import os
import sys
import glob
import torch 
import torchaudio
import numpy as np
import soundfile as sf

class SignalTrainLA2ADataset(torch.utils.data.Dataset):
    """ SignalTrain LA2A dataset. Source: [10.5281/zenodo.3824876](https://zenodo.org/record/3824876)."""
    def __init__(self, root_dir, subset="train", length=16384, preload=False, half=True, fraction=1.0, use_soundfile=False):
        """
        Args:
            root_dir (str): Path to the root directory of the SignalTrain dataset.
            subset (str, optional): Pull data either from "train", "val", "test", or "full" subsets. (Default: "train")
            length (int, optional): Number of samples in the returned examples. (Default: 40)
            preload (bool, optional): Read in all data into RAM during init. (Default: False)
            half (bool, optional): Store the float32 audio as float16. (Default: True)
            fraction (float, optional): Fraction of the data to load from the subset. (Default: 1.0)
            use_soundfile (bool, optional): Use the soundfile library to load instead of torchaudio. (Default: False)
        """
        self.root_dir = root_dir
        self.subset = subset
        self.length = length
        self.preload = preload
        self.half = half
        self.fraction = fraction
        self.use_soundfile = use_soundfile

        if self.subset == "full":
            self.target_files = glob.glob(os.path.join(self.root_dir, "**", "target_*.wav"))
            self.input_files  = glob.glob(os.path.join(self.root_dir, "**", "input_*.wav"))
        else:
            # get all the target files files in the directory first
            self.target_files = glob.glob(os.path.join(self.root_dir, self.subset.capitalize(), "target_*.wav"))
            self.input_files  = glob.glob(os.path.join(self.root_dir, self.subset.capitalize(), "input_*.wav"))

        self.examples = [] 
        self.minutes = 0  # total number of hours of minutes in the subset

        # ensure that the sets are ordered correctlty
        self.target_files.sort()
        self.input_files.sort()

        # get the parameters 
        self.params = [(float(f.split("__")[1].replace(".wav","")), float(f.split("__")[2].replace(".wav",""))) for f in self.target_files]

        # loop over files to count total length
        for idx, (tfile, ifile, params) in enumerate(zip(self.target_files, self.input_files, self.params)):

            ifile_id = int(os.path.basename(ifile).split("_")[1])
            tfile_id = int(os.path.basename(tfile).split("_")[1])
            if ifile_id != tfile_id:
                raise RuntimeError(f"Found non-matching file ids: {ifile_id} != {tfile_id}! Check dataset.")

            md = torchaudio.info(tfile)
            num_frames = md.num_frames

            if self.preload:
                sys.stdout.write(f"* Pre-loading... {idx+1:3d}/{len(self.target_files):3d} ...\r")
                sys.stdout.flush()
                input, sr  = self.load(ifile)
                target, sr = self.load(tfile)

                num_frames = int(np.min([input.shape[-1], target.shape[-1]]))
                if input.shape[-1] != target.shape[-1]:
                    print(os.path.basename(ifile), input.shape[-1], os.path.basename(tfile), target.shape[-1])
                    raise RuntimeError("Found potentially corrupt file!")
                if self.half:
                    input = input.half()
                    target = target.half()
            else:
                input = None
                target = None

            # create one entry for each patch
            self.file_examples = []
            for n in range((num_frames // self.length)):
                offset = int(n * self.length)
                end = offset + self.length
                self.file_examples.append({"idx": idx, 
                                           "target_file" : tfile,
                                           "input_file" : ifile,
                                           "input_audio" : input[:,offset:end] if input is not None else None,
                                           "target_audio" : target[:,offset:end] if input is not None else None,
                                           "params" : params,
                                           "offset": offset,
                                           "frames" : num_frames})

            # add to overall file examples
            self.examples += self.file_examples
        
        # use only a fraction of the subset data if applicable
        if self.subset == "train":
            classes = set([ex['params'] for ex in self.examples])
            n_classes = len(classes) # number of unique compressor configurations
            fraction_examples = int(len(self.examples) * self.fraction)
            n_examples_per_class = int(fraction_examples / n_classes)
            n_min_total = ((self.length * n_examples_per_class * n_classes) / md.sample_rate) / 60 
            n_min_per_class = ((self.length * n_examples_per_class) / md.sample_rate) / 60 
            print(sorted(classes))
            print(f"Total Examples: {len(self.examples)}     Total classes: {n_classes}")
            print(f"Fraction examples: {fraction_examples}    Examples/class: {n_examples_per_class}")
            print(f"Training with {n_min_per_class:0.2f} min per class    Total of {n_min_total:0.2f} min")

            if n_examples_per_class <= 0: 
                raise ValueError(f"Fraction `{self.fraction}` set too low. No examples selected.")

            sampled_examples = []

            for config_class in classes: # select N examples from each class
                class_examples = [ex for ex in self.examples if ex["params"] == config_class]
                example_indices = np.random.randint(0, high=len(class_examples), size=n_examples_per_class)
                class_examples = [class_examples[idx] for idx in example_indices]
                extra_factor = int(1/self.fraction)
                sampled_examples += class_examples * extra_factor

            self.examples = sampled_examples

        self.minutes = ((self.length * len(self.examples)) / md.sample_rate) / 60 

        # we then want to get the input files
        print(f"Located {len(self.examples)} examples totaling {self.minutes:0.2f} min in the {self.subset} subset.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.preload:
            audio_idx = self.examples[idx]["idx"]
            offset = self.examples[idx]["offset"]
            input = self.examples[idx]["input_audio"]
            target = self.examples[idx]["target_audio"]
        else:
            offset = self.examples[idx]["offset"] 
            input, sr  = torchaudio.load(self.examples[idx]["input_file"], 
                                        num_frames=self.length, 
                                        frame_offset=offset, 
                                        normalize=False)
            target, sr = torchaudio.load(self.examples[idx]["target_file"], 
                                        num_frames=self.length, 
                                        frame_offset=offset, 
                                        normalize=False)
            if self.half:
                input = input.half()
                target = target.half()

        # at random with p=0.5 flip the phase 
        if np.random.rand() > 0.5:
            input *= -1
            target *= -1

        # then get the tuple of parameters
        params = torch.tensor(self.examples[idx]["params"]).unsqueeze(0)
        params[:,1] /= 100

        return input, target, params

    def load(self, filename):
        if self.use_soundfile:
            x, sr = sf.read(filename, always_2d=True)
            x = torch.tensor(x.T)
        else:
            x, sr = torchaudio.load(filename, normalize=False)
        return x, sr


class SolidStateBusCompDataset(torch.utils.data.Dataset):
    """ Diff-SSL-G-Comp. Source: [10.48550/arXiv.2504.04589](https://huggingface.co/datasets/amphion/SolidStateBusComp)."""
    def __init__(self, root_dir, subset="train", length=16384, preload=False, half=True, fraction=1.0, use_soundfile=False):
        """
        Args:
            root_dir (str): Path to the root directory of the Diff-SSL-G-Comp dataset.
            subset (str, optional): Pull data either from "train", "val", "test", or "full" subsets. (Default: "train")
            length (int, optional): Number of samples in the returned examples. (Default: 16384)
            preload (bool, optional): Read in all data into RAM during init. (Default: False)
            half (bool, optional): Store the float32 audio as float16. (Default: True)
            fraction (float, optional): Fraction of the data to load from the subset. (Default: 1.0)
            use_soundfile (bool, optional): Use the soundfile library to load instead of torchaudio. (Default: False)
        """
        self.root_dir = root_dir
        self.subset = subset
        self.length = length
        self.preload = preload
        self.half = half
        self.fraction = fraction
        self.use_soundfile = use_soundfile

        # get input files from processed_normalized directory
        input_dir = os.path.join(self.root_dir, "processed_normalized")
        if not os.path.exists(input_dir):
            raise RuntimeError(f"Input directory not found: {input_dir}")
        
        input_files = glob.glob(os.path.join(input_dir, "*.wav"))
        if len(input_files) == 0:
            raise RuntimeError(f"No input files found in {input_dir}")
        
        input_files.sort()
        
        # get all parameter folders from processed_ground_truth
        gt_dir = os.path.join(self.root_dir, "processed_ground_truth")
        if not os.path.exists(gt_dir):
            raise RuntimeError(f"Ground truth directory not found: {gt_dir}")
        
        param_folders = [d for d in os.listdir(gt_dir) 
                        if os.path.isdir(os.path.join(gt_dir, d)) and d.startswith("threshold_")]
        param_folders.sort()
        
        if len(param_folders) == 0:
            raise RuntimeError(f"No parameter folders found in {gt_dir}")
        
        print(f"Found {len(input_files)} input files and {len(param_folders)} parameter configurations")
        
        # parse parameters from folder names
        # format: threshold_X_attack_Y_release_Z_ratio_W
        self.param_configs = []
        for folder in param_folders:
            parts = folder.split("_")
            try:
                threshold = float(parts[1])
                attack = float(parts[3])
                release = float(parts[5])
                ratio = float(parts[7])
                self.param_configs.append({
                    "folder": folder,
                    "params": (threshold, attack, release, ratio)
                })
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse parameters from folder: {folder}")
                continue
        
        print(f"Successfully parsed {len(self.param_configs)} parameter configurations")
        
        self.examples = []
        self.minutes = 0
        
        # create mapping between input and output files
        # input format: XX_UnmasteredWAV.wav
        # output format: XX-exported.wav
        file_pairs = []
        
        for input_file in input_files:
            input_basename = os.path.basename(input_file)
            # extract the song ID (e.g., "54" from "54_UnmasteredWAV.wav")
            song_id = input_basename.split("_")[0]
            
            # find corresponding output files in all parameter folders
            for param_config in self.param_configs:
                output_file = os.path.join(gt_dir, param_config["folder"], f"{song_id}-exported.wav")
                if os.path.exists(output_file):
                    file_pairs.append({
                        "input_file": input_file,
                        "target_file": output_file,
                        "params": param_config["params"],
                        "song_id": song_id
                    })
        
        if len(file_pairs) == 0:
            raise RuntimeError("No matching input/output file pairs found!")
        
        print(f"Created {len(file_pairs)} input/output file pairs")
        
        # split into train/val/test subsets
        # use song IDs to ensure no song appears in multiple subsets
        unique_song_ids = sorted(list(set([pair["song_id"] for pair in file_pairs])))
        n_songs = len(unique_song_ids)
        
        if self.subset == "full":
            selected_song_ids = unique_song_ids
        else:
            # split: 70% train, 15% val, 15% test
            n_train = int(n_songs * 0.70)
            n_val = int(n_songs * 0.15)
            
            if self.subset == "train":
                selected_song_ids = unique_song_ids[:n_train]
            elif self.subset == "val":
                selected_song_ids = unique_song_ids[n_train:n_train+n_val]
            elif self.subset == "test":
                selected_song_ids = unique_song_ids[n_train+n_val:]
            else:
                raise ValueError(f"Unknown subset: {self.subset}. Use 'train', 'val', 'test', or 'full'")
        
        # filter file pairs by selected song IDs
        selected_pairs = [pair for pair in file_pairs if pair["song_id"] in selected_song_ids]
        print(f"Selected {len(selected_pairs)} pairs for {self.subset} subset")
        
        # process each file pair
        for idx, pair in enumerate(selected_pairs):
            input_file = pair["input_file"]
            target_file = pair["target_file"]
            params = pair["params"]
            
            # get file info
            md = torchaudio.info(target_file)
            num_frames = md.num_frames
            
            if self.preload:
                sys.stdout.write(f"* Pre-loading... {idx+1:3d}/{len(selected_pairs):3d} ...\r")
                sys.stdout.flush()
                input_audio, sr = self.load(input_file)
                target_audio, sr = self.load(target_file)
                
                num_frames = int(np.min([input_audio.shape[-1], target_audio.shape[-1]]))
                if input_audio.shape[-1] != target_audio.shape[-1]:
                    print(f"\nWarning: Length mismatch - {os.path.basename(input_file)}: {input_audio.shape[-1]}, "
                          f"{os.path.basename(target_file)}: {target_audio.shape[-1]}")
                    # Trim to shortest length
                    input_audio = input_audio[:, :num_frames]
                    target_audio = target_audio[:, :num_frames]
                
                if self.half:
                    input_audio = input_audio.half()
                    target_audio = target_audio.half()
            else:
                input_audio = None
                target_audio = None
            
            # create one entry for each patch
            for n in range(num_frames // self.length):
                offset = int(n * self.length)
                end = offset + self.length
                self.examples.append({
                    "idx": idx,
                    "target_file": target_file,
                    "input_file": input_file,
                    "input_audio": input_audio[:, offset:end] if input_audio is not None else None,
                    "target_audio": target_audio[:, offset:end] if target_audio is not None else None,
                    "params": params,
                    "offset": offset,
                    "frames": num_frames
                })
        
        # apply fraction sampling if training
        if self.subset == "train" and self.fraction < 1.0:
            classes = set([ex['params'] for ex in self.examples])
            n_classes = len(classes)
            fraction_examples = int(len(self.examples) * self.fraction)
            n_examples_per_class = max(1, int(fraction_examples / n_classes))
            n_min_total = ((self.length * n_examples_per_class * n_classes) / md.sample_rate) / 60
            n_min_per_class = ((self.length * n_examples_per_class) / md.sample_rate) / 60
            
            print(f"\nTotal Examples: {len(self.examples)}     Total classes: {n_classes}")
            print(f"Fraction examples: {fraction_examples}    Examples/class: {n_examples_per_class}")
            print(f"Training with {n_min_per_class:0.2f} min per class    Total of {n_min_total:0.2f} min")
            
            sampled_examples = []
            for config_class in classes:
                class_examples = [ex for ex in self.examples if ex["params"] == config_class]
                if len(class_examples) < n_examples_per_class:
                    sampled_examples += class_examples
                else:
                    example_indices = np.random.choice(len(class_examples), size=n_examples_per_class, replace=False)
                    sampled_examples += [class_examples[idx] for idx in example_indices]
            
            self.examples = sampled_examples
        
        self.minutes = ((self.length * len(self.examples)) / md.sample_rate) / 60
        print(f"Located {len(self.examples)} examples totaling {self.minutes:0.2f} min in the {self.subset} subset.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.preload:
            input = self.examples[idx]["input_audio"]
            target = self.examples[idx]["target_audio"]
        else:
            offset = self.examples[idx]["offset"]
            input, sr = torchaudio.load(self.examples[idx]["input_file"],
                                       num_frames=self.length,
                                       frame_offset=offset,
                                       normalize=False)
            target, sr = torchaudio.load(self.examples[idx]["target_file"],
                                        num_frames=self.length,
                                        frame_offset=offset,
                                        normalize=False)
            if self.half:
                input = input.half()
                target = target.half()
        
        # at random with p=0.5 flip the phase
        if np.random.rand() > 0.5:
            input *= -1
            target *= -1
        
        # get the 4-parameter tuple: (threshold, attack, release, ratio)
        params = torch.tensor(self.examples[idx]["params"], dtype=torch.float32).unsqueeze(0)
        
        # Normalize parameters to reasonable ranges
        # threshold: typically -20 to 0 dB -> normalize to [0, 1]
        # attack: typically 0.1 to 30 ms -> normalize to [0, 1]
        # release: typically 0.1 to 1.2 s -> normalize to [0, 1]
        # ratio: typically 2 to 10 -> normalize to [0, 1]
        params_normalized = params.clone()
        params_normalized[:, 0] = (params[:, 0] + 20) / 20  # threshold
        params_normalized[:, 1] = params[:, 1] / 30  # attack
        params_normalized[:, 2] = params[:, 2] / 1.2  # release
        params_normalized[:, 3] = (params[:, 3] - 2) / 8  # ratio
        
        return input, target, params_normalized

    def load(self, filename):
        if self.use_soundfile:
            x, sr = sf.read(filename, always_2d=True)
            x = torch.tensor(x.T)
        else:
            x, sr = torchaudio.load(filename, normalize=False)
        return x, sr
