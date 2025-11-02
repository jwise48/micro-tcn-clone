import os
import glob
import torch
import torchsummary
from itertools import product
import pytorch_lightning as pl
from argparse import ArgumentParser

from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel
from microtcn.data import SignalTrainLA2ADataset

torch.backends.cudnn.benchmark = True
"""
Preference of 'medium' for a good balance of speed/accuracy.
Use 'high' for maximum speed  with slightly lower precision.
"""
torch.set_float32_matmul_precision('medium')  # Add this line

"""
The 14 Model Configurations:
1. uTCN-300 (causal, 1% train data) - tiny dataset test
2. uTCN-100 (causal, 100% data) - smaller receptive field
3. uTCN-300 (causal, 100% data) - standard micro-TCN
4. uTCN-1000 (causal, 100% data) - larger receptive field
5. uTCN-100 (non-causal, 100% data)
6. uTCN-300 (non-causal, 100% data)
7. uTCN-1000 (non-causal, 100% data)
8. TCN-300 (non-causal, 10 blocks instead of 4)
9. uTCN-300 (causal, 10% data) - ablation study
10. LSTM-32 (baseline comparison)
11. uTCN-300 (causal, different dilation pattern: 3-60-5)
12. uTCN-300 (causal, L1 loss only)
13. uTCN-300 (non-causal, 30 blocks - deeper)
14. uTCN-324-16 (non-causal, 16 channels instead of 32)

"""

train_configs = [
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 0.01,
     "batch_size" : 32
    },
    {"name" : "uTCN-100",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-1000",
     "model_type" : "tcn",
     "nblocks" : 5,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-100",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-1000",
     "model_type" : "tcn",
     "nblocks" : 5,
     "dilation_growth" : 10,
     "kernel_size" : 5,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "TCN-300",
     "model_type" : "tcn",
     "nblocks" : 10,
     "dilation_growth" : 2,
     "kernel_size" : 15,
     "causal" : False,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 0.10,
     "batch_size" : 32
    },
    {"name" : "LSTM-32",
     "model_type" : "lstm",
     "num_layers" : 1,
     "hidden_size" : 32,
     "train_fraction" : 1.00,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 3,
     "dilation_growth" : 60,
     "kernel_size" : 5,
     "causal" : True,
     "train_fraction" : 1.0,
     "batch_size" : 32
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 4,
     "dilation_growth" : 10,
     "kernel_size" : 13,
     "causal" : True,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
     "train_loss" : "l1"
    },
    {"name" : "uTCN-300",
     "model_type" : "tcn",
     "nblocks" : 30,
     "dilation_growth" : 2,
     "kernel_size" : 15,
     "causal" : False,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
    },
    {"name" : "uTCN-324-16",
     "model_type" : "tcn",
     "nblocks" : 10,
     "dilation_growth" : 2,
     "kernel_size" : 15,
     "causal" : False,
     "train_fraction" : 1.0,
     "batch_size" : 32,
     "max_epochs" : 60,
     "channel_width" : 16,
    },
]

if __name__ == '__main__':
    n_configs = len(train_configs)

    for idx, tconf in enumerate(train_configs):

        #if (idx+1) not in [14]: continue - ex: [3, 6, 8, 10, 14]
        # if you only want to train a specific model

        parser = ArgumentParser()

        # add PROGRAM level args
        parser.add_argument('--model_type', type=str, default='tcn', help='tcn or lstm')
        parser.add_argument('--root_dir', type=str, default='./data')
        parser.add_argument('--preload', action="store_true")
        parser.add_argument('--sample_rate', type=int, default=44100)
        parser.add_argument('--shuffle', type=bool, default=True)
        parser.add_argument('--train_subset', type=str, default='train')
        parser.add_argument('--val_subset', type=str, default='val')
        parser.add_argument('--train_length', type=int, default=65536)
        parser.add_argument('--train_fraction', type=float, default=1.0)
        parser.add_argument('--eval_length', type=int, default=131072)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=12)

        # add trainer options expected for PyTorch Lightning 2.x compatibility
        parser.add_argument('--max_epochs', type=int, default=60)
        parser.add_argument('--precision', type=str, default='16-mixed')
        parser.add_argument('--accelerator', type=str, default='auto')
        parser.add_argument('--devices', type=int, default=1)
        parser.add_argument('--default_root_dir', type=str, default='./lightning_logs')

        # THIS LINE IS KEY TO PULL THE MODEL NAME
        temp_args, _ = parser.parse_known_args()

        print(f"* Training config {idx+1}/{n_configs}")
        print(tconf)
    
        # let the model add what it wants
        if temp_args.model_type == 'tcn':
            parser = TCNModel.add_model_specific_args(parser)
        elif temp_args.model_type == 'lstm':
            parser = LSTMModel.add_model_specific_args(parser)

        # parse them args
        args = parser.parse_args()

        # set the seed
        pl.seed_everything(42)

        # only run 60 epochs
        args.max_epochs = 60

        # init the trainer and model 
        if tconf["model_type"] == 'tcn':
            specifier =  f"{idx+1}-{tconf['name']}"
            specifier += "__causal" if tconf['causal'] else "__noncausal"
            specifier += f"__{tconf['nblocks']}-{tconf['dilation_growth']}-{tconf['kernel_size']}"
            specifier += f"__fraction-{tconf['train_fraction']}-bs{tconf['batch_size']}"
        elif tconf["model_type"] == 'lstm':
            specifier =  f"{idx+1}-{tconf['name']}"
            specifier += f"__{tconf['num_layers']}-{tconf['hidden_size']}"
            specifier += f"__fraction-{tconf['train_fraction']}-bs{tconf['batch_size']}"

        if "max_epochs" in tconf:
            args.max_epochs = tconf["max_epochs"]
        else:
            args.max_epochs = 60

        if "train_loss" in tconf:
            args.train_loss = tconf["train_loss"]
            specifier += f"__loss-{tconf['train_loss']}"

        # Set precision to use PyTorch Lightning 2.x string format
        args.precision = "16-mixed"

        args.default_root_dir = os.path.join("lightning_logs", "bulk", specifier)
        print(args.default_root_dir)
        
        # Create trainer with explicit arguments (PyTorch Lightning 2.x compatible)
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            precision=args.precision,
            accelerator=args.accelerator if hasattr(args, 'accelerator') else 'auto',
            devices=args.devices if hasattr(args, 'devices') else 1,
            default_root_dir=args.default_root_dir,
        )

        # setup the dataloaders
        # Check precision for half-precision training (handle both string and numeric formats)
        use_half = args.precision in ["16-mixed", "16-true", "16", 16]
        
        train_dataset = SignalTrainLA2ADataset(args.root_dir, 
                                        subset=args.train_subset,
                                        fraction=tconf["train_fraction"],
                                        half=use_half,
                                        preload=args.preload,
                                        length=args.train_length)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                    shuffle=args.shuffle,
                                                    batch_size=tconf["batch_size"],
                                                    num_workers=args.num_workers,
                                                    pin_memory=True)

        val_dataset = SignalTrainLA2ADataset(args.root_dir, 
                                        preload=args.preload,
                                        half=use_half,
                                        subset=args.val_subset,
                                        length=args.eval_length)

        val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                    shuffle=False,
                                                    batch_size=8,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True)

        # create the model with args
        dict_args = vars(args)
        dict_args["nparams"] = 2

        if tconf["model_type"] == 'tcn':
            dict_args["nblocks"] = tconf["nblocks"]
            dict_args["dilation_growth"] = tconf["dilation_growth"]
            dict_args["kernel_size"] = tconf["kernel_size"]
            dict_args["causal"] = tconf["causal"]
            if "channel_width" in tconf:
                dict_args["channel_width"] = tconf["channel_width"]
            model = TCNModel(**dict_args)
        elif tconf["model_type"] == 'lstm':
            dict_args["num_layers"] = tconf["num_layers"]
            dict_args["hidden_size"] = tconf["hidden_size"]
            model = LSTMModel(**dict_args)

        # summary 
        torchsummary.summary(model, [(1,65536), (1,2)], device="cpu")

        # train!
        trainer.fit(model, train_dataloader, val_dataloader)
