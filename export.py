import os
import glob
import torch
from argparse import ArgumentParser

from microtcn.tcn import TCNModel
from microtcn.lstm import LSTMModel

def load_model(model_dir, gpu=False):

    checkpoint_path = glob.glob(os.path.join(model_dir,
                                            "lightning_logs",
                                            "version_0",
                                            "checkpoints",
                                            "*"))[0]

    hparams_file = os.path.join(model_dir, "hparams.yaml")
    batch_size = int(os.path.basename(model_dir).split('-')[-1][2:])
    model_type = os.path.basename(model_dir).split('-')[1]
    epoch = int(os.path.basename(checkpoint_path).split('-')[0].split('=')[-1])

    map_location = "cuda:0" if gpu else "cpu"

    if model_type == "LSTM":
        model = LSTMModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location
        )

    else:
        model = TCNModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location
        )

    return model

def export_model(model_dir, save_dir):
    """Export a single trained model to TorchScript format.
    
    Args:
        model_dir: Path to the directory containing the trained model
        save_dir: Path to directory where exported model will be saved
        
    Returns:
        str: Path to the exported model file, or None if export failed
    """
    try:
        model_id = os.path.basename(model_dir)
        print(f"Exporting model: {model_id}")
        
        model = load_model(model_dir)
        
        # remove training-only components before TorchScript conversion
        # these loss functions are not needed for inference and cause
        # TorchScript compilation errors
        if hasattr(model, 'l1'):
            delattr(model, 'l1')
        if hasattr(model, 'stft'):
            delattr(model, 'stft')
        
        # set to eval mode for inference
        model.eval()
        
        script = model.to_torchscript()
        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        export_path = os.path.join(save_dir, f"traced_{model_id}.pt")
        torch.jit.save(script, export_path)
        print(f"Model exported to: {export_path}")
        
        return export_path
        
    except Exception as e:
        print(f"ERROR: Failed to export model from {model_dir}")
        print(f"Error details: {str(e)}")
        return None


def export_all_models(model_dir, save_dir):
    """Export all trained models from a directory to TorchScript format.
    
    Args:
        model_dir: Path to directory containing multiple trained model subdirectories
        save_dir: Path to directory where exported models will be saved
    """
    models = sorted(glob.glob(os.path.join(model_dir, "*")))
    
    for idx, model_path in enumerate(models):
        export_model(model_path, save_dir)


if __name__ == '__main__':

    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--model_dir', type=str, default='./lightning_logs/bulk')
    parser.add_argument('--save_dir', type=str, default='./models')

    # parse them args
    args = parser.parse_args()

    export_all_models(args.model_dir, args.save_dir)
