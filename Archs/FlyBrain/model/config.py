from pathlib import Path

def get_config():
    return {
        "batch_size": 4,
        "num_epochs": 10,
        "lr": 1e-4,
        "img_height": 480,
        "img_width": 640,
        "root_dir": "C:/Users/deshu/.cache/kagglehub/datasets/soumikrakshit/nyu-depth-v2/versions/1/nyu_data/",
        "model_folder": "weights",
        "model_basename": "flydepth_model_",
        "preload": None,  
        "experiment_name": "runs/flydepth",
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pth"
    return str(Path('.') / model_folder / model_filename)
