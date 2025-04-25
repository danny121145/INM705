import segmentation_models_pytorch as smp
import yaml

def load_model(config_path="config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model = smp.Unet(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"],
    )
    return model