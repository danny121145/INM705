import os
import yaml
import wandb
import torch
from tqdm import tqdm
from dataset import MontgomeryDataset, get_transforms
from model import load_model

# 1. Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Set device
device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

# 3. Init Weights and Biases
wandb.init(project="INM705_LungSegmentation", config=config)

# 4. Create dataset and dataloader
train_dataset = MontgomeryDataset(
    img_dir=os.path.normpath(config["data"]["img_dir"]),
    mask_dir=os.path.normpath(config["data"]["mask_dir"]),
    transform=get_transforms(train=True)  # from dataset.py
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=int(config["data"]["batch_size"]),
    shuffle=True
)

# 5. Load model
model = load_model(config_path="config.yaml").to(device)
wandb.watch(model, log="all")

# 6. Loss + Optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["lr"]))

# 7. Training loop
epochs = int(config["training"]["epochs"])
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        wandb.log({"batch_loss": loss.item()})

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
    wandb.log({"epoch_loss": avg_loss})

# 8. Save model
os.makedirs("checkpoints", exist_ok=True)
model_path = "checkpoints/unet_final.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")
artifact = wandb.Artifact("lung_model", type="model")
artifact.add_file(model_path)
wandb.log_artifact(artifact)

wandb.finish()
