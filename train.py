import os
import yaml
import wandb
import torch
from tqdm import tqdm
from dataset import MontgomeryDataset, get_transforms
from model import load_model
from torch.utils.data import DataLoader, random_split

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

# Init Weights and Biases
wandb.init(project="INM705_LungSegmentation", config=config)

# Create dataset and dataloader
train_dataset = MontgomeryDataset(
    img_dir=os.path.normpath(config["data"]["img_dir"]),
    mask_dir=os.path.normpath(config["data"]["mask_dir"]),
    transform=get_transforms(train=True)  # from dataset.py
)

# Split dataset into training and validation sets
val_split = 0.2
val_size = int(len(train_dataset) * val_split)
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=int(config["data"]["batch_size"]), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=int(config["data"]["batch_size"]), shuffle=False)

print(f"Training on {train_size} images, validating on {val_size} images.")

# Load model
model = load_model(config_path="config.yaml").to(device)
wandb.watch(model, log="all")

# Loss + Optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["lr"]))

# early stopping parameters
best_val_loss = float("inf")
patience = 5
patience_counter = 0
os.makedirs("checkpoints", exist_ok=True)

# Training loop
epochs = int(config["training"]["epochs"])
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        wandb.log({"batch_loss": loss.item()})

    avg_train_loss = train_loss / len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    wandb.log({
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "epoch": epoch + 1
    })

    # early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "checkpoints/unet_best.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

# Save final model
model_path = "checkpoints/unet_final.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ Final model saved to {model_path}")

# Log final model to WandB
artifact = wandb.Artifact("lung_model_final", type="model")
artifact.add_file(model_path)
wandb.log_artifact(artifact)

wandb.finish()
