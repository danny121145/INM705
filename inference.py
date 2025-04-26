import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dataset import get_transforms
from model import load_model
import cv2

# Config 
image_path = "archive/Montgomery/MontgomerySet/CXR_png/MCUCXR_0194_1.png"  # test image
checkpoint_path = "checkpoints/unet_best.pth"
output_path = "inference_output/predicted_mask.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Derive ground-truth path from image
base_name = os.path.basename(image_path).replace(".png", "")
gt_mask_path = f"archive/Montgomery/MontgomerySet/masks/{base_name}_mask.png"

# Prepare folders
os.makedirs("inference_output", exist_ok=True)

# Load model
model = load_model("config.yaml")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# Load and preprocess image
raw_img = Image.open(image_path).convert("RGB")
dummy_mask = np.zeros((raw_img.height, raw_img.width), dtype=np.uint8)  # match image height & width
preprocessed = get_transforms(train=False)(image=np.array(raw_img), mask=dummy_mask)
input_tensor = preprocessed["image"].unsqueeze(0).to(device)

# test-time augumentation (TTA)
with torch.no_grad():
    # normal prediction
    pred1 = model(input_tensor)

    # horizontal flip prediction
    flipped_img = torch.flip(input_tensor, dims=[3])
    pred2 = torch.flip(model(flipped_img), dims=[3])

    # average predictions
    output = (pred1 + pred2) / 2
    pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = (pred_mask > 0.5).astype(np.uint8)

# Save and show result
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(raw_img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(binary_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(raw_img)
plt.imshow(binary_mask, alpha=0.4, cmap="Reds")
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.savefig(output_path)
print(f"✅ Saved result to {output_path}")
plt.show()

# Load predicted mask and ground truth mask
pred_mask = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

# Resize ground truth to match prediction shape
gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]))

# Binarize both
pred_bin = (pred_mask > 50).astype(np.uint8)
gt_bin = (gt_mask > 50).astype(np.uint8)

# Compute IoU
intersection = np.logical_and(pred_bin, gt_bin).sum()
union = np.logical_or(pred_bin, gt_bin).sum()
iou_score = intersection / union if union != 0 else 0

# Compute Dice Score
dice_score = (2 * intersection) / (pred_bin.sum() + gt_bin.sum()) if (pred_bin.sum() + gt_bin.sum()) != 0 else 0

print(f"📐 IoU Score: {iou_score:.4f}")
print(f"🧮 Dice Score: {dice_score:.4f}")