import os
import cv2
import numpy as np

def combine_masks(mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    left_dir = os.path.join(mask_dir, "leftMask")
    right_dir = os.path.join(mask_dir, "rightMask")

    left_files = {f: os.path.join(left_dir, f) for f in os.listdir(left_dir) if f.endswith(".png")}
    right_files = {f: os.path.join(right_dir, f) for f in os.listdir(right_dir) if f.endswith(".png")}

    common_keys = set(left_files.keys()) & set(right_files.keys())

    print(f"Found {len(common_keys)} matching filenames in left + right masks")

    for file_name in sorted(common_keys):
        left_path = left_files[file_name]
        right_path = right_files[file_name]

        left_mask = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_mask = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        if left_mask is None or right_mask is None:
            print(f"⚠️ Skipping {file_name} due to read error")
            continue

        combined_mask = cv2.bitwise_or(left_mask, right_mask)

        base_name = file_name.replace(".png", "")
        save_path = os.path.join(output_dir, f"{base_name}_mask.png")
        cv2.imwrite(save_path, combined_mask)
        print(f"✅ Combined mask saved: {save_path}")

    # show mismatched files
    missing = set(left_files.keys()).symmetric_difference(set(right_files.keys()))
    if missing:
        print("\n⚠️ Mismatched mask files (not in both):")
        for m in missing:
            print(f"  - {m}")

if __name__ == "__main__":
    combine_masks(
        mask_dir=r"C:\Users\danie\OneDrive\Desktop\INM705\archive\Montgomery\MontgomerySet\ManualMask",
        output_dir=r"C:\Users\danie\OneDrive\Desktop\INM705\archive\Montgomery\MontgomerySet\masks"
    )
