Lung Segmentation using U-Net with ResNet-50 Backbone
This project performs semantic segmentation of lung fields in chest X-rays using a U-Net architecture with a ResNet-50 encoder.

All training, validation, and results tracking were performed using PyTorch and Weights & Biases (WandB).

Dataset
The dataset used is the Montgomery County Chest X-ray Set.
You can download it from Kaggle:
Montgomery Dataset on Kaggle

Important:
The dataset (images and masks) must be placed in the correct folder structure before running the code.
Example folder structure:

archive/
└── Montgomery/
    └── MontgomerySet/
        ├── CXR_png/         # Chest X-ray images
        └── masks/           # Combined left+right lung masks
        
Setup Instructions

Clone the Repository:
git clone https://github.com/danny121145/INM705.git
cd INM705

Download the Dataset:
Manually download the Montgomery dataset from Kaggle and place it inside the archive/ directory as shown above.

Create a Virtual Environment and Install Requirements:
If you have Bash (Git Bash, WSL, or VS Code Terminal):
bash setup.sh

Alternatively, you can manually do:
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt

How to Run the Project
You must run the following scripts in order:

1. Preprocessing Masks
   
First, combine left and right lung masks into one mask:
python preprocess.py

This will generate the combined masks into the archive/Montgomery/MontgomerySet/masks/ directory.

2. Train the Model
   
Start the model training:
python train.py

This will:
Train the model
Save the trained model checkpoints inside /checkpoints
Log all training and validation progress to Weights & Biases (WandB)
✅ After training, you will have a model checkpoint ready.

3. Run Inference
Test the trained model on new images:
python inference.py

This will:
Load a saved checkpoint
Predict lung masks for selected test images
Save and visualize the outputs inside the /inference_output folder
Calculate and display IoU and Dice scores
