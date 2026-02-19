# Plant Detection App 🌿

## Overview

The **Plant Detection App** is a state-of-the-art tool designed to help farmers and agriculturalists identify plant diseases quickly and accurately. Leveraging advanced Deep Learning models, specifically **Vision Transformer (ViT)** and **Swin Transformer**, the app analyzes images of plant leaves to detect diseases across a wide variety of crops.

## Key Features

-   **Start-of-the-Art AI**: powered by Vision Transformer and Swin Transformer architectures.
-   **High Accuracy**: Trained on the Plant Village dataset containing over 87,000 images.
-   **Wide Coverage**: Capable of detecting 38 different disease classes across apples, tomatoes, corn, potatoes, and more.
-   **Real-time Results**: Get instant diagnosis with just a single image upload.
-   **User-Friendly Interface**: Simple, clean, and responsive UI built with Streamlit.

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Dataset**
    This script will download the Plant Village dataset from Google Drive and extract it to the `dataset/` directory.
    ```bash
    python download_data.py
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## Training

To train the model yourself, use the `train.py` script.

**Train Vision Transformer (ViT):**
```bash
python train.py --data_dir dataset/Plant_leaf_diseases_dataset_with_augmentation --model vit --epochs 10 --batch_size 32
```

**Train Swin Transformer:**
```bash
python train.py --data_dir dataset/Plant_leaf_diseases_dataset_with_augmentation --model swin --epochs 10 --batch_size 32
```

## Evaluation

To evaluate a trained model:

```bash
python evaluate.py --data_dir dataset/Plant_leaf_diseases_dataset_with_augmentation --model vit --checkpoint checkpoints/best_model_vit.pth
```
