# Building Segmentation of Hazy Aerial Images via Collaborative Decoupling Knowledge Learning

This project implements a two-stage training strategy for remote sensing image dehazing and object/segmentation-based vision tasks.

## 🛠 Prerequisites

### Environment Setup

Create the Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate CDKL
```

Alternatively, ensure you have the necessary dependencies installed:
- Python 3.x
- PyTorch
- Additional dependencies: `torchvision`, `PIL`, `cv2`, `numpy`, `tqdm`, `thop`.

## 🚀 Training Process

The training is divided into two major steps:

### Step 1: Teacher Network Training (T-Net)
Train the teacher network using clear images to establish a high-quality vision baseline.

1.  **Script**: [T_Net_train.py](file:///home/user/H/ObjectDetection2023_revised/github_vision/T_Net_train.py)
2.  **Configuration**: 
    - Open [config.py](file:///home/user/H/ObjectDetection2023_revised/github_vision/config.py).
    - Modify `train_data_root` and `val_data_root` to point to your clear image datasets.
    - Set `load_Tmodel_path = None` if starting from scratch.
3.  **Command**:
    ```bash
    python T_Net_train.py
    ```

### Step 2: Student Network Training with Knowledge Distillation (S-Net)
Train the student network using both hazy and clear images, guided by the pre-trained teacher network.

1.  **Script**: [S_Net_train.py](file:///home/user/H/ObjectDetection2023_revised/github_vision/S_Net_train.py)
2.  **Configuration**:
    - Open [config.py](file:///home/user/H/ObjectDetection2023_revised/github_vision/config.py).
    - Ensure `train_data_root` and `val_data_root` point to the dataset containing paired hazy and clear images.
    - **Crucial**: Set `load_Tmodel_path` to the path of the `.pth` file generated in Step 1.
    - Set `load_model_path = None` if starting student training for the first time.
3.  **Command**:
    ```bash
    python S_Net_train.py
    ```

## 🔍 Testing and Inference

Run inference on hazy images using the trained student network.

1.  **Script**: [test.py](file:///home/user/H/ObjectDetection2023_revised/github_vision/test.py)
2.  **Configuration**:
    - Open `test.py`.
    - Update the `imgs` path in the `if __name__ == '__main__':` block to point to your test hazy images.
    - Ensure `selectnet = 'S-net'`.
    - Set `outgt_path` for segmentation results and `outclear_path` for dehazed image results.
    - Verify `opt.load_model_path` in `config.py` points to your best S-Net checkpoint.
3.  **Command**:
    ```bash
    python test.py
    ```

## 📂 Configuration Details

Most hyperparameters and paths are managed in [config.py](file:///home/user/H/ObjectDetection2023_revised/github_vision/config.py):

| Parameter | Description |
| :--- | :--- |
| `train_data_root` | Path to training dataset |
| `val_data_root` | Path to validation dataset |
| `load_Tmodel_path` | Pre-trained Teacher model path (for distillation) |
| `load_model_path` | Pre-trained Student model path (for testing/resuming) |
| `batch_size` | Training batch size |
| `max_epoch` | Maximum number of epochs |

## 📝 Notes
- Ensure your dataset structure follows the expected format (e.g., `hazy/`, `clear/`, `gt/` folders).
- Data loading logic for matching hazy (`_1.jpg`) and clear images is handled in the custom dataset classes.

## 📄 Publication
This work is currently under review/submission at **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**.
