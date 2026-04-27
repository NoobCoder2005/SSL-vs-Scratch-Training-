# SSL vs Scratch Training for Medical CT Segmentation

This project compares two training strategies for multi-class CT organ segmentation:

- **SSL pretraining + finetuning** (SimCLR-style encoder pretraining, then UNet segmentation)
- **Training from scratch** (UNet segmentation without SSL initialization)

It also includes:

- Evaluation scripts (Dice metrics + qualitative predictions)
- A Streamlit app for interactive inference on 2D images and 3D NIfTI volumes

## Repository

GitHub repository name: `SSL-vs-Scratch-Training-`

## Project Structure

```text
medical_segmentation_project/
  main.py
  requirements.txt
  data/
  models/
  losses/
  training/
    pretrain_ssl.py
    train_segmentation.py
  evaluation/
    eval_segmentation.py
  streamlit_app.py
  checkpoints/
  results/
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data Format

Use a patient-root directory with one folder per patient. The code expects CT scans and segmentation data according to the dataset loader logic in `data/dataset_loader.py`.

If you are just testing the pipeline, use `--dummy` mode.

Optional split files:

- `train.txt`, `val.txt`, `test.txt`
- One patient ID per line

## Training

### 1) SSL Pretraining (SimCLR)

```bash
python main.py ssl --data_root "PATH_TO_DATA" --train_list train.txt --val_list val.txt --epochs 10 --batch_size 32
```

Output:

- `checkpoints/encoder.pth`

### 2) Segmentation With SSL Initialization

```bash
python main.py seg --data_root "PATH_TO_DATA" --train_list train.txt --val_list val.txt --encoder_ckpt checkpoints/encoder.pth --save_path checkpoints/unet_ssl.pth --epochs_head 5 --epochs_finetune 10
```

### 3) Segmentation From Scratch (No SSL)

```bash
python main.py seg --data_root "PATH_TO_DATA" --train_list train.txt --val_list val.txt --save_path checkpoints/unet_from_scratch.pth --epochs_head 0 --epochs_finetune 10
```

## Evaluation

Evaluate a trained model on a test split:

```bash
python evaluation/eval_segmentation.py --data_root "PATH_TO_DATA" --test_list test.txt --ckpt checkpoints/unet_ssl.pth --out_dir results/eval_ssl
```

Outputs include:

- `metrics.json`
- `metrics.csv`
- qualitative predictions in `results/.../predictions/`

## Streamlit Inference App

Launch:

```bash
streamlit run streamlit_app.py
```

In the sidebar, set checkpoint path (for example):

- `checkpoints/unet_from_scratch.pth`
- `checkpoints/unet_ssl.pth`

The app supports:

- 2D image uploads (`.png`, `.jpg`, `.jpeg`)
- 3D CT volumes (`.nii`, `.nii.gz`) with auto-selected representative slice

## Useful Options

Common flags in training scripts:

- `--device auto|cpu|cuda`
- `--amp` for mixed precision on CUDA
- `--num_workers`
- `--batch_size`

For a quick smoke test:

```bash
python main.py ssl --dummy --epochs 1
python main.py seg --dummy --epochs_head 1 --epochs_finetune 1
```

## Notes

- This project uses **5 segmentation classes** by default (background + organs).
- Intermediate checkpoints and result images can be large; consider adding them to `.gitignore` if needed.
