# A Unified Framework for Semi-Supervised Image Segmentation and Registration

This repository contains the official implementation of the paper accepted at ISBI 2025:
**"A Unified Framework for Semi-Supervised Image Segmentation and Registration"**

Paper URL: [https://arxiv.org/html/2502.03229v1](https://arxiv.org/html/2502.03229v1)

## Overview

UniSegReg is an end-to-end framework that jointly performs medical image segmentation and registration with an iterative quality assessment-based data expansion strategy. The approach leverages partially labeled data and iteratively expands the training dataset by evaluating segmentation quality.

### Key Features:

- Joint segmentation and registration training
- Iterative quality assessment-based data expansion
- Works with both 2D and 3D medical images
- End-to-end trainable architecture

## Repository Structure

```
UniSegReg/
├── snmi/                    # Core code for models, datasets, and training
├── models/                  # Model implementation files
│   ├── __init__.py
│   ├── joint_seg.py         # Segmentation model implementation
│   └── joint_reg.py         # Registration model implementation
├── train2D.py               # Training script for 2D images
├── train3D.py               # Training script for 3D images
└── README.md
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/username/UniSegReg.git
cd UniSegReg
```

2. Configure your environment:
- Python 3.6+
- PyTorch 1.7+
- CUDA (for GPU acceleration)

3. Install required packages:
```bash
pip install torch torchvision numpy scipy scikit-image SimpleITK matplotlib tensorboard tqdm
```

4. Prepare your dataset:
   - Update the `PATH_TO_2D_DATASET` or `PATH_TO_3D_DATASET` variables in the training scripts
   - Organize your dataset with the following structure:
     ```
     dataset/
     ├── train/
     ├── valid/
     ├── test/
     └── source_test/
     ```
   - Ensure images have the proper suffixes as specified in the config section

## Usage

### 2D Joint Training

Configure the dataset path in `train2D.py`:
```python
PATH_TO_2D_DATASET = "/path/to/your/2D/dataset"
```

Run the training script:
```bash
python train2D.py
```

### 3D Joint Training

Configure the dataset path in `train3D.py`:
```python
PATH_TO_3D_DATASET = "/path/to/your/3D/dataset"
```

Run the training script:
```bash
python train3D.py
```

## Method

The UniSegReg framework consists of three main components:

1. **Segmentation Module**: Based on a U-Net architecture for medical image segmentation
2. **Registration Module**: Deformable registration network based on a multi-resolution approach
3. **Quality Assessment**: Evaluates segmentation quality to expand training data iteratively

In the iterative training procedure:
1. Train segmentation and registration models initially on labeled data
2. Use these models to segment unlabeled images
3. Evaluate segmentation quality to identify reliable predictions
4. Add high-quality predictions to training set for next iteration
5. Repeat until desired performance or convergence

## Configuration

The training scripts include configuration classes (`cfg`) that control various aspects of the training:

- Dataset paths and file suffixes
- Training parameters (batch size, learning rate, etc.)
- Model architecture parameters
- Loss functions and weights
- Testing and evaluation settings

You can modify these parameters to suit your specific dataset and requirements.

## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@article{li2024unisegreg,
  title={A Unified Framework for Semi-Supervised Image Segmentation and Registration},
  author={Li, Ruizhe and Figueredo, Grazziela and Auer, Dorothee and Dineen, Rob and Morgan, Paul and Chen, Xin},
  journal={arXiv preprint arXiv:2502.03229},
  year={2025}
}
```

## License

This project is licensed under the MIT License.