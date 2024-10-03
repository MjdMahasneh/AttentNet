# AttentNet: Fully Convolutional 3D Attention for Lung Nodule Detection

This repository contains the code and resources for [AttentNEt](https://arxiv.org/abs/2407.14464), an automated lung nodule detection system based on 3D convolutional attention mechanisms. AttentNet operates in two stages:
1. **Region Proposal Network (RPN)**: Proposes candidate nodule locations.
2. **False Positive (FP) Reduction**: Refines the predictions to reduce false positives.

## Key Features
- **3D Convolutional Attention**: Efficient processing of 3D medical images, reducing computational overhead.
- **Two-Stage Detection**: Combines RPN and FP reduction stages for accurate nodule detection.
- **LUNA16 Dataset**: Tested and evaluated on the LUNA16 lung nodule dataset.

## Requirements

- `opencv-python`
- `numpy`
- `torch`
- `sklearn`
- `tqdm`
- `scipy`


## Installation
Clone the repository:

```bash
    git clone https://github.com/MjdMahasneh/AttentNet/tree/master
```

## Usage

- **`./RPN/main.py`**: Implements the Region Proposal Network (RPN) for generating lung nodule candidates, utilizing 3D convolutional attention for enhanced feature extraction and localization.
- **`./FP_reduction/main.py`**: Implements the False Positive (FP) reduction stage, which refines the candidates produced by the RPN by reducing false positives, improving the overall detection accuracy.

###### NOTE: this repo is work in progress and will be updated soon with more features and improvements.
