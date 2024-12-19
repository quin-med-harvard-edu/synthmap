# SynthMap: Generative Model for MRI Myelin Water Fraction Mapping

This repository contains the official implementation of "SynthMap: a generative model for synthesis of 3D datasets for quantitative MRI parameter mapping of myelin water fraction" (MIDL 2022).

## Overview

SynthMap is a framework that combines:
- MR physics signal decay model
- Probabilistic multi-component parametric T2 model 
- Contrast-agnostic spatial generative model

to improve estimation of T2 distributions from multi-echo T2 data. The model generates synthetic training data for deep neural networks without requiring real data.

Key features:
- Generates large synthetic datasets (120,000+ slices) for training
- Robust to noise compared to conventional methods
- Works with limited real training data
- Superior accuracy over NNLS-based methods
- Compatible with clinical scan protocols

## Installation

```bash
# Clone repository
git clone https://github.com/sergeicu/synthmap

# Create conda environment
conda env create -f environment.yaml

# Activate environment
conda activate synthmap
```

## Repository Structure

```
.
├── competing_methods/      # Implementation of baseline methods (NNLS, ANN)
├── generate_training_data/ # Data generation pipeline
│   ├── generate_mr_signal/     # MR signal synthesis
│   └── generate_parameter_maps/ # Parameter map generation
└── train/                 # Training pipeline and models
    ├── data/             # Data loading and preprocessing
    ├── models/           # Model architectures (U-Net etc.)
    └── options/          # Training configurations
```

## Usage

### 1. Generate Training Data

```bash
# Configure parameters
cd generate_training_data/generate_parameter_maps
cp configs/example.yaml configs/custom.yaml
# Edit custom.yaml as needed

# Generate parameter maps
python generate2_csf.py --config configs/custom.yaml

# Generate MR signals
cd ../generate_mr_signal
python generate_MR_signal_pytorch_anima5_csf_IESvaried.py
```

### 2. Train Model

```bash
cd train
cp configs/example.yaml configs/custom.yaml
# Edit training configuration as needed

# Start training
python train_model.py --config configs/custom.yaml
```

### 3. Test Model

```bash
python test_model.py --weights path/to/weights --data path/to/test/data
```

## Methods

### Signal Decay Model

The model uses a discrete parametric model of T2 decay rates with three components:
- Myelin bound water
- Intra-extra axonal space bound water  
- Cerebro-spinal fluid free water

### Data Generation

1. Sample parameters from uniform distributions based on tissue types
2. Apply spatial transforms and augmentations:
   - Affine transformations
   - Non-rigid deformations
   - Partial volume effects
   - B1 field inhomogeneity

### Network Architecture

- U-Net based architecture
- 4 resolution blocks
- 3 CNN layers per block with ReLU activation
- 50% dropout after each CNN layer
- 64 feature maps in first layer
- 3x3 CNN kernels
- Max pooling between blocks

## Results

The model demonstrates:
- Superior noise robustness compared to conventional methods
- Consistent performance across different SNR levels
- Accurate myelin water fraction maps
- Low variance in estimates

See paper for detailed quantitative results.

## Citation

```bibtex
@inproceedings{vasylechko2022synthmap,
  title={SynthMap: a generative model for synthesis of 3D datasets for quantitative MRI parameter mapping of myelin water fraction},
  author={Vasylechko, Serge Didenko and Warfield, Simon K and Kurugol, Sila and Afacan, Onur},
  booktitle={MIDL},
  year={2022}
}
```

## License

This project is licensed under [LICENSE] - see the LICENSE file for details.

## Acknowledgments

This work was supported by:
- NIDDK, NIBIB, NINDS and NLM of the National Institutes of Health (R01DK125561, R21DK123569, R21EB029627, R01NS121657, R01LM013608, S10OD0250111)
- United States-Israel Binational Science Foundation (BSF) grant 2019056
- National Multiple Sclerosis Society pilot grant PP-1905-34002
