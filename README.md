<img src="hypercomp_logo.png" alt="HyPerComp Inc." width="200"/>

[HyPerComp Inc.](https://www.hypercomp.net)

---

# CV-DeepONet: Complex-Valued Deep Operator Network for 3D Maxwell's Equations

This repository contains the code and data for training a Complex-Valued Deep Operator Network (CV-DeepONet) as a surrogate model for time-harmonic Maxwell's equations. Two benchmark cases are provided: scattering by a metallic sphere and scattering by a metallic almond-shaped target.

## Paper

If you use this code or data, please cite:

> Qile Jiang, Marc Salvadori, Dale Ota, Vijaya Shankar, Khemraj Shukla. *Complex valued Deep Operator Network (DeepONet) for three dimensional Maxwell's equations: G in C^{m x n}.* Journal of Computational Physics, vol. 562, p. 114993, 2026. https://doi.org/10.1016/j.jcp.2026.114993

The paper is available at: https://www.sciencedirect.com/science/article/pii/S0021999126003463

## Acknowledgements

This work was supported by [HyPerComp Inc.](https://www.hypercomp.net)

The complex-valued neural network layers are built using the [cvnn](https://pypi.org/project/cvnn/) library.

## Repository Structure

```
CV_DeepONet/
├── sphere/                   # Sphere scattering case
│   ├── maxwell_cvnn.py       # Model definition and training
│   ├── data_handler.py       # Data loading utilities
│   ├── utils.py              # Logging and helper functions
│   ├── create_input_file.py  # Script to regenerate input_file.yaml
│   ├── input_file.yaml       # Hyperparameter configuration
│   └── data/
│       ├── data_configs.txt  # Mesh configuration info
│       └── pec/              # Field data files (101 frequencies)
│
└── almond/                   # Almond scattering case
    ├── almond.py             # Model definition and training
    ├── almond_l2_joint.py    # Variant with L2 joint loss
    ├── input_file.yaml       # Hyperparameter configuration
    ├── input_file_l2.yaml    # Configuration for L2 joint variant
    └── E-H_files/            # Field data files (61 frequencies)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Sphere case

```bash
cd sphere
python maxwell_cvnn.py
```

The script reads the configuration from `input_file.yaml` and loads field data from `data/pec/`. To modify the hyperparameters, edit `input_file.yaml` directly or run `python create_input_file.py` to regenerate it.

### Almond case

```bash
cd almond
python almond.py
```

For the L2 joint loss variant:

```bash
cd almond
python almond_l2_joint.py
```

The almond scripts read from `input_file.yaml` (or `input_file_l2.yaml` for the joint variant) and load field data from `E-H_files/`.

## Data

Each case provides E and H field data computed by a frequency-domain solver over a range of frequencies on a fixed mesh. The sphere data covers 101 frequency samples from 0.05 to 0.6 GHz, and the almond data covers 61 frequency samples over a similar range. The network is trained to interpolate the scattered fields at arbitrary query frequencies and spatial locations.
