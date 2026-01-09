# PAM Bifurcation Embedding using Echo State Networks

This repository contains the code and data for predicting bifurcation patterns in Pneumatic Artificial Muscle (PAM) systems using Echo State Networks (ESN), as published in **AIP Chaos**.

## Overview

The project demonstrates **bifurcation embedding** of electrical resistance in PAM systems using reservoir computing. The method captures complex bifurcation patterns and nonlinear dynamics in soft mechanical actuators.

### Key Features

- **Objective**: Predict bifurcation patterns in PAM electrical resistance
- **Input signals**: Pressure control, Applied load
- **Output**: Electrical resistance prediction with bifurcation embedding
- **Method**: Echo State Network (ESN) with Ridge regression readout
- **Focus**: Bifurcation prediction and nonlinear dynamics modeling

## Repository Structure

```
├── ESNClasses.py                      # Core ESN implementation classes
├── ESN_PAM_bifurcation_embedding.ipynb  # Main Jupyter notebook demonstrating the method
└── data/
    └── PAM1_BifurcationData/
        ├── PAM_timeseries_load_change.txt  # Main experimental data
        ├── A8_bifurcation.txt              # Bifurcation data
        ├── A8_bifurcation_edit.txt         # Edited bifurcation data
        └── readme.txt                      # Data column descriptions
```

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

## Installation

```bash
pip install numpy scipy matplotlib
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/RemiAJR/PAM_bifurcation_embedding.git
   cd PAM_bifurcation_embedding
   ```

2. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook ESN_PAM_bifurcation_embedding.ipynb
   ```

The notebook demonstrates:
- Loading and preprocessing PAM experimental data
- Initializing ESN components (input layer, reservoir, output layer)
- Training the ESN on specific load conditions
- Predicting bifurcation diagrams across various loads
- Performance evaluation using RMSE and NRMSE metrics

## ESN Components

The `ESNClasses.py` module provides the following classes:

- **`Module`**: Base class for all ESN components
- **`Linear`**: Linear transformation layer for input processing
- **`ESN`**: Core Echo State Network reservoir with configurable parameters
- **`RidgeReadout`**: Ridge regression-based output layer for training

### Key ESN Parameters

- **Reservoir dimension**: 1000 neurons
- **Spectral radius**: 0.95
- **Activation function**: tanh with state transformation

## Data Format

The PAM experimental data (`PAM_timeseries_load_change.txt`) contains 33 columns with time-multiplexed measurements:
- Timestep
- Input pressure and load signals
- Resistance, length, and pressure measurements (5 multiplexing cycles)
- Control signals for pressure and load

See `data/PAM1_BifurcationData/readme.txt` for detailed column descriptions.

## Citation

If you use this code in your research, please cite the associated paper published in **AIP Chaos**.

## License

Please refer to the repository license for usage terms.
