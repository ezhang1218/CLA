# CLA — Contrastive Liquid Association

Contrastive Liquid Association (CLA) is a dimension reduction (DR) method for high-dimensional datasets with **multiple or continuous treatments**. Unlike standard contrastive DR, which is limited to binary case-control comparison, CLA identifies the latent direction of variation that changes systematically with treatment intensity relative to the control group. 

## Repository Structure

```
.
├── data/               # Preprocessed datasets for T4D7_BC and Senescent experiments
├── main.py             # Core CLA implementation
├── simulations.py      # Simulation studies
├── T47D_BC.ipynb       # Application: T47D breast cancer cells
├── LPS_Nutlin.ipynb    # Application: Liposarcoma cells (Nutlin-3a)
├── LPS_Abema.ipynb     # Application: Liposarcoma cells (Abemaciclib)
└── Senescent.ipynb     # Application: Human epithelial senescence time-course
```

## Quickstart

```python
import numpy as np
from main import cla, plot_cla_eigenvectors

# X: (n, p) feature matrix
# Z: (n,) treatment vector — 0 for control, >0 for treated
T, evals, evecs = cla(X, Z, center=True)

plot_cla_eigenvectors(evecs, feature_names, n_components=2)
```

## Requirements

```
pip install numpy scipy matplotlib anndata scanpy
```
