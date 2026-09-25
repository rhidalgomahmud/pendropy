# PENDROPy

**PENDROPy** is a Python framework for the numerical analysis of pendant drops with elastic fluid–fluid interfaces.

This repository is part of the Master's Thesis:

> **Numerical modelling of elastic fluid–fluid interfaces: determination of the dilatational modulus from pendant-drop profiles**

It contains the numerical functions, representative datasets, and examples used for reference and elastic pendant-drop analysis.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/rhidalgomahmud/pendropy.git
cd pendropy
pip install -e .
```

## Repository structure

```text
pendropy/
├── pendropy/
│   └── functions/
│       ├── pendant_drop_functions.py
│       └── noise_functions.py
├── data/
│   ├── reference_profiles.h5
│   ├── reference_inverse.h5
│   ├── elastic_profiles.h5
│   └── elastic_inverse.h5
├── examples/
│   ├── example_usage.py
│   ├── read_reference_profiles.py
│   ├── read_reference_inverse.py
│   ├── read_elastic_profiles.py
│   └── read_elastic_inverse.py
├── README.md
└── pyproject.toml
```

## Usage

A complete example of the numerical workflow is provided in:

```text
examples/example_usage.py
```

It covers the reference forward and inverse problems, followed by the elastic forward and inverse analyses.

The `read_*.py` examples show how to access and visualise the provided HDF5 datasets.

A basic calculation can be performed as:

```python
import pendropy as pdp

result = pdp.reference_forward(
    Wo=0.6,
    V=20.0
)
```

## Data

The `data/` directory contains representative results generated for the Master's Thesis:

- `reference_profiles.h5` — reference pendant-drop profiles.
- `reference_inverse.h5` — reference inverse results.
- `elastic_profiles.h5` — elastic pendant-drop profiles.
- `elastic_inverse.h5` — elastic inverse results.
