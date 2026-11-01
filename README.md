# Seaside

[![PyPI version](https://img.shields.io/pypi/v/poseigen_seaside)](https://pypi.org/project/poseigen_seaside/)
[![Python versions](https://img.shields.io/pypi/pyversions/poseigen_seaside)](https://pypi.org/project/poseigen_seaside/)
[![License](https://img.shields.io/pypi/l/poseigen_seaside)](https://pypi.org/project/poseigen_seaside/)

Seaside is the core utility package in the Poseigen ecosystem. It provides shared helpers for numerical workflows, metrics, and plotting, and serves as the foundation for other Poseigen packages.

Part of the [Poseigen ecosystem](https://github.com/Husam94/poseigen).

## Installation

Requires Python 3.12 or newer. Install from PyPI:

```bash
python -m pip install poseigen_seaside
```

## Package

Seaside is organized into three modules:

- **`basics`** provides general-purpose utilities for array handling, transformations, rolling-window operations, and data helpers.
- **`metrics`** provides evaluation functions for model and distribution-based scoring.
- **`visuals`** provides plotting utilities for line plots, heatmaps, and scatter-plus-histogram visualizations.

## Usage

Import the modules you need:

```python
import poseigen_seaside.basics as se
import poseigen_seaside.metrics as mex
import poseigen_seaside.visuals as viz
```
