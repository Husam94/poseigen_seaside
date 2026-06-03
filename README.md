# Seaside

Seaside is the core utility package in the Poseigen family.
It provides shared helpers for numerical workflows, metrics, and plotting, and serves as the foundation for other Poseigen packages.

## Features

Seaside is organized into three modules:

- Basics: general-purpose utilities for array handling, transformations, rolling-window operations, and data helpers.
- Metrics: evaluation functions for model and distribution-based scoring.
- Visuals: plotting utilities for line plots, heatmaps, and scatter-plus-histogram visualizations.

## Installation

Install from PyPI:

```bash
pip install poseigen_seaside
```

For local development, install from source using your preferred editable-install workflow.

## Usage

Import from the package namespace:

```python
import poseigen_seaside as seaside
```

Or import modules directly:

```python
from poseigen_seaside import basics, metrics, visuals
```

## Project Status

Seaside is in active development and is intended to be the shared base package for the Poseigen ecosystem.

## License

This project is released under the MIT License.

