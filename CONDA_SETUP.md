# Conda Environment Setup Guide

## Understanding the Files

### `requirements.txt`

**What it does:**

- A simple text file listing Python package dependencies
- Used by `pip` (Python's default package installer)
- Format: one package per line with optional version constraints
- Example: `numpy>=1.24.0` means "install numpy version 1.24.0 or higher"

**When to use:**

- Simple Python environments
- Installing dependencies with `pip`
- CI/CD pipelines
- Docker containers

**How to use:**

```bash
pip install -r requirements.txt
```

### `setup.py`

**What it does:**

- Python package installation script using `setuptools`
- Makes your project installable as a Python package
- Defines package metadata (name, version, author, etc.)
- Specifies dependencies, entry points, and package structure
- Allows installing your project in "editable" mode: `pip install -e .`

**When to use:**

- Installing your project as a package
- Distributing your code to others
- Creating command-line tools (entry_points)

**How to use:**

```bash
# Install in editable mode (changes to code are immediately available)
pip install -e .

# Or regular install
pip install .
```

### `environment.yml` (Conda-specific)

**What it does:**

- Conda environment configuration file
- Defines both conda and pip dependencies
- Specifies Python version and conda channels
- Can recreate entire environment from scratch

**When to use:**

- Creating conda environments
- Reproducible scientific computing environments
- Managing complex dependencies with C/C++ libraries

**How to use:**

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate multiagent-system
```

---

## Conda Setup Instructions

### Option 1: Using `environment.yml` (Recommended)

```bash
# 1. Navigate to project directory
cd /path/to/Multiagent-System

# 2. Create conda environment from file
conda env create -f environment.yml

# 3. Activate the environment
conda activate multiagent-system

# 4. Verify installation
python --version  # Should show Python 3.9+
python -c "import numpy, pandas, torch; print('All packages installed!')"
```

### Option 2: Manual Conda Setup

```bash
# 1. Create a new conda environment with Python 3.9+
conda create -n multiagent-system python=3.9

# 2. Activate the environment
conda activate multiagent-system

# 3. Install conda packages (prefer conda-forge channel)
conda install -c conda-forge numpy pandas scipy matplotlib seaborn scikit-learn

# 4. Install pip-only packages
pip install -r requirements.txt
```

### Option 3: Using `setup.py` with Conda

```bash
# 1. Create conda environment with Python
conda create -n multiagent-system python=3.9 pip

# 2. Activate environment
conda activate multiagent-system

# 3. Install your project and all dependencies
pip install -e .

# This automatically:
# - Reads requirements.txt
# - Installs all dependencies
# - Installs your project in editable mode
```

---

## Managing the Conda Environment

### Useful Commands

```bash
# List all environments
conda env list

# Activate environment
conda activate multiagent-system

# Deactivate environment
conda deactivate

# Update environment from updated environment.yml
conda env update -f environment.yml --prune

# Export current environment to file
conda env export > environment.yml

# Remove environment (if needed)
conda env remove -n multiagent-system

# List installed packages in environment
conda list

# Install additional packages
conda install package-name
# or
pip install package-name
```

---

## Troubleshooting

### Issue: Package not found

```bash
# Add conda-forge channel (often has more packages)
conda config --add channels conda-forge

# Then try installing again
conda install package-name
```

### Issue: Version conflicts

```bash
# Update conda
conda update conda

# Try installing specific version
conda install package-name=version

# Or use pip for that specific package
pip install package-name==version
```

### Issue: CUDA/GPU support for PyTorch

```bash
# For CUDA-enabled PyTorch, install separately
# Check https://pytorch.org/get-started/locally/ for your system
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

## Quick Start Summary

```bash
# One-liner setup (recommended)
conda env create -f environment.yml && conda activate multiagent-system

# Then run your experiments
python examples/quick_start.py
```

---

## Key Differences Summary

| Feature           | requirements.txt  | setup.py             | environment.yml          |
| ----------------- | ----------------- | -------------------- | ------------------------ |
| **Purpose**       | List dependencies | Install as package   | Define conda env         |
| **Installer**     | pip               | pip/setuptools       | conda                    |
| **Package types** | Python only       | Python only          | Python + conda packages  |
| **Use case**      | Simple deps       | Package distribution | Scientific computing     |
| **Complexity**    | Simple            | Medium               | Complex (handles C deps) |

---

## Recommendation

For this project:

- **Use `environment.yml`** if you want conda's dependency resolution and scientific computing packages
- **Use `setup.py`** if you want to install your project as a package: `pip install -e .`
- **Use `requirements.txt`** if you prefer pip and are comfortable managing dependencies manually

Most users should use: **`conda env create -f environment.yml`** ✅
