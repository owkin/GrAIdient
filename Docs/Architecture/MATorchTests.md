# 👷 MATorchTests

`MATorchTests` contains integration tests that allow to compare `MAKit` models 
with their equivalent in `PyTorch`.

The goal is to demonstrate a good level of reproducibility and 
interoperability with `PyTorch`.

These tests are run when branches are merged to the Release branches.
Once a PR is "ready to review", please run these tests via the action trigger 
in [GitHub](https://github.com/owkin/MAKit/actions) .

## Setup

These tests require a special `Python` environment. 

```bash
conda create --name matorch python=3.9
conda activate matorch
cd Tests/MATorchTests/Base
pip install -e .
```

Now, let us run the tests from Xcode or a `bash` command:

```bash
swift test --filter MATorchTests
```

It is finally possible to clean the environment 🌍

```bash     
conda deactivate
conda env remove --name matorch
```

## Steps 

1. Create a model in `MAKit` and `PyTorch`.
1. Get the weigths from the `PyTorch` model and load them  in the `MAKit` model.
1. Load data from `PyTorch` and set it on both models.
1. Compute forward, apply dummy loss then the backward pass.
1. Compare the gradient norm in the very first layer of both models.
