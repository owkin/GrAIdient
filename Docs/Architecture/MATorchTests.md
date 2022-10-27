# 👷 MATorchTests

`MATorchTests` contains integration tests that allow to compare `MAKit` models 
with their equivalent in `PyTorch`.

The goal is to demonstrate a good level of reproducibility and 
interoperability with `PyTorch`.

These tests are run when branches are merged to the release branches.
Once a PR is ready to be merged, please run these tests via the action trigger 
in GitHub.

## Setup

These tests require a special `Python` environment. 

```bash
conda create --name matorch python=3.7
conda activate matorch
cd Tests/MATorchTests/Base
pip install -e .
```

You should be able to run the tests right from XCode or 
with a `bash` command:

```bash
swift test --filter MATorchTests
```

You may eventually clean the environment with:

```bash     
conda deactivate
conda env remove --name matorch
```

## Steps 

1. Create a model in `MAKit` and `PyTorch`.
1. Get the weigths from the `PyTorch` model and load them  in the `MAKit` model.
1. Load data from `PyTorch` and set it on both models.
1. Compute forward, apply dummy loss then the backward pass.
1. Compare the gradient norm on the very first layer in both models.
