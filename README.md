
# MAKit

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](
https://github.com/owkin/MAKit/blob/main/LICENSE) 
[![tests](https://github.com/owkin/MAKit/actions/workflows/unit-tests.yml/badge.svg?branch=master)](
https://github.com/owkin/MAKit/actions/workflows/unit-tests.yml) 
[![tests](https://github.com/owkin/MAKit/actions/workflows/integration-tests.yml/badge.svg?branch=master)](
https://github.com/owkin/MAKit/actions/workflows/integration-tests.yml) 
[![tests](https://github.com/owkin/MAKit/actions/workflows/examples.yml/badge.svg?branch=master)](
https://github.com/owkin/MAKit/actions/workflows/examples.yml)

Maximal Activation Kit. \
A deep-learning framework for computer vision.

This framework aims at promoting full control and understanding of the main 
operations needed to train deep learning models. 

## Swift Package Manager

Add the following dependency to your `Package.swift` manifest:

```swift
.package(url: "https://github.com/owkin/MAKit.git", .branch("main")),
```

## Main Concepts

[Main Concepts](Docs/Concepts/CONCEPTS.md)

## Examples 

[Examples](Docs/Examples/EXAMPLES.md)

# MAKitTests

# MATorchTests

`MATorchTests` contains integration tests that allow to compare `MAKit` models 
with their equivalent in `PyTorch`.

The goal is to demonstrate a good level of reproducibility and 
interoperability with `PyTorch`.

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
