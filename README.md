# MAKit

Maximal Activation Kit. 
A deep-learning framework for computer vision.

This framework aims at promoting a total control and understanding of the main 
operations needed to train deep learning models. 

The `Layer` is the essential component needed to build an explicit graph of 
operations from the data input to the ground truth. 

The `Model` is a component that wraps the different layers together to 
call functions in a simple way. 

The API explicitly exposes the following functions: 

- forward pass
- backward pass
- forward pass for the gradient checking.

## Model

## Layer

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
2. Get the weigths from the `PyTorch` model and load them  in the `MAKit` model.
3. Load data from `PyTorch` and set it on both models.
4. Compute forward, apply dummy loss then the backward pass.
5. Compare the gradient norm on the very first layer in both models.

# MAExamples

`MAExamples` contains examples that show how to interact with `MAKit`. 
The examples are organised as tests so as to demonstrate some capabilities. 
They may be run in local (Xcode or command line) 
or in the CI (release/main branches or explicit trigger).

The following examples are currently available: 

- VGGExample

## VGG Example

This example trains a simple model on the GPU. 
The dataset used is CIFAR 10. 
We want to train the model to discriminate between 2 labels 
(not the 10 available).

### Setup

This example has some `Python` dependencies. In order to run 
the example, we first have to setup the environment: 

```bash
conda create --name maexamples python=3.7
conda activate maexamples
cd Tests/MAExamples/Base
pip install -e .
```

Then you should be able to run the tests right from XCode or 
with a `bash` command:

```bash
swift test --filter MAExamples
```

Or to run the tests in the production model: 

```bash
swift test -c release --filter MAExamples
```

You may eventually clean the environment with:

```bash     
conda deactivate
conda env remove --name maexamples
```

### Steps

1. Dump the training and testing datasets.
2. Evaluate a random model on the testing dataset: watch a bad performance.  
3. Train a model on the training dataset.
4. Evaluate the trained model on the testing dataset: 
   watch a better performance.
