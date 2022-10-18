# MAKit

Maximal Activation Kit. 
A deep-learning framework for computer vision.

It aims at promoting a total control and understanding of the main 
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

# MAExamples

`MAExamples` contains examples that show how to interact with `MAKit`. 
The examples are organised as tests so as to demonstrate some capabilities. 
They may be run in local or in the CI.

The following examples are currently available: 

- VGGExample

## VGG Example

### Setup

This example has some Python dependencies. In order to run 
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

You may eventually clean the environment with:

```bash     
conda deactivate
conda env remove --name maexamples
```
