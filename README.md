# MAKit

Maximal Activation Kit. \
A deep-learning framework for computer vision.

This framework aims at promoting full control and understanding of the main 
operations needed to train deep learning models. 

## Layer

The `Layer` is the essential component of `MAKit`. 
A layer has 2 principle characteristics: an operation and a shape. 

In `MAKit` there is a strict correlation between the two. This enforces 
the prominent role of the layer in the framework at the expense of the 
operations as pure mathematical objects. 

Said differently, the layer is the only object that can 
be handled and appended to a model. A layer may be composed of many 
internal operations (for example the batch normalization) but these 
operations can not be mixed into any model.

### Shape

The layer shape is the equivalent of a PyTorch `Tensor`. It 
characterizes the layer internal state, the `neural structure`. 
For now there are only two available shapes: `1D` or `2D` that are visible 
in the layer name.

### API

The layer exposes 3 main APIs: 

- forward: the direct propagation of data from the first layers to the last ones
- backward: the retro propagation of gradients 
  from the last layers to the first ones
- forwardGC: the direct propagation of data to evaluate the gradients 
  during the gradient checking
  
Each of these API can be run in the CPU or GPU.

The CPU execution mode should only be run for debug or testing. 
The GPU execution mode is the official way to go. 

## GPU Mode

In order to enforce GPU execution model, use: 

```swift
MAKit.Opti.GPU = true
```

Optionally, you can specify external GPU to be used in a desired order 
(first in the list will be used with top priority): 

```swift
MAKit.Opti.gpuNamedPriority = ["<GPU name1>", "<GPU name2>", ...]
```

Then, in order to initialize the GPU kernel: 

```swift
_ = MetalKernel.get
```

For now we have considered the most low level elements of `MAKit`: the layers 
and their 3 main APIs which are in fact 6 if we consider the execution mode: 
CPU or GPU.

Let us wrap these different layers into a `Model`.

## Model

The `Model` is a convenient way to consider a list of layers. 
It allows to call the different APIs with the special logic of any 
deep learning model. 

- Forward pass logic: first layer in the list executes forward, then second
  executes forward, ... last layer executes forward
- ForwardGC pass logic: first layer in the list executes forwardGC, then second
  executes forwardGC, ... last layer executes forwardGC
- Backward pass logic: last layer in the list executes backward, ..., 
  second layer executes backward, first layer executes backward

Plus, as a convenient wrapping arount the layers, the `Model` routes the 
execution mode so that the CPU and GPU are only exposed 
in the `Layer` component.

So now, how do we create a model ?

### Graph of Layers

As we saw in the [previous paragraph](#model), the order in which the layers 
appear in the model is very important. This order has to be defined 
explicitly by the designer of the deep learning model. Once more, this enforces 
one model where layers are clearly chained from the beginning to the end 
without any operations being resolved at runtime.

The `ModelContext` is used in order to build this graph of layers. 
Once created, you are naturally editing a graph of layers where each layer 
that receives the context will be appended 
to the list of layers of the same model. 

<ins>Example</ins>: 

```swift
let context = ModelContext(name: "MyModel", models: [])
let params = MAKit.Model.Params(context: context)

var layer: Layer1D = Input1D(nbNeurons: 1, params: params)
layer = FullyConnected(
    layerPrev: layer, nbNeurons: 5,
    activation: ReLU.str, biases: true,
    params: params
)
layer = MSE1D(layerPrev: layer, params: params)

let model = context.model
```

Please note that the notion of graph of layers goes beyond the models. 
We could design a graph of layers composed of several models. 

<ins>Example</ins>: 

```swift
var context = ModelContext(name: "CNN", models: [])
var params = MAKit.Model.Params(context: context)

var layer: Layer2D = Input2D(
    nbChannels: 3, width: 256, height: 256, params: params
)
layer = Convolution2D(
    layerPrev: layer, size: 3, nbChannels: 64, stride: 1,
    activation: ReLU.str, biases: false, bn: true, params: params
)

let cnn = context.model

context = ModelContext(name: "Classifier", models: [cnn])
params = MAKit.Model.Params(context: context)

let head: Layer1D = FullyConnected(
    layerPrev: layer, nbNeurons: 1,
    activation: ReLU.str, biases: true, params: params
)
_ = MSE1D(layerPrev: head, params: params)

let classifier = context.model
```

### Layer `id` and `layerPrev`

Each layer has an `id` which must be unique in the graph of layers. 
That being said, we could have multiple copies of graph of layers in which 
the `id` are unique in their graph but duplicated in the other graphs.
This is typically handy for multi GPU setup.

<ins>Example</ins>: 

layerA1 -> layerA2 -> ... -> layerA42 \
layerB1 -> layerB2 -> ... -> layerB42 \
layerC1 -> layerC2 -> ... -> layerC42

where 
layerA1.id == 0, layerA2.id == 1, ... layerA42 == 41 \
layerB1.id == 0, layerB2.id == 1, ... layerB42 == 41 \
layerC1.id == 0, layerC2.id == 1, ... layerC42 == 41

The `id` serves as a way to find the links between the layers when we 
dump a model on the disk and try to load it later. 

This is a two step use case: we first have to load the model from the disk. 
We obtain a `BaseModel` with a list of layers where each layer's `id` is 
well defined but not their `layerPrev`. This base model does not expose the 
main APIs and we must create a `Model` out of this base model. During the 
creation of this model, the links `layerPrev` of the different layers 
will be initialized and the full API will be available for this model.

<ins>Example</ins>: \
Let us take the CNN and Classifier from the 
[previous paragraph](#graph-of-layers): 

```swift
let encoder = PropertyListEncoder()

var data = try! encoder.encode(cnn)
try! data.write(to: URL(fileURLWithPath: "/path/to/model1.plist"))
data = try! encoder.encode(classifier)
try! data.write(to: URL(fileURLWithPath: "/path/to/model2.plist"))

data = try! Data(contentsOf: URL(fileURLWithPath: "/path/to/model1.plist"))
let baseCNN = try! PropertyListDecoder().decode(
    BaseModel.self, from: data
)
data = try! Data(contentsOf: URL(fileURLWithPath: "/path/to/model2.plist"))
let baseClassifier = try! PropertyListDecoder().decode(
    BaseModel.self, from: data
)

let cnnFromDisk = Model(model: baseCNN, modelsPrev: [])
let classifierFromDisk = Model(model: baseClassifier, modelsPrev: [cnnFromDisk])
```

## Optimizer

### Optimizer Scheduler

### Variable Scheduler

## Gradient Checking

## How to Extend ?

### Create a New Layer

### Craete a New Activation Function

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

### Known Issues

The following error may follow on local setup: 
`UnpicklingError: invalid load key, ‘v’`.

This may occur when lfs data files are not pulled. 
To fix this: 

```bash     
git lfs pull
```

### Steps

1. Dump the training and testing datasets.
1. Evaluate a random model on the testing dataset: watch a bad performance.  
1. Train a model on the training dataset.
1. Evaluate the trained model on the testing dataset: 
   watch a better performance.
