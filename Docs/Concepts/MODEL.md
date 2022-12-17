# 📚 Model

The `Model` is a convenient way to consider a list of layers. 
It allows to call the different APIs with the special logic of any 
deep learning model. 

- Forward pass logic: first layer in the list executes forward, then second
  executes forward, ... last layer executes forward
- ForwardGC pass logic: first layer in the list executes forwardGC, then second
  executes forwardGC, ... last layer executes forwardGC
- Backward pass logic: last layer in the list executes backward, ..., 
  second layer executes backward, first layer executes backward

As a container of layers, the `Model` also routes the 
execution mode, taking into account the `GrAI.Opti.GPU` global variable 
defined in the [previous chapter](GPU.md). 
This way, CPU and GPU are only exposed in the `Layer` component.

So now, how do we create a model ?

## Graph of Layers

As we saw in the [previous paragraph](#-model), the order in which the layers 
appear in the model is very important. This order has to be defined 
explicitly by the designer of the deep learning model. This enforces 
one model where layers are clearly chained from the beginning to the end 
without any hidden operations occuring at runtime.

The `ModelContext` is used to build this graph of layers. 
Once created, the model context object allows to edit the graph of layers: 
the model context is given to every newly created layers as a parameter 
for these layers to be appended to the list of layers of the model context. 

<ins>Example</ins>: 

```swift
let context = ModelContext(name: "MyModel", models: [])
let params = GrAI.Model.Params(context: context)

// First layer to be appended to the list of layers of the context model.
var layer: Layer1D = Input1D(nbNeurons: 1, params: params)

// Second layer to be appended.
layer = FullyConnected(
    layerPrev: layer, nbNeurons: 5,
    activation: ReLU.str, biases: true,
    params: params
)

// Last layer.
layer = MSE1D(layerPrev: layer, params: params)

// Get back the model that actually contains the list of layers.
let model = context.model
```

Please note that the notion of graph of layers can include several models. 
This will prove usefull for complex training such as GAN.

<ins>Example</ins>: 

```swift
// Create first model context.
var context = ModelContext(name: "CNN", models: [])
var params = GrAI.Model.Params(context: context)

var layer: Layer2D = Input2D(
    nbChannels: 3, width: 256, height: 256, params: params
)
layer = Convolution2D(
    layerPrev: layer, size: 3, nbChannels: 64, stride: 1,
    activation: ReLU.str, biases: false, bn: true, params: params
)

// Get our first model.
let cnn = context.model

// Note that cnn is a dependency for our second model context.
context = ModelContext(name: "Classifier", models: [cnn])
params = GrAI.Model.Params(context: context)

let head: Layer1D = FullyConnected(
    layerPrev: layer, nbNeurons: 1,
    activation: ReLU.str, biases: true, params: params
)
_ = MSE1D(layerPrev: head, params: params)

// Get our second model.
let classifier = context.model
```

## Initialize Links

Each layer has an `id` which must be unique in the graph of layers. 
That being said, we could have multiple copies of graph of layers in which 
the `id` are unique in their graph but duplicated in the other graphs.
This is typically handy for a multi GPU setup.

<ins>Example</ins>: 

layerA1 -> layerA2 -> ... -> layerA42 \
layerB1 -> layerB2 -> ... -> layerB42 \
layerC1 -> layerC2 -> ... -> layerC42

where: \
layerA1.id == 0, layerA2.id == 1, ... layerA42 == 41 \
layerB1.id == 0, layerB2.id == 1, ... layerB42 == 41 \
layerC1.id == 0, layerC2.id == 1, ... layerC42 == 41

The `id` serves as a way to match the links between the layers. 
One could have stored the direct links `layerPrev` instead of the `id` but 
then we would have issues to serialize and deserialize the model. 

Thus, initializing the links is a two steps process.

1. We first deal with a `BaseModel` object. This object may be 
directly retrieved through the model context 
(example: `let classifier = context.model` 
in the [previous paragraph](#graph-of-layers)) or loaded from the disk 
(we will see an example of that later). This base model consists in 
a list of layers where each layer's `id` is 
well defined but not their `layerPrev`. This base model does not expose the 
main APIs. 

1. In order to get access to the main APIs we must create a `Model` 
out of the previous base model. During the 
creation of this model, the links `layerPrev` of the different layers 
will be initialized and the full API will become available.

<ins>Example</ins>: \
Let us take the CNN and Classifier from the 
[previous paragraph](#graph-of-layers): 

Let us imagine the two models have been trained and we want to save them 
to the disk: 

```swift
// Use JSONEncoder to encode a readable file 
// or PropertyListEncoder to encode a binary file 
// (takes less space on the disk).
let encoder = JSONEncoder()

// Encode first model.
var data = try! encoder.encode(cnn)
// Save first model to the disk.
try! data.write(to: URL(fileURLWithPath: "/path/to/model1.plist"))

// Encode second model.
data = try! encoder.encode(classifier)
// Save second model to the disk.
try! data.write(to: URL(fileURLWithPath: "/path/to/model2.plist"))
```

Then we want to be able to load the 2 models from the disk and get access 
to the full `Model` API:

```swift
// Load the data of our first model from the disk.
data = try! Data(contentsOf: URL(fileURLWithPath: "/path/to/model1.plist"))
// Initialize a base model out of it. 
// Use PropertyListDecoder instead to decode a binary file.
let baseCNN = try! JSONDecoder().decode(
    BaseModel.self, from: data
)

// Load the data of our second model from the disk.
data = try! Data(contentsOf: URL(fileURLWithPath: "/path/to/model2.plist"))
// Initialize a base model ouf of it.
// Use PropertyListDecoder instead to decode a binary file.
let baseClassifier = try! JSONDecoder().decode(
    BaseModel.self, from: data
)

// Initialize the links of our first model.
let cnn = Model(model: baseCNN, modelsPrev: [])
// Initialize the links of our second model, taking into account 
// its dependency on the first model.
let classifier = Model(model: baseClassifier, modelsPrev: [cnn])
```

## Initialize Weights

Once the links have been initialized, it is nearly time 
to train or run the model in the GPU execution mode 
(or just test/debug on the CPU). 

### Set Model's Weights

One of the `Model`'s APIs is `weights`. It allows to set the values 
of weights and biases for the different layers of the model. 

Yet, it does not load these values into the layers. Consider it as a cache 
for later use: 

```swift
cnn.weights = myCNNWeights
classifier.weights = myClassifierWeights
```

### Model Loaded from the Disk

When a model has been loaded from the disk 
(see [previous paragraph](#initialize-links)), there is no need to use 
the `weights` API: the cache for the weights and biases values is already set 
with the values loaded from the disk.

### Initialize "Hard Resources"
 
The last thing to do is to initialize the "hard resources".
These are resources that may be time consuming to initialize 
depending on the size of the model:  

- the weights and biases
- the batch normalization (weights, biases and stats) 

To trigger the process: 

```swift
cnn.initKernel(.Training)
classifier.initKernel(.Training)
``` 

or 

```swift
cnn.initKernel(.Inference)
classifier.initKernel(.Inference)
``` 

What happens is that the cache that has been set earlier will now 
be fully loaded into the kernel of the different layers. 

- CPU mode: the weights, biases... will be loaded into 
  the layers' objects needed for computing the forward and the backward pass
- GPU mode: the weights, biases... will be uploaded 
  to the GPU device

So now, what would have happened if the cache for weights and biases had 
not been set earlier ?

=> The values for weights would have been initialized "randomly" 
while the values for biases would have been initialized to 0. 

To cap it all, the `weights` API is not necessary in the following situations: 

- The model has been loaded from the disk 
- We want to train a model from scratch

But the `initKernel` API is always necessary for the model to be ready to 
train/run.

## Model Transformation

In some scenario, we need to transform the model and preserve the 
"hard resources" to avoid losing time initializing resources that 
are independent of the transformation concerned. Hence, the use of the 
`inPlace` parameter.

<ins>Example</ins>: 

```swift
newCNN = Model.resize(
    models: [cnn],
    imageWidth: 256,
    imageHeight: 256,
    inPlace: true
)[0]
```

In this situation, we do not need to call the `initKernel` API.

## Next Chapter

Previous chapter: [GPU Mode](GPU.md). \
Next chapter: [Training Flow](TRAINING.md).
