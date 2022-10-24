# Training Flow

Let us consider a model containing the graph of layers:  

layer1 -> layer2 -> ... -> layer42

Let us suppose that the model links are initialized so as its "hard resources".
We want to go through the typical training flow: 

```swift
// Let us assume data is defined.
let data: [[Double]]
// Let us assume groundTruth is defined.
let groundTruth: [[Double]]
// Let us assume model is defined.
let model: Model

let firstLayer = layer1 as! Input1D
let lastLayer = layer42 as! MSE1D

// Set data.
try! firstLayer.setDataGPU(data)

// Reset gradient validity inside the kernel
// for backward pass and update the batch size
model.updateKernel(batchSize: data.count)

// Forward.
try! model.forward()

// Apply loss derivative.
try! lastLayer.lossDerivativeGPU(groundTruth)

// Backward.
try! model.backward()

// Update weights.
try! model.update()

// Get loss result.
let loss = try! lastLayer.getLossGPU(groundTruth)
```

First note that the `setData`, `lossDerivative` and `getLoss` are not exposed 
at the model level. They are specific to the concerned layer. As a low level 
component, beware not to forget the `GPU` in the API name.

Then note that the `setData` enables the forward pass while 
`lossDerivative` enables the backward pass. 

`getLoss` is a just an indicator but has no real role in the training flow.

## Next Chapter

Previous chapter: [Model](MODEL.md). \
Next chapter: [Optimizer](OPTIMIZER.md).
