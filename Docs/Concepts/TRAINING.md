# 📚 Training Flow

Let us consider a model containing the following graph of layers:  

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

// Reset gradient validity for backward pass 
// and update the batch size.
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

Please note that the `setData`, `lossDerivative` and `getLoss` are not exposed 
at the model level. They are specific to some layers. 
So beware not to forget the `GPU` in the API name.

Also note how the `setData` enables the forward pass while 
`lossDerivative` enables the backward pass. 

Finally, `getLoss` is a just an indicator 
but has no real role in the training flow.

## Next Chapter

Previous chapter: [Model](MODEL.md). \
Next chapter: [Optimizer](OPTIMIZER.md).
