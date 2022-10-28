# 📚 Plugin

It is possible to create a new repository that depends on MAKit and 
to create custom new layers and activation functions that extend the ones 
in MAKit.

## Create a New Layer

To create a custom layer, extend a base layer defined in 
`Layer1D.Base` or `Layer2D.Base`.

### Custom "First" Layer

Extend `LayerInput1D` or `LayerInput2D`.

The goal of such layers is to give access to API that sets data: 
`setDataCPU`, `setDataGPU`. 

Note that this layer could also be an intermediate layer. 
As such, the layer has access to `computeForward` which allows 
(when set to `true`) to rely on its own `layerPrev` layer's `forward` API 
to pass data. 
By default `computeForward = false` and the data must be set via the 
`setData` API.

### Custom "Last" Layer

Extend `LayerOutput1D`.

The goal of such layers is to give access to API that sets ground truth: 

- `collectGradientsApprox` to evaluate gradient estimations 
   during gradient checking
- `getLossCPU`, `getLossGPU` to retrieve the loss indicator
- `lossDerivativeCPU`, `lossDerivativeGPU` to initialize the backward pass 
  thanks to the derivative of loss
 
### Custom "Merge" Layer

Extend `LayerMerge2D`. 

The goal of such layers is to compute an operation that depends on 2 or 
more previous layers.

They get access to the `getMergedGraph` API to help define the 
`forwardGCCPU` and `forwardGCGPU` because of the specificity of the gradient 
of weights whose modifications have been populated before the common ancestor 
of the "merged graph" or after this common ancestor.

### Custom "Learning" Layer

Extend `LayerUpdate`.

The goal of such layers is to provide weights (and biases) 
whose gradients are computed 
during the `backward` pass and can be updated by an `Optimizer`.

In order to do that, one has to implement the different API of `LayerUpdate`: 

- `weightsCPU`, `weightsGPU`: get or set the weights 
- `computeDeltaWeights`: allows to change the training flow so gradients of 
  weights are not computed for this layer
- `accumulateDeltaWeights`: allows to accumulate the gradients of weights 
  through different steps
- `initWeightsCPU`, `initWeightsGPU`: initialize weights (randomly or 
  set by `weightsCPU` and `weightsGPU`)
- `collectWeightsCPU`, `collectWeightsGPU`: retrieve the different weights 
  (for updating them during training or comparing their gradients 
   with estimations during gradient checking)

### Plugin Layer 

Once the new layer, say `MyLayer` has been implemented, declare it on the 
registry so that it can be serialized/deserialized from the disk among any 
other layers of the model: 

```swift
MAKit.Model.Layer.append(registry: buildRegistry([
    MyLayer.self,
]))
```

If some new Metal operations have to be loaded to run `MyLayer`, declare 
them to `MetalKernel`: 

```swift
let metalLib = Bundle.main.url(
    forResource: "default",
    withExtension: "metallib"
)!
let listKernels =
[
    "myFunction1",
    "myFunction2"
]
MetalKernel.get.buildKernels(
    libraryURL: metalLib,
    kernelNames: listKernels
)
```

## Create a New Activation Function

Let us take an implementation example of a new activation function: 

```swift
import MAKit

public class TanH: ActivationFunction
{
    public static let str = "TanH"
    
    public override var forwardKernel: String
    {
        get {
            return "forwardTanH"
        }
    }
    public override var backwardKernel: String
    {
        get {
            return "backwardTanH"
        }
    }
    
    init()
    {
        super.init(TanH.str)
    }
    
    required public init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
    }
    
    public override func coeffInitWeights(nPrev: Int, nCur: Int) -> Double
    {
        sqrt(1.0 / Double(nPrev))
    }
    
    public override func apply(_ x: Double) -> Double
    {
        tanh(x)
    }
    
    public override func derivate(_ x: Double) -> Double
    {
        let fx = self.apply(x)
        return 1 - fx * fx
    }
}
```

Let us create a small factory for it: 

```swift
class ActivationKernelImpl: ActivationKernel
{
    static let _kernels: [String:ActivationKernelImpl] =
    [
        TanH.str: TanHKernel(),
    ]
    
    func build(_ name: String) -> ActivationFunction?
    {
        for (nameTmp, kernel) in ActivationKernelImpl._kernels
        {
            if nameTmp == name
            {
                return kernel.build()
            }
        }
        return nil
    }
    
    func build() -> ActivationFunction
    {
        fatalError("Not implemented.")
    }
}

private class TanHKernel: ActivationKernelImpl
{
    override func build() -> ActivationFunction
    {
        return TanH()
    }
}
```

We are now able to register this new activation function so that it can 
be used as the underlying function of `Activation1D` or `Activation2D`: 

```swift
// Enable to serialize/deserialize the activation function.
MAKit.Model.Activation.append(registry: buildRegistry([
    TanH.self,
]))
// Enable to use the activation function inside existing layers.
MAKit.Model.Activation.append(kernel: ActivationKernelImpl())
```

Let us see the example corresponding to the Metal kernels:

```C
#include <metal_stdlib>
using namespace metal;

kernel void forwardTanH(
    constant uint * pNbElems,
    device float * tmps,
    device float * outs,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    tmps[id] = outs[id];
    outs[id] = tanh(tmps[id]);
}

kernel void backwardTanH(
    const device float * tmps,
    constant uint * pNbElems,
    device float * delta,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    
    if (pNbElems)
    {
        nbElems = pNbElems[0];
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float tmp = tanh(tmps[id]);
    float derivative = 1.0 - tmp * tmp;
    delta[id] *= derivative;
}
``` 

We finally have to declare these operations to the `MetalKernel`: 

```swift 
let metalLib = Bundle.main.url(
    forResource: "default",
    withExtension: "metallib"
)!
let listKernels =
[
    // Activation
    "forwardTanH",
    "backwardTanH"
]
MetalKernel.get.buildKernels(
    libraryURL: metalLib,
    kernelNames: listKernels
)
```

## Previous Chapter

Previous chapter: [Gradient Checking](GRADIENT_CHECKING.md).
