# 📚 Layer

The `Layer` is the essential component of `GrAIdient`. 
A layer has 2 principle characteristics: an operation and a shape. 

In `GrAIdient` there is a strict correlation between the two. This enforces 
the prominent role of the layer in the framework at the expense of the 
operations as pure mathematical objects. 

Said differently, the layer is the only object that can 
be handled by a model. A layer may be composed of many 
internal operations (for example the batch normalization) but these 
operations can not be mixed into any model without being previously wrapped 
inside a layer.

## Shape

The layer shape is the equivalent of a PyTorch `Tensor`. It 
characterizes the layer internal state, its `neural structure`. 
For now there are only two available shapes: `1D` or `2D`. These suffixes 
most of the time appear in the layer's name itself: 
example `Layer1D`, `Layer2D`.

## API

The layer exposes 3 main APIs: 

- `forward`: the direct propagation of data 
- `backward`: the retro propagation of gradients 
- `forwardGC`: the direct propagation of data to evaluate the gradients 
  during the gradient checking
  
Each of these API can be run in two execution contexts: CPU or GPU. 

As a low-level component, the layer exposes this execution context 
in the name of the 3 previous APIs, which in fact results in the 6 final APIs: 
`forwardCPU`, `forwardGPU`, 
`backwardCPU`, `backwardGPU`, `forwardGCCPU` and `forwardGCGPU`.

The CPU execution mode should only be run for debug or testing. 
The GPU execution mode is the standard way to go. 

## Next Chapter

Next chapter: [GPU Mode](GPU.md).
