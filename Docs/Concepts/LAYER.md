# Layer

The `Layer` is the essential component of `MAKit`. 
A layer has 2 principle characteristics: an operation and a shape. 

In `MAKit` there is a strict correlation between the two. This enforces 
the prominent role of the layer in the framework at the expense of the 
operations as pure mathematical objects. 

Said differently, the layer is the only object that can 
be handled and appended to a model. A layer may be composed of many 
internal operations (for example the batch normalization) but these 
operations can not be mixed into any model.

## Shape

The layer shape is the equivalent of a PyTorch `Tensor`. It 
characterizes the layer internal state, the `neural structure`. 
For now there are only two available shapes: `1D` or `2D` that are visible 
in the layer name.

## API

The layer exposes 3 main APIs: 

- forward: the direct propagation of data from the first layers to the last ones
- backward: the retro propagation of gradients 
  from the last layers to the first ones
- forwardGC: the direct propagation of data to evaluate the gradients 
  during the gradient checking
  
Each of these API can be run in the CPU or GPU.

The CPU execution mode should only be run for debug or testing. 
The GPU execution mode is the official way to go. 

## Next Chapter

Next chapter: [GPU Model](GPU.md).
