# GPU Mode

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

## Next Chapter

Previous chapter: [Layer](LAYER.md). \
Next chapter: [Model](MODEL.md).
