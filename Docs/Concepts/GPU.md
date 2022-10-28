# 📚 GPU Mode

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

Let us wrap our different layers into a `Model`.

## Next Chapter

Previous chapter: [Layer](LAYER.md). \
Next chapter: [Model](MODEL.md).
