# 📚 GPU Mode

In order to enforce the GPU execution mode, set this global variable: 

```swift
MAKit.Opti.GPU = true
```

Optionally, you can also set the following global variable in order to 
precise the list of external GPU to use 
(first in the list will be used with top priority): 

```swift
MAKit.Opti.gpuNamedPriority = ["<GPU name1>", "<GPU name2>", ...]
```

Then, it is time to initialize the GPU kernel: 

```swift
_ = MetalKernel.get
```

## Next Chapter

Previous chapter: [Layer](LAYER.md). \
Next chapter: [Model](MODEL.md).
