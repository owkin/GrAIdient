<br />

<p align="center">
    <img src="Docs/grAIdient.png" width="600">
</p>

--------------------------------------------------------------------------------

<div align="center">

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 
[![tests](https://github.com/owkin/GrAIdient/actions/workflows/unit-tests.yml/badge.svg?branch=main)](
https://github.com/owkin/GrAIdient/actions/workflows/unit-tests.yml) 
[![tests](https://github.com/owkin/GrAIdient/actions/workflows/integration-tests.yml/badge.svg?branch=main)](
https://github.com/owkin/GrAIdient/actions/workflows/integration-tests.yml) 
[![tests](https://github.com/owkin/GrAIdient/actions/workflows/examples.yml/badge.svg?branch=main)](
https://github.com/owkin/GrAIdient/actions/workflows/examples.yml)

</div>

<br />
Ready for the grAIt descent? <br />
<br />

GrAIdient is a framework that allows deep learning models to be developed 
using the internal GPU of a Mac, unlocking researchers to more easily train 
and run AI models on their own computers.

GrAIdient exposes the graph of layers, 
providing a unique way to design deep learning models for greater understanding, 
control and reproducibility. 

Though deeply grounded to the data driven pipeline, the goal is to challenge 
the very understanding of deep learning models and inject human intelligence 
where relevant; to transition from black box models to white box models, 
and all the gradients in between. 

Check out this toy [VGG example](Tests/GrAIExamples/VGGExample.swift) and 
its [documentation](Docs/Examples/VGG) to get started with GrAIdient today.

## ✨ Key Features

- flat design with direct access to the graph of layers and the backward pass
- run natively on Mac Intel GPU, eGPU, Mac M1, M2...
- compiled language
- gradient checking
- PyTorch interoperability
- gradients per batch & per sample (ready for differential privacy)
- debug at the neuron level

## 📦 Swift Package Manager

Add the following dependency to your `Package.swift` manifest:

```swift
.package(url: "https://github.com/owkin/GrAIdient.git", .branch("main")),
```

## 📚 Documentation

The documentation is divided into several sections: 

- [Main Concepts](Docs/Concepts/CONCEPTS.md)
- [Architecture](Docs/Architecture/ARCHITECTURE.md)
- [Examples](Docs/Examples/EXAMPLES.md)

## 👨‍💻 Contributing

Read below to learn how to take part in improving GrAIdient.

### Changelog

All notable changes to GrAIdient will be documented in the
[changelog](CHANGELOG.md).

### Contributing Guide

Read our [contributing guide](Docs/Contributing/CONTRIBUTING.md) 
to learn about our development process 
and how to build and test your changes to GrAIdient.

### Code of Conduct

GrAIdient has adopted a Code of Conduct that we expect 
project participants to adhere to. 
Please read the [full text](Docs/Contributing/CODE_OF_CONDUCT.md)
so that you can understand what actions will and will not be tolerated.

### Licence

GrAIdient, GrAITestsUtils and GrAITests are [MIT licenced](LICENSE).

GrAIExamples and GrAITorchTests both depend on PythonKit and are 
[Apache 2.0 licensed](Tests/LICENSE). 
