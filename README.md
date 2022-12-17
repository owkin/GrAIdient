
# GrAIdient

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 
[![tests](https://github.com/owkin/GrAIdient/actions/workflows/unit-tests.yml/badge.svg?branch=main)](
https://github.com/owkin/GrAIdient/actions/workflows/unit-tests.yml) 
[![tests](https://github.com/owkin/GrAIdient/actions/workflows/integration-tests.yml/badge.svg?branch=main)](
https://github.com/owkin/GrAIdient/actions/workflows/integration-tests.yml) 
[![tests](https://github.com/owkin/GrAIdient/actions/workflows/examples.yml/badge.svg?branch=main)](
https://github.com/owkin/GrAIdient/actions/workflows/examples.yml)

Ready for the GrAIt descent ? \
GrAIdient is the grey area between black box and white box models. 

It aims at promoting full control and understanding of the main 
operations needed to train deep learning models in the computer vision area. 

Disclaimer: If you cannot find an operation that suits your needs, 
please consider building it into the GrAIdient framework. 

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
[changelog](Docs/Contributing/CHANGELOG.md).

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

GrAIdient, MATestUtils and GrAITests are [MIT licenced](LICENSE).

GrAIExamples and MATorchTests both depend on PythonKit and are 
[Apache 2.0 licensed](Tests/LICENSE). 
