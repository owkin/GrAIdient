
# MAKit

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 
[![tests](https://github.com/owkin/MAKit/actions/workflows/unit-tests.yml/badge.svg?branch=master)](
https://github.com/owkin/MAKit/actions/workflows/unit-tests.yml) 
[![tests](https://github.com/owkin/MAKit/actions/workflows/integration-tests.yml/badge.svg?branch=master)](
https://github.com/owkin/MAKit/actions/workflows/integration-tests.yml) 
[![tests](https://github.com/owkin/MAKit/actions/workflows/examples.yml/badge.svg?branch=master)](
https://github.com/owkin/MAKit/actions/workflows/examples.yml)

Maximal Activation Kit. \
A deep-learning framework for computer vision.

This framework aims at promoting full control and understanding of the main 
operations needed to train deep learning models in the computer vision area. 

## 📦 Swift Package Manager

Add the following dependency to your `Package.swift` manifest:

```swift
.package(url: "https://github.com/owkin/MAKit.git", .branch("main")),
```

## 📚 Documentation

The documentation is divided into several sections: 

- [Main Concepts](Docs/Concepts/CONCEPTS.md)
- [Architecture](Docs/Architecture/ARCHITECTURE.md)
- [Examples](Docs/Examples/EXAMPLES.md)

## 👨‍💻 Contributing

Read below to learn how to take part in improving MAKit.

### Changelog

All notable changes to MAKit will be documented in 
[this file](Docs/Contributing/CHANGELOG.md).

### Contributing Guide

Read our [contributing guide](Docs/Contributing/CONTRIBUTING.md) 
to learn about our development process 
and how to build and test your changes to MAKit.

### Code of Conduct

MAKit has adopted a Code of Conduct that we expect 
project participants to adhere to. 
Please read the [full text](Docs/Contributing/CODE_OF_CONDUCT.md)
so that you can understand what actions will and will not be tolerated.

### Licence

MAKit, MATestUtils and MAKitTests are [MIT licenced](LICENSE).

MAExamples and MATorchTests both depend on PythonKit and are 
[Apache 2.0 licensed](Tests/LICENSE). 
