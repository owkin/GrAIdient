# 🚀 Auto Encoder Example

This is the documentation of a 
[toy Auto Encoder model](../../Tests/GrAIExamples/AutoEncoderExample.swift), 
trained on the GPU. 
The dataset used is CIFAR 10. 

We want to train the model to encode and generate images of ships (label 8).

Here is a subset of the data input images.

<table align="center" cellspacing="0" cellpadding="0">
    <tr>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_0.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_1.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_2.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_3.png"></td>
    </tr>
    <tr>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_4.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_5.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_6.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_7.png"></td>
    </tr>
    <tr>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_8.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_9.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_10.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_11.png"></td>
    </tr>
    <tr>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_12.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_13.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_14.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_15.png"></td>
    </tr>
</table>

## Setup

This example has some `Python` dependencies. In order to run 
the example, we first have to setup the environment: 

```bash
conda create --name graiexamples python=3.9
conda activate graiexamples
cd Tests/GrAIExamples/Base
pip install -e .
```

Now, let us run the tests from Xcode or a `bash` command (here with compiler 
optimization):

```bash
swift test -c release --filter GrAIExamples
```

It is finally possible to clean the environment 🌍

```bash     
conda deactivate
conda env remove --name graiexamples
```

## Steps

Each train example uses a `CIFARAutoEncoderTrainer`. 
The latter is responsible for initializing the training dataset 
before the actual training takes place.

1. Train a simple auto encoder model.
1. Train a UNet like auto encoder model.
1. Train a StyleGAN like auto encoder model.

## Further tests

Further tests are available at 
[AutoEncoderTests](../../Tests/GrAIExamples/AutoEncoderTests.swift).

The test `testTrain` compares the training of a `SimpleAutoEncoder` 
in GrAIdient and in PyTorch to show that the same `loss` is computed 
throughout the training.
