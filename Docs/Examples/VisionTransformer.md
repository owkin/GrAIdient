# 🚀 Vision Transformer Example

This is the documentation of a [toy Vision Transformer model](../../Tests/GrAIExamples/VisionTransformer.swift), 
trained on the GPU. 
The dataset used is CIFAR 10. 

We want to train the model to discriminate between 2 labels 
(not the 10 available of the CIFAR 10 dataset): the 8 (ships) or the 5 (dogs).

Here is a subset of images we find for the label 8 (ships) vs label 5 (dogs).

<table align="center" cellspacing="0" cellpadding="0">
    <tr>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_0.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_1.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_2.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_3.png"></td>
        <td> </td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_0.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_1.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_2.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_3.png"></td>
    </tr>
    <tr>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_4.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_5.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_6.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_7.png"></td>
        <td> </td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_4.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_5.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_6.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_7.png"></td>
    </tr>
    <tr>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_8.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_9.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_10.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_11.png"></td>
        <td> </td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_8.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_9.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_10.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_11.png"></td>
    </tr>
    <tr>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_12.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_13.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_14.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR8_15.png"></td>
        <td> </td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_12.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_13.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_14.png"></td>
        <td><img src="../../Tests/GrAIExamples/Base/data/out/CIFAR5_15.png"></td>
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

1. Dump the training dataset.  
1. Train a simple Vision Transformer model on the training dataset.
