# 🚀 VGG Example

This is the documentation of a [toy VGG model](../../Tests/GrAIExamples/VGGExample.swift) 
(simplified from the VGG architecture), trained on the GPU. 
The dataset used is CIFAR 10. 

We want to train the model to discriminate between 2 labels 
(not the 10 available of the CIFAR 10 dataset): the 8 (ships) or the 5 (dogs).

Here is a subset of images we find for the label 8 (ships) vs label 5 (dogs).

<table align="center" cellspacing="0" cellpadding="0">
    <tr>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_0.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_1.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_2.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_3.png"></td>
        <td> </td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_0.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_1.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_2.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_3.png"></td>
    </tr>
    <tr>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_4.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_5.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_6.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_7.png"></td>
        <td> </td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_4.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_5.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_6.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_7.png"></td>
    </tr>
    <tr>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_8.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_9.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_10.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_11.png"></td>
        <td> </td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_8.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_9.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_10.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_11.png"></td>
    </tr>
    <tr>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_12.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_13.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_14.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR8_15.png"></td>
        <td> </td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_12.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_13.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_14.png"></td>
        <td><img src="../../Tests/data/out/cifar-10/CIFAR5_15.png"></td>
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

1. Dump the training and testing datasets.
1. Dump some images corresponding to the 2 labels 
   we want to discriminate against.
1. Evaluate a random model on the testing dataset: watch a bad performance.  
1. Train a model on the training dataset.
1. Evaluate the trained model on the testing dataset: 
   watch a better performance.
   
## Benchmarks

To benchmark the time performance of the a VGG like model, look at 
[VGGBenchmark](../../Tests/GrAIExamples/VGGBenchmark.swift) and rename 
`_test_TrainVGG` and `_test_EvalVGG` into `test_TrainVGG` and `test_EvalVGG`.

The test `test_TrainVGG` will measure the time spent for training the VGG 
model for 20 steps.

The test `test_EvalVGG` will measure the time spent for running the VGG model 
in inference for 20 steps.

Note that for both tests, the data is random and fixed once and for all.
