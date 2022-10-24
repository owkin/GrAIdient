# VGG Example

This example trains a simple model on the GPU. 
The dataset used is CIFAR 10. 
We want to train the model to discriminate between 2 labels 
(not the 10 available).

Here is a subset of images we find for the label 8 (ships) vs label 5 (dogs).

<table align="center" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none;">
    <tr>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_0.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_1.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_2.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_3.png" alt=""></td>
        <td> </td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_0.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_1.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_2.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_3.png" alt=""></td>
    </tr>
    <tr>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_4.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_5.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_6.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_7.png" alt=""></td>
        <td> </td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_4.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_5.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_6.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_7.png" alt=""></td>
    </tr>
    <tr>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_8.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_9.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_10.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_11.png" alt=""></td>
        <td> </td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_8.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_9.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_10.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_11.png" alt=""></td>
    </tr>
    <tr>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_12.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_13.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_14.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR8_15.png" alt=""></td>
        <td> </td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_12.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_13.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_14.png" alt=""></td>
        <td><img src="../../Tests/MAExamples/Base/data/out/CIFAR5_15.png" alt=""></td>
    </tr>
</table>

## Setup

This example has some `Python` dependencies. In order to run 
the example, we first have to setup the environment: 

```bash
conda create --name maexamples python=3.7
conda activate maexamples
cd Tests/MAExamples/Base
pip install -e .
```

Then you should be able to run the tests right from XCode or 
with a `bash` command:

```bash
swift test --filter MAExamples
```

Or to run the tests in the production model: 

```bash
swift test -c release --filter MAExamples
```

You may eventually clean the environment with:

```bash     
conda deactivate
conda env remove --name maexamples
```

## Known Issues

The following error may follow on local setup: 
`UnpicklingError: invalid load key, ‘v’`.

This may occur when lfs data files are not pulled. 
To fix this: 

```bash     
git lfs pull
```

## Steps

1. Dump the training and testing datasets.
1. Dump some images corresponding to the 2 labels 
   we want to discriminate against.
1. Evaluate a random model on the testing dataset: watch a bad performance.  
1. Train a model on the training dataset.
1. Evaluate the trained model on the testing dataset: 
   watch a better performance.
