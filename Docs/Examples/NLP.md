#  🚀 NLP Example

This is the documentation for running 
[LLMs](../../Tests/GrAIExamples/NLPExample.swift) on the GPU.

## Setup

This example has some `Python` dependencies. In order to run 
the example, we first have to setup the environment: 

```bash
conda create --name graiexamples python=3.9
conda activate graiexamples
cd Tests/GrAIExamples/Base
pip install -e .
```

Then: 
- download weights from 
[MistralAI](https://docs.mistral.ai/getting-started/open_weight_models/).
- Update `_modelPath` in the 
[NLPExample](../../Tests/GrAIExamples/NLPExample.swift) file with the 
previous downloaded weights. 
- Optionnally update `_prompt`.
- Rename `_testGenerate` into `testGenerate`. 
- Run the test.

It is finally possible to clean the environment 🌍

```bash     
conda deactivate
conda env remove --name graiexamples
```

## Steps

1. Generate text from a prompt.   

## Further tests

Further tests are available at 
[NLPExampleTests](../../Tests/GrAIExamples/NLPExampleTests.swift). 
In order to run them, rename 
`_testPredict1` and `_testPredict32` into `testPredict1` and `testPredict32`.

The test `testPredict1` compares the first step of generation 
of a toy LLM (just one transformer block) in GrAIdient and in PyTorch.   

The test `testPredict32` runs the first step of generation 
of a full LLM in GrAIdient and compares the expected result from PyTorch.
