#  🚀 LLM Example

This is the documentation for running 
[LLMs](../../Tests/GrAIExamples/LLMExample.swift) on the GPU.

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
[MistralAI](https://docs.mistral.ai/getting-started/open_weight_models/) 
and / or
[Llama](https://llama.meta.com/llama-downloads/) 
- Update `_modelPathMistral`, `_modelPathLlama2`, `_modelPathLlama3` in the 
[LLMExample](../../Tests/GrAIExamples/LLMExample.swift) file with the 
previous downloaded weights. 
- Optionnally update `_prompt`.
- Rename `_testGenerateMistral`, `_testGenerateLlama2` and `_testGenerateLlama3` 
into 
`testGenerateMistral`, `testGenerateLlama2` and `testGenerateLlama3`. 
- Run the tests.

It is finally possible to clean the environment 🌍

```bash     
conda deactivate
conda env remove --name graiexamples
```

## Steps

1. Generate text from a prompt with Mistral 7B Instruct model.
1. Generate text from a prompt with Llama 2 7B Chat model.
1. Generate text from a prompt with Llama 3 8B Instruct model.  

## Further tests

Further tests are available at 
[LLMExampleTests](../../Tests/GrAIExamples/LLMExampleTests.swift). 
In order to run them, rename 
`_testPredict1` and `_testPredict32` into `testPredict1` and `testPredict32`.

The test `testPredict1` compares the first step of generation 
of a toy LLM (just one transformer block) in GrAIdient and in PyTorch.   

The test `testPredict32` runs the first step of generation 
of a full LLM in GrAIdient and compares the expected result from PyTorch.
