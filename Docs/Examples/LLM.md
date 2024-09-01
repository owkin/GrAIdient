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
- Download weights from 
[MistralAI](https://docs.mistral.ai/getting-started/open_weight_models/) 
(mistral-7B-Instruct-v0.3)
and / or
[Llama](https://llama.meta.com/llama-downloads/) 
(llama-2-7b-chat or Meta-Llama-3-8B-Instruct) 
and / or Gemma2 from [HuggingFace](https://huggingface.co/google/gemma-2-2b-it) 
(Gemma-2-2b-it).
- Update `_modelPathMistral`, `_modelPathLlama2`, `_modelPathLlama3`, 
`_modelPathGemma2` in the 
[LLMExample](../../Tests/GrAIExamples/LLMExample.swift) file with the 
previous downloaded weights. 
- Optionnally update `_prompt`.
- Rename `_testGenerateMistral`, `_testGenerateLlama2`, `_testGenerateLlama3` 
and `_testGenerateGemma2`
into 
`testGenerateMistral`, `testGenerateLlama2`, `testGenerateLlama3` and 
`testGenerateGemma2`. 
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
1. Generata text from a prompt with Gemme 2 2B Instruct model.

## Further tests

Further tests are available at 
[LLMExampleTests](../../Tests/GrAIExamples/LLMExampleTests.swift). 
In order to run them, rename 
`_testPredict1` and `_testPredict32` into `testPredict1` and `testPredict32`.

The test `testPredict1` compares the first step of generation 
of a toy LLM (just one transformer block) in GrAIdient and in PyTorch.   

The test `testPredict32` runs the first step of generation 
of a full LLM in GrAIdient and compares the expected result from PyTorch.
