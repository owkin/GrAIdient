# 👷 MAKitTests

`MAKitTests` contains unit tests of `MAKit`. 

Some of them are systematic and repeated over many different setups to ensure 
that every layer, optimizer, activation function ... is tested.

## Systematic Tests

- gradient checking tests: compare gradients of weights 
  computed through `backward` to an estimation computed through `forwardGC`
  
- flow tests: compare gradients computed in CPU and GPU

- reset tests: compare gradients computed in CPU and GPU after a reset

- reverse tests: compare gradients computed in CPU and GPU after exchanging 
  the execution context 
  (the model CPU will be executed on the GPU and vice versa)
  
- inference tests: compare loss in CPU and GPU during the inference phase

- load tests: compare loss in CPU and GPU after loading models from the disk

- transform tests: compare loss in CPU and GPU after applying transforms on 
  the model
  
- norm tests: test that we can clip the gradients of weights

## Other Tests 

Other tests are not repeated over many different setups. 
They concern some specific components:

- `MetalKernelTests`: ensure that every Metal kernel builds correctly
- `OptimizerVariablesTests`: test scheduler for optimizers and variables
- `UpdateManagementTests`: test we can control the training flow
