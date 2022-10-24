# MAKitTests

`MAKitTests` contains unit tests of `MAKit`: 

- gradient checking tests: compare gradients of weights 
  computed through `backward` to an estimation computed through `forwardGC`
  
- flow tests: compare gradients computed in CPU and GPU

- reset tests: compare gradients computed in CPU and GPU after a reset

- reverse tests: compare gradients computed in CPU and GPU after exchanging 
  the execution context 
  (the model CPU will be executed on the GPU and vice versa)
  
- inference tests: compare loss in CPU and GPU in the inference phase

- load tests: compare loss in CPU and GPU after loading models from the disk

- transform tests: compare loss in CPU and GPU after applying transforms on 
  the model
  
- norm tests: test that we can clip the gradient of weights

These tests are running on different setup to ensure that: 
every layer, optimizer, activation function ... is tested.
