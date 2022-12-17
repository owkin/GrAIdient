# 📚 Optimizer

The `Optimizer` is responsible for using the gradients of weights 
in order to update the weights values in the different layers that have them. 

The gradients of weights are computed during the backward pass.

## Optimizer Scheduler

Here, we setup the optimizer of a model: 

```swift 
var optimizerParams: GrAI.Optimizer.Params()
// Initialize one optimizer for the whole training loop: Adam.
optimizerParams.optimizer = ConstEpochsScheduler(.Adam)

model.setupOptimizers(params: optimizerParams)
```

It is also possible to setup optimizers that are scheduled upon 
the training loop current step: 

```swift 
// Let us assume nbLoops is defined: the number of steps per epoch.
let nLoops: Int 
let epoch1 = 5
let epoch2 = 10

var optimizerParams: GrAI.Optimizer.Params()
optimizerParams.nbLoops = nbLoops
// Initialize optimizers Adam and SGD to be used according to the current step 
// of the training loop.
optimizerParams.optimizer = ListEpochsScheduler(
    epochs: [epoch1, epoch2],
    optimizers: [.Adam, .SGD]
)

model.setupOptimizers(params: optimizerParams)
```

If defined so, do not forget to inform the optimizer about the step 
evolution during the training loop: 

```swift 
model.step()
```

## Variable Scheduler

The different variables that are defined for the optimizers are 
visible in the `OptimizerImpl` component.

For example at `OptimizerImpl.initVariables` the variables `alpha` and 
`lambda` are initialized with default values. 

It is possible to override the default variables with the optimizer parameters:

```swift
var optimizerParams: GrAI.Optimizer.Params()
optimizerParams.optimizer = ConstEpochsScheduler(.Adam)

// Initialize alpha value to be constant over time.
optimizerParams.variables["alpha"] = ConstEpochsVar(
    value: ConstVal(0.05)
)
        
model.setupOptimizers(params: optimizerParams)
```

It is also possible to define a scheduler for the values of these variables: 

```swift
// Let us assume nbLoops is defined: the number of setps per epoch.
let nLoops: Int 
let nbEpochs: 10

var optimizerParams: GrAI.Optimizer.Params()
optimizerParams.nbLoops = nbLoops
// Initialize optimizer to be constant over time.
optimizerParams.optimizer = ConstEpochsScheduler(.Adam)

// Initialize alpha value to follow a trigonometric ascending function.
optimizerParams.variables["alpha"] = MultEpochsVar(
    epoch0: nbEpochs,
    epochMul: 1,
    value: TrigonometricAscending(
        min: 0.0001,
        max: 0.001
    )
)
        
model.setupOptimizers(params: optimizerParams)
```

or 

```swift
// Let us assume nbLoops is defined: the number of setps per epoch.
let nLoops: Int 
let epoch1 = 5
let epoch2 = 10

var optimizerParams: GrAI.Optimizer.Params()
optimizerParams.nbLoops = nbLoops
// Initialize optimizer to be constant over time.
optimizerParams.optimizer = ConstEpochsScheduler(.Adam)

// Initialize alpha value to follow an exponential ascending function 
// for some steps then change for a linear descending function.
optimizerParams.variables["alpha"] = ListEpochsVar(
    epochs: [epoch1, epoch2],
    values: [
        ExponentialAscending(
            min: 0.0001,
            max: 0.001
        ),
        LinearDescending(
            min: 0.0001,
            max: 0.001
        )
    ]
)
        
model.setupOptimizers(params: optimizerParams)
```

If defined so, do not forget to inform the optimizer about the step 
evolution: 

```swift 
model.step()
```

## Next Chapter

Previous chapter: [Training Flow](TRAINING.md). \
Next chapter: [Gradient Checking](GRADIENT_CHECKING.md).
