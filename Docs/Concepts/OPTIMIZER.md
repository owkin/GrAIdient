# Optimizer

The `Optimizer` is responsible for using the gradients of weights 
in order to update the weights values. 

The gradients of weights are computed during the backward pass.

## Optimizer Scheduler

In order to setup the optimizer: 

```swift 
var optimizerParams: MAKit.Optimizer.Params()
optimizerParams.optimizer = ConstEpochsScheduler(.Adam)

model.setupOptimizers(params: optimizerParams)
```

It is also possible to setup an optimizer that is scheduled upon step: 

```swift 
// Let us assume nbLoops is defined: the number of setps per epoch.
let nLoops: Int 
let epoch1 = 5
let epoch2 = 10

var optimizerParams: MAKit.Optimizer.Params()
optimizerParams.nbLoops = nbLoops
optimizerParams.optimizer = ListEpochsScheduler(
    epochs: [epoch1, epoch2],
    optimizers: [.Adam, .SGD]
)

model.setupOptimizers(params: optimizerParams)
```

If defined as so, do not forget to inform the optimizer about the step 
evolution: 

```swift 
model.step()
```

## Variable Scheduler

The different variables that a defined for the different optimizers are 
visible in the `OptimizerImpl` component.

For example at `OptimizerImpl.initVariables` the variables `alpha` and 
`lambda` are initialized with default values. 

It is possible to override the default variables with the optimizer parameters 
of the [previous paragraph](#optimizer-scheduler).

```swift
var optimizerParams: MAKit.Optimizer.Params()
optimizerParams.optimizer = ConstEpochsScheduler(.Adam)

optimizerParams.variables["alpha"] = ConstEpochsVar(
    value: ConstVal(0.05)
)
        
model.setupOptimizers(params: optimizerParams)
```

It is also possible to define scheduler for variables: 

```swift
// Let us assume nbLoops is defined: the number of setps per epoch.
let nLoops: Int 
let nbEpochs: 10

var optimizerParams: MAKit.Optimizer.Params()
optimizerParams.nbLoops = nbLoops
optimizerParams.optimizer = ConstEpochsScheduler(.Adam)

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

var optimizerParams: MAKit.Optimizer.Params()
optimizerParams.nbLoops = nbLoops
optimizerParams.optimizer = ConstEpochsScheduler(.Adam)

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

If defined as so, do not forget to inform the optimizer about the step 
evolution: 

```swift 
model.step()
```

## Next Chapter

Previous chapter: [Training Flow](TRAINING.md). \
Next chapter: [Gradient Checking](GRADIENT_CHECKING.md).
