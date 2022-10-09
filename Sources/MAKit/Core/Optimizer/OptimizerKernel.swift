//
// OptimizerKernel.swift
// MAKit
//
// Created by Jean-François Reboud on 08/10/2022.
//

/// The state of the optimization process.
public class OptimizerKernel
{
    /// Parameters of the optimizer algorithm.
    var params: MAKit.Optimizer.Params! = nil
    
    /// Create the current algorithm.
    var algo: OptimizerAlgorithm
    {
        get {
            return params.optimizer.buildOptimizer(kernel: self)
        }
    }
    
    ///
    /// Update the values of the internal variables.
    ///
    /// - Parameter variables: The internal variables to update.
    ///
    func setup(_ variables: [OptimizerVariable])
    {
        for variable in variables
        {
            if let timeVariable = params.variables[variable.name]
            {
                variable.time = timeVariable
            }
        }
    }
    
    ///
    /// Get the value of an internal variable.
    ///
    /// - Parameter variable: The variable to get the value from.
    /// - Returns: The current value of the variable.
    ///
    func getValue(_ variable: OptimizerVariable) -> Double?
    {
        return variable.getValue(
            step: params.step,
            epoch: params.epoch,
            nbLoops: params.nbLoops
        )
    }
    ///
    /// Get the percent value of an internal variable.
    ///
    /// - Parameter variable: The variable to get the percent value from.
    /// - Returns: The current percent value of the variable.
    ///
    func getPercent(_ variable: OptimizerVariable) -> Double?
    {
        return variable.getPercent(
            step: params.step,
            epoch: params.epoch,
            nbLoops: params.nbLoops
        )
    }
    
    /// Increment internal time state.
    func incT()
    {
        params.t += 1
    }
    /// Increment internal step state.
    func incStep()
    {
        params.step += 1
    }
    /// Increment internal epoch state.
    func incEpoch()
    {
        params.epoch += 1
    }
}

/// The API for something that may be variable during the training loop.
public protocol HandleTime
{
    ///
    /// Get the next epoch target.
    ///
    /// - Parameter epoch: The current epoch.
    /// - Returns: (The final epoch of the current stage, The first epoch of the current stage).
    ///
    func getEpochTarget(epoch: Int) -> (Int, Int)
    
    ///
    /// Get the current ratio of progress from the first epoch of the current stage
    /// to the final epoch of the current stage.
    ///
    /// - Parameters:
    ///     - step: The current step.
    ///     - epoch: The current epoch.
    ///     - nbLoops: The number of loops per epoch.
    /// - Returns: The progress ratio.
    ///
    func getProgressRatio(step: Int, epoch: Int, nbLoops: Int) -> Double
}

extension HandleTime
{
    ///
    /// Get the current ratio of progress from the first epoch of the current stage
    /// to the final epoch of the current stage.
    ///
    /// - Parameters:
    ///     - step: The current step.
    ///     - epoch: The current epoch.
    ///     - nbLoops: The number of loops per epoch.
    /// - Returns: The progress ratio.
    ///
    public func getProgressRatio(step: Int, epoch: Int, nbLoops: Int) -> Double
    {
        let t = step
        var td = Double(t % nbLoops)
        td /= Double(nbLoops)
        
        let (epochTarget, epochBlock) = getEpochTarget(epoch: epoch)
        let T = epoch - epochBlock
        
        return (Double(T) + td) / Double(epochTarget)
    }
}

/// An epoch thread with no variation at all.
public class ConstEpochsTime: HandleTime
{
    ///
    /// Get the next epoch target.
    ///
    /// - Parameter epoch: The current epoch.
    /// - Returns: (The final epoch of the current stage, The first epoch of the current stage).
    ///
    public func getEpochTarget(epoch: Int) -> (Int, Int)
    {
        return (0, 0)
    }
}

/// An epoch thread where next epoch target is a multiple of previous epoch target.
public class MultEpochsTime: HandleTime
{
    /// The very first epoch target.
    let _epoch0: Int
    /// The multipliciative factor.
    let _epochMul: Int
    
    ///
    /// Create a multiplicative epoch target thread.
    ///
    /// - Parameters:
    ///     - epoch0: The very first epoch target.
    ///     - epochMul: The multiplicative factor.
    ///
    init(epoch0: Int, epochMul: Int)
    {
        _epoch0 = epoch0
        _epochMul = epochMul
    }
    
    ///
    /// Get the next epoch target.
    ///
    /// - Parameter epoch: The current epoch.
    /// - Returns: (The final epoch of the current stage, The first epoch of the current stage).
    ///
    public func getEpochTarget(epoch: Int) -> (Int, Int)
    {
        var epochBlock = 0
        var epochTarget = _epoch0
        while epochBlock + epochTarget <= epoch
        {
            epochBlock += epochTarget
            epochTarget *= _epochMul
        }
        return (epochTarget, epochBlock)
    }
}

/// An epoch thread where the list of target epochs are specified.
public class ListEpochsTime: HandleTime
{
    /// The list of epoch targets.
    let _epochs: [Int]
    
    ///
    /// Create an epoch thread with a list of target epochs.
    ///
    /// - Parameter epochs: The list of target epochs.
    ///
    init(epochs: [Int])
    {
        _epochs = epochs
    }
    
    ///
    /// Get the next epoch target.
    ///
    /// - Parameter epoch: The current epoch.
    /// - Returns: (The final epoch of the current stage,
    ///            The first epoch of the current stage,
    ///            The index of the first epoch of the current stage in `_epochs`).
    ///
    func getEpochTarget(epoch: Int) -> (Int, Int, Int)
    {
        var epochBlock = 0
        var curEpoch = 0
        
        var epochTarget = _epochs[curEpoch]
        curEpoch += 1
        
        while epochBlock + epochTarget <= epoch
        {
            if _epochs.count <= curEpoch
            {
                fatalError("'epoch': \(epoch) out of bound.")
            }
            
            epochBlock += epochTarget
            epochTarget = _epochs[curEpoch] - _epochs[curEpoch-1]
            curEpoch += 1
        }
        
        return (epochTarget, epochBlock, curEpoch-1)
    }
    
    ///
    /// Get the next epoch target.
    ///
    /// - Parameter epoch: The current epoch.
    /// - Returns: (The final epoch of the current stage, The first epoch of the current stage).
    ///
    public func getEpochTarget(epoch: Int) -> (Int, Int)
    {
        let (epochTarget, epochBlock, _) = getEpochTarget(epoch: epoch)
        return (epochTarget, epochBlock)
    }
}

/// The API for an optimizer algorithm that may be variable during the training loop.
public protocol TimeScheduler: HandleTime
{
    ///
    /// Build an optimizer algorithm.
    ///
    /// - Parameter kernel: The state of the optimization process.
    /// - Returns: The optimizer algorithm.
    ///
    func buildOptimizer(kernel: OptimizerKernel) -> OptimizerAlgorithm
}

/// A scheduler that will always build the same optimizer algorithm.
public class ConstEpochsScheduler: ConstEpochsTime, TimeScheduler
{
    /// The factory used to build an optimizer algorithm.
    let _handleOptimizer: BuildOptimizer
    
    ///
    /// Create a scheduler that will always build the same optimizer algorithm.
    ///
    /// - Parameter optimizer: The optimizer algorithm to build.
    ///
    public init(_ optimizer: MAKit.Optimizer.Class)
    {
        _handleOptimizer = BuildOptimizer(optimizer)
        super.init()
    }
    
    ///
    /// Build an optimizer algorithm.
    ///
    /// - Parameter kernel: The state of the optimization process.
    /// - Returns: The optimizer algorithm.
    ///
    public func buildOptimizer(kernel: OptimizerKernel) -> OptimizerAlgorithm
    {
        return _handleOptimizer.build(kernel: kernel)
    }
}

/// A scheduler that can build different optimizer algorithms at different epochs targets.
public class ListEpochsScheduler: ListEpochsTime, TimeScheduler
{
    /// The list of factories to build the optimizer algorithms.
    let _listHandleOptimizer: [BuildOptimizer]
    
    ///
    /// Create a scheduler that can build different optimizer algorithms through time.
    ///
    /// - Parameters:
    ///     - epochs: The list of epochs target.
    ///     - optimizers: The corresponding list of optimizer algorithms to build.
    ///
    public init(epochs: [Int], optimizers: [MAKit.Optimizer.Class])
    {
        var listHandleOptimizer = [BuildOptimizer]()
        for optimizer in optimizers
        {
            listHandleOptimizer.append(BuildOptimizer(optimizer))
        }
        _listHandleOptimizer = listHandleOptimizer
        super.init(epochs: epochs)
    }
    
    ///
    /// Build an optimizer algorithm.
    ///
    /// - Parameter kernel: The state of the optimization process.
    /// - Returns: The optimizer algorithm.
    ///
    public func buildOptimizer(kernel: OptimizerKernel) -> OptimizerAlgorithm
    {
        let (_, _, epochIndex) = getEpochTarget(epoch: kernel.params.epoch)
        return _listHandleOptimizer[epochIndex].build(kernel: kernel)
    }
}

/// A factory to build an optimizer algorithm.
class BuildOptimizer
{
    /// The optimizer algorithm to build.
    let _optimizerClass: MAKit.Optimizer.Class
    
    ///
    /// Create a factory that can build an optimizer algorithm.
    ///
    /// - Parameter optimizerClass: The optimizer algorithm to build.
    ///
    init(_ optimizerClass: MAKit.Optimizer.Class)
    {
        _optimizerClass = optimizerClass
    }
    
    ///
    /// Build an optimizer algorithm.
    ///
    /// - Parameter kernel: The state of the optimization process.
    /// - Returns: The optimizer algorithm.
    ///
    func build(kernel: OptimizerKernel) -> OptimizerAlgorithm
    {
        let optimizer: OptimizerImpl
        switch _optimizerClass
        {
        case .SGDMomentum:
            optimizer = SGDMomentumOptimizer(kernel)
        case .Adam:
            optimizer = AdamOptimizer(kernel)
        case .AMSGrad:
            optimizer = AMSGradOptimizer(kernel)
        case .AdamRectified:
            optimizer = AdamRectifiedOptimizer(kernel)
        case .AdaBound:
            optimizer = AdaBoundOptimizer(kernel)
        case .AMSBound:
            optimizer = AMSBoundOptimizer(kernel)
        default:
            optimizer = SGDOptimizer(kernel)
        }
        optimizer.initVariables()
        
        let algorithm = OptimizerAlgorithm(optimizer)
        return algorithm
    }
}
