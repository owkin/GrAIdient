//
// OptimizerKernel.swift
// MAKit
//
// Created by Jean-François Reboud on 08/10/2022.
//

public class OptimizerKernel
{
    var params: MAKit.Optimizer.Params! = nil
    
    var algo: OptimizerAlgorithm
    {
        get {
            return params.optimizer.buildOptimizer(kernel: self)
        }
    }
    
    init() {}
    
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
    
    func getValue(_ variable: OptimizerVariable) -> Double?
    {
        return variable.getValue(
            step: params.step,
            epoch: params.epoch,
            nbLoops: params.nbLoops
        )
    }
    func getPercent(_ variable: OptimizerVariable) -> Double?
    {
        return variable.getPercent(
            step: params.step,
            epoch: params.epoch,
            nbLoops: params.nbLoops
        )
    }
    
    func incT()
    {
        params.t += 1
    }
    func incStep()
    {
        params.step += 1
    }
    func incEpoch()
    {
        params.epoch += 1
    }
}

public protocol HandleTime
{
    func getEpochTarget(epoch: Int) -> (Int, Int)
    func getProgressRatio(step: Int, epoch: Int, nbLoops: Int) -> Double
}

extension HandleTime
{
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

public class ConstEpochsTime: HandleTime
{
    public func getEpochTarget(epoch: Int) -> (Int, Int)
    {
        return (0, 0)
    }
}

public class MultEpochsTime: HandleTime
{
    let _epoch0: Int
    let _epochMul: Int
    
    init(epoch0: Int, epochMul: Int)
    {
        _epoch0 = epoch0
        _epochMul = epochMul
    }
    
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

public class ListEpochsTime: HandleTime
{
    let _epochs: [Int]
    
    init(epochs: [Int])
    {
        _epochs = epochs
    }
    
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
    
    public func getEpochTarget(epoch: Int) -> (Int, Int)
    {
        let (epochTarget, epochBlock, _) = getEpochTarget(epoch: epoch)
        return (epochTarget, epochBlock)
    }
}

public protocol TimeScheduler: HandleTime
{
    func buildOptimizer(kernel: OptimizerKernel) -> OptimizerAlgorithm
}

public class ConstEpochsScheduler: ConstEpochsTime, TimeScheduler
{
    let _handleOptimizer: BuildOptimizer
    
    public init(_ optimizer: MAKit.Optimizer.Class)
    {
        _handleOptimizer = BuildOptimizer(optimizer)
        super.init()
    }
    
    public func buildOptimizer(kernel: OptimizerKernel) -> OptimizerAlgorithm
    {
        return _handleOptimizer.build(kernel: kernel)
    }
}

public class ListEpochsScheduler: ListEpochsTime, TimeScheduler
{
    let _listHandleOptimizer: [BuildOptimizer]
    
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
    
    public func buildOptimizer(kernel: OptimizerKernel) -> OptimizerAlgorithm
    {
        let (_, _, epochIndex) = getEpochTarget(epoch: kernel.params.epoch)
        return _listHandleOptimizer[epochIndex].build(kernel: kernel)
    }
}

class BuildOptimizer
{
    let _optimizerClass: MAKit.Optimizer.Class
    
    init(_ optimizerClass: MAKit.Optimizer.Class)
    {
        _optimizerClass = optimizerClass
    }
    
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
