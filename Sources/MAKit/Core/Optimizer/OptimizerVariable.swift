//
// OptimizerVariable.swift
// MAKit
//
// Created by Jean-François Reboud on 08/10/2022.
//

import Foundation

class OptimizerVariable
{
    let name: String
    var time: TimeVariable
    
    init(name: String, time: TimeVariable)
    {
        self.name = name
        self.time = time
    }
    
    func getValue(step: Int, epoch: Int, nbLoops: Int) -> Double?
    {
        return time.getValue(
            step: step,
            epoch: epoch,
            nbLoops: nbLoops
        )
    }
    
    func getPercent(step: Int, epoch: Int, nbLoops: Int) -> Double?
    {
        return time.getPercent(
            step: step,
            epoch: epoch,
            nbLoops: nbLoops
        )
    }
}

public protocol TimeVariable: HandleTime
{
    func getValue(step: Int, epoch: Int, nbLoops: Int) -> Double?
    func getPercent(step: Int, epoch: Int, nbLoops: Int) -> Double?
}

public class ConstEpochsVar: ConstEpochsTime, TimeVariable
{
    let _value: ValueVariable
    
    public init(value: ValueVariable)
    {
        _value = value
        super.init()
    }
    
    public func getValue(step: Int, epoch: Int, nbLoops: Int) -> Double?
    {
        let progress = getProgressRatio(
            step: step,
            epoch: epoch,
            nbLoops: nbLoops
        )
        return _value.getValue(progress: progress)
    }
    
    public func getPercent(step: Int, epoch: Int, nbLoops: Int) -> Double?
    {
        let progress = getProgressRatio(
            step: step,
            epoch: epoch,
            nbLoops: nbLoops
        )
        return _value.getPercent(progress: progress)
    }
}

public class MultEpochsVar: MultEpochsTime, TimeVariable
{
    let _value: ValueVariable
    
    public init(epoch0: Int, epochMul: Int, value: ValueVariable)
    {
        _value = value
        super.init(epoch0: epoch0, epochMul: epochMul)
    }
    
    public func getValue(step: Int, epoch: Int, nbLoops: Int) -> Double?
    {
        let progress = getProgressRatio(
            step: step,
            epoch: epoch,
            nbLoops: nbLoops
        )
        return _value.getValue(progress: progress)
    }
    
    public func getPercent(step: Int, epoch: Int, nbLoops: Int) -> Double?
    {
        let progress = getProgressRatio(
            step: step,
            epoch: epoch,
            nbLoops: nbLoops
        )
        return _value.getPercent(progress: progress)
    }
}

public class ListEpochsVar: ListEpochsTime, TimeVariable
{
    let _values: [ValueVariable]
    
    public init(epochs: [Int], values: [ValueVariable])
    {
        var listHandleValue = [ValueVariable]()
        for value in values
        {
            listHandleValue.append(value)
        }
        
        _values = listHandleValue
        super.init(epochs: epochs)
    }
    
    public func getValue(step: Int, epoch: Int, nbLoops: Int) -> Double?
    {
        let (_, _, epochIndex) = getEpochTarget(epoch: epoch)
        
        let progress = getProgressRatio(
            step: step,
            epoch: epoch,
            nbLoops: nbLoops
        )
        return _values[epochIndex].getValue(progress: progress)
    }
    
    public func getPercent(step: Int, epoch: Int, nbLoops: Int) -> Double?
    {
        let (_, _, epochIndex) = getEpochTarget(epoch: epoch)
        
        let progress = getProgressRatio(
            step: step,
            epoch: epoch,
            nbLoops: nbLoops
        )
        return _values[epochIndex].getPercent(progress: progress)
    }
}

public protocol ValueVariable
{
    func getValue(progress: Double) -> Double?
    func getPercent(progress: Double) -> Double?
}

public class NullVal: ValueVariable
{
    public init() {}
    
    public func getValue(progress: Double) -> Double?
    {
        return nil
    }
    
    public func getPercent(progress: Double) -> Double?
    {
        return nil
    }
}

public class ConstVal: ValueVariable
{
    let _value: Double
    
    public init(_ value: Double)
    {
        _value = value
    }
    
    public func getValue(progress: Double) -> Double?
    {
        return _value
    }
    
    public func getPercent(progress: Double) -> Double?
    {
        return 100
    }
}

public class MinMaxVal: ValueVariable
{
    let _min: Double
    let _max: Double
    
    public required init(min: Double, max: Double)
    {
        _min = min
        _max = max
    }
    
    public func getValue(progress: Double) -> Double?
    {
        fatalError("Not implemented.")
    }
    
    public func getPercent(progress: Double) -> Double?
    {
        fatalError("Not implemented.")
    }
}

public class DescendingVal: MinMaxVal
{
    public override func getPercent(progress: Double) -> Double?
    {
        let cur = getValue(progress: progress)!
        return (_max - cur) * 100.0 / (_max - _min)
    }
}

public class AscendingVal: MinMaxVal
{
    public override func getPercent(progress: Double) -> Double?
    {
        let cur = getValue(progress: progress)!
        return (cur - _min) * 100.0 / (_max - _min)
    }
}

public class LinearAscending: AscendingVal
{
    public override func getValue(progress: Double) -> Double?
    {
        return _min + progress * (_max - _min)
    }
}

public class LinearDescending: DescendingVal
{
    public override func getValue(progress: Double) -> Double?
    {
        return _max - progress * (_max - _min)
    }
}

public class TrigonometricAscending: AscendingVal
{
    public override func getValue(progress: Double) -> Double?
    {
        let factor = sin(Double.pi/2.0 * progress)
        return _min + (_max - _min) * factor
    }
}

public class TrigonometricDescending: DescendingVal
{
    public override func getValue(progress: Double) -> Double?
    {
        let factor = sin(Double.pi/2.0 * progress)
        return _max - (_max - _min) * factor
    }
}

public class ExponentialAscending: AscendingVal
{
    public override func getValue(progress: Double) -> Double?
    {
        return _min * pow(_max/_min, progress)
    }
}

public class ExponentialDescending: DescendingVal
{
    public override func getValue(progress: Double) -> Double?
    {
        return _max * pow(_min/_max, progress)
    }
}
