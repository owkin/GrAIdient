//
// OptimizerVariable.swift
// MAKit
//
// Created by Jean-François Reboud on 08/10/2022.
//

import Foundation

/// A "variable" that may be variable during the training loop.
class OptimizerVariable
{
    /// Name of the variable.
    let name: String
    
    /// The time dependency of the variable.
    var time: TimeVariable
    
    ///
    /// Create a variable that may vary through time.
    ///
    /// - Parameters:
    ///     - name: The name of the variable.
    ///     - time: The time dependency of the variable.
    ///
    init(name: String, time: TimeVariable)
    {
        self.name = name
        self.time = time
    }
    
    ///
    /// Get the variable value.
    ///
    /// - Parameters:
    ///     - step: The current step.
    ///     - nbLoops: The number of steps per epoch.
    /// - Returns: The variable value.
    ///
    func getValue(step: Int, nbLoops: Int) -> Double?
    {
        return time.getValue(
            step: step,
            nbLoops: nbLoops
        )
    }
    
    ///
    /// Get the variable percent value.
    ///
    /// - Parameters:
    ///     - step: The current step.
    ///     - nbLoops: The number of steps per epoch.
    /// - Returns: The variable percent value.
    ///
    func getPercent(step: Int, nbLoops: Int) -> Double?
    {
        return time.getPercent(
            step: step,
            nbLoops: nbLoops
        )
    }
}

/// The API for a "variable" that may be variable during the training loop.
public protocol TimeVariable: HandleTime
{
    ///
    /// Get the variable value.
    ///
    /// - Parameters:
    ///     - step: The current step.
    ///     - nbLoops: The number of steps per epoch.
    /// - Returns: The variable value.
    ///
    func getValue(step: Int, nbLoops: Int) -> Double?
    
    ///
    /// Get the variable percent value.
    ///
    /// - Parameters:
    ///     - step: The current step.
    ///     - nbLoops: The number of steps per epoch.
    /// - Returns: The variable percent value.
    ///
    func getPercent(step: Int, nbLoops: Int) -> Double?
}

/// A variable that will always have the same value.
public class ConstEpochsVar: ConstEpochsTime, TimeVariable
{
    /// The variable value.
    let _value: ValueVariable
    
    ///
    /// Create a variable that will always have the same value.
    ///
    /// - Parameter value: The variable value.
    ///
    public init(value: ValueVariable)
    {
        _value = value
        super.init()
    }
    
    ///
    /// Get the variable value.
    ///
    /// - Parameters:
    ///     - step: The current step.
    ///     - nbLoops: The number of steps per epoch.
    /// - Returns: The variable value.
    ///
    public func getValue(step: Int, nbLoops: Int) -> Double?
    {
        let progress = getProgressRatio(
            step: step,
            nbLoops: nbLoops
        )
        return _value.getValue(progress: progress)
    }
    
    ///
    /// Get the variable percent value.
    ///
    /// - Parameters:
    ///     - step: The current step.
    ///     - nbLoops: The number of steps per epoch.
    /// - Returns: The variable percent value.
    ///
    public func getPercent(step: Int, nbLoops: Int) -> Double?
    {
        let progress = getProgressRatio(
            step: step,
            nbLoops: nbLoops
        )
        return _value.getPercent(progress: progress)
    }
}

/// A variable that will change according to a multiplicative epoch thread.
public class MultEpochsVar: MultEpochsTime, TimeVariable
{
    /// The variable value.
    let _value: ValueVariable
    
    ///
    /// Create a variable that follows a `MultEpochsTime` epoch thread.
    ///
    /// - Parameters:
    ///     - epoch0: The very first epoch target.
    ///     - epochMul: The multiplicative factor.
    ///     - value: The variable value.
    ///
    public init(epoch0: Int, epochMul: Int, value: ValueVariable)
    {
        _value = value
        super.init(epoch0: epoch0, epochMul: epochMul)
    }
    
    ///
    /// Get the variable value.
    ///
    /// - Parameters:
    ///     - step: The current step.
    ///     - nbLoops: The number of steps per epoch.
    /// - Returns: The variable value.
    ///
    public func getValue(step: Int, nbLoops: Int) -> Double?
    {
        let progress = getProgressRatio(
            step: step,
            nbLoops: nbLoops
        )
        return _value.getValue(progress: progress)
    }
    
    ///
    /// Get the variable percent value.
    ///
    /// - Parameters:
    ///     - step: The current step.
    ///     - nbLoops: The number of steps per epoch.
    /// - Returns: The variable percent value.
    ///
    public func getPercent(step: Int, nbLoops: Int) -> Double?
    {
        let progress = getProgressRatio(
            step: step,
            nbLoops: nbLoops
        )
        return _value.getPercent(progress: progress)
    }
}

/// A variable with a list of values through time.
public class ListEpochsVar: ListEpochsTime, TimeVariable
{
    /// The list of variable values.
    let _values: [ValueVariable]
    
    ///
    /// Create a variable that follows a `ListEpochsTime` epoch thread.
    ///
    ///
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
    
    ///
    /// Get the variable value.
    ///
    /// - Parameters:
    ///     - step: The current step.
    ///     - nbLoops: The number of steps per epoch.
    /// - Returns: The variable value.
    ///
    public func getValue(step: Int, nbLoops: Int) -> Double?
    {
        let epoch = step / nbLoops
        let (_, _, epochIndex) = getEpochTarget(epoch: epoch)
        
        let progress = getProgressRatio(
            step: step,
            nbLoops: nbLoops
        )
        return _values[epochIndex].getValue(progress: progress)
    }
    
    ///
    /// Get the variable percent value.
    ///
    /// - Parameters:
    ///     - step: The current step.
    ///     - nbLoops: The number of steps per epoch.
    /// - Returns: The variable percent value.
    ///
    public func getPercent(step: Int, nbLoops: Int) -> Double?
    {
        let epoch = step / nbLoops
        let (_, _, epochIndex) = getEpochTarget(epoch: epoch)
        
        let progress = getProgressRatio(
            step: step,
            nbLoops: nbLoops
        )
        return _values[epochIndex].getPercent(progress: progress)
    }
}

/// The API to get a variable value depending on time progress.
public protocol ValueVariable
{
    ///
    /// Get the variable value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable value.
    ///
    func getValue(progress: Double) -> Double?
    ///
    /// Get the variable percent value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable percent value.
    ///
    func getPercent(progress: Double) -> Double?
}

/// A variable with no value at all.
public class NullVal: ValueVariable
{
    public init() {}
    
    ///
    /// Get the variable value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable value.
    ///
    public func getValue(progress: Double) -> Double?
    {
        return nil
    }
    
    ///
    /// Get the variable percent value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable percent value.
    ///
    public func getPercent(progress: Double) -> Double?
    {
        return nil
    }
}

/// A variable value that stays the same.
public class ConstVal: ValueVariable
{
    /// The variable value.
    let _value: Double
    
    ///
    /// Create a variable value.
    ///
    /// - Parameter value: The variable value.
    ///
    public init(_ value: Double)
    {
        _value = value
    }
    
    ///
    /// Get the variable value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable value.
    ///
    public func getValue(progress: Double) -> Double?
    {
        return _value
    }
    
    ///
    /// Get the variable percent value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable percent value.
    ///
    public func getPercent(progress: Double) -> Double?
    {
        return 100
    }
}

/// A variable value that varies in a range.
public class MinMaxVal: ValueVariable
{
    /// The minimal variable value.
    let _min: Double
    /// The maximal variable value.
    let _max: Double
    
    ///
    /// Create a variable value in a range.
    ///
    /// - Parameters:
    ///     - min: The minimal value of the range.
    ///     - max: The maximal value of the range.
    ///
    public required init(min: Double, max: Double)
    {
        _min = min
        _max = max
    }
    
    ///
    /// Get the variable value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable value.
    ///
    public func getValue(progress: Double) -> Double?
    {
        fatalError("Not implemented.")
    }
    
    ///
    /// Get the variable percent value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable percent value.
    ///
    public func getPercent(progress: Double) -> Double?
    {
        fatalError("Not implemented.")
    }
}

/// A variable value that varies from the max to the min of a range.
public class DescendingVal: MinMaxVal
{
    ///
    /// Get the variable percent value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable percent value.
    ///
    public override func getPercent(progress: Double) -> Double?
    {
        let cur = getValue(progress: progress)!
        return (_max - cur) * 100.0 / (_max - _min)
    }
}

/// A variable value that varies from the min to the max of a range.
public class AscendingVal: MinMaxVal
{
    ///
    /// Get the variable percent value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable percent value.
    ///
    public override func getPercent(progress: Double) -> Double?
    {
        let cur = getValue(progress: progress)!
        return (cur - _min) * 100.0 / (_max - _min)
    }
}

/// A variable value that grows continuously.
public class LinearAscending: AscendingVal
{
    ///
    /// Get the variable value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable value.
    ///
    public override func getValue(progress: Double) -> Double?
    {
        return _min + progress * (_max - _min)
    }
}

/// A variable value that shrinks continuously.
public class LinearDescending: DescendingVal
{
    ///
    /// Get the variable value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable value.
    ///
    public override func getValue(progress: Double) -> Double?
    {
        return _max - progress * (_max - _min)
    }
}

/// A variable value that grows as a sinus.
public class TrigonometricAscending: AscendingVal
{
    ///
    /// Get the variable value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable value.
    ///
    public override func getValue(progress: Double) -> Double?
    {
        let factor = sin(Double.pi/2.0 * progress)
        return _min + (_max - _min) * factor
    }
}

/// A variable value that shrinks as a sinus.
public class TrigonometricDescending: DescendingVal
{
    ///
    /// Get the variable value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable value.
    ///
    public override func getValue(progress: Double) -> Double?
    {
        let factor = sin(Double.pi/2.0 * progress)
        return _max - (_max - _min) * factor
    }
}

/// A variable value that grows as an exponential.
public class ExponentialAscending: AscendingVal
{
    ///
    /// Get the variable value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable value.
    ///
    public override func getValue(progress: Double) -> Double?
    {
        return _min * pow(_max/_min, progress)
    }
}

/// A variable value that shrinks as an exponential.
public class ExponentialDescending: DescendingVal
{
    ///
    /// Get the variable value.
    ///
    /// - Parameter progress: The time progress (from 0 to 1).
    /// - Returns: The variable value.
    ///
    public override func getValue(progress: Double) -> Double?
    {
        return _max * pow(_min/_max, progress)
    }
}
