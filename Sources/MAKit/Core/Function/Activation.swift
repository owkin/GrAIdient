//
// Activation.swift
// MAKit
//
// Created by Jean-François Reboud on 05/10/2022.
//

import Foundation

/// Activation function to be used in a Layer.
open class ActivationFunction: Codable
{
    let _name: String
    
    /// Get the name of the function.
    var name: String
    {
        get {
            return _name
        }
    }
    
    /// Forward GPU kernel.
    open var forwardKernel: String
    {
        get {
            fatalError("Not implemented.")
        }
    }
    /// Backward GPU kernel.
    open var backwardKernel: String
    {
        get {
            fatalError("Not implemented.")
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case name
    }
    
    ///
    /// Create an activation function.
    ///
    /// - Parameter name: Name of the function.
    ///
    public init(_ name: String)
    {
        _name = name
    }
    
    ///
    /// Decode from the disk.
    ///
    /// Throw an error if reading from the decoder fails, or
    /// if the data read is corrupted or otherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
    required public init(from decoder: Decoder) throws
    {
        let container = try decoder.container(keyedBy: Keys.self)
        _name = try container.decode(String.self, forKey: .name)
    }
    
    ///
    /// Encode to the disk.
    ///
    /// If the value fails to encode anything, `encoder` will encode an empty
    /// keyed container in its place.
    ///
    /// Throw an error if any values are invalid for the given
    /// encoder's format.
    /// 
    /// - Parameter encoder: The encoder to write data to.
    ///
    public func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        try container.encode(_name, forKey: .name)
    }
    
    ///
    /// Coefficient to apply during the weights initialization.
    ///
    /// - Parameters:
    ///     - nPrev: The number of input connections.
    ///     - nCur: The number of output connections.
    ///
    open func coeffInitWeights(nPrev: Int, nCur: Int) -> Double
    {
        return sqrt(2.0 / Double(nPrev + nCur))
    }
    
    ///
    /// Forward CPU.
    ///
    /// - Parameter x: The input.
    /// - Returns: The output.
    ///
    open func apply(_ x: Double) -> Double
    {
        fatalError("Not implemented.")
    }
    
    ///
    /// Backward CPU.
    ///
    /// - Parameter x: The input.
    /// - Returns: The output.
    ///
    open func derivate(_ x: Double) -> Double
    {
        fatalError("Not implemented.")
    }
    
    // TODO: add elements here.
}

/// ReLU activation function.
public class ReLU: ActivationFunction
{
    public static let str = "ReLU"
    
    /// Forward GPU kernel.
    public override var forwardKernel: String
    {
        get {
            return "forwardReLU"
        }
    }
    /// Backward GPU kernel.
    public override var backwardKernel: String
    {
        get {
            return "backwardReLU"
        }
    }
    
    /// Create a ReLU activation function.
    init()
    {
        super.init(ReLU.str)
    }
    
    ///
    /// Decode from the disk.
    ///
    /// Throw an error if reading from the decoder fails, or
    /// if the data read is corrupted or otherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
    required public init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
    }
    
    ///
    /// Coefficient to apply during the weights initialization.
    ///
    /// - Parameters:
    ///     - nPrev: The number of input connections.
    ///     - nCur: The number of output connections.
    ///
    public override func coeffInitWeights(nPrev: Int, nCur: Int) -> Double
    {
        return sqrt(2.0 / Double(nPrev))
    }
    
    ///
    /// Forward CPU.
    ///
    /// - Parameter x: The input.
    /// - Returns: The output.
    ///
    public override func apply(_ x: Double) -> Double
    {
        if x < 0
        {
            return 0.0
        }
        return x
    }
    
    ///
    /// Backward CPU.
    ///
    /// - Parameter x: The input.
    /// - Returns: The output.
    ///
    public override func derivate(_ x: Double) -> Double
    {
        if x < 0
        {
            return 0.0
        }
        return 1
    }
}

/// LeakyReLU activation function.
public class LeakyReLU: ActivationFunction
{
    let Ɛ = 0.01
    
    public static let str = "LeakyReLU"
    
    /// Forward GPU kernel.
    public override var forwardKernel: String
    {
        get {
            return "forwardLeakyReLU"
        }
    }
    /// Backward GPU kernel.
    public override var backwardKernel: String
    {
        get {
            return "backwardLeakyReLU"
        }
    }
    
    /// Create a LeakyReLU activation function.
    init()
    {
        super.init(LeakyReLU.str)
    }
    
    ///
    /// Decode from the disk.
    ///
    /// Throw an error if reading from the decoder fails, or
    /// if the data read is corrupted or otherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
    required public init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
    }
    
    ///
    /// Coefficient to apply during the weights initialization.
    ///
    /// - Parameters:
    ///     - nPrev: The number of input connections.
    ///     - nCur: The number of output connections.
    ///
    public override func coeffInitWeights(nPrev: Int, nCur: Int) -> Double
    {
        return sqrt(2.0 / Double(nPrev))
    }
    
    ///
    /// Forward CPU.
    ///
    /// - Parameter x: The input.
    /// - Returns: The output.
    ///
    public override func apply(_ x: Double) -> Double
    {
        if x < 0
        {
            return Ɛ * x
        }
        return x
    }
    
    ///
    /// Backward CPU.
    ///
    /// - Parameter x: The input.
    /// - Returns: The output.
    ///
    public override func derivate(_ x: Double) -> Double
    {
        if x < 0
        {
            return Ɛ
        }
        return 1
    }
}

/// SoftReLU activation function.
public class SoftReLU: ActivationFunction
{
    let Ɛ = 0.01
    
    public static let str = "SoftReLU"
    
    /// Forward GPU kernel.
    public override var forwardKernel: String
    {
        get {
            return "forwardSoftReLU"
        }
    }
    /// Backward GPU kernel.
    public override var backwardKernel: String
    {
        get {
            return "backwardSoftReLU"
        }
    }
    
    /// Create a SoftReLU activation function.
    init()
    {
        super.init(SoftReLU.str)
    }
    
    ///
    /// Decode from the disk.
    ///
    /// Throw an error if reading from the decoder fails, or
    /// if the data read is corrupted or otherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
    required public init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
    }
    
    ///
    /// Coefficient to apply during the weights initialization.
    ///
    /// - Parameters:
    ///     - nPrev: The number of input connections.
    ///     - nCur: The number of output connections.
    ///
    public override func coeffInitWeights(nPrev: Int, nCur: Int) -> Double
    {
        return sqrt(2.0 / Double(nPrev))
    }
    
    ///
    /// Forward CPU.
    ///
    /// - Parameter x: The input.
    /// - Returns: The output.
    ///
    public override func apply(_ x: Double) -> Double
    {
        return Ɛ * x + (1 - Ɛ) * log(1 + exp(x))
    }
    
    ///
    /// Backward CPU.
    ///
    /// - Parameter x: The input.
    /// - Returns: The output.
    ///
    public override func derivate(_ x: Double) -> Double
    {
        return Ɛ + (1 - Ɛ) / (1 + exp(-x))
    }
}

/// Factory API to build an activation function.
public protocol IActivationKernel
{
    ///
    /// Build an activation function.
    ///
    /// - Parameter name: Name of the function to build.
    /// - Returns: The activation function.
    ///
    func build(_ name: String) -> ActivationFunction?
}

/// Factory to build an activation function.
class ActivationKernel: IActivationKernel
{
    static let _kernels: [String:ActivationKernel] =
    [
        ReLU.str: ReLUKernel(),
        LeakyReLU.str: LeakyReLUKernel(),
        SoftReLU.str: SoftReLUKernel(),
    ]
    
    ///
    /// Build an activation function.
    ///
    /// - Parameter name: Name of the function to build.
    /// - Returns: The activation function.
    ///
    func build(_ name: String) -> ActivationFunction?
    {
        for (nameTmp, kernel) in ActivationKernel._kernels
        {
            if nameTmp == name
            {
                return kernel.build()
            }
        }
        return nil
    }
    
    /// Build an activation function.
    func build() -> ActivationFunction
    {
        fatalError("Not implemented.")
    }
}

/// Factory to build a ReLU function.
private class ReLUKernel: ActivationKernel
{
    /// Build a ReLU function.
    override func build() -> ActivationFunction
    {
        return ReLU()
    }
}

/// Factory to build a ReLU function.
private class LeakyReLUKernel: ActivationKernel
{
    /// Build a LeakyReLU function.
    override func build() -> ActivationFunction
    {
        return LeakyReLU()
    }
}

/// Factory to build a ReLU function.
private class SoftReLUKernel: ActivationKernel
{
    /// Build a SoftReLU function.
    override func build() -> ActivationFunction
    {
        return SoftReLU()
    }
}