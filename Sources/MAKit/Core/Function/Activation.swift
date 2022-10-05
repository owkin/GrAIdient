//
// Activation.swift
// MAKit
//
// Created by Jean-François Reboud on 05/10/2022.
//

import Foundation

open class ActivationFunction: Codable
{
    let _name: String
    
    var name: String
    {
        get {
            return _name
        }
    }
    
    open var forwardKernel: String
    {
        get {
            fatalError("Not implemented.")
        }
    }
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
    
    public init(_ name: String)
    {
        _name = name
    }
    
    required public init(from decoder: Decoder) throws
    {
        let container = try decoder.container(keyedBy: Keys.self)
        _name = try container.decode(String.self, forKey: .name)
    }
    
    public func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        try container.encode(_name, forKey: .name)
    }
    
    open func coeffInitWeights(nPrev: Int, nCur: Int) -> Double
    {
        return sqrt(2.0 / Double(nPrev + nCur))
    }
    
    open func apply(_ x: Double) -> Double
    {
        fatalError("Not implemented.")
    }
    
    open func derivate(_ x: Double) -> Double
    {
        fatalError("Not implemented.")
    }
    
    // TODO: add elements here.
}

public class ReLU: ActivationFunction
{
    public static let str = "ReLU"
    
    public override var forwardKernel: String
    {
        get {
            return "forwardReLU"
        }
    }
    public override var backwardKernel: String
    {
        get {
            return "backwardReLU"
        }
    }
    
    init()
    {
        super.init(ReLU.str)
    }
    
    required public init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
    }
    
    public override func coeffInitWeights(nPrev: Int, nCur: Int) -> Double
    {
        return sqrt(2.0 / Double(nPrev))
    }
    
    public override func apply(_ x: Double) -> Double
    {
        if x < 0
        {
            return 0.0
        }
        return x
    }
    
    public override func derivate(_ x: Double) -> Double
    {
        if x < 0
        {
            return 0.0
        }
        return 1
    }
}

public class LeakyReLU: ActivationFunction
{
    let Ɛ = 0.01
    
    public static let str = "LeakyReLU"
    
    public override var forwardKernel: String
    {
        get {
            return "forwardLeakyReLU"
        }
    }
    public override var backwardKernel: String
    {
        get {
            return "backwardLeakyReLU"
        }
    }
    
    init()
    {
        super.init(LeakyReLU.str)
    }
    
    required public init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
    }
    
    public override func coeffInitWeights(nPrev: Int, nCur: Int) -> Double
    {
        return sqrt(2.0 / Double(nPrev))
    }
    
    public override func apply(_ x: Double) -> Double
    {
        if x < 0
        {
            return Ɛ * x
        }
        return x
    }
    
    public override func derivate(_ x: Double) -> Double
    {
        if x < 0
        {
            return Ɛ
        }
        return 1
    }
}

public class SoftReLU: ActivationFunction
{
    let Ɛ = 0.01
    
    public static let str = "SoftReLU"
    
    public override var forwardKernel: String
    {
        get {
            return "forwardSoftReLU"
        }
    }
    public override var backwardKernel: String
    {
        get {
            return "backwardSoftReLU"
        }
    }
    
    init()
    {
        super.init(SoftReLU.str)
    }
    
    required public init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
    }
    
    public override func coeffInitWeights(nPrev: Int, nCur: Int) -> Double
    {
        return sqrt(2.0 / Double(nPrev))
    }
    
    public override func apply(_ x: Double) -> Double
    {
        return Ɛ * x + (1 - Ɛ) * log(1 + exp(x))
    }
    
    public override func derivate(_ x: Double) -> Double
    {
        return Ɛ + (1 - Ɛ) / (1 + exp(-x))
    }
}

public protocol IActivationKernel
{
    func build(_ name: String) -> ActivationFunction?
}

class ActivationKernel: IActivationKernel
{
    static let _kernels: [String:ActivationKernel] =
    [
        ReLU.str: ReLUKernel(),
        LeakyReLU.str: LeakyReLUKernel(),
        SoftReLU.str: SoftReLUKernel(),
    ]
    
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
    
    func build() -> ActivationFunction
    {
        fatalError("Not implemented.")
    }
}

private class ReLUKernel: ActivationKernel
{
    override func build() -> ActivationFunction
    {
        return ReLU()
    }
}

private class LeakyReLUKernel: ActivationKernel
{
    override func build() -> ActivationFunction
    {
        return LeakyReLU()
    }
}

private class SoftReLUKernel: ActivationKernel
{
    override func build() -> ActivationFunction
    {
        return SoftReLU()
    }
}
