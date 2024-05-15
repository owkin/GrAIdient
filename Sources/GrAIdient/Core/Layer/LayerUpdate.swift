//
// LayerUpdate.swift
// GrAIdient
//
// Created by Jean-François Reboud on 05/10/2022.
//

import Foundation
import Accelerate

/// Error occuring in an output layer.
public enum LossError: Error
{
    /// Ground truth has an unexected value.
    case GroundTruthValue
}

extension LossError: CustomStringConvertible
{
    public var description: String
    {
        switch self
        {
        case .GroundTruthValue:
            return "Ground truth has an unexpected value."
        }
    }
}

/// Running phase of a model.
public enum Phase
{
    case Training, Inference
}

/// API for a layer that have learning weights.
public protocol LayerUpdate: Layer
{
    /// Weights in the CPU execution context.
    var weightsCPU: [Float] { get set }
    /// Weights in the GPU execution context.
    var weightsGPU: [Float] { get set }
    
    /// Whether to compute weights' gradients or not.
    var computeDeltaWeights: Bool { get set }
    /// Whether gradients of weights must be accumulated or not.
    var accumulateDeltaWeights: Bool { get set }
    
    ///
    /// Initialize weights in the CPU execution context.
    ///
    /// Note that we do not initialize the delta which are dependent on the batch size.
    ///
    func initWeightsCPU()
    ///
    /// Initialize weights in the GPU execution context.
    ///
    /// Note that we do not initialize the delta which are dependent on the batch size.
    ///
    func initWeightsGPU()
    
    /// Get the weights in the CPU execution context.
    func collectWeightsCPU() -> [IWeightArrays]
    /// Get the weights in the GPU execution context.
    func collectWeightsGPU() -> [IWeightBuffers]
}

/// API for the GPU buffers needed to update the weights.
public protocol IWeightBuffers
{
    /// GPU device where the buffers are sent.
    var deviceID: Int { get }
    /// Number of elements in the different buffers.
    var nbElems: Int { get }
    
    /// Weights buffer: the buffer to be update.
    var w: FloatBuffer { get }
    /// Gradients buffer.
    var g: FloatBuffer { get }
    /// Momentum buffer.
    var m: FloatBuffer { get }
    /// Velocity buffer.
    var v: FloatBuffer { get }
    /// Velocity normalized buffer.
    var vHat: FloatBuffer { get }
    
    /// Clean the momentum..., preserving the weights.
    func reset()
}

extension IWeightBuffers
{
    /// GPU device where the buffers are sent.
    public var deviceID: Int
    {
        get {
            return w.deviceID
        }
    }
    /// Number of elements in the different buffers.
    public var nbElems: Int
    {
        get {
            return w.nbElems
        }
    }
}

/// GPU buffers needed to update the weights.
public class WeightBuffers: IWeightBuffers
{
    /// Weights buffer: the buffer to be update.
    public let w: FloatBuffer
    /// Gradients buffer.
    public let g: FloatBuffer
    /// Momentum buffer.
    public let m: FloatBuffer
    /// Velocity buffer.
    public let v: FloatBuffer
    /// Velocity normalized buffer.
    public let vHat: FloatBuffer
    
    ///
    /// Create a container of buffers.
    ///
    /// - Parameters:
    ///     - nbElems: Number of elements in the different buffers.
    ///     - deviceID: GPU device where the buffers are sent.
    ///
    init(nbElems: Int, deviceID: Int)
    {
        w = FloatBuffer(nbElems: nbElems, deviceID: deviceID)
        g = FloatBuffer(nbElems: nbElems, deviceID: deviceID)
        m = FloatBuffer(nbElems: nbElems, deviceID: deviceID)
        v = FloatBuffer(
            nbElems: nbElems, deviceID: deviceID, forceFloat: true
        )
        vHat = FloatBuffer(
            nbElems: nbElems, deviceID: deviceID, forceFloat: true
        )
    }
    
    /// Clean the buffers.
    public func reset()
    {
        // do not touch w
        g.reset()
        m.reset()
        v.reset()
        vHat.reset()
    }
}

/// Method used to initialize weights values (not considering the biases).
public enum WeightInitClass
{
    case XavierUniform, XavierNormal, KaimingUniform, KaimingNormal
}

public protocol LayerWeightInit: LayerUpdate
{
    /// Number of weights values (not considering the biases).
    var weightListSize: Int { get }
    /// Method used to initialize weights values.
    var weightInitClass: WeightInitClass { get set }
    
    /// Get the number of input and output connections.
    var connectivityIO: (Int, Int) { get }
    /// Get coefficient (depending on activation function) to apply during the weights initialization.
    var coeffInitWeights: Float { get }
}

extension LayerWeightInit
{
    /// Number of weights values.
    public var weightListSize: Int
    {
        get {
            let io = connectivityIO
            return io.0 * io.1
        }
    }
    
    /// Get coefficient (depending on activation function) to apply during the weights initialization.
    public var coeffInitWeights: Float
    {
        get {
            return 1.0
        }
    }
    
    /// 
    /// Generate list of weights values.
    ///
    /// - Returns: The generated list of values.
    ///
    public func generateWeightsList() -> [Float]
    {
        let nbElems = weightListSize
        let weightsList: [Float]
        switch weightInitClass {
        case .XavierUniform:
            weightsList = Self.XavierUniform(
                nbElems: nbElems,
                connectivityIO: connectivityIO
            )
        case .XavierNormal:
            weightsList = Self.XavierNormal(
                nbElems: nbElems,
                connectivityIO: connectivityIO
            )
        case .KaimingUniform:
            weightsList = Self.KaimingUniform(
                nbElems: nbElems,
                coeff: coeffInitWeights,
                connectivityIO: connectivityIO
            )
        case .KaimingNormal:
            weightsList = Self.KaimingNormal(
                nbElems: nbElems,
                coeff: coeffInitWeights,
                connectivityIO: connectivityIO
            )
        }
        return weightsList
    }
    
    ///
    /// Generate weights values.
    ///
    /// - Parameters:
    ///     - out: The output buffer.
    ///     - deviceID: GPU device.
    ///
    public func generateWeightsList(
        out: FloatBuffer,
        deviceID: Int)
    {
        let nbElems = weightListSize
        switch weightInitClass {
        case .XavierUniform:
            Self.XavierUniform(
                nbElems: nbElems,
                connectivityIO: connectivityIO,
                out: out,
                deviceID: deviceID
            )
        case .XavierNormal:
            Self.XavierNormal(
                nbElems: nbElems,
                connectivityIO: connectivityIO,
                out: out,
                deviceID: deviceID
            )
        case .KaimingUniform:
            Self.KaimingUniform(
                nbElems: nbElems,
                coeff: coeffInitWeights,
                connectivityIO: connectivityIO,
                out: out,
                deviceID: deviceID
            )
        case .KaimingNormal:
            Self.KaimingNormal(
                nbElems: nbElems,
                coeff: coeffInitWeights,
                connectivityIO: connectivityIO,
                out: out,
                deviceID: deviceID
            )
        }
    }
    
    ///
    /// Xavier uniform initialization method.
    ///
    /// - Parameters:
    ///     - nbElems: Number of weights to initialize.
    ///     - connectivityIO: Number of input and output connections.
    /// - Returns: Weights values.
    ///
    static func XavierUniform(
        nbElems: Int,
        connectivityIO: (Int, Int)) -> [Float]
    {
        var values = [Float]()
        let bound = sqrt(6) / sqrt(Float(connectivityIO.0 + connectivityIO.1))
        for _ in 0..<nbElems
        {
            values.append(Float.random(in: -bound..<bound))
        }
        return values
    }
    
    ///
    /// Xavier uniform initialization method.
    ///
    /// - Parameters:
    ///     - nbElems: Number of weights to initialize.
    ///     - connectivityIO: Number of input and output connections.
    ///     - out: The output buffer.
    ///     - deviceID: GPU device.
    ///
    static func XavierUniform(
        nbElems: Int,
        connectivityIO: (Int, Int),
        out: FloatBuffer,
        deviceID: Int)
    {
        var array = [Float](repeating: 0.0, count: nbElems)
        array.withUnsafeMutableBufferPointer
        {
            ptr in
            
            let bound = 
                sqrt(6) / sqrt(Float(connectivityIO.0 + connectivityIO.1))
            guard var arrayDescriptor = BNNSNDArrayDescriptor(
                data: ptr,
                shape: .vector(nbElems)),
            let randomNumberGenerator = BNNSCreateRandomGenerator(
                BNNSRandomGeneratorMethodAES_CTR,
                nil) else
            {
                fatalError()
            }
            
            BNNSRandomFillUniformFloat(
                randomNumberGenerator,
                &arrayDescriptor,
                -bound,
                bound
            )
            
            BNNSDestroyRandomGenerator(randomNumberGenerator)
        }
        out.initialize(array: &array)
    }
    
    ///
    /// Xavier normal initialization method.
    ///
    /// - Parameters:
    ///     - nbElems: Number of weights to initialize.
    ///     - connectivityIO: Number of input and output connections.
    /// - Returns: Weights values.
    ///
    static func XavierNormal(
        nbElems: Int,
        connectivityIO: (Int, Int)) -> [Float]
    {
        var values = [Float]()
        let std = sqrt(2) / sqrt(Float(connectivityIO.0 + connectivityIO.1))
        for _ in 0..<nbElems
        {
            values.append(randomNormal(mean: 0.0, standardDeviation: std))
        }
        return values
    }
    
    ///
    /// Xavier normal initialization method.
    ///
    /// - Parameters:
    ///     - nbElems: Number of weights to initialize.
    ///     - connectivityIO: Number of input and output connections.
    ///     - out: The output buffer.
    ///     - deviceID: GPU device.
    ///
    static func XavierNormal(
        nbElems: Int,
        connectivityIO: (Int, Int),
        out: FloatBuffer,
        deviceID: Int)
    {
        var array = [Float](repeating: 0.0, count: nbElems)
        array.withUnsafeMutableBufferPointer
        {
            ptr in
            
            let std = sqrt(2) / sqrt(Float(connectivityIO.0 + connectivityIO.1))
            guard var arrayDescriptor = BNNSNDArrayDescriptor(
                data: ptr,
                shape: .vector(nbElems)),
            let randomNumberGenerator = BNNSCreateRandomGenerator(
                BNNSRandomGeneratorMethodAES_CTR,
                nil) else
            {
                fatalError()
            }
            
            BNNSRandomFillNormalFloat(
                randomNumberGenerator,
                &arrayDescriptor,
                0.0,
                std
            )
            
            BNNSDestroyRandomGenerator(randomNumberGenerator)
        }
        out.initialize(array: &array)
    }
    
    ///
    /// Kaiming uniform initialization method.
    ///
    /// - Parameters:
    ///     - nbElems: Number of weights to initialize.
    ///     - coeff: Multiplicative coefficient.
    ///     - connectivityIO: Number of input and output connections.
    /// - Returns: Weights values.
    ///
    static func KaimingUniform(
        nbElems: Int,
        coeff: Float,
        connectivityIO: (Int, Int)) -> [Float]
    {
        var values = [Float]()
        let bound = sqrt(3) * coeff / sqrt(Float(connectivityIO.0))
        for _ in 0..<nbElems
        {
            values.append(Float.random(in: -bound..<bound))
        }
        return values
    }
    
    ///
    /// Kaiming uniform initialization method.
    ///
    /// - Parameters:
    ///     - nbElems: Number of weights to initialize.
    ///     - coeff: Multiplicative coefficient.
    ///     - connectivityIO: Number of input and output connections.
    ///     - out: The output buffer.
    ///     - deviceID: GPU device.
    ///
    static func KaimingUniform(
        nbElems: Int,
        coeff: Float,
        connectivityIO: (Int, Int),
        out: FloatBuffer,
        deviceID: Int)
    {
        var array = [Float](repeating: 0.0, count: nbElems)
        array.withUnsafeMutableBufferPointer
        {
            ptr in
            
            let bound = sqrt(3) * coeff / sqrt(Float(connectivityIO.0))
            guard var arrayDescriptor = BNNSNDArrayDescriptor(
                data: ptr,
                shape: .vector(nbElems)),
            let randomNumberGenerator = BNNSCreateRandomGenerator(
                BNNSRandomGeneratorMethodAES_CTR,
                nil) else
            {
                fatalError()
            }
            
            BNNSRandomFillUniformFloat(
                randomNumberGenerator,
                &arrayDescriptor,
                -bound,
                bound
            )
            
            BNNSDestroyRandomGenerator(randomNumberGenerator)
        }
        out.initialize(array: &array)
    }
    
    ///
    /// Xavier normal initialization method.
    ///
    /// - Parameters:
    ///     - nbElems: Number of weights to initialize.
    ///     - coeff: Multiplicative coefficient.
    ///     - connectivityIO: Number of input and output connections.
    /// - Returns: Weights values.
    ///
    static func KaimingNormal(
        nbElems: Int,
        coeff: Float,
        connectivityIO: (Int, Int)) -> [Float]
    {
        var values = [Float]()
        let std = coeff / sqrt(Float(connectivityIO.0))
        for _ in 0..<nbElems
        {
            values.append(randomNormal(mean: 0.0, standardDeviation: std))
        }
        return values
    }
    
    ///
    /// Kaiming normal initialization method.
    ///
    /// - Parameters:
    ///     - nbElems: Number of weights to initialize.
    ///     - coeff: Multiplicative coefficient.
    ///     - connectivityIO: Number of input and output connections.
    ///     - out: The output buffer.
    ///     - deviceID: GPU device.
    ///
    static func KaimingNormal(
        nbElems: Int,
        coeff: Float,
        connectivityIO: (Int, Int),
        out: FloatBuffer,
        deviceID: Int)
    {
        var array = [Float](repeating: 0.0, count: nbElems)
        array.withUnsafeMutableBufferPointer
        {
            ptr in
            
            let std = coeff / sqrt(Float(connectivityIO.0))
            guard var arrayDescriptor = BNNSNDArrayDescriptor(
                data: ptr,
                shape: .vector(nbElems)),
            let randomNumberGenerator = BNNSCreateRandomGenerator(
                BNNSRandomGeneratorMethodAES_CTR,
                nil) else
            {
                fatalError()
            }
            
            BNNSRandomFillNormalFloat(
                randomNumberGenerator,
                &arrayDescriptor,
                0.0,
                std
            )
            
            BNNSDestroyRandomGenerator(randomNumberGenerator)
        }
        out.initialize(array: &array)
    }
}

///
/// Generate numbers from a normal distribution.
///
/// - Parameters:
///     - mean: Mean of the normal distribution.
///     - standardDeviation: Standard deviation of the normal distribution.
/// - Returns: A number from the normal distribution.
///
public func randomNormal<T: BinaryFloatingPoint>(
    mean: T,
    standardDeviation: T) -> T
{
    let u1 = Double.random(in: 0..<1)
    let u2 = Double.random(in: 0..<1)
    let randStdNormal = sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
    let randNormal = mean + standardDeviation * T(randStdNormal)
    return randNormal
}
