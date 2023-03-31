//
// LayerUpdate.swift
// GrAIdient
//
// Created by Jean-François Reboud on 05/10/2022.
//

import Foundation

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
    var w: MetalBuffer<Float> { get }
    /// Gradients buffer.
    var g: MetalBuffer<Float> { get }
    /// Momentum buffer.
    var m: MetalBuffer<Float> { get }
    /// Velocity buffer.
    var v: MetalBuffer<Float> { get }
    /// Velocity normalized buffer.
    var vHat: MetalBuffer<Float> { get }
    
    /// Clean the momentum..., preserving the weights.
    func reset()
}

extension IWeightBuffers
{
    /// Get the weights as a private buffer.
    var w_p: MetalPrivateBuffer<Float>?
    {
        get {
            return w as? MetalPrivateBuffer<Float>
        }
    }
    /// Get the weights as a shared buffer.
    var w_s: MetalSharedBuffer<Float>?
    {
        get {
            return w as? MetalSharedBuffer<Float>
        }
    }
    
    /// Get the gradient buffer as a private buffer.
    var g_p: MetalPrivateBuffer<Float>?
    {
        get {
            return g as? MetalPrivateBuffer<Float>
        }
    }
    /// Get the gradient buffer as a shared buffer.
    var g_s: MetalSharedBuffer<Float>?
    {
        get {
            return g as? MetalSharedBuffer<Float>
        }
    }
}

/// GPU buffers needed to update the weights.
class WeightBuffers: IWeightBuffers
{
    /// Number of elements in the different buffers.
    let nbElems: Int
    /// GPU device where the buffers are sent.
    let deviceID: Int
    
    var _w: MetalBuffer<Float>! = nil
    var _g: MetalBuffer<Float>! = nil
    var _m: MetalBuffer<Float>! = nil
    var _v: MetalBuffer<Float>! = nil
    var _vHat: MetalBuffer<Float>! = nil
    
    ///
    /// Create a container of buffers.
    ///
    /// - Parameters:
    ///     - nbElems: Number of elements in the different buffers.
    ///     - deviceID: GPU device where the buffers are sent.
    ///
    init(nbElems: Int, deviceID: Int)
    {
        self.nbElems = nbElems
        self.deviceID = deviceID
    }
    
    /// Weights buffer: the buffer to be update.
    var w: MetalBuffer<Float>
    {
        get {
            if _w == nil
            {
                _w = MetalPrivateBuffer<Float>(nbElems, deviceID: deviceID)
            }
            return _w
        }
    }
    
    /// Gradients buffer.
    var g: MetalBuffer<Float>
    {
        get {
            if _g == nil
            {
                _g = MetalPrivateBuffer<Float>(nbElems, deviceID: deviceID)
            }
            return _g
        }
    }
    
    /// Momentum buffer.
    var m: MetalBuffer<Float>
    {
        get {
            if _m == nil
            {
                _m = MetalPrivateBuffer<Float>(nbElems, deviceID: deviceID)
            }
            return _m
        }
    }
    
    /// Velocity buffer.
    var v: MetalBuffer<Float>
    {
        get {
            if _v == nil
            {
                _v = MetalPrivateBuffer<Float>(nbElems, deviceID: deviceID)
            }
            return _v
        }
    }
    
    /// Velocity normalized buffer.
    var vHat: MetalBuffer<Float>
    {
        get {
            if _vHat == nil
            {
                _vHat = MetalPrivateBuffer<Float>(nbElems, deviceID: deviceID)
            }
            return _vHat
        }
    }
    
    /// Clean the momentum..., preserving the weights.
    func reset()
    {
        // do not touch _w
        _g = nil
        _m = nil
        _v = nil
        _vHat = nil
    }
}
