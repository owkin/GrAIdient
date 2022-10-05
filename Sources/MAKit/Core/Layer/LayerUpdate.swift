//
// LayerUpdate.swift
// MAKit
//
// Created by Jean-François Reboud on 05/10/2022.
//

import Foundation

/// Running phase of a model.
public enum Phase
{
    case Training, Inference
}

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

public protocol IWeightBuffers
{
    var deviceID: Int { get }
    var nbElems: Int { get }
    
    var w: MetalBuffer<Float> { get }
    var g: MetalBuffer<Float> { get }
    var m: MetalBuffer<Float> { get }
    var v: MetalBuffer<Float> { get }
    var vHat: MetalBuffer<Float> { get }
    
    /// Clean the momentum, preserving the weights.
    func reset()
}

extension IWeightBuffers
{
    var w_p: MetalPrivateBuffer<Float>?
    {
        get {
            return w as? MetalPrivateBuffer<Float>
        }
    }
    var w_s: MetalSharedBuffer<Float>?
    {
        get {
            return w as? MetalSharedBuffer<Float>
        }
    }
    
    var g_p: MetalPrivateBuffer<Float>?
    {
        get {
            return g as? MetalPrivateBuffer<Float>
        }
    }
    var g_s: MetalSharedBuffer<Float>?
    {
        get {
            return g as? MetalSharedBuffer<Float>
        }
    }
}

class WeightBuffers: IWeightBuffers
{
    let nbElems: Int
    let deviceID: Int
    
    var _w: MetalBuffer<Float>! = nil
    var _g: MetalBuffer<Float>! = nil
    var _m: MetalBuffer<Float>! = nil
    var _v: MetalBuffer<Float>! = nil
    var _vHat: MetalBuffer<Float>! = nil
    
    init(nbElems: Int,
         deviceID: Int)
    {
        self.nbElems = nbElems
        self.deviceID = deviceID
    }
    
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
    
    /// Clean the momentum, preserving the weights.
    func reset()
    {
        // do not touch _w
        _g = nil
        _m = nil
        _v = nil
        _vHat = nil
    }
}
