//
// LayerInput.swift
// MAKit
//
// Created by Jean-François Reboud on 09/10/2022.
//

class InputArrays<T: Layer>
{
    unowned let _layer: T
    let nbElems: Int
    
    var _m: [Double] = []
    var _v: [Double] = []
    var _vHat: [Double] = []
    
    init(layer: T, nbElems: Int)
    {
        _layer = layer
        self.nbElems = nbElems
    }
    
    var m: [Double]
    {
        get {
            if _m.count == 0
            {
                _m = [Double](repeating: 0.0, count: nbElems)
            }
            return _m
        }
        set {
            if newValue.count != nbElems
            {
                fatalError(
                    "Momentum do not have the expected number of elements."
                )
            }
            _m = newValue
        }
    }
    var v: [Double]
    {
        get {
            if _v.count == 0
            {
                _v = [Double](repeating: 0.0, count: nbElems)
            }
            return _v
        }
        set {
            if newValue.count != nbElems
            {
                fatalError(
                    "Velocity do not have the expected number of elements."
                )
            }
            _v = newValue
        }
    }
    var vHat: [Double]
    {
        get {
            if _vHat.count == 0
            {
                _vHat = [Double](repeating: 0.0, count: nbElems)
            }
            return _vHat
        }
        set {
            if newValue.count != nbElems
            {
                fatalError(
                    "Velocity do not have the expected number of elements."
                )
            }
            _vHat = newValue
        }
    }
    
    /// Clean the momentum, preserving the weights.
    func reset()
    {
        _m = []
        _v = []
        _vHat = []
    }
}

class InputBuffers<T: Layer>
{
    unowned let _layer: T
    let nbElems: Int
    let deviceID: Int
    
    var _m: MetalBuffer<Float>! = nil
    var _v: MetalBuffer<Float>! = nil
    var _vHat: MetalBuffer<Float>! = nil
    
    init(layer: T,
         nbElems: Int,
         deviceID: Int)
    {
        _layer = layer
        self.nbElems = nbElems
        self.deviceID = deviceID
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
        _m = nil
        _v = nil
        _vHat = nil
    }
}
