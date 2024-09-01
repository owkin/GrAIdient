//
// LayerInput.swift
// GrAIdient
//
// Created by Jean-François Reboud on 09/10/2022.
//

/// Arrays needed to update the inputs of a layer.
class InputArrays<T: Layer>
{
    /// The link to the layer.
    unowned let _layer: T
    /// Number of elements in the different arrays.
    let nbElems: Int
    
    var _m: [Double] = []
    var _v: [Double] = []
    var _vHat: [Double] = []
    
    ///
    /// Create a container of arrays.
    ///
    /// - Parameters:
    ///     - layer: The link to the layer.
    ///     - nbElems: Number of elements in the different arrays.
    ///
    init(layer: T, nbElems: Int)
    {
        _layer = layer
        self.nbElems = nbElems
    }
    
    /// Momentum array.
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
    /// Velocity array.
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
    /// Veclocity normalized array.
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
    
    /// Clean the momentum..., preserving the weights.
    func reset()
    {
        _m = []
        _v = []
        _vHat = []
    }
}

/// GPU buffers needed to update the inputs of a layer.
class InputBuffers<T: Layer>
{
    /// The link to the layer.
    unowned let _layer: T
    
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
    ///     - layer: The link to the layer.
    ///     - nbElems: Number of elements in the different buffers.
    ///     - deviceID: GPU device where the buffers are sent.
    ///
    init(layer: T,
         nbElems: Int,
         deviceID: Int)
    {
        _layer = layer
        m = FloatBuffer(nbElems: nbElems, deviceID: deviceID)
        v = FloatBuffer(nbElems: nbElems, deviceID: deviceID)
        vHat = FloatBuffer(nbElems: nbElems, deviceID: deviceID)
    }
    
    /// Clean the momentum..., preserving the weights.
    func reset()
    {
        m.reset()
        v.reset()
        vHat.reset()
    }
}
