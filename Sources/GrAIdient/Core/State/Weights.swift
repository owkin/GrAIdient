//
// Weights.swift
// GrAIdient
//
// Created by Jean-François Reboud on 05/10/2022.
//

/// API for the arrays needed to update the weights.
public protocol IWeightArrays
{
    /// Number of elementss in the different arrays.
    var nbElems: Int { get }
    
    /// Weights array: the array to update.
    var w: [Double] { get set }
    /// Gradients array.
    var g: [Double] { get set }
    /// Momentum array.
    var m: [Double] { get set }
    /// Velocity array.
    var v: [Double] { get set }
    /// Veclocity normalized array.
    var vHat: [Double] { get set }
    
    /// Clean the momentum..., preserving the weights.
    func reset()
}

/// Arrays needed to update the weights.
class WeightArrays: IWeightArrays
{
    /// Number of elements in the different arrays.
    let nbElems: Int
    
    var _w: [Double] = []
    var _g: [Double] = []
    var _m: [Double] = []
    var _v: [Double] = []
    var _vHat: [Double] = []
    
    ///
    /// Create a container of arrays.
    ///
    /// - Parameter nbElems: Number of elements in the different arrays.
    ///
    init(_ nbElems: Int)
    {
        self.nbElems = nbElems
    }
    
    /// Weights array: the array to update.
    var w: [Double]
    {
        get {
            if _w.count == 0
            {
                _w = [Double](repeating: 0.0, count: nbElems)
            }
            return _w
        }
        set {
            if newValue.count != nbElems
            {
                fatalError(
                    "Weights do not have the expected number of elements."
                )
            }
            _w = newValue
        }
    }
    /// Gradients array.
    var g: [Double]
    {
        get {
            if _g.count == 0
            {
                _g = [Double](repeating: 0.0, count: nbElems)
            }
            return _g
        }
        set {
            if newValue.count != nbElems
            {
                fatalError(
                    "Gradients do not have the expected number of elements."
                )
            }
            _g = newValue
        }
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
        _g = []
        _m = []
        _v = []
        _vHat = []
    }
}

/// Arrays needed to update the weights, shaped in grids.
class WeightGrids: WeightArrays
{
    //             j  0 ---------->[width-1,0]
    //0           i|
    //             |
    //             |
    //             |
    //             |
    //[0,height-1] v               [width-1,height-1]
    
    /// The width of the grid shape.
    public let width: Int
    /// The height of the grid shape.
    public let height: Int
    
    ///
    /// Create a container of arrays, shaped in grids.
    ///
    /// - Parameters:
    ///     - width: The width of the grid shape.
    ///     - height: The height of the grid shape.
    ///
    public init(width: Int, height: Int)
    {
        self.width = width
        self.height = height
        super.init(height * width)
    }

    ///
    /// Get the weights with a grid subscript.
    ///
    /// - Parameters:
    ///     - i: The row element.
    ///     - j: The column element.
    /// - Returns: The weight retrieved.
    ///
    func w(_ i: Int, _ j: Int) -> Double
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            return super.w[j + i * width]
        }
        else
        {
            fatalError("Indices (\(i), \(j)) out of bound.")
        }
    }
    ///
    /// Set the weights with a grid subscript.
    ///
    /// - Parameters:
    ///     - i: The row element.
    ///     - j: The column element.
    ///     - val: The value to set.
    ///
    func w(_ i: Int, _ j: Int, _ val: Double)
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            super.w[j + i * width] = val
        }
        else
        {
            fatalError("Indices (\(i), \(j)) out of bound.")
        }
    }
    
    ///
    /// Get the gradients with a grid subscript.
    ///
    /// - Parameters:
    ///     - i: The row element.
    ///     - j: The column element.
    /// - Returns: The gradient retrieved.
    ///
    func g(_ i: Int, _ j: Int) -> Double
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            return super.g[j + i * width]
        }
        else
        {
            fatalError("Indices (\(i), \(j)) out of bound.")
        }
    }
    ///
    /// Set the gradients with a grid subscript.
    ///
    /// - Parameters:
    ///     - i: The row element.
    ///     - j: The column element.
    ///     - val: The value to set.
    ///
    func g(_ i: Int, _ j: Int, _ val: Double)
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            super.g[j + i * width] = val
        }
        else
        {
            fatalError("Indices (\(i), \(j)) out of bound.")
        }
    }
    
    ///
    /// Get the momentum with a grid subscript.
    ///
    /// - Parameters:
    ///     - i: The row element.
    ///     - j: The column element.
    /// - Returns: The momentum retrieved.
    ///
    func m(_ i: Int, _ j: Int) -> Double
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            return super.m[j + i * width]
        }
        else
        {
            fatalError("Indices (\(i), \(j)) out of bound.")
        }
    }
    ///
    /// Set the momentum with a grid subscript.
    ///
    /// - Parameters:
    ///     - i: The row element.
    ///     - j: The column element.
    ///     - val: The value to set.
    ///
    func m(_ i: Int, _ j: Int, _ val: Double)
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            super.m[j + i * width] = val
        }
        else
        {
            fatalError("Indices (\(i), \(j)) out of bound.")
        }
    }
    
    ///
    /// Get the velocity with a grid subscript.
    ///
    /// - Parameters:
    ///     - i: The row element.
    ///     - j: The column element.
    /// - Returns: The velocity retrieved.
    ///
    func v(_ i: Int, _ j: Int) -> Double
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            return super.v[j + i * width]
        }
        else
        {
            fatalError("Indices (\(i), \(j)) out of bound.")
        }
    }
    ///
    /// Set the velocity with a grid subscript.
    ///
    /// - Parameters:
    ///     - i: The row element.
    ///     - j: The column element.
    ///     - val: The value to set.
    ///
    func v(_ i: Int, _ j: Int, _ val: Double)
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            super.v[j + i * width] = val
        }
        else
        {
            fatalError("Indices (\(i), \(j)) out of bound.")
        }
    }
    
    ///
    /// Get the velocity normalized with a grid subscript.
    ///
    /// - Parameters:
    ///     - i: The row element.
    ///     - j: The column element.
    /// - Returns: The velocity normalized retrieved.
    ///
    func vHat(_ i: Int, _ j: Int) -> Double
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            return super.vHat[j + i * width]
        }
        else
        {
            fatalError("Indices (\(i), \(j)) out of bound.")
        }
    }
    ///
    /// Set the velocity normalized with a grid subscript.
    ///
    /// - Parameters:
    ///     - i: The row element.
    ///     - j: The column element.
    ///     - val: The value to set.
    ///
    func vHat(_ i: Int, _ j: Int, _ val: Double)
    {
        if i >= 0 && j >= 0 && j < width && i < height
        {
            super.vHat[j + i * width] = val
        }
        else
        {
            fatalError("Indices (\(i), \(j)) out of bound.")
        }
    }
}
