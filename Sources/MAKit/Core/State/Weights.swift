//
// Weights.swift
// MAKit
//
// Created by Jean-François Reboud on 05/10/2022.
//

public protocol IWeightArrays
{
    var nbElems: Int { get }
    
    var w: [Double] { get set }
    var g: [Double] { get set }
    var m: [Double] { get set }
    var v: [Double] { get set }
    var vHat: [Double] { get set }
    
    /// Clean the momentum, preserving the weights.
    func reset()
}

class WeightArrays: IWeightArrays
{
    let nbElems: Int
    
    var _w: [Double] = []
    var _g: [Double] = []
    var _m: [Double] = []
    var _v: [Double] = []
    var _vHat: [Double] = []
    
    init(_ nbElems: Int)
    {
        self.nbElems = nbElems
    }
    
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
        _g = []
        _m = []
        _v = []
        _vHat = []
    }
}

class WeightGrids: WeightArrays
{
    public let width: Int
    public let height: Int
    
    public init(width: Int, height: Int)
    {
        self.width = width
        self.height = height
        super.init(height * width)
    }
    
    //             j  0 ---------->[width-1,0]
    //0           i|
    //             |
    //             |
    //             |
    //             |
    //[0,height-1] v               [width-1,height-1]
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
