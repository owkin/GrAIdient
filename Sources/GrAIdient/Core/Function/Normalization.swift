//
// Normalization.swift
// GrAIdient
//
// Created by Jean-François Reboud on 09/10/2022.
//

import Accelerate

/// Normalization function to be used in a layer.
class Normalization
{
    /// Slight modification to avoid "divide by 0" errors.
    static let _Ɛ: Double = 1e-5

    ///
    /// Forward Gradient Checking CPU.
    ///
    /// - Parameters:
    ///     - outs: The data to normalize.
    ///     - β: A bias to add to the normalization result.
    ///     - Ɣ: A weight to scale the normalization result.
    /// - Returns: The data normalized.
    ///
    static func forwardGC(outs: [Double],
                          β: Double,
                          Ɣ: Double) -> [Double]
    {
        let μ = vDSP.mean(outs)
        let tmp1 = vDSP.add(-μ, outs)
        let σ2 = vDSP.meanSquare(tmp1)
        let xHat = vDSP.divide(tmp1, sqrt(σ2 + _Ɛ))
        let outsNew = vDSP.add(β, vDSP.multiply(Ɣ, xHat))
        return outsNew
    }
    
    ///
    /// Forward Gradient Checking LayerNorm CPU.
    ///
    /// - Parameters:
    ///     - outs: The data to normalize.
    ///     - β: The biases to add to the normalization result.
    ///     - Ɣ: The weights to scale the normalization result.
    /// - Returns: The data normalized.
    ///
    static func forwardGC(outs: [Double],
                          β: [Double],
                          Ɣ: [Double]) -> [Double]
    {
        let μ = vDSP.mean(outs)
        let tmp1 = vDSP.add(-μ, outs)
        let σ2 = vDSP.meanSquare(tmp1)
        let xHat = vDSP.divide(tmp1, sqrt(σ2 + _Ɛ))
        let outsNew = vDSP.add(β, vDSP.multiply(Ɣ, xHat))
        return outsNew
    }
    
    ///
    /// Forward Gradient Checking RMSNorm CPU.
    ///
    /// - Parameters:
    ///     - outs: The data to normalize.
    ///     - Ɣ: The weights to scale the normalization result.
    /// - Returns: The data normalized.
    ///
    static func forwardΣGC(outs: [Double],
                          Ɣ: [Double]) -> [Double]
    {
        let σ2 = vDSP.meanSquare(outs)
        let xHat = vDSP.divide(outs, sqrt(σ2 + _Ɛ))
        let outsNew = vDSP.multiply(Ɣ, xHat)
        return outsNew
    }

    ///
    /// Forward Training CPU.
    ///
    /// - Parameters:
    ///     - outs: The data to normalize.
    ///     - β: A bias to add to the normalization result.
    ///     - Ɣ: A weight to scale the normalization result.
    /// - Returns: (The data normalized,
    ///            The data normalized without taking into account the bias and the weight,
    ///            The average of the data,
    ///            The deviation of the data).
    ///
    static func forward(outs: [Double],
                        β: Double,
                        Ɣ: Double) -> (outsNew: [Double],
                                       xHat: [Double],
                                       μ: Double,
                                       σ2: Double)
    {
        
        let μ = vDSP.mean(outs)
        let tmp1 = vDSP.add(-μ, outs)
        let σ2 = vDSP.meanSquare(tmp1)
        let xHat = vDSP.divide(tmp1, sqrt(σ2 + _Ɛ))
        let outsNew = vDSP.add(β, vDSP.multiply(Ɣ, xHat))
        
        return (outsNew: outsNew,
                xHat: xHat,
                μ: μ,
                σ2: σ2)
    }
    
    ///
    /// Forward LayerNorm CPU.
    ///
    /// - Parameters:
    ///     - outs: The data to normalize.
    ///     - β: The biases to add to the normalization result.
    ///     - Ɣ: The weights to scale the normalization result.
    /// - Returns: (The data normalized,
    ///            The data normalized without taking into account the bias and the weight,
    ///            The average of the data,
    ///            The deviation of the data).
    ///
    static func forward(outs: [Double],
                        β: [Double],
                        Ɣ: [Double]) -> (outsNew: [Double],
                                         xHat: [Double],
                                         μ: Double,
                                         σ2: Double)
    {
        
        let μ = vDSP.mean(outs)
        let tmp1 = vDSP.add(-μ, outs)
        let σ2 = vDSP.meanSquare(tmp1)
        let xHat = vDSP.divide(tmp1, sqrt(σ2 + _Ɛ))
        let outsNew = vDSP.add(β, vDSP.multiply(Ɣ, xHat))
        
        return (outsNew: outsNew,
                xHat: xHat,
                μ: μ,
                σ2: σ2)
    }
    
    ///
    /// Forward RMSNorm CPU.
    ///
    /// - Parameters:
    ///     - outs: The data to normalize.
    ///     - Ɣ: The weights to scale the normalization result.
    ///     - addUnitOffset: Whether to add unit offset or not.
    /// - Returns: (The data normalized,
    ///            The data normalized without taking into account the bias and the weight,
    ///            The deviation of the data).
    ///
    static func forwardΣ(outs: [Double],
                         Ɣ: [Double],
                         addUnitOffset: Bool) -> (outsNew: [Double],
                                                  xHat: [Double],
                                                  σ2: Double)
    {
        let σ2 = vDSP.meanSquare(outs)
        let xHat = vDSP.divide(outs, sqrt(σ2 + _Ɛ))
        var outsNew = vDSP.multiply(Ɣ, xHat)
        if addUnitOffset
        {
            outsNew = vDSP.add(xHat, outsNew)
        }
        
        return (outsNew: outsNew,
                xHat: xHat,
                σ2: σ2)
    }

    ///
    /// Forward Inference CPU.
    ///
    /// - Parameters:
    ///     - outs: The data to normalize.
    ///     - μ: A global average of data.
    ///     - σ2: A global deviation of data.
    ///     - β: A bias to add to the normalization result.
    ///     - Ɣ: A weight to scale the normalization result.
    /// - Returns: The data normalized.
    ///
    static func forward(outs: [Double],
                        μ: Double,
                        σ2: Double,
                        β: Double,
                        Ɣ: Double) -> [Double]
    {
        let tmp1 = vDSP.add(-μ, outs)
        let xHat = vDSP.divide(tmp1, sqrt(σ2 + _Ɛ))
        return vDSP.add(β, vDSP.multiply(Ɣ, xHat))
    }

    ///
    /// Backward Training CPU.
    ///
    /// - Parameters:
    ///     - delta: The gradients to back propagate.
    ///     - xHat: The data normalized without taking into account the bias and the weight.
    ///     - σ2: The deviation of the data.
    ///     - Ɣ: The weight that scaled the normalization result.
    /// - Returns: (The gradient taking into account the normalization,
    ///            The gradient of β,
    ///            The gradient of Ɣ).
    ///
    static func backward(delta: [Double],
                         xHat: [Double],
                         σ2: Double,
                         Ɣ: Double) -> (deltaNew: [Double],
                                        dβ: Double,
                                        dƔ: Double)
    {
        let nbElems = delta.count
        let factor = 1.0 / (Double(nbElems) * sqrt(σ2 + _Ɛ))
        
        let sum0 = vDSP.sum(delta)
        let mult0 = vDSP.multiply(delta, xHat)
        
        let sum1 = Ɣ * sum0
        let sum2 = Ɣ * vDSP.sum(mult0)
        
        let tmp1 = vDSP.add(
            multiplication: (delta, Ɣ * Double(nbElems)),
            multiplication: (xHat, -sum2))
        let deltaNew = vDSP.add(
            multiplication: (tmp1, factor), -factor * sum1)
        
        let dƔ = vDSP.sum(mult0)
        let dβ = sum0
        
        return (deltaNew: deltaNew,
                dβ: dβ,
                dƔ: dƔ)
    }
    
    ///
    /// Backward LayerNorm CPU.
    ///
    /// - Parameters:
    ///     - delta: The gradients to back propagate.
    ///     - xHat: The data normalized without taking into account the bias and the weight.
    ///     - σ2: The deviation of the data.
    ///     - Ɣ: The weights that scaled the normalization result.
    /// - Returns: The gradient taking into account the normalization.
    ///
    static func backward(delta: [Double],
                         xHat: [Double],
                         σ2: Double,
                         Ɣ: [Double]) -> [Double]
    {
        let nbElems = delta.count
        let factor = 1.0 / (Double(nbElems) * sqrt(σ2 + _Ɛ))
        
        let Ɣdelta = vDSP.multiply(Ɣ, delta)
        let sum1 = vDSP.sum(Ɣdelta)
        let sum2 = vDSP.sum(vDSP.multiply(Ɣdelta, xHat))
        
        let tmp1 = vDSP.add(
            multiplication: (Ɣdelta, Double(nbElems)),
            multiplication: (xHat, -sum2))
        let deltaNew = vDSP.add(
            multiplication: (tmp1, factor), -factor * sum1)
        
        return deltaNew
    }
    
    ///
    /// Backward RMSNorm CPU.
    ///
    /// - Parameters:
    ///     - delta: The gradients to back propagate.
    ///     - xHat: The data normalized without taking into account the bias and the weight.
    ///     - σ2: The deviation of the data.
    ///     - Ɣ: The weights that scaled the normalization result.
    ///     - addUnitOffset: Whether to add unit offset or not.
    /// - Returns: The gradient taking into account the normalization.
    ///
    static func backwardΣ(delta: [Double],
                          xHat: [Double],
                          σ2: Double,
                          Ɣ: [Double],
                          addUnitOffset: Bool) -> [Double]
    {
        let nbElems = delta.count
        let factor = 1.0 / (Double(nbElems) * sqrt(σ2 + _Ɛ))
        
        let Ɣdelta: [Double]
        if addUnitOffset
        {
            Ɣdelta = vDSP.multiply(vDSP.add(1, Ɣ), delta)
        }
        else
        {
            Ɣdelta = vDSP.multiply(Ɣ, delta)
        }
        
        let sum2 = vDSP.sum(vDSP.multiply(Ɣdelta, xHat))
        
        let tmp1 = vDSP.add(
            multiplication: (Ɣdelta, Double(nbElems)),
            multiplication: (xHat, -sum2))
        let deltaNew = vDSP.add(
            multiplication: (tmp1, factor), 0)
        
        return deltaNew
    }

    ///
    /// Backward Inference CPU.
    ///
    /// - Parameters:
    ///     - delta: The gradients to back propagate.
    ///     - σ2: The deviation of the data.
    ///     - Ɣ: The weight that scaled the normalization result.
    /// - Returns: The gradient taking into account the normalization.
    ///
    static func backward(delta: [Double],
                         σ2: Double,
                         Ɣ: Double) -> [Double]
    {
        return vDSP.multiply(Ɣ / sqrt(σ2 + _Ɛ), delta)
    }
}
