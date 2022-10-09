//
// Normalization.swift
// MAKit
//
// Created by Jean-François Reboud on 09/10/2022.
//

import Accelerate

class Normalization
{
    static let _Ɛ: Double = 1e-5

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

    static func backward(delta: [Double],
                         σ2: Double,
                         Ɣ: Double) -> [Double]
    {
        return vDSP.multiply(Ɣ / sqrt(σ2 + _Ɛ), delta)
    }
}
