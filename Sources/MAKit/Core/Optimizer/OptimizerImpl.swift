//
// OptimizerImpl.swift
// MAKit
//
// Created by Jean-François Reboud on 08/10/2022.
//

import MetalKit

class OptimizerImpl
{
    let _kernel: OptimizerKernel
    
    private var _variables = [OptimizerVariable]()
    
    var params: MAKit.Optimizer.Params
    {
        get {
            return _kernel.params
        }
    }
    
    var alpha: Double
    {
        get {
            return _kernel.getValue(_variables[0])!
        }
    }
    var alphaPercent: Double
    {
        get {
            return _kernel.getPercent(_variables[0])!
        }
    }
    
    var lambda: Double?
    {
        get {
            return _kernel.getValue(_variables[1])
        }
    }
    var lambdaPercent: Double?
    {
        get {
            return _kernel.getPercent(_variables[1])
        }
    }
    
    var lowerBound: Double?
    {
        get {
            return nil
        }
    }
    var lowerBoundPercent: Double?
    {
        get {
            return nil
        }
    }
    
    var upperBound: Double?
    {
        get {
            return nil
        }
    }
    var upperBoundPercent: Double?
    {
        get {
            return nil
        }
    }
    
    init(_ kernel: OptimizerKernel)
    {
        _kernel = kernel
    }
    
    func initVariables()
    {
        _variables =
        [
            OptimizerVariable(
                name: "alpha",
                time: ConstEpochsVar(value: ConstVal(0.001))
            ),
            OptimizerVariable(
                name: "lambda",
                time: ConstEpochsVar(value: NullVal())
            )
        ]
        _kernel.setup(_variables)
    }
    
    func stepCPU(_ weight: IWeightArrays)
    {
        fatalError("Not implemented.")
    }
    
    func stepGPU(_ weight: IWeightBuffers)
    {
        fatalError("Not implemented.")
    }
    
    func incT()
    {
        _kernel.incT()
    }
    
    func incStep()
    {
        _kernel.incStep()
    }
    
    func incEpoch()
    {
        _kernel.incEpoch()
    }
}

class SGDOptimizer: OptimizerImpl
{
    override func stepCPU(_ weights: IWeightArrays)
    {
        var wVar = weights
        for i in 0..<weights.nbElems
        {
            var g = weights.g[i]
            if let λ = lambda
            {
                g += λ * weights.w[i]
            }
            wVar.w[i] -= alpha * g
        }
    }
    
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float] = [Float(alpha)]
        let pLambda: [Float] = [lambda != nil ? Float(lambda!) : 0.0]
        
        let command = MetalKernel.get.createCommand(
            "weightsSGD", deviceID: weights.deviceID)
        
        command.setBuffer(weights.g.metal, atIndex: 0)
        command.setBytes(pNbElems, atIndex: 1)
        command.setBytes(pAlpha, atIndex: 2)
        command.setBytes(pLambda, atIndex: 3)
        command.setBuffer(weights.w.metal, atIndex: 4)
        
        let threads = command.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
        let threadsPerGrid = MTLSize(width: nbElems, height: 1, depth: 1)
        command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
    }
}

class SGDMomentumOptimizer: OptimizerImpl
{
    let _β1 = 0.9
    
    override func stepCPU(_ weights: IWeightArrays)
    {
        var wVar = weights
        for i in 0..<weights.nbElems
        {
            var g = weights.g[i]
            if let λ = lambda
            {
                g += λ * weights.w[i]
            }
            
            let v = _β1 * weights.m[i] + alpha * g
            wVar.m[i] = v
            wVar.w[i] -= v
        }
    }
    
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float] = [Float(alpha)]
        let pLambda: [Float] = [lambda != nil ? Float(lambda!) : 0.0]
        
        let command = MetalKernel.get.createCommand(
            "weightsMomentum", deviceID: weights.deviceID)
        
        command.setBuffer(weights.g.metal, atIndex: 0)
        command.setBytes(pNbElems, atIndex: 1)
        command.setBytes(pAlpha, atIndex: 2)
        command.setBytes(pLambda, atIndex: 3)
        command.setBuffer(weights.w.metal, atIndex: 4)
        command.setBuffer(weights.m.metal, atIndex: 5)
        
        let threads = command.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
        let threadsPerGrid = MTLSize(width: nbElems, height: 1, depth: 1)
        command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
    }
}

class AdamOptimizer: OptimizerImpl
{
    let _β1 = 0.9
    let _β2 = 0.999
    let _Ɛ = 0.00000001
    
    override func stepCPU(_ weights: IWeightArrays)
    {
        var wVar = weights
        let t = Double(_kernel.params.t)
        
        for i in 0..<weights.nbElems
        {
            var g = weights.g[i]
            if let λ = lambda
            {
                g += λ * weights.w[i]
            }
            
            var m = _β1 * weights.m[i] + (1 - _β1) * g
            var v = _β2 * weights.v[i] + (1 - _β2) * g * g
            
            wVar.m[i] = m
            wVar.v[i] = v
            
            m /= (1 - pow(_β1, t))
            v /= (1 - pow(_β2, t))
            
            wVar.w[i] -= alpha * m / (sqrt(v) + _Ɛ)
        }
    }
    
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        let t = Double(_kernel.params.t)
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float] = [Float(alpha)]
        let pLambda: [Float] = [lambda != nil ? Float(lambda!) : 0.0]
        let pT: [Float] = [Float(t)]
        
        let command = MetalKernel.get.createCommand(
            "weightsAdam", deviceID: weights.deviceID)
        
        command.setBuffer(weights.g.metal, atIndex: 0)
        command.setBytes(pNbElems, atIndex: 1)
        command.setBytes(pAlpha, atIndex: 2)
        command.setBytes(pLambda, atIndex: 3)
        command.setBytes(pT, atIndex: 4)
        command.setBuffer(weights.w.metal, atIndex: 5)
        command.setBuffer(weights.m.metal, atIndex: 6)
        command.setBuffer(weights.v.metal, atIndex: 7)
        
        let threads = command.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
        let threadsPerGrid = MTLSize(width: nbElems, height: 1, depth: 1)
        command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
    }
}

class AMSGradOptimizer: OptimizerImpl
{
    let _β1 = 0.9
    let _β2 = 0.999
    let _Ɛ = 0.00000001
    
    override func stepCPU(_ weights: IWeightArrays)
    {
        var wVar = weights
        let t = Double(_kernel.params.t)
        
        for i in 0..<weights.nbElems
        {
            var g = weights.g[i]
            if let λ = lambda
            {
                g += λ * weights.w[i]
            }
            
            var m = _β1 * weights.m[i] + (1 - _β1) * g
            let v = _β2 * weights.v[i] + (1 - _β2) * g * g
            var vHat = max(v, weights.vHat[i])
            
            wVar.m[i] = m
            wVar.v[i] = v
            wVar.vHat[i] = vHat
            
            m /= (1 - pow(_β1, t))
            vHat /= (1 - pow(_β2, t))
            
            wVar.w[i] -= alpha * m / (sqrt(vHat) + _Ɛ)
        }
    }
    
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        let t = Double(_kernel.params.t)
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float] = [Float(alpha)]
        let pLambda: [Float] = [lambda != nil ? Float(lambda!) : 0.0]
        let pT: [Float] = [Float(t)]
        
        let command = MetalKernel.get.createCommand(
            "weightsAMSGrad", deviceID: weights.deviceID)
        
        command.setBuffer(weights.g.metal, atIndex: 0)
        command.setBytes(pNbElems, atIndex: 1)
        command.setBytes(pAlpha, atIndex: 2)
        command.setBytes(pLambda, atIndex: 3)
        command.setBytes(pT, atIndex: 4)
        command.setBuffer(weights.w.metal, atIndex: 5)
        command.setBuffer(weights.m.metal, atIndex: 6)
        command.setBuffer(weights.v.metal, atIndex: 7)
        command.setBuffer(weights.vHat.metal, atIndex: 8)
        
        let threads = command.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
        let threadsPerGrid = MTLSize(width: nbElems, height: 1, depth: 1)
        command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
    }
}

class AdamRectifiedOptimizer: OptimizerImpl
{
    let _β1 = 0.9
    let _β2 = 0.999
    let _Ɛ = 0.00000001
    
    override func stepCPU(_ weights: IWeightArrays)
    {
        var wVar = weights
        let t = Double(_kernel.params.t)
        let ρinf = 2.0 / (1.0 - _β2) - 1.0
        
        for i in 0..<weights.nbElems
        {
            var g = weights.g[i]
            if let λ = lambda
            {
                g += λ * weights.w[i]
            }
            
            var m = _β1 * weights.m[i] + (1 - _β1) * g
            let v = _β2 * weights.v[i] + (1 - _β2) * g * g
            
            wVar.m[i] = m
            wVar.v[i] = v
            
            m /= (1 - pow(_β1, t))
            let ρ = ρinf - 2.0 * t * pow(_β2, t) / (1 - pow(_β2, t))
            
            if ρ > 5.0
            {
                let l = sqrt((1 - pow(_β2, t)) / (v + _Ɛ))
                let r = sqrt(((ρ - 4.0) * (ρ - 2.0) * ρinf) /
                             ((ρinf - 4.0) * (ρinf - 2.0) * ρ))
                                    
                wVar.w[i] -= alpha * m * r * l
            }
            else
            {
                wVar.w[i] -= alpha * m
            }
        }
    }
    
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        let t = Double(_kernel.params.t)
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float] = [Float(alpha)]
        let pLambda: [Float] = [lambda != nil ? Float(lambda!) : 0.0]
        let pT: [Float] = [Float(t)]
        
        let command = MetalKernel.get.createCommand(
            "weightsAdamRectified", deviceID: weights.deviceID)
        
        command.setBuffer(weights.g.metal, atIndex: 0)
        command.setBytes(pNbElems, atIndex: 1)
        command.setBytes(pAlpha, atIndex: 2)
        command.setBytes(pLambda, atIndex: 3)
        command.setBytes(pT, atIndex: 4)
        command.setBuffer(weights.w.metal, atIndex: 5)
        command.setBuffer(weights.m.metal, atIndex: 6)
        command.setBuffer(weights.v.metal, atIndex: 7)
        
        let threads = command.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
        let threadsPerGrid = MTLSize(width: nbElems, height: 1, depth: 1)
        command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
    }
}

class BoundOptimizer: OptimizerImpl
{
    private var _variables = [OptimizerVariable]()
    
    override var lowerBound: Double?
    {
        get {
            return _kernel.getValue(_variables[0])
        }
    }
    override var lowerBoundPercent: Double?
    {
        get {
            return _kernel.getPercent(_variables[0])
        }
    }
    
    override var upperBound: Double?
    {
        get {
            return _kernel.getValue(_variables[1])
        }
    }
    override var upperBoundPercent: Double?
    {
        get {
            return _kernel.getPercent(_variables[1])
        }
    }
    
    override func initVariables()
    {
        super.initVariables()
        
        _variables =
        [
            OptimizerVariable(
                name: "lowerBound",
                time: ConstEpochsVar(value: ConstVal(0.0001))
            ),
            OptimizerVariable(
                name: "upperBound",
                time: ConstEpochsVar(value: ConstVal(0.005))
            )
        ]
        _kernel.setup(_variables)
    }
}

class AdaBoundOptimizer: BoundOptimizer
{
    let _β1 = 0.9
    let _β2 = 0.999
    let _Ɛ = 0.00000001
    
    override func stepCPU(_ weights: IWeightArrays)
    {
        var wVar = weights
        let t = Double(_kernel.params.t)
        
        for i in 0..<weights.nbElems
        {
            var g = weights.g[i]
            if let λ = lambda
            {
                g += λ * weights.w[i]
            }
            
            let m = _β1 * weights.m[i] + (1 - _β1) * g
            let v = _β2 * weights.v[i] + (1 - _β2) * g * g
            
            wVar.m[i] = m
            wVar.v[i] = v
            
            var alphaHat = alpha *
                sqrt(1 - pow(_β2, t)) / ((sqrt(v) + _Ɛ) * (1 - pow(_β1, t)))
            if alphaHat < lowerBound!
            {
                alphaHat = lowerBound!
            }
            else if alphaHat > upperBound!
            {
                alphaHat = upperBound!
            }
            
            wVar.w[i] -= alphaHat * m
        }
    }
    
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        let t = Double(_kernel.params.t)
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float] = [Float(alpha)]
        let pLambda: [Float] = [lambda != nil ? Float(lambda!) : 0.0]
        let pT: [Float] = [Float(t)]
        let pLowerBound: [Float] = [Float(lowerBound!)]
        let pUpperBound: [Float] = [Float(upperBound!)]
        
        let command = MetalKernel.get.createCommand(
            "weightsAdaBound", deviceID: weights.deviceID)
        
        command.setBuffer(weights.g.metal, atIndex: 0)
        command.setBytes(pNbElems, atIndex: 1)
        command.setBytes(pAlpha, atIndex: 2)
        command.setBytes(pLambda, atIndex: 3)
        command.setBytes(pT, atIndex: 4)
        command.setBytes(pLowerBound, atIndex: 5)
        command.setBytes(pUpperBound, atIndex: 6)
        command.setBuffer(weights.w.metal, atIndex: 7)
        command.setBuffer(weights.m.metal, atIndex: 8)
        command.setBuffer(weights.v.metal, atIndex: 9)
        
        let threads = command.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
        let threadsPerGrid = MTLSize(width: nbElems, height: 1, depth: 1)
        command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
    }
}

class AMSBoundOptimizer: BoundOptimizer
{
    let _β1 = 0.9
    let _β2 = 0.999
    let _Ɛ = 0.00000001
    
    override func stepCPU(_ weights: IWeightArrays)
    {
        var wVar = weights
        let t = Double(_kernel.params.t)
        
        for i in 0..<weights.nbElems
        {
            var g = weights.g[i]
            if let λ = lambda
            {
                g += λ * weights.w[i]
            }
            
            let m = _β1 * weights.m[i] + (1 - _β1) * g
            let v = _β2 * weights.v[i] + (1 - _β2) * g * g
            let vHat = max(v, weights.vHat[i])
            
            wVar.m[i] = m
            wVar.v[i] = v
            wVar.vHat[i] = vHat
            
            var alphaHat = alpha * sqrt(1 - pow(_β2, t)) /
                ((sqrt(vHat) + _Ɛ) * (1 - pow(_β1, t)))
            if alphaHat < lowerBound!
            {
                alphaHat = lowerBound!
            }
            else if alphaHat > upperBound!
            {
                alphaHat = upperBound!
            }
            
            wVar.w[i] -= alphaHat * m
        }
    }
    
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        let t = Double(_kernel.params.t)
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float] = [Float(alpha)]
        let pLambda: [Float] = [lambda != nil ? Float(lambda!) : 0.0]
        let pT: [Float] = [Float(t)]
        let pLowerBound: [Float] = [Float(lowerBound!)]
        let pUpperBound: [Float] = [Float(upperBound!)]
        
        let command = MetalKernel.get.createCommand(
            "weightsAMSBound", deviceID: weights.deviceID)
        
        command.setBuffer(weights.g.metal, atIndex: 0)
        command.setBytes(pNbElems, atIndex: 1)
        command.setBytes(pAlpha, atIndex: 2)
        command.setBytes(pLambda, atIndex: 3)
        command.setBytes(pT, atIndex: 4)
        command.setBytes(pLowerBound, atIndex: 5)
        command.setBytes(pUpperBound, atIndex: 6)
        command.setBuffer(weights.w.metal, atIndex: 7)
        command.setBuffer(weights.m.metal, atIndex: 8)
        command.setBuffer(weights.v.metal, atIndex: 9)
        command.setBuffer(weights.vHat.metal, atIndex: 10)
        
        let threads = command.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
        let threadsPerGrid = MTLSize(width: nbElems, height: 1, depth: 1)
        command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
        command.enqueue()
    }
}
