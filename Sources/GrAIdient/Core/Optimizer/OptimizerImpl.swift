//
// OptimizerImpl.swift
// GrAIdient
//
// Created by Jean-François Reboud on 08/10/2022.
//

import Foundation

/// Abstract part of the implementation of the weight update algorithm.
class OptimizerImpl
{
    /// Object that contains the state of the optimization process.
    let _kernel: OptimizerKernel
    
    /// List of variables needed during the weight update.
    private var _variables = [OptimizerVariable]()
    
    /// Get the parameters of the optimizer algorithm.
    var params: GrAI.Optimizer.Params
    {
        get {
            return _kernel.params
        }
    }
    
    /// Get alpha (learning rate) value.
    var alpha: Double
    {
        get {
            return _kernel.getValue(_variables[0])!
        }
    }
    /// Get alpha (learning rate) percent value.
    var alphaPercent: Double
    {
        get {
            return _kernel.getPercent(_variables[0])!
        }
    }
    
    /// Get lambda (weight decay) value.
    var lambda: Double?
    {
        get {
            return _kernel.getValue(_variables[1])
        }
    }
    /// Get lambda (weight decay) percent value.
    var lambdaPercent: Double?
    {
        get {
            return _kernel.getPercent(_variables[1])
        }
    }
    
    /// Get lower bound value.
    var lowerBound: Double?
    {
        get {
            return nil
        }
    }
    /// Get lower bound percent value.
    var lowerBoundPercent: Double?
    {
        get {
            return nil
        }
    }
    
    /// Get upper bound value.
    var upperBound: Double?
    {
        get {
            return nil
        }
    }
    /// Get upper bound percent value.
    var upperBoundPercent: Double?
    {
        get {
            return nil
        }
    }
    
    ///
    /// Create the implementation of the weight update algorithm.
    ///
    /// - Parameter kernel: The state of the optimization process.
    ///
    init(_ kernel: OptimizerKernel)
    {
        _kernel = kernel
    }
    
    /// Set default values to the list of internal variables.
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
    
    ///
    /// The weight update function in CPU execution context.
    ///
    /// - Parameter weight: The weight arrays to update.
    ///
    func stepCPU(_ weight: IWeightArrays)
    {
        fatalError("Not implemented.")
    }
    
    ///
    /// The weight update function in GPU execution context.
    ///
    /// - Parameter weight: The weight buffers to update.
    ///
    func stepGPU(_ weight: IWeightBuffers)
    {
        fatalError("Not implemented.")
    }
    
    /// Increment internal time state.
    func incT()
    {
        _kernel.incT()
    }
    
    /// Increment internal step state.
    func incStep()
    {
        _kernel.incStep()
    }
}

/// Simple SGD optimizer.
class SGDOptimizer: OptimizerImpl
{
    ///
    /// The weight update function in CPU execution context.
    ///
    /// - Parameter weight: The weight arrays to update.
    ///
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
    
    ///
    /// The weight update function in GPU execution context.
    ///
    /// - Parameter weight: The weight buffers to update.
    ///
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float16] = [Float16(alpha)]
        let pLambda: [Float16] = [lambda != nil ? Float16(lambda!) : 0.0]
        
        let command = MetalKernel.get.createCommand(
            "weightsSGD", deviceID: weights.deviceID
        )
        command.setBuffer(weights.g.metal, atIndex: 0)
        command.setBytes(pNbElems, atIndex: 1)
        command.setBytes(pAlpha, atIndex: 2)
        command.setBytes(pLambda, atIndex: 3)
        command.setBuffer(weights.w.metal, atIndex: 4)
        
        command.dispatchThreads(nbElems)
        command.enqueue()
    }
}

/// SGD with momentum optimizer.
class SGDMomentumOptimizer: OptimizerImpl
{
    let _β1 = 0.9
    
    ///
    /// The weight update function in CPU execution context.
    ///
    /// - Parameter weight: The weight arrays to update.
    ///
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
    
    ///
    /// The weight update function in GPU execution context.
    ///
    /// - Parameter weight: The weight buffers to update.
    ///
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float16] = [Float16(alpha)]
        let pLambda: [Float16] = [lambda != nil ? Float16(lambda!) : 0.0]
        
        let command = MetalKernel.get.createCommand(
            "weightsMomentum", deviceID: weights.deviceID
        )
        command.setBuffer(weights.g.metal, atIndex: 0)
        command.setBytes(pNbElems, atIndex: 1)
        command.setBytes(pAlpha, atIndex: 2)
        command.setBytes(pLambda, atIndex: 3)
        command.setBuffer(weights.w.metal, atIndex: 4)
        command.setBuffer(weights.m.metal, atIndex: 5)
        
        command.dispatchThreads(nbElems)
        command.enqueue()
    }
}

/// Adam optimizer.
class AdamOptimizer: OptimizerImpl
{
    let _β1 = 0.9
    let _β2 = 0.999
    let _Ɛ = 0.00000001
    
    ///
    /// The weight update function in CPU execution context.
    ///
    /// - Parameter weight: The weight arrays to update.
    ///
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
    
    ///
    /// The weight update function in GPU execution context.
    ///
    /// - Parameter weight: The weight buffers to update.
    ///
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        let t = Double(_kernel.params.t)
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float16] = [Float16(alpha)]
        let pLambda: [Float16] = [lambda != nil ? Float16(lambda!) : 0.0]
        let pT: [Float16] = [Float16(t)]
        
        let command = MetalKernel.get.createCommand(
            "weightsAdam", deviceID: weights.deviceID
        )
        command.setBuffer(weights.g.metal, atIndex: 0)
        command.setBytes(pNbElems, atIndex: 1)
        command.setBytes(pAlpha, atIndex: 2)
        command.setBytes(pLambda, atIndex: 3)
        command.setBytes(pT, atIndex: 4)
        command.setBuffer(weights.w.metal, atIndex: 5)
        command.setBuffer(weights.m.metal, atIndex: 6)
        command.setBuffer(weights.v.metal, atIndex: 7)
        
        command.dispatchThreads(nbElems)
        command.enqueue()
    }
}

/// AMSGrad optimizer.
class AMSGradOptimizer: OptimizerImpl
{
    let _β1 = 0.9
    let _β2 = 0.999
    let _Ɛ = 0.00000001
    
    ///
    /// The weight update function in CPU execution context.
    ///
    /// - Parameter weight: The weight arrays to update.
    ///
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
    
    ///
    /// The weight update function in GPU execution context.
    ///
    /// - Parameter weight: The weight buffers to update.
    ///
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        let t = Double(_kernel.params.t)
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float16] = [Float16(alpha)]
        let pLambda: [Float16] = [lambda != nil ? Float16(lambda!) : 0.0]
        let pT: [Float16] = [Float16(t)]
        
        let command = MetalKernel.get.createCommand(
            "weightsAMSGrad", deviceID: weights.deviceID
        )
        command.setBuffer(weights.g.metal, atIndex: 0)
        command.setBytes(pNbElems, atIndex: 1)
        command.setBytes(pAlpha, atIndex: 2)
        command.setBytes(pLambda, atIndex: 3)
        command.setBytes(pT, atIndex: 4)
        command.setBuffer(weights.w.metal, atIndex: 5)
        command.setBuffer(weights.m.metal, atIndex: 6)
        command.setBuffer(weights.v.metal, atIndex: 7)
        command.setBuffer(weights.vHat.metal, atIndex: 8)
        
        command.dispatchThreads(nbElems)
        command.enqueue()
    }
}

/// Adam rectified optimizer.
class AdamRectifiedOptimizer: OptimizerImpl
{
    let _β1 = 0.9
    let _β2 = 0.999
    let _Ɛ = 0.00000001
    
    ///
    /// The weight update function in CPU execution context.
    ///
    /// - Parameter weight: The weight arrays to update.
    ///
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
    
    ///
    /// The weight update function in GPU execution context.
    ///
    /// - Parameter weight: The weight buffers to update.
    ///
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        let t = Double(_kernel.params.t)
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float16] = [Float16(alpha)]
        let pLambda: [Float16] = [lambda != nil ? Float16(lambda!) : 0.0]
        let pT: [Float16] = [Float16(t)]
        
        let command = MetalKernel.get.createCommand(
            "weightsAdamRectified", deviceID: weights.deviceID
        )
        command.setBuffer(weights.g.metal, atIndex: 0)
        command.setBytes(pNbElems, atIndex: 1)
        command.setBytes(pAlpha, atIndex: 2)
        command.setBytes(pLambda, atIndex: 3)
        command.setBytes(pT, atIndex: 4)
        command.setBuffer(weights.w.metal, atIndex: 5)
        command.setBuffer(weights.m.metal, atIndex: 6)
        command.setBuffer(weights.v.metal, atIndex: 7)
        
        command.dispatchThreads(nbElems)
        command.enqueue()
    }
}

/// An optimizer with lower and upper bounds.
class BoundOptimizer: OptimizerImpl
{
    /// List of variables needed during the weight update.
    private var _variables = [OptimizerVariable]()
    
    /// Get lower bound value.
    override var lowerBound: Double?
    {
        get {
            return _kernel.getValue(_variables[0])
        }
    }
    /// Get lower bound percent value.
    override var lowerBoundPercent: Double?
    {
        get {
            return _kernel.getPercent(_variables[0])
        }
    }
    
    /// Get upper bound value.
    override var upperBound: Double?
    {
        get {
            return _kernel.getValue(_variables[1])
        }
    }
    /// Get upper bound percent value.
    override var upperBoundPercent: Double?
    {
        get {
            return _kernel.getPercent(_variables[1])
        }
    }
    
    /// Set default values to the list of internal variables.
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

/// Ada bound optimizer.
class AdaBoundOptimizer: BoundOptimizer
{
    let _β1 = 0.9
    let _β2 = 0.999
    let _Ɛ = 0.00000001
    
    ///
    /// The weight update function in CPU execution context.
    ///
    /// - Parameter weight: The weight arrays to update.
    ///
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
    
    ///
    /// The weight update function in GPU execution context.
    ///
    /// - Parameter weight: The weight buffers to update.
    ///
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        let t = Double(_kernel.params.t)
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float16] = [Float16(alpha)]
        let pLambda: [Float16] = [lambda != nil ? Float16(lambda!) : 0.0]
        let pT: [Float16] = [Float16(t)]
        let pLowerBound: [Float16] = [Float16(lowerBound!)]
        let pUpperBound: [Float16] = [Float16(upperBound!)]
        
        let command = MetalKernel.get.createCommand(
            "weightsAdaBound", deviceID: weights.deviceID
        )
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
        
        command.dispatchThreads(nbElems)
        command.enqueue()
    }
}

/// AMS bound optimizer.
class AMSBoundOptimizer: BoundOptimizer
{
    let _β1 = 0.9
    let _β2 = 0.999
    let _Ɛ = 0.00000001
    
    ///
    /// The weight update function in CPU execution context.
    ///
    /// - Parameter weight: The weight arrays to update.
    ///
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
    
    ///
    /// The weight update function in GPU execution context.
    ///
    /// - Parameter weight: The weight buffers to update.
    ///
    override func stepGPU(_ weights: IWeightBuffers)
    {
        let nbElems = weights.nbElems
        let t = Double(_kernel.params.t)
        
        let pNbElems: [UInt32] = [UInt32(nbElems)]
        let pAlpha: [Float16] = [Float16(alpha)]
        let pLambda: [Float16] = [lambda != nil ? Float16(lambda!) : 0.0]
        let pT: [Float16] = [Float16(t)]
        let pLowerBound: [Float16] = [Float16(lowerBound!)]
        let pUpperBound: [Float16] = [Float16(upperBound!)]
        
        let command = MetalKernel.get.createCommand(
            "weightsAMSBound", deviceID: weights.deviceID
        )
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
        
        command.dispatchThreads(nbElems)
        command.enqueue()
    }
}
