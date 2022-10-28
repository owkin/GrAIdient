//
// OptimizerAlgorithm.swift
// MAKit
//
// Created by Jean-François Reboud on 08/10/2022.
//

import MetalKit

/// Error occuring on a bad request to a `LayerUpdate`.
public enum UpdateError: Error
{
    /// Layer has not been visited by any backward pass.
    case Dirty
}

extension UpdateError: CustomStringConvertible
{
    public var description: String
    {
        switch self
        {
        case .Dirty:
            return "Layer internal state has not been updated " +
                   "through backward pass."
        }
    }
}

/// Algorithm used to update the weights of a `LayerUpdate`.
public class OptimizerAlgorithm
{
    /// Implementation of the weight update algorithm.
    let _optimizer: OptimizerImpl
    
    /// Get alpha (learning rate) value.
    var alpha: Double
    {
        get {
            return _optimizer.alpha
        }
    }
    /// Get alpha (learning rate) percent value.
    var alphaPercent: Double
    {
        get {
            return _optimizer.alphaPercent
        }
    }
    
    /// Get lambda (weight decay) value.
    var lambda: Double?
    {
        get {
            return _optimizer.lambda
        }
    }
    /// Get lambda (weight decay) percent value.
    var lambdaPercent: Double?
    {
        get {
            return _optimizer.lambdaPercent
        }
    }
    
    /// Get lower bound value.
    var lowerBound: Double?
    {
        get {
            return _optimizer.lowerBound
        }
    }
    /// Get lower bound percent value.
    var lowerBoundPercent: Double?
    {
        get {
            return _optimizer.lowerBoundPercent
        }
    }
    
    /// Get upper bound value.
    var upperBound: Double?
    {
        get {
            return _optimizer.upperBound
        }
    }
    /// Get upper bound percent value.
    var upperBoundPercent: Double?
    {
        get {
            return _optimizer.upperBoundPercent
        }
    }
    
    ///
    /// Create a new algorithm thanks to the implementation of the weight update algorithm.
    ///
    /// - Parameter optimizer: Implementation of the weight update algorithm.
    ///
    init(_ optimizer: OptimizerImpl)
    {
        _optimizer = optimizer
    }
    
    ///
    /// Update the weights of the layers conforming to `LayerUpdate` in the CPU execution context.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameters:
    ///     - layers: The list of layers that potentially contain weights to update.
    ///     - gradientNorm: A norm to scale the weights' gradients.
    ///
    func udpateCPU(layers: [Layer], gradientNorm: Double?) throws
    {
        incT()
        
        if _optimizer.params.gradientClipping
        {
            var gNorm: Double = 0.0
            if let gNormTmp = gradientNorm
            {
                gNorm = gNormTmp
            }
            else
            {
                gNorm = try getGradientNormCPU(layers)
            }
            try clipGradientCPU(
                layers: layers,
                gradientNorm: gNorm,
                normThreshold: _optimizer.params.normThreshold
            )
        }
        
        try updateWeightsCPU(layers)
    }
    
    ///
    /// Update the weights of the layers conforming to `LayerUpdate` in the GPU execution context.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameters:
    ///     - layers: The list of layers that potentially contain weights to update.
    ///     - gradientNorm: A norm to scale the weights' gradients.
    ///
    func udpateGPU(layers: [Layer], gradientNorm: Float?) throws
    {
        incT()
        
        if _optimizer.params.gradientClipping
        {
            var gNorm: Float = 0.0
            if let gNormTmp = gradientNorm
            {
                gNorm = gNormTmp
            }
            else
            {
                gNorm = try getGradientNormGPU(layers)
            }
            try clipGradientGPU(
                layers: layers,
                gradientNorm: gNorm,
                normThreshold: _optimizer.params.normThreshold
            )
        }
    
        try updateWeightsGPU(layers)
    }
    
    ///
    /// Multiply the weights' gradients by a scalar in the CPU execution context.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameters:
    ///     - layers: The list of layers to consider.
    ///     - factor: The coefficient to multiply the weights' gradients by.
    ///
    func multiplyGradientCPU(layers: [Layer], factor: Double) throws
    {
        for layer in layers
        {
            let desc = layer.description + String(layer.id)
            MAKit.Time.start()
            
            if let layerUpdate = layer as? LayerUpdate,
                layerUpdate.computeDeltaWeights
            {
                if layerUpdate.dirty
                {
                    throw UpdateError.Dirty
                }
                
                for var arrays in layerUpdate.collectWeightsCPU() {
                for i in 0..<arrays.nbElems
                {
                    arrays.g[i] *= factor
                }}
            }
            
            try MAKit.Time.stop(id: "MultiplyGradientCPU", description: desc)
        }
    }
    
    ///
    /// Multiply the weights' gradients by a scalar in the GPU execution context.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameters:
    ///     - layers: The list of layers to consider.
    ///     - factor: The coefficient to multiply the weights' gradients by.
    ///
    func multiplyGradientGPU(layers: [Layer], factor: Float) throws
    {
        for layer in layers
        {
            let desc = layer.description + String(layer.id)
            MAKit.Time.start()
            
            if let layerUpdate = layer as? LayerUpdate,
                layerUpdate.computeDeltaWeights
            {
                if layerUpdate.dirty
                {
                    throw UpdateError.Dirty
                }
                
                for buffers in layerUpdate.collectWeightsGPU()
                {
                    let nbElems = buffers.g.nbElems
                    
                    let pNbElems: [UInt32] = [UInt32(nbElems)]
                    let pFactor: [Float] = [Float(factor)]
                    
                    let command = MetalKernel.get.createCommand(
                        "multiplyGradients", deviceID: layer.deviceID
                    )
                    command.setBytes(pNbElems, atIndex: 0)
                    command.setBytes(pFactor, atIndex: 1)
                    command.setBuffer(buffers.g.metal, atIndex: 2)
                    
                    let threads = command.threadExecutionWidth
                    let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
                    let threadsPerGrid = MTLSize(width: nbElems,
                                                 height: 1,
                                                 depth: 1)
                    command.dispatchThreads(
                        threadsPerGrid: threadsPerGrid,
                        threadsPerThreadgroup: threadsPerThreadgroup
                    )
                    command.enqueue()
                }
            }
            
            try MAKit.Time.stop(id: "MultiplyGradientGPU", description: desc)
        }
    }
    
    ///
    /// Get gradient norm in the CPU execution context.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameter layers: The list of layers to consider.
    /// - Returns: The gradient norm.
    ///
    func getGradientNormCPU(_ layers: [Layer]) throws -> Double
    {
        var gradientSquaredNorm: Double = 0.0
        for layer in layers
        {
            let desc = layer.description + String(layer.id)
            MAKit.Time.start()
            
            if let layerUpdate = layer as? LayerUpdate,
               layerUpdate.computeDeltaWeights
            {
                if layerUpdate.dirty
                {
                    throw UpdateError.Dirty
                }
                
                for arrays in layerUpdate.collectWeightsCPU() {
                for g in arrays.g
                {
                    gradientSquaredNorm += g * g
                }}
            }
            
            try MAKit.Time.stop(id: "GetGradientNormCPU", description: desc)
        }
        return sqrt(gradientSquaredNorm)
    }
    
    ///
    /// Get gradient norm in the GPU execution context.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameter layers: The list of layers to consider.
    /// - Returns: The gradient norm.
    ///
    func getGradientNormGPU(_ layers: [Layer]) throws -> Float
    {
        var partialGradSum: Float = 0.0
        for layer in layers
        {
            let desc = layer.description + String(layer.id)
            MAKit.Time.start()
            
            if let layerUpdate = layer as? LayerUpdate,
               layerUpdate.computeDeltaWeights
            {
                if layerUpdate.dirty
                {
                    throw UpdateError.Dirty
                }
                
                for buffers in layerUpdate.collectWeightsGPU()
                {
                    let buffer: UnsafeMutableBufferPointer<Float>
                    if let g_p = buffers.g_p
                    {
                        MetalKernel.get.download([g_p])
                        buffer = g_p.shared.buffer
                    }
                    else if let g_s = buffers.g_s
                    {
                        MetalKernel.get.download([g_s])
                        buffer = g_s.buffer
                    }
                    else
                    {
                        fatalError("Unreachable.")
                    }
                    
                    for i in 0..<buffers.g.nbElems
                    {
                        let partialGrad = buffer[i]
                        partialGradSum += partialGrad * partialGrad
                    }
                }
            }
            
            try MAKit.Time.stop(id: "GetGradientNormGPU", description: desc)
        }
        return sqrt(partialGradSum)
    }
    
    ///
    /// Get the weights'  gradients in the CPU execution context.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameter layers: The list of layers to consider.
    /// - Returns: The list of weights' gradients.
    ///
    func getGradientsCPU(_ layers: [Layer]) throws -> [Double]
    {
        var gradients = [Double]()
        for layer in layers
        {
            let desc = layer.description + String(layer.id)
            MAKit.Time.start()
            
            if let layerUpdate = layer as? LayerUpdate,
               layerUpdate.computeDeltaWeights
            {
                if layerUpdate.dirty
                {
                    throw UpdateError.Dirty
                }
                
                for arrays in layerUpdate.collectWeightsCPU() {
                for g in arrays.g
                {
                    gradients.append(g)
                }}
            }
            
            try MAKit.Time.stop(id: "GetGradientsCPU", description: desc)
        }
        return gradients
    }
    
    ///
    /// Get the weights'  gradients in the GPU execution context.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameter layers: The list of layers to consider.
    /// - Returns: The list of weights' gradients.
    ///
    func getGradientsGPU(_ layers: [Layer]) throws -> [Float]
    {
        var gradients = [Float]()
        for layer in layers
        {
            let desc = layer.description + String(layer.id)
            MAKit.Time.start()
            
            if let layerUpdate = layer as? LayerUpdate,
               layerUpdate.computeDeltaWeights
            {
                if layerUpdate.dirty
                {
                    throw UpdateError.Dirty
                }
                
                for buffers in layerUpdate.collectWeightsGPU()
                {
                    let buffer: UnsafeMutableBufferPointer<Float>
                    if let g_p = buffers.g_p
                    {
                        MetalKernel.get.download([g_p])
                        buffer = g_p.shared.buffer
                    }
                    else if let g_s = buffers.g_s
                    {
                        MetalKernel.get.download([g_s])
                        buffer = g_s.buffer
                    }
                    else
                    {
                        fatalError("Unreachable.")
                    }
                    
                    for i in 0..<buffers.g.nbElems
                    {
                        gradients.append(buffer[i])
                    }
                }
            }
            
            try MAKit.Time.stop(id: "GetGradientsGPU", description: desc)
        }
        return gradients
    }
    
    /// Increment internal time state.
    func incT()
    {
        _optimizer.incT()
    }
    
    /// Increment internal step state.
    func incStep()
    {
        _optimizer.incStep()
    }
    
    ///
    /// Clip the weights'  gradients in the CPU execution context.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameters:
    ///     - layers: The list of layers to consider.
    ///     - gradientNorm: The norm of the weights' gradients.
    ///     - normThreshold: The threshold above which we must clip the weights' gradients.
    ///
    func clipGradientCPU(layers: [Layer],
                         gradientNorm: Double,
                         normThreshold: Double) throws
    {
        if gradientNorm > normThreshold {
        for layer in layers
        {
            let desc = layer.description + String(layer.id)
            MAKit.Time.start()
            
            if let layerUpdate = layer as? LayerUpdate,
               layerUpdate.computeDeltaWeights
            {
                if layerUpdate.dirty
                {
                    throw UpdateError.Dirty
                }
                
                for var arrays in layerUpdate.collectWeightsCPU() {
                for i in 0..<arrays.nbElems
                {
                    arrays.g[i] *= normThreshold / gradientNorm
                }}
            }
            
            try MAKit.Time.stop(id: "ClipGradientCPU", description: desc)
        }}
    }
    
    ///
    /// Clip the weights'  gradients in the GPU execution context.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameters:
    ///     - layers: The list of layers to consider.
    ///     - gradientNorm: The norm of the weights' gradients.
    ///     - normThreshold: The threshold above which we must clip the weights' gradients.
    ///
    func clipGradientGPU(layers: [Layer],
                         gradientNorm: Float,
                         normThreshold: Double) throws
    {
        if gradientNorm > Float(normThreshold) {
        for layer in layers
        {
            let desc = layer.description + String(layer.id)
            MAKit.Time.start()
            
            if let layerUpdate = layer as? LayerUpdate,
               layerUpdate.computeDeltaWeights
            {
                if layerUpdate.dirty
                {
                    throw UpdateError.Dirty
                }
                
                for buffers in layerUpdate.collectWeightsGPU()
                {
                    let nbElems = buffers.g.nbElems
                    
                    let pNbElems: [UInt32] = [UInt32(nbElems)]
                    let pGradientNorm: [Float] = [Float(gradientNorm)]
                    let pNormThreshold: [Float] = [Float(normThreshold)]
                    
                    let command = MetalKernel.get.createCommand(
                        "clipGradients", deviceID: layer.deviceID
                    )
                    command.setBytes(pNbElems, atIndex: 0)
                    command.setBytes(pGradientNorm, atIndex: 1)
                    command.setBytes(pNormThreshold, atIndex: 2)
                    command.setBuffer(buffers.g.metal, atIndex: 3)
                    
                    let threads = command.threadExecutionWidth
                    let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
                    let threadsPerGrid = MTLSize(width: nbElems,
                                                 height: 1,
                                                 depth: 1)
                    command.dispatchThreads(
                        threadsPerGrid: threadsPerGrid,
                        threadsPerThreadgroup: threadsPerThreadgroup
                    )
                    command.enqueue()
                }
            }
            
            try MAKit.Time.stop(id: "ClipGradientGPU", description: desc)
        }}
    }
    
    ///
    /// Update the weights of the layers conforming to `LayerUpdate` in the CPU execution context.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameter layers: The list of layers that potentially contain weights to update
    ///
    func updateWeightsCPU(_ layers: [Layer]) throws
    {
        for layer in layers
        {
            let desc = layer.description + String(layer.id)
            MAKit.Time.start()
            
            if let layerUpdate = layer as? LayerUpdate,
               layerUpdate.computeDeltaWeights
            {
                if layerUpdate.dirty
                {
                    throw UpdateError.Dirty
                }
                
                stepCPU(layerUpdate.collectWeightsCPU())
            }
            
            try MAKit.Time.stop(id: "UpdateWeightsCPU", description: desc)
        }
    }
    
    ///
    /// Update the weights of the layers conforming to `LayerUpdate` in the GPU execution context.
    ///
    /// Throw an error when layers have not been visited by any backward pass.
    ///
    /// - Parameter layers: The list of layers that potentially contain weights to update
    ///
    func updateWeightsGPU(_ layers: [Layer]) throws
    {
        for layer in layers
        {
            let desc = layer.description + String(layer.id)
            MAKit.Time.start()
            
            if let layerUpdate = layer as? LayerUpdate,
               layerUpdate.computeDeltaWeights
            {
                if layerUpdate.dirty
                {
                    throw UpdateError.Dirty
                }
                
                stepGPU(layerUpdate.collectWeightsGPU())
            }
            
            try MAKit.Time.stop(id: "UpdateWeightsGPU", description: desc)
        }
    }
    
    ///
    /// Update the weights  in the CPU execution context.
    ///
    /// - Parameter weights: The list of weight arrays to update.
    ///
    func stepCPU(_ weights: [IWeightArrays])
    {
        for weightsTmp in weights
        {
            _optimizer.stepCPU(weightsTmp)
        }
    }
    
    ///
    /// Update the weights  in the GPU execution context.
    ///
    /// - Parameter weights: The list of weight buffers to update.
    ///
    func stepGPU(_ weights: [IWeightBuffers])
    {
        for weightsTmp in weights
        {
            _optimizer.stepGPU(weightsTmp)
        }
    }
}
