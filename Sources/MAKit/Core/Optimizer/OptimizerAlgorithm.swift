//
// OptimizerAlgorithm.swift
// MAKit
//
// Created by Jean-François Reboud on 08/10/2022.
//

import MetalKit

public enum UpdateError: Error
{
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

public class OptimizerAlgorithm
{
    let _optimizer: OptimizerImpl
    
    var alpha: Double
    {
        get {
            return _optimizer.alpha
        }
    }
    var alphaPercent: Double
    {
        get {
            return _optimizer.alphaPercent
        }
    }
    
    var lambda: Double?
    {
        get {
            return _optimizer.lambda
        }
    }
    var lambdaPercent: Double?
    {
        get {
            return _optimizer.lambdaPercent
        }
    }
    
    var lowerBound: Double?
    {
        get {
            return _optimizer.lowerBound
        }
    }
    var lowerBoundPercent: Double?
    {
        get {
            return _optimizer.lowerBoundPercent
        }
    }
    
    var upperBound: Double?
    {
        get {
            return _optimizer.upperBound
        }
    }
    var upperBoundPercent: Double?
    {
        get {
            return _optimizer.upperBoundPercent
        }
    }
    
    init(_ optimizer: OptimizerImpl)
    {
        _optimizer = optimizer
    }
    
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
                        "multiplyGradients", deviceID: layer.deviceID)
                    
                    command.setBytes(pNbElems, atIndex: 0)
                    command.setBytes(pFactor, atIndex: 1)
                    command.setBuffer(buffers.g.metal, atIndex: 2)
                    
                    let threads = command.threadExecutionWidth
                    let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
                    let threadsPerGrid = MTLSize(width: nbElems,
                                                 height: 1,
                                                 depth: 1)
                    command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
                    command.enqueue()
                }
            }
            
            try MAKit.Time.stop(id: "MultiplyGradientGPU", description: desc)
        }
    }
    
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
    
    func incT()
    {
        _optimizer.incT()
    }
    
    func incStep()
    {
        _optimizer.incStep()
    }
    
    func incEpoch()
    {
        _optimizer.incEpoch()
    }
    
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
                        "clipGradients", deviceID: layer.deviceID)
                    
                    command.setBytes(pNbElems, atIndex: 0)
                    command.setBytes(pGradientNorm, atIndex: 1)
                    command.setBytes(pNormThreshold, atIndex: 2)
                    command.setBuffer(buffers.g.metal, atIndex: 3)
                    
                    let threads = command.threadExecutionWidth
                    let threadsPerThreadgroup = MTLSizeMake(threads, 1, 1)
                    let threadsPerGrid = MTLSize(width: nbElems,
                                                 height: 1,
                                                 depth: 1)
                    command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                                threadsPerThreadgroup: threadsPerThreadgroup)
                    command.enqueue()
                }
            }
            
            try MAKit.Time.stop(id: "ClipGradientGPU", description: desc)
        }}
    }
    
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
    
    func stepCPU(_ weights: [IWeightArrays])
    {
        for weightsTmp in weights
        {
            _optimizer.stepCPU(weightsTmp)
        }
    }
    
    func stepGPU(_ weights: [IWeightBuffers])
    {
        for weightsTmp in weights
        {
            _optimizer.stepGPU(weightsTmp)
        }
    }
}
