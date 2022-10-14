//
// AvgPool2D.swift
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

import MetalKit

public class AvgPool2D: Layer1D
{
    public init(layerPrev: Layer2D, params: MAKit.Model.Params)
    {
        super.init(layerPrev: layerPrev,
                   nbNeurones: layerPrev.nbFilters,
                   params: params)
    }
    
    public required init(from decoder: Decoder) throws
    {
        try super.init(from: decoder)
    }
    
    public override func copy(
        mapping: Dictionary<Int, Layer>,
        inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let layerPrev = mapping[idPrev] as! Layer2D
        
        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
            
        let layer = AvgPool2D(layerPrev: layerPrev, params: params)
        return layer
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for depth in 0..<nbNeurones
            {
                neurones.get(depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
            }
            
            let neuronesPrev = layerPrev.neurones
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbNeurones
                {
                    var mean = 0.0
            
                    for i in 0..<heightPrev {
                    for j in 0..<widthPrev
                    {
                        mean += neuronesPrev[depth].get(i, j)!
                            .gc[batch][elem].out
                    }}
                    
                    mean /= Double(heightPrev * widthPrev)
                    neurones.get(depth)!.gc[batch][elem].out = mean
                }
            }}
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try forwardGCCPU()
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronesPrev = layerPrev.neurones
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurones
                {
                    var mean = 0.0
                    for i in 0..<heightPrev {
                    for j in 0..<widthPrev
                    {
                        mean += neuronesPrev[depth].get(i, j)!.v[elem].out
                    }}
                    
                    mean /= Double(heightPrev * widthPrev)
                    neurones.get(depth)!.v[elem].out = mean
                }
            }
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            let pNbNeurones: [UInt32] = [UInt32(nbNeurones)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            
            let command = MetalKernel.get.createCommand(
                "avgPoolForward", deviceID: deviceID)
            
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbNeurones, atIndex: 1)
            command.setBytes(pDimensionsPrev, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBuffer(outs.metal, atIndex: 4)
            
            let threadsPerThreadgroup = MTLSizeMake(8, 8, 1)
            let threadsPerGrid = MTLSize(width: nbNeurones,
                                         height: batchSize,
                                         depth: 1)
            command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                               threadsPerThreadgroup: threadsPerThreadgroup)
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronesPrev = layerPrev.neurones
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurones
                {
                    let deltaCur = neurones.get(depth)!.v[elem].delta
                    
                    for i in 0..<heightPrev {
                    for j in 0..<widthPrev
                    {
                        if layerPrev.dirty
                        {
                            neuronesPrev[depth].get(i, j)!.v[elem].delta =
                                deltaCur / Double(heightPrev * widthPrev)
                        }
                        else
                        {
                            neuronesPrev[depth].get(i, j)!.v[elem].delta +=
                                deltaCur / Double(heightPrev * widthPrev)
                        }
                    }}
                }
            }
            propagateDirty()
        }
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throws an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            let pNbNeurones: [UInt32] = [UInt32(nbNeurones)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "avgPoolBackward", deviceID: deviceID)
            
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbNeurones, atIndex: 1)
            command.setBytes(pDimensionsPrev, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBytes(pDirty, atIndex: 4)
            command.setBuffer(layerPrev.delta.metal, atIndex: 5)
            
            let threadsPerThreadgroup = MTLSizeMake(8, 8, 8)
            let threadsPerGrid = MTLSize(width: widthPrev,
                                         height: heightPrev,
                                         depth: nbNeurones * batchSize)
            command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                               threadsPerThreadgroup: threadsPerThreadgroup)
            command.enqueue()
            
            propagateDirty()
        }
    }
}
