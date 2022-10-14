//
// SelectNeurone.swift
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

import MetalKit

public class SelectNeurone: Layer1D, LayerResize
{
    public let targetI: Int
    public let targetJ: Int
    public let ratioI: Double
    public let ratioJ: Double
    
    private enum Keys: String, CodingKey
    {
        case targetI
        case targetJ
        case ratioI
        case ratioJ
    }
    
    public init(layerPrev: Layer2D,
                targetI: Int,
                targetJ: Int,
                params: MAKit.Model.Params)
    {
        self.targetI = targetI
        self.targetJ = targetJ
        self.ratioI = Double(targetI) / Double(layerPrev.height)
        self.ratioJ = Double(targetJ) / Double(layerPrev.width)
        
        super.init(layerPrev: layerPrev,
                   nbNeurones: layerPrev.nbFilters,
                   params: params)
    }
    
    public init(layerPrev: Layer2D,
                ratioI: Double,
                ratioJ: Double,
                params: MAKit.Model.Params)
    {
        self.targetI = Int(Double(layerPrev.height) * ratioI)
        self.targetJ = Int(Double(layerPrev.width) * ratioJ)
        self.ratioI = ratioI
        self.ratioJ = ratioJ
        
        super.init(layerPrev: layerPrev,
                   nbNeurones: layerPrev.nbFilters,
                   params: params)
    }
    
    public required init(from decoder: Decoder) throws
    {
        let values = try decoder.container(keyedBy: Keys.self)
        targetI = try values.decode(Int.self, forKey: Keys.targetI)
        targetJ = try values.decode(Int.self, forKey: Keys.targetJ)
        ratioI = try values.decode(Double.self, forKey: Keys.ratioI)
        ratioJ = try values.decode(Double.self, forKey: Keys.ratioJ)
        try super.init(from: decoder)
    }
    
    public override func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        try container.encode(targetI, forKey: Keys.targetI)
        try container.encode(targetJ, forKey: Keys.targetJ)
        try container.encode(ratioI, forKey: Keys.ratioI)
        try container.encode(ratioJ, forKey: Keys.ratioJ)
        try super.encode(to: encoder)
    }
    
    ///
    /// Create a new instance of `Layer` with same values as this.
    ///
    /// - Parameters:
    ///     - mapping: Dictionary allowing to find the layer associated to some id.
    ///     This dictionary is particularly useful when the different layers cannot access
    ///     their `layerPrev`.
    ///     - inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: A new instance of `Layer`. When `inPlace` is false, `initKernel` is
    /// necessary in order to recreate hard resources.
    ///
    public override func copy(
        mapping: Dictionary<Int, Layer>,
        inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let layerPrev = mapping[idPrev] as! Layer2D
        
        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
            
        let layer = SelectNeurone(
            layerPrev: layerPrev,
            targetI: targetI,
            targetJ: targetJ,
            params: params
        )
        return layer
    }
    
    ///
    /// Resize this layer.
    ///
    /// - Parameters:
    ///     - imageWidth: New size width.
    ///     - imageHeight: New size height.
    ///     - mapping: Dictionary allowing to find the layer associated to some id.
    ///     This dictionary is particularly useful when the different layers cannot access
    ///     their `layerPrev`.
    ///
    /// - Returns: A new instance of `Layer`. When `inPlace` is false, `initKernel` is
    ///  necessary in order to recreate hard resources.
    ///
    public func resize(
        imageWidth: Int,
        imageHeight: Int,
        mapping: Dictionary<Int, Layer>,
        inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let layerPrev = mapping[idPrev] as! Layer2D
        
        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
        
        let layer = SelectNeurone(
            layerPrev: layerPrev,
            ratioI: self.ratioI,
            ratioJ: self.ratioJ,
            params: params
        )
        return layer
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            for depth in 0..<nbNeurones
            {
                neurones.get(depth)!.initGC(batchSize: batchSize,
                                            nbGC: nbGC)
            }
            
            let neuronesPrev = layerPrev.neurones
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbNeurones
                {
                    let outPrev = neuronesPrev[depth].get(targetI, targetJ)!
                    neurones.get(depth)!.gc[batch][elem].out =
                        outPrev.gc[batch][elem].out
                }
            }}
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try forwardGCCPU()
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronesPrev = layerPrev.neurones
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurones
                {
                    let outPrev =
                        neuronesPrev[depth].get(targetI, targetJ)!.v[elem].out
                    neurones.get(depth)!.v[elem].out = outPrev
                }
            }
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let widthPrev = layerPrev.width
            let heightPrev = layerPrev.height
            
            let pTarget: [UInt32] = [UInt32(targetI), UInt32(targetJ)]
            let pNbNeurones: [UInt32] = [UInt32(nbNeurones)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            
            let command = MetalKernel.get.createCommand("selectNeuroneForward",
                                                        deviceID: deviceID)
            
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pTarget, atIndex: 1)
            command.setBytes(pNbNeurones, atIndex: 2)
            command.setBytes(pDimensionsPrev, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBuffer(outs.metal, atIndex: 5)
            
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
            
            if layerPrev.dirty
            {
                for elem in 0..<batchSize
                {
                    for i in 0..<heightPrev {
                    for j in 0..<widthPrev {
                    for depth in 0..<nbNeurones
                    {
                        neuronesPrev[depth].get(i, j)!.v[elem].delta = 0.0
                    }}}
                }
            }
            
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurones
                {
                    let neuronePrev = neuronesPrev[depth].get(targetI, targetJ)!
                    let delta = neurones.get(depth)!.v[elem].delta
                    
                    neuronePrev.v[elem].delta += delta
                }
            }
        }
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            let pTarget: [UInt32] = [UInt32(targetI), UInt32(targetJ)]
            let pNbNeurones: [UInt32] = [UInt32(nbNeurones)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createCommand(
                "selectNeuroneBackward", deviceID: deviceID)
            
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pTarget, atIndex: 1)
            command.setBytes(pNbNeurones, atIndex: 2)
            command.setBytes(pDimensionsPrev, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pDirty, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal, atIndex: 6)
            
            let threadsPerThreadgroup = MTLSizeMake(8, 8, 8)
            let threadsPerGrid = MTLSize(width: widthPrev,
                                         height: heightPrev,
                                         depth: nbNeurones * batchSize)
            command.dispatchThreads(threadsPerGrid: threadsPerGrid,
                               threadsPerThreadgroup: threadsPerThreadgroup)
            command.enqueue()
        }
    }
}
