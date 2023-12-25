//
// SelectNeurons2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

///
/// Layer with a 1D shape neural structure.
///
/// This layer transforms a 2D layer into a 1D layer, selecting neurons at a precise position in the input grids
/// into as many output neurons.
///
public class SelectNeurons2D: Layer1D, LayerResize
{
    /// Row of the selected neuron.
    public let targetI: Int
    /// Column of the selected neuron.
    public let targetJ: Int
    /// Height ratio of the selected neuron.
    public let ratioI: Double
    /// Width ratio of the selected neuron.
    public let ratioJ: Double
    
    private enum Keys: String, CodingKey
    {
        case targetI
        case targetJ
        case ratioI
        case ratioJ
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - targetI: Row of the selected neuron.
    ///     - targetJ: Column of the selected neuron.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                targetI: Int,
                targetJ: Int,
                params: GrAI.Model.Params)
    {
        self.targetI = targetI
        self.targetJ = targetJ
        self.ratioI = Double(targetI) / Double(layerPrev.height)
        self.ratioJ = Double(targetJ) / Double(layerPrev.width)
        
        super.init(layerPrev: layerPrev,
                   nbNeurons: layerPrev.nbChannels,
                   params: params)
    }
    
    ///
    /// Create a layer with a 1D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - ratioI: Height ratio of the selected neuron.
    ///     - ratioJ: Width ratio of the selected neuron.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                ratioI: Double,
                ratioJ: Double,
                params: GrAI.Model.Params)
    {
        self.targetI = Int(Double(layerPrev.height) * ratioI)
        self.targetJ = Int(Double(layerPrev.width) * ratioJ)
        self.ratioI = ratioI
        self.ratioJ = ratioJ
        
        super.init(layerPrev: layerPrev,
                   nbNeurons: layerPrev.nbChannels,
                   params: params)
    }
    
    ///
    /// Decode from the disk.
    ///
    /// Throw an error if reading from the decoder fails, or
    /// if the data read is corrupted or otherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
    public required init(from decoder: Decoder) throws
    {
        let values = try decoder.container(keyedBy: Keys.self)
        targetI = try values.decode(Int.self, forKey: Keys.targetI)
        targetJ = try values.decode(Int.self, forKey: Keys.targetJ)
        ratioI = try values.decode(Double.self, forKey: Keys.ratioI)
        ratioJ = try values.decode(Double.self, forKey: Keys.ratioJ)
        try super.init(from: decoder)
    }
    
    ///
    /// Encode to the disk.
    ///
    /// If the value fails to encode anything, `encoder` will encode an empty
    /// keyed container in its place.
    ///
    /// Throw an error if any values are invalid for the given
    /// encoder's format.
    ///
    /// - Parameter encoder: The encoder to write data to.
    ///
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
    /// Create a layer with same values as this.
    ///
    /// - Parameters:
    ///     - mapping: Dictionary allowing to find the layer associated to some id.
    ///     This dictionary is particularly useful when the different layers cannot access
    ///     their `layerPrev`.
    ///     - inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: A new layer. When `inPlace` is false, `initKernel` is
    /// necessary in order to recreate hard resources.
    ///
    public override func copy(
        mapping: Dictionary<Int, Layer>,
        inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let layerPrev = mapping[idPrev] as! Layer2D
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = SelectNeurons2D(
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
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        let layer = SelectNeurons2D(
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
            for depth in 0..<nbNeurons
            {
                neurons.get(depth)!.initGC(batchSize: batchSize, nbGC: nbGC)
            }
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC
            {
                for depth in 0..<nbNeurons
                {
                    let outPrev = neuronsPrev[depth].get(targetI, targetJ)!
                    neurons.get(depth)!.gc[batch][elem].out =
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
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurons
                {
                    let outPrev =
                        neuronsPrev[depth].get(targetI, targetJ)!.v[elem].out
                    neurons.get(depth)!.v[elem].out = outPrev
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
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            
            let command = MetalKernel.get.createEncoder(
                "selectNeurons2DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pTarget, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pDimensionsPrev, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBuffer(outs.metal, atIndex: 5)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: batchSize
            )
            command.endEncoding()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let neuronsPrev = layerPrev.neurons
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            if layerPrev.dirty
            {
                for elem in 0..<batchSize
                {
                    for i in 0..<heightPrev {
                    for j in 0..<widthPrev {
                    for depth in 0..<nbNeurons
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta = 0.0
                    }}}
                }
            }
            
            for elem in 0..<batchSize
            {
                for depth in 0..<nbNeurons
                {
                    let neuronPrev = neuronsPrev[depth].get(targetI, targetJ)!
                    let delta = neurons.get(depth)!.v[elem].delta
                    
                    neuronPrev.v[elem].delta += delta
                }
            }
            propagateDirty()
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
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensionsPrev: [UInt32] = [UInt32(widthPrev),
                                             UInt32(heightPrev)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let command = MetalKernel.get.createEncoder(
                "selectNeurons2DBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pTarget, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pDimensionsPrev, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pDirty, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: widthPrev * nbNeurons,
                height: heightPrev * batchSize
            )
            command.endEncoding()
            
            propagateDirty()
        }
    }
}
