//
// FTFrequences2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 03/12/2022.
//

import Foundation

///
/// Layer with a 2D shape neural structure.
///
/// This layer creates frequences to be used by an Inverse Real Discrete Fourier Transform.
///
public class FTFrequences2D: LayerInput2D, LayerResize
{
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - nbChannels: Number of channels.
    ///     - dimension: Height & width of each channel.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(nbChannels: Int, dimension: Int,
                params: GrAI.Model.Params) throws
    {
        if nbChannels % 2 != 0
        {
            throw LayerError.Init(
                message: "FTFrequences2D input channels " +
                         "should be a multiple of 2."
            )
        }
        super.init(layerPrev: nil,
                   nbChannels: nbChannels,
                   height: dimension,
                   width: dimension,
                   params: params)
        computeDelta = false
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
        try super.init(from: decoder)
        computeDelta = false
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
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        let layer = try! FTFrequences2D(
            nbChannels: nbChannels, dimension: width,
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
    ///     - inPlace: Whether hard resources should be copied as is.
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
        if imageWidth != imageHeight
        {
            fatalError("FTFrequences2D input channels should be a square")
        }
        
        let context = ModelContext(name: "", curID: 0)
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        let layer = try! FTFrequences2D(
            nbChannels: nbChannels, dimension: imageWidth,
            params: params
        )
        return layer
    }
    
    private func _getScaleValue(i: Int, j: Int) -> Double
    {
        var freq: Double =
            sqrt(Double(i * i + j * j)) / Double(width)
        freq = max(freq, 1.0 / Double(width))
        return (1.0 / freq) * Double(width)
    }
    
    ///
    /// API to set data in the CPU execution context.
    ///
    /// Throw an error if batch size is not coherent.
    ///
    /// - Parameter batchSize: The batch size of data.
    ///
    public func setDataCPU(batchSize: Int) throws
    {
        try checkStateCPU(batchSize: batchSize)
        
        for elem in 0..<batchSize {
        for depth in 0..<nbChannels
        {
            let end = width % 2 == 0 ? width / 2 : (width - 1) / 2
            if width % 2 == 0
            {
                var curI = 0
                for i in 0..<end
                {
                    var curJ = 0
                    for j in 0..<end
                    {
                        neurons[depth].get(curI, curJ)!.v[elem].out =
                            _getScaleValue(i: i, j: j)
                        curJ += 1
                    }
                    for j in (1...end).reversed()
                    {
                        neurons[depth].get(curI, curJ)!.v[elem].out =
                            _getScaleValue(i: i, j: j)
                        curJ += 1
                    }
                    curI += 1
                }
                for i in (1...end).reversed()
                {
                    var curJ = 0
                    for j in 0..<end
                    {
                        neurons[depth].get(curI, curJ)!.v[elem].out =
                            _getScaleValue(i: i, j: j)
                        curJ += 1
                    }
                    for j in (1...end).reversed()
                    {
                        neurons[depth].get(curI, curJ)!.v[elem].out =
                            _getScaleValue(i: i, j: j)
                        curJ += 1
                    }
                    curI += 1
                }
            }
            else
            {
                var curI = 0
                for i in 0...end
                {
                    var curJ = 0
                    for j in 0...end
                    {
                        neurons[depth].get(curI, curJ)!.v[elem].out =
                            _getScaleValue(i: i, j: j)
                        curJ += 1
                    }
                    for j in (1...end).reversed()
                    {
                        neurons[depth].get(curI, curJ)!.v[elem].out =
                            _getScaleValue(i: i, j: j)
                        curJ += 1
                    }
                    curI += 1
                }
                for i in (1...end).reversed()
                {
                    var curJ = 0
                    for j in 0...end
                    {
                        neurons[depth].get(curI, curJ)!.v[elem].out =
                            _getScaleValue(i: i, j: j)
                        curJ += 1
                    }
                    for j in (1...end).reversed()
                    {
                        neurons[depth].get(curI, curJ)!.v[elem].out =
                            _getScaleValue(i: i, j: j)
                        curJ += 1
                    }
                    curI += 1
                }
            }
        }}
    }
    
    ///
    /// API to set data in the GPU execution context.
    ///
    /// Throw an error if batch size is not coherent.
    ///
    /// - Parameter batchSize: The batch size of data.
    ///
    public func setDataGPU(batchSize: Int) throws
    {
        try checkStateForwardGPU(batchSize: batchSize)
        
        let pNbChannels: [UInt32] = [UInt32(nbChannels)]
        let pNbBatch: [UInt32] = [UInt32(batchSize)]
        let pDimension: [UInt32] = [UInt32(width)]
        
        let command = MetalKernel.get.createCommand(
            "setDataFTFrequences2D", deviceID: deviceID
        )
        command.setBytes(pNbChannels, atIndex: 0)
        command.setBytes(pDimension, atIndex: 1)
        command.setBytes(pNbBatch, atIndex: 2)
        command.setBuffer(outs.metal, atIndex: 3)
        
        command.dispatchThreads(
            width: width * nbChannels,
            height: height * batchSize
        )
        command.enqueue()
    }
}
