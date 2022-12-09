//
// ResizeBilinearJitter.swift
// MAKit
//
// Created by Jean-François Reboud on 09/12/2022.
//

import Foundation

///
/// Layer with a 2D shape neural structure.
///
/// Bilnear resize input grids.
///
/// Note that this layer enforces deterministic dimensions for the output grids.
/// The final dimensions being the maximal scale of the dimensions of the input grids.
/// For intermediate scales, padding is used in order to complete the missing values.
///
public class ResizeBilinearJitter: Layer2D
{
    let _scalesList: [Double]
    
    var _offsetI: Int = 0
    var _offsetJ: Int = 0
    var _doNotRandom: Bool = false
    
    var _widthResize: Int = 0
    var _heightResize: Int = 0
    
    private enum Keys: String, CodingKey
    {
        case scalesList
        case jitterDimension
        case doNotRandom
        case offsetI
        case offsetJ
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - scalesList: List of scales to apply to (heightPrev, widthPrev) dimensions.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                scalesList: [Double],
                params: MAKit.Model.Params)
    {
        _scalesList = scalesList
        
        if scalesList.count == 0
        {
            fatalError("`scalesList` should have at least one element.")
        }
        for scale in scalesList
        {
            if scale == 0
            {
                fatalError("Only non 0 scales are possible.")
            }
        }
        
        let nbChannels = layerPrev.nbChannels
        let heightPrev = layerPrev.height
        let widthPrev = layerPrev.width
        
        var width: Int = 10000
        var height: Int = 10000
        for scale in scalesList
        {
            width = min(width, Int(round(scale * Double(widthPrev))))
            height = min(height, Int(round(scale * Double(heightPrev))))
        }
        
        super.init(layerPrev: layerPrev,
                   nbChannels: nbChannels,
                   height: height,
                   width: width,
                   params: params)
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - scalesList: List of scales to apply to (heightPrev, widthPrev) dimensions.
    ///     - offsetI: Height offset.
    ///     - offsetJ: Width offset.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                scalesList: [Double],
                offsetI: Int,
                offsetJ: Int,
                params: MAKit.Model.Params)
    {
        _scalesList = scalesList
        _doNotRandom = true
        _offsetI = offsetI
        _offsetJ = offsetJ
        
        if scalesList.count == 0
        {
            fatalError("`scalesList` should have at least one element.")
        }
        for scale in scalesList
        {
            if scale == 0
            {
                fatalError("Only non 0 scales are possible.")
            }
        }
        
        let nbChannels = layerPrev.nbChannels
        let heightPrev = layerPrev.height
        let widthPrev = layerPrev.width
        
        var width: Int = 10000
        var height: Int = 10000
        for scale in scalesList
        {
            width = min(width, Int(round(scale * Double(widthPrev))))
            height = min(height, Int(round(scale * Double(heightPrev))))
        }
        
        super.init(layerPrev: layerPrev,
                   nbChannels: nbChannels,
                   height: height,
                   width: width,
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
        _scalesList = try values.decode(
            [Double].self, forKey: Keys.scalesList
        )
        _doNotRandom = try values.decode(Bool.self, forKey: Keys.doNotRandom)
        _offsetI = try values.decode(Int.self, forKey: Keys.offsetI)
        _offsetJ = try values.decode(Int.self, forKey: Keys.offsetJ)
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
        try container.encode(_scalesList, forKey: Keys.scalesList)
        try container.encode(_doNotRandom, forKey: Keys.doNotRandom)
        try container.encode(_offsetI, forKey: Keys.offsetI)
        try container.encode(_offsetJ, forKey: Keys.offsetJ)
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
        
        let params = MAKit.Model.Params(context: context)
        params.context.curID = id
            
        let layer: ResizeBilinearJitter
        if !_doNotRandom
        {
            layer = ResizeBilinearJitter(
                layerPrev: layerPrev,
                scalesList: _scalesList,
                params: params
            )
        }
        else
        {
            layer = ResizeBilinearJitter(
                layerPrev: layerPrev,
                scalesList: _scalesList,
                offsetI: _offsetI,
                offsetJ: _offsetJ,
                params: params
            )
        }
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
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    neurons[depth].get(i, j)!.initGC(
                        batchSize: batchSize,
                        nbGC: nbGC
                    )
                }}
            }
        
            let ratioInOutI = Double(_heightResize - 1) / Double(height - 1)
            let ratioInOutJ = Double(_widthResize - 1) / Double(width - 1)
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let iPrev = Double(i) * ratioInOutI
                    let jPrev = Double(j) * ratioInOutJ
                    
                    let iPrevInf = Int(floor(iPrev))
                    let iPrevSup = Int(ceil(iPrev))
                    let jPrevInf = Int(floor(jPrev))
                    let jPrevSup = Int(ceil(jPrev))
                    
                    let iWeight = ratioInOutI * Double(i) - Double(iPrevInf)
                    let jWeight = ratioInOutJ * Double(j) - Double(jPrevInf)
                    
                    let outPrev11 = neuronsPrev[depth].get(
                        iPrevInf+_offsetI, jPrevInf+_offsetJ
                    )!.gc[batch][elem].out
                    let outPrev12 = neuronsPrev[depth].get(
                        iPrevInf+_offsetI, jPrevSup+_offsetJ
                    )!.gc[batch][elem].out
                    let outPrev21 = neuronsPrev[depth].get(
                        iPrevSup+_offsetI, jPrevInf+_offsetJ
                    )!.gc[batch][elem].out
                    let outPrev22 = neuronsPrev[depth].get(
                        iPrevSup+_offsetI, jPrevSup+_offsetJ
                    )!.gc[batch][elem].out
                    
                    var out = outPrev11 * (1.0 - jWeight) * (1.0 - iWeight)
                    out += outPrev12 * jWeight * (1.0 - iWeight)
                    out += outPrev21 * (1.0 - jWeight) * iWeight
                    out += outPrev22 * jWeight * iWeight
                    
                    neurons[depth].get(i, j)!.gc[batch][elem].out = out
                }}
            }}}
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
            
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            _heightResize = height
            _widthResize = width
            
            if _scalesList.count > 1
            {
                let randIndex = Int.random(in: 0..<_scalesList.count)
                let ratioInOut = _scalesList[randIndex]
                
                _widthResize = Int(round(Double(width) / ratioInOut))
                _heightResize = Int(round(Double(height) / ratioInOut))
            }
            
            let ratioInOutI = Double(_heightResize - 1) / Double(height - 1)
            let ratioInOutJ = Double(_widthResize - 1) / Double(width - 1)
            let jitterDimensionI = heightPrev - _heightResize
            let jitterDimensionJ = widthPrev - _heightResize
            
            if !_doNotRandom
            {
                if jitterDimensionI == 0
                {
                    _offsetI = 0
                }
                else
                {
                    _offsetI = Int.random(in: 0..<jitterDimensionI)
                }
                if jitterDimensionJ == 0
                {
                    _offsetJ = 0
                }
                else
                {
                    _offsetJ = Int.random(in: 0..<jitterDimensionJ)
                }
            }
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let iPrev = Double(i) * ratioInOutI
                    let jPrev = Double(j) * ratioInOutJ
                    
                    let iPrevInf = Int(floor(iPrev))
                    let iPrevSup = Int(ceil(iPrev))
                    let jPrevInf = Int(floor(jPrev))
                    let jPrevSup = Int(ceil(jPrev))
                    
                    let iWeight = ratioInOutI * Double(i) - Double(iPrevInf)
                    let jWeight = ratioInOutJ * Double(j) - Double(jPrevInf)
                    
                    let outPrev11 = neuronsPrev[depth].get(
                        iPrevInf+_offsetI, jPrevInf+_offsetJ)!.v[elem].out
                    let outPrev12 = neuronsPrev[depth].get(
                        iPrevInf+_offsetI, jPrevSup+_offsetJ)!.v[elem].out
                    let outPrev21 = neuronsPrev[depth].get(
                        iPrevSup+_offsetI, jPrevInf+_offsetJ)!.v[elem].out
                    let outPrev22 = neuronsPrev[depth].get(
                        iPrevSup+_offsetI, jPrevSup+_offsetJ)!.v[elem].out
                    
                    var out = outPrev11 * (1.0 - iWeight) * (1.0 - jWeight)
                    out += outPrev12 * (1.0 - iWeight) * jWeight
                    out += outPrev21 * iWeight * (1.0 - jWeight)
                    out += outPrev22 * iWeight * jWeight
                    
                    neurons[depth].get(i, j)!.v[elem].out = out
                }}
            }}
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
            
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            _heightResize = height
            _widthResize = width
            
            if _scalesList.count > 1
            {
                let randIndex = Int.random(in: 0..<_scalesList.count)
                let ratioInOut = _scalesList[randIndex]
                
                _widthResize = Int(round(Double(width) / ratioInOut))
                _heightResize = Int(round(Double(height) / ratioInOut))
            }
            
            let jitterDimensionI = heightPrev - _heightResize
            let jitterDimensionJ = widthPrev - _heightResize
            
            if !_doNotRandom
            {
                if jitterDimensionI == 0
                {
                    _offsetI = 0
                }
                else
                {
                    _offsetI = Int.random(in: 0..<jitterDimensionI)
                }
                if jitterDimensionJ == 0
                {
                    _offsetJ = 0
                }
                else
                {
                    _offsetJ = Int.random(in: 0..<jitterDimensionJ)
                }
            }
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pDimensionsPrev: [UInt32] = [
                UInt32(widthPrev), UInt32(heightPrev)
            ]
            let pDimensionsResize: [UInt32] = [
                UInt32(_widthResize), UInt32(_heightResize)
            ]
            let pCropOffsets: [UInt32] = [UInt32(_offsetJ), UInt32(_offsetI)]
            
            let command = MetalKernel.get.createCommand(
                "resizeBilinearCropForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbChannels, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pDimensionsPrev, atIndex: 3)
            command.setBytes(pDimensionsResize, atIndex: 4)
            command.setBytes(pCropOffsets, atIndex: 5)
            command.setBytes(pNbBatch, atIndex: 6)
            command.setBuffer(outs.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: width * nbChannels,
                height: height * batchSize
            )
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        if let layerPrev = self.layerPrev as? Layer2D, mustComputeBackward
        {
            let widthPrev = layerPrev.width
            let heightPrev = layerPrev.height
            let neuronsPrev = layerPrev.neurons
            
            if layerPrev.dirty
            {
                for elem in 0..<batchSize {
                for depth in 0..<nbChannels
                {
                    for i in 0..<heightPrev {
                    for j in 0..<widthPrev
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta = 0.0
                    }}
                }}
            }
            
            let ratioInOutI = Double(_heightResize - 1) / Double(height - 1)
            let ratioInOutJ = Double(_widthResize - 1) / Double(width - 1)
            
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let iPrev = Double(i) * ratioInOutI
                    let jPrev = Double(j) * ratioInOutJ
                    
                    let iPrevInf = Int(floor(iPrev))
                    let iPrevSup = Int(ceil(iPrev))
                    let jPrevInf = Int(floor(jPrev))
                    let jPrevSup = Int(ceil(jPrev))
                    
                    let iWeight = ratioInOutI * Double(i) - Double(iPrevInf)
                    let jWeight = ratioInOutJ * Double(j) - Double(jPrevInf)
                    
                    let delta = neurons[depth].get(i, j)!.v[elem].delta
                    
                    neuronsPrev[depth].get(
                        iPrevInf+_offsetI, jPrevInf+_offsetJ
                    )!.v[elem].delta += delta * (1.0-iWeight) * (1.0-jWeight)
                    neuronsPrev[depth].get(
                        iPrevInf+_offsetI, jPrevSup+_offsetJ
                    )!.v[elem].delta += delta * (1.0 - iWeight) * jWeight
                    neuronsPrev[depth].get(
                        iPrevSup+_offsetI, jPrevInf+_offsetJ
                    )!.v[elem].delta += delta * iWeight * (1.0 - jWeight)
                    neuronsPrev[depth].get(
                        iPrevSup+_offsetI, jPrevSup+_offsetJ
                    )!.v[elem].delta += delta * iWeight * jWeight
                }}
            }}
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
            
            var command: MetalCommand
            if layerPrev.dirty
            {
                let nbElems = layerPrev.delta.nbElems
                let pNbElems: [UInt32] = [UInt32(nbElems)]
                
                command = MetalKernel.get.createCommand(
                    "reset", deviceID: deviceID
                )
                command.setBytes(pNbElems, atIndex: 0)
                command.setBuffer(layerPrev.delta.metal, atIndex: 1)
                
                command.dispatchThreads(nbElems)
                command.enqueue()
            }
            
            let widthPrev = layerPrev.width
            let heightPrev = layerPrev.height
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pDimensionsPrev: [UInt32] = [
                UInt32(widthPrev), UInt32(heightPrev)
            ]
            let pDimensionsResize: [UInt32] = [
                UInt32(_widthResize), UInt32(_heightResize)
            ]
            let pCropOffsets: [UInt32] = [UInt32(_offsetJ), UInt32(_offsetI)]
            
            command = MetalKernel.get.createCommand(
                "resizeBilinearCropBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbChannels, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pDimensionsPrev, atIndex: 3)
            command.setBytes(pDimensionsResize, atIndex: 4)
            command.setBytes(pCropOffsets, atIndex: 5)
            command.setBytes(pNbBatch, atIndex: 6)
            command.setBuffer(layerPrev.delta.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: widthPrev * nbChannels,
                height: heightPrev * batchSize
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
