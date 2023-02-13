//
// ResizeBilinearPad.swift
// GrAIdient
//
// Created by Jean-François Reboud on 07/12/2022.
//

import Foundation

///
/// Layer with a 2D shape neural structure.
///
/// Bilinear resize input grids then potentially pad missing values.
///
/// Note that this layer enforces deterministic dimensions for the output grids.
/// The final dimensions being the maximal scale of the dimensions of the input grids.
///
/// When one unique scale is used, no padding will be used and the final dimensions
/// will correspond to the scale of the dimensions of the input grids.
///
public class ResizeBilinearPad: Layer2D
{
    let _scalesList: [Double]
    let _padValue: Double
    
    var _widthResize: Int = 0
    var _heightResize: Int = 0
    
    private enum Keys: String, CodingKey
    {
        case scalesList
        case padValue
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - scalesList: List of scales to apply to (heightPrev, widthPrev) dimensions.
    ///     - padValue: Value to set on the created borders.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                scalesList: [Double],
                padValue: Double,
                params: GrAI.Model.Params)
    {
        _padValue = padValue
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
        
        var width: Int = 0
        var height: Int = 0
        for scale in scalesList
        {
            width = max(width, Int(round(scale * Double(widthPrev))))
            height = max(height, Int(round(scale * Double(heightPrev))))
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
        _padValue = try values.decode(
            Double.self, forKey: Keys.padValue
        )
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
        try container.encode(_padValue, forKey: Keys.padValue)
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
            
        let layer = ResizeBilinearPad(
            layerPrev: layerPrev,
            scalesList: _scalesList,
            padValue: _padValue,
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
            
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            let ratioInOutI = Double(heightPrev - 1) / Double(_heightResize - 1)
            let ratioInOutJ = Double(widthPrev - 1) / Double(_widthResize - 1)
            let padDimensionI = (height - _heightResize) / 2
            let padDimensionJ = (width - _widthResize) / 2
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    if i < padDimensionI || i >= height - padDimensionI ||
                       j < padDimensionJ || j >= width - padDimensionJ
                    {
                        neurons[depth].get(i, j)!.gc[batch][elem].out =
                            _padValue
                    }
                    else
                    {
                        let I = i-padDimensionI
                        let J = j-padDimensionJ
                        
                        let iPrev = Double(I) * ratioInOutI
                        let jPrev = Double(J) * ratioInOutJ
                        
                        let iPrevInf = Int(floor(iPrev))
                        let iPrevSup = Int(ceil(iPrev))
                        let jPrevInf = Int(floor(jPrev))
                        let jPrevSup = Int(ceil(jPrev))
                        
                        let iWeight = ratioInOutI * Double(I) - Double(iPrevInf)
                        let jWeight = ratioInOutJ * Double(J) - Double(jPrevInf)
                        
                        let outPrev11 = neuronsPrev[depth].get(
                            iPrevInf, jPrevInf)!.gc[batch][elem].out
                        let outPrev12 = neuronsPrev[depth].get(
                            iPrevInf, jPrevSup)!.gc[batch][elem].out
                        let outPrev21 = neuronsPrev[depth].get(
                            iPrevSup, jPrevInf)!.gc[batch][elem].out
                        let outPrev22 = neuronsPrev[depth].get(
                            iPrevSup, jPrevSup)!.gc[batch][elem].out
                        
                        var out = outPrev11 * (1.0 - jWeight) * (1.0 - iWeight)
                        out += outPrev12 * jWeight * (1.0 - iWeight)
                        out += outPrev21 * (1.0 - jWeight) * iWeight
                        out += outPrev22 * jWeight * iWeight
                        
                        neurons[depth].get(i, j)!.gc[batch][elem].out = out
                    }
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
                
                _widthResize = Int(round(ratioInOut * Double(widthPrev)))
                _heightResize = Int(round(ratioInOut * Double(heightPrev)))
            }
            
            let ratioInOutI = Double(heightPrev - 1) / Double(_heightResize - 1)
            let ratioInOutJ = Double(widthPrev - 1) / Double(_widthResize - 1)
            let padDimensionI = (height - _heightResize) / 2
            let padDimensionJ = (width - _widthResize) / 2
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    if i < padDimensionI || i >= height - padDimensionI ||
                       j < padDimensionJ || j >= width - padDimensionJ
                    {
                        neurons[depth].get(i, j)!.v[elem].out = _padValue
                    }
                    else
                    {
                        let I = i-padDimensionI
                        let J = j-padDimensionJ
                        
                        let iPrev = Double(I) * ratioInOutI
                        let jPrev = Double(J) * ratioInOutJ
                        
                        let iPrevInf = Int(floor(iPrev))
                        let iPrevSup = Int(ceil(iPrev))
                        let jPrevInf = Int(floor(jPrev))
                        let jPrevSup = Int(ceil(jPrev))
                        
                        let iWeight = ratioInOutI * Double(I) - Double(iPrevInf)
                        let jWeight = ratioInOutJ * Double(J) - Double(jPrevInf)
                        
                        let outPrev11 = neuronsPrev[depth].get(
                            iPrevInf, jPrevInf)!.v[elem].out
                        let outPrev12 = neuronsPrev[depth].get(
                            iPrevInf, jPrevSup)!.v[elem].out
                        let outPrev21 = neuronsPrev[depth].get(
                            iPrevSup, jPrevInf)!.v[elem].out
                        let outPrev22 = neuronsPrev[depth].get(
                            iPrevSup, jPrevSup)!.v[elem].out
                        
                        var out = outPrev11 * (1.0 - iWeight) * (1.0 - jWeight)
                        out += outPrev12 * (1.0 - iWeight) * jWeight
                        out += outPrev21 * iWeight * (1.0 - jWeight)
                        out += outPrev22 * iWeight * jWeight
                        
                        neurons[depth].get(i, j)!.v[elem].out = out
                    }
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
                
                _widthResize = Int(round(ratioInOut * Double(widthPrev)))
                _heightResize = Int(round(ratioInOut * Double(heightPrev)))
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
            let pPadValue: [Float] = [Float(_padValue)]
            
            let command = MetalKernel.get.createCommand(
                "resizeBilinearPadForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbChannels, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pDimensionsPrev, atIndex: 3)
            command.setBytes(pDimensionsResize, atIndex: 4)
            command.setBytes(pPadValue, atIndex: 5)
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
            
            let ratioInOutI = Double(heightPrev - 1) / Double(_heightResize - 1)
            let ratioInOutJ = Double(widthPrev - 1) / Double(_widthResize - 1)
            let padDimensionI = (height - _heightResize) / 2
            let padDimensionJ = (width - _widthResize) / 2
            
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<_heightResize {
                for j in 0..<_widthResize
                {
                    let iPrev = Double(i) * ratioInOutI
                    let jPrev = Double(j) * ratioInOutJ
                    
                    let iPrevInf = Int(floor(iPrev))
                    let iPrevSup = Int(ceil(iPrev))
                    let jPrevInf = Int(floor(jPrev))
                    let jPrevSup = Int(ceil(jPrev))
                    
                    let iWeight = ratioInOutI * Double(i) - Double(iPrevInf)
                    let jWeight = ratioInOutJ * Double(j) - Double(jPrevInf)
                    
                    let delta = neurons[depth].get(
                        i+padDimensionI, j+padDimensionJ)!.v[elem].delta
                    
                    neuronsPrev[depth].get(iPrevInf, jPrevInf)!.v[elem].delta +=
                        delta * (1.0 - iWeight) * (1.0 - jWeight)
                    neuronsPrev[depth].get(iPrevInf, jPrevSup)!.v[elem].delta +=
                        delta * (1.0 - iWeight) * jWeight
                    neuronsPrev[depth].get(iPrevSup, jPrevInf)!.v[elem].delta +=
                        delta * iWeight * (1.0 - jWeight)
                    neuronsPrev[depth].get(iPrevSup, jPrevSup)!.v[elem].delta +=
                        delta * iWeight * jWeight
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
            
            command = MetalKernel.get.createCommand(
                "resizeBilinearPadBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbChannels, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pDimensionsPrev, atIndex: 3)
            command.setBytes(pDimensionsResize, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal, atIndex: 6)
            
            command.dispatchThreads(
                width: widthPrev * nbChannels,
                height: heightPrev * batchSize
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}