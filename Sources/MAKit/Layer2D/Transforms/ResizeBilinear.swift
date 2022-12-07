//
// ResizeBilinear.swift
// MAKit
//
// Created by Jean-François Reboud on 07/12/2022.
//

import Foundation

///
/// Layer with a 2D shape neural structure.
///
/// Bilnear resize input grids.
///
/// Note that this layer should be followed by a layer that can enforce dimensions:
/// - AdaptiveAvgPool2D
/// - AvgPool2D
///
public class ResizeBilinear: Layer2D
{
    let _scalesList: [Double]
    
    var _ratioInOutI: Double = 1.0
    var _ratioInOutJ: Double = 1.0
    
    private enum Keys: String, CodingKey
    {
        case scalesList
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - scalesList: List of scales to apply to (height, width) dimensions.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                scalesList: [Double],
                params: MAKit.Model.Params)
    {
        if scalesList.count == 0
        {
            fatalError("`scalesList` should have at least one element.")
        }
        _scalesList = scalesList
        
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
            
        let layer = ResizeBilinear(
            layerPrev: layerPrev,
            scalesList: _scalesList,
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
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let iProj = Double(i) * _ratioInOutI
                    let jProj = Double(j) * _ratioInOutJ
                    
                    let iProjInf = Int(floor(iProj))
                    let iProjSup = Int(ceil(iProj))
                    let jProjInf = Int(floor(jProj))
                    let jProjSup = Int(ceil(jProj))
                    
                    let iWeight = _ratioInOutI * Double(i) - Double(iProjInf)
                    let jWeight = _ratioInOutJ * Double(j) - Double(jProjInf)
                    
                    let outPrev11 = neuronsPrev[depth].get(
                        iProjInf, jProjInf)!.gc[batch][elem].out
                    let outPrev12 = neuronsPrev[depth].get(
                        iProjInf, jProjSup)!.gc[batch][elem].out
                    let outPrev21 = neuronsPrev[depth].get(
                        iProjSup, jProjInf)!.gc[batch][elem].out
                    let outPrev22 = neuronsPrev[depth].get(
                        iProjSup, jProjSup)!.gc[batch][elem].out
                    
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
        //
        // WARNING: dangerous behavior.
        // - reset inner structure
        //
        if _scalesList.count > 1
        {
            resetKernelCPU()
        }
        
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
            let heightPrev = layerPrev.height
            let widthPrev = layerPrev.width
            
            //
            // WARNING: dangerous behavior.
            // - update dimension structure
            // - reset inner structure
            // - re build inner structure
            //
            if _scalesList.count > 1
            {
                let randIndex = Int.random(in: 0..<_scalesList.count)
                let ratioInOut = _scalesList[randIndex]
                width = Int(round(ratioInOut * Double(widthPrev)))
                height = Int(round(ratioInOut * Double(heightPrev)))
                
                resetKernelCPU()
            }
            try checkStateCPU(batchSize: batchSize)
            
            _ratioInOutI = Double(heightPrev - 1) / Double(height - 1)
            _ratioInOutJ = Double(widthPrev - 1) / Double(width - 1)
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let iPrev = Double(i) * _ratioInOutI
                    let jPrev = Double(j) * _ratioInOutJ
                    
                    let iPrevInf = Int(floor(iPrev))
                    let iPrevSup = Int(ceil(iPrev))
                    let jPrevInf = Int(floor(jPrev))
                    let jPrevSup = Int(ceil(jPrev))
                    
                    let iWeight = _ratioInOutI * Double(i) - Double(iPrevInf)
                    let jWeight = _ratioInOutJ * Double(j) - Double(jPrevInf)
                    
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
            
            //
            // WARNING: dangerous behavior.
            // - update dimension structure
            //
            if _scalesList.count > 1
            {
                let randIndex = Int.random(in: 0..<_scalesList.count)
                let ratioInOut = _scalesList[randIndex]
                width = Int(round(ratioInOut * Double(widthPrev)))
                height = Int(round(ratioInOut * Double(heightPrev)))
            }
            
            _ratioInOutI = Double(heightPrev - 1) / Double(height - 1)
            _ratioInOutJ = Double(widthPrev - 1) / Double(width - 1)
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pDimensionsPrev: [UInt32] = [
                UInt32(widthPrev), UInt32(heightPrev)
            ]
            let pRatioInOut: [Float] = [
                Float(_ratioInOutJ), Float(_ratioInOutI)
            ]
            
            let command = MetalKernel.get.createCommand(
                "resizeBilinearForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbChannels, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pDimensionsPrev, atIndex: 3)
            command.setBytes(pRatioInOut, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBuffer(outs.metal, atIndex: 6)
            
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
            
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let iPrev = Double(i) * _ratioInOutI
                    let jPrev = Double(j) * _ratioInOutJ
                    
                    let iPrevInf = Int(floor(iPrev))
                    let iPrevSup = Int(ceil(iPrev))
                    let jPrevInf = Int(floor(jPrev))
                    let jPrevSup = Int(ceil(jPrev))
                    
                    let iWeight = _ratioInOutI * Double(i) - Double(iPrevInf)
                    let jWeight = _ratioInOutJ * Double(j) - Double(jPrevInf)
                    
                    let delta = neurons[depth].get(i, j)!.v[elem].delta
                    
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
            
            /*for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<heightPrev {
                for j in 0..<widthPrev
                {
                    let iProj = Double(i) / _ratioInOutI
                    let jProj = Double(j) / _ratioInOutJ
                    
                    let iProjInf = Int(floor(iProj))
                    let iProjSup = Int(ceil(iProj))
                    let jProjInf = Int(floor(jProj))
                    let jProjSup = Int(ceil(jProj))
                    
                    let iPrevInf = Double(iProjInf) * _ratioInOutI
                    let iPrevSup = Double(iProjSup) * _ratioInOutI
                    let jPrevInf = Double(jProjInf) * _ratioInOutJ
                    let jPrevSup = Double(jProjSup) * _ratioInOutJ
                    
                    let iPrevInfInf = floor(iPrevInf)
                    let iPrevSupInf = floor(iPrevSup)
                    let jPrevInfInf = floor(jPrevInf)
                    let jPrevSupInf = floor(jPrevSup)
                    
                    let iwPrevInf = iPrevInf - iPrevInfInf
                    let iwPrevSup = iPrevSup - iPrevSupInf
                    let jwPrevInf = jPrevInf - jPrevInfInf
                    let jwPrevSup = jPrevSup - jPrevSupInf
                    
                    let delta11 =
                        neurons[depth].get(iProjInf, jProjInf)!.v[elem].delta
                    let delta12 =
                        neurons[depth].get(iProjInf, jProjSup)!.v[elem].delta
                    let delta21 =
                        neurons[depth].get(iProjSup, jProjInf)!.v[elem].delta
                    let delta22 =
                        neurons[depth].get(iProjSup, jProjSup)!.v[elem].delta
                    
                    var deltaCur = delta11 * iwPrevInf * jwPrevInf
                    deltaCur += delta12 * iwPrevInf * (1.0 - jwPrevSup)
                    deltaCur += delta21 * (1.0 - iwPrevSup) * jwPrevInf
                    deltaCur += delta22 * (1.0 - iwPrevSup) * (1.0 - jwPrevSup)
                   
                    if layerPrev.dirty
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta =
                            deltaCur
                    }
                    else
                    {
                        neuronsPrev[depth].get(i, j)!.v[elem].delta +=
                            deltaCur
                    }
                }}
            }}*/
            
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
            let pRatioInOut: [Float] = [
                Float(_ratioInOutJ), Float(_ratioInOutI)
            ]
            
            command = MetalKernel.get.createCommand(
                "resizeBilinearBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbChannels, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pDimensionsPrev, atIndex: 3)
            command.setBytes(pRatioInOut, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBuffer(layerPrev.delta.metal, atIndex: 6)
            
            command.dispatchThreads(width: nbChannels, height: batchSize)
            command.enqueue()
            
            propagateDirty()
        }
    }
}
