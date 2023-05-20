//
// ColorJitterHSV.swift
// GrAIdient
//
//  Created by Jean-François Reboud on 19/05/2023.
//

import Foundation

/// Error occuring when range could not be built.
public enum RangeError: Error
{
    /// Values specifiied are not coherent.
    case ValueError
}

extension RangeError: CustomStringConvertible
{
    public var description: String
    {
        switch self
        {
        case .ValueError:
            return "Values specified are not coherent."
        }
    }
}

/// A bounded interval.
public struct Range<T: BinaryFloatingPoint & Codable>: Codable
{
    let min: T
    let max: T
    
    ///
    /// Create the bounded interval.
    ///
    /// Throw an error when parameter values are not coherent.
    ///
    /// - Parameters:
    ///     - min: The minimum value of the interval.
    ///     - max: The maximum value of the interval.
    ///
    public init(min: T, max: T) throws
    {
        self.min = min
        self.max = max
        if max < min
        {
            throw RangeError.ValueError
        }
    }
}

///
/// Layer with a 2D shape neural structure.
///
/// This layer add some noise (in the HSV space) to the RGB channels of the previous layer.
///
public class ColorJitterHSV: Layer2D
{
    let _rangeH: Range<Double>
    let _rangeS: Range<Double>
    let _rangeV: Range<Double>
    
    private enum Keys: String, CodingKey
    {
        case rangeH
        case rangeS
        case rangeV
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - rangeH: Range of noise in the hue dimension.
    ///     - rangeS: Range of noise in the saturation dimension.
    ///     - rangeV: Range of noise in the value dimension.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                rangeH: Range<Double>,
                rangeS: Range<Double>,
                rangeV: Range<Double>,
                params: GrAI.Model.Params)
    {
        _rangeH = rangeH
        _rangeS = rangeS
        _rangeV = rangeV
        
        if rangeH.min < 0 || rangeH.max > 360 ||
           rangeS.min < 0 || rangeS.max > 1 ||
           rangeV.min < 0 || rangeV.max > 1
        {
            fatalError()
        }
        
        let width = layerPrev.width
        let height = layerPrev.height
        let nbChannels = layerPrev.nbChannels
        
        if nbChannels != 3
        {
            fatalError("Previous layer should have 3 channels: RGB.")
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
        _rangeH = try values.decode(Range<Double>.self, forKey: Keys.rangeH)
        _rangeS = try values.decode(Range<Double>.self, forKey: Keys.rangeS)
        _rangeV = try values.decode(Range<Double>.self, forKey: Keys.rangeV)
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
        try container.encode(_rangeH, forKey: Keys.rangeH)
        try container.encode(_rangeS, forKey: Keys.rangeS)
        try container.encode(_rangeV, forKey: Keys.rangeV)
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
            
        let layer = ColorJitterHSV(
            layerPrev: layerPrev,
            rangeH: _rangeH,
            rangeS: _rangeS,
            rangeV: _rangeV,
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
        fatalError("Not implemented.")
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
            
            let noiseH = Double.random(in: _rangeH.min..._rangeH.max)
            let noiseS = Double.random(in: _rangeS.min..._rangeS.max)
            let noiseV = Double.random(in: _rangeV.min..._rangeV.max)
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize
            {
                for row in 0..<height {
                for col in 0..<width
                {
                    var r = neuronsPrev[0].get(row, col)!.v[elem].out
                    var g = neuronsPrev[1].get(row, col)!.v[elem].out
                    var b = neuronsPrev[2].get(row, col)!.v[elem].out
                    
                    let maxValue = max(r, g, b)
                    let minValue = min(r, g, b)
                    let delta = maxValue - minValue
                    
                    var h: Double
                    if delta == 0
                    {
                        h = 0.0
                    }
                    else if maxValue == r
                    {
                        h = (g - b) / delta
                    }
                    else if maxValue == g
                    {
                        h = (g - b) / delta + 2.0
                    }
                    else
                    {
                        h = (g - b) / delta + 4.0
                    }
                    h *= 60.0
                    
                    var s: Double = 0.0
                    if maxValue != 0
                    {
                        s = delta / maxValue
                    }
                    
                    var v = maxValue
                    
                    h += noiseH; h = max(h, 0.0); h = min(h, 360.0)
                    s += noiseS; s = max(s, 0.0); s = min(s, 1.0)
                    v += noiseV; v = max(v, 0.0); v = min(v, 1.0)
                    
                    if s == 0.0
                    {
                        r = v; g = v; b = v
                    }
                    
                    let angle = h
                    let sector = angle / 60 // Sector
                    let i = floor(sector)
                    let f = sector - i // Factorial part of h
                    
                    let p = v * (1 - s)
                    let q = v * (1 - (s * f))
                    let t = v * (1 - (s * (1 - f)))
                    
                    switch(i) {
                    case 0:
                        r = v; g = t; b = p
                    case 1:
                        r = q; g = v; b = p
                    case 2:
                        r = p; g = v; b = t
                    case 3:
                        r = p; g = q; b = v
                    case 4:
                        r = t; g = p; b = v
                    default:
                        r = v; g = p; b = q
                    }
                    
                    neurons[0].get(row, col)!.v[elem].out = r
                    neurons[1].get(row, col)!.v[elem].out = g
                    neurons[2].get(row, col)!.v[elem].out = b
                }}
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
            
            let noiseH = Double.random(in: _rangeH.min..._rangeH.max)
            let noiseS = Double.random(in: _rangeS.min..._rangeS.max)
            let noiseV = Double.random(in: _rangeV.min..._rangeV.max)
            
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pNoise: [Float] = [Float(noiseH), Float(noiseS), Float(noiseV)]
            
            let command = MetalKernel.get.createCommand(
                "colorJitterHSVForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNoise, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pNbBatch, atIndex: 3)
            command.setBuffer(outs.metal, atIndex: 4)
            
            command.dispatchThreads(
                width: height * width,
                height: batchSize
            )
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        fatalError("Not implemented.")
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        fatalError("Not implemented.")
    }
}
