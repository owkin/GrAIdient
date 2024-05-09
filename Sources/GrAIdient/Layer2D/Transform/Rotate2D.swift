//
// Rotate2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 08/12/2022.
//

import Foundation

///
/// Layer with a 2D shape neural structure.
///
/// Rotate input grids.
///
public class Rotate2D: Layer2D
{
    let _anglesList: [Double]
    let _minAngle: Double?
    let _maxAngle: Double?
    
    let _padValue: Double
    
    var _angle: Double = 0.0
    
    private enum Keys: String, CodingKey
    {
        case anglesList
        case minAngle
        case maxAngle
        case padValue
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - anglesList: List of rotation angles to apply.
    ///     - padValue: Value to set on the missing values.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                anglesList: [Double],
                padValue: Double,
                params: GrAI.Model.Params) throws
    {
        _padValue = padValue
        _anglesList = anglesList
        _minAngle = nil
        _maxAngle = nil
        
        if anglesList.count == 0
        {
            throw LayerError.Init(
                message: "`anglesList` should have at least one element."
            )
        }
        
        let nbChannels = layerPrev.nbChannels
        let height = layerPrev.height
        let width = layerPrev.width
        
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
    ///     - minAngle: Minimum angle rotation to apply.
    ///     - maxAngle: Maximum angle rotation to apply.
    ///     - padValue: Value to set on the missing values.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                minAngle: Double,
                maxAngle: Double,
                padValue: Double,
                params: GrAI.Model.Params) throws
    {
        _padValue = padValue
        _anglesList = []
        _minAngle = minAngle
        _maxAngle = maxAngle
        
        if minAngle >= maxAngle
        {
            throw LayerError.Init(message: "`minAngle` is not coherent.")
        }
        
        let nbChannels = layerPrev.nbChannels
        let height = layerPrev.height
        let width = layerPrev.width
        
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
        _anglesList = try values.decode(
            [Double].self, forKey: Keys.anglesList
        )
        _minAngle = try values.decodeIfPresent(
            Double.self, forKey: Keys.minAngle
        )
        _maxAngle = try values.decodeIfPresent(
            Double.self, forKey: Keys.maxAngle
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
        try container.encode(_anglesList, forKey: Keys.anglesList)
        if let minAngle = _minAngle
        {
            try container.encode(minAngle, forKey: Keys.minAngle)
        }
        if let maxAngle = _maxAngle
        {
            try container.encode(maxAngle, forKey: Keys.maxAngle)
        }
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
            
        let layer: Rotate2D
        if _anglesList.count != 0
        {
            layer = try! Rotate2D(
                layerPrev: layerPrev,
                anglesList: _anglesList,
                padValue: _padValue,
                params: params
            )
        }
        else if let minAngle = _minAngle, let maxAngle = _maxAngle
        {
            layer = try! Rotate2D(
                layerPrev: layerPrev,
                minAngle: minAngle,
                maxAngle: maxAngle,
                padValue: _padValue,
                params: params
            )
        }
        else
        {
            fatalError()
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
            
            let centerI = Double(height - 1) / 2.0
            let centerJ = Double(width - 1) / 2.0
            
            let neuronsPrev = layerPrev.neurons
            for batch in 0..<batchSize {
            for elem in 0..<nbGC {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let prevJ: Double =
                        cos(-_angle) * (Double(j) - centerJ) +
                        sin(-_angle) * (Double(i) - centerI) + centerJ
                    let prevI: Double =
                        cos(-_angle) * (Double(i) - centerI) -
                        sin(-_angle) * (Double(j) - centerJ) + centerI
                    
                    if round(prevJ) < 0 || round(prevJ) >= Double(width) ||
                       round(prevI) < 0 || round(prevI) >= Double(height)
                    {
                        neurons[depth].get(i, j)!.gc[batch][elem].out =
                            _padValue
                    }
                    else
                    {
                        neurons[depth].get(i, j)!.gc[batch][elem].out =
                            neuronsPrev[depth].get(
                                Int(round(prevI)), Int(round(prevJ))
                            )!.gc[batch][elem].out
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
        
            let angle: Double
            if _anglesList.count != 0
            {
                let randIndex = Int.random(in: 0..<_anglesList.count)
                angle = _anglesList[randIndex]
            }
            else if _anglesList.count == 0,
                 let minAngle = _minAngle, let maxAngle = _maxAngle
            {
                angle = Double.random(in: minAngle...maxAngle)
            }
            else
            {
                fatalError()
            }
            _angle = angle * Double.pi / 180.0
            
            let centerI = Double(height - 1) / 2.0
            let centerJ = Double(width - 1) / 2.0
            
            let neuronsPrev = layerPrev.neurons
            for elem in 0..<batchSize {
            for depth in 0..<nbChannels
            {
                for i in 0..<height {
                for j in 0..<width
                {
                    let prevJ: Double =
                        cos(-_angle) * (Double(j) - centerJ) +
                        sin(-_angle) * (Double(i) - centerI) + centerJ
                    let prevI: Double =
                        cos(-_angle) * (Double(i) - centerI) -
                        sin(-_angle) * (Double(j) - centerJ) + centerI
                    
                    if round(prevJ) < 0 || round(prevJ) >= Double(width) ||
                       round(prevI) < 0 || round(prevI) >= Double(height)
                    {
                        neurons[depth].get(i, j)!.v[elem].out = _padValue
                    }
                    else
                    {
                        neurons[depth].get(i, j)!.v[elem].out =
                            neuronsPrev[depth].get(
                                Int(round(prevI)), Int(round(prevJ))
                            )!.v[elem].out
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
            
            let angle: Double
            if _anglesList.count != 0
            {
                let randIndex = Int.random(in: 0..<_anglesList.count)
                angle = _anglesList[randIndex]
            }
            else if _anglesList.count == 0,
                 let minAngle = _minAngle, let maxAngle = _maxAngle
            {
                angle = Double.random(in: minAngle...maxAngle)
            }
            else
            {
                fatalError()
            }
            _angle = angle * Double.pi / 180.0
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pAngle: [Float] = [Float(_angle)]
            let pPadValue: [Float] = [Float(_padValue)]
            
            let command = MetalKernel.get.createCommand(
                "rotate2DForward", deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBytes(pNbChannels, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pAngle, atIndex: 3)
            command.setBytes(pPadValue, atIndex: 4)
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
            let centerI = Double(height - 1) / 2.0
            let centerJ = Double(width - 1) / 2.0
            
            let neuronsPrev = layerPrev.neurons
            if layerPrev.dirty
            {
                for elem in 0..<batchSize {
                for depth in 0..<nbChannels
                {
                    for i in 0..<height {
                    for j in 0..<width
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
                    let prevJ: Double =
                        cos(-_angle) * (Double(j) - centerJ) +
                        sin(-_angle) * (Double(i) - centerI) + centerJ
                    let prevI: Double =
                        cos(-_angle) * (Double(i) - centerI) -
                        sin(-_angle) * (Double(j) - centerJ) + centerI
                    
                    if round(prevJ) >= 0 && round(prevJ) < Double(width) &&
                       round(prevI) >= 0 && round(prevI) < Double(height)
                    {
                        neuronsPrev[depth].get(
                            Int(round(prevI)), Int(round(prevJ))
                        )!.v[elem].delta +=
                            neurons[depth].get(i, j)!.v[elem].delta
                    }
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
            
            let pNbChannels: [UInt32] = [UInt32(nbChannels)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pDimensions: [UInt32] = [UInt32(width), UInt32(height)]
            let pAngle: [Float] = [Float16(_angle)]
            
            command = MetalKernel.get.createCommand(
                "rotate2DBackward", deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBytes(pNbChannels, atIndex: 1)
            command.setBytes(pDimensions, atIndex: 2)
            command.setBytes(pAngle, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBuffer(layerPrev.delta.metal, atIndex: 5)
            
            command.dispatchThreads(
                width: width * nbChannels,
                height: height * batchSize
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
}
