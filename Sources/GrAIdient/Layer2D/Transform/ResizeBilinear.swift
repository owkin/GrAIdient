//
// ResizeBilinear.swift
// GrAIdient
//
// Created by Jean-François Reboud on 14/04/2023.
//

import Foundation

/// Layer with a 2D shape neural structure.
public class ResizeBilinear: ResizeBilinearPad
{
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - dimension: Height & width of each channel.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: Layer2D,
                dimension: Int,
                params: GrAI.Model.Params) throws
    {
        if layerPrev.height != layerPrev.width
        {
            throw LayerError.Init(
                message: "ResizeBilinear only supports squared images."
            )
        }
        super.init(
            layerPrev: layerPrev,
            scalesList: [Double(dimension) / Double(layerPrev.height)],
            padValue: 0.0,
            params: params
        )
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
            
        let dimension = Double(layerPrev.height) * _scalesList[0]
        let layer = try! ResizeBilinear(
            layerPrev: layerPrev,
            dimension: Int(round(dimension)),
            params: params
        )
        return layer
    }
}
