//
// LayerMerge2D.swift
// GrAIdient
//
// Created by Jean-François Reboud on 09/10/2022.
//

/// Layer that is connected with more than 1 previous layer.
open class LayerMerge2D: Layer2D
{
    /// List of links to the previous layers in the model.
    public var layersPrev = [Layer]()
    /// List of identifiers of the previous layers in the model.
    public let idsPrev: [Int]
    
    /// Whether backward pass should continue backward or not.
    public override var mustComputeBackward: Bool
    {
        get {
            for layerPrev in layersPrev
            {
                if layerPrev.computeDelta
                {
                    return true
                }
            }
            return false
        }
    }
    
    /// Downscale factor of the resolution (height and width).
    public override var strideFactor: Double
    {
        if let value = strideFactorCache
        {
            return value
        }
        
        var valueFirst: Double! = nil
        for layerPrev in layersPrev
        {
            if let layerPrevTmp = layerPrev as? Layer2D
            {
                let value = layerPrevTmp.strideFactor
                if valueFirst == nil
                {
                    valueFirst = value
                }
                else if value != valueFirst
                {
                    fatalError("Branches have not same 'strideFactor'.")
                }
            }
        }
        
        strideFactorCache = valueFirst
        return valueFirst
    }
    
    /// The size of the input image this layer is looking at.
    public override var receptiveField: Int
    {
        if let value = receptiveFieldCache
        {
            return value
        }
        
        var valueMax: Int! = nil
        for layerPrev in layersPrev
        {
            if let layerPrevTmp = layerPrev as? Layer2D
            {
                let value = layerPrevTmp.receptiveField
                if valueMax == nil || value > valueMax
                {
                    valueMax = value
                }
            }
        }
        
        receptiveFieldCache = valueMax
        return valueMax
    }
    
    private enum Keys: String, CodingKey
    {
        case idsPrev
    }
    
    ///
    /// Create a layer with a 2D shape neural structure.
    ///
    /// - Parameters:
    ///     - layersPrev: List of previous layers that have been queued to the model.
    ///     - nbChannels: Number of channels.
    ///     - height: Height of each channel.
    ///     - width: Width of each channel.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layersPrev: [Layer],
                nbChannels: Int, height: Int, width: Int,
                params: GrAI.Model.Params)
    {
        var idsPrev = [Int]()
        for layer in layersPrev
        {
            idsPrev.append(layer.id)
        }
        self.idsPrev = idsPrev
        
        super.init(layerPrev: layersPrev[0],
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
        let container = try decoder.container(keyedBy: Keys.self)
        idsPrev = try container.decode([Int].self, forKey: .idsPrev)
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
        try container.encode(idsPrev, forKey: .idsPrev)
        try super.encode(to: encoder)
    }
    
    ///
    /// Find the `layerPrev` associated to the layer's `idPrev`.
    ///
    /// - Parameter layers: The potential layers where to find the layer's `idPrev`.
    ///
    public override func initLinks(_ layers: [Layer])
    {
        layersPrev = [Layer]()
        for id in idsPrev
        {
            for testLayer in layers
            {
                if testLayer.id == id
                {
                    layersPrev.append(testLayer)
                    break
                }
            }
        }
    }
    
    ///
    /// Update the backward dirty flag for `layerPrev` instance.
    ///
    /// - Parameter dirty: The boolean value for the dirty flag.
    ///
    public override func propagateDirty(_ dirty: Bool = false)
    {
        for num in 0..<layersPrev.count
        {
            layersPrev[num].dirty = dirty
        }
    }
    
    ///
    /// Get the different layers (a "graph") between the first common ancestor and this.
    ///
    /// - Returns: (The list of different layers after the common ancestor,
    ///            The list of different layers id after the common ancestor).
    ///
    private func _getMergedGraph() -> ([Layer], [Int])
    {
        var layersBranches = [Layer?]()
        for layer in layersPrev
        {
            layersBranches.append(layer)
        }
        
        let layersEqual =
        {
            () -> Bool in
            let firstLayer = layersBranches.first!
            for layer in layersBranches
            {
                if layer !== firstLayer
                {
                    return false
                }
            }
            return true
        }
        
        var layersIndex = [Int]()
        var layers = [Layer]()
        while !layersEqual()
        {
            var idMax = -1
            var indexMax = -1
            
            for (index, layer) in layersBranches.enumerated()
            {
                if let layerTmp = layer
                {
                    let id = layerTmp.id
                    if id > idMax
                    {
                        idMax = id
                        indexMax = index
                    }
                }
            }
            if indexMax < 0
            {
                break
            }
            
            let layerMax = layersBranches[indexMax]!
            layersBranches[indexMax] = layerMax.layerPrev
            
            layersIndex.append(indexMax)
            layers.append(layerMax)
        }
        
        return (layers, layersIndex)
    }
    
    ///
    /// Get every layers (a "graph") between the very first of the `Model` and this.
    ///
    /// - Parameter layerPrev: The different layers found in the "graph".
    ///
    public override func getGraph(_ layers: inout [Layer])
    {
        layers.append(self)
        
        let layersMerged = _getMergedGraph().0
        layers += layersMerged
        
        layersMerged.last?.layerPrev?.getGraph(&layers)
    }
    
    ///
    /// Get every layers (a "graph") between the very first of the `Model` and this.
    ///
    /// The main difficulty with a `LayerMerge` is that we must take into account the origin of the
    /// weight modifications for estimating their gradient during the Gradient Checking.
    /// When we look at the "graph" of a `LayerMerge` we must consider the last common ancestor
    /// before the fork.
    /// The weights originating before the fork should only undergo a "simple forward" from the
    /// layers that appear after the fork.
    /// But the weights modifications that pop after a fork should have a particular behavior as they
    /// are populating a new weight modification that is related to one precise branch.
    ///
    /// - Returns: (Number of  weight modifications that occur before the fork,
    ///            Index of the different layers after the fork,
    ///            Number of weight modifications associated with the different layers after the fork).
    ///
    public func getMergedGraph() -> (nbSameElems: Int,
                                     layersIndex: [Int],
                                     nbElems: [Int])
    {
        var (layersMerged, layersIndex) = _getMergedGraph()
        
        var nbSameElems = 0
        if let commonAncestor = layersMerged.last!.layerPrev
        {
            nbSameElems = commonAncestor.nbGC
        }
        
        layersMerged = layersMerged.reversed()
        layersIndex = layersIndex.reversed()
        
        var nbElems = [Int]()
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: layersPrev.count)
        for (index, layer) in zip(layersIndex, layersMerged)
        {
            let nbElemsTmp = layer.nbGC
            let nbDiffElems = nbElemsTmp - nbLastElems[index]
            
            nbLastElems[index] += nbDiffElems
            nbElems.append(nbDiffElems)
        }
        
        return (nbSameElems: nbSameElems,
                layersIndex: layersIndex,
                nbElems: nbElems)
    }
}
