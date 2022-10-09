//
// LayerMerge2D.swift
// MAKit
//
// Created by Jean-François Reboud on 09/10/2022.
//

open class LayerMerge2D: Layer2D
{
    var _layersPrev = [Layer]()
    let _idsPrev: [Int]
    
    public override var mustComputeBackward: Bool
    {
        get {
            for layerPrev in _layersPrev
            {
                if layerPrev.computeDelta
                {
                    return true
                }
            }
            return false
        }
    }
    
    public override var strideFactor: Double
    {
        if let value = strideFactorCache
        {
            return value
        }
        
        var valueFirst: Double! = nil
        for layerPrev in _layersPrev
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
    
    public override var receptiveField: Int
    {
        if let value = receptiveFieldCache
        {
            return value
        }
        
        var valueMax: Int! = nil
        for layerPrev in _layersPrev
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
    
    public init(layersPrev: [Layer],
                nbFilters: Int, height: Int, width: Int,
                params: MAKit.Model.Params)
    {
        var idsPrev = [Int]()
        for layer in layersPrev
        {
            idsPrev.append(layer.id)
        }
        _idsPrev = idsPrev
        
        super.init(layerPrev: layersPrev[0],
                   nbFilters: nbFilters,
                   height: height,
                   width: width,
                   params: params)
    }
    
    public required init(from decoder: Decoder) throws
    {
        let container = try decoder.container(keyedBy: Keys.self)
        _idsPrev = try container.decode([Int].self, forKey: .idsPrev)
        try super.init(from: decoder)
    }
    
    public override func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        try container.encode(_idsPrev, forKey: .idsPrev)
        try super.encode(to: encoder)
    }
    
    public override func initLinks(_ layers: [Layer])
    {
        _layersPrev = [Layer]()
        for id in _idsPrev
        {
            for testLayer in layers
            {
                if testLayer.id == id
                {
                    _layersPrev.append(testLayer)
                    break
                }
            }
        }
    }
    
    public override func propagateDirty(_ dirty: Bool = false)
    {
        for num in 0..<_layersPrev.count
        {
            _layersPrev[num].dirty = dirty
        }
    }
    
    private func _getMergedGraph() -> ([Layer], [Int])
    {
        var layersBranches = [Layer]()
        for layer in _layersPrev
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
            var idMax = 0
            var indexMax = -1
            
            for (index, layer) in layersBranches.enumerated()
            {
                let id = layer.id
                if id > idMax
                {
                    idMax = id
                    indexMax = index
                }
            }
            
            let layerMax = layersBranches[indexMax]
            layersBranches[indexMax] = layerMax.layerPrev!
            
            layersIndex.append(indexMax)
            layers.append(layerMax)
        }
        
        return (layers, layersIndex)
    }
    
    public override func getGraph(_ layers: inout [Layer])
    {
        layers.append(self)
        
        let layersMerged = _getMergedGraph().0
        layers += layersMerged
        
        layersMerged.last?.layerPrev?.getGraph(&layers)
    }
    
    public func getMergedGraph() -> (nbSameElems: Int,
                                     layersIndex: [Int],
                                     nbElems: [Int])
    {
        var (layersMerged, layersIndex) = _getMergedGraph()
        
        let sameLayer = layersMerged.last!.layerPrev!
        let nbSameElems = sameLayer.nbGC
        
        layersMerged = layersMerged.reversed()
        layersIndex = layersIndex.reversed()
        
        var nbElems = [Int]()
        var nbLastElems = [Int](repeating: nbSameElems,
                                count: _layersPrev.count)
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
