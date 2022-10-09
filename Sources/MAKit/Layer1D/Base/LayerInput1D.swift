//
// LayerInput1D.swift
// MAKit
//
// Created by Jean-François Reboud on 09/10/2022.
//

open class LayerInput1D: Layer1D
{
    // By default InputLayer is the first layer.
    // But it could be used as an intermediate layer between two networks.
    public var computeForward = false
    
    public override var nbGC: Int
    {
        get {
            return 0
        }
    }
    
    public override func initLinks(_ layers: [Layer])
    {
        super.initLinks(layers)
        if layerPrev != nil
        {
            computeDelta = true
        }
    }
}
