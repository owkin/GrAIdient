//
// LayerInput1D.swift
// MAKit
//
// Created by Jean-François Reboud on 09/10/2022.
//

/// First layer of a model.
open class LayerInput1D: Layer1D
{
    ///
    /// Whether we must call the forward for this layer or not.
    ///
    /// When this layer is the first of a model, user is supposed to pass data with an explicit API.
    /// When this layer is an intermediate layer of the model, user expects to rely to the forward API:
    /// in that case, set `computeForward` to true.
    ///
    public var computeForward = false
    
    /// Number of different weigths for which we are estimating the gradient during Gradient Checking.
    public override var nbGC: Int
    {
        get {
            return 0
        }
    }
    
    ///
    /// Find the `layerPrev` associated to the layer's `idPrev`.
    ///
    /// - Parameter layers: The potential layers where to find the layer's `idPrev`.
    ///
    public override func initLinks(_ layers: [Layer])
    {
        super.initLinks(layers)
        if layerPrev != nil
        {
            computeDelta = true
        }
    }
}
