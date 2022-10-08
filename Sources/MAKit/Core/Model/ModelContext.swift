//
// ModelContext.swift
// MAKit
//
// Created by Jean-François Reboud on 05/10/2022.
//

import Foundation

///
/// An object that is passed during the creation of the different layers in order for them to be linked to the
/// `Model` where they belong.
///
public class ModelContext
{
    /// The id of the next layer to be built.
    public var curID = 0
    /// The pointer toward the model that is being built.
    public var model: BaseModel
    
    ///
    /// Create a model context.
    ///
    /// - Parameters:
    ///     - name: Name of the model.
    ///     - curID: Id of the next layer to be built.
    ///
    public init(name: String, curID: Int)
    {
        self.model = BaseModel(name: name)
        self.curID = curID
    }
    
    ///
    /// Create a model context, taking into account that previous models may have been built before.
    ///
    /// This API is necessary when building several models where layers reference `layerPrev`
    /// among other models than the one they are belonging to.
    ///
    /// - Parameters:
    ///     - name: Name of the model.
    ///     - models: The previous models built.
    ///
    public init(name: String, models: [BaseModel])
    {
        var maxId = 0
        for model in models
        {
            let lastId = model.layers.last!.id
            if lastId > maxId
            {
                maxId = lastId
            }
        }
        
        self.model = BaseModel(name: name)
        self.curID = maxId + 1
    }
    
    ///
    /// Populate model with a new layer.
    ///
    /// - Parameters:
    ///     - layer: The layer to be added to the model.
    ///     - hidden: Whether the layer is hidden inside another layer.
    ///     
    func update(layer: Layer, hidden: Bool)
    {
        if !hidden
        {
            model.append(layer)
        }
        curID += 1
    }
}
