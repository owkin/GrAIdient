//
// ModelContext.swift
// MAKit
//
// Created by Jean-François Reboud on 05/10/2022.
//

import Foundation

///
/// An object that is passed during the creation of the different layers for them to be added in their
/// `Model` queue.
///
/// The main problem adressed by the model context is to be able to edit the very graph of the
/// deep-learning calls of the model. The basic bricks being the layers, they have to be able to refer to
/// previous ones. Plus, we want these references to be written to the disk in order to load them later.
/// Hence the need of an id for each layer.
/// So now, why not using a global unique id ? This global unique id would be less convenient
/// with multiple mirrored models in the context of federated learning on multiple GPU at once.
/// The current solution allows to consider several "threads" of models.
/// Each thread is composed of models where the id are unique in the given "thread".
/// But looking at another "thread" of mirrored models, we would find the exact same ids.
/// As a conclusion, it appears that the id is not unique, but closely related to the model in which it is used.
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
