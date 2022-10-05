//
// ModelContext.swift
// MAKit
//
// Created by Jean-François Reboud on 05/10/2022.
//

import Foundation

public class ModelContext
{
    public var curID = 0
    public var model: BaseModel
    
    public init(name: String, curID: Int)
    {
        self.model = BaseModel(name: name)
        self.curID = curID
    }
    
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
    
    func update(layer: Layer, hidden: Bool)
    {
        if !hidden
        {
            model.append(layer)
        }
        curID += 1
    }
}
