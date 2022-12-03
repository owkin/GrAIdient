//
// FTFrequences2DMSE1DCase.swift
// MAKitTests
//
// Created by Jean-François Reboud on 03/12/2022.
//

import MAKit

///
/// A class that will test a model with a structural hypothesis:
/// the model last layer is a MSE1D layer,
/// the model first layer is a Input2D,
/// the model second layer is a FTFrequences2D.
///
class FTFrequences2DMSE1DCase: Input2DMSE1DCase
{
    ///
    /// A function to create/set data to the model.
    ///
    /// - Parameters:
    ///     - inputs: The data to set.
    ///     - model: The model.
    /// - Returns: (The data, the batch size).
    ///
    override func setData(
        _ inputs: [[Double]]?,
        _ model: Model) -> ([[Double]], Int)
    {
        let secondLayer = model.layers[1] as! FTFrequences2D
        if MAKit.Opti.GPU
        {
            try! secondLayer.setDataGPU(batchSize: getBatchSize(model))
        }
        else
        {
            try! secondLayer.setDataCPU(batchSize: getBatchSize(model))
        }
        return super.setData(inputs, model)
    }
}
