//
// Input2DMSE2DCase.swift
// GrAITests
//
// Created by Jean-François Reboud on 05/03/2023.
//

import XCTest
import GrAIdient
import GrAITestsUtils

///
/// A class that will test a model with a structural hypothesis:
/// the model last layer is a MSE2D layer, the model first layer is a Input2D.
///
class Input2DMSE2DCase: XCTestCase, IOCase
{
    var height = 6
    var width = 6
    
    /// Batch size of data.
    var batchSize: Int = -1
    /// Optimizer parameters.
    var optimizerParams = GrAI.Optimizer.Params()
    
    /// Systematic call before test begins.
    override func setUp()
    {
        batchSize = 5
        _ = MetalKernel.get
        GrAI.Opti.GPU = true
        
        setOptimizerParams(params: &optimizerParams)
        optimizerParams.nbLoops = 2
    }
    
    ///
    /// A function to create/set ground truth to the model.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth to set.
    ///     - model: The model.
    /// - Returns: The ground truth.
    ///
    func setLoss(_ groundTruth: [[Double]]?, _ model: Model) -> [[Double]]
    {
        let lastLayer = model.layers.last as! MSE2D
        let gt: [[Double]]
        if let groundTruthTmp = groundTruth
        {
            gt = groundTruthTmp
        }
        else
        {
            gt = buildData(dim1: getBatchSize(model), dim2: height * width)
        }
        
        if GrAI.Opti.GPU
        {
            try! lastLayer.lossDerivativeGPU(
                gt.reduce([], +), batchSize: gt.count, format: .Neuron
            )
        }
        else
        {
            try! lastLayer.lossDerivativeCPU(
                gt.reduce([], +), batchSize: gt.count, format: .Neuron
            )
        }
        return gt
    }
    
    ///
    /// A function to get loss of a model.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth to set.
    ///     - model: The model.
    /// - Returns: The loss value.
    ///
    func getLoss(_ groundTruth: [[Double]], _ model: Model) -> Double
    {
        let lastLayer = model.layers.last as! MSE2D
        if GrAI.Opti.GPU
        {
            return Double(try! lastLayer.getLossGPU(
                groundTruth.reduce([], +),
                batchSize: groundTruth.count,
                format: .Neuron
            ))
        }
        else
        {
            return try! lastLayer.getLossCPU(
                groundTruth.reduce([], +),
                batchSize: groundTruth.count,
                format: .Neuron
            )
        }
    }
    
    ///
    /// A function to get the gradients of weights approximations..
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - model: The model.
    /// - Returns: The gradients of weights approximations.
    ///
    func getGradientsApprox(
        _ groundTruth: [[Double]],
        _ model: Model) -> [Double]
    {
        let lastLayer = model.layers.last as! MSE2D
        return try! lastLayer.collectGradientsApprox(
            groundTruth.reduce([], +),
            batchSize: groundTruth.count,
            format: .Neuron
        )
    }
    
    ///
    /// A function to create/set data to the model.
    ///
    /// - Parameters:
    ///     - inputs: The data to set.
    ///     - model: The model.
    /// - Returns: (The data, the batch size).
    ///
    func setData(_ inputs: [[Double]]?, _ model: Model) -> ([[Double]], Int)
    {
        let firstLayer = model.layers.first as! Input2D
        let ins: [[Double]]
        if let insTmp = inputs
        {
            ins = insTmp
        }
        else
        {
            ins = buildData(dim1: getBatchSize(model), dim2: height * width)
        }
        
        if GrAI.Opti.GPU
        {
            try! firstLayer.setDataGPU(
                ins.reduce([], +), batchSize: ins.count, format: .Neuron
            )
        }
        else
        {
            try! firstLayer.setDataCPU(
                ins.reduce([], +), batchSize: ins.count, format: .Neuron
            )
        }
        return (ins, ins.count)
    }
    
    ///
    /// Copy a model.
    ///
    /// We must call the `initKernel` API.
    ///
    /// - Parameter model: The model.
    /// - Returns: The transformed model.
    ///
    func copy(_ model: Model) -> Model
    {
        let modelNew = Model.copy(models: [model], inPlace: false)[0]
        modelNew.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        return modelNew
    }
    
    ///
    /// Copy a model in place.
    ///
    /// No need to call the `initKernel` API.
    ///
    /// - Parameter model: The model.
    /// - Returns: The transformed model.
    ///
    func copyInPlace(_ model: Model) -> Model
    {
        let modelNew = Model.copy(models: [model], inPlace: true)[0]
        modelNew.setupOptimizers(params: optimizerParams)
        modelNew.phase = .Inference
        return modelNew
    }
    
    ///
    /// Resize a model.
    ///
    /// We must call the `initKernel` API.
    ///
    /// - Parameter model: The model.
    /// - Returns: The transformed model.
    ///
    func resize(_ model: Model) -> Model
    {
        let modelsNew = Model.resize(models: [model],
                                     imageWidth: 2 * width,
                                     imageHeight: 2 * height,
                                     inPlace: false)
        let modelNew = Model.resize(models: modelsNew,
                                    imageWidth: width,
                                    imageHeight: height,
                                    inPlace: false)[0]
        modelNew.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        return modelNew
    }
    
    ///
    /// Resize a model in place.
    ///
    /// No need to call the `initKernel` API.
    ///
    /// - Parameter model: The model.
    /// - Returns: The transformed model.
    ///
    func resizeInPlace(_ model: Model) -> Model
    {
        let modelsNew = Model.resize(models: [model],
                                     imageWidth: 2 * width,
                                     imageHeight: 2 * height,
                                     inPlace: true)
        let modelNew = Model.resize(models: modelsNew,
                                    imageWidth: width,
                                    imageHeight: height,
                                    inPlace: true)[0]
        modelNew.updateKernel(batchSize: batchSize)
        modelNew.setupOptimizers(params: optimizerParams)
        modelNew.phase = .Inference
        return modelNew
    }
    
    /// A list of functions that transform the model into another one.
    var transforms: [(Model) -> Model]
    {
        get {
            return [copy, copyInPlace, resize, resizeInPlace]
        }
    }
}
