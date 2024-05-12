//
// Input1DLinearError1DCase.swift
// GrAITests
//
// Created by Jean-François Reboud on 10/10/2022.
//

import XCTest
import GrAIdient
import GrAITestsUtils

///
/// A class that will test a model with a structural hypothesis:
/// the model last layer is a LinearError1D layer, the model first layer is a Input1D.
/// 
class Input1DLinearError1DCase: XCTestCase, IOCase
{
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
        GrAI.Precision.float = true
        
        setOptimizerParams(params: &optimizerParams)
        optimizerParams.nbLoops = 3
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
        let lastLayer = model.layers.last as! LinearError1D
        let gt: [[Double]]
        if let groundTruthTmp = groundTruth
        {
            gt = groundTruthTmp
        }
        else
        {
            gt = buildData(dim1: getBatchSize(model), dim2: 1)
        }
        
        if GrAI.Opti.GPU
        {
            try! lastLayer.lossDerivativeGPU()
        }
        else
        {
            try! lastLayer.lossDerivativeCPU()
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
        let lastLayer = model.layers.last as! LinearError1D
        if GrAI.Opti.GPU
        {
            return Double(try! lastLayer.getLossGPU(
                groundTruth, batchSize: groundTruth.count, nbNeurons: 1
            ))
        }
        else
        {
            return try! lastLayer.getLossCPU(
                groundTruth, batchSize: groundTruth.count, nbNeurons: 1
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
        let lastLayer = model.layers.last as! LinearError1D
        return try! lastLayer.collectGradientsApprox(
            groundTruth, batchSize: groundTruth.count, nbNeurons: 1
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
    func setData(
        _ inputs: [[Float]]?,
        _ model: Model
    ) -> ([[Float]], Int)
    {
        let firstLayer = model.layers.first as! Input1D
        let ins: [[Float]]
        if let insTmp = inputs
        {
            ins = insTmp
        }
        else
        {
            ins = buildData(dim1: getBatchSize(model), dim2: 1)
        }
        
        if GrAI.Opti.GPU
        {
            try! firstLayer.setDataGPU(
                ins, batchSize: ins.count, nbNeurons: 1
            )
        }
        else
        {
            try! firstLayer.setDataCPU(
                ins, batchSize: ins.count, nbNeurons: 1
            )
        }
        return (ins, ins.count)
    }
    
    ///
    /// Copy a model and call the `initKernel` API.
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
    /// Copy a model in place: do not call the `initKernel` API.
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
    
    /// A list of functions that transform the model into another one.
    var transforms: [(Model) -> Model]
    {
        get {
            return [copy, copyInPlace]
        }
    }
}
