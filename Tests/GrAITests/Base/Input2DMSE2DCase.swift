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
class Input2DMSE2DCase: MSE2DCase
{
    var height = 6
    var width = 6
    
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
    
    ///
    /// Run Gradient Checking test.
    ///
    /// The goal is to compare the gradients of weights that are computed through `backward`
    /// to an estimation that is being computed through `forwardGC`.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: GradTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.000001)
    {
        let model = trainer.model!
        let lastLayer = model.layers.last as! MSE1D
        let layersGraph = model.getGraph(lastLayer)
        
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    layersGraph: layersGraph,
                    setData: self.setData,
                    setLoss: self.setLoss,
                    getGradientsApprox: self.getGradientsApprox)
                {
                    (gradDiff: Double) in
                    if gradDiff > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Flow test.
    ///
    /// The goal is to compare the gradients of weights computed in the CPU execution context with
    /// the gradients of weights computed in the GPU execution context.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: FlowTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.000001)
    {
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    setData: self.setData,
                    setLoss: self.setLoss)
                {
                    (gradDiff: Double) in
                    if gradDiff > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Flow Reset test.
    ///
    /// The goal is to compare the gradients of weights computed in the CPU execution context with
    /// the gradients of weights computed in the GPU execution context.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: FlowResetTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.000001)
    {
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    setData: self.setData,
                    setLoss: self.setLoss)
                {
                    (gradDiff: Double) in
                    if gradDiff > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Flow Reverse test.
    ///
    /// The goal is to compare the gradients of weights computed in the CPU execution context with
    /// the gradients of weights computed in the GPU execution context.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: FlowReverseTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.000001)
    {
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    setData: self.setData,
                    setLoss: self.setLoss)
                {
                    (gradDiff: Double) in
                    if gradDiff > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Inference test.
    ///
    /// The goal is to compare the losses computed in the CPU execution context with
    /// the losses computed in the GPU execution context during the inference phase.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: InferenceTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.001)
    {
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    setData: self.setData,
                    setLoss: self.setLoss,
                    getLoss: self.getLoss)
                {
                    (lossDiff: Double) in
                    if lossDiff > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Loading test.
    ///
    /// The goal is to compare the losses computed in the CPU execution after havinng loaded the
    /// model from the disk and do the same in the GPU execution context.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: LoadTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.001)
    {
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    setData: self.setData,
                    setLoss: self.setLoss,
                    getLoss: self.getLoss)
                {
                    (diffCPU: Double, diffGPU: Double) in
                    if diffCPU > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                    if diffGPU > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Transform tests.
    ///
    /// The goal is to compare the losses computed in the CPU execution
    /// after transforming the model and do the same in the GPU execution context.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: TransformTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.001)
    {
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    transforms: [
                        self.copy, self.copyInPlace,
                        self.resize, self.resizeInPlace
                    ],
                    setData: self.setData,
                    setLoss: self.setLoss,
                    getLoss: self.getLoss)
                {
                    (diffCPU: Double, diffGPU: Double) in
                    if diffCPU > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                    if diffGPU > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Clipping test.
    ///
    /// The goal is to compare the norm of the gradients of the weights with a threshold.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///     - normClipping: The threshold above which gradients must be cut.
    ///
    func run(
        _ trainer: NormTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.001,
        normClipping: Double = 0.001)
    {
        optimizerParams.gradientClipping = true
        optimizerParams.normThreshold = normClipping
        trainer.optimizerParams = optimizerParams
        
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    setData: self.setData,
                    setLoss: self.setLoss)
                {
                    (normDiff: Double) throws in
                    if normDiff > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
}
