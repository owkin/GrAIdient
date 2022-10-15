//
// Input2DMSE1DCase.swift
// MAKitTests
//
// Created by Jean-François Reboud on 15/10/2022.
//

import XCTest
import MAKit
import MAKitTestsUtils

///
/// A class that will test a model with a structural hypothesis:
/// the model last layer is a MSE1D layer, the model first layer is a Input2D.
///
class Input2DMSE1DCase: MSE1DCase
{
    let height = 12
    let width = 12
     
    ///
    /// Create synthetic data.
    ///
    /// - Parameter nbElems: The number of elements of the data.
    /// - Returns: The created data.
    ///
    func build2DData<T: BinaryFloatingPoint>(_ nbElems: Int) -> [T]
    {
        var data = [T]()
        for _ in 0..<nbElems
        {
            data.append(T(Double.random(in: -1.0..<1.0)))
        }
        return data
    }
    
    ///
    /// A function to create/set data to the model.
    ///
    /// - Parameters:
    ///     - inputs: The data to set.
    ///     - model: The model.
    /// - Returns: (The data, the batch size).
    ///
    func setData(_ inputs: [Double]?, _ model: Model) -> ([Double], Int)
    {
        let firstLayer = model.layers.first as! Input2D
        let ins: [Double]
        if let insTmp = inputs
        {
            ins = insTmp
        }
        else
        {
            ins = build2DData(getBatchSize(model) * height * width)
        }
        
        if MAKit.Opti.GPU
        {
            try! firstLayer.setDataGPU(
                ins,
                batchSize: getBatchSize(model),
                format: .Neuron
            )
        }
        else
        {
            try! firstLayer.setDataCPU(
                ins,
                batchSize: getBatchSize(model),
                format: .Neuron
            )
        }
        return (ins, ins.count / (height * width))
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
            deviceID: DEVICE_ID_DEFAULT
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
            deviceID: DEVICE_ID_DEFAULT
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
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func run(_ trainer: GradTrainer)
    {
        let model = trainer.model!
        let lastLayer = model.layers.last as! MSE1D
        let layersGraph = model.getGraph(lastLayer)
        
        retryNumeric(nbRetry: NB_RETRY)
        {
            () throws in
            try trainer.run(
                layersGraph: layersGraph,
                setData: self.setData,
                setLoss: self.setLoss,
                getGradientsApprox: self.getGradientsApprox)
            {
                (gradDiff: Double) in
                if gradDiff > 0.000001
                {
                    throw TestError.Numeric
                }
            }
        }
    }
    
    ///
    /// Run Flow test.
    ///
    /// The goal is to compare the gradients of weights computed in the CPU execution context with
    /// the gradients of weights computed in the GPU execution context.
    ///
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func run(_ trainer: FlowTrainer)
    {
        retryNumeric(nbRetry: NB_RETRY)
        {
            () throws in
            try trainer.run(
                setData: self.setData,
                setLoss: self.setLoss)
            {
                (gradDiff: Double) in
                if gradDiff > 0.000001
                {
                    throw TestError.Numeric
                }
            }
        }
    }
    
    ///
    /// Run Flow Reset test.
    ///
    /// The goal is to compare the gradients of weights computed in the CPU execution context with
    /// the gradients of weights computed in the GPU execution context.
    ///
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func run(_ trainer: FlowResetTrainer)
    {
        retryNumeric(nbRetry: NB_RETRY)
        {
            () throws in
            try trainer.run(
                setData: self.setData,
                setLoss: self.setLoss)
            {
                (gradDiff: Double) in
                if gradDiff > 0.000001
                {
                    throw TestError.Numeric
                }
            }
        }
    }
    
    ///
    /// Run Flow Reverse test.
    ///
    /// The goal is to compare the gradients of weights computed in the CPU execution context with
    /// the gradients of weights computed in the GPU execution context.
    ///
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func run(_ trainer: FlowReverseTrainer)
    {
        retryNumeric(nbRetry: NB_RETRY)
        {
            () throws in
            try trainer.run(
                setData: self.setData,
                setLoss: self.setLoss)
            {
                (gradDiff: Double) in
                if gradDiff > 0.000001
                {
                    throw TestError.Numeric
                }
            }
        }
    }
    
    ///
    /// Run Inference test.
    ///
    /// The goal is to compare the losses computed in the CPU execution context with
    /// the losses computed in the GPU execution context during the inference phase.
    ///
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func run(_ trainer: InferenceTrainer)
    {
        retryNumeric(nbRetry: NB_RETRY)
        {
            () throws in
            try trainer.run(
                setData: self.setData,
                setLoss: self.setLoss,
                getLoss: self.getLoss)
            {
                (lossDiff: Double) in
                if lossDiff > 0.001
                {
                    throw TestError.Numeric
                }
            }
        }
    }
    
    ///
    /// Run Loading test.
    ///
    /// The goal is to compare the losses computed in the CPU execution after havinng loaded the
    /// model from the disk and do the same in the GPU execution context.
    ///
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func run(_ trainer: LoadTrainer)
    {
        retryNumeric(nbRetry: NB_RETRY)
        {
            () throws in
            try trainer.run(
                setData: self.setData,
                setLoss: self.setLoss,
                getLoss: self.getLoss)
            {
                (diffCPU: Double, diffGPU: Double) in
                if diffCPU > 0.001
                {
                    throw TestError.Numeric
                }
                if diffGPU > 0.001
                {
                    throw TestError.Numeric
                }
            }
        }
    }
    
    ///
    /// Run Copy test.
    ///
    /// The goal is to compare the losses computed in the CPU execution
    /// after copying the model and do the same in the GPU execution context.
    ///
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func runCopy(_ trainer: TransformTrainer)
    {
        retryNumeric(nbRetry: NB_RETRY)
        {
            () throws in
            try trainer.run(
                transform: self.copy,
                setData: self.setData,
                setLoss: self.setLoss,
                getLoss: self.getLoss)
            {
                (diffCPU: Double, diffGPU: Double) in
                if diffCPU > 0.001
                {
                    throw TestError.Numeric
                }
                if diffGPU > 0.001
                {
                    throw TestError.Numeric
                }
            }
        }
    }
    
    ///
    /// Run Copy In Place test.
    ///
    /// The goal is to compare the losses computed in the CPU execution
    /// after copying the model in place and do the same in the GPU execution context.
    ///
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func runCopyInPlace(_ trainer: TransformTrainer)
    {
        retryNumeric(nbRetry: NB_RETRY)
        {
            () throws in
            try trainer.run(
                transform: self.copyInPlace,
                setData: self.setData,
                setLoss: self.setLoss,
                getLoss: self.getLoss)
            {
                (diffCPU: Double, diffGPU: Double) throws in
                if diffCPU > 0.001
                {
                    throw TestError.Numeric
                }
                if diffGPU > 0.001
                {
                    throw TestError.Numeric
                }
            }
        }
    }
    
    ///
    /// Run Resize test.
    ///
    /// The goal is to compare the losses computed in the CPU execution
    /// after resizing the model and do the same in the GPU execution context.
    ///
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func runResize(_ trainer: TransformTrainer)
    {
        retryNumeric(nbRetry: NB_RETRY)
        {
            () throws in
            try trainer.run(
                transform: self.resize,
                setData: self.setData,
                setLoss: self.setLoss,
                getLoss: self.getLoss)
            {
                (diffCPU: Double, diffGPU: Double) in
                if diffCPU > 0.001
                {
                    throw TestError.Numeric
                }
                if diffGPU > 0.001
                {
                    throw TestError.Numeric
                }
            }
        }
    }
    
    ///
    /// Run Resize In Place test.
    ///
    /// The goal is to compare the losses computed in the CPU execution
    /// after resizing the model in place and do the same in the GPU execution context.
    ///
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func runResizeInPlace(_ trainer: TransformTrainer)
    {
        retryNumeric(nbRetry: NB_RETRY)
        {
            () throws in
            try trainer.run(
                transform: self.resizeInPlace,
                setData: self.setData,
                setLoss: self.setLoss,
                getLoss: self.getLoss)
            {
                (diffCPU: Double, diffGPU: Double) throws in
                if diffCPU > 0.001
                {
                    throw TestError.Numeric
                }
                if diffGPU > 0.001
                {
                    throw TestError.Numeric
                }
            }
        }
    }
    
    ///
    /// Run Clipping test.
    ///
    /// The goal is to compare the norm of the gradients of the weights with a threshold.
    ///
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func run(_ trainer: NormTrainer)
    {
        let normClipping = 0.001
        optimizerParams.gradientClipping = true
        optimizerParams.normThreshold = normClipping
        trainer.optimizerParams = optimizerParams
        
        retryNumeric(nbRetry: NB_RETRY)
        {
            () throws in
            try trainer.run(
                setData: self.setData,
                setLoss: self.setLoss)
            {
                (normDiff: Double) throws in
                if normDiff > 0.000001
                {
                    throw TestError.Numeric
                }
            }
        }
    }
}
