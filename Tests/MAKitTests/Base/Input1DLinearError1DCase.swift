//
// Input1DLinearError1DCase.swift
// MAKitTests
//
// Created by Jean-François Reboud on 10/10/2022.
//

import XCTest
import Retry
import MAKit
import MAKitTestsUtils

///
/// A class that will test a model with a structural hypothesis:
/// the model last layer is a LinearError1D layer, the model first layer is a Input1D.
/// 
class Input1DLinearError1DCase: LinearError1DCase
{
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
            ins = build1DData(dim1: getBatchSize(model), dim2: 1)
        }
        
        if MAKit.Opti.GPU
        {
            try! firstLayer.setDataGPU(ins)
        }
        else
        {
            try! firstLayer.setDataCPU(ins)
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
            deviceID: DEVICE_ID_DEFAULT
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
        let lastLayer = model.layers.last as! LinearError1D
        let layersGraph = model.getGraph(lastLayer)
        
        retry(max: NB_RETRY)
        {
            try trainer.run(
                layersGraph: layersGraph,
                setData: self.setData,
                setLoss: self.setLoss,
                getGradientsApprox: self.getGradientsApprox)
            {
                (gradDiff: Double) throws in
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
        retry(max: NB_RETRY)
        {
            try trainer.run(
                setData: self.setData,
                setLoss: self.setLoss)
            {
                (gradDiff: Double) throws in
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
        retry(max: NB_RETRY)
        {
            try trainer.run(
                setData: self.setData,
                setLoss: self.setLoss)
            {
                (gradDiff: Double) throws in
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
        retry(max: NB_RETRY)
        {
            try trainer.run(
                setData: self.setData,
                setLoss: self.setLoss)
            {
                (gradDiff: Double) throws in
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
        retry(max: NB_RETRY)
        {
            try trainer.run(
                setData: self.setData,
                setLoss: self.setLoss,
                getLoss: self.getLoss)
            {
                (lossDiff: Double) throws in
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
        retry(max: NB_RETRY)
        {
            try trainer.run(
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
    /// Run Copy test.
    ///
    /// The goal is to compare the losses computed in the CPU execution
    /// after copying the model and do the same in the GPU execution context.
    ///
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func runCopy(_ trainer: TransformTrainer)
    {
        retry(max: NB_RETRY)
        {
            try trainer.run(
                transform: self.copy,
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
    /// Run Copy In Place test.
    ///
    /// The goal is to compare the losses computed in the CPU execution
    /// after copying the model in place and do the same in the GPU execution context.
    ///
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func runCopyInPlace(_ trainer: TransformTrainer)
    {
        retry(max: NB_RETRY)
        {
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
    /// Run Clipping test.
    ///
    /// The goal is to compare the norm of the gradients of the weights with a threshold.
    ///
    /// - Parameter trainer: The testing pipeline to run.
    ///
    func run(_ trainer: NormTrainer)
    {
        let normClipping = 0.03
        optimizerParams.gradientClipping = true
        optimizerParams.normThreshold = normClipping
        trainer.optimizerParams = optimizerParams
        
        retry(max: NB_RETRY)
        {
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
