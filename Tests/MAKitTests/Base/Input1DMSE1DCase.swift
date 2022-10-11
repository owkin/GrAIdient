//
// Input1DMSE1DCase.swift
// MAKitTests
//
// Created by Jean-François Reboud on 10/10/2022.
//

import XCTest
import MAKit
import MAKitTestsUtils

/// A class that will test a model with a structural hypothesis:
/// the model last layer is a MSE1D layer, the model first layer is a Input1D.
class Input1DMSE1DCase: MSE1DCase
{
    ///
    /// A function to create/set data to the model.
    ///
    /// - Parameters:
    ///     - inputs: The data to set.
    ///     - model: The model.
    /// - Returns: (The data, the batch size).
    ///
    public func setData(
        _ inputs: [[Float]]?,
        _ model: Model) -> ([[Float]], Int)
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
        let lastLayer = model.layers.last as! MSE1D
        let layersGraph = model.getGraph(lastLayer)
        trainer.run(
            layersGraph: layersGraph,
            setData: setData,
            setLoss: setLoss,
            getGradientsApprox: getGradientsApprox)
        {
            (gradDiff: Double) in
            XCTAssert(gradDiff <= 0.000001)
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
        trainer.run(
            setData: setData,
            setLoss: setLoss)
        {
            (gradDiff: Double) in
            XCTAssert(gradDiff <= 0.000001)
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
        trainer.run(
            setData: setData,
            setLoss: setLoss)
        {
            (gradDiff: Double) in
            XCTAssert(gradDiff <= 0.000001)
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
        trainer.run(
            setData: setData,
            setLoss: setLoss)
        {
            (gradDiff: Double) in
            XCTAssert(gradDiff <= 0.000001)
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
        trainer.run(
            setData: setData,
            setLoss: setLoss,
            getLoss: getLoss)
        {
            (lossDiff: Double) in
            XCTAssert(lossDiff <= 0.001)
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
        trainer.run(
            setData: setData,
            setLoss: setLoss,
            getLoss: getLoss)
        {
            (diffCPU: Double, diffGPU: Double) in
            XCTAssert(diffCPU <= 0.001)
            XCTAssert(diffGPU <= 0.001)
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
        trainer.run(
            transform: copy,
            setData: setData,
            setLoss: setLoss,
            getLoss: getLoss)
        {
            (diffCPU: Double, diffGPU: Double) in
            XCTAssert(diffCPU <= 0.001)
            XCTAssert(diffGPU <= 0.001)
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
        trainer.run(
            transform: copyInPlace,
            setData: setData,
            setLoss: setLoss,
            getLoss: getLoss)
        {
            (diffCPU: Double, diffGPU: Double) in
            XCTAssert(diffCPU <= 0.001)
            XCTAssert(diffGPU <= 0.001)
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
        
        trainer.run(
            setData: setData,
            setLoss: setLoss)
        {
            (normDiff: Double) in
            XCTAssert(normDiff <= 0.000001)
        }
    }
}
