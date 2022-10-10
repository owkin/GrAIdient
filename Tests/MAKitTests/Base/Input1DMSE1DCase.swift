//
// Input1DMSE1DCase.swift
// MAKitTests
//
// Created by Jean-François Reboud on 10/10/2022.
//

import XCTest
import MAKit
import MAKitTestsUtils

class Input1DMSE1DCase: MSE1DCase
{
    public func setData(
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
    
    func copyInPlace(_ model: Model) -> Model
    {
        let modelNew = Model.copy(models: [model], inPlace: true)[0]
        modelNew.setupOptimizers(params: optimizerParams)
        modelNew.phase = .Inference
        return modelNew
    }
    
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
