//
// LayerSeqTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 27/02/2023.
//

import XCTest
import GrAIdient
import GrAITestsUtils

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class LayerSeqGradTests: Input2DMSE1DCase
{
    override func setUp()
    {
        super.setUp()
        
        optimizerParams.nbLoops = 2
        GrAI.Loop.gradientChecking = true
    }
    
    private func _buildTrainer(_ model: String) -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            _buildModel(model: model, context: context)
        }
        return trainer
    }
    
    private func _buildModel(model: String, context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        var layerSeq: LayerSeq
        switch model
        {
        case "FullyConnectedPatch":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            
        case "Sum":
            let otherLayer1: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            let otherLayer2: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! SumSeq(
                layersPrev: [layerSeq, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Concat1":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 2, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! Concat1Seq(
                layersPrev: [layerSeq, otherLayer],
                params: params
            )
            
        case "Concat2":
            let otherLayer1: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 3,
                activation: SoftReLU.str, biases: true, params: params
            )
            let otherLayer2: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 6,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 9,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = Concat2Seq(
                layersPrev: [layerSeq, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Constant12":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 2,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = Constant12Seq(
                sequence: 4, nbNeurons: 2, params: params
            )
            (layerSeq as! Constant12Seq).weightsCPU = [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
            ]
            
            layerSeq = try! SumSeq(
                layersPrev: [layerSeq, otherLayer], params: params
            )
            
        case "Constant2":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = Constant2Seq(
                sequence: 9, nbNeurons: 5, params: params
            )
            (layerSeq as! Constant2Seq).weightsCPU = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            layerSeq = try! SumSeq(
                layersPrev: [layerSeq, otherLayer], params: params
            )
            
        case "FullyConnectedSeq":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5 * 3,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 4 * 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            
        case "LayerNorm":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = LayerNormSeq(
                layerPrev: layerSeq, activation: nil, params: params
            )
            
        case "Query":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 6,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 6,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! QuerySeq(
                query: layerSeq, key: otherLayer, nbHeads: 2, params: params
            )
            
        case "QuerySelf":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 3 * 6,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! QuerySelfSeq(
                layerPrev: layerSeq,
                query: 0, key: 1, nbBlocksPrev: 3, nbHeads: 2,
                params: params
            )
            
        case "Softmax":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 15,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! SoftmaxSeq(
                layerPrev: layerSeq, nbHeads: 3, params: params
            )
            
        case "Value":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 6,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 6,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 2 * 4,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! ValueSeq(
                value: otherLayer, score: layerSeq, nbHeads: 2, params: params
            )
            
        case "ValueSelf":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 3 * 6,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 6,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 2 * 4,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! ValueSelfSeq(
                value: otherLayer, score: layerSeq,
                offset: 2, nbBlocksPrev: 3, nbHeads: 2, params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layerSeq, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
    }
    
    func testFullyConnectedPatchCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    func testFullyConnectedPatchGPU() throws
    {
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    func testFullyConnectedPatchSampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    func testSumCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    func testSumGPU() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    func testConcat1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Concat1")
        run(trainer)
    }
    
    func testConcat1GPU() throws
    {
        let trainer = _buildTrainer("Concat1")
        run(trainer)
    }
    
    func testConcat2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Concat2")
        run(trainer)
    }
    
    func testConcat2GPU() throws
    {
        let trainer = _buildTrainer("Concat2")
        run(trainer)
    }
    
    func testConstant12CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Constant12")
        run(trainer)
    }
    
    func testConstant12GPU() throws
    {
        let trainer = _buildTrainer("Constant12")
        run(trainer)
    }
    
    func testConstant2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    func testConstant2GPU() throws
    {
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    func testConstant2SampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    func testFullyConnectedSeqCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    func testFullyConnectedSeqGPU() throws
    {
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    func testFullyConnectedSeqSampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    func testLayerNormSeqCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("LayerNorm")
        run(trainer)
    }
    
    func testLayerNormSeqGPU() throws
    {
        let trainer = _buildTrainer("LayerNorm")
        run(trainer)
    }
    
    func testQuerySeqCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Query")
        run(trainer)
    }
    
    func testQuerySeqGPU() throws
    {
        let trainer = _buildTrainer("Query")
        run(trainer)
    }
    
    func testQuerySelfSeqCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("QuerySelf")
        run(trainer)
    }
    
    func testQuerySelfSeqGPU() throws
    {
        let trainer = _buildTrainer("QuerySelf")
        run(trainer)
    }
    
    func testSoftmaxSeqCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    func testSoftmaxSeqGPU() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    func testValueSeqCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Value")
        run(trainer)
    }
    
    func testValueSeqGPU() throws
    {
        let trainer = _buildTrainer("Value")
        run(trainer)
    }
    
    func testValueSelfSeqCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("ValueSelf")
        run(trainer)
    }
    
    func testValueSelfSeqGPU() throws
    {
        let trainer = _buildTrainer("ValueSelf")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class LayerSeqFlowTests: Input2DMSE1DCase
{
    private func _buildTrainer(_ model: String) -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    func buildModel(model: String, context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        var layerSeq: LayerSeq
        switch model
        {
        case "FullyConnectedPatch":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
           
        case "Sum":
            let otherLayer1: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            let otherLayer2: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! SumSeq(
                layersPrev: [layerSeq, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Concat1":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 2, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! Concat1Seq(
                layersPrev: [layerSeq, otherLayer],
                params: params
            )
            
        case "Concat2":
            let otherLayer1: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 3,
                activation: LeakyReLU.str, biases: true, params: params
            )
            let otherLayer2: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 6,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 9,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = Concat2Seq(
                layersPrev: [layerSeq, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Constant12":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 2,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = Constant12Seq(
                sequence: 9, nbNeurons: 2, params: params
            )
            (layerSeq as! Constant12Seq).weightsCPU = [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                17, 18, 19
            ]
            
            layerSeq = try! SumSeq(
                layersPrev: [layerSeq, otherLayer], params: params
            )
            
        case "Constant2":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = Constant2Seq(
                sequence: 9, nbNeurons: 5, params: params
            )
            (layerSeq as! Constant2Seq).weightsCPU = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            layerSeq = try! SumSeq(
                layersPrev: [layerSeq, otherLayer], params: params
            )
            
        case "FullyConnectedSeq":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 4,
                activation: LeakyReLU.str, biases: true, params: params
            )
            
        case "LayerNorm":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = LayerNormSeq(
                layerPrev: layerSeq, activation: nil, params: params
            )
            
        case "Query":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 6,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 6,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! QuerySeq(
                query: layerSeq, key: otherLayer, nbHeads: 2, params: params
            )
            
        case "QuerySelf":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 3 * 6,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = try! QuerySelfSeq(
                layerPrev: layerSeq,
                query: 0, key: 1, nbBlocksPrev: 3, nbHeads: 2,
                params: params
            )
            
        case "Softmax":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 15,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! SoftmaxSeq(
                layerPrev: layerSeq, nbHeads: 3, params: params
            )
            
        case "Value":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 6,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 6,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 2 * 4,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! ValueSeq(
                value: otherLayer, score: layerSeq, nbHeads: 2, params: params
            )
            
        case "ValueSelf":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 3 * 6,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 6,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 2 * 4,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! ValueSelfSeq(
                value: otherLayer, score: layerSeq,
                offset: 2, nbBlocksPrev: 3, nbHeads: 2, params: params
            )
            
        case "VQ":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = VQSeq(layerPrev: layerSeq, K: 5, params: params)
            (layerSeq as! VQSeq).beta = 0.25
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layerSeq, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
    }
    
    func testFullyConnectedPatch() throws
    {
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    func testFullyConnectedPatchSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    func testConcat1() throws
    {
        let trainer = _buildTrainer("Concat1")
        run(trainer)
    }
    
    func testConcat2() throws
    {
        let trainer = _buildTrainer("Concat2")
        run(trainer)
    }
    
    func testConstant12() throws
    {
        let trainer = _buildTrainer("Constant12")
        run(trainer)
    }
    
    func testConstant2() throws
    {
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    func testConstant2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    func testFullyConnectedSeq() throws
    {
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    func testFullyConnectedSeqSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    func testLayerNormSeq() throws
    {
        let trainer = _buildTrainer("LayerNorm")
        run(trainer)
    }
    
    func testQuerySeq() throws
    {
        let trainer = _buildTrainer("Query")
        run(trainer)
    }
    
    func testQuerySelfSeq() throws
    {
        let trainer = _buildTrainer("QuerySelf")
        run(trainer)
    }
    
    func testSoftmaxSeq() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    func testValueSeq() throws
    {
        let trainer = _buildTrainer("Value")
        run(trainer)
    }
    
    func testValueSelfSeq() throws
    {
        let trainer = _buildTrainer("ValueSelf")
        run(trainer)
    }
    
    func testVQ() throws
    {
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
    
    func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class LayerSeq4FlowTests: Input2DMSE1DCase
{
    private func _buildTrainer(_ model: String) -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    func buildModel(model: String, context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 4, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        var layerSeq: LayerSeq
        switch model
        {
        case "Sum":
            let otherLayer1: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 4 * 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            let otherLayer2: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 4 * 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 4 * 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! SumSeq(
                layersPrev: [layerSeq, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Concat1":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 4 * 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 2, nbNeurons: 4 * 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! Concat1Seq(
                layersPrev: [layerSeq, otherLayer],
                params: params
            )
            
        case "Constant12":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 4,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = Constant12Seq(
                sequence: 4, nbNeurons: 4, params: params
            )
            (layerSeq as! Constant12Seq).weightsCPU = [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0
            ]
            
            layerSeq = try! SumSeq(
                layersPrev: [layerSeq, otherLayer], params: params
            )
            
        case "Constant2":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 4 * 2,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = Constant2Seq(
                sequence: 9, nbNeurons: 4 * 2, params: params
            )
            (layerSeq as! Constant2Seq).weightsCPU = [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
            ]
            
            layerSeq = try! SumSeq(
                layersPrev: [layerSeq, otherLayer], params: params
            )
            
        case "FullyConnectedSeq":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 4 * 3,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 4 * 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            
        case "LayerNorm":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 4 * 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = LayerNormSeq(
                layerPrev: layerSeq, activation: nil, params: params
            )
            
        case "Query":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 4 * 2 * 3,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 4 * 2 * 3,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! QuerySeq(
                query: layerSeq, key: otherLayer, nbHeads: 2, params: params
            )
            
        case "QuerySelf":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 3 * 4 * 2 * 3,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! QuerySelfSeq(
                layerPrev: layerSeq,
                query: 0, key: 1, nbBlocksPrev: 3, nbHeads: 2,
                params: params
            )
            
        case "Softmax":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 4 * 3 * 3,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! SoftmaxSeq(
                layerPrev: layerSeq, nbHeads: 3, params: params
            )
            
        case "Value":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 4 * 2 * 3,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 4 * 3,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 4 * 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! ValueSeq(
                value: otherLayer, score: layerSeq, nbHeads: 2, params: params
            )
            
        case "ValueSelf":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 3 * 4 * 2 * 3,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 4 * 3,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 4 * 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = try! ValueSelfSeq(
                value: otherLayer, score: layerSeq,
                offset: 2, nbBlocksPrev: 3, nbHeads: 2, params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layerSeq, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
    }
    
    func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    func testConcat1() throws
    {
        let trainer = _buildTrainer("Concat1")
        run(trainer)
    }
    
    func testConstant12() throws
    {
        let trainer = _buildTrainer("Constant12")
        run(trainer)
    }
    
    func testConstant2() throws
    {
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    func testFullyConnectedSeq() throws
    {
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    func testLayerNormSeq() throws
    {
        let trainer = _buildTrainer("LayerNorm")
        run(trainer)
    }
    
    func testQuerySeq() throws
    {
        let trainer = _buildTrainer("Query")
        run(trainer)
    }
    
    func testQuerySelfSeq() throws
    {
        let trainer = _buildTrainer("QuerySelf")
        run(trainer)
    }
    
    func testSoftmaxSeq() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    func testValueSeq() throws
    {
        let trainer = _buildTrainer("Value")
        run(trainer)
    }
    
    func testValueSelfSeq() throws
    {
        let trainer = _buildTrainer("ValueSelf")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class LayerSeqFlowResetTests: LayerSeqFlowTests
{
    override func setUp()
    {
        super.setUp()
        
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
    }
    
    private func _buildTrainer(_ model: String) -> FlowResetTrainer
    {
        let trainer = FlowResetTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testFullyConnectedPatch() throws
    {
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    override func testFullyConnectedPatchSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    override func testConcat1() throws
    {
        let trainer = _buildTrainer("Concat1")
        run(trainer)
    }
    
    override func testConcat2() throws
    {
        let trainer = _buildTrainer("Concat2")
        run(trainer)
    }
    
    override func testConstant12() throws
    {
        let trainer = _buildTrainer("Constant12")
        run(trainer)
    }
    
    override func testConstant2() throws
    {
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    override func testConstant2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    override func testFullyConnectedSeq() throws
    {
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    override func testFullyConnectedSeqSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    override func testLayerNormSeq() throws
    {
        let trainer = _buildTrainer("LayerNorm")
        run(trainer)
    }
    
    override func testQuerySeq() throws
    {
        let trainer = _buildTrainer("Query")
        run(trainer)
    }
    
    override func testQuerySelfSeq() throws
    {
        let trainer = _buildTrainer("QuerySelf")
        run(trainer)
    }
    
    override func testSoftmaxSeq() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    override func testValueSeq() throws
    {
        let trainer = _buildTrainer("Value")
        run(trainer)
    }
    
    override func testValueSelfSeq() throws
    {
        let trainer = _buildTrainer("ValueSelf")
        run(trainer)
    }
    
    override func testVQ() throws
    {
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
    
    override func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class LayerSeqFlowReverseTests: LayerSeqFlowTests
{
    override func setUp()
    {
        super.setUp()
        
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
    }
    
    private func _buildTrainer(_ model: String) -> FlowReverseTrainer
    {
        let trainer = FlowReverseTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testFullyConnectedPatch() throws
    {
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    override func testFullyConnectedPatchSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    override func testConcat1() throws
    {
        let trainer = _buildTrainer("Concat1")
        run(trainer)
    }
    
    override func testConcat2() throws
    {
        let trainer = _buildTrainer("Concat2")
        run(trainer)
    }
    
    override func testConstant12() throws
    {
        let trainer = _buildTrainer("Constant12")
        run(trainer)
    }
    
    override func testConstant2() throws
    {
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    override func testConstant2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    override func testFullyConnectedSeq() throws
    {
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    override func testFullyConnectedSeqSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    override func testLayerNormSeq() throws
    {
        /*let trainer = _buildTrainer("LayerNorm")
        run(trainer, nbRetry: 5)*/
    }
    
    override func testQuerySeq() throws
    {
        let trainer = _buildTrainer("Query")
        run(trainer)
    }
    
    override func testQuerySelfSeq() throws
    {
        let trainer = _buildTrainer("QuerySelf")
        run(trainer)
    }
    
    override func testSoftmaxSeq() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    override func testValueSeq() throws
    {
        let trainer = _buildTrainer("Value")
        run(trainer)
    }
    
    override func testValueSelfSeq() throws
    {
        let trainer = _buildTrainer("ValueSelf")
        run(trainer)
    }
    
    override func testVQ() throws
    {
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
    
    override func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class LayerSeqFlowAccumulateTests: Input2DMSE1DCase
{
    private func _buildTrainer(_ model: String) -> FlowTrainer
    {
        let trainer = FlowAccumulateTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    func buildModel(model: String, context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        var layerSeq: LayerSeq
        switch model
        {
        case "FullyConnectedPatch":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            
        case "Constant12":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 2,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = Constant12Seq(
                sequence: 4, nbNeurons: 2, params: params
            )
            (layerSeq as! Constant12Seq).weightsCPU = [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
            ]
            
            layerSeq = try! SumSeq(
                layersPrev: [layerSeq, otherLayer], params: params
            )
            
        case "Constant2":
            let otherLayer: LayerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = Constant2Seq(
                sequence: 9, nbNeurons: 5, params: params
            )
            (layerSeq as! Constant2Seq).weightsCPU = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            layerSeq = try! SumSeq(
                layersPrev: [layerSeq, otherLayer], params: params
            )
            
        case "FullyConnectedSeq":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 4,
                activation: LeakyReLU.str, biases: true, params: params
            )
            
        case "LayerNorm":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = LayerNormSeq(
                layerPrev: layerSeq, activation: nil, params: params
            )
            
        case "VQ":
            layerSeq = try! FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = VQSeq(layerPrev: layerSeq, K: 5, params: params)
            (layerSeq as! VQSeq).beta = 0.25
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layerSeq, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
    }
    
    func testFullyConnectedPatch() throws
    {
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    func testFullyConnectedPatchSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    func testConstant12() throws
    {
        let trainer = _buildTrainer("Constant12")
        run(trainer)
    }
    
    func testConstant2() throws
    {
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    func testConstant2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    func testFullyConnectedSeq() throws
    {
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    func testFullyConnectedSeqSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    func testLayerNormSeq() throws
    {
        let trainer = _buildTrainer("LayerNorm")
        run(trainer)
    }
    
    func testVQ() throws
    {
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
    
    func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class LayerSeqInferenceTests: LayerSeqFlowTests
{
    private func _buildTrainer(_ model: String) -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testFullyConnectedPatch() throws
    {
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    override func testFullyConnectedPatchSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    override func testConcat1() throws
    {
        let trainer = _buildTrainer("Concat1")
        run(trainer)
    }
    
    override func testConcat2() throws
    {
        let trainer = _buildTrainer("Concat2")
        run(trainer)
    }
    
    override func testConstant12() throws
    {
        let trainer = _buildTrainer("Constant12")
        run(trainer)
    }
    
    override func testConstant2() throws
    {
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    override func testConstant2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    override func testFullyConnectedSeq() throws
    {
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    override func testFullyConnectedSeqSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    override func testLayerNormSeq() throws
    {
        let trainer = _buildTrainer("LayerNorm")
        run(trainer)
    }
    
    override func testQuerySeq() throws
    {
        let trainer = _buildTrainer("Query")
        run(trainer)
    }
    
    override func testQuerySelfSeq() throws
    {
        let trainer = _buildTrainer("QuerySelf")
        run(trainer)
    }
    
    override func testSoftmaxSeq() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    override func testValueSeq() throws
    {
        let trainer = _buildTrainer("Value")
        run(trainer)
    }
    
    override func testValueSelfSeq() throws
    {
        let trainer = _buildTrainer("ValueSelf")
        run(trainer)
    }
    
    override func testVQ() throws
    {
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
    
    override func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class LayerSeqLoadTests: LayerSeqFlowTests
{
    private func _buildTrainer(_ model: String) -> LoadTrainer
    {
        let trainer = LoadTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testFullyConnectedPatch() throws
    {
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    override func testFullyConnectedPatchSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    override func testConcat1() throws
    {
        let trainer = _buildTrainer("Concat1")
        run(trainer)
    }
    
    override func testConcat2() throws
    {
        let trainer = _buildTrainer("Concat2")
        run(trainer)
    }
    
    override func testConstant12() throws
    {
        let trainer = _buildTrainer("Constant12")
        run(trainer)
    }
    
    override func testConstant2() throws
    {
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    override func testConstant2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    override func testFullyConnectedSeq() throws
    {
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    override func testFullyConnectedSeqSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    override func testLayerNormSeq() throws
    {
        let trainer = _buildTrainer("LayerNorm")
        run(trainer)
    }
    
    override func testQuerySeq() throws
    {
        let trainer = _buildTrainer("Query")
        run(trainer)
    }
    
    override func testQuerySelfSeq() throws
    {
        let trainer = _buildTrainer("QuerySelf")
        run(trainer)
    }
    
    override func testSoftmaxSeq() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    override func testValueSeq() throws
    {
        let trainer = _buildTrainer("Value")
        run(trainer)
    }
    
    override func testValueSelfSeq() throws
    {
        let trainer = _buildTrainer("ValueSelf")
        run(trainer)
    }
    
    override func testVQ() throws
    {
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
    
    override func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class LayerSeqTransformTests: LayerSeqFlowTests
{
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
                    transforms: [self.copy, self.copyInPlace],
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
    
    private func _buildTrainer(_ model: String) -> TransformTrainer
    {
        let trainer = TransformTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testFullyConnectedPatch() throws
    {
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    override func testFullyConnectedPatchSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedPatch")
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    override func testConcat1() throws
    {
        let trainer = _buildTrainer("Concat1")
        run(trainer)
    }
    
    override func testConcat2() throws
    {
        let trainer = _buildTrainer("Concat2")
        run(trainer)
    }
    
    override func testConstant12() throws
    {
        let trainer = _buildTrainer("Constant12")
        run(trainer)
    }
    
    override func testConstant2() throws
    {
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    override func testConstant2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant2")
        run(trainer)
    }
    
    override func testFullyConnectedSeq() throws
    {
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    override func testFullyConnectedSeqSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    override func testLayerNormSeq() throws
    {
        let trainer = _buildTrainer("LayerNorm")
        run(trainer)
    }
    
    override func testQuerySeq() throws
    {
        let trainer = _buildTrainer("Query")
        run(trainer)
    }
    
    override func testQuerySelfSeq() throws
    {
        let trainer = _buildTrainer("QuerySelf")
        run(trainer)
    }
    
    override func testSoftmaxSeq() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    override func testValueSeq() throws
    {
        let trainer = _buildTrainer("Value")
        run(trainer)
    }
    
    override func testValueSelfSeq() throws
    {
        let trainer = _buildTrainer("ValueSelf")
        run(trainer)
    }
    
    override func testVQ() throws
    {
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
    
    override func testVQSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("VQ")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class SelectSeqGradTests: Input2DMSE1DCase
{
    private func _buildTrainer() -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            _buildModel(context: context)
        }
        return trainer
    }
    
    private func _buildModel(context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        let layerSeq: LayerSeq = try! FullyConnectedPatch(
            layerPrev: layer, patch: width / 3, nbNeurons: 5,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        var head: Layer1D = SelectSeq(
            layerPrev: layerSeq,
            targetSeq: 3,
            params: params
        )
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
    }
    
    func testSelectCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testSelectGPU() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class SelectSeqFlowTests: Input2DMSE1DCase
{
    private func _buildTrainer() -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    func buildModel(context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        let layerSeq: LayerSeq = try! FullyConnectedPatch(
            layerPrev: layer, patch: width / 3, nbNeurons: 5,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        var head: Layer1D = SelectSeq(
            layerPrev: layerSeq,
            targetSeq: 3,
            params: params
        )
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
    }
    
    func testSelect() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class SelectSeqFlowResetTests: SelectSeqFlowTests
{
    override func setUp()
    {
        super.setUp()
        
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
    }
    
    private func _buildTrainer() -> FlowResetTrainer
    {
        let trainer = FlowResetTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testSelect() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class SelectSeqFlowReverseTests: SelectSeqFlowTests
{
    override func setUp()
    {
        super.setUp()
        
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
    }
    
    private func _buildTrainer() -> FlowReverseTrainer
    {
        let trainer = FlowReverseTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testSelect() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class SelectSeqInferenceTests: SelectSeqFlowTests
{
    private func _buildTrainer() -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testSelect() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class SelectSeqLoadTests: SelectSeqFlowTests
{
    private func _buildTrainer() -> LoadTrainer
    {
        let trainer = LoadTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testSelect() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class SelectSeqTransformTests: SelectSeqFlowTests
{
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
                    transforms: [self.copy, self.copyInPlace],
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
    
    private func _buildTrainer() -> TransformTrainer
    {
        let trainer = TransformTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testSelect() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class VQSeqFlowTests: Input2DVQSeqCase
{
    private func _buildTrainer() -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    func buildModel(context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        let layerSeq: LayerSeq = try! FullyConnectedPatch(
            layerPrev: layer, patch: width / 3, nbNeurons: 5,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        _ = VQSeq(layerPrev: layerSeq, K: 5, params: params)
    }
    
    func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class VQSeqFlowResetTests: VQSeqFlowTests
{
    override func setUp()
    {
        super.setUp()
        
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
    }
    
    private func _buildTrainer() -> FlowResetTrainer
    {
        let trainer = FlowResetTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class VQSeqFlowReverseTests: VQSeqFlowTests
{
    override func setUp()
    {
        super.setUp()
        
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
    }
    
    private func _buildTrainer() -> FlowReverseTrainer
    {
        let trainer = FlowReverseTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class VQSeqInferenceTests: VQSeqFlowTests
{
    private func _buildTrainer() -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class VQSeqLoadTests: VQSeqFlowTests
{
    private func _buildTrainer() -> LoadTrainer
    {
        let trainer = LoadTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class VQSeqTransformTests: VQSeqFlowTests
{
    private func _buildTrainer() -> TransformTrainer
    {
        let trainer = TransformTrainer(
            name: "LayerSeq",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// Tests for the VQGradSeq layer.
class VQGradSeqTests: XCTestCase
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
        optimizerParams.nbLoops = 3
    }
    
    ///
    /// Build the two branches of the model.
    ///
    /// - Returns:
    ///     (frist branch, last branch of the model).
    ///
    func buildModel() -> (Model, Model)
    {
        var context = ModelContext(name: "MainBranch", curID: 0)
        var params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        let layerSeq: LayerSeq = try! FullyConnectedPatch(
            layerPrev: layer, patch: width / 3, nbNeurons: 6,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layerSeq, params: params)
        
        head = try! FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
        
        let mainBranch = Model(model: context.model, modelsPrev: [])
        
        context = ModelContext(name: "VQBranch", models: [mainBranch])
        params = GrAI.Model.Params(context: context)
        
        _ = VQGradSeq(layerPrev: layerSeq, K: 5, params: params)
        
        let vqBranch = Model(model: context.model, modelsPrev: [mainBranch])
        
        return (mainBranch, vqBranch)
    }
    
    ///
    /// Get the current batch size of data.
    ///
    /// This function allows to simulate the fact that the batch size of data may be smalling during the
    /// last iteration of the training.
    ///
    /// - Parameter model: The model.
    /// - Returns: The batch size of data.
    ///
    func getBatchSize(_ model: Model) -> Int
    {
        if model.optimizerParams.step == model.optimizerParams.nbLoops-1
        {
            return batchSize / 2
        }
        else
        {
            return batchSize
        }
    }
    
    ///
    /// Create synthetic data.
    ///
    /// - Parameters:
    ///     - dim1: The first dimension of the data.
    ///     - dim2: The second dimension of the data.
    /// - Returns: The created data.
    ///
    func buildData<T: BinaryFloatingPoint>(dim1: Int, dim2: Int) -> [[T]]
    {
        var data = [[T]]()
        for _ in 0..<dim1
        {
            var data1 = [T]()
            for _ in 0..<dim2
            {
                data1.append(T(Double.random(in: -1.0..<1.0)))
            }
            data.append(data1)
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
                ins.reduce([], +),
                batchSize: ins.count,
                nbChannels: 1, height: height, width: width,
                format: .Neuron
            )
        }
        else
        {
            try! firstLayer.setDataCPU(
                ins.reduce([], +),
                batchSize: ins.count,
                nbChannels: 1, height: height, width: width,
                format: .Neuron
            )
        }
        return (ins, ins.count)
    }
    
    func testInference()
    {
        let (mainCPU, vqCPU) = buildModel()
        let (mainGPU, vqGPU) = buildModel()
        
        GrAI.Opti.CPU = true
        randomSelectWeightsInitializationScheme(model: mainCPU)
        randomSelectWeightsInitializationScheme(model: vqCPU)
        
        mainCPU.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        vqCPU.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        mainGPU.weights = mainCPU.weights
        vqGPU.weights = vqCPU.weights
        
        GrAI.Opti.GPU = true
        mainGPU.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        vqGPU.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let lastLayerCPU = mainCPU.layers.last as! MSE1D
        let vqLayerCPU = vqCPU.layers.last as! VQGradSeq
        let lastLayerGPU = mainGPU.layers.last as! MSE1D
        let vqLayerGPU = vqGPU.layers.last as! VQGradSeq
        
        lastLayerCPU.coeff = -1.0
        lastLayerGPU.coeff = -1.0
        vqLayerCPU.magnitudeCoeff = 1.1
        vqLayerGPU.magnitudeCoeff = 1.1
        
        var numLoop = 0
        while numLoop < optimizerParams.nbLoops
        {
            GrAI.Opti.CPU = true
            
            let (inputs, batchSize) = setData(nil, mainCPU)
            mainCPU.updateKernel(batchSize: batchSize)
            vqCPU.updateKernel(batchSize: batchSize)
            
            try! mainCPU.forward()
            try! lastLayerCPU.lossDerivativeCPU(
                [[Double]](repeating: [0.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainCPU.backward()
            try! mainCPU.update()
            
            try! vqCPU.forward()
            try! vqLayerCPU.lossDerivativeCPU()
            let lossCPU: Double = vqLayerCPU.getLossCPU()
            try! vqCPU.update()
            
            GrAI.Opti.GPU = true
            
            _ = setData(inputs, mainGPU)
            mainGPU.updateKernel(batchSize: batchSize)
            vqGPU.updateKernel(batchSize: batchSize)
            
            try! mainGPU.forward()
            try! lastLayerGPU.lossDerivativeGPU(
                [[Double]](repeating: [0.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainGPU.backward()
            try! mainGPU.update()
            
            try! vqGPU.forward()
            try! vqLayerGPU.lossDerivativeGPU()
            let lossGPU: Double = try! vqLayerGPU.getLossGPU()
            try! vqGPU.update()
            
            let diff = (lossGPU - lossCPU) * (lossGPU - lossCPU) /
                       (lossCPU * lossCPU + lossGPU * lossGPU)
            XCTAssert(diff < 0.001)
            
            mainCPU.incStep()
            vqCPU.incStep()
            mainGPU.incStep()
            vqGPU.incStep()
            numLoop += 1
        }
    }
    
    func testLoad()
    {
        GrAI.Opti.GPU = true
        var (mainBranch, vqBranch) = buildModel()
        
        randomSelectWeightsInitializationScheme(model: mainBranch)
        randomSelectWeightsInitializationScheme(model: vqBranch)
        
        mainBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        vqBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let folderURL = FileManager.default.temporaryDirectory
        let mainPath =
            folderURL.appendingPathComponent("testMain.plist").path
        let vqPath =
            folderURL.appendingPathComponent("testVQ.plist").path
        
        let encoder = PropertyListEncoder()
    
        var data = try! encoder.encode(mainBranch)
        try! data.write(to: URL(fileURLWithPath: mainPath))
        
        data = try! encoder.encode(vqBranch)
        try! data.write(to: URL(fileURLWithPath: vqPath))
        
        data = try! Data(contentsOf: URL(fileURLWithPath: mainPath))
        let mainBase = try! PropertyListDecoder().decode(
            BaseModel.self, from: data
        )
        data = try! Data(contentsOf: URL(fileURLWithPath: vqPath))
        let vqBase = try! PropertyListDecoder().decode(
            BaseModel.self, from: data
        )
        
        mainBranch = Model(model: mainBase, modelsPrev: [])
        vqBranch = Model(model: vqBase, modelsPrev: [mainBranch])
        
        mainBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        vqBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let lastLayer = mainBranch.layers.last as! MSE1D
        let vqLayer = vqBranch.layers.last as! VQGradSeq
        
        lastLayer.coeff = -1.0
        vqLayer.magnitudeCoeff = 1.1
        
        var numLoop = 0
        while numLoop < optimizerParams.nbLoops
        {
            let (_, batchSize) = setData(nil, mainBranch)
            mainBranch.updateKernel(batchSize: batchSize)
            vqBranch.updateKernel(batchSize: batchSize)
            
            try! mainBranch.forward()
            try! lastLayer.lossDerivativeGPU(
                [[Double]](repeating: [0.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainBranch.backward()
            try! mainBranch.update()
            
            try! vqBranch.forward()
            try! vqLayer.lossDerivativeGPU()
            let lossVal: Double = try! vqLayer.getLossGPU()
            try! vqBranch.update()
            
            print(lossVal)
            
            mainBranch.incStep()
            vqBranch.incStep()
            numLoop += 1
        }
    }
    
    func testTransform()
    {
        GrAI.Opti.GPU = true
        var (mainBranch, vqBranch) = buildModel()
        
        randomSelectWeightsInitializationScheme(model: mainBranch)
        randomSelectWeightsInitializationScheme(model: vqBranch)
        
        mainBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        vqBranch.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let branches = Model.copy(
            models: [mainBranch, vqBranch],
            inPlace: true
        )
        mainBranch = branches[0]
        vqBranch = branches[1]
        
        mainBranch.setupOptimizers(params: optimizerParams)
        vqBranch.setupOptimizers(params: optimizerParams)
        mainBranch.phase = .Inference
        vqBranch.phase = .Inference
        
        let lastLayer = mainBranch.layers.last as! MSE1D
        let vqLayer = vqBranch.layers.last as! VQGradSeq
        
        lastLayer.coeff = -1.0
        vqLayer.magnitudeCoeff = 1.1
        
        var numLoop = 0
        while numLoop < optimizerParams.nbLoops
        {
            let (_, batchSize) = setData(nil, mainBranch)
            mainBranch.updateKernel(batchSize: batchSize)
            vqBranch.updateKernel(batchSize: batchSize)
            
            try! mainBranch.forward()
            try! lastLayer.lossDerivativeGPU(
                [[Double]](repeating: [0.0], count: batchSize),
                batchSize: batchSize,
                nbNeurons: 1
            )
            try! mainBranch.backward()
            try! mainBranch.update()
            
            try! vqBranch.forward()
            try! vqLayer.lossDerivativeGPU()
            let lossVal: Double = try! vqLayer.getLossGPU()
            try! vqBranch.update()
            
            print(lossVal)
            
            mainBranch.incStep()
            vqBranch.incStep()
            numLoop += 1
        }
    }
}
