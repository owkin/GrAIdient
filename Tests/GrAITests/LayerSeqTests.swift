//
// LayerSeqTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 27/02/2023.
//

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
            layerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            
        case "Sum":
            let otherLayer1: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            let otherLayer2: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = SumSeq(
                layersPrev: [layerSeq, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Concat1":
            let otherLayer1: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = Concat1Seq(
                layersPrev: [layerSeq, otherLayer1],
                params: params
            )
            
        case "Concat2":
            let otherLayer1: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 3,
                activation: SoftReLU.str, biases: true, params: params
            )
            let otherLayer2: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 6,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 9,
                activation: SoftReLU.str, biases: true, params: params
            )
            layerSeq = Concat2Seq(
                layersPrev: [layerSeq, otherLayer1, otherLayer2],
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layerSeq, params: params)
        
        head = FullyConnected(
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
            layerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
           
        case "Sum":
            let otherLayer1: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            let otherLayer2: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = SumSeq(
                layersPrev: [layerSeq, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Concat1":
            let otherLayer1: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = Concat1Seq(
                layersPrev: [layerSeq, otherLayer1],
                params: params
            )
            
        case "Concat2":
            let otherLayer1: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 3,
                activation: LeakyReLU.str, biases: true, params: params
            )
            let otherLayer2: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 6,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 9,
                activation: LeakyReLU.str, biases: true, params: params
            )
            layerSeq = Concat2Seq(
                layersPrev: [layerSeq, otherLayer1, otherLayer2],
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        var head: Layer1D = AvgPoolSeq(layerPrev: layerSeq, params: params)
        
        head = FullyConnected(
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
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class LayerSeqTransformTests: LayerSeqFlowTests
{
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
}
