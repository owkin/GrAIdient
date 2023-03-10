//
// LayerSeqDirtyTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 05/03/2023.
//

import GrAIdient
import GrAITestsUtils

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class LayerSeqDirtyGradTests: Input2DMSE1DCase
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
        
        let layerSeq = FullyConnectedPatch(
            layerPrev: layer, patch: width / 3, nbNeurons: 5,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        var firstLayer: LayerSeq = layerSeq
        var secondLayer: LayerSeq
        
        switch model
        {
        case "FullyConnectedSeq":
            secondLayer = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            
        case "LayerNorm":
            secondLayer = LayerNormSeq(
                layerPrev: layerSeq, activation: nil, params: params
            )
            
        case "Softmax":
            secondLayer = SoftmaxSeq(
                layerPrev: layerSeq, params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        firstLayer = SumSeq(
            layersPrev: [firstLayer, secondLayer], params: params
        )
        
        var head: Layer1D = AvgPoolSeq(layerPrev: firstLayer, params: params)
        
        head = FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
    }
    
    func testFLCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    func testFLGPU() throws
    {
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
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class LayerSeqDirtyFlowTests: Input2DMSE1DCase
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
        
        let layerSeq = FullyConnectedPatch(
            layerPrev: layer, patch: width / 3, nbNeurons: 5,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        var firstLayer: LayerSeq = layerSeq
        var secondLayer: LayerSeq
        
        switch model
        {
        case "Sum":
            let otherLayer: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            secondLayer = SumSeq(
                layersPrev: [firstLayer, otherLayer],
                params: params
            )
            
        case "Concat2":
            let otherLayer: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 3,
                activation: LeakyReLU.str, biases: true, params: params
            )
            secondLayer = Concat2Seq(
                layersPrev: [firstLayer, otherLayer],
                params: params
            )
            secondLayer = FullyConnectedSeq(
                layerPrev: secondLayer, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            
        case "FullyConnectedSeq":
            secondLayer = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            
        case "LayerNorm":
            secondLayer = LayerNormSeq(
                layerPrev: layerSeq, activation: nil, params: params
            )
            
        case "QueryQuery":
            let otherLayer: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            secondLayer = QuerySeq(
                query: layerSeq, key: otherLayer, params: params
            )
            secondLayer = FullyConnectedSeq(
                layerPrev: secondLayer, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            
        case "QueryKey":
            let otherLayer: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: width / 3, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            secondLayer = QuerySeq(
                query: otherLayer, key: layerSeq, params: params
            )
            secondLayer = FullyConnectedSeq(
                layerPrev: secondLayer, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            
        case "Softmax":
            secondLayer = SoftmaxSeq(
                layerPrev: layerSeq, params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        firstLayer = SumSeq(
            layersPrev: [firstLayer, secondLayer], params: params
        )
        
        var head: Layer1D = AvgPoolSeq(layerPrev: firstLayer, params: params)
        
        head = FullyConnected(
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
    
    func testConcat2() throws
    {
        let trainer = _buildTrainer("Concat2")
        run(trainer)
    }
    
    func testFLSeq() throws
    {
        let trainer = _buildTrainer("FullyConnectedSeq")
        run(trainer)
    }
    
    func testLayerNormSeq() throws
    {
        let trainer = _buildTrainer("LayerNorm")
        run(trainer)
    }
    
    func testQueryQuerySeq() throws
    {
        let trainer = _buildTrainer("QueryQuery")
        run(trainer)
    }
    
    func testQueryKeySeq() throws
    {
        let trainer = _buildTrainer("QueryKey")
        run(trainer)
    }
    
    func testSoftmaxSeq() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
}
