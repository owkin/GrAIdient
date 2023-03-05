//
// LayerSeqDirtyTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 05/03/2023.
//

import GrAIdient
import GrAITestsUtils

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
            layerPrev: layer, patch: 2, nbNeurons: 5,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        var firstLayer: LayerSeq = layerSeq
        var secondLayer: LayerSeq
        
        switch model
        {
        case "Sum":
            let otherLayer: LayerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true, params: params
            )
            secondLayer = SumSeq(
                layersPrev: [firstLayer, otherLayer],
                params: params
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
}
