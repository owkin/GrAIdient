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
            layerSeq = FullyConnectedPatch(
                layerPrev: layer, patch: 2, nbNeurons: 5,
                activation: SoftReLU.str, biases: true, params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        let head: Layer1D = AvgPoolSeq(layerPrev: layerSeq, params: params)
        
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
}
