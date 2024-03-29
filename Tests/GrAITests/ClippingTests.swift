//
// ClippingTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 10/10/2022.
//

import GrAIdient
import GrAITestsUtils

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class ClippingTests: Input1DMSE1DCase
{
    private func _buildTrainer() -> NormTrainer
    {
        let trainer = NormTrainer(
            name: "Clipping",
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
        
        var layer: Layer1D = Input1D(nbNeurons: 1, params: params)
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 12,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        layer = MSE1D(layerPrev: layer, params: params)
    }
    
    func testClippingCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testClippingGPU() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}
