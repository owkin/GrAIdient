//
// Activation1DTests.swift
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
class Activation1DGradTests: Input1DMSE1DCase
{
    override func setUp()
    {
        super.setUp()
        GrAI.Loop.gradientChecking = true
    }
    
    private func _buildTrainer(model: String, activation: String?)
        -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "Activation1D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            _buildModel(model: model, activation: activation, context: context)
        }
        return trainer
    }
    
    private func _buildModel(
        model: String,
        activation: String?,
        context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer1D = Input1D(nbNeurons: 1, params: params)
        
        layer = FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: SoftReLU.str, biases: true,
            params: params
        )
        
        switch model
        {
        case "FullyConnected":
            layer = FullyConnected(
                layerPrev: layer, nbNeurons: 12,
                activation: activation, biases: true,
                params: params
            )
            
        case "Activation":
            layer = Activation1D(
                layerPrev: layer,
                activation: activation!,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        layer = FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: SoftReLU.str, biases: true,
            params: params
        )
        
        layer = MSE1D(layerPrev: layer, params: params)
    }
    
    func testFLNoActivationCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: nil
        )
        run(trainer)
    }
    
    func testFLNoActivationGPU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: nil
        )
        run(trainer)
    }
    
    func testFLReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: ReLU.str
        )
        run(trainer)
    }
    
    func testFLReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: ReLU.str
        )
        run(trainer)
    }
    
    func testFLLeakyReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: LeakyReLU.str
        )
        run(trainer)
    }
    
    func testFLLeakyReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: LeakyReLU.str
        )
        run(trainer)
    }
    
    func testFLSoftReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: SoftReLU.str
        )
        run(trainer)
    }
    
    func testFLSoftReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: SoftReLU.str
        )
        run(trainer)
    }
    
    func testFLSigmoidCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: Sigmoid.str
        )
        run(trainer)
    }
    
    func testFLSigmoidGPU() throws
    {
        let trainer = _buildTrainer(
            model: "FullyConnected", activation: Sigmoid.str
        )
        run(trainer)
    }
    
    func testReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: ReLU.str
        )
        run(trainer)
    }
    
    func testReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: ReLU.str
        )
        run(trainer)
    }
    
    func testLeakyReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: LeakyReLU.str
        )
        run(trainer)
    }
    
    func testLeakyReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: LeakyReLU.str
        )
        run(trainer)
    }
    
    func testSoftReLUCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: SoftReLU.str
        )
        run(trainer)
    }
    
    func testSoftReLUGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: SoftReLU.str
        )
        run(trainer)
    }
    
    func testSigmoidCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(
            model: "Activation", activation: Sigmoid.str
        )
        run(trainer)
    }
    
    func testSigmoidGPU() throws
    {
        let trainer = _buildTrainer(
            model: "Activation", activation: Sigmoid.str
        )
        run(trainer)
    }
}