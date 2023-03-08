//
// ActivationSeqTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 08/03/2023.
//

import GrAIdient
import GrAITestsUtils

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class ActivationSeqGradTests: Input2DMSE1DCase
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
            name: "ActivationSeq",
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
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        var layerSeq: LayerSeq = FullyConnectedPatch(
            layerPrev: layer, patch: width / 3, nbNeurons: 5,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        switch model
        {
        case "FullyConnected":
            layerSeq = FullyConnectedSeq(
                layerPrev: layerSeq, nbNeurons: 5,
                activation: activation, biases: true,
                params: params
            )
            
        case "Activation":
            layerSeq = ActivationSeq(
                layerPrev: layerSeq,
                activation: activation!,
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
