//
// Layer1DDirtyTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 01/01/2023.
//

import GrAIdient
import GrAITestsUtils

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer1DDirtyGradTests: Input1DMSE1DCase
{
    override func setUp()
    {
        super.setUp()
        GrAI.Loop.gradientChecking = true
    }
    
    private func _buildTrainer(_ model: String) -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "Layer1D",
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
        
        var layer: Layer1D = Input1D(nbNeurons: 1, params: params)
        
        layer = FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: SoftReLU.str, biases: true,
            params: params
        )
        
        var firstLayer: Layer1D = layer
        var secondLayer: Layer1D
        
        switch model
        {
        case "FullyConnected":
            firstLayer = FullyConnected(
                layerPrev: layer, nbNeurons: 12,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            secondLayer = FullyConnected(
                layerPrev: firstLayer, nbNeurons: 12,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            
        case "Activation":
            secondLayer = Activation1D(
                layerPrev: layer,
                activation: SoftReLU.str,
                params: params
            )
            
        case "SelectNeurons":
            secondLayer = SelectNeurons1D(
                layerPrev: layer,
                neurons: [1, 4],
                coeffs: [0.6, 0.4],
                params: params
            )
            secondLayer = FullyConnected(
                layerPrev: secondLayer, nbNeurons: 5,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            
        case "Softmax":
            secondLayer = Softmax1D(layerPrev: layer, size: 5, params: params)
            
        default:
            fatalError("Unreachable.")
        }
        
        layer = Sum1D(
            layersPrev: [firstLayer, secondLayer], params: params
        )
        
        var head: Layer1D = FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
    }
    
    func testFLCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    func testFLGPU() throws
    {
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    func testActivationCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Activation")
        run(trainer)
    }
    
    func testActivationGPU() throws
    {
        let trainer = _buildTrainer("Activation")
        run(trainer)
    }
    
    func testSelectNeuronsCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("SelectNeurons")
        run(trainer)
    }
    
    func testSelectNeuronsGPU() throws
    {
        let trainer = _buildTrainer("SelectNeurons")
        run(trainer)
    }
    
    func testSoftmaxCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    func testSoftmaxGPU() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer1DDirtyFlowTests: Input1DMSE1DCase
{
    private func _buildTrainer(_ model: String) -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "Layer1D",
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
        
        var layer: Layer1D = Input1D(nbNeurons: 1, params: params)
        
        layer = FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        var firstLayer: Layer1D = layer
        var secondLayer: Layer1D
        
        switch model
        {
        case "FullyConnected":
            firstLayer = FullyConnected(
                layerPrev: layer, nbNeurons: 12,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            secondLayer = FullyConnected(
                layerPrev: firstLayer, nbNeurons: 12,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            
        case "Activation":
            secondLayer = Activation1D(
                layerPrev: layer,
                activation: LeakyReLU.str,
                params: params
            )
            
        case "SelectNeurons":
            secondLayer = SelectNeurons1D(
                layerPrev: layer,
                neurons: [1, 4],
                coeffs: [0.6, 0.4],
                params: params
            )
            secondLayer = FullyConnected(
                layerPrev: secondLayer, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            
        case "Concat":
            let otherLayer: Layer1D = FullyConnected(
                layerPrev: layer, nbNeurons: 6,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            secondLayer = Concat1D(
                layersPrev: [firstLayer, otherLayer],
                params: params
            )
            secondLayer = FullyConnected(
                layerPrev: secondLayer, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            
        case "Sum":
            let otherLayer: Layer1D = FullyConnected(
                layerPrev: layer, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            secondLayer = Sum1D(
                layersPrev: [firstLayer, otherLayer],
                params: params
            )
            
        case "Softmax":
            secondLayer = Softmax1D(layerPrev: layer, size: 5, params: params)
            
        case "DotProduct":
            let otherLayer: Layer1D = FullyConnected(
                layerPrev: layer, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            secondLayer = DotProduct1D(
                layersPrev: [firstLayer, otherLayer], size: 5, params: params
            )
            secondLayer = FullyConnected(
                layerPrev: secondLayer, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        layer = Sum1D(
            layersPrev: [firstLayer, secondLayer], params: params
        )
        
        var head: Layer1D = FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
    }
    
    func testFL() throws
    {
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    func testActivation() throws
    {
        let trainer = _buildTrainer("Activation")
        run(trainer)
    }
    
    func testSelectNeurons() throws
    {
        let trainer = _buildTrainer("SelectNeurons")
        run(trainer)
    }
    
    func testConcat() throws
    {
        let trainer = _buildTrainer("Concat")
        run(trainer)
    }
    
    func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    func testSoftmax() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    func testDotProduct() throws
    {
        let trainer = _buildTrainer("DotProduct")
        run(trainer)
    }
}