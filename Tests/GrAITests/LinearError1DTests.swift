//
// LinearError1DTests.swift
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
class LinearError1DGradTests: Input1DLinearError1DCase
{
    override func setUp()
    {
        super.setUp()
        GrAI.Loop.gradientChecking = true
    }
    
    private func _buildTrainer(_ model: String) -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "LinearError1D",
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
        
        switch model
        {
        case "FullyConnected":
            layer = FullyConnected(
                layerPrev: layer, nbNeurons: 12,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            
        case "Activation":
            layer = Activation1D(
                layerPrev: layer,
                activation: SoftReLU.str,
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
        
        layer = LinearError1D(layerPrev: layer, params: params)
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
    
    func testFLSampleGPU() throws
    {
        GrAI.Gradient.sample = true
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
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class LinearError1DFlowTests: Input1DLinearError1DCase
{
    private func _buildTrainer(_ model: String) -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "LinearError",
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
        
        switch model
        {
        case "FullyConnected":
            layer = FullyConnected(
                layerPrev: layer, nbNeurons: 12,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            
        case "Activation":
            layer = Activation1D(
                layerPrev: layer,
                activation: LeakyReLU.str,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        layer = FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        layer = LinearError1D(layerPrev: layer, params: params)
    }
    
    func testFL() throws
    {
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    func testFLSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    func testActivation() throws
    {
        let trainer = _buildTrainer("Activation")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class LinearError1DFlowResetTests: LinearError1DFlowTests
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
            name: "LinearError",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testFL() throws
    {
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    override func testFLSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer("Activation")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class LinearError1DFlowReverseTests: LinearError1DFlowTests
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
            name: "LinearError",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testFL() throws
    {
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    override func testFLSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer("Activation")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class LinearError1DInferenceTests: LinearError1DFlowTests
{
    private func _buildTrainer(_ model: String) -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "LinearError",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testFL() throws
    {
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    override func testFLSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer("Activation")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class LinearError1DLoadTests: LinearError1DFlowTests
{
    private func _buildTrainer(_ model: String) -> LoadTrainer
    {
        let trainer = LoadTrainer(
            name: "LinearError",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testFL() throws
    {
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    override func testFLSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer("Activation")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class LinearError1DTransformTests: LinearError1DFlowTests
{
    private func _buildTrainer(_ model: String) -> TransformTrainer
    {
        let trainer = TransformTrainer(
            name: "LinearError",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, context: context)
        }
        return trainer
    }
    
    override func testFL() throws
    {
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    override func testFLSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("FullyConnected")
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer("Activation")
        run(trainer)
    }
}