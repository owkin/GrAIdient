//
// Layer1DTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 10/10/2022.
//

import XCTest
import GrAIdient
import GrAITestsUtils

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer1DGradTests: Input1DMSE1DCase
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
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: SoftReLU.str, biases: true,
            params: params
        )
        
        switch model
        {
        case "FullyConnected":
            layer = try! FullyConnected(
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
            
        case "SelectNeurons":
            layer = SelectNeurons1D(
                layerPrev: layer,
                neurons: [1, 4],
                coeffs: [0.6, 0.4],
                params: params
            )
            
        case "Concat":
            let otherLayer1: Layer1D = try! FullyConnected(
                layerPrev: layer, nbNeurons: 9,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            let otherLayer2: Layer1D = try! FullyConnected(
                layerPrev: layer, nbNeurons: 6,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            layer = try! FullyConnected(
                layerPrev: layer, nbNeurons: 3,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            layer = Concat1D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Sum":
            let otherLayer1: Layer1D = try! FullyConnected(
                layerPrev: layer, nbNeurons: 10,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            let otherLayer2: Layer1D = try! FullyConnected(
                layerPrev: layer, nbNeurons: 10,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            layer = try! FullyConnected(
                layerPrev: layer, nbNeurons: 10,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            layer = try! Sum1D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Softmax":
            layer = try! FullyConnected(
                layerPrev: layer, nbNeurons: 15,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            
            layer = try! Softmax1D(layerPrev: layer, nbHeads: 3, params: params)
            
        case "DotProduct":
            let otherLayer: Layer1D = try! FullyConnected(
                layerPrev: layer, nbNeurons: 12,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            layer = try! FullyConnected(
                layerPrev: layer, nbNeurons: 12,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            layer = try! DotProduct1D(
                layersPrev: [layer, otherLayer], size: 3, params: params
            )
            
        case "Constant":
            var otherLayer: Layer1D = Constant1D(
                nbNeurons: 5, params: params
            )
            (otherLayer as! Constant1D).weightsCPU = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            otherLayer = try! FullyConnected(
                layerPrev: otherLayer, nbNeurons: 5,
                activation: SoftReLU.str, biases: true,
                params: params
            )
            layer = try! Sum1D(
                layersPrev: [layer, otherLayer], params: params
            )
            
        case "LayerOutput":
            layer = MSE1D(layerPrev: layer, params: params)
            
        case "Dropout1":
            layer = Dropout1D(layerPrev: layer, coeff: 0.0, params: params)
            
        case "Dropout2":
            layer = Dropout1D(layerPrev: layer, coeff: 1.0, params: params)
            
        default:
            fatalError("Unreachable.")
        }
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: SoftReLU.str, biases: true,
            params: params
        )
        
        layer = MSE1D(layerPrev: layer, params: params)
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
    
    func testConcatCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Concat")
        run(trainer)
    }
    
    func testConcatGPU() throws
    {
        let trainer = _buildTrainer("Concat")
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
    
    func testDotProductCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("DotProduct")
        run(trainer)
    }
    
    func testDotProductGPU() throws
    {
        let trainer = _buildTrainer("DotProduct")
        run(trainer)
    }
    
    func testConstantCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    func testConstantGPU() throws
    {
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    func testConstantSampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    func testLayerOutputCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("LayerOutput")
        run(trainer)
    }
    
    func testLayerOutputGPU() throws
    {
        let trainer = _buildTrainer("LayerOutput")
        run(trainer)
    }
    
    func testDropout1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Dropout1")
        run(trainer)
    }
    
    func testDropout1GPU() throws
    {
        let trainer = _buildTrainer("Dropout1")
        run(trainer)
    }
    
    func testDropout2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer("Dropout2")
        run(trainer)
    }
    
    func testDropout2GPU() throws
    {
        let trainer = _buildTrainer("Dropout2")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer1DFlowTests: Input1DMSE1DCase
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
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        switch model
        {
        case "FullyConnected":
            layer = try! FullyConnected(
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
            
        case "SelectNeurons":
            layer = SelectNeurons1D(
                layerPrev: layer,
                neurons: [1, 4],
                coeffs: [0.6, 0.4],
                params: params
            )
            
        case "Concat":
            let otherLayer1: Layer1D = try! FullyConnected(
                layerPrev: layer, nbNeurons: 9,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            let otherLayer2: Layer1D = try! FullyConnected(
                layerPrev: layer, nbNeurons: 6,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            layer = try! FullyConnected(
                layerPrev: layer, nbNeurons: 3,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            layer = Concat1D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Sum":
            let otherLayer1: Layer1D = try! FullyConnected(
                layerPrev: layer, nbNeurons: 10,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            let otherLayer2: Layer1D = try! FullyConnected(
                layerPrev: layer, nbNeurons: 10,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            layer = try! FullyConnected(
                layerPrev: layer, nbNeurons: 10,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            layer = try! Sum1D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Softmax":
            layer = try! FullyConnected(
                layerPrev: layer, nbNeurons: 15,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            
            layer = try! Softmax1D(layerPrev: layer, nbHeads: 3, params: params)
            
        case "DotProduct":
            let otherLayer: Layer1D = try! FullyConnected(
                layerPrev: layer, nbNeurons: 12,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            layer = try! FullyConnected(
                layerPrev: layer, nbNeurons: 12,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            layer = try! DotProduct1D(
                layersPrev: [layer, otherLayer], size: 3, params: params
            )
            
        case "Constant":
            var otherLayer: Layer1D = Constant1D(
                nbNeurons: 5, params: params
            )
            (otherLayer as! Constant1D).weightsCPU = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            otherLayer = try! FullyConnected(
                layerPrev: otherLayer, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            layer = try! Sum1D(
                layersPrev: [layer, otherLayer], params: params
            )
            
        case "LayerOutput":
            layer = MSE1D(layerPrev: layer, params: params)
            
        case "Dropout":
            layer = Dropout1D(layerPrev: layer, coeff: 0.5, params: params)
            
        default:
            fatalError("Unreachable.")
        }
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        layer = MSE1D(layerPrev: layer, params: params)
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
    
    func testConstant() throws
    {
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    func testConstantSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    func testLayerOutput() throws
    {
        let trainer = _buildTrainer("LayerOutput")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class Layer1DFlowPrecisionTests: Layer1DFlowTests
{
    private func _buildTrainer(_ model: String) -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
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
        run(trainer, diffThreshold: 0.002)
    }
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer("SelectNeurons")
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer("Concat")
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer, diffThreshold: 0.002)
    }
    
    override func testSoftmax() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer, diffThreshold: 0.002)
    }
    
    override func testDotProduct() throws
    {
        let trainer = _buildTrainer("DotProduct")
        run(trainer, diffThreshold: 0.002)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    override func testConstantSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    override func testLayerOutput() throws
    {
        let trainer = _buildTrainer("LayerOutput")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer1DFlowResetTests: Layer1DFlowTests
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
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer("SelectNeurons")
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer("Concat")
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    override func testSoftmax() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    override func testDotProduct() throws
    {
        let trainer = _buildTrainer("DotProduct")
        run(trainer)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    override func testConstantSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    override func testLayerOutput() throws
    {
        let trainer = _buildTrainer("LayerOutput")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer1DFlowReverseTests: Layer1DFlowTests
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
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer("SelectNeurons")
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer("Concat")
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    override func testSoftmax() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    override func testDotProduct() throws
    {
        let trainer = _buildTrainer("DotProduct")
        run(trainer)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    override func testConstantSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    override func testLayerOutput() throws
    {
        let trainer = _buildTrainer("LayerOutput")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer1DFlowAccumulateTests: Input1DMSE1DCase
{
    private func _buildTrainer(_ model: String) -> FlowTrainer
    {
        let trainer = FlowAccumulateTrainer(
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
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        switch model
        {
        case "FullyConnected":
            layer = try! FullyConnected(
                layerPrev: layer, nbNeurons: 12,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            
        case "Constant":
            var otherLayer: Layer1D = Constant1D(
                nbNeurons: 5, params: params
            )
            (otherLayer as! Constant1D).weightsCPU = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            otherLayer = try! FullyConnected(
                layerPrev: otherLayer, nbNeurons: 5,
                activation: LeakyReLU.str, biases: true,
                params: params
            )
            layer = try! Sum1D(
                layersPrev: [layer, otherLayer], params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        layer = MSE1D(layerPrev: layer, params: params)
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
    
    func testConstant() throws
    {
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    func testConstantSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class Layer1DInferenceTests: Layer1DFlowTests
{
    private func _buildTrainer(_ model: String) -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
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
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer("SelectNeurons")
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer("Concat")
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    override func testSoftmax() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    override func testDotProduct() throws
    {
        let trainer = _buildTrainer("DotProduct")
        run(trainer)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    override func testConstantSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    override func testLayerOutput() throws
    {
        let trainer = _buildTrainer("LayerOutput")
        run(trainer)
    }
    
    // Test should be Ok:
    // it is normal that the Flow part is Ko because CPU and GPU models
    // do not share same dropout state.
    // Anyway, the final check is done in inference, where both models
    // should operate the same way.
    func testDropout() throws
    {
        let trainer = _buildTrainer("Dropout")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class Layer1DLoadTests: Layer1DFlowTests
{
    private func _buildTrainer(_ model: String) -> LoadTrainer
    {
        let trainer = LoadTrainer(
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
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer("SelectNeurons")
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer("Concat")
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    override func testSoftmax() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    override func testDotProduct() throws
    {
        let trainer = _buildTrainer("DotProduct")
        run(trainer)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    override func testConstantSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    override func testLayerOutput() throws
    {
        let trainer = _buildTrainer("LayerOutput")
        run(trainer)
    }
    
    // Test should be Ok:
    // it is normal that the Flow part is Ko because CPU and GPU models
    // do not share same dropout state.
    // Anyway, the final check is done in inference, where both models
    // should operate the same way.
    func testDropout() throws
    {
        let trainer = _buildTrainer("Dropout")
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class Layer1DTransformTests: Layer1DFlowTests
{
    private func _buildTrainer(_ model: String) -> TransformTrainer
    {
        let trainer = TransformTrainer(
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
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer("SelectNeurons")
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer("Concat")
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer("Sum")
        run(trainer)
    }
    
    override func testSoftmax() throws
    {
        let trainer = _buildTrainer("Softmax")
        run(trainer)
    }
    
    override func testDotProduct() throws
    {
        let trainer = _buildTrainer("DotProduct")
        run(trainer)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    override func testConstantSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer("Constant")
        run(trainer)
    }
    
    override func testLayerOutput() throws
    {
        let trainer = _buildTrainer("LayerOutput")
        run(trainer)
    }
    
    // Test should be Ok:
    // it is normal that the Flow part is Ko because CPU and GPU models
    // do not share same dropout state.
    // Anyway, the final check is done in inference, where both models
    // should operate the same way.
    func testDropout() throws
    {
        let trainer = _buildTrainer("Dropout")
        run(trainer)
    }
}

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
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: SoftReLU.str, biases: true,
            params: params
        )
        
        switch model
        {
        case "FullyConnected":
            layer = try! FullyConnected(
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
        
        layer = try! FullyConnected(
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
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        switch model
        {
        case "FullyConnected":
            layer = try! FullyConnected(
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
        
        layer = try! FullyConnected(
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

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class BCE1DGradTests: Input1DBCE1DCase
{
    override func setUp()
    {
        super.setUp()
        GrAI.Loop.gradientChecking = true
    }
    
    private func _buildTrainer(_ model: String) -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "BCE1D",
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
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: SoftReLU.str, biases: true,
            params: params
        )
        
        switch model
        {
        case "FullyConnected":
            layer = try! FullyConnected(
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
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: Sigmoid.str, biases: true,
            params: params
        )
        
        layer = BCE1D(layerPrev: layer, params: params)
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
class BCE1DFlowTests: Input1DBCE1DCase
{
    private func _buildTrainer(_ model: String) -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "BCE1D",
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
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        switch model
        {
        case "FullyConnected":
            layer = try! FullyConnected(
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
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: Sigmoid.str, biases: true,
            params: params
        )
        
        layer = BCE1D(layerPrev: layer, params: params)
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
class BCE1DFlowResetTests: BCE1DFlowTests
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
            name: "BCE1D",
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
class BCE1DFlowReverseTests: BCE1DFlowTests
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
            name: "BCE1D",
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
class BCE1DInferenceTests: BCE1DFlowTests
{
    private func _buildTrainer(_ model: String) -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "BCE1D",
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
class BCE1DLoadTests: BCE1DFlowTests
{
    private func _buildTrainer(_ model: String) -> LoadTrainer
    {
        let trainer = LoadTrainer(
            name: "BCE1D",
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
class BCE1DTransformTests: BCE1DFlowTests
{
    private func _buildTrainer(_ model: String) -> TransformTrainer
    {
        let trainer = TransformTrainer(
            name: "BCE1D",
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
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class BCESigmoid1DGradTests: Input1DBCESigmoid1DCase
{
    override func setUp()
    {
        super.setUp()
        GrAI.Loop.gradientChecking = true
    }
    
    private func _buildTrainer(_ model: String) -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "BCESigmoid1D",
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
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: SoftReLU.str, biases: true,
            params: params
        )
        
        switch model
        {
        case "FullyConnected":
            layer = try! FullyConnected(
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
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: nil, biases: true,
            params: params
        )
        
        layer = BCESigmoid1D(layerPrev: layer, params: params)
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
class BCESigmoid1DFlowTests: Input1DBCESigmoid1DCase
{
    private func _buildTrainer(_ model: String) -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "BCESigmoid1D",
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
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        switch model
        {
        case "FullyConnected":
            layer = try! FullyConnected(
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
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: nil, biases: true,
            params: params
        )
        
        layer = BCESigmoid1D(layerPrev: layer, params: params)
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
class BCESigmoid1DFlowResetTests: BCESigmoid1DFlowTests
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
            name: "BCE1D",
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
class BCESigmoid1DFlowReverseTests: BCESigmoid1DFlowTests
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
            name: "BCE1D",
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
class BCESigmoid1DInferenceTests: BCESigmoid1DFlowTests
{
    private func _buildTrainer(_ model: String) -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "BCE1D",
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
class BCESigmoid1DLoadTests: BCESigmoid1DFlowTests
{
    private func _buildTrainer(_ model: String) -> LoadTrainer
    {
        let trainer = LoadTrainer(
            name: "BCE1D",
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
class BCESigmoid1DTransformTests: BCESigmoid1DFlowTests
{
    private func _buildTrainer(_ model: String) -> TransformTrainer
    {
        let trainer = TransformTrainer(
            name: "BCE1D",
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
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Dropout1DFlowTest: Input1DMSE1DCase
{
    override func setUp()
    {
        super.setUp()
        GrAI.Loop.gradientChecking = true
    }
    
    ///
    /// Create the model.
    ///
    /// - Returns:
    ///     The model created.
    ///
    func buildModel() -> Model
    {
        let context = ModelContext(name: "Dropout", curID: 0)
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer1D = Input1D(nbNeurons: 1, params: params)
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 5,
            activation: SoftReLU.str, biases: true,
            params: params
        )
        
        layer = Dropout1D(layerPrev: layer, coeff: 0.5, params: params)
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: SoftReLU.str, biases: true,
            params: params
        )
        
        layer = MSE1D(layerPrev: layer, params: params)
        
        return Model(model: context.model, modelsPrev: [])
    }
    
    func testFlow()
    {
        let modelCPU = buildModel()
        let modelGPU = buildModel()
        
        GrAI.Opti.CPU = true
        randomSelectWeightsInitializationScheme(model: modelCPU)
        
        modelCPU.initialize(
            params: optimizerParams,
            phase: .InferenceBackward,
            deviceID: DEVICE_ID
        )
        modelCPU.computeDeltaWeights = true
        
        modelGPU.weights = modelCPU.weights
        
        GrAI.Opti.GPU = true
        modelGPU.initialize(
            params: optimizerParams,
            phase: .InferenceBackward,
            deviceID: DEVICE_ID
        )
        modelGPU.computeDeltaWeights = true
        
        let firstLayerCPU = modelCPU.layers.first as! Input1D
        let firstLayerGPU = modelGPU.layers.first as! Input1D
        
        firstLayerCPU.computeDeltaWeights = false
        firstLayerGPU.computeDeltaWeights = false
        
        let lastLayerCPU = modelCPU.layers.last as! MSE1D
        let lastLayerGPU = modelGPU.layers.last as! MSE1D
        
        lastLayerCPU.coeff = -1.0
        lastLayerGPU.coeff = -1.0
        
        var numLoop = 0
        while numLoop < optimizerParams.nbLoops
        {
            let resultsCPU: [Double]
            GrAI.Opti.CPU = true
            
            var (inputs, batchSize) = setData(nil, modelCPU)
            modelCPU.updateKernel(batchSize: batchSize)
            try! modelCPU.forward()
            
            var gt = setLoss(nil, modelCPU)
            try! modelCPU.backward()
            try! modelCPU.update()
            
            resultsCPU = getGradients(model: modelCPU)
            
            let resultsGPU: [Double]
            GrAI.Opti.GPU = true
            
            (inputs, batchSize) = setData(inputs, modelGPU)
            modelGPU.updateKernel(batchSize: batchSize)
            try! modelGPU.forward()
            
            gt = setLoss(gt, modelGPU)
            try! modelGPU.backward()
            try! modelGPU.update()
            
            resultsGPU = getGradients(model: modelGPU)
            
            if let gradDiff = checkFlow(resultsCPU, resultsGPU)
            {
                XCTAssert(gradDiff < 0.000001)
            }
            
            modelCPU.incStep()
            modelGPU.incStep()
            numLoop += 1
        }
    }
}
