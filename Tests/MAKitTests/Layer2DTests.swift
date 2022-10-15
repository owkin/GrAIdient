//
// Layer2DTests.swift
// MAKitTests
//
// Created by Jean-François Reboud on 15/10/2022.
//

import MAKit
import MAKitTestsUtils

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2DGradTests: Input2DMSE1DCase
{
    override func setUpWithError() throws
    {
        try! super.setUpWithError()
        MAKit.Loop.gradientChecking = true
    }
    
    private func _buildTrainer(model: String, bn: Bool) -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            _buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    private func _buildModel(model: String, bn: Bool, context: ModelContext)
    {
        let params = MAKit.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        let firstLayer = layer
        var head: Layer1D? = nil
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        switch model
        {
        case "Convolution1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 1,
                activation: SoftReLU.str, biases: true, bn: bn, params: params
            )
            
        case "Convolution2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: SoftReLU.str, biases: true, bn: bn, params: params
            )
            
        case "ConvolutionStride1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: SoftReLU.str, biases: true, bn: bn, params: params
            )
            
        case "ConvolutionStride2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: SoftReLU.str, biases: true, bn: bn, params: params
            )
            
        case "BN":
            layer = BN2D(
                layerPrev: layer, activation: SoftReLU.str, params: params
            )
            
        case "MaxPool1":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            
        case "MaxPool2":
            layer = MaxPool2D(
                layerPrev: layer, size: 4, stride: 4, params: params
            )
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            
        case "MaxPool3":
            layer = MaxPool2D(
                layerPrev: layer, size: 3, stride: 2, params: params
            )
            
        case "AvgPooling":
            head = AvgPool2D(layerPrev: layer, params: params)
            
        case "AdaptiveAvgPool1":
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 12, params: params
            )
            
        case "AdaptiveAvgPool2":
            layer = AdaptiveAvgPool2D(layerPrev: layer, size: 5, params: params)
            
        case "AdaptiveAvgPool3":
            layer = AdaptiveAvgPool2D(layerPrev: layer, size: 6, params: params)
            
        case "AdaptiveAvgPool4":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 15, params: params
            )
            
        case "AdaptiveAvgPool5":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 12, params: params
            )
            
        case "Sum":
            let otherLayer: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            
            layer = Sum2D(layersPrev: [layer, otherLayer], params: params)
            
        case "Activation":
            layer = Activation2D(
                layerPrev: layer,
                activation: SoftReLU.str,
                params: params
            )
            
        case "SelectNeurons":
            head = SelectNeurons2D(
                layerPrev: layer,
                targetI: 3, targetJ: 6,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        head = FullyConnected(
            layerPrev: head != nil ? head! : layer, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head!, params: params)
    }
    
    func testConvolution1BNCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    func testConvolution1BNGPU() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    func testConvolution1BNSampleGPU() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    func testConvolution1NoBNCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution1NoBNGPU() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution1NoBNSampleGPU() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution2CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    func testConvolution2GPU() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    func testConvolution2SampleGPU() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride1CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride1GPU() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride1SampleGPU() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride2CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride2GPU() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride2SampleGPU() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testBN2DCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    func testBN2DGPU() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    func testMaxPool1CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    func testMaxPool1GPU() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    func testMaxPool2CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    func testMaxPool2GPU() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    func testMaxPool3CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    func testMaxPool3GPU() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    func testAvgPoolCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    func testAvgPoolGPU() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool1CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool1GPU() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool2CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool2GPU() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool3CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool3GPU() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool4CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool4GPU() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool5CPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool5GPU() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    func testSumCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    func testSumGPU() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    func testActivationCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    func testActivationGPU() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    func testSelectNeuronsCPU() throws
    {
        MAKit.Opti.CPU = true
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
    
    func testSelectNeuronsGPU() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2DFlowTests: Input2DMSE1DCase
{
    private func _buildTrainer(model: String, bn: Bool) -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    func buildModel(model: String, bn: Bool, context: ModelContext)
    {
        let params = MAKit.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        let firstLayer = layer
        var head: Layer1D? = nil
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 3, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        switch model
        {
        case "Convolution1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: bn, params: params
            )
            
        case "Convolution2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: bn, params: params
            )
            
        case "ConvolutionStride1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: true, bn: bn, params: params
            )
            
        case "ConvolutionStride2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: true, bn: bn, params: params
            )
            
        case "BN":
            layer = BN2D(
                layerPrev: layer, activation: LeakyReLU.str, params: params
            )
            
        case "MaxPool1":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            
        case "MaxPool2":
            layer = MaxPool2D(
                layerPrev: layer, size: 4, stride: 4, params: params
            )
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            
        case "MaxPool3":
            layer = MaxPool2D(
                layerPrev: layer, size: 3, stride: 2, params: params
            )
            
        case "AvgPooling":
            head = AvgPool2D(layerPrev: layer, params: params)
            
        case "AdaptiveAvgPool1":
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 12, params: params
            )
            
        case "AdaptiveAvgPool2":
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 5, params: params
            )
            
        case "AdaptiveAvgPool3":
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 6, params: params
            )
            
        case "AdaptiveAvgPool4":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 15, params: params
            )
            
        case "AdaptiveAvgPool5":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 12, params: params
            )
            
        case "Sum":
            let otherLayer: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            
            layer = Sum2D(layersPrev: [layer, otherLayer], params: params)
            
        case "Activation":
            layer = Activation2D(
                layerPrev: layer,
                activation: LeakyReLU.str,
                params: params
            )
            
        case "SelectNeurons":
            head = SelectNeurons2D(
                layerPrev: layer,
                targetI: 3, targetJ: 6,
                params: params
            )
            
        default:
            fatalError("Unreachable.")
        }
        
        head = FullyConnected(
            layerPrev: head != nil ? head! : layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head!, params: params)
    }
    
    func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    func testConvolution1BNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution1NoBNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    func testConvolution2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride1Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testBN2D() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    func testMaxPool1() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    func testMaxPool2() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    func testMaxPool3() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    func testAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool1() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool2() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool3() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool4() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    func testAdaptiveAvgPool5() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    func testSelectNeurons() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2DFlowResetTests: Layer2DFlowTests
{
    override func setUpWithError() throws
    {
        try! super.setUpWithError()
        
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
    }
    
    private func _buildTrainer(model: String, bn: Bool) -> FlowResetTrainer
    {
        let trainer = FlowResetTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    override func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1BNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolution2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN2D() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    override func testMaxPool1() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    override func testMaxPool2() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    override func testMaxPool3() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    override func testAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool1() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool2() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool3() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool4() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool5() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2DFlowReverseTests: Layer2DFlowTests
{
    override func setUpWithError() throws
    {
        try! super.setUpWithError()
        
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
    }
    
    private func _buildTrainer(model: String, bn: Bool) -> FlowReverseTrainer
    {
        let trainer = FlowReverseTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    override func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1BNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolution2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN2D() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    override func testMaxPool1() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    override func testMaxPool2() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    override func testMaxPool3() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    override func testAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool1() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool2() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool3() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool4() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool5() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class Layer2DInferenceTests: Layer2DFlowTests
{
    private func _buildTrainer(model: String, bn: Bool) -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    override func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1BNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolution2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN2D() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    override func testMaxPool1() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    override func testMaxPool2() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    override func testMaxPool3() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    override func testAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool1() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool2() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool3() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool4() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool5() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class Layer2DLoadTests: Layer2DFlowTests
{
    private func _buildTrainer(model: String, bn: Bool) -> LoadTrainer
    {
        let trainer = LoadTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    override func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1BNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolution2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride1Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConvolutionStride2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN2D() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    override func testMaxPool1() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        run(trainer)
    }
    
    override func testMaxPool2() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        run(trainer)
    }
    
    override func testMaxPool3() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        run(trainer)
    }
    
    override func testAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool1() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool2() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool3() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool4() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        run(trainer)
    }
    
    override func testAdaptiveAvgPool5() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        run(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        run(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        run(trainer)
    }
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class Layer2DTransformTests: Layer2DFlowTests
{
    private func _buildTrainer(model: String, bn: Bool) -> TransformTrainer
    {
        let trainer = TransformTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(model: model, bn: bn, context: context)
        }
        return trainer
    }
    
    override func testConvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        runCopy(trainer)
    }
    
    func testConvolution1BNCopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        runCopyInPlace(trainer)
    }
    
    func testConvolution1BNResize() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        runResize(trainer)
    }
    
    func testConvolution1BNResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        runResizeInPlace(trainer)
    }
    
    override func testConvolution1BNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        runCopy(trainer)
    }
    
    func testConvolution1BNSampleCopyInPlace() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        runCopyInPlace(trainer)
    }
    
    func testConvolution1BNSampleResize() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        runResize(trainer)
    }
    
    func testConvolution1BNSampleResizeInPlace() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        runResizeInPlace(trainer)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        runCopy(trainer)
    }
    
    func testConvolution1NoBNCopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testConvolution1NoBNResize() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        runResize(trainer)
    }
    
    func testConvolution1NoBNResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        runCopy(trainer)
    }
    
    func testConvolution1NoBNSampleCopyInPlace() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testConvolution1NoBNSampleResize() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        runResize(trainer)
    }
    
    func testConvolution1NoBNSampleResizeInPlace() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testConvolution2() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        runCopy(trainer)
    }
    
    func testConvolution2CopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testConvolution2Resize() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        runResize(trainer)
    }
    
    func testConvolution2ResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testConvolution2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        runCopy(trainer)
    }
    
    func testConvolution2SampleCopyInPlace() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testConvolution2SampleResize() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        runResize(trainer)
    }
    
    func testConvolution2SampleResizeInPlace() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testConvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        runCopy(trainer)
    }
    
    func testConvolutionStride1CopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testConvolutionStride1Resize() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        runResize(trainer)
    }
    
    func testConvolutionStride1ResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testConvolutionStride1Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        runCopy(trainer)
    }
    
    func testConvolutionStride1SampleCopyInPlace() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testConvolutionStride1SampleResize() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        runResize(trainer)
    }
    
    func testConvolutionStride1SampleResizeInPlace() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testConvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        runCopy(trainer)
    }
    
    func testConvolutionStride2CopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testConvolutionStride2Resize() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        runResize(trainer)
    }
    
    func testConvolutionStride2ResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testConvolutionStride2Sample() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        runCopy(trainer)
    }
    
    func testConvolutionStride2SampleCopyInPlace() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testConvolutionStride2SampleResize() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        runResize(trainer)
    }
    
    func testConvolutionStride2SampleResizeInPlace() throws
    {
        MAKit.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testBN2D() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        runCopy(trainer)
    }
    
    func testBN2DCopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testBN2DResize() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        runResize(trainer)
    }
    
    func testBN2DResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testMaxPool1() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        runCopy(trainer)
    }
    
    func testMaxPool1CopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testMaxPool1Resize() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        runResize(trainer)
    }
    
    func testMaxPool1ResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "MaxPool1", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testMaxPool2() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        runCopy(trainer)
    }
    
    func testMaxPool2CopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testMaxPool2Resize() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        runResize(trainer)
    }
    
    func testMaxPool2ResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "MaxPool2", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testMaxPool3() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        runCopy(trainer)
    }
    
    func testMaxPool3CopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testMaxPool3Resize() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        runResize(trainer)
    }
    
    func testMaxPool3ResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "MaxPool3", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testAvgPool() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        runCopy(trainer)
    }
    
    func testAvgPoolCopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testAvgPoolResize() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        runResize(trainer)
    }
    
    func testAvgPoolResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "AvgPooling", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testAdaptiveAvgPool1() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        runCopy(trainer)
    }
    
    func testAdaptiveAvgPool1CopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testAdaptiveAvgPool1Resize() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        runResize(trainer)
    }
    
    func testAdaptiveAvgPool1ResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool1", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testAdaptiveAvgPool2() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        runCopy(trainer)
    }
    
    func testAdaptiveAvgPool2CopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testAdaptiveAvgPool2Resize() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        runResize(trainer)
    }
    
    func testAdaptiveAvgPool2ResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool2", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testAdaptiveAvgPool3() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        runCopy(trainer)
    }
    
    func testAdaptiveAvgPool3CopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testAdaptiveAvgPool3Resize() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        runResize(trainer)
    }
    
    func testAdaptiveAvgPool3ResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool3", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testAdaptiveAvgPool4() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        runCopy(trainer)
    }
    
    func testAdaptiveAvgPool4CopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testAdaptiveAvgPool4Resize() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        runResize(trainer)
    }
    
    func testAdaptiveAvgPool4ResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool4", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testAdaptiveAvgPool5() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        runCopy(trainer)
    }
    
    func testAdaptiveAvgPool5CopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testAdaptiveAvgPool5Resize() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        runResize(trainer)
    }
    
    func testAdaptiveAvgPool5ResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "AdaptiveAvgPool5", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testSum() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        runCopy(trainer)
    }
    
    func testSumCopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testSumResize() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        runResize(trainer)
    }
    
    func testResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "Sum", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testActivation() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        runCopy(trainer)
    }
    
    func testActivationCopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testActivationResize() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        runResize(trainer)
    }
    
    func testActivationResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "Activation", bn: false)
        runResizeInPlace(trainer)
    }
    
    override func testSelectNeurons() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        runCopy(trainer)
    }
    
    func testSelectNeuronsCopyInPlace() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        runCopyInPlace(trainer)
    }
    
    func testSelectNeuronsResize() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        runResize(trainer)
    }
    
    func testSelectNeuronsResizeInPlace() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        runResizeInPlace(trainer)
    }
}
