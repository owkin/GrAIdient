//
// Layer2DTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 15/10/2022.
//

import GrAIdient
import GrAITestsUtils

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2DGradTests: Input2DMSE1DCase
{
    override func setUp()
    {
        super.setUp()
        
        optimizerParams.nbLoops = 2
        GrAI.Loop.gradientChecking = true
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
        let params = GrAI.Model.Params(context: context)
        
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
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Convolution2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "ConvolutionStride1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "ConvolutionStride2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
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
                layerPrev: layer, size: 6, params: params
            )
            
        case "AdaptiveAvgPool2":
            layer = AdaptiveAvgPool2D(layerPrev: layer, size: 2, params: params)
            
        case "AdaptiveAvgPool3":
            layer = AdaptiveAvgPool2D(layerPrev: layer, size: 3, params: params)
            
        case "AdaptiveAvgPool4":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 9, params: params
            )
            
        case "AdaptiveAvgPool5":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 6, params: params
            )
            
        case "Sum":
            let otherLayer1: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            let otherLayer2: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            layer = Sum2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Activation":
            layer = Activation2D(
                layerPrev: layer,
                activation: SoftReLU.str,
                params: params
            )
            
        case "SelectNeurons":
            head = SelectNeurons2D(
                layerPrev: layer,
                targetI: 1, targetJ: 3,
                params: params
            )
            
        case "IRDFT2RGB":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 6, stride: 2,
                activation: SoftReLU.str, biases: true, bn: bn, params: params
            )
            layer = IRDFT2RGB(layerPrev: layer, params: params)
            
        case "DecorrelateRGB":
            layer = DecorrelateRGB(
                layerPrev: layer,
                correlation: [
                    0.26, 0.26, 0.27,
                    0.09, 0.00, -0.09,
                    0.02, -0.05, 0.03
                ].map { $0 / 0.4619524 },
                params: params
            )
            
        case "LinearScale":
            layer = LinearScale2D(
                layerPrev: layer,
                weight: 2.0,
                bias: 3.0,
                params: params
            )
            
        case "Multiply":
            let otherLayer1: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            let otherLayer2: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            layer = Multiply2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Pad":
            layer = Pad2D(
                layerPrev: layer,
                padDimension: 3, padValue: 0.5,
                params: params
            )
            
        case "Crop":
            layer = Crop2D(
                layerPrev: layer,
                cropDimension: 3,
                params: params
            )
            
        case "ResizeBilinearPad":
            layer = ResizeBilinearPad(
                layerPrev: layer,
                scalesList: [0.8, 1.2], padValue: 0.5,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "Rotate":
            layer = Rotate2D(
                layerPrev: layer,
                anglesList: [20.0, 350.0], padValue: 0.5,
                params: params
            )
            
        case "ResizeBilinearCrop1":
            layer = ResizeBilinearCrop(
                layerPrev: layer,
                scalesList: [0.6, 0.8],
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "ResizeBilinearCrop2":
            layer = ResizeBilinearCrop(
                layerPrev: layer,
                scalesList: [0.8, 1.2],
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "Deconvolution1":
            layer = MaxPool2D(
                layerPrev: layer, size: 3, stride: 4, params: params
            )
            layer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 1,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Deconvolution2":
            layer = MaxPool2D(
                layerPrev: layer, size: 3, stride: 4, params: params
            )
            layer = Deconvolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "DeconvolutionStride1":
            layer = MaxPool2D(
                layerPrev: layer, size: 3, stride: 4, params: params
            )
            layer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "DeconvolutionStride2":
            layer = MaxPool2D(
                layerPrev: layer, size: 3, stride: 4, params: params
            )
            layer = Deconvolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: SoftReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Concat":
            let otherLayer1: Layer2D = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            let otherLayer2: Layer2D = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 9, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            layer = Concat2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "InstanceNorm":
            layer = InstanceNorm2D(
                layerPrev: layer, activation: SoftReLU.str, params: params
            )
            
        case "AdaIN":
            let otherLayer: Layer = Constant1D(
                nbNeurons: 6, params: params
            )
            (otherLayer as! Constant1D).weightsCPU = [
                0.5, -0.5, 1.5, -2.0, 3.0, 1.0
            ]
            layer = AdaIN(
                layersPrev: [layer, otherLayer], params: params
            )
            
        case "Constant":
            var otherLayer: Layer2D = Constant2D(
                nbChannels: 5, height: height, width: width, params: params
            )
            (otherLayer as! Constant2D).weightsCPU = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            otherLayer = Convolution2D(
                layerPrev: otherLayer, size: 1, nbChannels: 3, stride: 1,
                activation: SoftReLU.str, biases: true, bn: false,
                params: params
            )
            layer = Sum2D(
                layersPrev: [layer, otherLayer], params: params
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
        GrAI.Opti.CPU = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer)
    }
    
    func testConvolution1NoBNCPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution2CPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution2", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride1CPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testConvolutionStride2CPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    func testBNGPU() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer)
    }
    
    func testMaxPool1CPU() throws
    {
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
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
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
    
    func testSelectNeuronsGPU() throws
    {
        let trainer = _buildTrainer(model: "SelectNeurons", bn: false)
        run(trainer)
    }
    
    func testIRDFT2RGBCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    func testIRDFT2RGBGPU() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    func testDecorrelateRGBCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    func testDecorrelateRGBGPU() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    func testLinearScaleCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    func testLinearScaleGPU() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    func testMultiplyCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    func testMultiplyGPU() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    func testPadCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    func testPadGPU() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    func testCropCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    func testCropGPU() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearPadCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinearPad", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearPadGPU() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad", bn: false)
        run(trainer)
    }
    
    func testRotateCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    func testRotateGPU() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop1GPU() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop2GPU() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    func testDeconvolution1BNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer)
    }
    
    func testDeconvolution1BNGPU() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution1BNSampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution1NoBNCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    func testDeconvolution1NoBNGPU() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution1NoBNSampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    func testDeconvolution2GPU() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution2SampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolutionStride1CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testDeconvolutionStride1GPU() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolutionStride1SampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolutionStride2CPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testDeconvolutionStride2GPU() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolutionStride2SampleGPU() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testConcatCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    func testConcatGPU() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    func testInstanceNormCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer)
    }
    
    func testInstanceNormGPU() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer)
    }
    
    func testAdaINCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer)
    }
    
    func testAdaINGPU() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer)
    }
    
    func testConstantCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    func testConstantGPU() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
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
        let params = GrAI.Model.Params(context: context)
        
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
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Convolution2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "ConvolutionStride1":
            layer = Convolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "ConvolutionStride2":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
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
                layerPrev: layer, size: 6, params: params
            )
            
        case "AdaptiveAvgPool2":
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 2, params: params
            )
            
        case "AdaptiveAvgPool3":
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 3, params: params
            )
            
        case "AdaptiveAvgPool4":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 9, params: params
            )
            
        case "AdaptiveAvgPool5":
            layer = MaxPool2D(
                layerPrev: layer, size: 2, stride: 2, params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: 6, params: params
            )
            
        case "Sum":
            let otherLayer1: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            let otherLayer2: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            layer = Sum2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Activation":
            layer = Activation2D(
                layerPrev: layer,
                activation: LeakyReLU.str,
                params: params
            )
            
        case "SelectNeurons":
            head = SelectNeurons2D(
                layerPrev: layer,
                targetI: 1, targetJ: 3,
                params: params
            )
            
        case "IRDFT2RGB":
            layer = Convolution2D(
                layerPrev: layer, size: 2, nbChannels: 6, stride: 2,
                activation: SoftReLU.str, biases: true, bn: bn, params: params
            )
            layer = IRDFT2RGB(layerPrev: layer, params: params)
            
        case "DecorrelateRGB":
            layer = DecorrelateRGB(
                layerPrev: layer,
                correlation: [
                    0.26, 0.26, 0.27,
                    0.09, 0.00, -0.09,
                    0.02, -0.05, 0.03
                ].map { $0 / 0.4619524 },
                params: params
            )
            
        case "LinearScale":
            layer = LinearScale2D(
                layerPrev: layer,
                weight: 2.0,
                bias: 3.0,
                params: params
            )
            
        case "Multiply":
            let otherLayer1: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            let otherLayer2: Layer2D = Convolution2D(
                layerPrev: firstLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            layer = Multiply2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "Pad":
            layer = Pad2D(
                layerPrev: layer,
                padDimension: 3, padValue: 0.5,
                params: params
            )
            
        case "Crop":
            layer = Crop2D(
                layerPrev: layer,
                cropDimension: 3,
                offsetI: 2,
                offsetJ: 2,
                params: params
            )
            
        case "ResizeBilinearPad1":
            layer = ResizeBilinearPad(
                layerPrev: layer,
                scalesList: [0.8], padValue: 0.5,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "ResizeBilinearPad2":
            layer = ResizeBilinearPad(
                layerPrev: layer,
                scalesList: [1.2], padValue: 0.5,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "Rotate":
            layer = Rotate2D(
                layerPrev: layer,
                anglesList: [20.0], padValue: 0.5,
                params: params
            )
            
        case "ResizeBilinearCrop1":
            layer = ResizeBilinearCrop(
                layerPrev: layer,
                scale: 0.8,
                offsetI: 0,
                offsetJ: 0,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "ResizeBilinearCrop2":
            layer = ResizeBilinearCrop(
                layerPrev: layer,
                scale: 1.2,
                offsetI: 1,
                offsetJ: 1,
                params: params
            )
            layer = AdaptiveAvgPool2D(
                layerPrev: layer, size: width, params: params
            )
            
        case "Deconvolution1":
            layer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Deconvolution2":
            layer = Deconvolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 1,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "DeconvolutionStride1":
            layer = Deconvolution2D(
                layerPrev: layer, size: 3, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "DeconvolutionStride2":
            layer = Deconvolution2D(
                layerPrev: layer, size: 2, nbChannels: 5, stride: 2,
                activation: LeakyReLU.str, biases: !bn, bn: bn, params: params
            )
            
        case "Concat":
            let otherLayer1: Layer2D = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            let otherLayer2: Layer2D = Convolution2D(
                layerPrev: layer, size: 1, nbChannels: 9, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            layer = Concat2D(
                layersPrev: [layer, otherLayer1, otherLayer2],
                params: params
            )
            
        case "InstanceNorm":
            layer = InstanceNorm2D(
                layerPrev: layer, activation: LeakyReLU.str, params: params
            )
            
        case "AdaIN":
            let otherLayer: Layer = Constant1D(
                nbNeurons: 6, params: params
            )
            (otherLayer as! Constant1D).weightsCPU = [
                0.5, -0.5, 1.5, -2.0, 3.0, 1.0
            ]
            layer = AdaIN(
                layersPrev: [layer, otherLayer], params: params
            )
            
        case "Constant":
            var otherLayer: Layer2D = Constant2D(
                nbChannels: 5, height: height, width: width, params: params
            )
            (otherLayer as! Constant2D).weightsCPU = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            otherLayer = Convolution2D(
                layerPrev: otherLayer, size: 1, nbChannels: 3, stride: 1,
                activation: LeakyReLU.str, biases: true, bn: false,
                params: params
            )
            layer = Sum2D(
                layersPrev: [layer, otherLayer], params: params
            )
            
        case "VQ1":
            layer = VQ2D(layerPrev: layer, K: 5, beta: 0.25, params: params)
            
        case "VQ2":
            layer = VQ2D(layerPrev: layer, K: 5, beta: 0.25, params: params)
            (layer as! VQ2D).computeVQ = false
            
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
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testConvolution1BNSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    func testConvolution1NoBNSample() throws
    {
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer, diffThreshold: 0.0001)
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
    
    func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    func testPad() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearPad1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearPad2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    func testResizeBilinearCrop2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    func testDeconvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    func testDeconvolution1SampleNoBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    func testDeconvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testDeconvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    func testDeconvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testDeconvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer)
    }
    
    func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer)
    }
    
    func testConstant() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    func testVQ1() throws
    {
        let trainer = _buildTrainer(model: "VQ1", bn: false)
        run(trainer)
    }
    
    func testVQ1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ1", bn: false)
        run(trainer)
    }
                         
    func testVQ2() throws
    {
        let trainer = _buildTrainer(model: "VQ2", bn: false)
        run(trainer)
    }

    func testVQ2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ2", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2DFlowResetTests: Layer2DFlowTests
{
    override func setUp()
    {
        super.setUp()
        
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
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testConvolution1BNSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer, diffThreshold: 0.0001)
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
    
    override func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    override func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    override func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    override func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    override func testPad() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    override func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1SampleNoBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    override func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer)
    }
    
    override func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    override func testVQ1() throws
    {
        let trainer = _buildTrainer(model: "VQ1", bn: false)
        run(trainer)
    }
    
    override func testVQ1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ1", bn: false)
        run(trainer)
    }
                         
    override func testVQ2() throws
    {
        let trainer = _buildTrainer(model: "VQ2", bn: false)
        run(trainer)
    }

    override func testVQ2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ2", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class Layer2DFlowReverseTests: Layer2DFlowTests
{
    override func setUp()
    {
        super.setUp()
        
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
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testConvolution1BNSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer, diffThreshold: 0.0001)
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
    
    override func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    override func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    override func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    override func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    override func testPad() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    override func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, diffThreshold: 0.0001)
    }
    
    override func testDeconvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1SampleNoBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    override func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer)
    }
    
    override func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    override func testVQ1() throws
    {
        let trainer = _buildTrainer(model: "VQ1", bn: false)
        run(trainer)
    }
    
    override func testVQ1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ1", bn: false)
        run(trainer)
    }
                         
    override func testVQ2() throws
    {
        let trainer = _buildTrainer(model: "VQ2", bn: false)
        run(trainer)
    }

    override func testVQ2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ2", bn: false)
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class Layer2DInferenceTests: Layer2DFlowTests
{
    override func setUp()
    {
        super.setUp()
        optimizerParams.nbLoops = 2
    }
    
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
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    override func testConvolution1BNSample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Convolution1", bn: true)
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    override func testConvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Convolution1", bn: false)
        run(trainer)
    }
    
    override func testConvolution1NoBNSample() throws
    {
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN() throws
    {
        let trainer = _buildTrainer(model: "BN", bn: false)
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
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
    
    override func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    override func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    override func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    override func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    override func testPad() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    override func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    override func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer, nbRetry: 5, diffThreshold: 0.01)
    }
    
    override func testDeconvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1SampleNoBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    override func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer)
    }
    
    override func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    override func testVQ1() throws
    {
        let trainer = _buildTrainer(model: "VQ1", bn: false)
        run(trainer)
    }
    
    override func testVQ1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ1", bn: false)
        run(trainer)
    }
                         
    override func testVQ2() throws
    {
        let trainer = _buildTrainer(model: "VQ2", bn: false)
        run(trainer)
    }

    override func testVQ2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ2", bn: false)
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN() throws
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
    
    override func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    override func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    override func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    override func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    override func testPad() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    override func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer)
    }
    
    override func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer)
    }
    
    override func testDeconvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1SampleNoBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    override func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer)
    }
    
    override func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    override func testVQ1() throws
    {
        let trainer = _buildTrainer(model: "VQ1", bn: false)
        run(trainer)
    }
    
    override func testVQ1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ1", bn: false)
        run(trainer)
    }
            
    /// Do not run this test as computeVQ is not recovered from the disk.
    override func testVQ2() throws
    {
        /*let trainer = _buildTrainer(model: "VQ2", bn: false)
        run(trainer)*/
    }

    /// Do not run this test as computeVQ is not recovered from the disk.
    override func testVQ2Sample() throws
    {
        /*GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ2", bn: false)
        run(trainer)*/
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
        run(trainer)
    }
    
    override func testConvolution1BNSample() throws
    {
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
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
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "ConvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testBN() throws
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
    
    override func testIRDFT2RGB() throws
    {
        let trainer = _buildTrainer(model: "IRDFT2RGB", bn: false)
        run(trainer)
    }
    
    override func testDecorrelateRGB() throws
    {
        let trainer = _buildTrainer(model: "DecorrelateRGB", bn: false)
        run(trainer)
    }
    
    override func testLinearScale() throws
    {
        let trainer = _buildTrainer(model: "LinearScale", bn: false)
        run(trainer)
    }
    
    override func testMultiply() throws
    {
        let trainer = _buildTrainer(model: "Multiply", bn: false)
        run(trainer)
    }
    
    override func testPad() throws
    {
        let trainer = _buildTrainer(model: "Pad", bn: false)
        run(trainer)
    }
    
    override func testCrop() throws
    {
        let trainer = _buildTrainer(model: "Crop", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearPad2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearPad2", bn: false)
        run(trainer)
    }
    
    override func testRotate() throws
    {
        let trainer = _buildTrainer(model: "Rotate", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop1() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop1", bn: false)
        run(trainer)
    }
    
    override func testResizeBilinearCrop2() throws
    {
        let trainer = _buildTrainer(model: "ResizeBilinearCrop2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1BN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer)
    }
    
    override func testDeconvolution1SampleBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: true)
        run(trainer)
    }
    
    override func testDeconvolution1NoBN() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution1SampleNoBN() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2() throws
    {
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolution2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "Deconvolution2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride1", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2() throws
    {
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testDeconvolutionStride2Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "DeconvolutionStride2", bn: false)
        run(trainer)
    }
    
    override func testConcat() throws
    {
        let trainer = _buildTrainer(model: "Concat", bn: false)
        run(trainer)
    }
    
    override func testInstanceNorm() throws
    {
        let trainer = _buildTrainer(model: "InstanceNorm", bn: false)
        run(trainer)
    }
    
    override func testAdaIN() throws
    {
        let trainer = _buildTrainer(model: "AdaIN", bn: false)
        run(trainer)
    }
    
    override func testConstant() throws
    {
        let trainer = _buildTrainer(model: "Constant", bn: false)
        run(trainer)
    }
    
    override func testVQ1() throws
    {
        let trainer = _buildTrainer(model: "VQ1", bn: false)
        run(trainer)
    }
    
    override func testVQ1Sample() throws
    {
        GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ1", bn: false)
        run(trainer)
    }
           
    /// Do not run this test as computeVQ is not copied.
    override func testVQ2() throws
    {
        /*let trainer = _buildTrainer(model: "VQ2", bn: false)
        run(trainer)*/
    }

    /// Do not run this test as computeVQ is not copied.
    override func testVQ2Sample() throws
    {
        /*GrAI.Gradient.sample = true
        let trainer = _buildTrainer(model: "VQ2", bn: false)
        run(trainer)*/
    }
}

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class MSE2DGradTests: Input2DMSE2DCase
{
    override func setUp()
    {
        super.setUp()
        GrAI.Loop.gradientChecking = true
    }
    
    private func _buildTrainer() -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "Layer2D",
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
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 1, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        _ = MSE2D(layerPrev: layer, params: params)
    }
    
    func testLossCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testLossGPU() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class MSE2DFlowTests: Input2DMSE2DCase
{
    private func _buildTrainer() -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    func buildModel(context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 1, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        _ = MSE2D(layerPrev: layer, params: params)
    }
    
    func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class MSE2DFlowResetTests: MSE2DFlowTests
{
    private func _buildTrainer() -> FlowResetTrainer
    {
        let trainer = FlowResetTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class MSE2DFlowReverseTests: MSE2DFlowTests
{
    private func _buildTrainer() -> FlowReverseTrainer
    {
        let trainer = FlowReverseTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class MSE2DFlowInferenceTests: MSE2DFlowTests
{
    private func _buildTrainer() -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class MSE2DLoadTests: MSE2DFlowTests
{
    private func _buildTrainer() -> LoadTrainer
    {
        let trainer = LoadTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class MSE2DTransformTests: MSE2DFlowTests
{
    private func _buildTrainer() -> TransformTrainer
    {
        let trainer = TransformTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testLoss() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Gradient Checking
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DGradTests: FTFrequences2DMSE1DCase
{
    override func setUp()
    {
        super.setUp()
        
        optimizerParams.nbLoops = 2
        GrAI.Loop.gradientChecking = true
    }
    
    private func _buildTrainer() -> GradTrainer
    {
        let trainer = GradTrainer(
            name: "Layer2D",
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
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        let frequences: Layer2D = FTFrequences2D(
            nbChannels: 6, dimension: width, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: SoftReLU.str, biases: true, bn: false, params: params
        )
        
        layer = Multiply2D(
            layersPrev: [layer, frequences], params: params
        )
        
        var head: Layer1D = FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: SoftReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
    }
    
    func testEvenCPU() throws
    {
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testEvenGPU() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testOddCPU() throws
    {
        height = 7
        width = 7
        GrAI.Opti.CPU = true
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testOddGPU() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DFlowTests: FTFrequences2DMSE1DCase
{
    private func _buildTrainer() -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    func buildModel(context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer2D = Input2D(
            nbChannels: 1, width: width, height: height, params: params
        )
        let frequences: Layer2D = FTFrequences2D(
            nbChannels: 6, dimension: width, params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 1, nbChannels: 6, stride: 1,
            activation: LeakyReLU.str, biases: true, bn: false, params: params
        )
        
        layer = Multiply2D(
            layersPrev: [layer, frequences], params: params
        )
        
        var head: Layer1D = FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true, params: params
        )
        
        head = MSE1D(layerPrev: head, params: params)
    }
    
    func testEven() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testOdd() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DFlowResetTests: FTFrequences2DFlowTests
{
    private func _buildTrainer() -> FlowResetTrainer
    {
        let trainer = FlowResetTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testEven() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    override func testOdd() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DFlowReverseTests: FTFrequences2DFlowTests
{
    private func _buildTrainer() -> FlowReverseTrainer
    {
        let trainer = FlowReverseTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testEven() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    override func testOdd() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU Loss in inference mode with CPU one.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DFlowInferenceTests: FTFrequences2DFlowTests
{
    private func _buildTrainer() -> InferenceTrainer
    {
        let trainer = InferenceTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testEven() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    override func testOdd() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// loaded model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DLoadTests: FTFrequences2DFlowTests
{
    private func _buildTrainer() -> LoadTrainer
    {
        let trainer = LoadTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testEven() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    override func testOdd() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU/CPU Losses in inference mode with the one obtained from a
// transformed model.
// We expect to see errors ~ 1e-3 and less.
// -----------------------------------------------------------------------------
class FTFrequences2DTransformTests: FTFrequences2DFlowTests
{
    private func _buildTrainer() -> TransformTrainer
    {
        let trainer = TransformTrainer(
            name: "Layer2D",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            buildModel(context: context)
        }
        return trainer
    }
    
    override func testEven() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    override func testOdd() throws
    {
        height = 7
        width = 7
        let trainer = _buildTrainer()
        run(trainer)
    }
}
