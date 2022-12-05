//
// MATorchTests.swift
// MATorchTests
//
// Created by Jean-François Reboud on 19/10/2022.
//

import XCTest
import MAKit

/// Compare models created by MAKit and PyTorch.
final class MATorchTests: XCTestCase
{
    /// Size of one image (height and width are the same).
    let _size = 32
    
    /// Initialize test.
    override func setUp()
    {
        setPythonLib()
        _ = MetalKernel.get
        MAKit.Opti.GPU = true
    }
    
    ///
    /// Compute the gradient norm on the first layer of the model.
    ///
    /// - Parameter model: The model we want to evaludate the gradient norm on.
    ///
    func _getGradientNorm(_ model: Model) -> Double
    {
        // Create the context to build a graph of layers
        // that come after the layers inside `model`.
        let context = ModelContext(name: "ModelTest", models: [model])
        let params = MAKit.Model.Params(context: context)
        
        // Append a loss layer.
        let lastLayer = MSE1D(
            layerPrev: model.layers.last! as! Layer1D,
            params: params
        )
        lastLayer.coeff = 1.0 / 2.0
        
        // Initialize the finalModel with the links (`layerPrev` updated).
        let finalModel = Model(model: context.model, modelsPrev: [model])
        
        // Initialize for inference.
        finalModel.initKernel(phase: .Inference)
        // The final model contains the layers of `model` and the loss layer.
        finalModel.layers = model.layers + context.model.layers
        
        let optimizerParams = getOptimizerParams(nbLoops: 1)
        finalModel.setupOptimizers(params: optimizerParams)
        
        let groundTruth: [[Double]] = [[0.0]]
        let firstLayer: Input2D = finalModel.layers.first as! Input2D
        
        // Update internal batch size.
        finalModel.updateKernel(batchSize: 1)
        
        // Forward.
        try! finalModel.forward()
        
        // Apply loss derivative.
        try! lastLayer.lossDerivativeGPU(groundTruth)
        
        // Backward.
        try! finalModel.backward()
        
        // Get the gradient norm on the first layer.
        let gradNormOutput: Double =
            try! finalModel.getGradientNorm(layers: [firstLayer])
        return gradNormOutput
    }
    
    /// Test that model1 backward pass returns the same gradient norm in MAKit and PyTorch.
    func testModel1()
    {
        // Build model.
        let model = ModelTest1.build(_size)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeTest1GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that model2 backward pass returns the same gradient norm in MAKit and PyTorch.
    func testModel2()
    {
        // Build model.
        let model = ModelTest2.build(_size)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeTest2GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that model3 backward pass returns the same gradient norm in MAKit and PyTorch.
    func testModel3()
    {
        // Build model.
        let model = ModelTest3.build(_size)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        let secondLayer: FTFrequences2D = model.layers[1] as! FTFrequences2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getComplexData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        try! secondLayer.setDataGPU(batchSize: 1)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeTest3GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
}
