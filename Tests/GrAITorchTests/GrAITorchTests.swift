//
// GrAITorchTests.swift
// GrAITorchTests
//
// Created by Jean-François Reboud on 19/10/2022.
//

import XCTest
import GrAIdient

/// Compare models created by GrAIdient and PyTorch.
final class GrAITorchTests: XCTestCase
{
    /// Size of one image (height and width are the same).
    let _size = 32
    /// Kernel split size of one image (height and width are the same).
    let _patch = 8
    
    /// Initialize test.
    override func setUp()
    {
        setPythonLib()
        _ = MetalKernel.get
        GrAI.Opti.GPU = true
    }
    
    ///
    /// Compute the gradient norm on the first layer of the model.
    ///
    /// - Parameter model: The model we want to evalulate the gradient norm on.
    /// - Returns: The gradient norm on the first layer.
    ///
    func _getGradientNorm(_ model: Model) -> Double
    {
        // Create the context to build a graph of layers
        // that come after the layers inside `model`.
        let context = ModelContext(name: "ModelTest", models: [model])
        let params = GrAI.Model.Params(context: context)
        
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
    
    /// Test that model1 backward pass returns the same gradient norm in GrAIdient and PyTorch.
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
    
    /// Test that model2 backward pass returns the same gradient norm in GrAIdient and PyTorch.
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
    
    /// Test that model3 backward pass returns the same gradient norm in GrAIdient and PyTorch.
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
    
    /// Test that model4 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModel4()
    {
        // Build model.
        let model = ModelTest4.build(_size)
        
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
        let expectedNorm: Double = Double(computeTest4GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that model5 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModel5()
    {
        // Build model.
        let model = ModelTest5.build(_size)
        
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
        let expectedNorm: Double = Double(computeTest5GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that model6 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModel6()
    {
        // Build model.
        let model = ModelTest6.build(_size)
        
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
        let expectedNorm: Double = Double(computeTest6GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that model7 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModel7()
    {
        // Build model.
        let model = ModelTest7.build(_size)
        
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
        let expectedNorm: Double = Double(computeTest7GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that model8 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModel8()
    {
        // Build model.
        let model = ModelTest8.build(_size)
        
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
        let expectedNorm: Double = Double(computeTest8GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that model9 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModel9()
    {
        // Build model.
        let model = ModelTest9.build(size: _size, patch: _patch)
        
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
        let expectedNorm: Double = Double(computeTest9GradNorm(
            size: _size, patch: _patch
        ))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that model10 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModel10()
    {
        // Build model.
        let model = ModelTest10.build(size: _size, patch: _patch)
        
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
        let expectedNorm: Double = Double(computeTest10GradNorm(
            size: _size, patch: _patch
        ))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
}
