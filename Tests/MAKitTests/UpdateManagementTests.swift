//
// UpdateManagementTests.swift
// MAKitTests
//
//  Created by Jean-François Reboud on 15/10/2022.
//

import XCTest
import MAKit
import MATestsUtils

class UpdateManagementTests: XCTestCase
{
    var optimizerParams = MAKit.Optimizer.Params()
    
    override func setUp()
    {
        _ = MetalKernel.get
        MAKit.Opti.GPU = true
        
        setOptimizerParams(params: &optimizerParams)
        optimizerParams.nbLoops = 1
    }
    
    // Test that we can choose to stop the backward pass
    // and update some parts of a model.
    // Use case: Input1D in CPU mode.
    func testUpdateManagement1DCPU() throws
    {
        MAKit.Opti.CPU = true
        let context = ModelContext(name: "model1", curID: 0)
        
        let params = MAKit.Model.Params(context: context)
        
        let layer1 = Input1D(nbNeurons: 1, params: params)
        
        let model1 = Model(model: context.model, modelsPrev: [])
        
        context.model = BaseModel(name: "model2")
        
        let layer2 = Input1D(layerPrev: layer1, params: params)
        
        let lastLayer = MSE1D(layerPrev: layer2, params: params)
        
        let model2 = Model(model: context.model, modelsPrev: [model1])
        
        model1.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        model2.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let groundTruth: [[Double]] = [[0.0]]
        
        let inputData1: [[Float]] = [[0.0]]
        let inputData2: [[Float]] = [[1.0]]
        
        // Test that by default, layer1 does not forward to layer2.
        
        try! layer1.setDataCPU(inputData1)
        try! layer2.setDataCPU(inputData2)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        var output1: Float = layer1.getOutsCPU(elem: 0)[0]
        var output2: Float = layer2.getOutsCPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 == 1.0)
        
        // Test that layer1 does forward to layer2.
        layer2.computeForward = true
        
        try! layer1.setDataCPU(inputData1)
        try! layer2.setDataCPU(inputData2)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        output1 = layer1.getOutsCPU(elem: 0)[0]
        output2 = layer2.getOutsCPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 == 0.0)
        
        // Test that layer1 and layer2 are not updated because
        // computeDeltaWeights == false for both of them.
        // Beware that by default layer2.computeDelta == true as it is
        // an intermediate layer.
        layer1.computeDelta = true
        layer2.computeForward = false
        
        try! layer1.setDataCPU(inputData1)
        try! layer2.setDataCPU(inputData2)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        try! lastLayer.applyGradientCPU(groundTruth)
        
        try! model2.backward()
        try! model1.backward()
        
        try! model1.update()
        try! model2.update()
        
        output1 = layer1.getOutsCPU(elem: 0)[0]
        output2 = layer2.getOutsCPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 == 1.0)
        
        // Test that both layer1 and layer2 are updated.
        layer1.computeDeltaWeights = true
        layer2.computeDeltaWeights = true
        
        try! layer1.setDataCPU(inputData1)
        try! layer2.setDataCPU(inputData2)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        try! lastLayer.applyGradientCPU(groundTruth)
        
        try! model2.backward()
        try! model1.backward()
        
        try! model1.update()
        try! model2.update()
        
        output1 = layer1.getOutsCPU(elem: 0)[0]
        output2 = layer2.getOutsCPU(elem: 0)[0]
        
        XCTAssert(output1 != 0.0 && output2 != 1.0)
        
        // Test that layer1 is not updated.
        layer1.computeDelta = false
        
        try! layer1.setDataCPU(inputData1)
        try! layer2.setDataCPU(inputData2)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        try! lastLayer.applyGradientCPU(groundTruth)
        
        try! model2.backward()
        try! model1.backward()
        
        try! model2.update()
        do
        {
            try model1.update()
            XCTAssert(false)
        }
        catch UpdateError.Dirty
        {
            XCTAssert(true)
        }
        
        output1 = layer1.getOutsCPU(elem: 0)[0]
        output2 = layer2.getOutsCPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 != 1.0)
    }
    
    // Test that we can choose to stop the backward pass
    // and update some parts of a model.
    // Use case: Input1D in GPU mode.
    func testUpdateManagement1DGPU() throws
    {
        let context = ModelContext(name: "model1", curID: 0)
        
        let params = MAKit.Model.Params(context: context)
        
        let layer1 = Input1D(nbNeurons: 1, params: params)
        
        let model1 = Model(model: context.model, modelsPrev: [])
        
        context.model = BaseModel(name: "model2")
        
        let layer2 = Input1D(layerPrev: layer1, params: params)
        
        let lastLayer = MSE1D(layerPrev: layer2, params: params)
        
        let model2 = Model(model: context.model, modelsPrev: [model1])
        
        model1.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        model2.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let groundTruth = MetalSharedBuffer<Float>(
            1, deviceID: DEVICE_ID
        )
        groundTruth.buffer[0] = 0
        MetalKernel.get.upload([groundTruth])
        
        let inputData1: [[Float]] = [[0.0]]
        let inputData2: [[Float]] = [[1.0]]
        
        // Test that by default, layer1 does not forward to layer2.
        
        try! layer1.setDataGPU(inputData1)
        try! layer2.setDataGPU(inputData2)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        var output1: Float = layer1.getOutsGPU(elem: 0)[0]
        var output2: Float = layer2.getOutsGPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 == 1.0)
        
        // Test that layer1 does forward to layer2.
        layer2.computeForward = true
        
        try! layer1.setDataGPU(inputData1)
        try! layer2.setDataGPU(inputData2)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        output1 = layer1.getOutsGPU(elem: 0)[0]
        output2 = layer2.getOutsGPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 == 0.0)
        
        // Test that layer1 and layer2 are not updated because
        // computeDeltaWeights == false for both of them.
        // Beware that by default layer2.computeDelta == true as it is
        // an intermediate layer.
        layer1.computeDelta = true
        layer2.computeForward = false
        
        try! layer1.setDataGPU(inputData1)
        try! layer2.setDataGPU(inputData2)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        try! lastLayer.applyGradientGPU(groundTruth, batchSize: 1)
        
        try! model2.backward()
        try! model1.backward()
        
        try! model1.update()
        try! model2.update()
        
        output1 = layer1.getOutsGPU(elem: 0)[0]
        output2 = layer2.getOutsGPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 == 1.0)
        
        // Test that both layer1 and layer2 are updated.
        layer1.computeDeltaWeights = true
        layer2.computeDeltaWeights = true
        
        try! layer1.setDataGPU(inputData1)
        try! layer2.setDataGPU(inputData2)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        try! lastLayer.applyGradientGPU(groundTruth, batchSize: 1)
        
        try! model2.backward()
        try! model1.backward()
        
        try! model1.update()
        try! model2.update()
        
        output1 = layer1.getOutsGPU(elem: 0)[0]
        output2 = layer2.getOutsGPU(elem: 0)[0]
        
        XCTAssert(output1 != 0.0 && output2 != 1.0)
        
        // Test that layer1 is not updated.
        layer1.computeDelta = false
        
        try! layer1.setDataGPU(inputData1)
        try! layer2.setDataGPU(inputData2)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        try! lastLayer.applyGradientGPU(groundTruth, batchSize: 1)
        
        try! model2.backward()
        try! model1.backward()
        
        try! model2.update()
        do
        {
            try model1.update()
            XCTAssert(false)
        }
        catch UpdateError.Dirty
        {
            XCTAssert(true)
        }
        
        output1 = layer1.getOutsGPU(elem: 0)[0]
        output2 = layer2.getOutsGPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 != 1.0)
    }
    
    // Test that we can choose to stop the backward pass
    // and update some parts of a model.
    // Use case: Input2D in CPU mode.
    func testUpdateManagement2DCPU() throws
    {
        MAKit.Opti.CPU = true
        let context = ModelContext(name: "model1", curID: 0)
        
        let params = MAKit.Model.Params(context: context)
        
        let layer1 = Input2D(nbChannels: 1, width: 1, height: 1, params: params)
        
        let model1 = Model(model: context.model, modelsPrev: [])
        
        context.model = BaseModel(name: "model2")
        
        let layer2 = Input2D(layerPrev: layer1, params: params)
        
        let layer3 = AvgPool2D(layerPrev: layer2, params: params)
        
        let lastLayer = MSE1D(layerPrev: layer3, params: params)
        
        let model2 = Model(model: context.model, modelsPrev: [model1])
        
        model1.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        model2.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let groundTruth: [[Double]] = [[0.0]]
        
        let inputData1: [Float] = [0.0]
        let inputData2: [Float] = [1.0]
        
        // Test that by default, layer1 does not forward to layer2.
        
        try! layer1.setDataCPU(inputData1, batchSize: 1, format: .Neuron)
        try! layer2.setDataCPU(inputData2, batchSize: 1, format: .Neuron)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        var output1: Float = layer1.getOutsCPU(elem: 0)[0]
        var output2: Float = layer2.getOutsCPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 == 1.0)
        
        // Test that layer1 does forward to layer2.
        layer2.computeForward = true
        
        try! layer1.setDataCPU(inputData1, batchSize: 1, format: .Neuron)
        try! layer2.setDataCPU(inputData2, batchSize: 1, format: .Neuron)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        output1 = layer1.getOutsCPU(elem: 0)[0]
        output2 = layer2.getOutsCPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 == 0.0)
        
        // Test that layer1 and layer2 are not updated because
        // computeDeltaWeights == false for both of them.
        // Beware that by default layer2.computeDelta == true as it is
        // an intermediate layer.
        layer1.computeDelta = true
        layer2.computeForward = false
        
        try! layer1.setDataCPU(inputData1, batchSize: 1, format: .Neuron)
        try! layer2.setDataCPU(inputData2, batchSize: 1, format: .Neuron)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        try! lastLayer.applyGradientCPU(groundTruth)
        
        try! model2.backward()
        try! model1.backward()
        
        try! model1.update()
        try! model2.update()
        
        output1 = layer1.getOutsCPU(elem: 0)[0]
        output2 = layer2.getOutsCPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 == 1.0)
        
        // Test that both layer1 and layer2 are updated.
        layer1.computeDeltaWeights = true
        layer2.computeDeltaWeights = true
        
        try! layer1.setDataCPU(inputData1, batchSize: 1, format: .Neuron)
        try! layer2.setDataCPU(inputData2, batchSize: 1, format: .Neuron)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        try! lastLayer.applyGradientCPU(groundTruth)
        
        try! model2.backward()
        try! model1.backward()
        
        try! model1.update()
        try! model2.update()
        
        output1 = layer1.getOutsCPU(elem: 0)[0]
        output2 = layer2.getOutsCPU(elem: 0)[0]
        
        XCTAssert(output1 != 0.0 && output2 != 1.0)
        
        // Test that layer1 is not updated.
        layer1.computeDelta = false
        
        try! layer1.setDataCPU(inputData1, batchSize: 1, format: .Neuron)
        try! layer2.setDataCPU(inputData2, batchSize: 1, format: .Neuron)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        try! lastLayer.applyGradientCPU(groundTruth)
        
        try! model2.backward()
        try! model1.backward()
        
        try! model2.update()
        do
        {
            try model1.update()
            XCTAssert(false)
        }
        catch UpdateError.Dirty
        {
            XCTAssert(true)
        }
        
        output1 = layer1.getOutsCPU(elem: 0)[0]
        output2 = layer2.getOutsCPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 != 1.0)
    }
    
    // Test that we can choose to stop the backward pass
    // and update some parts of a model.
    // Use case: Input2D in GPU mode.
    func testUpdateManagement2DGPU() throws
    {
        let context = ModelContext(name: "model1", curID: 0)
        
        let params = MAKit.Model.Params(context: context)
        
        let layer1 = Input2D(nbChannels: 1, width: 1, height: 1, params: params)
        
        let model1 = Model(model: context.model, modelsPrev: [])
        
        context.model = BaseModel(name: "model2")
        
        let layer2 = Input2D(layerPrev: layer1, params: params)
        
        let layer3 = AvgPool2D(layerPrev: layer2, params: params)
        
        let lastLayer = MSE1D(layerPrev: layer3, params: params)
        
        let model2 = Model(model: context.model, modelsPrev: [model1])
        
        model1.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        model2.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID
        )
        
        let groundTruth = MetalSharedBuffer<Float>(
            1, deviceID: DEVICE_ID
        )
        groundTruth.buffer[0] = 0
        MetalKernel.get.upload([groundTruth])
        
        let inputData1: [Float] = [0.0]
        let inputData2: [Float] = [1.0]
        
        // Test that by default, layer1 does not forward to layer2.
        
        try! layer1.setDataGPU(inputData1, batchSize: 1, format: .Neuron)
        try! layer2.setDataGPU(inputData2, batchSize: 1, format: .Neuron)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        var output1: Float = layer1.getOutsGPU(elem: 0)[0]
        var output2: Float = layer2.getOutsGPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 == 1.0)
        
        // Test that layer1 does forward to layer2.
        layer2.computeForward = true
        
        try! layer1.setDataGPU(inputData1, batchSize: 1, format: .Neuron)
        try! layer2.setDataGPU(inputData2, batchSize: 1, format: .Neuron)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        output1 = layer1.getOutsGPU(elem: 0)[0]
        output2 = layer2.getOutsGPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 == 0.0)
        
        // Test that layer1 and layer2 are not updated because
        // computeDeltaWeights == false for both of them.
        // Beware that by default layer2.computeDelta == true as it is
        // an intermediate layer.
        layer1.computeDelta = true
        layer2.computeForward = false
        
        try! layer1.setDataGPU(inputData1, batchSize: 1, format: .Neuron)
        try! layer2.setDataGPU(inputData2, batchSize: 1, format: .Neuron)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        try! lastLayer.applyGradientGPU(groundTruth, batchSize: 1)
        
        try! model2.backward()
        try! model1.backward()
        
        try! model1.update()
        try! model2.update()
        
        output1 = layer1.getOutsGPU(elem: 0)[0]
        output2 = layer2.getOutsGPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 == 1.0)
        
        // Test that both layer1 and layer2 are updated.
        layer1.computeDeltaWeights = true
        layer2.computeDeltaWeights = true
        
        try! layer1.setDataGPU(inputData1, batchSize: 1, format: .Neuron)
        try! layer2.setDataGPU(inputData2, batchSize: 1, format: .Neuron)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        try! lastLayer.applyGradientGPU(groundTruth, batchSize: 1)
        
        try! model2.backward()
        try! model1.backward()
        
        try! model1.update()
        try! model2.update()
        
        output1 = layer1.getOutsGPU(elem: 0)[0]
        output2 = layer2.getOutsGPU(elem: 0)[0]
        
        XCTAssert(output1 != 0.0 && output2 != 1.0)
        
        // Test that layer1 is not updated.
        layer1.computeDelta = false
        
        try! layer1.setDataGPU(inputData1, batchSize: 1, format: .Neuron)
        try! layer2.setDataGPU(inputData2, batchSize: 1, format: .Neuron)
        
        model1.updateKernel(batchSize: 1)
        model2.updateKernel(batchSize: 1)
        
        try! model1.forward()
        try! model2.forward()
        
        try! lastLayer.applyGradientGPU(groundTruth, batchSize: 1)
        
        try! model2.backward()
        try! model1.backward()
        
        try! model2.update()
        do
        {
            try model1.update()
            XCTAssert(false)
        }
        catch UpdateError.Dirty
        {
            XCTAssert(true)
        }
        
        output1 = layer1.getOutsGPU(elem: 0)[0]
        output2 = layer2.getOutsGPU(elem: 0)[0]
        
        XCTAssert(output1 == 0.0 && output2 != 1.0)
    }
}
