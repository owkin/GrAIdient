//
// VGGExample.swift
// MAExamples
//
// Created by Jean-François Reboud on 17/10/2022.
//

import XCTest
import MAKit

final class VGGExample: XCTestCase
{
    let _outputDir = NSTemporaryDirectory()
    
    let _batchSize = 256
    let _size = 32
    
    override func setUp()
    {
        setPythonLib()
        MAKit.Opti.CPU = true
    }
    
    func _getOptimizerParams(nbLoops: Int) -> MAKit.Optimizer.Params
    {
        var optimizerParams = MAKit.Optimizer.Params()
        optimizerParams.nbLoops = nbLoops
        
        optimizerParams.optimizer = ConstEpochsScheduler(
            MAKit.Optimizer.Class.AdamRectified
        )
        
        optimizerParams.variables["alpha"] = ConstEpochsVar(
            value: ConstVal(1e-3)
        )
        optimizerParams.variables["lambda"] = ConstEpochsVar(
            value: ConstVal(1e-6)
        )
        return optimizerParams
    }
    
    func _buildModel(bn: Bool) -> Model
    {
        let context = ModelContext(name: "VGG", models: [])
        let params = MAKit.Model.Params(context: context)
        
        var layer: Layer2D
        layer = Input2D(
            nbChannels: 3,
            width: _size, height: _size,
            params: params
        )
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        layer = MaxPool2D(layerPrev: layer, size: 2, stride: 2, params: params)
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 16, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 16, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        layer = MaxPool2D(layerPrev: layer, size: 2, stride: 2, params: params)
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 32, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 32, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 32, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        layer = MaxPool2D(layerPrev: layer, size: 2, stride: 2, params: params)
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 64, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 64, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 64, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        layer = MaxPool2D(layerPrev: layer, size: 2, stride: 2, params: params)
        
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 64, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 64, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 64, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(layerPrev: layer, params: params)
        
        head = FullyConnected(
            layerPrev: head, nbNeurons: 512,
            activation: ReLU.str, biases: true, params: params
        )
        head = FullyConnected(
            layerPrev: head, nbNeurons: 512,
            activation: ReLU.str, biases: true, params: params
        )
        head = FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: ReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
        
        return Model(model: context.model, modelsPrev: [])
    }
    
    func testTrainVGG()
    {
        CIFAR.dumpTrain(
            datasetPath: _outputDir + "/dataset8",
            label: 8,
            size: _size
        )
        CIFAR.dumpTrain(
            datasetPath: _outputDir + "/dataset5",
            label: 5,
            size: _size
        )
        let cifar8 = CIFAR.loadDataset(
            datasetPath: _outputDir + "/dataset8",
            size: _size
        )
        let cifar5 = CIFAR.loadDataset(
            datasetPath: _outputDir + "/dataset5",
            size: _size
        )
        
        let minTmp = min(cifar8.nbSamples, cifar5.nbSamples)
        cifar8.keep(minTmp)
        cifar5.keep(minTmp)
        
        let params = _getOptimizerParams(nbLoops: _batchSize)
        cifar8.initSamples(batchSize: _batchSize / 2)
        cifar5.initSamples(batchSize: _batchSize / 2)
        
        let vgg = _buildModel(bn: true)
        vgg.initialize(params: params, phase: .Training)
        
        for epoch in 0..<20
        {
            cifar8.shuffle()
            cifar5.shuffle()
            
            for numLoop in 0..<cifar8.nbLoops
            {
                let samples8 = cifar8.getSamples()!
                let samples5 = cifar5.getSamples()!
                
                vgg.updateKernel(batchSize: samples8.count)
            }
        }
    }
}
