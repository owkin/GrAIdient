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
    
    let _batchSize = 64
    let _size = 32
    
    override func setUp()
    {
        setPythonLib()
        MAKit.Opti.GPU = true
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
        
        layer = MaxPool2D(layerPrev: layer, size: 2, stride: 2, params: params)
        
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
            layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        layer = Convolution2D(
            layerPrev: layer, size: 3, nbChannels: 8, stride: 1,
            activation: ReLU.str, biases: true, bn: bn,
            params: params
        )
        
        var head: Layer1D = AvgPool2D(layerPrev: layer, params: params)

        head = FullyConnected(
            layerPrev: head, nbNeurons: 1,
            activation: ReLU.str, biases: true, params: params
        )
        
        _ = MSE1D(layerPrev: head, params: params)
        
        return Model(model: context.model, modelsPrev: [])
    }
    
    func _preprocess(
        _ data: [UInt8],
        mean: (Double, Double, Double),
        imageFormat: ImageFormat) -> [Double]
    {
        let batchSize = data.count / (_size * _size * 3)
        var newData = [Double](repeating: 0.0, count: data.count)
        
        switch imageFormat
        {
        case .RGB:
            for elem in 0..<batchSize
            {
                for i in 0..<_size {
                for j in 0..<_size
                {
                    let offset = j + (elem * _size + i) * _size
                    
                    for channel in 0..<3
                    {
                        let valeur = Double(data[3 * offset + channel])
                        switch channel
                        {
                        case 0:
                            newData[3 * offset + channel] =
                                (valeur - mean.0) / 255
                        case 1:
                            newData[3 * offset + channel] =
                                (valeur - mean.1) / 255
                        case 2:
                            newData[3 * offset + channel] =
                                (valeur - mean.2) / 255
                        default:
                            break
                        }
                    }
                }}
            }
        case .Neuron:
            for elem in 0..<batchSize
            {
                for i in 0..<_size {
                for j in 0..<_size
                {
                    let offsetSet = j + (elem * _size + i) * _size
                    
                    for channel in 0..<3
                    {
                        let offsetDebutGet = (channel + 3 * elem) * _size
                        let offsetGet = j + (offsetDebutGet + i) * _size
                        
                        let valeur = Double(data[offsetGet])
                        switch channel
                        {
                        case 0:
                            newData[3 * offsetSet + channel] =
                                (valeur - mean.0) / 255
                        case 1:
                            newData[3 * offsetSet + channel] =
                                (valeur - mean.1) / 255
                        case 2:
                            newData[3 * offsetSet + channel] =
                                (valeur - mean.2) / 255
                        default:
                            break
                        }
                    }
                }}
            }
        }
        return newData
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
        
        let params = _getOptimizerParams(nbLoops: _batchSize)
        cifar8.initSamples(batchSize: _batchSize / 2)
        cifar5.initSamples(batchSize: _batchSize / 2)
        cifar8.keep(500)
        cifar5.keep(500)
        
        let vgg = _buildModel(bn: true)
        vgg.initialize(params: params, phase: .Training)
        
        let firstLayer: Input2D = vgg.layers.first as! Input2D
        let lastLayer: MSE1D = vgg.layers.last as! MSE1D
        
        let mean = (123.675, 116.28, 103.53)
        for epoch in 0..<5
        {
            print("EPOCH \(epoch)/19.")
            cifar8.shuffle()
            cifar5.shuffle()
            
            for step in 0..<cifar8.nbLoops
            {
                let samples8 = cifar8.getSamples()!
                let samples5 = cifar5.getSamples()!
                
                let flat8 = samples8.reduce([], +)
                let flat5 = samples5.reduce([], +)
                let data = _preprocess(
                    flat8 + flat5,
                    mean: mean,
                    imageFormat: .Neuron
                )
                
                var gt8 = [[Double]]()
                for _ in 0..<samples8.count
                {
                    gt8.append([0.0])
                }
                var gt5 = [[Double]]()
                for _ in 0..<samples5.count
                {
                    gt5.append([1.0])
                }
                let gt = gt8 + gt5
                
                vgg.updateKernel(batchSize: gt.count)
                try! firstLayer.setDataGPU(
                    data,
                    batchSize: gt.count,
                    format: .Neuron
                )
                
                try! vgg.forward()
                try! lastLayer.applyGradientGPU(gt)
                try! vgg.backward()
                try! vgg.update()
                
                let loss = try! lastLayer.getLossGPU(gt)
                print("Step \(step)/\(cifar8.nbLoops-1): \(sqrt(loss)).")
                
                vgg.incStep()
            }
            vgg.incEpoch()
        }
    }
}
