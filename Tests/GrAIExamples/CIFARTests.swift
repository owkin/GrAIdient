//
// CIFARTests.swift
// GrAIExamples
//
// Created by Jean-François Reboud on 16/10/2022.
//

import XCTest

/// Test that we can sample the CIFAR dataset.
final class CIFARTests: XCTestCase
{
    /// Directory to dump outputs from the tests.
    let _outputDir = NSTemporaryDirectory()
    
    /// Batch size of data.
    let _batchSize = 256
    /// Size of one image (height and width are the same).
    let _size = 32
    
    /// Mean of the preprocessing to apply to data.
    let _mean: (Float, Float, Float) = (123.675, 116.28, 103.53)
    /// Deviation of the preprocessing to apply to data.
    let _std: (Float, Float, Float) = (58.395, 57.12, 57.375)
    
    /// Initialize test.
    override func setUp()
    {
        setPythonLib()
    }
    
    /// Test1: dump training dataset and load it.
    func test1_DumpLoad()
    {
        let datasetPath = _outputDir + "/datasetTrain"
        CIFAR.dumpTrain(
            datasetPath: datasetPath,
            label: 0,
            size: _size
        )
        _ = CIFAR.loadDataset(
            datasetPath: datasetPath,
            size: _size
        )
    }
    
    /// Test2: load training dataset and get samples until the last one.
    func test2_Samples()
    {
        let datasetPath = _outputDir + "/datasetTrain"
        let cifar = CIFAR.loadDataset(
            datasetPath: datasetPath,
            size: _size
        )
        cifar.initSamples(batchSize: _batchSize)
        
        var nbLoops = 0
        var lastLoop = false
        while let samples = cifar.getSamples()
        {
            XCTAssert(!lastLoop)
            if samples.count != _batchSize
            {
                lastLoop = true
            }
            else
            {
                XCTAssert(samples.count == _batchSize)
            }
            
            for sample in samples
            {
                XCTAssert(sample.count == cifar.sizeDataBlock)
            }
            nbLoops += 1
        }
        print("Number of loops per epoch: " + String(nbLoops))
        XCTAssert(nbLoops == cifar.nbLoops)
        XCTAssert(cifar.getSamples() == nil)
    }
    
    /// Test3: load training dataset, shuffle and get samples until the last one.
    func test3_ShuffleSamples()
    {
        let datasetPath = _outputDir + "/datasetTrain"
        let cifar = CIFAR.loadDataset(
            datasetPath: datasetPath,
            size: _size
        )
        cifar.initSamples(batchSize: _batchSize)
        
        var nbLoops = 0
        var lastLoop = false
        while let samples = cifar.getSamples()
        {
            XCTAssert(!lastLoop)
            if samples.count != _batchSize
            {
                lastLoop = true
            }
            else
            {
                XCTAssert(samples.count == _batchSize)
            }
            
            for sample in samples
            {
                XCTAssert(sample.count == cifar.sizeDataBlock)
            }
            nbLoops += 1
        }
        print("Number of loops per epoch: " + String(nbLoops))
        XCTAssert(nbLoops == cifar.nbLoops)
    }
    
    /// Test4: dump testing dataset and load it.
    func test4_DumpTest()
    {
        let datasetPath = _outputDir + "/datasetTest"
        CIFAR.dumpTest(
            datasetPath: datasetPath,
            label: 0,
            size: _size
        )
        _ = CIFAR.loadDataset(
            datasetPath: datasetPath,
            size: _size
        )
    }
    
    /// Test5: iterate on CIFAR, preprocess and compare with PyTorch results.
    func test5_PreprocessSamples()
    {
        let cifar = CIFAR.loadDataset(
            datasetPath: _outputDir + "/datasetTrain",
            size: _size
        )
        cifar.initSamples(batchSize: _batchSize)
        
        let iterator = CIFAR.buildIterator(
            train: true,
            batchSize: _batchSize,
            label: 0,
            shuffle: false
        )
        
        var nbLoops = 0
        var lastLoop = false
        var batchSize = 0
        var samples2 = [Float]()
        
        while let samples1 = cifar.getSamples()
        {
            (samples2, batchSize) = CIFAR.getSamples(iterator)
            
            XCTAssert(!lastLoop)
            if samples1.count != _batchSize
            {
                lastLoop = true
            }
            else
            {
                XCTAssert(samples1.count == _batchSize)
                XCTAssert(batchSize == _batchSize)
            }
            
            // Pre processing.
            let data: [Float] = preprocess(
                samples1,
                height: _size,
                width: _size,
                mean: _mean,
                std: _std,
                imageFormat: .Neuron
            )
            
            for (elem1, elem2) in zip(data, samples2)
            {
                XCTAssertEqual(elem1, elem2, accuracy: 0.0001)
            }
            nbLoops += 1
        }
        
        print("Number of loops per epoch: " + String(nbLoops))
        XCTAssert(nbLoops == cifar.nbLoops)
        XCTAssert(cifar.getSamples() == nil)
        
        (samples2, batchSize) = CIFAR.getSamples(iterator)
        XCTAssert(samples2.count == 0)
        XCTAssert(batchSize == 0)
    }
}
