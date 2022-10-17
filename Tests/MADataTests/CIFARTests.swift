//
// CIFARTests.swift
// MADataTests
//
// Created by Jean-François Reboud on 16/10/2022.
//

import XCTest

final class CIFARTests: XCTestCase
{
    static let _homeDir = FileManager.default.homeDirectoryForCurrentUser
    
    let _pythonLibrary =
        "\(_homeDir.path)/opt/anaconda3/envs/makit-test/lib/libpython3.7m.dylib"
    let _dataInputDir =
        "\(_homeDir.path)/DocumentsNonSync/Projet/Python/data/CIFAR"
    let _datasetOutputDir = "\(_homeDir.path)/DocumentsNonSync/Projet/Output"
    
    let _batchSize = 256
    let _size = 32
    
    /*func testDumpLoad()
    {
        CIFAR.dumpDataset(pythonLibrary: _pythonLibrary,
                          dataInputDir: _dataInputDir,
                          datasetOutputDir: _datasetOutputDir,
                          label: 0,
                          size: _size)
        _ = CIFAR.loadDataset(datasetDir: _datasetOutputDir, size: _size)
    }
    
    func testSamples()
    {
        let cifar = CIFAR.loadDataset(datasetDir: _datasetOutputDir,
                                      size: _size)
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
    
    func testShuffleSamples()
    {
        let cifar = CIFAR.loadDataset(datasetDir: _datasetOutputDir,
                                      size: _size)
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
    }*/
    
    func testDumpTest()
    {
        CIFAR.dumpTest(pythonLibrary: _pythonLibrary,
                       dataInputDir: _dataInputDir,
                       datasetOutputDir: _datasetOutputDir,
                       label: 0,
                       size: _size)
        _ = CIFAR.loadDataset(datasetDir: _datasetOutputDir, size: _size)
    }
}
