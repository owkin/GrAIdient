//
// CIFARTests.swift
// MAExamples
//
// Created by Jean-François Reboud on 16/10/2022.
//

import XCTest

let PYTHON_LIB =
    FileManager.default.homeDirectoryForCurrentUser.path +
    "/opt/anaconda3/envs/maexamples/lib/libpython3.7m.dylib"

func setPythonLib()
{
    if ProcessInfo.processInfo.environment["PYTHON_LIBRARY"] == nil
    {
        let task = Process()
        task.launchPath = "/usr/bin/which"
        task.arguments = ["python"]
        
        let pipe = Pipe()
        task.standardOutput = pipe
        task.launch()
        task.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: String.Encoding.utf8)!
        
        if output.count > 0
        {
            var url = URL(fileURLWithPath: output)
            url = url.deletingLastPathComponent().deletingLastPathComponent()
            url = url.appendingPathComponent("lib")
            url = url.appendingPathComponent("libpython3.7m.dylib")
            setenv("PYTHON_LIBRARY", url.path, 1)
        }
        else
        {
            setenv("PYTHON_LIBRARY", PYTHON_LIB, 1)
        }
    }
}

final class CIFARTests: XCTestCase
{
    let _outputDir = NSTemporaryDirectory()
    
    let _batchSize = 256
    let _size = 32
    
    override func setUp()
    {
        setPythonLib()
    }
    
    func testDumpLoad()
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
    
    func testSamples()
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
    
    func testShuffleSamples()
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
    
    func testDumpTest()
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
}
