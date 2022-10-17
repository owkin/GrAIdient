//
// CIFARTests.swift
// MADataTests
//
// Created by Jean-François Reboud on 16/10/2022.
//

import XCTest
import Foundation

let PYTHON_LIB =
    FileManager.default.homeDirectoryForCurrentUser.path +
    "/opt/anaconda3/envs/maexamples/lib/libpython3.7m.dylib"

func getPythonLib() -> String
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
        return url.path
    }
    else
    {
        return PYTHON_LIB
    }
}

final class CIFARTests: XCTestCase
{
    let _outputDir = NSTemporaryDirectory()
    
    let _batchSize = 256
    let _size = 32
    
    /*func testDumpLoad()
    {
        CIFAR.dumpDataset(pythonLibrary: _pythonLibrary,
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
        CIFAR.dumpTest(
            pythonLibrary: getPythonLib(),
            datasetOutputDir: _outputDir,
            label: 0,
            size: _size
        )
        _ = CIFAR.loadDataset(datasetDir: _outputDir, size: _size)
    }
}
