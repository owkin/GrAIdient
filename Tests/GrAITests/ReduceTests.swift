//
// ReduceTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 17/05/2023.
//

import XCTest
import GrAIdient

/// Test reduce sum kernel.
class ReduceSumTests: XCTestCase
{
    var _buffer: FloatBuffer! = nil
    var _array = [Float]()
    
    override func setUp()
    {
        _ = MetalKernel.get
        GrAI.Opti.GPU = true
    }
    
    private func _testBuffer(dim1: Int, dim2: Int, shared: Bool)
    {
        _array = [Float](repeating: 0.0, count: dim1 * dim2)
        _buffer = FloatBuffer(nbElems: dim1 * dim2, deviceID: 0, shared: shared)
        
        for elem1 in 0..<dim1 {
        for elem2 in 0..<dim2
        {
            let offset = elem2 * dim1 + elem1
            let value = Float.random(in: 0..<1)
            _array[offset] = value
        }}
        _buffer.initialize(array: &_array)
        
        var resultsCPU = [Float]()
        for elem2 in 0..<dim2
        {
            var sum: Float = 0.0
            for elem1 in 0..<dim1
            {
                let offset = elem2 * dim1 + elem1
                sum += _array[offset]
            }
            resultsCPU.append(sum)
        }
        
        reduceSum(
            inBuffer: _buffer.metal,
            outBuffer: _buffer.metal,
            dim1: dim1, dim2: dim2,
            deviceID: 0
        )
        
        let resultsGPU = _buffer.download()
        for (resultCPU, resultGPU) in zip(resultsCPU, resultsGPU)
        {
            let diffPercent =
                abs(resultCPU - resultGPU) / resultCPU * 100.0
            XCTAssert(diffPercent < 0.1)
        }
    }
    
    func testVerySmallFloat()
    {
        let dim1 = 2
        let dim2 = 5
        GrAI.Precision.float = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testVerySmallFloat16()
    {
        let dim1 = 2
        let dim2 = 5
        GrAI.Precision.float16 = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testSmallFloat()
    {
        let dim1 = 50
        let dim2 = 5
        GrAI.Precision.float = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testSmallFloat16()
    {
        let dim1 = 50
        let dim2 = 5
        GrAI.Precision.float16 = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testBigFloat()
    {
        let dim1 = 2000
        let dim2 = 5
        GrAI.Precision.float = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testBigFloat16()
    {
        let dim1 = 2000
        let dim2 = 5
        GrAI.Precision.float16 = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testVeryBigFloat()
    {
        let dim1 = 10000
        let dim2 = 5
        GrAI.Precision.float = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testVeryBigFloat16()
    {
        let dim1 = 10000
        let dim2 = 5
        GrAI.Precision.float16 = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
}

/// Test reduce max kernel.
class ReduceMaxTests: XCTestCase
{
    var _buffer: FloatBuffer! = nil
    var _array = [Float]()
    
    override func setUp()
    {
        _ = MetalKernel.get
    }
    
    private func _testBuffer(dim1: Int, dim2: Int, shared: Bool)
    {
        _array = [Float](repeating: 0.0, count: dim1 * dim2)
        _buffer = FloatBuffer(nbElems: dim1 * dim2, deviceID: 0, shared: shared)
        
        for elem1 in 0..<dim1 {
        for elem2 in 0..<dim2
        {
            let offset = elem2 * dim1 + elem1
            let value = Float.random(in: 0..<1)
            _array[offset] = value
        }}
        _buffer.initialize(array: &_array)
        
        var resultsCPU = [Float]()
        for elem2 in 0..<dim2
        {
            var val: Float = _array[elem2 * dim1]
            for elem1 in 0..<dim1
            {
                let offset = elem2 * dim1 + elem1
                val = max(_array[offset], val)
            }
            resultsCPU.append(val)
        }
        
        reduceMax(
            inBuffer: _buffer.metal,
            outBuffer: _buffer.metal,
            dim1: dim1, dim2: dim2,
            deviceID: 0
        )
        
        let resultsGPU = _buffer.download()
        for (resultCPU, resultGPU) in zip(resultsCPU, resultsGPU)
        {
            let diffPercent =
                abs(resultCPU - resultGPU) / resultCPU * 100.0
            XCTAssert(diffPercent < 0.05)
        }
    }
    
    func testVerySmallFloat()
    {
        let dim1 = 2
        let dim2 = 5
        GrAI.Precision.float = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testVerySmallFloat16()
    {
        let dim1 = 2
        let dim2 = 5
        GrAI.Precision.float16 = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testSmallFloat()
    {
        let dim1 = 50
        let dim2 = 5
        GrAI.Precision.float = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testSmallFloat16()
    {
        let dim1 = 50
        let dim2 = 5
        GrAI.Precision.float16 = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testBigFloat()
    {
        let dim1 = 2000
        let dim2 = 5
        GrAI.Precision.float = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testBigFloat16()
    {
        let dim1 = 2000
        let dim2 = 5
        GrAI.Precision.float16 = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testVeryBigFloat()
    {
        let dim1 = 10000
        let dim2 = 5
        GrAI.Precision.float = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
    
    func testVeryBigFloat16()
    {
        let dim1 = 10000
        let dim2 = 5
        GrAI.Precision.float16 = true
        _testBuffer(dim1: dim1, dim2: dim2, shared: true)
        _testBuffer(dim1: dim1, dim2: dim2, shared: false)
    }
}
