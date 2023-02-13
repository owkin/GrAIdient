//
// ConcurrencyTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 07/12/2022.
//

import XCTest
import GrAIdient

/// Test Concurrency utils.
class ConcurrencyTests: XCTestCase
{
    private func _testArray(_ data: inout [Int], nbElems: Int)
    {
        data.withUnsafeMutableBufferPointer
        {
            dataPtr in
            Concurrency.slice(nbElems)
            {
                (elem: Int) in
                dataPtr[elem] = elem
            }
        }
        Concurrency.slice(nbElems)
        {
            (elem: Int) in
            XCTAssert(data[elem] == elem)
        }
    }
    
    func testBigSlice()
    {
        let nbElems = 5000
        var data = [Int](repeating: 0, count: nbElems)
        _testArray(&data, nbElems: nbElems)
    }
    
    func testSmallSlice()
    {
        let nbElems = 3
        var data = [Int](repeating: 0, count: nbElems)
        _testArray(&data, nbElems: nbElems)
    }
    
    func testVerySmallSlice()
    {
        let nbElems = 1
        var data = [Int](repeating: 0, count: nbElems)
        _testArray(&data, nbElems: nbElems)
    }
    
    func testNoneSlice()
    {
        let nbElems = 1
        var data = [Int](repeating: 0, count: nbElems)
        _testArray(&data, nbElems: 0)
    }
}
