//
// Buffer.swift
// GrAIdient
//
// Created by Jean-François Reboud on 08/05/2024.
//

import Foundation
import Accelerate

///
/// Copy array to buffer.
///
/// - Parameters:
///     - array: input array
///     - buffer: output buffer
///     - start: start index in `array`
///     - nbElems: Number of elements to copy.
///
func copyFloatArrayToBuffer(
    array: inout [Float],
    buffer: UnsafeMutableBufferPointer<Float>,
    start: Int,
    nbElems: Int)
{
    if #available(macOS 13.0, *)
    {
        copyArrayToBuffer(
            array: &array,
            buffer: buffer,
            start: start, 
            nbElems: nbElems
        )
    }
    else
    {
        fatalError()
    }
}

@available(macOS 13.0, *)
///
/// Copy array to buffer.
///
/// - Parameters:
///     - array: input array
///     - buffer: output buffer
///     - start: start index in `array`
///     - nbElems: Number of elements to copy.
///
func copyArrayToBuffer<T: BNNSScalar>(
    array: inout [T],
    buffer: UnsafeMutableBufferPointer<T>,
    start: Int,
    nbElems: Int)
{
    var dest = BNNSNDArrayDescriptor(
        data: buffer,
        shape: .vector(nbElems)
    )!
    
    array.withUnsafeMutableBufferPointer
    {
        ptr in
        
        let base = ptr.baseAddress
        let offset = base?.advanced(by: start)
        let bufferPtr = UnsafeMutableBufferPointer<T>(
            start: offset, count: nbElems
        )
        
        var src = BNNSNDArrayDescriptor(
            data: bufferPtr,
            shape: .vector(nbElems)
        )!
        
        BNNSCopy(&dest, &src, nil)
    }
}
