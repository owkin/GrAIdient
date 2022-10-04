//
// MetalBuffer.swift
// MAKit
//
// Created by Jean-François Reboud on 04/10/2022.
//

import Metal

public extension UnsafeMutableRawPointer
{
    func toArray<T>(to type: T.Type, capacity count: Int) -> [T]
    {
        let pointer = bindMemory(to: type, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}

public class MetalBuffer<T>
{
    let _nbElems: Int
    
    var _metal: MTLBuffer? = nil
    let _deviceID: Int
    
    var deviceID: Int
    {
        get {
            return _deviceID
        }
    }
    
    public var nbElems: Int
    {
        get {
            return _nbElems
        }
    }
    
    public var metal: MTLBuffer
    {
        get {
            fatalError("Not implemented.")
        }
    }
    
    public init(_ nbElems: Int, deviceID: Int)
    {
        _deviceID = deviceID
        _nbElems = nbElems
    }
}

public class MetalPrivateBuffer<T>: MetalBuffer<T>
{
    var _shared: MetalSharedBuffer<T>? = nil
    
    public override var metal: MTLBuffer
    {
        get {
            if _metal == nil
            {
                _metal = MetalKernel.get.createBuffer(
                    _nbElems * MemoryLayout<T>.size,
                    options: MTLResourceOptions.storageModePrivate,
                    deviceID: _deviceID)
            }
            return _metal!
        }
    }
    
    public var shared: MetalSharedBuffer<T>
    {
        get {
            if _shared == nil
            {
                _shared = MetalSharedBuffer<T>(nbElems, deviceID: _deviceID)
            }
            return _shared!
        }
    }
}

public class MetalSharedBuffer<T>: MetalBuffer<T>
{
    let _bufferPtr: UnsafeMutableBufferPointer<T>
    let _memory: UnsafeMutableRawPointer
    let _allocationSize: Int
    
    public var buffer: UnsafeMutableBufferPointer<T>
    {
        get {
            return _bufferPtr
        }
    }
    public var array: [T]
    {
        get {
            return _memory.toArray(to: T.self, capacity: nbElems)
        }
    }
    
    public override var metal: MTLBuffer
    {
        get {
            let metalKernel = MetalKernel.get
            if _metal == nil
            {
                _metal = metalKernel.createBuffer(
                    metalBuffer: self,
                    options: MTLResourceOptions.storageModeManaged,
                    deviceID: _deviceID)
            }
            return _metal!
        }
    }
    
    var allocationSize: Int
    {
        get {
            return _allocationSize
        }
    }
    var memory: UnsafeMutableRawPointer
    {
        get {
            return _memory
        }
    }
    
    public override init(_ nbElems: Int, deviceID: Int)
    {
        let byteLength = nbElems * MemoryLayout<T>.size
        
        var memory: UnsafeMutableRawPointer? = nil
        let alignment = 0x1000
        
        let allocationSize = (byteLength + alignment - 1) & (~(alignment - 1))
        posix_memalign(&memory, alignment, allocationSize)
        memset(memory, 0, byteLength)
        
        _memory = memory!
        _allocationSize = allocationSize
        
        let pointer = memory?.bindMemory(to: T.self, capacity: nbElems)
        _bufferPtr = UnsafeMutableBufferPointer(start: pointer,
                                                count: nbElems)
        super.init(nbElems, deviceID: deviceID)
    }
    
    deinit {
        free(_memory)
    }
}
