//
// MetalBuffer.swift
// MAKit
//
// Created by Jean-François Reboud on 04/10/2022.
//

import Metal

/// Abstract array of elements that can be sent to the GPU.
public class MetalBuffer<T>
{
    /// Number of elements in the array.
    public internal(set) var nbElems: Int
    /// GPU ID where the array will be sent.
    public internal(set) var deviceID: Int
    
    /// GPU resource cache.
    var _metal: MTLBuffer? = nil
    
    /// Get the GPU resource.
    public var metal: MTLBuffer
    {
        get {
            fatalError("Not implemented.")
        }
    }
    
    ///
    /// Create an object to interact with a GPU resource.
    ///
    /// - Parameters:
    ///     - nbElems: The number of elements in the array.
    ///     - deviceID: GPU ID where the array will be sent.
    ///
    public init(_ nbElems: Int, deviceID: Int)
    {
        self.deviceID = deviceID
        self.nbElems = nbElems
    }
}

///
/// An array of elements that can be sent to the GPU.
///
/// The GPU resource is intended to stay on the GPU itself (limited access with CPU).
///
public class MetalPrivateBuffer<T>: MetalBuffer<T>
{
    /// Cache toward a shared version.
    var _shared: MetalSharedBuffer<T>? = nil
    
    /// Build and get the GPU resource (no access with CPU).
    public override var metal: MTLBuffer
    {
        get {
            if _metal == nil
            {
                _metal = MetalKernel.get.createBuffer(
                    nbElems * MemoryLayout<T>.size,
                    options: MTLResourceOptions.storageModePrivate,
                    deviceID: deviceID
                )
            }
            return _metal!
        }
    }
    
    /// Get a GPU resource that may be synchorized with CPU.
    public var shared: MetalSharedBuffer<T>
    {
        get {
            if _shared == nil
            {
                _shared = MetalSharedBuffer<T>(nbElems, deviceID: deviceID)
            }
            return _shared!
        }
    }
}

///
/// An array of elements that can be sent to the GPU.
///
/// The GPU resource may be synchronized with CPU (memory is duplicated in the CPU and the GPU
/// when GPU and CPU hardware are not unified).
///
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
    
    /// Build and get a GPU resource that may be synchorized with CPU.
    public override var metal: MTLBuffer
    {
        get {
            let metalKernel = MetalKernel.get
            if _metal == nil
            {
                _metal = metalKernel.createBuffer(self)
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
    
    ///
    /// Create an object to interact with a GPU resource.
    ///
    /// - Parameters:
    ///     - nbElems: The number of elements in the array.
    ///     - deviceID: GPU ID where the array will be sent.
    ///
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

public extension UnsafeMutableRawPointer
{
    func toArray<T>(to type: T.Type, capacity count: Int) -> [T]
    {
        let pointer = bindMemory(to: type, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}
