//
// MetalBuffer.swift
// GrAIdient
//
// Created by Jean-François Reboud on 04/10/2022.
//

import Metal

/// Wrapper of Metal float buffer.
public class FloatBuffer
{
    /// Number of elements in the buffer.
    public let nbElems: Int
    /// GPU device where the buffer is sent.
    public let deviceID: Int
    /// Whether to create a shared buffer or a private one.
    public let shared: Bool
    
    /// Float buffer.
    var _float: MetalBuffer<Float>? = nil
    /// Float16 buffer.
    var _float16: MetalBuffer<UInt16>? = nil
    
    /// Get Metal buffer.
    public var metal: MTLBuffer
    {
        get {
            if GrAI.Precision.float16
            {
                if _float16 == nil
                {
                    if shared
                    {
                        _float16 = MetalSharedBuffer<UInt16>(
                            nbElems, deviceID: deviceID
                        )
                    }
                    else
                    {
                        _float16 = MetalPrivateBuffer<UInt16>(
                            nbElems, deviceID: deviceID
                        )
                    }
                }
                return _float16!.metal
            }
            else
            {
                if _float == nil
                {
                    if shared
                    {
                        _float = MetalSharedBuffer<Float>(
                            nbElems, deviceID: deviceID
                        )
                    }
                    else
                    {
                        _float = MetalPrivateBuffer<Float>(
                            nbElems, deviceID: deviceID
                        )
                    }
                }
                return _float!.metal
            }
        }
    }
    
    ///
    /// Create a wrapper of Metal buffer.
    ///
    /// - Parameters:
    ///     - nbElems: The number of elements in the array.
    ///     - deviceID: GPU ID where the array will be sent.
    ///     - shared: Whether to create a shared buffer or a private one.
    ///
    public init(nbElems: Int, deviceID: Int, shared: Bool = false)
    {
        self.deviceID = deviceID
        self.nbElems = nbElems
        self.shared = shared
    }
    
    /// Clean the buffers.
    func reset()
    {
        _float = nil
        _float16 = nil
    }
    
    /// Initialize Metal buffer.
    public func initialize()
    {
        if GrAI.Precision.float16
        {
            if _float16 == nil
            {
                if shared
                {
                    _float16 = MetalSharedBuffer<UInt16>(
                        nbElems, deviceID: deviceID
                    )
                }
                else
                {
                    let buffer = MetalPrivateBuffer<UInt16>(
                        nbElems, deviceID: deviceID
                    )
                    _float16 = buffer
                    _ = buffer.shared
                }
            }
            _float16!.upload()
        }
        else
        {
            if _float == nil
            {
                if shared
                {
                    _float = MetalSharedBuffer<Float>(
                        nbElems, deviceID: deviceID
                    )
                }
                else
                {
                    let buffer = MetalPrivateBuffer<Float>(
                        nbElems, deviceID: deviceID
                    )
                    _float = buffer
                    _ = buffer.shared
                }
            }
            _float!.upload()
        }
    }
    
    ///
    /// Initialize Metal buffer.
    ///
    /// - Parameters:
    ///     - array: Input array.
    ///     - start: Start offset.
    ///
    public func initialize(
        array: inout [Float],
        start: Int = 0)
    {
        if GrAI.Precision.float16
        {
            if _float16 == nil
            {
                if shared
                {
                    _float16 = MetalSharedBuffer<UInt16>(
                        nbElems, deviceID: deviceID
                    )
                }
                else
                {
                    _float16 = MetalPrivateBuffer<UInt16>(
                        nbElems, deviceID: deviceID
                    )
                }
            }
            setupHalfBuffer(
                array: &array,
                out: _float16!,
                start: start,
                nbElems: nbElems,
                deviceID: deviceID
            )
        }
        else
        {
            if _float == nil
            {
                if shared
                {
                    _float = MetalSharedBuffer<Float>(
                        nbElems, deviceID: deviceID
                    )
                }
                else
                {
                    _float =  MetalPrivateBuffer<Float>(
                        nbElems, deviceID: deviceID
                    )
                }
            }
            setupFloatBuffer(
                array: &array,
                out: _float!,
                start: start,
                nbElems: nbElems,
                deviceID: deviceID
            )
        }
    }
    
    /// Retrieve Metal buffer content.
    public func download() -> [Float]
    {
        if GrAI.Precision.float16
        {
            if _float16 == nil
            {
                if shared
                {
                    _float16 = MetalSharedBuffer<UInt16>(
                        nbElems, deviceID: deviceID
                    )
                }
                else
                {
                    _float16 = MetalPrivateBuffer<UInt16>(
                        nbElems, deviceID: deviceID
                    )
                }
            }
            return getHalfBuffer(_float16!).array
        }
        else
        {
            if _float == nil
            {
                if shared
                {
                    _float = MetalSharedBuffer<Float>(
                        nbElems, deviceID: deviceID
                    )
                }
                else
                {
                    _float = MetalPrivateBuffer<Float>(
                        nbElems, deviceID: deviceID
                    )
                }
            }
            return [Float](_float!.download())
        }
    }
}

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
    
    ///
    /// Download the content of the buffer to the CPU.
    ///
    /// - Returns: the CPU buffer.
    ///
    public func download() -> UnsafeMutableBufferPointer<T>
    {
        fatalError("Not implemented.")
    }
    
    /// Upload the content of the buffer to the GPU.
    public func upload()
    {
        fatalError("Not implemented.")
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
    
    ///
    /// Download the content of the buffer to the CPU.
    ///
    /// - Returns: the CPU buffer.
    ///
    public override func download() -> UnsafeMutableBufferPointer<T>
    {
        MetalKernel.get.download([self])
        return shared.buffer
    }
    
    /// Upload the content of the buffer to the GPU.
    public override func upload()
    {
        MetalKernel.get.upload([self])
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
    /// The inner data buffer.
    public let buffer: UnsafeMutableBufferPointer<T>
    
    /// Explicit allocated memory pointer.
    let memory: UnsafeMutableRawPointer
    /// Byte size allocation.
    let allocationSize: Int
    
    /// Get access to the inner data as an array.
    public var array: [T]
    {
        get {
            return memory.toArray(to: T.self, capacity: nbElems)
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
        var memoryTmp: UnsafeMutableRawPointer? = nil
        let alignment = 0x1000
        
        allocationSize = (byteLength + alignment - 1) & (~(alignment - 1))
        posix_memalign(&memoryTmp, alignment, allocationSize)
        memset(memoryTmp, 0, byteLength)
        
        memory = memoryTmp!
        let pointer = memoryTmp?.bindMemory(to: T.self, capacity: nbElems)
        buffer = UnsafeMutableBufferPointer(start: pointer, count: nbElems)
        
        super.init(nbElems, deviceID: deviceID)
    }
    
    ///
    /// Free memory.
    ///
    /// This is necessary as memory was initialized with a specific method.
    ///
    deinit {
        free(memory)
    }
    
    ///
    /// Download the content of the buffer to the CPU.
    ///
    /// - Returns: the CPU buffer.
    ///
    public override func download() -> UnsafeMutableBufferPointer<T>
    {
        MetalKernel.get.download([self])
        return buffer
    }
    
    /// Upload the content of the buffer to the GPU.
    public override func upload()
    {
        MetalKernel.get.upload([self])
    }
}

public extension UnsafeMutableRawPointer
{
    ///
    /// Convert buffer pointer to array.
    ///
    /// - Parameters:
    ///     - to: The type for each elements of the array.
    ///     - capacity: size of the array.
    /// - Returns: The converted array.
    ///
    func toArray<T>(to type: T.Type, capacity count: Int) -> [T]
    {
        let pointer = bindMemory(to: type, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}
