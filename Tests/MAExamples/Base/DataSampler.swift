//
// DataSampler.swift
// MAExamples
//
// Created by Jean-François Reboud on 16/10/2022.
//

import Foundation

/// API to interact with a data sampler.
public protocol DataSampler
{
    /// Number of different elements.
    var nbSamples: Int { get }
    /// Number of loops to see every elements.
    var nbLoops: Int { get }
    /// Number of elements retrieved per loop.
    var batchSize: Int { get set }
    
    ///
    /// Part internal data into small batches.
    ///
    /// - Parameter batchSize: The number of elements per batch.
    ///
    func initSamples(batchSize: Int)
    
    ///
    /// Keep a sub part of elements in the original data.
    ///
    /// - Parameter nbSamples: The number of elements to keep.
    ///
    func keep(_ nbSamples: Int)
    
    /// Shuffle the index associated to each element.
    func shuffle()
}

/// Data sampler generic implementation.
open class DataSamplerImpl<Type>: DataSampler
{
    /// The internal data.
    let _data: [Type]
    
    /// Number of elements retrieved per loop.
    var _batchSize: Int? = nil
    
    /// List of indices associated to each elements.
    var _dataIndex: [Int]? = nil
    /// Current loop index.
    var _curIndex: Int? = nil
    
    /// Size of one internal element.
    open var sizeDataBlock: Int
    {
        get {
            fatalError("Not implemented.")
        }
    }
    
    /// Number of different elements.
    public var nbSamples: Int
    {
        get {
            guard let dataIndex = _dataIndex else
            {
                fatalError("Call `initSamples` before.")
            }
            return dataIndex.count
        }
    }
    
    /// Number of loops to see every elements.
    public var nbLoops: Int
    {
        get {
            guard let batchSize = _batchSize else
            {
                fatalError("Call `initSamples` before.")
            }
            return Int(ceil(Float(nbSamples) / Float(batchSize)))
        }
    }
    
    /// Number of elements retrieved per loop.
    public var batchSize: Int
    {
        get {
            if _batchSize == nil
            {
                fatalError("Call `initSamples` before.")
            }
            return _batchSize!
        }
        set {
            _batchSize = newValue
        }
    }
    
    ///
    /// Create a data sampler.
    ///
    /// - Parameter data: The internal data.
    ///
    public init(data: [Type])
    {
        _data = data
    }
    
    ///
    /// Part internal data into small batches.
    ///
    /// - Parameter batchSize: The number of elements per batch.
    ///
    public func initSamples(batchSize: Int)
    {
        _batchSize = batchSize
        _dataIndex = Array(0..<_data.count / sizeDataBlock)
        _curIndex = 0
    }
    
    ///
    /// Keep a sub part of elements in the original data.
    ///
    /// - Parameter nbSamples: The number of elements to keep.
    ///
    public func keep(_ nbSamples: Int)
    {
        guard let dataIndex = _dataIndex else
        {
            fatalError("Call `initSamples` before.")
        }
        
        _dataIndex = Array(dataIndex[0..<nbSamples])
        _curIndex = 0
    }
    
    /// Shuffle the index associated to each element.
    public func shuffle()
    {
        guard var dataIndex = _dataIndex else
        {
            fatalError("Call `initSamples` before.")
        }
        dataIndex.shuffle()
        _dataIndex = dataIndex
        _curIndex = 0
    }
    
    ///
    /// Get the elements in the batch associated to the current loop.
    ///
    /// - Returns: The batch of elements.
    ///
    public func getSamples() -> [[Type]]?
    {
        guard var curIndex = _curIndex else
        {
            fatalError("Call `initSamples` or `shuffle` before.")
        }
        
        let samples = _getSamples(&curIndex)
        _curIndex = curIndex
        
        return samples
    }
    
    ///
    /// Get the elements in the batch associated to the current loop.
    ///
    /// - Parameter curIndex: Current loop.
    /// - Returns: The batch of elements.
    ///
    private func _getSamples(_ curIndex: inout Int) -> [[Type]]?
    {
        guard let dataIndex = _dataIndex,
              let batchSize = _batchSize else
        {
            fatalError("Call `initSamples` or `shuffle` before.")
        }
        
        if curIndex >= dataIndex.count
        {
            return nil
        }
        
        var samples = [[Type]]()
        func appendBlock(offsetBlock: Int)
        {
            var samplesTmp = [Type]()
            let offset = offsetBlock * sizeDataBlock
            for block in 0..<sizeDataBlock
            {
                samplesTmp.append(_data[offset + block])
            }
            samples.append(samplesTmp)
        }
        
        var nbSamples = 0
        while nbSamples < batchSize && curIndex < dataIndex.count
        {
            let offsetBlock = dataIndex[curIndex]
            appendBlock(offsetBlock: offsetBlock)
            nbSamples += 1
            curIndex += 1
        }
        
        if nbSamples == 0
        {
            return nil
        }
        return samples
    }
}
