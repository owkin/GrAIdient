//
// DataSampler.swift
// MAData
//
// Created by Jean-François Reboud on 16/10/2022.
//

import Foundation

public protocol DataSampler
{
    var nbSamples: Int { get }
    var nbLoops: Int { get }
    var batchSize: Int { get set }
    
    func clone() -> Self
    func initSamples(batchSize: Int)
    
    func keep(_ nbBlocks: Int)
    func remove(_ nbBlocks: Int)
    
    func getSampledDataIndex(_ sampleFactor: Double) -> (subSampled: [Int],
                                                         notSampled: [Int])
    func setDataIndex(_ dataIndex: [Int])
    func shuffle()
}

open class DataSamplerImpl<Type>: DataSampler
{
    let _data: [Type]
    
    var _batchSize: Int? = nil
    
    var _dataIndex: [Int]? = nil
    var _curIndex: Int? = nil
    
    open var sizeDataBlock: Int
    {
        get {
            fatalError("Not implemented.")
        }
    }
    
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
    
    public init(data: [Type])
    {
        _data = data
    }
    
    public init(copyFrom: DataSamplerImpl<Type>)
    {
        _data = copyFrom._data
        _batchSize = copyFrom._batchSize
        _dataIndex = copyFrom._dataIndex
        _curIndex = 0
    }
    
    open func clone() -> Self
    {
        return DataSamplerImpl<Type>(copyFrom: self) as! Self
    }
    
    public func initSamples(batchSize: Int)
    {
        _batchSize = batchSize
        _dataIndex = Array(0..<_data.count / sizeDataBlock)
        _curIndex = 0
    }
    
    public func keep(_ nbBlocks: Int)
    {
        guard let dataIndex = _dataIndex else
        {
            fatalError("Call `initSamples` before.")
        }
        
        _dataIndex = Array(dataIndex[0..<nbBlocks])
        _curIndex = 0
    }
    
    public func remove(_ nbBlocks: Int)
    {
        guard let dataIndex = _dataIndex else
        {
            fatalError("Call `initSamples` before.")
        }
        
        _dataIndex = Array(dataIndex[nbBlocks..<dataIndex.count])
        _curIndex = 0
    }
    
    public func getSampledDataIndex(_ sampleFactor: Double)
        -> (subSampled: [Int], notSampled: [Int])
    {
        guard let dataIndex = _dataIndex else
        {
            fatalError("Call `initSamples` before.")
        }
        
        var subSampled = [Int]()
        var notSampled = [Int]()
        
        var curFactor: Double = sampleFactor
        for dataIndexTmp in dataIndex
        {
            if curFactor >= 1.0
            {
                subSampled.append(dataIndexTmp)
                curFactor = 0.0
            }
            else
            {
                notSampled.append(dataIndexTmp)
            }
            curFactor += sampleFactor
        }
        
        return (subSampled: subSampled, notSampled: notSampled)
    }
    
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
    
    public func setDataIndex(_ dataIndex: [Int])
    {
        _dataIndex = dataIndex
        _curIndex = 0
    }
    
    public func getSamples(blockIndex: Int) -> [[Type]]?
    {
        var curIndex = blockIndex
        return _getSamples(&curIndex)
    }
    
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
