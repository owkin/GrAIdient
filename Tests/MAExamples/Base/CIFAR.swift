//
// CIFAR.swift
// MADataTests
//
// Created by Jean-François Reboud on 16/10/2022.
//

import Foundation
import PythonKit

public class CIFAR: DataSamplerImpl<UInt8>
{
    let _size: Int
    
    public override var sizeDataBlock: Int
    {
        return _size * _size * 3
    }
    
    init(data: [UInt8], size: Int)
    {
        _size = size
        super.init(data: data)
    }
    
    init(copyFrom: CIFAR)
    {
        _size = copyFrom._size
        super.init(copyFrom: copyFrom)
    }
    
    public override func clone() -> Self
    {
        return CIFAR(copyFrom: self) as! Self
    }
    
    public static func dumpTrain(
        datasetPath: String,
        label: Int,
        size: Int)
    {
        let cifar = Python.import("cifar")
        
        var dataset = [UInt8]()
        for dataFile in 1...5
        {
            let data = cifar.load_CIFAR_data(dataFile, label, size)
            dataset += Array<UInt8>(data)!
        }
        
        let datasetData = Data(
            bytes: &dataset,
            count: dataset.count * MemoryLayout<UInt8>.stride
        )
        try! datasetData.write(to: URL(fileURLWithPath: datasetPath))
    }
    
    public static func dumpTest(
        datasetPath: String,
        label: Int,
        size: Int)
    {
        let cifar = Python.import("cifar")
        
        var dataset = [UInt8]()
        let data = cifar.load_CIFAR_test(label, size)
        dataset += Array<UInt8>(data)!
        
        let datasetData = Data(
            bytes: &dataset,
            count: dataset.count * MemoryLayout<UInt8>.stride
        )
        try! datasetData.write(to: URL(fileURLWithPath: datasetPath))
    }
    
    public static func loadDataset(datasetPath: String, size: Int) -> CIFAR
    {
        let datasetData = try! Data(
            contentsOf: URL(fileURLWithPath: datasetPath)
        )
        let dataset: [UInt8] = datasetData.withUnsafeBytes {
            (pointer: UnsafeRawBufferPointer) -> [UInt8] in
            let buffer = pointer.bindMemory(to: UInt8.self)
            return buffer.map { UInt8($0) }
        }
        return CIFAR(data: dataset, size: size)
    }
}

