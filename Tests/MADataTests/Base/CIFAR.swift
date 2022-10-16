//
// CIFAR.swift
// MADataTests
//
// Created by Jean-François Reboud on 16/10/2022.
//

import Foundation
import Darwin
import PythonKit
import MAData

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
    
    public static func dumpDataset(
        pythonLibrary: String,
        dataInputDir: String,
        datasetOutputDir: String,
        label: Int,
        size: Int)
    {
        setenv("PYTHON_LIBRARY", pythonLibrary, 1)
        let motionData = Python.import("data")
        
        var features = [UInt8]()
        for dataFile in 1...5
        {
            let data = motionData.load_CIFAR_data(
                dataInputDir, dataFile, label, size)
            features += Array<UInt8>(data)!
        }
        
        let featuresPath = datasetOutputDir + "/features"
        let featuresData = Data(bytes: &features,
                            count: features.count * MemoryLayout<UInt8>.stride)
        try! featuresData.write(to: URL(fileURLWithPath: featuresPath))
    }
    
    public static func dumpTest(
        pythonLibrary: String,
        dataInputDir: String,
        datasetOutputDir: String,
        label: Int,
        size: Int)
    {
        setenv("PYTHON_LIBRARY", pythonLibrary, 1)
        let motionData = Python.import("data")
        
        var features = [UInt8]()
        let data = motionData.load_CIFAR_test(dataInputDir, label, size)
        features += Array<UInt8>(data)!
        
        let featuresPath = datasetOutputDir + "/features"
        let featuresData = Data(bytes: &features,
                            count: features.count * MemoryLayout<UInt8>.stride)
        try! featuresData.write(to: URL(fileURLWithPath: featuresPath))
    }
    
    public static func loadDataset(datasetDir: String, size: Int) -> CIFAR
    {
        let featuresData = try! Data(contentsOf:
            URL(fileURLWithPath: datasetDir + "/features"))
        
        let features: [UInt8] = featuresData.withUnsafeBytes {
            (pointer: UnsafeRawBufferPointer) -> [UInt8] in
            let buffer = pointer.bindMemory(to: UInt8.self)
            return buffer.map { UInt8($0) }
        }
        
        return CIFAR(data: features, size: size)
    }
}

