//
// CIFAR.swift
// GrAIExamples
//
// Created by Jean-François Reboud on 16/10/2022.
//

import Foundation
import PythonKit

/// CIFAR data sampler.
class CIFAR: DataSamplerImpl<UInt8>
{
    /// Size of one image (height and width are the same).
    private let _size: Int
    
    /// Size of one internal element.
    override var sizeDataBlock: Int
    {
        return _size * _size * 3
    }
    
    ///
    /// Create a data sampler.
    ///
    /// - Parameters:
    ///     - data: The internal data.
    ///     - size: The image size (height and width are the same).
    ///
    init(data: [UInt8], size: Int)
    {
        _size = size
        super.init(data: data)
    }
    
    ///
    /// Dump the training dataset to the disk.
    ///
    /// - Parameters:
    ///     - datasetPath: The path where to dump the dataset.
    ///     - label: The label we want the data associated to.
    ///     - size: The image size (height and width are the same).
    ///
    static func dumpTrain(
        datasetPath: String,
        label: Int,
        size: Int)
    {
        let pythonLib = Python.import("python_lib")
        
        var dataset = [UInt8]()
        for dataFile in 1...5
        {
            let data = pythonLib.load_CIFAR_train(dataFile, label, size)
            dataset += Array<UInt8>(data)!
        }
        
        let datasetData = Data(
            bytes: &dataset,
            count: dataset.count * MemoryLayout<UInt8>.stride
        )
        try! datasetData.write(to: URL(fileURLWithPath: datasetPath))
    }
    
    ///
    /// Dump the testing dataset to the disk.
    ///
    /// - Parameters:
    ///     - datasetPath: The path where to dump the dataset.
    ///     - label: The label we want the data associated to.
    ///     - size: The image size (height and width are the same).
    ///
    static func dumpTest(
        datasetPath: String,
        label: Int,
        size: Int)
    {
        let pythonLib = Python.import("python_lib")
        
        var dataset = [UInt8]()
        let data = pythonLib.load_CIFAR_test(label, size)
        dataset += Array<UInt8>(data)!
        
        let datasetData = Data(
            bytes: &dataset,
            count: dataset.count * MemoryLayout<UInt8>.stride
        )
        try! datasetData.write(to: URL(fileURLWithPath: datasetPath))
    }
    
    ///
    /// Load a dataset from the disk.
    ///
    /// - Parameters:
    ///     - datasetPath: The path to load the dataset from.
    ///     - size: The image size (height and width are the same).
    /// - Returns: The CIFAR data sampler.
    ///
    static func loadDataset(datasetPath: String, size: Int) -> CIFAR
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
    
    ///
    /// Build an iterator on CIFAR dataset.
    ///
    /// - Parameters:
    ///     - train: Train of test dataset.
    ///     - batchSize: The batch size.
    ///     - label: The label we want the data associated to.
    ///     - shuffle: Whether to shuffle indices of data.
    ///
    /// - Returns: A Python iterator.
    ///
    static func buildIterator(
        train: Bool,
        batchSize: Int,
        label: Int,
        shuffle: Bool) -> PythonObject
    {
        let pythonLib = Python.import("python_lib")
        return pythonLib.iter_CIFAR(train, batchSize, label, shuffle)
    }
    
    ///
    /// Load next data from a Python iterator.
    ///
    /// - Parameter iterator: The Python iterator.
    ///
    static func getSamples(_ iterator: PythonObject) -> ([Float], Int)
    {
        let pythonLib = Python.import("python_lib")
        let data = pythonLib.next_data_CIFAR(iterator)
        
        let samples = [Float](data.tuple2.0)!
        let batchSize = Int(data.tuple2.1)!
        
        return (samples, batchSize)
    }
}
