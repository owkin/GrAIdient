//
// MSE1DCase.swift
// MAKitTests
//
//  Created by Jean-François Reboud on 10/10/2022.
//

import XCTest
import MAKit
import MAKitTestsUtils

///
/// A class that will test a model with a structural hypothesis:
/// the model last layer is a MSE1D layer.
///
class MSE1DCase: XCTestCase
{
    /// Batch size of data.
    var batchSize: Int! = nil
    /// Optimizer parameters.
    var optimizerParams = MAKit.Optimizer.Params()
    
    /// Ground truth buffer.
    var _gtBuffer: MetalSharedBuffer<Float>! = nil
    
    /// Systematic call before test begins.
    override func setUpWithError() throws
    {
        batchSize = 5
        MAKit.Opti.GPU = true
        
        setOptimizerParams(params: &optimizerParams)
        optimizerParams.nbLoops = 3
    }
    
    ///
    /// Get the current batch size of data.
    ///
    /// This function allows to simulate the fact that the batch size of data may be smalling during the
    /// last iteration of the training.
    ///
    /// - Parameter model: The model.
    /// - Returns: The batch size of data.
    ///
    func getBatchSize(_ model: Model) -> Int
    {
        if model.optimizerParams.step == model.optimizerParams.nbLoops-1
        {
            return batchSize / 2
        }
        else
        {
            return batchSize
        }
    }
    
    ///
    /// Create synthetic data.
    ///
    /// - Parameters:
    ///     - dim1: The first dimension of the data.
    ///     - dim2: The second dimension of the data.
    /// - Returns: The created data.
    ///
    func build1DData<T: BinaryFloatingPoint>(dim1: Int, dim2: Int) -> [[T]]
    {
        var data = [[T]]()
        for _ in 0..<dim1
        {
            var data1 = [T]()
            for _ in 0..<dim2
            {
                data1.append(T(Double.random(in: -1.0..<1.0)))
            }
            data.append(data1)
        }
        return data
    }
    
    ///
    /// Upload data to a GPU buffer.
    ///
    /// - Parameters:
    ///     - data: The data to upload.
    ///     - buffer: The GPU buffer that should contain the data.
    ///
    func uploadData<T: BinaryFloatingPoint>(
        data: [[T]], buffer: MetalBuffer<Float>)
    {
        let dim2 = data.first!.count
        let bufferPtr: UnsafeMutableBufferPointer<Float>
        if let sBuffer = buffer as? MetalSharedBuffer<Float>
        {
            // Wait for previous loop to end to avoid race condition.
            MetalKernel.get.download([sBuffer])
            
            bufferPtr = sBuffer.buffer
        }
        else if let pBuffer = buffer as? MetalPrivateBuffer<Float>
        {
            // Wait for previous loop to end to avoid race condition.
            MetalKernel.get.download([pBuffer])
            
            bufferPtr = pBuffer.shared.buffer
        }
        else
        {
            fatalError("Unreachable.")
        }
        
        for (i, dataI) in data.enumerated()
        {
            for (j, dataIJ) in dataI.enumerated()
            {
                bufferPtr[j + i * dim2] = Float(dataIJ)
            }
        }
        
        if let sBuffer = buffer as? MetalSharedBuffer<Float>
        {
            MetalKernel.get.upload([sBuffer])
        }
        else if let pBuffer = buffer as? MetalPrivateBuffer<Float>
        {
            MetalKernel.get.upload([pBuffer])
        }
    }
    
    ///
    /// A function to create/set ground truth to the model.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth to set.
    ///     - model: The model.
    /// - Returns: The ground truth.
    ///
    func setLoss(_ groundTruth: [[Double]]?, _ model: Model) -> [[Double]]
    {
        let lastLayer = model.layers.last as! MSE1D
        let gt: [[Double]]
        if let groundTruthTmp = groundTruth
        {
            gt = groundTruthTmp
        }
        else
        {
            gt = build1DData(dim1: getBatchSize(model), dim2: 1)
        }
        
        if MAKit.Opti.GPU
        {
            if _gtBuffer == nil
            {
                _gtBuffer = MetalSharedBuffer<Float>(
                    lastLayer.nbNeurons * batchSize,
                    deviceID: DEVICE_ID_DEFAULT
                )
            }
            uploadData(data: gt, buffer: _gtBuffer)
            try! lastLayer.applyGradientGPU(
                _gtBuffer,
                batchSize: getBatchSize(model)
            )
        }
        else
        {
            try! lastLayer.applyGradientCPU(gt)
        }
        return gt
    }
    
    ///
    /// A function to get loss of a model.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth to set.
    ///     - model: The model.
    /// - Returns: The loss value.
    ///
    func getLoss(_ groundTruth: [[Double]], _ model: Model) -> Double
    {
        let lastLayer = model.layers.last as! MSE1D
        if MAKit.Opti.GPU
        {
            if _gtBuffer == nil
            {
                _gtBuffer = MetalSharedBuffer<Float>(
                    lastLayer.nbNeurons * batchSize,
                    deviceID: DEVICE_ID_DEFAULT
                )
            }
            uploadData(data: groundTruth, buffer: _gtBuffer)
            return Double(try! lastLayer.getLossGPU(
                _gtBuffer,
                batchSize: getBatchSize(model)
            ))
        }
        else
        {
            return try! lastLayer.getLossCPU(groundTruth)
        }
    }
    
    ///
    /// A function to get the gradients of weights approximations..
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - model: The model.
    /// - Returns: The gradients of weights approximations.
    ///
    func getGradientsApprox(
        _ groundTruth: [[Double]],
        _ model: Model) -> [Double]
    {
        let lastLayer = model.layers.last as! MSE1D
        return try! lastLayer.collectGradientsApprox(groundTruth)
    }
}
