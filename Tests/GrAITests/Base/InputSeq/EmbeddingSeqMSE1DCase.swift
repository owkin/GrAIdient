//
// EmbeddingSeqMSE1DCase.swift
// GrAITests
//
// Created by Jean-François Reboud on 11/06/2024.
//

import XCTest
import GrAIdient
import GrAITestsUtils

///
/// A class that will test a model with a structural hypothesis:
/// the model last layer is a MSE1D layer, the model first layer is an EmbeddingSeq.
///
class EmbeddingSeqMSE1DCase: XCTestCase, Input1DCase, IOCase
{
    /// Batch size of data.
    var batchSize: Int = -1
    /// Length of the sequence.
    var sequence: Int = -1
    /// Vocabulary size.
    var vocabularySize: Int = -1
    /// Optimizer parameters.
    var optimizerParams = GrAI.Optimizer.Params()
    
    /// Systematic call before test begins.
    override func setUp()
    {
        batchSize = 5
        sequence = 35
        vocabularySize = 120
        _ = MetalKernel.get
        
        GrAI.Opti.GPU = true
        GrAI.Precision.float = true
        
        setOptimizerParams(params: &optimizerParams)
        optimizerParams.nbLoops = 3
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
            gt = buildData(dim1: getBatchSize(model), dim2: 1)
        }
        
        if GrAI.Opti.GPU
        {
            try! lastLayer.lossDerivativeGPU(
                gt, batchSize: gt.count, nbNeurons: 1
            )
        }
        else
        {
            try! lastLayer.lossDerivativeCPU(
                gt, batchSize: gt.count, nbNeurons: 1
            )
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
        if GrAI.Opti.GPU
        {
            return Double(try! lastLayer.getLossGPU(
                groundTruth, batchSize: groundTruth.count, nbNeurons: 1
            ))
        }
        else
        {
            return try! lastLayer.getLossCPU(
                groundTruth, batchSize: groundTruth.count, nbNeurons: 1
            )
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
        return try! lastLayer.collectGradientsApprox(
            groundTruth, batchSize: groundTruth.count, nbNeurons: 1
        )
    }
    
    ///
    /// Create synthetic data.
    ///
    /// - Parameters:
    ///     - batchSize: Batch size of the data.
    ///     - sequence: Length of the sequence.
    ///     - vocabularySize: Vocabulary size.
    /// - Returns: The created data.
    ///
    func buildData(
        batchSize: Int,
        sequence: Int,
        vocabularySize: Int) -> [[Int]]
    {
        var data = [[Int]]()
        for _ in 0..<batchSize
        {
            var data1 = [Int]()
            for _ in 0..<sequence
            {
                data1.append(Int.random(in: 0..<vocabularySize))
            }
            data.append(data1)
        }
        return data
    }
    
    ///
    /// A function to create/set data to the model.
    ///
    /// - Parameters:
    ///     - inputs: The data to set.
    ///     - model: The model.
    /// - Returns: (The data, the batch size).
    ///
    public func setData(
        _ inputs: [[Int]]?,
        _ model: Model) -> ([[Int]], Int)
    {
        let firstLayer = model.layers.first as! EmbeddingSeq
        let ins: [[Int]]
        if let insTmp = inputs
        {
            ins = insTmp
        }
        else
        {
            ins = buildData(
                batchSize: getBatchSize(model), 
                sequence: sequence,
                vocabularySize: vocabularySize
            )
        }
        
        if GrAI.Opti.GPU
        {
            try! firstLayer.setDataGPU(
                ins, batchSize: ins.count, sequence: sequence
            )
        }
        else
        {
            try! firstLayer.setDataCPU(
                ins, batchSize: ins.count, sequence: sequence
            )
        }
        return (ins, ins.count)
    }
}
