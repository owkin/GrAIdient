//
// Input1DBCESigmoid1DCase.swift
// GrAITests
//
//  Created by Jean-François Reboud on 07/07/2023.
//

import XCTest
import GrAIdient
import GrAITestsUtils

///
/// A class that will test a model with a structural hypothesis:
/// the model last layer is a BCE1D layer, the model first layer is a Input1D.
///
class Input1DBCESigmoid1DCase: XCTestCase, Input1DCase, IOCase
{
    /// Batch size of data.
    var batchSize: Int = -1
    /// Optimizer parameters.
    var optimizerParams = GrAI.Optimizer.Params()
    
    /// Systematic call before test begins.
    override func setUp()
    {
        batchSize = 5
        _ = MetalKernel.get
        GrAI.Opti.GPU = true
        
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
        let lastLayer = model.layers.last as! BCESigmoid1D
        let gt: [[Double]]
        if let groundTruthTmp = groundTruth
        {
            gt = groundTruthTmp
        }
        else
        {
            var values = [[Double]]()
            let batchSize = getBatchSize(model)
            for _ in 0..<batchSize / 2
            {
                values.append([0.0])
            }
            for _ in batchSize / 2..<batchSize
            {
                values.append([1.0])
            }
            gt = values
        }
        
        if GrAI.Opti.GPU
        {
            try! lastLayer.lossDerivativeGPU(gt)
        }
        else
        {
            try! lastLayer.lossDerivativeCPU(gt)
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
        let lastLayer = model.layers.last as! BCESigmoid1D
        if GrAI.Opti.GPU
        {
            return Double(try! lastLayer.getLossGPU(groundTruth))
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
        let lastLayer = model.layers.last as! BCESigmoid1D
        return try! lastLayer.collectGradientsApprox(groundTruth)
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
        _ inputs: [[Float]]?,
        _ model: Model) -> ([[Float]], Int)
    {
        let firstLayer = model.layers.first as! Input1D
        let ins: [[Float]]
        if let insTmp = inputs
        {
            ins = insTmp
        }
        else
        {
            ins = buildData(dim1: getBatchSize(model), dim2: 1)
        }
        
        if GrAI.Opti.GPU
        {
            try! firstLayer.setDataGPU(ins)
        }
        else
        {
            try! firstLayer.setDataCPU(ins)
        }
        return (ins, ins.count)
    }
}
