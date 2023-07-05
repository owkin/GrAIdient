//
// IOCase.swift
// GrAITests
//
// Created by Jean-François Reboud on 05/07/2023.
//

import XCTest
import GrAIdient
import GrAITestsUtils

let NB_RETRY = 3

/// Use case with functions to get / set inputs and outputs.
protocol IOCase
{
    associatedtype DataT
    associatedtype LossT
    
    /// Batch size of data.
    var batchSize: Int { get }
    /// Optimizer parameters.
    var optimizerParams: GrAI.Optimizer.Params { get }
    
    /// A list of functions that transform the model into another one.
    var transforms: [(Model)->Model] { get }
    
    /// A function to create/set data to the model.
    func setData(_: DataT?, _: Model) -> (DataT, Int)
    /// A function to get the loss of the model.
    func getLoss(_: LossT, _: Model) -> Double
    /// A function to create/set ground truth to the model.
    func setLoss(_: LossT?, _: Model) -> LossT
    
    /// A function that gets gradients of weights approximations.
    func getGradientsApprox(_: LossT, _: Model) -> [Double]
}

extension IOCase
{
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
    func buildData<T: BinaryFloatingPoint>(dim1: Int, dim2: Int) -> [[T]]
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
    /// Run Gradient Checking test.
    ///
    /// The goal is to compare the gradients of weights that are computed through `backward`
    /// to an estimation that is being computed through `forwardGC`.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: GradTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.000001)
    {
        let model = trainer.model!
        let lastLayer = model.layers.last!
        let layersGraph = model.getGraph(lastLayer)
        
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    layersGraph: layersGraph,
                    setData: self.setData,
                    setLoss: self.setLoss,
                    getGradientsApprox: self.getGradientsApprox)
                {
                    (gradDiff: Double) in
                    if gradDiff > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Flow test.
    ///
    /// The goal is to compare the gradients of weights computed in the CPU execution context with
    /// the gradients of weights computed in the GPU execution context.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: FlowTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.000001)
    {
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    setData: self.setData,
                    setLoss: self.setLoss)
                {
                    (gradDiff: Double) in
                    if gradDiff > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Flow Reset test.
    ///
    /// The goal is to compare the gradients of weights computed in the CPU execution context with
    /// the gradients of weights computed in the GPU execution context.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: FlowResetTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.000001)
    {
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    setData: self.setData,
                    setLoss: self.setLoss)
                {
                    (gradDiff: Double) in
                    if gradDiff > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Flow Reverse test.
    ///
    /// The goal is to compare the gradients of weights computed in the CPU execution context with
    /// the gradients of weights computed in the GPU execution context.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: FlowReverseTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.000001)
    {
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    setData: self.setData,
                    setLoss: self.setLoss)
                {
                    (gradDiff: Double) in
                    if gradDiff > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Inference test.
    ///
    /// The goal is to compare the losses computed in the CPU execution context with
    /// the losses computed in the GPU execution context during the inference phase.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: InferenceTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.001)
    {
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    setData: self.setData,
                    setLoss: self.setLoss,
                    getLoss: self.getLoss)
                {
                    (lossDiff: Double) in
                    if lossDiff > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Loading test.
    ///
    /// The goal is to compare the losses computed in the CPU execution after havinng loaded the
    /// model from the disk and do the same in the GPU execution context.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: LoadTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.001)
    {
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    setData: self.setData,
                    setLoss: self.setLoss,
                    getLoss: self.getLoss)
                {
                    (diffCPU: Double, diffGPU: Double) in
                    if diffCPU > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                    if diffGPU > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Transform tests.
    ///
    /// The goal is to compare the losses computed in the CPU execution
    /// after transforming the model and do the same in the GPU execution context.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///
    func run(
        _ trainer: TransformTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.001)
    {
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    transforms: self.transforms,
                    setData: self.setData,
                    setLoss: self.setLoss,
                    getLoss: self.getLoss)
                {
                    (diffCPU: Double, diffGPU: Double) in
                    if diffCPU > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                    if diffGPU > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
    
    ///
    /// Run Clipping test.
    ///
    /// The goal is to compare the norm of the gradients of the weights with a threshold.
    ///
    /// - Parameters:
    ///     - trainer: The testing pipeline to run.
    ///     - nbRetry: The maximum number we can retry the test.
    ///     - diffThreshold: The threshold above which the relative difference is too high.
    ///     - normClipping: The threshold above which gradients must be cut.
    ///
    func run(
        _ trainer: NormTrainer,
        nbRetry: Int = NB_RETRY,
        diffThreshold: Double = 0.001,
        normClipping: Double = 0.001)
    {
        var params = optimizerParams
        params.gradientClipping = true
        params.normThreshold = normClipping
        trainer.optimizerParams = params
        
        retryNumeric(
            nbRetry: nbRetry,
            {
                () throws in
                try trainer.run(
                    setData: self.setData,
                    setLoss: self.setLoss)
                {
                    (normDiff: Double) throws in
                    if normDiff > diffThreshold
                    {
                        throw TestError.Numeric
                    }
                }
            },
            {
                () in
                XCTAssert(false)
            }
        )
    }
}
