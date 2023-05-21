//
// CIFARAutoEncoderTrainer.swift
// GrAIExamples
//
// Created by Jean-François Reboud on 21/05/2023.
//

import Foundation
import GrAIdient

/// Error occuring when trainer cannot be built.
public enum TrainerError: Error
{
    /// Model size is not coherent.
    case Size
    /// Model structure is not expected.
    case Structural
}

extension TrainerError: CustomStringConvertible
{
    public var description: String
    {
        switch self
        {
        case .Size:
            return "Model size is not coherent with trainer."
        case .Structural:
            return "Model first layer should be an Input2D."
        }
    }
}

/// Train an auto encoder model on CIFAR dataset.
class CIFARAutoEncoderTrainer
{
    /// Directory to dump outputs from the tests.
    let _outputDir = NSTemporaryDirectory()
    
    /// Size of one image (height and width are the same).
    let _size = 32
    
    /// Mean of the preprocessing to apply to data.
    let _mean: (Float, Float, Float) = (123.675, 116.28, 103.53)
    /// Deviation of the preprocessing to apply to data.
    let _std: (Float, Float, Float) = (58.395, 57.12, 57.375)
    
    /// Dataset to get the data from.
    var _dataset: CIFAR! = nil
    /// Final model that is being trained.
    var _model: Model! = nil
    /// Base model to train.
    let _baseModel: Model
    
    ///
    /// Create the trainer.
    ///
    /// Throw an error if the original model's first layer is not an `Input2D` or the size of the latter
    /// is not the size expected by the trainer.
    ///
    /// - Parameter model: The original model (auto encoder structure) to train.
    ///
    init(model: Model) throws
    {
        guard let firstLayer = model.layers.first as? Input2D else
        {
            throw TrainerError.Structural
        }
        
        let height = firstLayer.height
        let width = firstLayer.width
        if height != _size || width != _size
        {
            throw TrainerError.Size
        }
       
        _baseModel = model
    }
    
    ///
    /// Create the final model (containing the original one + some additional layers) to train.
    ///
    /// - Returns: The final model to train.
    ///
    private func _buildModel() -> Model
    {
        let context = ModelContext(name: "Final", models: [_baseModel])
        let params = GrAI.Model.Params(context: context)
        
        _ = MSE2D(
            layerPrev: _baseModel.layers.last as! Layer2D,
            params: params
        )
        
        var model = Model(name: "Final")
        model.layers = _baseModel.layers + context.model.layers
        model = Model(model: model, modelsPrev: [])
        
        return model
    }
    
    ///
    /// Get optimizer parameters for model training.
    ///
    /// - Parameter nbLoops: Number of steps per epoch.
    /// - Returns: The optimizer parameters.
    ///
    func _getOptimizerParams(nbLoops: Int) -> GrAI.Optimizer.Params
    {
        var optimizerParams = GrAI.Optimizer.Params()
        optimizerParams.nbLoops = nbLoops
        
        // Simple optimizer scheduler: always the same optimizer during
        // the training.
        optimizerParams.optimizer = ConstEpochsScheduler(
            GrAI.Optimizer.Class.Adam
        )
        
        // Simple variable scheduler: always the same variable during
        // the training.
        optimizerParams.variables["alpha"] = ConstEpochsVar(
            value: ConstVal(1e-3)
        )
        optimizerParams.variables["lambda"] = ConstEpochsVar(
            value: ConstVal(1e-6)
        )
        
        // Other schedulers can be built thanks to `GrAI.Optimizer.Params`.
        return optimizerParams
    }
    
    ///
    /// Initialize dataset, model and optimizer parameters.
    ///
    /// - Parameters:
    ///     - batchSize: The number of samples per batch of data.
    ///     - label: The class of the CIFAR dataset to use.
    ///     - keep: The number of elements to keep in the dataset.
    ///
    func initTrain(batchSize: Int, label: Int, keep: Int? = nil)
    {
        // Create dataset.
        CIFAR.dumpTrain(
            datasetPath: _outputDir + "/datasetTrain\(label)",
            label: label,
            size: _size
        )
        
        // Load dataset.
        _dataset = CIFAR.loadDataset(
            datasetPath: _outputDir + "/datasetTrain\(label)",
            size: _size
        )
        _dataset.initSamples(batchSize: batchSize)
        if let nbElems = keep
        {
            _dataset.keep(nbElems)
        }
        
        // Get optimizer parameters for iterating over batch size elements.
        let params = _getOptimizerParams(nbLoops: batchSize)
        
        // Build model
        _model = _buildModel()
        
        // Initialize for training.
        _model.initialize(params: params, phase: .Training)
    }
    
    ///
    /// One training step.
    ///
    /// - Returns: The loss on the last training step.
    ///
    func step() -> Float
    {
        let firstLayer: Input2D = _model.layers.first as! Input2D
        let lastLayer: MSE2D = _model.layers.last as! MSE2D
        
        // Get data.
        let samples = _dataset.getSamples()!
        let batchSize = samples.count
        
        // Pre processing.
        let data = preprocess(
            samples,
            height: _size,
            width: _size,
            mean: _mean,
            std: _std,
            imageFormat: .Neuron
        )
        
        // Reset gradient validity for backward pass
        // and update the batch size.
        _model.updateKernel(batchSize: batchSize)
        
        // Set data.
        try! firstLayer.setDataGPU(
            data,
            batchSize: batchSize,
            format: .Neuron
        )
        
        // Forward.
        try! _model.forward()
        
        // Apply loss derivative.
        try! lastLayer.lossDerivativeGPU(
            firstLayer.outs,
            batchSize: batchSize
        )
        
        // Backward.
        try! _model.backward()
        
        // Update weights.
        try! _model.update()
        
        // Get loss result.
        // Note that backward is explicitly
        // enabled by `applyGradient` whereas `getLoss` is
        // just an indicator.
        let loss = try! lastLayer.getLossGPU(
            firstLayer.outs,
            batchSize: batchSize
        )
        
        // Update internal step.
        // This is not mandatory except if we used another
        // optimizer scheduler: see `_getOptimizerParams`.
        _model.incStep()
        
        return loss
    }
    
    ///
    /// Run the training on multiple steps and multiple epochs.
    ///
    /// - Parameters:
    ///     - batchSize: The number of samples per batch of data.
    ///     - label: The class of the CIFAR dataset to use.
    ///     - nbEpochs: The number of epochs for the training to continue.
    ///     - keep: The number of elements to keep in the dataset.
    ///     
    func run(batchSize: Int, label: Int, nbEpochs: Int, keep: Int? = nil)
    {
        initTrain(
            batchSize: batchSize,
            label: label,
            keep: keep
        )
        
        for epoch in 0..<nbEpochs
        {
            print("EPOCH \(epoch)/\(nbEpochs-1).")
            _dataset.shuffle()
            
            var runningLoss = 0.0
            var nbSteps = 0
            
            for _ in 0..<_dataset.nbLoops
            {
                let loss = self.step()
                print("Step \(nbSteps)/\(_dataset.nbLoops-1): \(sqrt(loss)).")
                
                runningLoss += Double(loss)
                nbSteps += 1
                
                if nbSteps % 50 == 0
                {
                    print(
                        "Running loss: " +
                        "\(sqrt(runningLoss / 50.0))."
                    )
                    runningLoss = 0.0
                }
            }
        }
    }
}
