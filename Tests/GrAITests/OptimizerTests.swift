//
// OptimizerTests.swift
// GrAITests
//
// Created by Jean-François Reboud on 10/10/2022.
//

import GrAIdient
import GrAITestsUtils

// -----------------------------------------------------------------------------
// Compare GPU gradients with CPU ones through time.
// We expect to see errors ~ 1e-7 and less.
// -----------------------------------------------------------------------------
class OptimizerFlowTests: Input1DMSE1DCase
{
    override func setUp()
    {
        batchSize = 5
        _ = MetalKernel.get
        
        GrAI.Opti.GPU = true
        GrAI.Precision.float = true
        
        setOptimizerParams(params: &optimizerParams)
        optimizerParams.nbLoops = 10
    }
    
    private func _buildTrainer() -> FlowTrainer
    {
        let trainer = FlowTrainer(
            name: "Optimizer",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            _buildModel(context: context)
        }
        return trainer
    }
    
    fileprivate func _buildModel(context: ModelContext)
    {
        let params = GrAI.Model.Params(context: context)
        
        var layer: Layer1D = Input1D(nbNeurons: 1, params: params)
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 12,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        layer = try! FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        layer = MSE1D(layerPrev: layer, params: params)
    }
    
    func testSGD() throws
    {
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testSGDDecay() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testSGDMomentum() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .SGDMomentum)
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testSGDMomentumDecay() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .SGDMomentum,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testAdam() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testAdamDecay() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testAMSGrad() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AMSGrad)
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testAMSGradDecay() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AMSGrad,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testAdamRectified() throws
    {
        optimizerParams.nbLoops = 5
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AdamRectified)
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testAdamRectifiedDecay() throws
    {
        optimizerParams.nbLoops = 5
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AdamRectified,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testAdaBound() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AdaBound)
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testAdaBoundDecay() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AdaBound,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testAMSBound() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AMSBound)
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testAMSBoundDecay() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AMSBound,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer)
    }
}

// -----------------------------------------------------------------------------
// Compare GPU gradients with Float precision versus Float16 precision.
// We expect to see errors ~ 1e-4 and less.
// -----------------------------------------------------------------------------
class OptimizerFlowPrecisionTests: OptimizerFlowTests
{
    override func setUp()
    {
        batchSize = 5
        _ = MetalKernel.get
        
        GrAI.Opti.GPU = true
        GrAI.Precision.float = true
        
        setOptimizerParams(params: &optimizerParams)
        optimizerParams.nbLoops = 10
    }
    
    private func _buildTrainer() -> FlowPrecisionTrainer
    {
        let trainer = FlowPrecisionTrainer(
            name: "Optimizer",
            params: optimizerParams
        )
        trainer.build()
        {
            (context: ModelContext) in
            _buildModel(context: context)
        }
        return trainer
    }
    
    override func testSGD() throws
    {
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.002)
    }
    
    override func testSGDDecay() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testSGDMomentum() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .SGDMomentum)
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testSGDMomentumDecay() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .SGDMomentum,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAdam() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam)
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAdamDecay() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .Adam,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAMSGrad() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AMSGrad)
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAMSGradDecay() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AMSGrad,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAdamRectified() throws
    {
        optimizerParams.nbLoops = 5
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AdamRectified)
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAdamRectifiedDecay() throws
    {
        optimizerParams.nbLoops = 5
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AdamRectified,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAdaBound() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AdaBound)
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAdaBoundDecay() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AdaBound,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAMSBound() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AMSBound)
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
    
    override func testAMSBoundDecay() throws
    {
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AMSBound,
                           lambda: 1e-3)
        let trainer = _buildTrainer()
        run(trainer, diffThreshold: 0.005)
    }
}
