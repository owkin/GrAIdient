//
// OptimizerTests.swift
// MAKitTests
//
// Created by Jean-François Reboud on 10/10/2022.
//

import MAKit
import MATestsUtils

class OptimizerTests: Input1DMSE1DCase
{
    override func setUpWithError() throws
    {
        batchSize = 5
        MAKit.Opti.GPU = true
        
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
    
    private func _buildModel(context: ModelContext)
    {
        let params = MAKit.Model.Params(context: context)
        
        var layer: Layer1D = Input1D(nbNeurons: 1, params: params)
        
        layer = FullyConnected(
            layerPrev: layer, nbNeurons: 1,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        layer = FullyConnected(
            layerPrev: layer, nbNeurons: 12,
            activation: LeakyReLU.str, biases: true,
            params: params
        )
        
        layer = FullyConnected(
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
        setOptimizerParams(params: &optimizerParams,
                           optimizerClass: .AdamRectified)
        let trainer = _buildTrainer()
        run(trainer)
    }
    
    func testAdamRectifiedDecay() throws
    {
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
