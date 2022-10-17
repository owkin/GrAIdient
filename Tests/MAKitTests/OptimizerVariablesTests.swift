//
// OptimizerVariablesTests.swift
// MAKitTests
//
// Created by Jean-François Reboud on 10/10/2022.
//

import XCTest
import MAKit

class OptimizerVariablesTests: XCTestCase
{
    var optimizerParams = MAKit.Optimizer.Params()
    
    override func setUp()
    {
        let firstEpoch = 10
        let secondEpoch = 20
        let nbEpochs = 30
        
        optimizerParams.nbLoops = 10
        
        let min = 0.001
        let max = 0.01
    
        optimizerParams.optimizer = ListEpochsScheduler(
            epochs: [firstEpoch, secondEpoch, nbEpochs],
            optimizers: [
                MAKit.Optimizer.Class.AdaBound,
                MAKit.Optimizer.Class.SGD,
                MAKit.Optimizer.Class.AMSBound
            ]
        )
        
        optimizerParams.variables["alpha"] = ConstEpochsVar(
            value: ConstVal(min)
        )
        optimizerParams.variables["lambda"] = ListEpochsVar(
            epochs: [firstEpoch, secondEpoch, nbEpochs],
            values: [NullVal(), ConstVal(min), NullVal()]
        )
        optimizerParams.variables["lowerBound"] = ListEpochsVar(
            epochs: [firstEpoch, secondEpoch, nbEpochs],
            values: [
                ExponentialAscending(min: min, max: max),
                ConstVal(min),
                TrigonometricDescending(min: min, max: max)
            ]
        )
        optimizerParams.variables["upperBound"] = ListEpochsVar(
            epochs: [firstEpoch, secondEpoch, nbEpochs],
            values: [
                LinearDescending(min: min, max: max),
                ConstVal(min),
                TrigonometricAscending(min: min, max: max)
            ]
        )
    }
    
    private func _checkLists(_ list1: [Double?], _ list2: [Double?])
    {
        for (elem1, elem2) in zip(list1, list2)
        {
            if elem1 != nil && elem2 != nil
            {
                XCTAssertEqual(elem1!, elem2!, accuracy: 0.001)
            }
            else
            {
                XCTAssert(elem1 == elem2)
            }
        }
    }
    
    // Test dynamic behavior of time dependent optimizer variables.
    func testOptimizerVariables()
    {
        let firstEpoch = 10
        let secondEpoch = 20
        let nbEpochs = 30
        
        let min = 0.001
        let max = 0.01
        
        let model = Model(name: "OptimizerVariables")
        model.setupOptimizers(params: optimizerParams)
        
        var listAlpha1 = [Double]()
        var listLambda1 = [Double?]()
        var listLowerBound1 = [Double?]()
        var listUpperBound1 = [Double?]()
        
        for _ in 0..<nbEpochs
        {
            for numLoop in 0..<optimizerParams.nbLoops
            {
                let alpha = model.alphaPercent
                let lambda = model.lambdaPercent
                let lowerBound = model.lowerBoundPercent
                let upperBound = model.upperBoundPercent
                
                if numLoop == 0
                {
                    listAlpha1.append(alpha)
                    listLambda1.append(lambda)
                    listLowerBound1.append(lowerBound)
                    listUpperBound1.append(upperBound)
                }
                
                model.incStep()
            }
            
            model.incEpoch()
        }
        
        var listAlpha2 = [Double]()
        var listLambda2 = [Double?]()
        var listLowerBound2 = [Double?]()
        var listUpperBound2 = [Double?]()
        
        for epoch in 0..<nbEpochs
        {
            let progress = 0.1 * Double(epoch % 10)
            let factor = sin(Double.pi/2.0 * progress)
            
            if epoch < firstEpoch
            {
                listAlpha2.append(100)
                listLambda2.append(nil)
                listLowerBound2.append(
                    (min * pow(max/min, progress) - min) * 100.0 / (max - min)
                )
                listUpperBound2.append(
                    (max - max + progress * (max - min)) * 100.0 / (max - min)
                )
            }
            else if epoch < secondEpoch
            {
                listAlpha2.append(100)
                listLambda2.append(100)
                listLowerBound2.append(nil)
                listUpperBound2.append(nil)
            }
            else
            {
                listAlpha2.append(100)
                listLambda2.append(nil)
                listLowerBound2.append(
                    (max - max + (max - min) * factor) * 100.0 / (max - min)
                )
                listUpperBound2.append(
                    (min + (max - min) * factor - min) * 100.0 / (max - min)
                )
            }
        }
        
        XCTAssert(listAlpha1 == listAlpha2)
        _checkLists(listLambda1, listLambda2)
        _checkLists(listLowerBound1, listLowerBound2)
        _checkLists(listUpperBound1, listUpperBound2)
    }
}
