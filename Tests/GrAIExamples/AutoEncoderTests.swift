//
// AutoEncoderTests.swift
// GrAIExamples
//
// Created by Jean-François Reboud on 21/05/2023.
//

import XCTest
import PythonKit
import GrAIdient

/// Compare models trained in GrAIdient and PyTorch.
final class AutoEncoderTests: XCTestCase
{
    /// Batch size of data.
    let _batchSize = 16
    
    /// The label associated to the data.
    let _label = 8
    
    /// Size of one image (height and width are the same).
    let _size = 32
    
    /// Initialize test.
    override func setUp()
    {
        setPythonLib()
        _ = MetalKernel.get
        GrAI.Opti.GPU = true
    }
    
    /// Compare loss in the training of a simple auto encoder model in GrAIdient and PyTorch.
    func testTrain() throws
    {
        let trainer1 = try CIFARAutoEncoderTrainer(
            model: SimpleAutoEncoder.build(_size),
            size: _size
        )
        trainer1.initTrain(
            batchSize: _batchSize,
            label: _label
        )
        
        let pythonLib = Python.import("python_lib")
        let trainer2 = pythonLib.train_simple_auto_encoder(
            _batchSize,
            _label
        )
        
        for _ in 0..<100
        {
            let computedLoss = trainer1.step()
            let expectedLoss = Float(
                pythonLib.step_simple_auto_encoder(trainer2)
            )!
            
            // Compare difference.
            let diffPercent =
                abs(computedLoss - expectedLoss) / abs(expectedLoss) * 100.0
            XCTAssert(diffPercent < 0.1)
        }
    }
}
