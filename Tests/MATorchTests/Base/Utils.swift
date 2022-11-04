//
// Utils.swift
// MATorchTests
//
// Created by Jean-François Reboud on 19/10/2022.
//

import Foundation
import PythonKit
import MAKit

/// Python library default path.
let PYTHON_LIB =
    FileManager.default.homeDirectoryForCurrentUser.path +
    "/opt/anaconda3/envs/matorch/lib/libpython3.9.dylib"

/// Set the Python library path.
func setPythonLib()
{
    if ProcessInfo.processInfo.environment["PYTHON_LIBRARY"] == nil
    {
        let task = Process()
        task.launchPath = "/usr/bin/which"
        task.arguments = ["python"]
        
        let pipe = Pipe()
        task.standardOutput = pipe
        task.launch()
        task.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: String.Encoding.utf8)!
        
        if output.count > 0
        {
            var url = URL(fileURLWithPath: output)
            url = url.deletingLastPathComponent().deletingLastPathComponent()
            url = url.appendingPathComponent("lib")
            url = url.appendingPathComponent("libpython3.9.dylib")
            setenv("PYTHON_LIBRARY", url.path, 1)
        }
        else
        {
            setenv("PYTHON_LIBRARY", PYTHON_LIB, 1)
        }
    }
}

///
/// Get data image from Python.
///
/// - Parameter size: The size of the image.
/// - Returns: The data image.
///
func getInputData(_ size: Int) -> [Float]
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.get_input_data(size)
    return [Float](data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTest1.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeTest1GradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_test1_grad_norm(size)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTest2.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeTest2GradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_test2_grad_norm(size)
    return Float(data)!
}

///
/// Get optimizer parameters for model training.
///
/// - Parameter nbLoops: Number of steps per epoch.
/// - Returns: The optimizer parameters.
///
func getOptimizerParams(nbLoops: Int) -> MAKit.Optimizer.Params
{
    var optimizerParams = MAKit.Optimizer.Params()
    optimizerParams.nbLoops = nbLoops
    
    // Simple optimizer scheduler: always the same optimizer during
    // the training.
    optimizerParams.optimizer = ConstEpochsScheduler(
        MAKit.Optimizer.Class.AdamRectified
    )
    
    // Simple variable scheduler: always the same variable during
    // the training.
    optimizerParams.variables["alpha"] = ConstEpochsVar(
        value: ConstVal(1e-3)
    )
    optimizerParams.variables["lambda"] = ConstEpochsVar(
        value: ConstVal(1e-3)
    )
    
    // Other schedulers can be built thanks to `MAKit.Optimizer.Params`.
    return optimizerParams
}
