//
// Utils.swift
// GrAITorchTests
//
// Created by Jean-François Reboud on 19/10/2022.
//

import Foundation
import PythonKit
import GrAIdient

/// Python library default path.
let PYTHON_LIB =
    FileManager.default.homeDirectoryForCurrentUser.path +
    "/opt/anaconda3/envs/graitorch/lib/libpython3.9.dylib"

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
/// Get data "complex" image from Python.
///
/// - Parameter size: The size of the image.
/// - Returns: The data image.
///
func getComplexData(_ size: Int) -> [Float]
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.get_complex_data(size)
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
/// Get gradient norm computed with PyTorch for ModelTest3.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeTest3GradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_test3_grad_norm(size)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTest4.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeTest4GradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_test4_grad_norm(size)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTest5.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeTest5GradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_test5_grad_norm(size)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTest6.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeTest6GradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_test6_grad_norm(size)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTest7.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeTest7GradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_test7_grad_norm(size)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTest8.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeTest8GradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_test8_grad_norm(size)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTest9.
///
/// - Parameters:
///     - size: The size of the input data.
///     - patch: The kernel split size of the input data.
/// - Returns: The gradient norm.
///
func computeTest9GradNorm(size: Int, patch: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_test9_grad_norm(size, patch)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTest10.
///
/// - Parameters:
///     - size: The size of the input data.
///     - patch: The kernel split size of the input data.
/// - Returns: The gradient norm.
///
func computeTest10GradNorm(size: Int, patch: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_test10_grad_norm(size, patch)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTest11.
///
/// - Parameters:
///     - size: The size of the input data.
///     - patch: The kernel split size of the input data.
/// - Returns: The gradient norm.
///
func computeTest11GradNorm(size: Int, patch: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_test11_grad_norm(size, patch)
    return Float(data)!
}

///
/// Get optimizer parameters for model training.
///
/// - Parameter nbLoops: Number of steps per epoch.
/// - Returns: The optimizer parameters.
///
func getOptimizerParams(nbLoops: Int) -> GrAI.Optimizer.Params
{
    var optimizerParams = GrAI.Optimizer.Params()
    optimizerParams.nbLoops = nbLoops
    
    // Simple optimizer scheduler: always the same optimizer during
    // the training.
    optimizerParams.optimizer = ConstEpochsScheduler(
        GrAI.Optimizer.Class.AdamRectified
    )
    
    // Simple variable scheduler: always the same variable during
    // the training.
    optimizerParams.variables["alpha"] = ConstEpochsVar(
        value: ConstVal(1e-3)
    )
    optimizerParams.variables["lambda"] = ConstEpochsVar(
        value: ConstVal(1e-3)
    )
    
    // Other schedulers can be built thanks to `GrAI.Optimizer.Params`.
    return optimizerParams
}
