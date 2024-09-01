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
    "/miniconda3/envs/graitorch/lib/libpython3.9.dylib"

/// Set the Python library path.
func setPythonLib()
{
    setenv("PYTHON_LIBRARY", PYTHON_LIB, 1)
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
/// Get data image from Python.
///
/// - Parameters:
///     - size: The size of the image.
///     - batchSize: The number of elements in the batch.
/// - Returns: The data image.
///
func getBatchData(size: Int, batchSize: Int) -> [Float]
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.get_batch_data(size, batchSize)
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
/// Get gradient norm computed with PyTorch for ModelTestConv1.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeConv1GradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_conv1_grad_norm(size)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTestConv2.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeConv2GradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_conv2_grad_norm(size)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTestFFT.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeFFTGradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_fft_grad_norm(size)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTestConvSK.
///
/// - Parameters:
///     - size: The size of the input data.
///     - stride: The stride of the model.
///     - kernel: The kernel size of the model.
/// - Returns: The gradient norm.
///
func computeConvSKGradNorm(size: Int, stride: Int, kernel: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_conv_sk_grad_norm(size, stride, kernel)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTestDeConvSK.
///
//// - Parameters:
///     - size: The size of the input data.
///     - stride: The stride of the model.
///     - kernel: The kernel size of the model.
/// - Returns: The gradient norm.
///
func computeDeConvSKGradNorm(size: Int, stride: Int, kernel: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_deconv_sk_grad_norm(size, stride, kernel)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTestCat.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeCatGradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_cat_grad_norm(size)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTestResize.
///
/// - Parameters:
///     - sizeInput: The size of the input data.
///     - sizeOutput: The output size of the resize operation.
/// - Returns: The gradient norm.
///
func computeResizeGradNorm(sizeInput: Int, sizeOutput: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_resize_grad_norm(sizeInput, sizeOutput)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTestPatchConv.
///
/// - Parameters:
///     - size: The size of the input data.
///     - patch: The kernel split size of the input data.
/// - Returns: The gradient norm.
///
func computePatchConvGradNorm(size: Int, patch: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_patch_conv_grad_norm(size, patch)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTestAttention1.
///
/// - Parameters:
///     - size: The size of the input data.
///     - patch: The kernel split size of the input data.
/// - Returns: The gradient norm.
///
func computeAttention1GradNorm(size: Int, patch: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_attention1_grad_norm(size, patch)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTestAttention2.
///
/// - Parameters:
///     - size: The size of the input data.
///     - patch: The kernel split size of the input data.
/// - Returns: The gradient norm.
///
func computeAttention2GradNorm(size: Int, patch: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_attention2_grad_norm(size, patch)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTestLayerNorm.
///
/// - Parameters:
///     - size: The size of the input data.
///     - patch: The kernel split size of the input data.
/// - Returns: The gradient norm.
///
func computeLayerNormGradNorm(size: Int, patch: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_layer_norm_grad_norm(size, patch)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTestAutoEncoder1.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeAutoEncoder1GradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_auto_encoder1_grad_norm(size)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTestAutoEncoder2.
///
/// - Parameter size: The size of the input data.
/// - Returns: The gradient norm.
///
func computeAutoEncoder2GradNorm(_ size: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_auto_encoder2_grad_norm(size)
    return Float(data)!
}

///
/// Get gradient norm computed with PyTorch for ModelTestGram.
///
/// - Parameters:
///     - size: The size of the input data.
///     - batchSize: The number of images in the batch.
/// - Returns: The gradient norm.
///
func computeGramGradNorm(size: Int, batchSize: Int) -> Float
{
    let pythonLib = Python.import("python_lib")
    let data = pythonLib.compute_gram_grad_norm(size, batchSize)
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
