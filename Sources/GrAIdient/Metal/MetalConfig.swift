//
// MetalConfig.swift
// GrAIdient
//
// Created by Jean-François Reboud on 18/05/2023.
//

let CONFIG_KERNELS =
[
    "Activation": [
        "forwardReLU",
        "backwardReLU",
        "forwardLeakyReLU",
        "backwardLeakyReLU",
        "forwardSoftReLU",
        "backwardSoftReLU",
        "forwardSigmoid",
        "backwardSigmoid",
        "forwardGELU",
        "backwardGELU",
    ],
    "Biases": [
        "reduceBiases",
    ],
    "BatchNorm": [
        "computeBNConvμ",
        "computeBNConvσ2",
        "forwardBNConvTraining",
        "forwardBNConvInference",
        "backwardWeightsBNConv",
        "backwardBNConvTraining",
        "backwardBNConvInference",
    ],
    "Convolution": [
        "convForward",
        "convBackward",
        "convBatchDerWeights",
        "convBatchDerBiases",
        "convDerWeights",
        "convDerBiases",
        "convReduceWeights",
    ],
    "Deconvolution": [
        "deconvForward",
        "deconvBackward",
        "deconvBatchDerWeights",
        "deconvDerWeights",
    ],
    "FullyConnected": [
        "flForward",
        "flBackward",
        "flBatchDerWeights",
        "flBatchDerBiases",
        "flDerWeights",
        "flDerBiases",
        "flReduceWeights",
    ],
    "FullyConnectedPatch": [
        "flPatchForward",
        "flPatchBackward",
        "flPatchBatchDerWeights",
        "flPatchBatchDerBiases",
        "flPatchDerWeights",
        "flPatchDerBiases",
        "flPatchReduceWeights",
    ],
    "FullyConnectedSeq": [
        "flSeqForward",
        "flSeq4Forward",
        "flSeqBackward",
        "flSeq4Backward",
        "flSeqBatchDerWeights",
        "flSeqBatch4DerWeights",
        "flSeqDerWeights",
        "flSeqReduceWeights",
    ],
    "InstanceNorm": [
        "computeInstanceNormConvμ",
        "computeInstanceNormConvσ2",
        "forwardInstanceNormConv",
        "forwardAdaIN",
        "backwardWeightsInstanceNormConv",
        "backward2AdaIN",
        "backwardInstanceNormConv",
        "backward1AdaIN",
    ],
    "Layer1D": [
        "MSE1DLoss",
        "MSE1DLossDerivative",
        "linearErrorLoss",
        "linearErrorLossDerivative",
        "selectNeurons1DForward",
        "selectNeurons1DBackward",
        "concat1DForward",
        "concat1DBackward",
        "softmax1DForward",
        "softmax1DBackward",
        "dotProduct1DForward",
        "dotProduct1DBackward",
        "constant1DForward",
        "BCE1DLoss",
        "BCE1DLossDerivative",
        "BCESigmoid1DLoss",
        "BCESigmoid1DLossDerivative",
        "dropout1DForward",
        "dropout1DBackward",
    ],
    "Layer2D": [
        "avgPoolForward",
        "avgPoolBackward",
        "maxPoolForward",
        "maxPoolBackward",
        "adaptiveAvgPoolForward1",
        "adaptiveAvgPoolForward2",
        "adaptiveAvgPoolBackward1",
        "adaptiveAvgPoolBackward2",
        "selectNeurons2DForward",
        "selectNeurons2DBackward",
        "IRDFT2RGBForward",
        "IRDFT2RGBBackward",
        "decorrelateRGBForward",
        "decorrelateRGBBackward",
        "linearScale2DForward",
        "linearScale2DBackward",
        "setDataFTFrequences2D",
        "pad2DForward",
        "pad2DBackward",
        "crop2DForward",
        "crop2DBackward",
        "resizeBilinearPadForward",
        "resizeBilinearPadBackward",
        "rotate2DForward",
        "rotate2DBackward",
        "resizeBilinearCropForward",
        "resizeBilinearCropBackward",
        "concat02DForward",
        "concat02DBackward",
        "concat12DForward",
        "concat12DBackward",
        "constant2DForward",
        "MSE2DLoss",
        "MSE2DLossDerivative",
        "selfCorrelate2DForward",
        "selfCorrelate2DBackward",
        "normalize12DForward",
        "normalize12DBackward",
        "computeSquaredNorm122D",
        "normalize122DForward",
        "computeDeltaTmp122D",
        "normalize122DBackward",
        "similarBatchError2DLoss",
        "similarBatchError2DLossDerivative",
        "similarError2DLossDerivative",
        "flipHorizontal2DForward",
        "flipHorizontal2DBackward",
        "flipVertical2DForward",
        "flipVertical2DBackward",
        "colorJitterHSVForward",
        "BCE2DLoss",
        "BCE2DLossDerivative",
        "BCESigmoid2DLoss",
        "BCESigmoid2DLossDerivative",
    ],
    "LayerMerge": [
        "sum1",
        "sum2",
        "multiplyForward",
        "multiplyBackward",
    ],
    "LayerNorm": [
        "computeLayerNormSeqμ",
        "computeLayerNormSeqσ2",
        "forwardLayerNormSeq",
        "backwardWeights1LayerNormSeq",
        "backwardWeights2LayerNormSeq",
        "backwardLayerNormSeq",
    ],
    "LayerSeq": [
        "avgPoolSeqForward",
        "avgPoolSeqBackward",
        "concat1SeqForward",
        "concat1SeqBackward",
        "concat2SeqForward",
        "concat2SeqBackward",
        "constant12SeqForward",
        "constant12SeqBackward",
        "constant2SeqForward",
        "querySeqForward",
        "queryQuerySeqBackward",
        "queryKeySeqBackward",
        "softmaxSeqForward",
        "softmaxSeqBackward",
        "valueSeqForward",
        "valueValueSeqBackward",
        "valueScoreSeqBackward",
        "selectSeqForward",
        "selectSeqBackward",
    ],
    "Optimizer": [
        "clipGradients",
        "multiplyGradients",
        "weightsSGD",
        "weightsMomentum",
        "weightsAdam",
        "weightsAMSGrad",
        "weightsAdamRectified",
        "weightsAdaBound",
        "weightsAMSBound",
    ],
    "Reduce": [
        "reduceSum64",
        "reduceSum",
        "reduceMax64",
        "reduceMax",
    ],
    "Reset": [
        "reset"
    ],
    "VQ2D": [
        "vq2DForward",
        "vq2DBackward",
        "vq2DBatchDerWeights",
        "vq2DDerWeights",
        "vq2DReduceWeights",
        "vq2DLoss",
        "vqGrad2DMax",
        "vqGrad2DForward"
    ],
    "VQSeq": [
        "vqSeqForward",
        "vqSeqBackward",
        "vqSeqBatchDerWeights",
        "vqSeqDerWeights",
        "vqSeqLoss",
        "vqGradSeqMax",
        "vqGradSeqForward"
    ]
]
