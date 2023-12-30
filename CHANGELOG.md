# 👨‍💻 Changelog

All notable changes to this project will be documented in this file.

## [unreleased]

⚙️ **core:** GELU vs GELUApprox ([113](https://github.com/owkin/GrAIdient/pull/113))\
🚀 **perf:** QuerySelf & ValueSelf ([112](https://github.com/owkin/GrAIdient/pull/112))\
🚀 **perf:** benchmark ViT base model ([111](https://github.com/owkin/GrAIdient/pull/111))\
🐛 **fix:** run on Apple Silicon ([110](https://github.com/owkin/GrAIdient/pull/110))\
⚙️ **core:** initForward,Backward model API ([109](https://github.com/owkin/GrAIdient/pull/109))\
🪜 **layer_1d:** Dropout1D ([#108](https://github.com/owkin/GrAIdient/pull/108))\
🪜 **feat:** VQGrad, VQGradSeq ([#107](https://github.com/owkin/GrAIdient/pull/107))

## 0.3.1 (2023-08-09)

### Bug Fixes

🐛 **fix**: input GPU check ([#106](https://github.com/owkin/GrAIdient/pull/106))

## 0.3.0 (2023-08-04)

### Features

🪜 **feat:** BCE1D, BCE2D, VQ2D & VQSeq as losses ([#101](https://github.com/owkin/GrAIdient/pull/101))\
🪜 **layer_seq:** VQSeq ([#100](https://github.com/owkin/GrAIdient/pull/100))\
🪜 **layer_2d:** loosen range contraint in ColorJitterHSV ([#98](https://github.com/owkin/GrAIdient/pull/98))\
🪜 **layer_2d:** SimilarityError2D & dirty losses ([#97](https://github.com/owkin/GrAIdient/pull/97))\
🪜 **layer_2d:** ColorJitterHSV, Image & ImageTests ([#93](https://github.com/owkin/GrAIdient/pull/93))\
🪜 **layer_2d:** Flip2D & config_kernels ([#92](https://github.com/owkin/GrAIdient/pull/92))\
🪜 **layer_2d:** SimilarityBatchError2D ([#88](https://github.com/owkin/GrAIdient/pull/88))\
🪜 **layer_2d:** Normalize2D ([#87](https://github.com/owkin/GrAIdient/pull/87))\
🪜 **layer_2d:** SelfCorrelate2D ([#86](https://github.com/owkin/GrAIdient/pull/86))\
🪜 **layer_2d**: VQ2D ([#81](https://github.com/owkin/GrAIdient/pull/81))\
🪜 **layer_seq**: Adding new layer SelectNeuronsSeq ([#77](https://github.com/owkin/GrAIdient/pull/77))\
⚙️ **core:** GELU activation function ([#73](https://github.com/owkin/GrAIdient/pull/73))\
🪜 **layer_seq:** ValueSeq ([#69](https://github.com/owkin/GrAIdient/pull/69))\
🪜 **layer_seq:** SoftmaxSeq ([#68](https://github.com/owkin/GrAIdient/pull/68))\
🪜 **layer_seq:** QuerySeq ([#67](https://github.com/owkin/GrAIdient/pull/67))\
🪜 **layer_seq:** LayerNormSeq & LayerNormalization ([#66](https://github.com/owkin/GrAIdient/pull/66))\
🪜 **layer_seq:** FullyConnectedSeq ([#65](https://github.com/owkin/GrAIdient/pull/65))\
🪜 **layer_seq:** Constant12Seq & Constant2Seq ([#64](https://github.com/owkin/GrAIdient/pull/64))\
🪜 **layer_seq:** Concat1Seq & Concat2Seq ([#63](https://github.com/owkin/GrAIdient/pull/63))\
🪜 **layer_seq:** SumSeq ([#62](https://github.com/owkin/GrAIdient/pull/62))\
🪜 **layer_2d:** MSE2D & LayerOutput2D ([#61](https://github.com/owkin/GrAIdient/pull/61))\
🪜 **layer_seq:** FullyConnectedPatch & base classes ([#60](https://github.com/owkin/GrAIdient/pull/60))\
🪜 **layer_2d:** Constant2D ([#56](https://github.com/owkin/GrAIdient/pull/56))\
🪜 **layer_2d:** AdaIN ([#55](https://github.com/owkin/GrAIdient/pull/55))\
🪜 **layer_2d:** InstanceNorm2D & InstanceNormalization ([#54](https://github.com/owkin/GrAIdient/pull/54))

### Bug Fixes

🐛 **layer_2d**: align Convolution & Deconvolution on PyTorch ([#84](https://github.com/owkin/GrAIdient/pull/84))\
🐛 **fix**: numerical stability of tanh for GELU ([#83](https://github.com/owkin/GrAIdient/pull/83))\
🐛 **fix:** numerical instability of Softmax ([#76](https://github.com/owkin/GrAIdient/pull/76))\
🐛 **fix:** update ValueSeq operation ([#72](https://github.com/owkin/GrAIdient/pull/72))

### Miscellaneous Tasks

🔨 **refactor:** throwable init ([#103](https://github.com/owkin/GrAIdient/pull/103))\
🔨 **refactor:** dims checks for inputs and outputs ([#102](https://github.com/owkin/GrAIdient/pull/102))\
🔨 **layer_2d:** expose indices in VQ2D ([#99](https://github.com/owkin/GrAIdient/pull/99))\
🔨 **core:** LayerWeightInit ([#96](https://github.com/owkin/GrAIdient/pull/96))\
🚨 **test**: FlowAccumulateTrainer ([#95](https://github.com/owkin/GrAIdient/pull/95))\
🚨 **examples**: compare training with PyTorch ([#94](https://github.com/owkin/GrAIdient/pull/94))\
🔨 **layer_2d:** remove computeVQ ([#91](https://github.com/owkin/GrAIdient/pull/91))\
🔨 **layer_2d:** API for random transforms ([#90](https://github.com/owkin/GrAIdient/pull/90))\
🚀 **perf:** enhance Normalize122D with reduce ([#89](https://github.com/owkin/GrAIdient/pull/89))\
🚨 **integration**: resize alignment with PyTorch ([#85](https://github.com/owkin/GrAIdient/pull/85))\
🔨 **layer_seq**: SelectSeq ([#82](https://github.com/owkin/GrAIdient/pull/82))\
🚀 **examples**: AutoEncoder models ([#79](https://github.com/owkin/GrAIdient/pull/79))\
🚀 **layer_seq**: factorize by nbHeads ([#78](https://github.com/owkin/GrAIdient/pull/78))\
🚀 **examples:** make Transformer example very simple ([#75](https://github.com/owkin/GrAIdient/pull/75))\
🚀 **examples:** adding Transformer training example ([#74](https://github.com/owkin/GrAIdient/pull/74))\
🚨 **integration:** update & validate LayerNormSeq ([#71](https://github.com/owkin/GrAIdient/pull/71))\
🚨 **integration:** validate MultiHeadAttention & fix Softmax stability ([#70](https://github.com/owkin/GrAIdient/pull/70))

## 0.2.0 (2023-02-27)

### Features

🪜 **layer_1d:** Softmax1D, DotProduct1D & Constant1D ([#49](https://github.com/owkin/GrAIdient/pull/49))\
🪜 **feat:** remove activation from layer ([#47](https://github.com/owkin/GrAIdient/pull/47))\
🪜 **feat:** LayerMerge1D, Sum1D, Concat1D, Concat2D ([#43](https://github.com/owkin/GrAIdient/pull/43))\
🪜 **layer_2d:** Deconvolution2D ([#42](https://github.com/owkin/GrAIdient/pull/42))\
🪜 **feat:** getDeltaWeightsGPU per sample API ([#41](https://github.com/owkin/GrAIdient/pull/41))

### Bug Fixes

🐛 **fix:** use buffers for neuron selection in SelectNeurons1D ([#50](https://github.com/owkin/GrAIdient/pull/50))\
🐛 **fix:** model context max id ([#45](https://github.com/owkin/GrAIdient/pull/45))\
🐛 **fix:** remove error when data input may indicate lower batch size ([#44](https://github.com/owkin/GrAIdient/pull/44))

### Miscellaneous Tasks

📚 **docs:** change project description and add links ([#57](https://github.com/owkin/GrAIdient/pull/57))\
📚 **docs:** PropertyListEncoder by default ([#51](https://github.com/owkin/GrAIdient/pull/51))\
🎉 **refactor:** logo ([#46](https://github.com/owkin/GrAIdient/pull/46))\
🎉 **refactor!:** re brand the framework ([#40](https://github.com/owkin/GrAIdient/pull/40))

## 0.1.1 (2022-12-16)

### Features

🪜 **layer_2d:** ResizeBilinearCrop ([#36](https://github.com/owkin/GrAIdient/pull/36))\
🚀 **perf:** enhance backwardGPU for ResizeBilinear ([#35](https://github.com/owkin/GrAIdient/pull/35))\
🪜 **layer_2d:** Rotate2D ([#34](https://github.com/owkin/GrAIdient/pull/34))\
🪜 **layer_2d:** ResizeBilinear ([#32](https://github.com/owkin/GrAIdient/pull/32))\
🪜 **layer_2d:** Pad2D & Jitter2D ([#30](https://github.com/owkin/GrAIdient/pull/30))\
🪜 **layer_2d:** add tests for non dirty status ([#27](https://github.com/owkin/GrAIdient/pull/27))\
🪜 **layer_2d:** FTFrequences2D & Multiply2D ([#25](https://github.com/owkin/GrAIdient/pull/25))\
🪜 **layer_2d:** LinearScale2D ([#24](https://github.com/owkin/GrAIdient/pull/24))\
🪜 **layer_2d:** DecorelateRGB ([#23](https://github.com/owkin/GrAIdient/pull/23))\
🪜 **layer_2d:** RDFT2Image ([#22](https://github.com/owkin/GrAIdient/pull/22))\
🪜 **core:** Sigmoid activation ([#21](https://github.com/owkin/GrAIdient/pull/21))\
🚀 **metal:** systematic dispatchThreads API ([#19](https://github.com/owkin/GrAIdient/pull/19))

### Bug Fixes 

🐛 **fix:** update correlation matrix coeffs ([#37](https://github.com/owkin/GrAIdient/pull/37))\
🐛 **fix:** ResizeBilinear to output deterministic dimensions ([#33](https://github.com/owkin/GrAIdient/pull/33))

### Miscellaneous Tasks

🔨 **refactor:** remove transaction ([#31](https://github.com/owkin/GrAIdient/pull/31))\
🚨 **integration:** activate DecorrelateRGB in test ([#29](https://github.com/owkin/GrAIdient/pull/29))\
🚨 **integration:** test IDFT and complex numbers ([#28](https://github.com/owkin/GrAIdient/pull/28))\
🔨 **test:** factorize transform tests ([#26](https://github.com/owkin/GrAIdient/pull/26))\
👷 **ci:** remove swift action ([#20](https://github.com/owkin/GrAIdient/pull/20))\
👷 **ci:** remove LFS ([#17](https://github.com/owkin/GrAIdient/pull/17))

## 0.1.0 (2022-10-28)

### Features

⚙️ **core:** remove incEpoch & applyGradient rename ([#11](https://github.com/owkin/GrAIdient/pull/11))\
🚀 **examples:** simple vgg trained on cifar ([#9](https://github.com/owkin/GrAIdient/pull/9))\
🪜 **layer_2d:** convolution, bn and other 2D layers ([#7](https://github.com/owkin/GrAIdient/pull/7))\
🪜 **layer_1d:** activation, fl, linear error, mse, select channels ([#5](https://github.com/owkin/GrAIdient/pull/5))\
⚙️ **core:** Layer architecture ([#4](https://github.com/owkin/GrAIdient/pull/4))\
⚙️ **core:** Optimizer architecture ([#3](https://github.com/owkin/GrAIdient/pull/3))\
⚙️ **core:** Model architecture ([#2](https://github.com/owkin/GrAIdient/pull/2))\
⚡️ **metal:** Metal architecture ([#1](https://github.com/owkin/GrAIdient/pull/1))

### Documentation

📚 update the readme and add documentation ([#12](https://github.com/owkin/GrAIdient/pull/12))

### Miscellaneous Tasks

🔧 **chore:** release 0.1.0 ([#13](https://github.com/owkin/GrAIdient/pull/13))\
🚀 **test:** reproducibility with PyTorch ([#10](https://github.com/owkin/GrAIdient/pull/10))\
🪜 **test:** layer2d ([#8](https://github.com/owkin/GrAIdient/pull/8))\
⚙️ **test:** optimizer, layer1d, clipping ([#6](https://github.com/owkin/GrAIdient/pull/6))
