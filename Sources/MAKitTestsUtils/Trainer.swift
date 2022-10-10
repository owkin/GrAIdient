//
// Trainer.swift
// MAKitTestsUtils
//
// Created by Jean-François Reboud on 10/10/2022.
//

import Foundation
import MAKit

public func setOptimizerParams(
    params: inout MAKit.Optimizer.Params,
    optimizerClass: MAKit.Optimizer.Class = .SGD,
    alpha: Double = 0.001,
    lambda: Double? = nil,
    lowerBound: Double = 0.0001,
    upperBound: Double = 0.005)
{
    params.optimizer = ConstEpochsScheduler(optimizerClass)
    
    let lambdaTime = lambda != nil ?
        ConstEpochsVar(value: ConstVal(lambda!)) :
        ConstEpochsVar(value: NullVal())
    
    params.variables =
    [
        "alpha": ConstEpochsVar(value: ConstVal(alpha)),
        "lambda": lambdaTime,
        "lowerBound": ConstEpochsVar(value: ConstVal(lowerBound)),
        "upperBound": ConstEpochsVar(value: ConstVal(upperBound))
    ]
}

public let DEVICE_ID_DEFAULT = 0

open class Trainer
{
    public let modelName: String
    public var optimizerParams: MAKit.Optimizer.Params
    
    public init(name: String, params: MAKit.Optimizer.Params)
    {
        modelName = name
        optimizerParams = params
    }
}

open class GradTrainer: Trainer
{
    public var model: Model! = nil
    
    public func build(_ buildFct: (ModelContext)->())
    {
        let context = ModelContext(name: modelName, curID: 0)
        buildFct(context)
        model = Model(model: context.model, modelsPrev: [])
    }
    
    public func run<DataT, LossT>(
        layersGraph: [Layer],
        setData: (DataT?, Model)->(DataT, Int),
        setLoss: (LossT?, Model)->(LossT),
        getGradientsApprox: (LossT, Model)->[Double],
        validate: (Double)->())
    {
        model.initialize(
            params: optimizerParams,
            phase: .Training,
            deviceID: DEVICE_ID_DEFAULT
        )
        
        if MAKit.Opti.GPU
        {
            fatalError("COUCOU1")
        }
        
        var epoch = 0
        let nbEpochsMax = 1
        while epoch < nbEpochsMax
        {
            var numLoop = 0
            while numLoop < optimizerParams.nbLoops
            {
                let (_, batchSize) = setData(nil, model)
                model.updateKernel(batchSize: batchSize)
                
                try! model.forward()
                try! model.forwardGradients()
                
                let gt = setLoss(nil, model)
                try! model.backward()
                try! model.update()
                
                let gradientsApprox = getGradientsApprox(gt, model)
                if let gradDiff = checkGradients(
                    model: model,
                    layersGraph: layersGraph,
                    gradientsApprox: gradientsApprox)
                {
                    validate(gradDiff)
                }
                
                model.incStep()
                numLoop += 1
            }
            
            model.incEpoch()
            epoch += 1
        }
    }
}

open class FlowTrainer: Trainer
{
    public var models: [Model] = []
    
    public var modelCPU: Model
    {
        get {
            return models[0]
        }
    }
    public var modelGPU: Model
    {
        get {
            return models[1]
        }
    }
    
    public func build(_ buildFct: (ModelContext)->())
    {
        var baseModels = [BaseModel]()
        
        let context = ModelContext(name: modelName + "CPU", curID: 0)
        buildFct(context)
        baseModels.append(context.model)
        
        context.model = BaseModel(name: modelName + "GPU")
        buildFct(context)
        baseModels.append(context.model)
        
        var models = [Model]()
        for baseModel in baseModels
        {
            models.append(Model(model: baseModel, modelsPrev: []))
        }
        self.models = models
    }

    public func initialize()
    {
        for i in 0...1
        {
            if i == 0
            {
                MAKit.Opti.CPU = true
            }
            
            if i > 0
            {
                models[i].weights = models[i-1].weights
            }
            
            if i == 1
            {
                MAKit.Opti.GPU = true
            }
            
            models[i].initialize(
                params: optimizerParams,
                phase: .Training,
                deviceID: DEVICE_ID_DEFAULT
            )
        }
    }
    
    public func run<DataT, LossT>(
        setData: (DataT?, Model)->(DataT, Int),
        setLoss: (LossT?, Model)->(LossT),
        validate: (Double)->())
    {
        initialize()
        
        var epoch = 0
        let nbEpochsMax = 1
        while epoch < nbEpochsMax
        {
            var numLoop = 0
            while numLoop < optimizerParams.nbLoops
            {
                let resultsCPU: [Double]
                MAKit.Opti.CPU = true
                
                var (inputs, batchSize) = setData(nil, modelCPU)
                modelCPU.updateKernel(batchSize: batchSize)
                try! modelCPU.forward()
                
                var gt = setLoss(nil, modelCPU)
                try! modelCPU.backward()
                try! modelCPU.update()
                
                resultsCPU = getGradients(model: modelCPU)
                
                let resultsGPU: [Double]
                MAKit.Opti.GPU = true
                
                (inputs, batchSize) = setData(inputs, modelGPU)
                modelGPU.updateKernel(batchSize: batchSize)
                try! modelGPU.forward()
                
                gt = setLoss(gt, modelGPU)
                try! modelGPU.backward()
                try! modelGPU.update()
                
                resultsGPU = getGradients(model: modelGPU)
                
                if let gradDiff = checkFlow(resultsCPU, resultsGPU)
                {
                    validate(gradDiff)
                }
                
                modelCPU.incStep()
                modelGPU.incStep()
                numLoop += 1
            }
            
            modelCPU.incEpoch()
            modelGPU.incEpoch()
            epoch += 1
        }
    }
}

open class FlowResetTrainer: FlowTrainer
{
    public func reset()
    {
        for i in 0...1
        {
            if i == 0
            {
                MAKit.Opti.CPU = true
            }
            else if i == 1
            {
                MAKit.Opti.GPU = true
            }
            models[i].resetKernel()
        }
    }
    
    public override func run<DataT, LossT>(
        setData: (DataT?, Model)->(DataT, Int),
        setLoss: (LossT?, Model)->(LossT),
        validate: (Double)->())
    {
        super.run(setData: setData, setLoss: setLoss){ (Double) in }
        
        reset()
        
        super.run(setData: setData, setLoss: setLoss, validate: validate)
    }
}

open class FlowReverseTrainer: FlowTrainer
{
    public func initializeReverse()
    {
        for i in 0...1
        {
            if i == 0
            {
                MAKit.Opti.CPU = true
                models[0].weights = models[0].weights
                MAKit.Opti.GPU = true
            }
            else if i == 1
            {
                MAKit.Opti.GPU = true
                models[1].weights = models[1].weights
                MAKit.Opti.CPU = true
            }
            
            models[i].initialize(
                params: optimizerParams,
                phase: .Training,
                deviceID: DEVICE_ID_DEFAULT
            )
        }
    }
    
    public override func run<DataT, LossT>(
        setData: (DataT?, Model)->(DataT, Int),
        setLoss: (LossT?, Model)->(LossT),
        validate: (Double)->())
    {
        super.run(setData: setData, setLoss: setLoss){ (Double) in }
        
        initializeReverse()
        
        var epoch = 0
        let nbEpochsMax = 1
        while epoch < nbEpochsMax
        {
            var numLoop = 0
            while numLoop < optimizerParams.nbLoops
            {
                let resultsGPU: [Double]
                MAKit.Opti.GPU = true
                
                var (inputs, batchSize) = setData(nil, modelCPU)
                modelCPU.updateKernel(batchSize: batchSize)
                try! modelCPU.forward()
                
                var gt = setLoss(nil, modelCPU)
                try! modelCPU.backward()
                try! modelCPU.update()
                
                resultsGPU = getGradients(model: modelCPU)
                
                let resultsCPU: [Double]
                MAKit.Opti.CPU = true
                
                (inputs, batchSize) = setData(inputs, modelGPU)
                modelGPU.updateKernel(batchSize: batchSize)
                try! modelGPU.forward()
                
                gt = setLoss(gt, modelGPU)
                try! modelGPU.backward()
                try! modelGPU.update()
                
                resultsCPU = getGradients(model: modelGPU)
                
                if let gradDiff = checkFlow(resultsCPU, resultsGPU)
                {
                    validate(gradDiff)
                }
                
                modelCPU.incStep()
                modelGPU.incStep()
                numLoop += 1
            }
            
            modelCPU.incEpoch()
            modelGPU.incEpoch()
            epoch += 1
        }
    }
}

open class InferenceTrainer: FlowTrainer
{
    public func run<DataT, LossT>(
        setData: (DataT?, Model)->(DataT, Int),
        setLoss: (LossT?, Model)->(LossT),
        getLoss: (LossT, Model)->Double,
        validate: (Double)->())
    {
        super.run(setData: setData, setLoss: setLoss){ (Double) in }
        
        modelCPU.phase = .Inference
        modelGPU.phase = .Inference
        
        var epoch = 0
        let nbEpochsMax = 1
        while epoch < nbEpochsMax
        {
            var numLoop = 0
            while numLoop < optimizerParams.nbLoops
            {
                MAKit.Opti.CPU = true
                var (inputs, batchSize) = setData(nil, modelCPU)
                modelCPU.updateKernel(batchSize: batchSize)
                try! modelCPU.forward()
                
                var gt = setLoss(nil, modelCPU)
                let lossCPU = getLoss(gt, modelCPU)
                
                MAKit.Opti.GPU = true
                (inputs, batchSize) = setData(inputs, modelGPU)
                modelGPU.updateKernel(batchSize: batchSize)
                try! modelGPU.forward()
                
                gt = setLoss(gt, modelGPU)
                let lossGPU = getLoss(gt, modelGPU)
                
                let diff = (lossGPU - lossCPU) * (lossGPU - lossCPU) /
                           (lossCPU * lossCPU + lossGPU * lossGPU)
                
                var warning = ""
                if diff > 0.0000001
                {
                    warning = "Inference Check Warning : "
                }
                let strDump = warning + String(diff)
                print(strDump)
                
                validate(diff)
                numLoop += 1
            }
            
            epoch += 1
        }
    }
}

open class LoadTrainer: FlowTrainer
{
    public func run<DataT, LossT>(
        setData: (DataT?, Model)->(DataT, Int),
        setLoss: (LossT?, Model)->(LossT),
        getLoss: (LossT, Model)->Double,
        validate: (Double, Double)->())
    {
        // 1. Train modelCPU and modelGPU.
        
        super.run(setData: setData, setLoss: setLoss){ (Double) in }
        
        // 2. Save modelCPU and modelGPU on the disk.
        
        let folderURL = FileManager.default.temporaryDirectory
        let modelCPUPath =
            folderURL.appendingPathComponent("testCPU.plist").path
        let modelGPUPath =
            folderURL.appendingPathComponent("testGPU.plist").path
        
        let encoder = PropertyListEncoder()
        
        MAKit.Opti.CPU = true
        var data = try! encoder.encode(modelCPU)
        try! data.write(to: URL(fileURLWithPath: modelCPUPath))
        
        MAKit.Opti.GPU = true
        data = try! encoder.encode(modelGPU)
        try! data.write(to: URL(fileURLWithPath: modelGPUPath))
        
        // 3. Run one step of Inference for modelCPU and modelGPU.
        
        modelCPU.phase = .Inference
        modelGPU.phase = .Inference
        
        let lossCPURef: Double
        let lossGPURef: Double
        
        MAKit.Opti.CPU = true
        var (inputs, batchSize) = setData(nil, modelCPU)
        modelCPU.updateKernel(batchSize: batchSize)
        try! modelCPU.forward()
        
        var gt = setLoss(nil, modelCPU)
        lossCPURef = getLoss(gt, modelCPU)
        
        MAKit.Opti.GPU = true
        (inputs, batchSize) = setData(inputs, modelGPU)
        modelGPU.updateKernel(batchSize: batchSize)
        try! modelGPU.forward()
        
        gt = setLoss(gt, modelGPU)
        lossGPURef = getLoss(gt, modelGPU)
        
        // 4. Load models from the disk.
        
        data = try! Data(contentsOf: URL(fileURLWithPath: modelCPUPath))
        let baseModelCPU = try! PropertyListDecoder().decode(BaseModel.self,
                                                             from: data)
        data = try! Data(contentsOf: URL(fileURLWithPath: modelGPUPath))
        let baseModelGPU = try! PropertyListDecoder().decode(BaseModel.self,
                                                             from: data)
        
        let modelCPU = Model(model: baseModelCPU, layersPrev: [])
        let modelGPU = Model(model: baseModelGPU, layersPrev: [])
                
        // 5. Run one step of Inference for loaded models.
        
        let lossCPUFromDisk: Double
        let lossGPUFromDisk: Double
        
        MAKit.Opti.CPU = true
        modelCPU.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID_DEFAULT
        )
        (inputs, batchSize) = setData(inputs, modelCPU)
        modelCPU.updateKernel(batchSize: batchSize)
        try! modelCPU.forward()
        
        gt = setLoss(gt, modelCPU)
        lossCPUFromDisk = getLoss(gt, modelCPU)
        
        MAKit.Opti.GPU = true
        modelGPU.initialize(
            params: optimizerParams,
            phase: .Inference,
            deviceID: DEVICE_ID_DEFAULT
        )
        (inputs, batchSize) = setData(inputs, modelGPU)
        modelGPU.updateKernel(batchSize: batchSize)
        try! modelGPU.forward()
        
        gt = setLoss(gt, modelGPU)
        lossGPUFromDisk = getLoss(gt, modelGPU)
        
        // 6. Compare results.
        
        let diffCPU =
            (lossCPUFromDisk - lossCPURef) * (lossCPUFromDisk - lossCPURef) /
            (lossCPUFromDisk * lossCPUFromDisk + lossCPURef * lossCPURef)
        let diffGPU =
            (lossGPUFromDisk - lossGPURef) * (lossGPUFromDisk - lossGPURef) /
            (lossGPUFromDisk * lossGPUFromDisk + lossGPURef * lossGPURef)
        
        var warning = ""
        let maxDiff = max(diffCPU, diffGPU)
        let maxIndex = diffCPU < diffGPU ? "GPU" : "CPU"
        if diffCPU > 0.0000001
        {
            warning = "Load Check Warning " + maxIndex + " : "
        }
        let strDump = warning + String(maxDiff)
        print(strDump)
        
        validate(diffCPU, diffGPU)
    }
}

open class TransformTrainer: FlowTrainer
{
    public func run<DataT, LossT>(
        transform: (Model)->Model,
        setData: (DataT?, Model)->(DataT, Int),
        setLoss: (LossT?, Model)->(LossT),
        getLoss: (LossT, Model)->Double,
        validate: (Double, Double)->())
    {
        // 1. Train modelCPU and modelGPU.
        
        super.run(setData: setData, setLoss: setLoss){ (Double) in }
        
        // 2. Run one step of Inference for modelCPU and modelGPU.
        
        modelCPU.phase = .Inference
        modelGPU.phase = .Inference
        
        let lossCPURef: Double
        let lossGPURef: Double
        
        MAKit.Opti.CPU = true
        var (inputs, batchSize) = setData(nil, modelCPU)
        modelCPU.updateKernel(batchSize: batchSize)
        try! modelCPU.forward()
        
        var gt = setLoss(nil, modelCPU)
        lossCPURef = getLoss(gt, modelCPU)
        
        MAKit.Opti.GPU = true
        (inputs, batchSize) = setData(inputs, modelGPU)
        modelGPU.updateKernel(batchSize: batchSize)
        try! modelGPU.forward()
        
        gt = setLoss(gt, modelGPU)
        lossGPURef = getLoss(gt, modelGPU)
        
        // 3. Create transformed models.
        
        MAKit.Opti.CPU = true
        let modelCPU: Model = transform(modelCPU)
        MAKit.Opti.GPU = true
        let modelGPU: Model = transform(modelGPU)
        
        // 4. Run one step of Inference on the new models.
        
        let lossCPUNew: Double
        let lossGPUNew: Double
        
        MAKit.Opti.CPU = true
        (inputs, batchSize) = setData(inputs, modelCPU)
        modelCPU.updateKernel(batchSize: batchSize)
        try! modelCPU.forward()
        
        gt = setLoss(gt, modelCPU)
        lossCPUNew = getLoss(gt, modelCPU)
        
        MAKit.Opti.GPU = true
        (inputs, batchSize) = setData(inputs, modelGPU)
        modelGPU.updateKernel(batchSize: batchSize)
        try! modelGPU.forward()
        
        gt = setLoss(gt, modelGPU)
        lossGPUNew = getLoss(gt, modelGPU)
        
        // 5. Compare results.
        
        let diffCPU =
            (lossCPUNew - lossCPURef) * (lossCPUNew - lossCPURef) /
            (lossCPUNew * lossCPUNew + lossCPURef * lossCPURef)
        let diffGPU =
            (lossGPUNew - lossGPURef) * (lossGPUNew - lossGPURef) /
            (lossGPUNew * lossGPUNew + lossGPURef * lossGPURef)
        
        var warning = ""
        let maxDiff = max(diffCPU, diffGPU)
        let maxIndex = diffCPU < diffGPU ? "GPU" : "CPU"
        if diffCPU > 0.0000001
        {
            warning = "Load Check Warning " + maxIndex + " : "
        }
        let strDump = warning + String(maxDiff)
        print(strDump)
        
        validate(diffCPU, diffGPU)
    }
}

open class NormTrainer: Trainer
{
    public var model: Model! = nil
    
    public override init(name: String,
                         params: MAKit.Optimizer.Params)
    {
        super.init(name: name, params: params)
    }
    
    public func build(_ buildFct: (ModelContext)->())
    {
        let context = ModelContext(name: modelName, curID: 0)
        buildFct(context)
        model = Model(model: context.model, modelsPrev: [])
    }
    
    public func run<DataT, LossT>(
        setData: (DataT?, Model)->(DataT, Int),
        setLoss: (LossT?, Model)->(LossT),
        validate: (Double)->())
    {
        model.initialize(
            params: optimizerParams,
            phase: .Training,
            deviceID: DEVICE_ID_DEFAULT
        )
        
        var epoch = 0
        let nbEpochsMax = 1
        let normClipping = optimizerParams.normThreshold
        while epoch < nbEpochsMax
        {
            var numLoop = 0
            while numLoop < optimizerParams.nbLoops
            {
                let (_, batchSize) = setData(nil, model)
                model.updateKernel(batchSize: batchSize)
                try! model.forward()
                
                _ = setLoss(nil, model)
                try! model.backward()
                try! model.update()
                
                let gradNorm = try! model.getGradientNorm()
                let diff = abs(gradNorm - normClipping) / abs(normClipping)
                
                var result = ""
                if diff > 0.0000001
                {
                    result = "Norm Check Warning : "
                }
                result += String(diff)
                print(result)
                
                validate(diff)
                
                model.incStep()
                numLoop += 1
            }
            
            model.incEpoch()
            epoch += 1
        }
    }
}
