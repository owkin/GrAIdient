//
// FlowChecking.swift
// MAKitTestsUtils
//
// Created by Jean-François Reboud on 10/10/2022.
//

import Foundation
import MAKit

public func checkFlow(_ ref: [Double], _ target: [Double]) -> Double?
{
    assert(ref.count == target.count)
    
    var isNull = true
    for (gradRef, grad) in zip(ref, target)
    {
        if grad != 0.0 || gradRef != 0.0
        {
            isNull = false
            break
        }
    }
    if isNull
    {
        print("Gradient is 0.0 !")
        return nil
    }
    
    var diff: Double = 0.0
    var grad: Double = 0.0
    var gradRef: Double = 0.0
    
    for (gradientRef, gradient) in zip(ref, target)
    {
        var tmp = gradient - gradientRef
        tmp *= tmp
        diff += tmp
        
        tmp = gradient
        tmp *= tmp
        grad += tmp
        
        tmp = gradientRef
        tmp *= tmp
        gradRef += tmp
    }
    
    diff = sqrt(diff)
    grad = sqrt(grad)
    gradRef = sqrt(gradRef)
    
    let result = diff / (grad + gradRef)
    
    var warning = ""
    if result > 0.0000001
    {
        warning = "GPUFlow Check Warning : "
    }
    let content = warning + String(result)
    print(content)
    
    return result
}

public func getGradients(model: Model) -> [Double]
{
    return try! model.collectGradients(layers: model.layers)
}
