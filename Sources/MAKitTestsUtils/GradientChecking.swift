//
// GradientChecking.swift
// MAKitTestsUtils
//
// Created by Jean-François Reboud on 10/10/2022.
//

import Foundation
import MAKit

///
/// Compare the gradients of weights with an estimation made through gradient checking.
///
/// - Parameters:
///     - model: The model.
///     - layersGraph: The list of layers (a "graph") to go through.
///     - gradientsApprox: The list of gradients of weights approximations.
/// - Returns: The relative difference between the two.
///
public func checkGradients(
    model: Model,
    layersGraph: [Layer],
    gradientsApprox: [Double]) -> Double?
{
    let gradients = try! model.collectGradients(layers: layersGraph)
    assert(gradientsApprox.count == gradients.count)
        
    var isNull = true
    for (gradientApprox, gradient) in zip(gradientsApprox, gradients)
    {
        if gradientApprox != 0.0 || gradient != 0.0
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
    var gradApprox: Double = 0.0
    
    for (gradientApprox, gradient) in zip(gradientsApprox, gradients)
    {
        var tmp = gradient - gradientApprox
        tmp *= tmp
        diff += tmp
        
        tmp = gradient
        tmp *= tmp
        grad += tmp
        
        tmp = gradientApprox
        tmp *= tmp
        gradApprox += tmp
    }
    diff = sqrt(diff)
    grad = sqrt(grad)
    gradApprox = sqrt(gradApprox)
    
    let result = diff / (grad + gradApprox)
    
    var warning = ""
    if result > 0.0000001
    {
        warning = "Gradient Check Warning : "
    }
    let content = warning + String(result)
    print(content)
        
    return result
}
