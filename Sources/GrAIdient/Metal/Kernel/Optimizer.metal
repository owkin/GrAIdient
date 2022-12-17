//
// Optimizer.metal
// GrAIdient
//
// Created by Jean-François Reboud on 09/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void clipGradients(
    constant uint * pNbElems,
    constant float * pGradientNorm,
    constant float * pNormThreshold,
    device float * grads,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    float gradientNorm;
    float normThreshold;
    
    if (pNbElems && pGradientNorm && pNormThreshold && grads)
    {
        nbElems = *pNbElems;
        gradientNorm = *pGradientNorm;
        normThreshold = *pNormThreshold;
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    grads[id] = grads[id] * normThreshold / gradientNorm;
}

kernel void multiplyGradients(
    constant uint * pNbElems,
    constant float * pFactor,
    device float * grads,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    float factor;
    
    if (pNbElems && pFactor && grads)
    {
        nbElems = *pNbElems;
        factor = *pFactor;
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    grads[id] = grads[id] * factor;
}

kernel void weightsSGD(
    const device float * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    device float * weights,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    float alpha, lambda;
    
    if (pNbElems && pAlpha && pLambda && grads && weights)
    {
        nbElems = pNbElems[0];
        alpha = *pAlpha;
        lambda = *pLambda;
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    weights[id] = weights[id] - alpha * g;
}

kernel void weightsMomentum(
    const device float * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    device float * weights,
    device float * mPtr,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    float alpha, lambda;
    float β1 = 0.9;
    
    if (pNbElems && pAlpha && pLambda && grads && weights && mPtr)
    {
        nbElems = pNbElems[0];
        alpha = *pAlpha;
        lambda = *pLambda;
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    
    float v = β1 * mPtr[id] + alpha * g;
    mPtr[id] = v;
    
    weights[id] = weights[id] - v;
}

kernel void weightsAdam(
    const device float * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    constant float * pT,
    device float * weights,
    device float * mPtr,
    device float * vPtr,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    float alpha, lambda;
    float t;
    float β1 = 0.9;
    float β2 = 0.999;
    float Ɛ = 0.00000001;
    
    if (pNbElems && pAlpha && pLambda && pT &&
        grads && weights && mPtr && vPtr)
    {
        nbElems = pNbElems[0];
        alpha = *pAlpha;
        lambda = *pLambda;
        t = *pT;
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    
    float m = β1 * mPtr[id] + (1 - β1) * g;
    float v = β2 * vPtr[id] + (1 - β2) * g * g;
    mPtr[id] = m;
    vPtr[id] = v;
    
    m /= (1 - pow(β1, t));
    v /= (1 - pow(β2, t));
    
    weights[id] = weights[id] - alpha * m / (sqrt(v) + Ɛ);
}

kernel void weightsAMSGrad(
    const device float * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    constant float * pT,
    device float * weights,
    device float * mPtr,
    device float * vPtr,
    device float * vHatPtr,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    float alpha, lambda;
    float t;
    float β1 = 0.9;
    float β2 = 0.999;
    float Ɛ = 0.00000001;
    
    if (pNbElems && pAlpha && pLambda && pT &&
        grads && weights && mPtr && vPtr && vHatPtr)
    {
        nbElems = pNbElems[0];
        alpha = *pAlpha;
        lambda = *pLambda;
        t = *pT;
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    
    float m = β1 * mPtr[id] + (1 - β1) * g;
    float v = β2 * vPtr[id] + (1 - β2) * g * g;
    float vHat = max(v, vHatPtr[id]);
    
    mPtr[id] = m;
    vPtr[id] = v;
    vHatPtr[id] = vHat;
    
    m /= (1 - pow(β1, t));
    vHat /= (1 - pow(β2, t));
    
    weights[id] = weights[id] - alpha * m / (sqrt(vHat) + Ɛ);
}

kernel void weightsAdamRectified(
    const device float * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    constant float * pT,
    device float * weights,
    device float * mPtr,
    device float * vPtr,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    float alpha, lambda;
    float t;
    float β1 = 0.9;
    float β2 = 0.999;
    float Ɛ = 0.00000001;
    float ρinf = 2.0 / (1.0 - β2) - 1.0;
    
    if (pNbElems && pAlpha && pLambda && pT &&
        grads && weights && mPtr && vPtr)
    {
        nbElems = pNbElems[0];
        alpha = *pAlpha;
        lambda = *pLambda;
        t = *pT;
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    
    float m = β1 * mPtr[id] + (1 - β1) * g;
    float v = β2 * vPtr[id] + (1 - β2) * g * g;
    mPtr[id] = m;
    vPtr[id] = v;
    
    m /= (1 - pow(β1, t));
    float ρ = ρinf - 2.0 * t * pow(β2, t) / (1 - pow(β2, t));
    
    if (ρ > 5.0)
    {
        float l = sqrt((1 - pow(β2, t)) / (v + Ɛ));
        float r = sqrt(((ρ - 4.0) * (ρ - 2.0) * ρinf) /
                       ((ρinf - 4.0) * (ρinf - 2.0) * ρ));
        
        weights[id] = weights[id] - alpha * m * r * l;
    }
    else
    {
        weights[id] = weights[id] - alpha * m;
    }
}

kernel void weightsAdaBound(
    const device float * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    constant float * pT,
    constant float * pLowerBound,
    constant float * pUpperBound,
    device float * weights,
    device float * mPtr,
    device float * vPtr,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    float alpha, lambda;
    float t;
    float β1 = 0.9;
    float β2 = 0.999;
    float Ɛ = 0.00000001;
    float lowerBound;
    float upperBound;
    
    if (pNbElems && pAlpha && pLambda && pT && pLowerBound && pUpperBound &&
        grads && weights && mPtr && vPtr)
    {
        nbElems = pNbElems[0];
        alpha = *pAlpha;
        lambda = *pLambda;
        t = *pT;
        lowerBound = *pLowerBound;
        upperBound = *pUpperBound;
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    
    float m = β1 * mPtr[id] + (1 - β1) * g;
    float v = β2 * vPtr[id] + (1 - β2) * g * g;
    
    mPtr[id] = m;
    vPtr[id] = v;
    
    float alphaHat = alpha *
        sqrt(1 - pow(β2, t)) / ((sqrt(v) + Ɛ) * (1 - pow(β1, t)));
    if (alphaHat < lowerBound)
    {
        alphaHat = lowerBound;
    }
    else if (alphaHat > upperBound)
    {
        alphaHat = upperBound;
    }
    
    weights[id] = weights[id] - alphaHat * m;
}

kernel void weightsAMSBound(
    const device float * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    constant float * pT,
    constant float * pLowerBound,
    constant float * pUpperBound,
    device float * weights,
    device float * mPtr,
    device float * vPtr,
    device float * vHatPtr,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    float alpha, lambda;
    float t;
    float β1 = 0.9;
    float β2 = 0.999;
    float Ɛ = 0.00000001;
    float lowerBound;
    float upperBound;
    
    if (pNbElems && pAlpha && pLambda && pT && pLowerBound && pUpperBound &&
        grads && weights && mPtr && vPtr && vHatPtr)
    {
        nbElems = pNbElems[0];
        alpha = *pAlpha;
        lambda = *pLambda;
        t = *pT;
        lowerBound = *pLowerBound;
        upperBound = *pUpperBound;
    }
    else
        return ;
    
    if (id >= nbElems)
    {
        return ;
    }
    
    float g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    
    float m = β1 * mPtr[id] + (1 - β1) * g;
    float v = β2 * vPtr[id] + (1 - β2) * g * g;
    float vHat = max(v, vHatPtr[id]);
    
    mPtr[id] = m;
    vPtr[id] = v;
    vHatPtr[id] = vHat;
    
    float alphaHat = alpha *
        sqrt(1 - pow(β2, t)) / ((sqrt(vHat) + Ɛ) * (1 - pow(β1, t)));
    if (alphaHat < lowerBound)
    {
        alphaHat = lowerBound;
    }
    else if (alphaHat > upperBound)
    {
        alphaHat = upperBound;
    }
    
    weights[id] = weights[id] - alphaHat * m;
}
