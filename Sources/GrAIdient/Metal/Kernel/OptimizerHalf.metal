//
// Optimizer.metal
// GrAIdient
//
// Created by Jean-François Reboud on 09/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void clipGradientsHalf(
    constant uint * pNbElems,
    constant float * pGradientNorm,
    constant float * pNormThreshold,
    device half * grads,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    half gradientNorm;
    half normThreshold;
    
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

kernel void multiplyGradientsHalf(
    constant uint * pNbElems,
    constant float * pFactor,
    device half * grads,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    half factor;
    
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

kernel void weightsSGDHalf(
    const device half * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    device half * weights,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    half alpha, lambda;
    
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
    
    half g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    weights[id] = weights[id] - alpha * g;
}

kernel void weightsMomentumHalf(
    const device half * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    device half * weights,
    device half * mPtr,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    half alpha, lambda;
    half β1 = 0.9;
    
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
    
    half g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    
    half v = β1 * mPtr[id] + alpha * g;
    mPtr[id] = v;
    
    weights[id] = weights[id] - v;
}

kernel void weightsAdamHalf(
    const device half * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    constant float * pT,
    device half * weights,
    device half * mPtr,
    device half * vPtr,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    half alpha, lambda;
    half t;
    half β1 = 0.9;
    half β2 = 0.999;
    half Ɛ = 0.0001;
    
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
    
    half g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    
    half m = β1 * mPtr[id] + (1 - β1) * g;
    half v = β2 * vPtr[id] + (1 - β2) * g * g;
    mPtr[id] = m;
    vPtr[id] = v;
    
    m /= (1 - pow(β1, t));
    v /= (1 - pow(β2, t));
    
    weights[id] = weights[id] - alpha * m / (sqrt(v) + Ɛ);
}

kernel void weightsAMSGradHalf(
    const device half * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    constant float * pT,
    device half * weights,
    device half * mPtr,
    device half * vPtr,
    device half * vHatPtr,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    half alpha, lambda;
    half t;
    half β1 = 0.9;
    half β2 = 0.999;
    half Ɛ = 0.0001;
    
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
    
    half g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    
    half m = β1 * mPtr[id] + (1 - β1) * g;
    half v = β2 * vPtr[id] + (1 - β2) * g * g;
    half vHat = max(v, vHatPtr[id]);
    
    mPtr[id] = m;
    vPtr[id] = v;
    vHatPtr[id] = vHat;
    
    m /= (1 - pow(β1, t));
    vHat /= (1 - pow(β2, t));
    
    weights[id] = weights[id] - alpha * m / (sqrt(vHat) + Ɛ);
}

kernel void weightsAdamRectifiedHalf(
    const device half * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    constant float * pT,
    device half * weights,
    device half * mPtr,
    device half * vPtr,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    half alpha, lambda;
    half t;
    half β1 = 0.9;
    half β2 = 0.999;
    half Ɛ = 0.0001;
    half ρinf = 2.0 / (1.0 - β2) - 1.0;
    
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
    
    half g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    
    half m = β1 * mPtr[id] + (1 - β1) * g;
    half v = β2 * vPtr[id] + (1 - β2) * g * g;
    mPtr[id] = m;
    vPtr[id] = v;
    
    m /= (1 - pow(β1, t));
    half ρ = ρinf - 2.0 * t * pow(β2, t) / (1 - pow(β2, t));
    
    if (ρ > 5.0)
    {
        half l = sqrt((1 - pow(β2, t)) / (v + Ɛ));
        half r = sqrt(((ρ - 4.0) * (ρ - 2.0) * ρinf) /
                       ((ρinf - 4.0) * (ρinf - 2.0) * ρ));
        
        weights[id] = weights[id] - alpha * m * r * l;
    }
    else
    {
        weights[id] = weights[id] - alpha * m;
    }
}

kernel void weightsAdaBoundHalf(
    const device half * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    constant float * pT,
    constant float * pLowerBound,
    constant float * pUpperBound,
    device half * weights,
    device half * mPtr,
    device half * vPtr,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    half alpha, lambda;
    half t;
    half β1 = 0.9;
    half β2 = 0.999;
    half Ɛ = 0.0001;
    half lowerBound;
    half upperBound;
    
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
    
    half g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    
    half m = β1 * mPtr[id] + (1 - β1) * g;
    half v = β2 * vPtr[id] + (1 - β2) * g * g;
    
    mPtr[id] = m;
    vPtr[id] = v;
    
    half alphaHat = alpha *
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

kernel void weightsAMSBoundHalf(
    const device half * grads,
    constant uint * pNbElems,
    constant float * pAlpha,
    constant float * pLambda,
    constant float * pT,
    constant float * pLowerBound,
    constant float * pUpperBound,
    device half * weights,
    device half * mPtr,
    device half * vPtr,
    device half * vHatPtr,
    uint id [[ thread_position_in_grid ]])
{
    uint nbElems;
    half alpha, lambda;
    half t;
    half β1 = 0.9;
    half β2 = 0.999;
    half Ɛ = 0.0001;
    half lowerBound;
    half upperBound;
    
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
    
    half g = grads[id];
    if (lambda != 0)
    {
        g += lambda * weights[id];
    }
    
    half m = β1 * mPtr[id] + (1 - β1) * g;
    half v = β2 * vPtr[id] + (1 - β2) * g * g;
    half vHat = max(v, vHatPtr[id]);
    
    mPtr[id] = m;
    vPtr[id] = v;
    vHatPtr[id] = vHat;
    
    half alphaHat = alpha *
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
