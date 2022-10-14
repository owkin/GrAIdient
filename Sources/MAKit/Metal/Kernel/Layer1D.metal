//
// Layer1D.metal
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void MSE1DLoss(
    const device float * outs,
    const device float * groundTruth,
    constant uint * pNbNeurones,
    constant uint * pNbBatch,
    device float * losses,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbBatch;
    
    if (pNbNeurones && pNbBatch && outs && groundTruth && losses)
    {
        nbNeurones = *pNbNeurones;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint elem = id;
    if (elem >= nbBatch)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint depth=0; depth<nbNeurones; depth++)
    {
        uint offset = depth + nbNeurones * elem;
    
        float gt = groundTruth[offset];
        float out = outs[offset];
        float diff = out - gt;
        
        tmp += diff * diff;
    }
    
    losses[elem] = tmp;
}

kernel void MSE1DApplyGradient(
    const device float * outs,
    const device float * groundTruth,
    constant uint * pNbNeurones,
    constant float * pCoeff,
    constant uint * pNbBatch,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    float coeff;
    uint nbBatch;
    
    if (pNbNeurones && pNbBatch && pCoeff && outs && groundTruth && deltaPrev)
    {
        nbNeurones = *pNbNeurones;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurones || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurones * elem;

    float gt = groundTruth[offset];
    float out = outs[offset];
    float diff = out - gt;
    
    deltaPrev[offset] = 2 * coeff * diff / float(nbNeurones * nbBatch);
}

kernel void linearErrorLoss(
    const device float * outs,
    const device float * groundTruth,
    constant uint * pNbNeurones,
    constant uint * pNbBatch,
    device float * losses,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbBatch;
    
    if (pNbNeurones && pNbBatch && outs && groundTruth && losses)
    {
        nbNeurones = *pNbNeurones;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint elem = id;
    if (elem >= nbBatch)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint depth=0; depth<nbNeurones; depth++)
    {
        uint offset = depth + nbNeurones * elem;
    
        float gt = groundTruth[offset];
        float out = outs[offset];
        float diff = out - gt;
        
        tmp += diff;
    }
    
    losses[elem] = tmp;
}

kernel void linearErrorApplyGradient(
    const device float * outs,
    constant uint * pNbNeurones,
    constant float * pCoeff,
    constant uint * pNbBatch,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    float coeff;
    uint nbBatch;
    
    if (pNbNeurones && pNbBatch && pCoeff && outs && deltaPrev)
    {
        nbNeurones = *pNbNeurones;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurones || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurones * elem;
    deltaPrev[offset] = coeff / float(nbNeurones * nbBatch);
}

kernel void selectChForward(
    const device float * outsPrev,
    constant uint * pNbNeurones,
    constant uint * pNbNeuronesPrev,
    constant uint * pChannels,
    constant float * pCoeffs,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbNeuronesPrev;
    uint nbBatch;
    
    if (pNbNeurones && pNbNeuronesPrev && pChannels && pCoeffs && pNbBatch &&
        outsPrev && outs)
    {
        nbNeurones = *pNbNeurones;
        nbNeuronesPrev = *pNbNeuronesPrev;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurones || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurones * elem;
    uint offsetPrev = pChannels[depth] + nbNeuronesPrev * elem;
    outs[offset] = pCoeffs[depth] * outsPrev[offsetPrev];
}

kernel void selectChBackward(
    const device float * delta,
    constant uint * pNbNeurones,
    constant uint * pNbNeuronesPrev,
    constant uint * pChannels,
    constant float * pCoeffs,
    constant uint * pNbBatch,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbNeuronesPrev;
    uint nbBatch;
    
    if (pNbNeurones && pNbNeuronesPrev && pChannels && pCoeffs && pNbBatch &&
        deltaPrev && delta)
    {
        nbNeurones = *pNbNeurones;
        nbNeuronesPrev = *pNbNeuronesPrev;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurones || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurones * elem;
    uint offsetPrev = pChannels[depth] + nbNeuronesPrev * elem;
    deltaPrev[offsetPrev] += pCoeffs[depth] * delta[offset];
}
