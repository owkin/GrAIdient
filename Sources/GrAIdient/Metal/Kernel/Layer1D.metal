//
// Layer1D.metal
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void MSE1DLoss(
    const device float * outs,
    const device float * groundTruth,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    device float * losses,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbNeurons && pNbBatch && outs && groundTruth && losses)
    {
        nbNeurons = *pNbNeurons;
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
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offset = depth + nbNeurons * elem;
    
        float gt = groundTruth[offset];
        float out = outs[offset];
        float diff = out - gt;
        
        tmp += diff * diff;
    }
    
    losses[elem] = tmp;
}

kernel void MSE1DLossDerivative(
    const device float * outs,
    const device float * groundTruth,
    constant uint * pNbNeurons,
    constant float * pCoeff,
    constant uint * pNbBatch,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    float coeff;
    uint nbBatch;
    
    if (pNbNeurons && pNbBatch && pCoeff && outs && groundTruth && deltaPrev)
    {
        nbNeurons = *pNbNeurons;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;

    float gt = groundTruth[offset];
    float out = outs[offset];
    float diff = out - gt;
    
    deltaPrev[offset] = 2 * coeff * diff / float(nbNeurons * nbBatch);
}

kernel void linearErrorLoss(
    const device float * outs,
    const device float * groundTruth,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    device float * losses,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbNeurons && pNbBatch && outs && groundTruth && losses)
    {
        nbNeurons = *pNbNeurons;
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
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offset = depth + nbNeurons * elem;
    
        float gt = groundTruth[offset];
        float out = outs[offset];
        float diff = out - gt;
        
        tmp += diff;
    }
    
    losses[elem] = tmp;
}

kernel void linearErrorLossDerivative(
    const device float * outs,
    constant uint * pNbNeurons,
    constant float * pCoeff,
    constant uint * pNbBatch,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    float coeff;
    uint nbBatch;
    
    if (pNbNeurons && pNbBatch && pCoeff && outs && deltaPrev)
    {
        nbNeurons = *pNbNeurons;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;
    deltaPrev[offset] = coeff / float(nbNeurons * nbBatch);
}

kernel void selectNeurons1DForward(
    const device float * outsPrev,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNeurons,
    constant float * pCoeffs,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    
    if (pNbNeurons && pNbNeuronsPrev && pNeurons && pCoeffs && pNbBatch &&
        outsPrev && outs)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;
    uint offsetPrev = pNeurons[depth] + nbNeuronsPrev * elem;
    outs[offset] = pCoeffs[depth] * outsPrev[offsetPrev];
}

kernel void selectNeurons1DBackward(
    const device float * delta,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNeurons,
    constant float * pCoeffs,
    constant uint * pNbBatch,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    
    if (pNbNeurons && pNbNeuronsPrev && pNeurons && pCoeffs && pNbBatch &&
        deltaPrev && delta)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;
    uint offsetPrev = pNeurons[depth] + nbNeuronsPrev * elem;
    deltaPrev[offsetPrev] += pCoeffs[depth] * delta[offset];
}
