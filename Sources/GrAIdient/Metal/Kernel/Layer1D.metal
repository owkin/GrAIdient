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

kernel void concat1DForward(
    const device float * outsPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint globalOffset;
    
    if (pGlobalOffset && pNbNeurons && pNbNeuronsPrev && pNbBatch &&
        outsPrev && outs)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        globalOffset = *pGlobalOffset;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeuronsPrev || elem >= nbBatch)
    {
        return ;
    }
    
    uint offsetPrev = depth + nbNeuronsPrev * elem;
    uint offset = globalOffset+depth + nbNeurons * elem;
    
    outs[offset] = outsPrev[offsetPrev];
}

kernel void concat1DBackward(
    const device float * delta,
    constant uint * pGlobalOffset,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint globalOffset;
    uint dirty;
    
    if (pGlobalOffset && pNbNeurons && pNbNeuronsPrev && pNbBatch && pDirty &&
        deltaPrev && delta)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        globalOffset = *pGlobalOffset;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeuronsPrev || elem >= nbBatch)
    {
        return ;
    }
    
    uint offsetPrev = depth + nbNeuronsPrev * elem;
    uint offset = globalOffset+depth + nbNeurons * elem;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = delta[offset];
    }
    else
    {
        deltaPrev[offsetPrev] += delta[offset];
    }
}

kernel void softmax1DForward(
    const device float * outsPrev,
    constant uint * pSize,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size;
    uint nbNeurons;
    uint nbBatch;
    
    if (pSize && pNbNeurons && pNbBatch && outsPrev && outs)
    {
        size = *pSize;
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    uint block = depth / size;
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    float sum1 = 0.0;
    for (uint j1=0; j1<size; j1++)
    {
        uint offset1 = j1+block*size + nbNeurons * elem;
        float outPrev = outsPrev[offset1];
        sum1 += exp(outPrev);
    }
    
    uint offset = depth + nbNeurons * elem;
    float outPrev = outsPrev[offset];
    outs[offset] = exp(outPrev) / sum1;
}

kernel void softmax1DBackward(
    const device float * outsPrev,
    const device float * delta,
    constant uint * pSize,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size;
    uint nbNeurons;
    uint nbBatch;
    uint dirty;
    
    if (pSize && pNbNeurons && pNbBatch && pDirty &&
        outsPrev && deltaPrev && delta)
    {
        size = *pSize;
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    uint block = depth / size;
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    float sum1 = 0.0;
    for (uint j1=0; j1<size; j1++)
    {
        uint offset1 = j1+block*size + nbNeurons * elem;
        float outPrev1 = outsPrev[offset1];
        sum1 += exp(outPrev1);
    }
    
    uint offset = depth + nbNeurons * elem;
    float outPrev = outsPrev[offset];
    float deltaCur = delta[offset];
    
    float sum2 = 0.0;
    for (uint j2=0; j2<size; j2++)
    {
        uint offset2 = j2+block*size + nbNeurons * elem;
        float outPrev2 = outsPrev[offset2];
        float deltaCur2 = delta[offset2];
        sum2 += exp(outPrev + outPrev2) * deltaCur2;
    }
    
    if (dirty)
    {
        deltaPrev[offset] = -sum2 / (sum1 * sum1) +
            exp(outPrev) * deltaCur / sum1;
    }
    else
    {
        deltaPrev[offset] += -sum2 / (sum1 * sum1) +
            exp(outPrev) * deltaCur / sum1;
    }
}
