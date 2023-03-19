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
    
    float cMax = outsPrev[0+block*size + nbNeurons * elem];
    for (uint j1=0; j1<size; j1++)
    {
        uint offset1 = j1+block*size + nbNeurons * elem;
        float outPrev = outsPrev[offset1];
        
        if (outPrev > cMax)
        {
            cMax = outPrev;
        }
    }
    
    float sum1 = 0.0;
    for (uint j1=0; j1<size; j1++)
    {
        uint offset1 = j1+block*size + nbNeurons * elem;
        float outPrev = outsPrev[offset1];
        sum1 += exp(outPrev - cMax);
    }
    
    uint offset = depth + nbNeurons * elem;
    float outPrev = outsPrev[offset];
    outs[offset] = exp(outPrev - cMax) / sum1;
}

kernel void softmax1DBackward(
    const device float * outs,
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
        deltaPrev && outs && delta)
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
    
    uint offset = depth + nbNeurons * elem;
    float outCur = outs[offset];
    float deltaCur = delta[offset];
    
    float sum1 = 0.0;
    for (uint j1=0; j1<size; j1++)
    {
        uint offset1 = j1+block*size + nbNeurons * elem;
        float outCur1 = outs[offset1];
        float deltaCur1 = delta[offset1];
        sum1 += outCur1 * deltaCur1;
    }
    
    if (dirty)
    {
        deltaPrev[offset] = outCur * (deltaCur - sum1);
    }
    else
    {
        deltaPrev[offset] += outCur * (deltaCur - sum1);
    }
}

kernel void dotProduct1DForward(
    const device float * outsPrev1,
    const device float * outsPrev2,
    constant int * pSize,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    
    if (pSize && pNbNeurons && pNbNeuronsPrev && pNbBatch &&
        outsPrev1 && outsPrev2 && outs)
    {
        size = *pSize;
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
    
    float sum = 0.0;
    for (uint j=0; j<size; j++)
    {
        uint offset = j+depth*size + nbNeuronsPrev * elem;
        float outPrev1 = outsPrev1[offset];
        float outPrev2 = outsPrev2[offset];
        sum += outPrev1 * outPrev2;
    }
    
    uint offset = depth + nbNeurons * elem;
    outs[offset] = sum;
}

kernel void dotProduct1DBackward(
    const device float * outsPrev,
    const device float * delta,
    constant int * pSize,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint dirty;
    
    if (pSize && pNbNeurons && pNbNeuronsPrev && pNbBatch && pDirty &&
        outsPrev && deltaPrev && delta)
    {
        size = *pSize;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    for (uint j=0; j<size; j++)
    {
        uint offsetPrev = j+depth*size + nbNeuronsPrev * elem;
        uint offset = depth + nbNeurons * elem;
        
        float outPrev = outsPrev[offsetPrev];
        float deltaCur = delta[offset];
        if (dirty)
        {
            deltaPrev[offsetPrev] = outPrev * deltaCur;
        }
        else
        {
            deltaPrev[offsetPrev] += outPrev * deltaCur;
        }
    }
}

kernel void constant1DForward(
    const device float * weights,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbNeurons && pNbBatch && weights && outs)
    {
        nbNeurons = *pNbNeurons;
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
    outs[offset] = weights[depth];
}
