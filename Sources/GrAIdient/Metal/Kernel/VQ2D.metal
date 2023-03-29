//
// VQ2D.metal
// GrAIdient
//
// Created by Jean-François Reboud on 29/03/2023.
//

#include <metal_stdlib>
using namespace metal;

kernel void vq2DForward(
    const device float * outsPrev,
    const device float * weights,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pK,
    constant uint * pNbBatch,
    constant uint * pComputeVQ,
    device float * outs,
    device int * indices,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint K;
    uint nbBatch;
    uint computeVQ;
    
    if (pNbChannels && pDimensions && pK && pNbBatch && pComputeVQ &&
        weights && outsPrev && outs && indices)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        K = *pK;
        nbBatch = *pNbBatch;
        computeVQ = *pComputeVQ;
    }
    else
        return ;
    
    uint elem = id[1];
    uint i = id[0] / width;
    uint j = id[0] % width;
    
    if (i * j >= height * width || elem >= nbBatch)
    {
        return ;
    }
    
    uint offsetStart = (0 + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    int minIndex = -1;
    float minValue = outsPrev[offset];
    for (uint k=0; k<K; k++)
    {
        float value = 0.0;
        for (uint depth=0; depth<nbChannels; depth++)
        {
            offsetStart = (depth + nbChannels * elem) * height;
            offset = j + (offsetStart + i) * width;
            
            uint offsetWeights = depth + nbChannels * k;
            
            float outPrev = outsPrev[offset];
            float vq = weights[offsetWeights];
            value += pow(outPrev - vq, 2.0);
        }
        value = sqrt(value);
        
        if (value < minValue)
        {
            minValue = value;
            minIndex = k;
        }
    }
    
    if (minIndex > 0)
    {
        for (uint depth=0; depth<nbChannels; depth++)
        {
            offsetStart = (depth + nbChannels * elem) * height;
            offset = j + (offsetStart + i) * width;
            
            uint offsetWeights = depth + nbChannels * minIndex;
            
            if (computeVQ)
            {
                outs[offset] = weights[offsetWeights];
            }
            else
            {
                outs[offset] = outsPrev[offset];
            }
        }
        indices[j + (elem * height + i) * width] = minIndex;
    }
}

kernel void vq2DBackward(
    const device float * outsPrev,
    const device float * delta,
    const device float * weights,
    const device int * indices,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pK,
    constant float * pBeta,
    constant uint * pNbBatch,
    constant uint * pCommit,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint K;
    float beta;
    uint nbBatch;
    uint commit;
    uint dirty;
    
    if (pNbChannels && pDimensions && pK && pBeta &&
        pNbBatch && pCommit && pDirty &&
        outsPrev && delta && weights && indices && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        K = *pK;
        beta = *pBeta;
        nbBatch = *pNbBatch;
        commit = *pCommit;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0] / width;
    uint elem = id[1] / height;
    uint i = id[1] % height;
    uint j = id[0] % width;
    uint coeff = nbBatch * height * width;
    
    if (i * elem >= height * nbBatch ||
        j * depth >= width * nbChannels)
    {
        return ;
    }
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    int minIndex = indices[j + (elem * height + i) * width];
    uint offsetWeights = depth + nbChannels * minIndex;
    
    float vq = weights[offsetWeights];
    float deltaCur = delta[offset];
    float outPrev = outsPrev[offset];
    
    if (dirty)
    {
        deltaPrev[offset] = deltaCur;
    }
    else
    {
        deltaPrev[offset] += deltaCur;
    }
    
    if (commit)
    {
        deltaPrev[offset] += beta / (float)coeff * (outPrev - vq);
    }
}

kernel void vq2DBatchDerWeights(
    const device float * outsPrev,
    const device float * weights,
    const device int * indices,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pK,
    constant uint * pNbBatch,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint K;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pK && pNbBatch &&
        outsPrev && weights && indices && grads)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        K = *pK;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint k = id[1];
    uint depth = id[0];
    uint coeff = nbBatch * height * width;
    
    if (depth >= nbChannels || k >= K)
    {
        return ;
    }
    
    float sum = 0.0;
    for (uint elem=0; elem<nbBatch; elem++){
    for (uint i=0; i<height; i++){
    for (uint j=0; j<width; j++)
    {
        int minIndex = indices[j + (elem * height + i) * width];
        if (minIndex == (int)k)
        {
            uint offsetStart = (depth + nbChannels * elem) * height;
            uint offset = j + (offsetStart + i) * width;
            
            int minIndex = indices[j + (elem * height + i) * width];
            uint offsetWeights = depth + nbChannels * minIndex;
            
            float vq = weights[offsetWeights];
            float outPrev = outsPrev[offset];
            
            sum += vq - outPrev;
        }
    }}}
    sum /= (float)coeff;
    
    grads[depth + nbChannels * k] += sum;
}

kernel void vq2DDerWeights(
    const device float * outsPrev,
    const device float * weights,
    const device int * indices,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pK,
    constant uint * pNbBatch,
    device float * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint K;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pK && pNbBatch &&
        outsPrev && weights && indices && deltaWeights)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        K = *pK;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint elem = id[1] / K;
    uint k = id[1] % K;
    uint depth = id[0];
    uint coeff = nbBatch * height * width;
    
    if (depth >= nbChannels || elem * k >= nbBatch * K)
    {
        return ;
    }
    
    float sum = 0.0;
    for (uint i=0; i<height; i++){
    for (uint j=0; j<width; j++)
    {
        int minIndex = indices[j + (elem * height + i) * width];
        if (minIndex == (int)k)
        {
            uint offsetStart = (depth + nbChannels * elem) * height;
            uint offset = j + (offsetStart + i) * width;
            
            int minIndex = indices[j + (elem * height + i) * width];
            uint offsetWeights = depth + nbChannels * minIndex;
            
            float vq = weights[offsetWeights];
            float outPrev = outsPrev[offset];
            
            sum += vq - outPrev;
        }
    }}
    sum /= (float)coeff;
    
    deltaWeights[depth + nbChannels * k + K * nbChannels * elem] += sum;
}

kernel void vq2DReduceWeights(
    const device float * deltaWeights,
    constant uint * pNbChannels,
    constant uint * pK,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint K;
    uint nbBatch;
    uint accumulate;
    
    if (pNbChannels && pK && pNbBatch && pAccumulate &&
        deltaWeights && grads)
    {
        nbChannels = *pNbChannels;
        K = *pK;
        nbBatch = *pNbBatch;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint k = id[1];
    uint depth = id[0];
    
    if (depth >= nbChannels || k >= K)
    {
        return ;
    }
    
    float sum = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offset = depth + nbChannels * k + K * nbChannels * elem;
        sum += deltaWeights[offset];
    }
    
    if (accumulate)
    {
        grads[depth + nbChannels * k] += sum;
    }
    else
    {
        grads[depth + nbChannels * k] = sum;
    }
}
