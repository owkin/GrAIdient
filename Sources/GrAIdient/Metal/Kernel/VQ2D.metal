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
    device float * outs,
    device int * indices,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint K;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pK && pNbBatch &&
        weights && outsPrev && outs && indices)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        K = *pK;
        nbBatch = *pNbBatch;
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
    
    int minIndex = -1;
    float minValue = 0.0;
    for (uint k=0; k<K; k++)
    {
        float value = 0.0;
        for (uint depth=0; depth<nbChannels; depth++)
        {
            uint offsetStart = (depth + nbChannels * elem) * height;
            uint offset = j + (offsetStart + i) * width;
            
            uint offsetWeights = depth + nbChannels * k;
            
            float outPrev = outsPrev[offset];
            float vq = weights[offsetWeights];
            value += pow(outPrev - vq, 2.0);
        }
        
        if (minIndex < 0 || value < minValue)
        {
            minValue = value;
            minIndex = k;
        }
    }
    
    if (minIndex >= 0)
    {
        for (uint depth=0; depth<nbChannels; depth++)
        {
            uint offsetStart = (depth + nbChannels * elem) * height;
            uint offset = j + (offsetStart + i) * width;
            
            uint offsetWeights = depth + nbChannels * minIndex;
            outs[offset] = weights[offsetWeights];
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
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint K;
    float beta;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensions && pK && pBeta && pNbBatch && pDirty &&
        outsPrev && delta && weights && indices && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        K = *pK;
        beta = *pBeta;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0] / width;
    uint elem = id[1] / height;
    uint i = id[1] % height;
    uint j = id[0] % width;
    
    if (i * elem >= height * nbBatch ||
        j * depth >= width * nbChannels)
    {
        return ;
    }
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    int minIndex = indices[j + (elem * height + i) * width];
    if (minIndex >= 0)
    {
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
        
        // Commitment term.
        deltaPrev[offset] += beta * 2.0 * (outPrev - vq);
    }
    else if (dirty)
    {
        deltaPrev[offset] = 0.0;
    }
}

kernel void vq2DBatchDerWeights(
    const device float * outsPrev,
    const device float * weights,
    const device int * indices,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pK,
    constant float * pCoeff,
    constant uint * pNbBatch,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint K;
    float coeff;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pK && pCoeff && pNbBatch &&
        outsPrev && weights && indices && grads)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        K = *pK;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
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
    for (uint elem=0; elem<nbBatch; elem++){
    for (uint i=0; i<height; i++){
    for (uint j=0; j<width; j++)
    {
        int minIndex = indices[j + (elem * height + i) * width];
        if (minIndex == (int)k)
        {
            uint offsetStart = (depth + nbChannels * elem) * height;
            uint offset = j + (offsetStart + i) * width;
            
            uint offsetWeights = depth + nbChannels * minIndex;
            
            float vq = weights[offsetWeights];
            float outPrev = outsPrev[offset];
            
            sum += vq - outPrev;
        }
    }}}
    sum *= coeff / (float)(nbBatch * nbChannels * height * width) * 2.0;
    
    grads[depth + nbChannels * k] += sum;
}

kernel void vq2DDerWeights(
    const device float * outsPrev,
    const device float * weights,
    const device int * indices,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pK,
    constant float * pCoeff,
    constant uint * pNbBatch,
    device float * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint K;
    float coeff;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pK && pCoeff && pNbBatch &&
        outsPrev && weights && indices && deltaWeights)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        K = *pK;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint elem = id[1] / K;
    uint k = id[1] % K;
    uint depth = id[0];
    
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
            
            uint offsetWeights = depth + nbChannels * minIndex;
            
            float vq = weights[offsetWeights];
            float outPrev = outsPrev[offset];
            
            sum += vq - outPrev;
        }
    }}
    sum *= coeff / (float)(nbBatch * nbChannels * height * width) * 2.0;
    
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

kernel void vq2DLoss(
    const device float * outsPrev,
    const device float * outs,
    const device int * indices,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device float * losses,
    uint id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch &&
        outsPrev && outs && indices && losses)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
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
    for (uint depth=0; depth<nbChannels; depth++)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        
        for (uint i=0; i<height; i++) {
        for (uint j=0; j<width; j++)
        {
            int minIndex = indices[j + (elem * height + i) * width];
            if (minIndex >= 0)
            {
                uint offset = j + (offsetStart + i) * width;
                
                float outPrev = outsPrev[offset];
                float vq = outs[offset];
                float diff = outPrev - vq;
                
                tmp += diff * diff;
            }
        }}
    }
    losses[elem] = tmp;
}

kernel void vqGradNorm2DMax(
     const device float * deltaPrev,
     constant uint * pNbChannels,
     constant uint * pDimensions,
     constant uint * pNbThreadgroups,
     constant uint * pNbBatch,
     device float * gradNorms,
     uint2 groupId [[ threadgroup_position_in_grid ]],
     uint2 threadId [[ thread_position_in_threadgroup ]],
     uint2 id [[ thread_position_in_grid ]])
{
    constexpr uint threadsPerThreadgroup = 64;
    threadgroup float normShared[threadsPerThreadgroup];
    
    uint height, width;
    uint nbChannels;
    uint nbThreadgroups;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbThreadgroups && pNbBatch &&
        deltaPrev && gradNorms)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        nbThreadgroups = *pNbThreadgroups;
        nbBatch = *pNbBatch;
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
    
    float norm = 0.0;
    for (uint depth=0; depth<nbChannels; depth++)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        uint offset = j + (offsetStart + i) * width;
        
        norm += pow(deltaPrev[offset], 2.0);
    }
    norm = sqrt(norm);
    
    normShared[threadId[0]] = norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride=threadsPerThreadgroup/2; stride>0; stride>>=1)
    {
        uint index = threadId[0] + groupId[0] * threadsPerThreadgroup;
        if (threadId[0] < stride &&
            (index + stride) < height * width)
        {
            normShared[threadId[0]] += normShared[threadId[0] + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (threadId[0] == 0)
    {
        uint offset = elem * nbThreadgroups + groupId[0];
        gradNorms[offset] = normShared[0];
    }
}

kernel void vqGradNorm2DForward(
    const device float * outsPrev,
    const device float * gradNorms,
    const device float * weights,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pK,
    constant float * pMagnitudeCoeff,
    constant uint * pNbBatch,
    device float * outs,
    device int * indices,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint K;
    float magnitudeCoeff;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pK && pMagnitudeCoeff && pNbBatch &&
        weights && gradNorms && outsPrev && outs && indices)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        magnitudeCoeff = *pMagnitudeCoeff;
        nbChannels = *pNbChannels;
        K = *pK;
        nbBatch = *pNbBatch;
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
    
    float norm = 0.0;
    for (uint depth=0; depth<nbChannels; depth++)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        uint offset = j + (offsetStart + i) * width;
        
        norm += pow(deltaPrev[offset], 2.0);
    }
    norm = sqrt(norm);
    
    if (norm >= gradNorms[elem] / magnitudeCoeff)
    {
        int minIndex = -1;
        float minValue = 0.0;
        for (uint k=0; k<K; k++)
        {
            float value = 0.0;
            for (uint depth=0; depth<nbChannels; depth++)
            {
                uint offsetStart = (depth + nbChannels * elem) * height;
                uint offset = j + (offsetStart + i) * width;
                
                uint offsetWeights = depth + nbChannels * k;
                
                float outPrev = outsPrev[offset];
                float vq = weights[offsetWeights];
                value += pow(outPrev - vq, 2.0);
            }
            
            if (minIndex < 0 || value < minValue)
            {
                minValue = value;
                minIndex = k;
            }
        }
        
        if (minIndex >= 0)
        {
            for (uint depth=0; depth<nbChannels; depth++)
            {
                uint offsetStart = (depth + nbChannels * elem) * height;
                uint offset = j + (offsetStart + i) * width;
                
                uint offsetWeights = depth + nbChannels * minIndex;
                outs[offset] = weights[offsetWeights];
            }
            indices[j + (elem * height + i) * width] = minIndex;
        }
    }
    else
    {
        indices[j + (elem * height + i) * width] = -1;
    }
}
