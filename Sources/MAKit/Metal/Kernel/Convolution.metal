//
// Convolution.metal
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void convForward(
    const device float * outsPrev,
    const device float * weights,
    const device float * biases,
    constant int * pStart,
    constant uint * pStride,
    constant uint * pNbChannels,
    constant uint * pNbChannelsPrev,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pDimWeights,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint weightHeight, weightWidth;
    uint nbChannels;
    uint nbChannelsPrev;
    int startI, startJ;
    int endI, endJ;
    uint stride;
    uint nbBatch;
    
    if (pStart && pStride && pNbChannels && pNbChannelsPrev &&
        pDimensions && pDimensionsPrev && pDimWeights && pNbBatch &&
        outsPrev && weights && biases && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        weightWidth = pDimWeights[0];
        weightHeight = pDimWeights[1];
        nbChannels = *pNbChannels;
        nbChannelsPrev = *pNbChannelsPrev;
        nbBatch = *pNbBatch;
        startI = pStart[0];
        endI = pStart[1];
        startJ = pStart[2];
        endJ = pStart[3];
        stride = pStride[0];
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    //uint depth = id % nbChannels;
    //uint elem = id / nbChannels;
    
    if (i >= height || j >= width) // || id[2] >= nbChannels * nbBatch)
    {
        return ;
    }
    
    for (uint elem=0; elem<nbBatch; elem++) {
    for (uint depth=0; depth<nbChannels; depth++)
    {
        uint offsetStart = (depth+nbChannels*elem)*height;
        
        float tmp = biases[depth];
        for (uint depthPrev=0; depthPrev<nbChannelsPrev; depthPrev++)
        {
            uint offsetStartPrev =
            (depthPrev + nbChannelsPrev*elem) * heightPrev;
            uint offsetStartWeights =
            (depthPrev + nbChannelsPrev * depth) * weightHeight;
            
            for (int k=startI; k<=endI; k++){
            for (int l=startJ; l<=endJ; l++)
            {
                if ((int)(stride*j)+l >= 0 && stride*j+l < widthPrev
                    && (int)(stride*i)+k >= 0 && stride*i+k < heightPrev)
                {
                    uint offsetPrev = stride*j+l +
                    (offsetStartPrev + stride*i+k)*widthPrev;
                    float outPrev = outsPrev[offsetPrev];
                    
                    uint offsetWeights = l-startJ +
                    (offsetStartWeights + k-startI) * weightWidth;
                    float w = weights[offsetWeights];
                    
                    tmp += outPrev * w;
                }
            }}
        }
        
        uint offset = j + (offsetStart + i)*width;
        outs[offset] = tmp;
    }}
}

kernel void convBackward(
    const device float * delta,
    const device float * weights,
    constant int * pStart,
    constant uint * pStride,
    constant uint * pNbChannels,
    constant uint * pNbChannelsPrev,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pDimWeights,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint3 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint weightHeight, weightWidth;
    uint nbChannels;
    uint nbChannelsPrev;
    int startI, startJ;
    int endI, endJ;
    uint stride;
    uint nbBatch;
    uint dirty;
    
    if (pStart && pStride && pNbChannels && pNbChannelsPrev &&
        pDimensions && pDimensionsPrev && pDimWeights && pNbBatch && pDirty &&
        delta && weights && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        weightWidth = pDimWeights[0];
        weightHeight = pDimWeights[1];
        nbChannels = *pNbChannels;
        nbChannelsPrev = *pNbChannelsPrev;
        nbBatch = *pNbBatch;
        startI = pStart[0];
        endI = pStart[1];
        startJ = pStart[2];
        endJ = pStart[3];
        stride = pStride[0];
        dirty = *pDirty;
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint depthPrev = id[2] % nbChannelsPrev;
    uint elem = id[2] / nbChannelsPrev;
    
    if (i >= heightPrev || j >= widthPrev ||
        id[2] >= nbChannelsPrev * nbBatch)
    {
        return ;
    }
    
    uint offsetStartPrev = (depthPrev + nbChannelsPrev * elem) * heightPrev;
    
    float tmp = 0.0;
    for (uint depth=0; depth<nbChannels; depth++)
    {
        uint offsetStartWeights =
            (depthPrev + nbChannelsPrev * depth) * weightHeight;
        uint offsetStart = (depth + nbChannels * elem) * height;
        
        for (int k=startI; k<=endI; k++){
        for (int l=startJ; l<=endJ; l++)
        {
            int i1, j1;
            if ((i-k) % stride != 0)
            {
                continue;
            }
            else if ((j-l) % stride != 0)
            {
                continue;
            }
            else
            {
                i1 = (i-k) / stride;
                j1 = (j-l) / stride;
            }
            if (j1 >= 0 && j1 < (int)width &&
                i1 >= 0 && i1 < (int)height)
            {
                uint offset = j1 + (offsetStart + i1) * width;
                float deltaCur = delta[offset];
                
                uint offsetWeights = l-startJ +
                    (offsetStartWeights + k-startI) * weightWidth;
                float w = weights[offsetWeights];
                
                tmp += deltaCur * w;
            }
        }}
    }
    
    uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
    if (dirty)
    {
        deltaPrev[offsetPrev] = tmp;
    }
    else
    {
        deltaPrev[offsetPrev] += tmp;
    }
}

kernel void convBatchDerWeights(
    const device float * outsPrev,
    const device float * delta,
    constant int * pStart,
    constant uint * pStride,
    constant uint * pNbChannels,
    constant uint * pNbChannelsPrev,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pDimWeights,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint weightHeight, weightWidth;
    uint nbChannels;
    uint nbChannelsPrev;
    int startI, startJ;
    int endI, endJ;
    uint stride;
    uint nbBatch;
    uint accumulate;
    
    if (pStart && pStride && pNbChannels && pNbChannelsPrev && pDimensions &&
        pDimensionsPrev && pDimWeights && pNbBatch && pAccumulate &&
        outsPrev && delta && grads)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        weightWidth = pDimWeights[0];
        weightHeight = pDimWeights[1];
        nbChannels = *pNbChannels;
        nbChannelsPrev = *pNbChannelsPrev;
        nbBatch = *pNbBatch;
        startI = pStart[0];
        endI = pStart[1];
        startJ = pStart[2];
        endJ = pStart[3];
        stride = pStride[0];
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    int weightsI = id[1] / nbChannelsPrev;
    int weightsJ = id[0] / nbChannels;
    uint depth = id[0] % nbChannels;
    uint depthPrev = id[1] % nbChannelsPrev;
    
    if (id[0] >= nbChannels * weightWidth ||
        id[1] >= nbChannelsPrev * weightHeight ||
        weightsI + startI > endI || weightsJ + startJ > endJ)
    {
        return ;
    }
    
    int i = weightsI + startI;
    int j = weightsJ + startJ;
    
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offsetStart =
            (depth + nbChannels * elem) * height;
        uint offsetStartPrev =
            (depthPrev + nbChannelsPrev * elem) * heightPrev;
        
        for (uint k=0; k<height; k++){
        for (uint l=0; l<width; l++)
        {
            if ((int)(stride*l)+j >= 0 && stride*l+j < widthPrev &&
                (int)(stride*k)+i >= 0 && stride*k+i < heightPrev)
            {
                uint offset = l + (offsetStart + k) * width;
                float deltaCur = delta[offset];
                
                uint offsetPrev = stride*l+j +
                    (offsetStartPrev + stride*k+i)*widthPrev;
                float outPrev = outsPrev[offsetPrev];
                
                tmp += deltaCur * outPrev;
            }
        }}
    }
    
    uint offsetStartWeights =
        (depthPrev + nbChannelsPrev * depth) * weightHeight;
    uint offsetWeights = j-startJ +
        (offsetStartWeights + i-startI) * weightWidth;
    
    if (accumulate)
    {
        grads[offsetWeights] += tmp;
    }
    else
    {
        grads[offsetWeights] = tmp;
    }
}

kernel void convBatchDerBiases(
    const device float * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    uint accumulate;
    
    if (pNbChannels && pDimensions && pNbBatch && pAccumulate &&
        delta && grads)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint depth = id;
    if (depth >= nbChannels)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        
        for (uint k=0; k<height; k++){
        for (uint l=0; l<width; l++)
        {
            uint offset = l + (offsetStart + k) * width;
            tmp += delta[offset];
        }}
    }
    
    if (accumulate)
    {
        grads[depth] += tmp;
    }
    else
    {
        grads[depth] = tmp;
    }
}

kernel void convDerWeights(
    const device float * outsPrev,
    const device float * delta,
    constant int * pStart,
    constant uint * pStride,
    constant uint * pNbChannels,
    constant uint * pNbChannelsPrev,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pDimWeights,
    constant uint * pNbBatch,
    device float * deltaWeights,
    uint3 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint weightHeight, weightWidth;
    uint nbChannels;
    uint nbChannelsPrev;
    int startI, startJ;
    int endI, endJ;
    uint stride;
    uint nbBatch;
    
    if (pStart && pStride && pNbChannels && pNbChannelsPrev && pDimensions &&
        pDimensionsPrev && pDimWeights && pNbBatch &&
        outsPrev && delta && deltaWeights)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        weightWidth = pDimWeights[0];
        weightHeight = pDimWeights[1];
        nbChannels = *pNbChannels;
        nbChannelsPrev = *pNbChannelsPrev;
        nbBatch = *pNbBatch;
        startI = pStart[0];
        endI = pStart[1];
        startJ = pStart[2];
        endJ = pStart[3];
        stride = pStride[0];
    }
    else
        return ;
    
    int weightsI = id[1] / nbChannelsPrev;
    int weightsJ = id[0] / nbChannels;
    uint depth = id[0] % nbChannels;
    uint depthPrev = id[1] % nbChannelsPrev;
    uint elem = id[2];
    
    if (id[0] >= nbChannels * weightWidth ||
        id[1] >= nbChannelsPrev * weightHeight ||
        elem >= nbBatch ||
        weightsI + startI > endI || weightsJ + startJ > endJ)
    {
        return ;
    }
    
    uint offsetStartGridWeights =
        elem * nbChannels * nbChannelsPrev * weightHeight;
    
    int i = weightsI + startI;
    int j = weightsJ + startJ;
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offsetStartPrev = (depthPrev + nbChannelsPrev * elem) * heightPrev;
    uint offsetStartWeights =
        (depthPrev + nbChannelsPrev * depth) * weightHeight;
    
    float tmp = 0.0;
    for (uint k=0; k<height; k++){
    for (uint l=0; l<width; l++)
    {
        if ((int)(stride*l)+j >= 0 && stride*l+j < widthPrev &&
            (int)(stride*k)+i >= 0 && stride*k+i < heightPrev)
        {
            uint offset = l + (offsetStart + k) * width;
            float deltaCur = delta[offset];
            
            uint offsetPrev = stride*l+j +
                (offsetStartPrev + stride*k+i)*widthPrev;
            float outPrev = outsPrev[offsetPrev];
            
            tmp += deltaCur * outPrev;
        }
    }}
    
    uint offsetWeights = j-startJ +
        (offsetStartGridWeights+offsetStartWeights+i-startI)*weightWidth;
    deltaWeights[offsetWeights] = tmp;
}

kernel void convDerBiases(
    const device float * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device float * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch && delta && deltaWeights)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbChannels || elem >= nbBatch)
    {
        return ;
    }
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    
    float tmp = 0.0;
    for (uint i=0; i<height; i++){
    for (uint j=0; j<width; j++)
    {
        uint offset = j + (offsetStart + i) * width;
        tmp += delta[offset];
    }}
    
    uint offsetWeights = elem * nbChannels + depth;
    deltaWeights[offsetWeights] = tmp;
}

kernel void convReduceWeights(
    const device float * deltaWeights,
    constant uint * pNbChannels,
    constant uint * pNbChannelsPrev,
    constant uint * pDimWeights,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbChannels;
    uint nbChannelsPrev;
    uint weightHeight, weightWidth;
    uint nbBatch;
    uint accumulate;
    
    if (pNbChannels && pNbChannelsPrev && pDimWeights && pNbBatch &&
        pAccumulate && deltaWeights && grads)
    {
        nbChannels = *pNbChannels;
        nbChannelsPrev = *pNbChannelsPrev;
        weightWidth = pDimWeights[0];
        weightHeight = pDimWeights[1];
        nbBatch = *pNbBatch;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint weightsI = id[1] / nbChannelsPrev;
    uint weightsJ = id[0] / nbChannels;
    uint depth = id[0] % nbChannels;
    uint depthPrev = id[1] % nbChannelsPrev;
    
    if (id[0] >= nbChannels * weightWidth ||
        id[1] >= nbChannelsPrev * weightHeight)
    {
        return ;
    }
    
    uint offsetStartWeights =
        (depthPrev + nbChannelsPrev * depth) * weightHeight;
    uint offsetWeights = weightsJ +
        (offsetStartWeights + weightsI) * weightWidth;
    
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offsetStart =
            elem * nbChannels * nbChannelsPrev * weightHeight;
        uint offset = weightsJ +
            (offsetStart + offsetStartWeights + weightsI) * weightWidth;
                
        tmp += deltaWeights[offset];
    }
    
    if (accumulate)
    {
        grads[offsetWeights] += tmp;
    }
    else
    {
        grads[offsetWeights] = tmp;
    }
}
