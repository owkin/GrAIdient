//
// Deconvolution.metal
// GrAIdient
//
// Created by Jean-François Reboud on 28/12/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void deconvForward(
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
    
    uint depth = id[0] / width;
    uint elem = id[1] / height;
    uint i = id[1] % height;
    uint j = id[0] % width;
    
    if (i * elem >= height * nbBatch ||
        j * depth >= width * nbChannels)
    {
        return ;
    }
    
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
            if ((i+k-endI) % stride == 0 && (j+l-endJ) % stride == 0)
            {
                int i1 = (i+k-endI) / stride;
                int j1 = (j+l-endJ) / stride;
                
                if (j1 >= 0 && j1 < (int)widthPrev &&
                    i1 >= 0 && i1 < (int)heightPrev)
                {
                    uint offsetPrev = j1 +
                        (offsetStartPrev + i1) * widthPrev;
                    float outPrev = outsPrev[offsetPrev];
                    
                    uint offsetWeights = l-startJ +
                        (offsetStartWeights + k-startI) * weightWidth;
                    float w = weights[offsetWeights];
                    
                    tmp += outPrev * w;
                }
            }
        }}
    }
    
    uint offset = j + (offsetStart + i)*width;
    outs[offset] = tmp;
}

kernel void deconvBackward(
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
    
    uint depthPrev = id[0] / widthPrev;
    uint elem = id[1] / heightPrev;
    uint i = id[1] % heightPrev;
    uint j = id[0] % widthPrev;
    
    if (i * elem >= heightPrev * nbBatch ||
        j * depthPrev >= widthPrev * nbChannelsPrev)
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
            if ((int)(stride*j)+endJ-l >= 0 && stride*j+endJ-l < width
                && (int)(stride*i)+endI-k >= 0 && stride*i+endI-k < height)
            {
                uint offset = stride*j+endJ-l +
                    (offsetStart + stride*i+endI-k) * width;
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

kernel void deconvBatchDerWeights(
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
            if ((i+k-endI) % stride == 0 && (j+l-endJ) % stride == 0)
            {
                int i1 = (i+k-endI) / stride;
                int j1 = (j+l-endJ) / stride;
                
                if (j1 >= 0 && j1 < (int)widthPrev &&
                    i1 >= 0 && i1 < (int)heightPrev)
                {
                    uint offset = l + (offsetStart + k) * width;
                    float deltaCur = delta[offset];
                    
                    uint offsetPrev = j1 +
                        (offsetStartPrev + i1)*widthPrev;
                    float outPrev = outsPrev[offsetPrev];
                    
                    tmp += deltaCur * outPrev;
                }
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

kernel void deconvDerWeights(
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
    
    uint remains = id[0];
    uint elem = remains / (weightWidth * nbChannels);
    remains = remains % (weightWidth * nbChannels);
    int weightsI = id[1] / nbChannelsPrev;
    int weightsJ = remains / nbChannels;
    uint depth = remains % nbChannels;
    uint depthPrev = id[1] % nbChannelsPrev;
    
    if (id[0] >= nbBatch * nbChannels * weightWidth ||
        id[1] >= nbChannelsPrev * weightHeight ||
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
        if ((i+k-endI) % stride == 0 && (j+l-endJ) % stride == 0)
        {
            int i1 = (i+k-endI) / stride;
            int j1 = (j+l-endJ) / stride;
            
            if (j1 >= 0 && j1 < (int)widthPrev &&
                i1 >= 0 && i1 < (int)heightPrev)
            {
                uint offset = l + (offsetStart + k) * width;
                float deltaCur = delta[offset];
                
                uint offsetPrev = j1 +
                    (offsetStartPrev + i1)*widthPrev;
                float outPrev = outsPrev[offsetPrev];
                
                tmp += deltaCur * outPrev;
            }
        }
    }}
    
    uint offsetWeights = j-startJ +
        (offsetStartGridWeights+offsetStartWeights+i-startI)*weightWidth;
    deltaWeights[offsetWeights] = tmp;
}
