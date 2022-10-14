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
    constant uint * pNbFilters,
    constant uint * pNbFiltersPrev,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pDimWeights,
    constant uint * pNbBatch,
    device float * outs,
    uint3 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint weightHeight, weightWidth;
    uint nbNeurones;
    uint nbNeuronesPrev;
    int startI, startJ;
    int endI, endJ;
    uint stride;
    uint nbBatch;
    
    if (pStart && pStride && pNbFilters && pNbFiltersPrev &&
        pDimensions && pDimensionsPrev && pDimWeights && pNbBatch &&
        outsPrev && weights && biases && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        weightWidth = pDimWeights[0];
        weightHeight = pDimWeights[1];
        nbNeurones = *pNbFilters;
        nbNeuronesPrev = *pNbFiltersPrev;
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
    uint depth = id[2] % nbNeurones;
    uint elem = id[2] / nbNeurones;
    
    if (i >= height || j >= width ||
        id[2] >= nbNeurones * nbBatch)
    {
        return ;
    }
    
    uint offsetStart = (depth+nbNeurones*elem)*height;
    
    float tmp = biases[depth];
    for (uint depthPrev=0; depthPrev<nbNeuronesPrev; depthPrev++)
    {
        uint offsetStartPrev =
            (depthPrev + nbNeuronesPrev*elem) * heightPrev;
        uint offsetStartWeights =
            (depthPrev + nbNeuronesPrev * depth) * weightHeight;
        
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
}

kernel void convBackward(
    const device float * delta,
    const device float * weights,
    constant int * pStart,
    constant uint * pStride,
    constant uint * pNbFilters,
    constant uint * pNbFiltersPrev,
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
    uint nbNeurones;
    uint nbNeuronesPrev;
    int startI, startJ;
    int endI, endJ;
    uint stride;
    uint nbBatch;
    uint dirty;
    
    if (pStart && pStride && pNbFilters && pNbFiltersPrev &&
        pDimensions && pDimensionsPrev && pDimWeights && pNbBatch && pDirty &&
        delta && weights && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        weightWidth = pDimWeights[0];
        weightHeight = pDimWeights[1];
        nbNeurones = *pNbFilters;
        nbNeuronesPrev = *pNbFiltersPrev;
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
    uint depthPrev = id[2] % nbNeuronesPrev;
    uint elem = id[2] / nbNeuronesPrev;
    
    if (i >= heightPrev || j >= widthPrev ||
        id[2] >= nbNeuronesPrev * nbBatch)
    {
        return ;
    }
    
    uint offsetStartPrev = (depthPrev + nbNeuronesPrev * elem) * heightPrev;
    
    float tmp = 0.0;
    for (uint depth=0; depth<nbNeurones; depth++)
    {
        uint offsetStartWeights =
            (depthPrev + nbNeuronesPrev * depth) * weightHeight;
        uint offsetStart = (depth + nbNeurones * elem) * height;
        
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
    constant uint * pNbFilters,
    constant uint * pNbFiltersPrev,
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
    uint nbNeurones;
    uint nbNeuronesPrev;
    int startI, startJ;
    int endI, endJ;
    uint stride;
    uint nbBatch;
    uint accumulate;
    
    if (pStart && pStride && pNbFilters && pNbFiltersPrev && pDimensions &&
        pDimensionsPrev && pDimWeights && pNbBatch && pAccumulate &&
        outsPrev && delta && grads)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        weightWidth = pDimWeights[0];
        weightHeight = pDimWeights[1];
        nbNeurones = *pNbFilters;
        nbNeuronesPrev = *pNbFiltersPrev;
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
    
    int weightsI = id[1] / nbNeuronesPrev;
    int weightsJ = id[0] / nbNeurones;
    uint depth = id[0] % nbNeurones;
    uint depthPrev = id[1] % nbNeuronesPrev;
    
    if (id[0] >= nbNeurones * weightWidth ||
        id[1] >= nbNeuronesPrev * weightHeight ||
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
            (depth + nbNeurones * elem) * height;
        uint offsetStartPrev =
            (depthPrev + nbNeuronesPrev * elem) * heightPrev;
        
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
        (depthPrev + nbNeuronesPrev * depth) * weightHeight;
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
    constant uint * pNbFilters,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbNeurones;
    uint nbBatch;
    uint accumulate;
    
    if (pNbFilters && pDimensions && pNbBatch && pAccumulate &&
        delta && grads)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbNeurones = *pNbFilters;
        nbBatch = *pNbBatch;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint depth = id;
    if (depth >= nbNeurones)
    {
        return ;
    }
    
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offsetStart = (depth + nbNeurones * elem) * height;
        
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
    constant uint * pNbFilters,
    constant uint * pNbFiltersPrev,
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
    uint nbNeurones;
    uint nbNeuronesPrev;
    int startI, startJ;
    int endI, endJ;
    uint stride;
    uint nbBatch;
    
    if (pStart && pStride && pNbFilters && pNbFiltersPrev && pDimensions &&
        pDimensionsPrev && pDimWeights && pNbBatch &&
        outsPrev && delta && deltaWeights)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        weightWidth = pDimWeights[0];
        weightHeight = pDimWeights[1];
        nbNeurones = *pNbFilters;
        nbNeuronesPrev = *pNbFiltersPrev;
        nbBatch = *pNbBatch;
        startI = pStart[0];
        endI = pStart[1];
        startJ = pStart[2];
        endJ = pStart[3];
        stride = pStride[0];
    }
    else
        return ;
    
    int weightsI = id[1] / nbNeuronesPrev;
    int weightsJ = id[0] / nbNeurones;
    uint depth = id[0] % nbNeurones;
    uint depthPrev = id[1] % nbNeuronesPrev;
    uint elem = id[2];
    
    if (id[0] >= nbNeurones * weightWidth ||
        id[1] >= nbNeuronesPrev * weightHeight ||
        elem >= nbBatch ||
        weightsI + startI > endI || weightsJ + startJ > endJ)
    {
        return ;
    }
    
    uint offsetStartGridWeights =
        elem * nbNeurones * nbNeuronesPrev * weightHeight;
    
    int i = weightsI + startI;
    int j = weightsJ + startJ;
    
    uint offsetStart = (depth + nbNeurones * elem) * height;
    uint offsetStartPrev = (depthPrev + nbNeuronesPrev * elem) * heightPrev;
    uint offsetStartWeights =
        (depthPrev + nbNeuronesPrev * depth) * weightHeight;
    
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
    constant uint * pNbFilters,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device float * deltaWeights,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbNeurones;
    uint nbBatch;
    
    if (pNbFilters && pDimensions && pNbBatch && delta && deltaWeights)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbNeurones = *pNbFilters;
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
    
    uint offsetStart = (depth + nbNeurones * elem) * height;
    
    float tmp = 0.0;
    for (uint i=0; i<height; i++){
    for (uint j=0; j<width; j++)
    {
        uint offset = j + (offsetStart + i) * width;
        tmp += delta[offset];
    }}
    
    uint offsetWeights = elem * nbNeurones + depth;
    deltaWeights[offsetWeights] = tmp;
}

kernel void convReduceWeights(
    const device float * deltaWeights,
    constant uint * pNbFilters,
    constant uint * pNbFiltersPrev,
    constant uint * pDimWeights,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device float * grads,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurones;
    uint nbNeuronesPrev;
    uint weightHeight, weightWidth;
    uint nbBatch;
    uint accumulate;
    
    if (pNbFilters && pNbFiltersPrev && pDimWeights && pNbBatch &&
        pAccumulate && deltaWeights && grads)
    {
        nbNeurones = *pNbFilters;
        nbNeuronesPrev = *pNbFiltersPrev;
        weightWidth = pDimWeights[0];
        weightHeight = pDimWeights[1];
        nbBatch = *pNbBatch;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint weightsI = id[1] / nbNeuronesPrev;
    uint weightsJ = id[0] / nbNeurones;
    uint depth = id[0] % nbNeurones;
    uint depthPrev = id[1] % nbNeuronesPrev;
    
    if (id[0] >= nbNeurones * weightWidth ||
        id[1] >= nbNeuronesPrev * weightHeight)
    {
        return ;
    }
    
    uint offsetStartWeights =
        (depthPrev + nbNeuronesPrev * depth) * weightHeight;
    uint offsetWeights = weightsJ +
        (offsetStartWeights + weightsI) * weightWidth;
    
    float tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offsetStart =
            elem * nbNeurones * nbNeuronesPrev * weightHeight;
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
