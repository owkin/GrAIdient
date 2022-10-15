//
// Layer2D.metal
// MAKit
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void avgPoolForward(
    const device float * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint heightPrev, widthPrev;
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbChannels && pDimensionsPrev && pNbBatch &&
        outsPrev && outs)
    {
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        nbNeurons = *pNbChannels;
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
        
    uint offsetStartPrev = (depth + nbNeurons * elem) * heightPrev;
    
    float tmp = 0.0;
    for (uint i=0; i<heightPrev; i++){
    for (uint j=0; j<widthPrev; j++)
    {
        uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
        tmp += outsPrev[offsetPrev];
    }}
    tmp /= heightPrev * widthPrev;
    
    uint offset = depth + nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void avgPoolBackward(
    const device float * delta,
    constant uint * pNbChannels,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint3 id [[ thread_position_in_grid ]])
{
    uint heightPrev, widthPrev;
    uint nbNeurons;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensionsPrev && pNbBatch && pDirty &&
        delta && deltaPrev)
    {
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        nbNeurons = *pNbChannels;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint depthPrev = id[2] % nbNeurons;
    uint elem = id[2] / nbNeurons;
    
    if (i >= heightPrev || j >= widthPrev ||
        id[2] >= nbNeurons * nbBatch)
    {
        return ;
    }
    
    uint offset = depthPrev + nbNeurons * elem;
    float deltaCur = delta[offset];
    
    uint offsetStartPrev = (depthPrev + nbNeurons * elem) * heightPrev;
    uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = deltaCur / (heightPrev * widthPrev);
    }
    else
    {
        deltaPrev[offsetPrev] += deltaCur / (heightPrev * widthPrev);
    }
}

kernel void maxPoolForward(
    const device float * outsPrev,
    constant int * pStart,
    constant uint * pStride,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device float * outs,
    device int * indicesMax,
    uint3 id [[ thread_position_in_grid ]])
{
    int start, end;
    uint stride;
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbNeurons;
    uint nbBatch;
    
    if (pStart && pStride && pNbChannels && pDimensions && pDimensionsPrev &&
        pNbBatch && outsPrev && outs && indicesMax)
    {
        start = pStart[0];
        end = pStart[1];
        stride = pStride[0];
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        nbNeurons = *pNbChannels;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint depth = id[2] % nbNeurons;
    uint elem = id[2] / nbNeurons;
    
    if (i >= height || j >= width ||
        id[2] >= nbNeurons * nbBatch)
    {
        return ;
    }
        
    uint offsetStartPrev = (depth + nbNeurons * elem) * heightPrev;
    uint offsetStart = (depth + nbNeurons * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    int indexMax = -1;
    float maxVal = -10000.0;
    for (int k=start; k<=end; k++){
    for (int l=start; l<=end; l++)
    {
        if ((int)(stride*j)+l >= 0 && stride*j+l < widthPrev
            && (int)(stride*i)+k >= 0 && stride*i+k < heightPrev)
        {
            uint offsetPrev = stride*j+l +
                (offsetStartPrev + stride*i+k)*widthPrev;
            
            float outPrev = outsPrev[offsetPrev];
            if (outPrev > maxVal)
            {
                indexMax = offsetPrev;
                indicesMax[offset] = offsetPrev;
                maxVal = outPrev;
            }
        }
    }}
    
    outs[offset] = maxVal;
    indicesMax[offset] = indexMax;
}

kernel void maxPoolBackward(
    const device float * delta,
    const device int * indicesMax,
    constant int * pStart,
    constant uint * pStride,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint3 id [[ thread_position_in_grid ]])
{
    int start, end;
    uint stride;
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbNeurons;
    uint nbBatch;
    uint dirty;
    
    if (pStart && pStride && pNbChannels && pDimensions && pDimensionsPrev &&
        pNbBatch && pDirty && delta && indicesMax && deltaPrev)
    {
        start = pStart[0];
        end = pStart[1];
        stride = pStride[0];
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        nbNeurons = *pNbChannels;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint depth = id[2] % nbNeurons;
    uint elem = id[2] / nbNeurons;
    
    if (i >= heightPrev || j >= widthPrev ||
        id[2] >= nbNeurons * nbBatch)
    {
        return ;
    }
    
    uint offsetStartPrev = (depth + nbNeurons * elem) * heightPrev;
    uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
    
    float tmp = 0.0;
    for (int k=start; k<=end; k++){
    for (int l=start; l<=end; l++)
    {
        int i1, j1;
        // i-k rather than i+k to take into account non symetric kernels.
        // Exemple: size of kernel 2 instead of 3.
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
            uint offsetStart = (depth + nbNeurons * elem) * height;
            uint offset = j1 + (offsetStart + i1) * width;
            
            if ((uint)indicesMax[offset] == offsetPrev)
            {
                tmp += delta[offset];
            }
        }
    }}
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = tmp;
    }
    else
    {
        deltaPrev[offsetPrev] += tmp;
    }
}

uint _startIndex(uint index, uint smallSize, uint bigSize)
{
    float val = float(index * bigSize) / smallSize;
    val = round(val * 1000) / 1000;
    return (uint)(floor(val));
}

uint _endIndex(uint index, uint smallSize, uint bigSize)
{
    return (uint)(ceil(float((index + 1) * bigSize) / smallSize));
}

kernel void adaptiveAvgPoolForward1(
    const device float * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device float * outs,
    uint3 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pNbBatch &&
        outsPrev && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        nbNeurons = *pNbChannels;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint depth = id[2] % nbNeurons;
    uint elem = id[2] / nbNeurons;
    
    if (i >= height || j >= width ||
        id[2] >= nbNeurons * nbBatch)
    {
        return ;
    }
    
    uint startI = _startIndex(i, height, heightPrev);
    uint endI = _endIndex(i, height, heightPrev);
    uint startJ = _startIndex(j, width, widthPrev);
    uint endJ = _endIndex(j, width, widthPrev);
    
    uint nbElemsI = endI - startI;
    uint nbElemsJ = endJ - startJ;
    uint nbElems = nbElemsI * nbElemsJ;
        
    uint offsetStartPrev = (depth + nbNeurons * elem) * heightPrev;
    uint offsetStart = (depth + nbNeurons * elem) * height;
    
    float tmp = 0.0;
    for (uint k=0; k<nbElemsI; k++) {
    for (uint l=0; l<nbElemsJ; l++)
    {
        uint offsetPrev = startJ+l + (offsetStartPrev + startI+k) * widthPrev;
        tmp += outsPrev[offsetPrev];
    }}
    
    uint offset = j + (offsetStart + i) * width;
    outs[offset] = tmp / (float)nbElems;
}

kernel void adaptiveAvgPoolForward2(
    const device float * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device int * nbElems,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pNbBatch &&
        outsPrev && nbElems && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        nbNeurons = *pNbChannels;
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
    
    uint offsetStart = (depth + nbNeurons * elem) * height;
    uint offsetStartPrev = (depth + nbNeurons * elem) * heightPrev;
    
    for (uint i = 0; i < heightPrev; i++) {
    for (uint j = 0; j < widthPrev; j++)
    {
        uint startI = _startIndex(i, heightPrev, height);
        uint endI = _endIndex(i, heightPrev, height);
        uint startJ = _startIndex(j, widthPrev, width);
        uint endJ = _endIndex(j, widthPrev, width);
        
        uint nbElemsI = endI - startI;
        uint nbElemsJ = endJ - startJ;
        
        uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
        float outPrev = outsPrev[offsetPrev];
        
        for (uint k = 0; k < nbElemsI; k++){
        for (uint l = 0; l < nbElemsJ; l++)
        {
            uint offset = startJ+l + (offsetStart + startI+k) * width;
            
            outs[offset] += outPrev;
            nbElems[offset] += 1;
        }}
    }}
    
    for (uint I = 0; I < height; I++){
    for (uint J = 0; J < width; J++)
    {
        uint offset = J + (offsetStart + I) * width;
        outs[offset] /= nbElems[offset];
    }}
}

kernel void adaptiveAvgPoolBackward1(
    const device float * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pNbBatch &&
        delta && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        nbNeurons = *pNbChannels;
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
    
    uint offsetStart = (depth + nbNeurons * elem) * height;
    uint offsetStartPrev = (depth + nbNeurons * elem) * heightPrev;
    
    for (uint i = 0; i < height; i++) {
    for (uint j = 0; j < width; j++)
    {
        uint startI = _startIndex(i, height, heightPrev);
        uint endI = _endIndex(i, height, heightPrev);
        uint startJ = _startIndex(j, width, widthPrev);
        uint endJ = _endIndex(j, width, widthPrev);
        
        uint nbElemsI = endI - startI;
        uint nbElemsJ = endJ - startJ;
        uint nbElems = nbElemsI * nbElemsJ;
        
        uint offset = j + (offsetStart + i) * width;
        float deltaCur = delta[offset] / (float)nbElems;
        
        for (uint k = 0; k < nbElemsI; k++){
        for (uint l = 0; l < nbElemsJ; l++)
        {
            uint offsetPrev = startJ+l +
                (offsetStartPrev + startI+k) * widthPrev;
            deltaPrev[offsetPrev] += deltaCur;
        }}
    }}
}

kernel void adaptiveAvgPoolBackward2(
    const device float * delta,
    const device int * nbElems,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pNbBatch &&
        delta && nbElems && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        nbNeurons = *pNbChannels;
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
    
    uint offsetStart = (depth + nbNeurons * elem) * height;
    uint offsetStartPrev = (depth + nbNeurons * elem) * heightPrev;
    
    for (uint i = 0; i < heightPrev; i++) {
    for (uint j = 0; j < widthPrev; j++)
    {
        uint startI = _startIndex(i, heightPrev, height);
        uint endI = _endIndex(i, heightPrev, height);
        uint startJ = _startIndex(j, widthPrev, width);
        uint endJ = _endIndex(j, widthPrev, width);
        
        uint nbElemsI = endI - startI;
        uint nbElemsJ = endJ - startJ;
        
        uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
        
        for (uint k = 0; k < nbElemsI; k++){
        for (uint l = 0; l < nbElemsJ; l++)
        {
            uint offset = startJ+l + (offsetStart + startI+k) * width;
            deltaPrev[offsetPrev] += delta[offset] / nbElems[offset];
        }}
    }}
}

kernel void selectNeuronForward(
    const device float * outsPrev,
    constant uint * pTarget,
    constant uint * pNbNeurons,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint targetI, targetJ;
    uint heightPrev, widthPrev;
    uint nbNeurons;
    uint nbBatch;
    
    if (pTarget && pNbNeurons && pDimensionsPrev && pNbBatch &&
        outsPrev && outs)
    {
        targetI = pTarget[0];
        targetJ = pTarget[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
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
        
    uint offsetStartPrev = (depth + nbNeurons * elem) * heightPrev;
    uint offsetPrev = targetJ +
        (offsetStartPrev + targetI) * widthPrev;
    uint offset = depth + nbNeurons * elem;
    
    outs[offset] = outsPrev[offsetPrev];
}

kernel void selectNeuronBackward(
    const device float * delta,
    constant uint * pTarget,
    constant uint * pNbNeurons,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint3 id [[ thread_position_in_grid ]])
{
    uint targetI, targetJ;
    uint heightPrev, widthPrev;
    uint nbNeurons;
    uint nbBatch;
    uint dirty;
    
    if (pTarget && pNbNeurons && pDimensionsPrev && pNbBatch && pDirty &&
        delta && deltaPrev)
    {
        targetI = pTarget[0];
        targetJ = pTarget[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint i = id[1];
    uint j = id[0];
    uint prevDepth = id[2] % nbNeurons;
    uint elem = id[2] / nbNeurons;
    
    if (i >= heightPrev || j >= widthPrev ||
        id[2] >= nbNeurons * nbBatch)
    {
        return ;
    }
    
    float deltaCur = 0.0;
    if (i == targetI && j == targetJ)
    {
        uint offset = prevDepth + nbNeurons * elem;
        deltaCur = delta[offset];
    }
    
    uint offsetStartPrev = (prevDepth + nbNeurons * elem) * heightPrev;
    uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = deltaCur;
    }
    else
    {
        deltaPrev[offsetPrev] += deltaCur;
    }
}