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
    constant uint * pNbNeurons,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint heightPrev, widthPrev;
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbNeurons && pDimensionsPrev && pNbBatch &&
        outsPrev && outs)
    {
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
    constant uint * pNbNeurons,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint heightPrev, widthPrev;
    uint nbNeurons;
    uint nbBatch;
    uint dirty;
    
    if (pNbNeurons && pDimensionsPrev && pNbBatch && pDirty &&
        delta && deltaPrev)
    {
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depthPrev = id[0] / widthPrev;
    uint elem = id[1] / heightPrev;
    uint i = id[1] % heightPrev;
    uint j = id[0] % widthPrev;
    
    if (i * elem >= heightPrev * nbBatch ||
        j * depthPrev >= widthPrev * nbNeurons)
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
    uint2 id [[ thread_position_in_grid ]])
{
    int start, end;
    uint stride;
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbChannels;
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
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
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
        
    uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
    uint offsetStart = (depth + nbChannels * elem) * height;
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
    uint2 id [[ thread_position_in_grid ]])
{
    int start, end;
    uint stride;
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbChannels;
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
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0] / widthPrev;
    uint elem = id[1] / heightPrev;
    uint i = id[1] % heightPrev;
    uint j = id[0] % widthPrev;
    
    if (i * elem >= heightPrev * nbBatch ||
        j * depth >= widthPrev * nbChannels)
    {
        return ;
    }
    
    uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
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
            uint offsetStart = (depth + nbChannels * elem) * height;
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
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pNbBatch &&
        outsPrev && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
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
    
    uint startI = _startIndex(i, height, heightPrev);
    uint endI = _endIndex(i, height, heightPrev);
    uint startJ = _startIndex(j, width, widthPrev);
    uint endJ = _endIndex(j, width, widthPrev);
    
    uint nbElemsI = endI - startI;
    uint nbElemsJ = endJ - startJ;
    uint nbElems = nbElemsI * nbElemsJ;
        
    uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
    uint offsetStart = (depth + nbChannels * elem) * height;
    
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
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pNbBatch &&
        outsPrev && nbElems && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
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
    uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
    
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
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pNbBatch &&
        delta && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
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
    uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
    
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
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pNbBatch &&
        delta && nbElems && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
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
    uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
    
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

kernel void selectNeurons2DForward(
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

kernel void selectNeurons2DBackward(
    const device float * delta,
    constant uint * pTarget,
    constant uint * pNbNeurons,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
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
    
    uint depthPrev = id[0] / widthPrev;
    uint elem = id[1] / heightPrev;
    uint i = id[1] % heightPrev;
    uint j = id[0] % widthPrev;
    
    if (i * elem >= heightPrev * nbBatch ||
        j * depthPrev >= widthPrev * nbNeurons)
    {
        return ;
    }
    
    float deltaCur = 0.0;
    if (i == targetI && j == targetJ)
    {
        uint offset = depthPrev + nbNeurons * elem;
        deltaCur = delta[offset];
    }
    
    uint offsetStartPrev = (depthPrev + nbNeurons * elem) * heightPrev;
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

kernel void IRDFT2RGBForward(
    const device float * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch && outsPrev && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
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
        
    uint offsetStartRealPrev = (2 * depth + 2 * nbChannels * elem) * height;
    uint offsetStartImPrev = (2 * depth + 1 + 2 * nbChannels * elem) * height;
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    float sum = 0.0;
    for (uint k=0; k<height; k++){
    for (uint l=0; l<width; l++)
    {
        uint offsetRealPrev = l + (offsetStartRealPrev + k) * width;
        uint offsetImPrev = l + (offsetStartImPrev + k) * width;
        
        float angle = 2.0 * M_PI_F;
        angle *= (float(i) / height * k + float(j) / width * l);
        
        sum += outsPrev[offsetRealPrev] * cos(angle) -
            outsPrev[offsetImPrev] * sin(angle);
    }}
    sum /= float(height * width);
    outs[offset] = sum;
}

kernel void IRDFT2RGBBackward(
    const device float * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensions &&
        pNbBatch && pDirty && delta && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
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
    
    uint offsetStartRealPrev = (2 * depth + 2 * nbChannels * elem) * height;
    uint offsetStartImPrev = (2 * depth + 1 + 2 * nbChannels * elem) * height;
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offsetRealPrev = j + (offsetStartRealPrev + i) * width;
    uint offsetImPrev = j + (offsetStartImPrev + i) * width;
    
    float sum1 = 0.0;
    float sum2 = 0.0;
    for (uint k=0; k<height; k++){
    for (uint l=0; l<width; l++)
    {
        uint offset = l + (offsetStart + k) * width;
        float deltaCur = delta[offset];
        
        float angle = 2.0 * M_PI_F;
        angle *= (float(i) / height * k + float(j) / width * l);
        
        sum1 += deltaCur * cos(angle);
        sum2 -= deltaCur * sin(angle);
    }}
    sum1 /= float(height * width);
    sum2 /= float(height * width);
    
    if (dirty)
    {
        deltaPrev[offsetRealPrev] = sum1;
        deltaPrev[offsetImPrev] = sum2;
    }
    else
    {
        deltaPrev[offsetRealPrev] += sum1;
        deltaPrev[offsetImPrev] += sum2;
    }
}

kernel void decorrelateRGBForward(
    const device float * outsPrev,
    constant float * correlation,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch &&
        outsPrev && correlation && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
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
    
    uint block = depth / 3;
    uint res = depth % 3;
        
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    float sum = 0.0;
    for (uint k=0; k<3; k++)
    {
        uint offsetStartPrev = (block * 3 + k + nbChannels * elem) * height;
        uint offsetPrev = j + (offsetStartPrev + i) * width;
        
        sum += outsPrev[offsetPrev] * correlation[res * 3 + k];
    }
    outs[offset] = sum;
}

kernel void decorrelateRGBBackward(
    const device float * delta,
    constant float * correlation,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensions && pNbBatch && pDirty &&
        delta && correlation && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
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
    
    uint block = depth / 3;
    uint res = depth % 3;
    
    uint offsetStartPrev = (depth + nbChannels * elem) * height;
    uint offsetPrev = j + (offsetStartPrev + i) * width;
    
    float sum = 0.0;
    for (uint k=0; k<3; k++)
    {
        uint offsetStart = (block * 3 + k + nbChannels * elem) * height;
        uint offset = j + (offsetStart + i) * width;
        
        sum += delta[offset] * correlation[k * 3 + res];
    }
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = sum;
    }
    else
    {
        deltaPrev[offsetPrev] += sum;
    }
}

kernel void linearScale2DForward(
    const device float * outsPrev,
    constant float * weights,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch &&
        outsPrev && weights && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
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
    
    outs[offset] = weights[0] * outsPrev[offset] + weights[1];
}

kernel void linearScale2DBackward(
    const device float * delta,
    constant float * weights,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensions && pNbBatch && pDirty &&
        delta && weights && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
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
    
    uint offsetStartPrev = (depth + nbChannels * elem) * height;
    uint offsetPrev = j + (offsetStartPrev + i) * width;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = delta[offsetPrev] * weights[0];
    }
    else
    {
        deltaPrev[offsetPrev] += delta[offsetPrev] * weights[0];
    }
}

float _getScaleValue(
    const uint i,
    const uint j,
    const uint dimension)
{
    float freq = sqrt(float(i * i + j * j)) / float(dimension);
    freq = max(freq, 1.0 / float(dimension));
    return (1.0 / freq) * float(dimension);
}

kernel void setDataFTFrequences2D(
    constant uint * pNbChannels,
    constant uint * pDimension,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint dimension;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimension && pNbBatch && outs)
    {
        dimension = *pDimension;
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0] / dimension;
    uint elem = id[1] / dimension;
    uint i = id[1] % dimension;
    uint j = id[0] % dimension;
    
    if (i * elem >= dimension * nbBatch ||
        j * depth >= dimension * nbChannels)
    {
        return ;
    }
        
    uint end = dimension % 2 == 0 ? dimension / 2 : (dimension - 1) / 2;
    uint jTmp = j;
    uint iTmp = i;
    if (dimension % 2 == 0)
    {
        if (jTmp >= end)
        {
            jTmp = jTmp - end + 1;
            jTmp = end + 1 - jTmp;
        }
        if (iTmp >= end)
        {
            iTmp = iTmp - end + 1;
            iTmp = end + 1 - iTmp;
        }
    }
    else
    {
        if (jTmp > end)
        {
            jTmp = jTmp - end;
            jTmp = end + 1 - jTmp;
        }
        if (iTmp > end)
        {
            iTmp = iTmp - end;
            iTmp = end + 1 - iTmp;
        }
    }
    
    uint offsetStart = (depth + nbChannels * elem) * dimension;
    uint offset = j + (offsetStart + i) * dimension;

    outs[offset] = _getScaleValue(iTmp, jTmp, dimension);
}

kernel void pad2DForward(
    const device float * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pPadDimension,
    constant float * pPadValue,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbChannels;
    uint padDimension;
    float padValue;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch &&
        pPadDimension && pPadValue && outsPrev && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        padDimension = *pPadDimension;
        padValue = *pPadValue;
        widthPrev = width - 2 * padDimension;
        heightPrev = height - 2 * padDimension;
        nbBatch = *pNbBatch;
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
    
    if (i < padDimension || i >= height - padDimension ||
        j < padDimension || j >= width - padDimension)
    {
        outs[offset] = padValue;
    }
    else
    {
        uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
        uint offsetPrev = j-padDimension +
            (offsetStartPrev + i-padDimension) * widthPrev;
        
        outs[offset] = outsPrev[offsetPrev];
    }
}

kernel void pad2DBackward(
    const device float * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pPadDimension,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbChannels;
    uint padDimension;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensions && pPadDimension && pNbBatch && pDirty &&
        delta && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        padDimension = *pPadDimension;
        widthPrev = width - 2 * padDimension;
        heightPrev = height - 2 * padDimension;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0] / widthPrev;
    uint elem = id[1] / heightPrev;
    uint i = id[1] % heightPrev;
    uint j = id[0] % widthPrev;
    
    if (i * elem >= heightPrev * nbBatch ||
        j * depth >= widthPrev * nbChannels)
    {
        return ;
    }
    
    uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
    uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j+padDimension +
        (offsetStart + i+padDimension) * width;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = delta[offset];
    }
    else
    {
        deltaPrev[offsetPrev] += delta[offset];
    }
}

kernel void jitter2DForward(
    const device float * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pJitterDimension,
    constant uint * pJitterOffsets,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbChannels;
    uint jitterDimension;
    uint offsetI, offsetJ;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch &&
        pJitterDimension && pJitterOffsets && outsPrev && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        jitterDimension = *pJitterDimension;
        offsetI = pJitterOffsets[0];
        offsetJ = pJitterOffsets[1];
        widthPrev = width + jitterDimension;
        heightPrev = height + jitterDimension;
        nbBatch = *pNbBatch;
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
        
    uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
    uint offsetPrev = j+offsetJ +
        (offsetStartPrev + i+offsetI) * widthPrev;
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    outs[offset] = outsPrev[offsetPrev];
}

kernel void jitter2DBackward(
    const device float * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pJitterDimension,
    constant uint * pJitterOffsets,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbChannels;
    uint jitterDimension;
    uint offsetI, offsetJ;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensions && pNbBatch &&
        pJitterDimension && pJitterOffsets && pDirty &&
        delta && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        jitterDimension = *pJitterDimension;
        offsetI = pJitterOffsets[0];
        offsetJ = pJitterOffsets[1];
        widthPrev = width + jitterDimension;
        heightPrev = height + jitterDimension;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0] / widthPrev;
    uint elem = id[1] / heightPrev;
    uint i = id[1] % heightPrev;
    uint j = id[0] % widthPrev;
    
    if (i * elem >= heightPrev * nbBatch ||
        j * depth >= widthPrev * nbChannels)
    {
        return ;
    }
    
    uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
    uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
    
    if (dirty &&
        (i < offsetI || i >= height + offsetI ||
         j < offsetJ || j >= width + offsetJ))
    {
        deltaPrev[offsetPrev] = 0.0;
    }
    else if (dirty)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        uint offset = j-offsetJ + (offsetStart + i-offsetI) * width;
        
        deltaPrev[offsetPrev] = delta[offset];
    }
    else if (i >= offsetI && i < height + offsetI &&
             j >= offsetJ && j < width + offsetJ)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        uint offset = j-offsetJ + (offsetStart + i-offsetI) * width;
        
        deltaPrev[offsetPrev] += delta[offset];
    }
}

kernel void resizeBilinearForward(
    const device float * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pDimensionsResize,
    constant float * pPadValue,
    constant uint * pNbBatch,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint heightResize, widthResize;
    uint nbChannels;
    float padValue;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pDimensionsResize &&
        pPadValue && pNbBatch && outsPrev && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        widthResize = pDimensionsResize[0];
        heightResize = pDimensionsResize[1];
        padValue = *pPadValue;
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
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
    
    float ratioInOutI = float(heightPrev - 1) / float(heightResize - 1);
    float ratioInOutJ = float(widthPrev - 1) / float(widthResize - 1);
    float padDimensionI = (width - widthResize) / 2;
    float padDimensionJ = (height - heightResize) / 2;
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    if (i < padDimensionI || i >= height - padDimensionI ||
        j < padDimensionJ || j >= width - padDimensionJ)
    {
        outs[offset] = padValue;
    }
    else
    {
        float I = i-padDimensionI;
        float J = j-padDimensionJ;
        
        float iPrev = float(I) * ratioInOutI;
        float jPrev = float(J) * ratioInOutJ;
        
        uint iPrevInf = floor(iPrev);
        uint iPrevSup = ceil(iPrev);
        uint jPrevInf = floor(jPrev);
        uint jPrevSup = ceil(jPrev);
        
        float iWeight = ratioInOutI * float(I) - float(iPrevInf);
        float jWeight = ratioInOutJ * float(J) - float(jPrevInf);
        
        uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
        uint offsetPrev11 = jPrevInf + (offsetStartPrev + iPrevInf) * widthPrev;
        uint offsetPrev12 = jPrevSup + (offsetStartPrev + iPrevInf) * widthPrev;
        uint offsetPrev21 = jPrevInf + (offsetStartPrev + iPrevSup) * widthPrev;
        uint offsetPrev22 = jPrevSup + (offsetStartPrev + iPrevSup) * widthPrev;
        
        float out = outsPrev[offsetPrev11] * (1.0 - iWeight) * (1.0 - jWeight);
        out += outsPrev[offsetPrev12] * (1.0 - iWeight) * jWeight;
        out += outsPrev[offsetPrev21] * iWeight * (1.0 - jWeight);
        out += outsPrev[offsetPrev22] * iWeight * jWeight;
        
        outs[offset] = out;
    }
}

kernel void resizeBilinearBackward(
    const device float * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pDimensionsResize,
    constant float * pPadValue,
    constant uint * pNbBatch,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint heightResize, widthResize;
    uint nbChannels;
    float padValue;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pDimensionsResize &&
        pPadValue && pNbBatch && delta && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        widthResize = pDimensionsResize[0];
        heightResize = pDimensionsResize[1];
        padValue = *pPadValue;
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (elem >= nbBatch || depth >= nbChannels)
    {
        return ;
    }
    
    float ratioInOutI = float(heightPrev - 1) / float(heightResize - 1);
    float ratioInOutJ = float(widthPrev - 1) / float(widthResize - 1);
    float padDimensionI = (width - widthResize) / 2;
    float padDimensionJ = (height - heightResize) / 2;
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
    
    for (uint i = 0; i < heightResize; i++) {
    for (uint j = 0; j < widthResize; j++)
    {
        float iPrev = float(i) * ratioInOutI;
        float jPrev = float(j) * ratioInOutJ;
        
        uint iPrevInf = floor(iPrev);
        uint iPrevSup = ceil(iPrev);
        uint jPrevInf = floor(jPrev);
        uint jPrevSup = ceil(jPrev);
        
        float iWeight = ratioInOutI * float(i) - float(iPrevInf);
        float jWeight = ratioInOutJ * float(j) - float(jPrevInf);
        
        uint offset = j+padDimensionJ + (offsetStart + i+padDimensionI) * width;
        float deltaCur = delta[offset];
        
        uint offsetPrev11 = jPrevInf + (offsetStartPrev + iPrevInf) * widthPrev;
        uint offsetPrev12 = jPrevSup + (offsetStartPrev + iPrevInf) * widthPrev;
        uint offsetPrev21 = jPrevInf + (offsetStartPrev + iPrevSup) * widthPrev;
        uint offsetPrev22 = jPrevSup + (offsetStartPrev + iPrevSup) * widthPrev;
        
        deltaPrev[offsetPrev11] += deltaCur * (1.0 - iWeight) * (1.0 - jWeight);
        deltaPrev[offsetPrev12] += deltaCur * (1.0 - iWeight) * jWeight;
        deltaPrev[offsetPrev21] += deltaCur * iWeight * (1.0 - jWeight);
        deltaPrev[offsetPrev22] += deltaCur * iWeight * jWeight;
    }}
}
