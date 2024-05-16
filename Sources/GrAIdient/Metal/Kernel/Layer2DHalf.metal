//
// Layer2D.metal
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void avgPoolForwardHalf(
    const device half * outsPrev,
    constant uint * pNbNeurons,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device half * outs,
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
    
    half tmp = 0.0;
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

kernel void avgPoolBackwardHalf(
    const device half * delta,
    constant uint * pNbNeurons,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
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
    half deltaCur = delta[offset];
    
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

kernel void maxPoolForwardHalf(
    const device half * outsPrev,
    constant int * pStart,
    constant uint * pStride,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device half * outs,
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
    half maxVal = -10000.0;
    for (int k=start; k<=end; k++){
    for (int l=start; l<=end; l++)
    {
        if ((int)(stride*j)+l >= 0 &&
            (int)(stride*j)+l < (int)widthPrev &&
            (int)(stride*i)+k >= 0 &&
            (int)(stride*i)+k < (int)heightPrev)
        {
            uint offsetPrev = (int)(stride*j)+l +
                (offsetStartPrev + (int)(stride*i)+k)*widthPrev;
            
            half outPrev = outsPrev[offsetPrev];
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

kernel void maxPoolBackwardHalf(
    const device half * delta,
    const device int * indicesMax,
    constant int * pStart,
    constant uint * pStride,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
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
    
    half tmp = 0.0;
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

kernel void adaptiveAvgPoolForward1Half(
    const device half * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device half * outs,
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
    
    half tmp = 0.0;
    for (uint k=0; k<nbElemsI; k++) {
    for (uint l=0; l<nbElemsJ; l++)
    {
        uint offsetPrev = startJ+l + (offsetStartPrev + startI+k) * widthPrev;
        tmp += outsPrev[offsetPrev];
    }}
    
    uint offset = j + (offsetStart + i) * width;
    outs[offset] = tmp / (half)nbElems;
}

kernel void adaptiveAvgPoolForward2Half(
    const device half * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device int * nbElems,
    device half * outs,
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
        half outPrev = outsPrev[offsetPrev];
        
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
        outs[offset] = float(outs[offset]) / nbElems[offset];
    }}
}

kernel void adaptiveAvgPoolBackward1Half(
    const device half * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device half * deltaPrev,
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
        half deltaCur = (float)delta[offset] / nbElems;
        
        for (uint k = 0; k < nbElemsI; k++){
        for (uint l = 0; l < nbElemsJ; l++)
        {
            uint offsetPrev = startJ+l +
                (offsetStartPrev + startI+k) * widthPrev;
            deltaPrev[offsetPrev] += deltaCur;
        }}
    }}
}

kernel void adaptiveAvgPoolBackward2Half(
    const device half * delta,
    const device int * nbElems,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device half * deltaPrev,
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

kernel void selectNeurons2DForwardHalf(
    const device half * outsPrev,
    constant uint * pTarget,
    constant uint * pNbNeurons,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device half * outs,
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

kernel void selectNeurons2DBackwardHalf(
    const device half * delta,
    constant uint * pTarget,
    constant uint * pNbNeurons,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
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
    
    half deltaCur = 0.0;
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

kernel void IRDFT2RGBForwardHalf(
    const device half * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * outs,
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

kernel void IRDFT2RGBBackwardHalf(
    const device half * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
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

kernel void decorrelateRGBForwardHalf(
    const device half * outsPrev,
    constant float * correlation,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * outs,
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
    
    half sum = 0.0;
    for (uint k=0; k<3; k++)
    {
        uint offsetStartPrev = (block * 3 + k + nbChannels * elem) * height;
        uint offsetPrev = j + (offsetStartPrev + i) * width;
        
        sum += outsPrev[offsetPrev] * correlation[res * 3 + k];
    }
    outs[offset] = sum;
}

kernel void decorrelateRGBBackwardHalf(
    const device half * delta,
    constant float * correlation,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
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
    
    half sum = 0.0;
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

kernel void linearScale2DForwardHalf(
    const device half * outsPrev,
    constant float * weights,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * outs,
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

kernel void linearScale2DBackwardHalf(
    const device half * delta,
    constant float * weights,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
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

kernel void setDataFTFrequences2DHalf(
    constant uint * pNbChannels,
    constant uint * pDimension,
    constant uint * pNbBatch,
    device half * outs,
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

kernel void pad2DForwardHalf(
    const device half * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pPadDimension,
    constant float * pPadValue,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbChannels;
    uint padDimension;
    half padValue;
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

kernel void pad2DBackwardHalf(
    const device half * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pPadDimension,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
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

kernel void crop2DForwardHalf(
    const device half * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pCropDimension,
    constant uint * pCropOffsets,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbChannels;
    uint cropDimension;
    uint offsetI, offsetJ;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch &&
        pCropDimension && pCropOffsets && outsPrev && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        cropDimension = *pCropDimension;
        offsetJ = pCropOffsets[0];
        offsetI = pCropOffsets[1];
        widthPrev = width + cropDimension;
        heightPrev = height + cropDimension;
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

kernel void crop2DBackwardHalf(
    const device half * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pCropDimension,
    constant uint * pCropOffsets,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint nbChannels;
    uint cropDimension;
    uint offsetI, offsetJ;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensions && pNbBatch &&
        pCropDimension && pCropOffsets && pDirty &&
        delta && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        cropDimension = *pCropDimension;
        offsetJ = pCropOffsets[0];
        offsetI = pCropOffsets[1];
        widthPrev = width + cropDimension;
        heightPrev = height + cropDimension;
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

kernel void resizeBilinearPadForwardHalf(
    const device half * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pDimensionsResize,
    constant uint * pPadDimensions,
    constant float * pPadValue,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint heightResize, widthResize;
    uint nbChannels;
    uint padStartI, padEndI;
    uint padStartJ, padEndJ;
    half padValue;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pDimensionsResize &&
        pPadDimensions && pPadValue && pNbBatch && outsPrev && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        widthResize = pDimensionsResize[0];
        heightResize = pDimensionsResize[1];
        padStartI = pPadDimensions[0];
        padEndI = pPadDimensions[1];
        padStartJ = pPadDimensions[2];
        padEndJ = pPadDimensions[3];
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
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    if (i < padStartI || i >= height - padEndI ||
        j < padStartJ || j >= width - padEndJ)
    {
        outs[offset] = padValue;
    }
    else
    {
        float I = i-padStartI;
        float J = j-padStartJ;
        
        float iPrev = I * ratioInOutI;
        float jPrev = J * ratioInOutJ;
        
        uint iPrevInf = floor(iPrev);
        uint iPrevSup = ceil(iPrev);
        uint jPrevInf = floor(jPrev);
        uint jPrevSup = ceil(jPrev);
        
        float iWeight = ratioInOutI * I - iPrevInf;
        float jWeight = ratioInOutJ * J - jPrevInf;
        
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

kernel void resizeBilinearPadBackwardHalf(
    const device half * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pDimensionsResize,
    constant uint * pPadDimensions,
    constant uint * pNbBatch,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint heightResize, widthResize;
    uint nbChannels;
    uint padStartI, padEndI;
    uint padStartJ, padEndJ;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pDimensionsResize &&
        pPadDimensions && pNbBatch && delta && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        widthResize = pDimensionsResize[0];
        heightResize = pDimensionsResize[1];
        padStartI = pPadDimensions[0];
        padEndI = pPadDimensions[1];
        padStartJ = pPadDimensions[2];
        padEndJ = pPadDimensions[3];
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
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
    
    float ratioInOutI = float(heightPrev - 1) / float(heightResize - 1);
    float ratioInOutJ = float(widthPrev - 1) / float(widthResize - 1);
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
    uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
    
    float kLow = (i-1.0) / ratioInOutI;
    float kHigh = (i+1.0) / ratioInOutI;
    float lLow = (j-1.0) / ratioInOutJ;
    float lHigh = (j+1.0) / ratioInOutJ;
    
    int kStart = ceil(kLow);
    int kEnd = floor(kHigh);
    int lStart = ceil(lLow);
    int lEnd = floor(lHigh);
    
    for (int k = kStart; k <= kEnd; k++) {
    for (int l = lStart; l <= lEnd; l++)
    {
        if (k >= 0 && k < (int)heightResize &&
            l >= 0 && l < (int)widthResize)
        {
            float kPrev = k * ratioInOutI;
            float lPrev = l * ratioInOutJ;
            
            uint kPrevInf = floor(kPrev);
            uint kPrevSup = ceil(kPrev);
            uint lPrevInf = floor(lPrev);
            uint lPrevSup = ceil(lPrev);
            
            float kWeight = ratioInOutI * k - kPrevInf;
            float lWeight = ratioInOutJ * l - lPrevInf;
            
            if (kPrevInf == i && lPrevInf == j)
            {
                uint offset = l+padStartJ +
                    (offsetStart + k+padStartI) * width;
                float deltaCur = delta[offset];
                
                deltaPrev[offsetPrev] +=
                    deltaCur * (1.0 - kWeight) * (1.0 - lWeight);
            }
            else if (kPrevInf == i && lPrevSup == j)
            {
                uint offset = l+padStartJ +
                    (offsetStart + k+padStartI) * width;
                float deltaCur = delta[offset];
                
                deltaPrev[offsetPrev] += deltaCur * (1.0 - kWeight) * lWeight;
            }
            else if (kPrevSup == i && lPrevInf == j)
            {
                uint offset = l+padStartJ +
                    (offsetStart + k+padStartI) * width;
                float deltaCur = delta[offset];
                
                deltaPrev[offsetPrev] += deltaCur * kWeight * (1.0 - lWeight);
            }
            else if (kPrevSup == i && lPrevSup == j)
            {
                uint offset = l+padStartJ +
                    (offsetStart + k+padStartI) * width;
                float deltaCur = delta[offset];
                
                deltaPrev[offsetPrev] += deltaCur * kWeight * lWeight;
            }
        }
    }}
}

kernel void rotate2DForwardHalf(
    const device half * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant float * pAngle,
    constant float * pPadValue,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    float angle, padValue;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pAngle && pPadValue && pNbBatch &&
        outsPrev && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        angle = *pAngle;
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
    
    float centerI = float(height - 1) / 2.0;
    float centerJ = float(width - 1) / 2.0;
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    float prevJ =
        cos(-angle) * (float(j) - centerJ) +
        sin(-angle) * (float(i) - centerI) + centerJ;
    float prevI =
        cos(-angle) * (float(i) - centerI) -
        sin(-angle) * (float(j) - centerJ) + centerI;
    
    if (round(prevJ) < 0 || round(prevJ) >= float(width) ||
        round(prevI) < 0 || round(prevI) >= float(height))
    {
        outs[offset] = padValue;
    }
    else
    {
        uint offsetPrev = round(prevJ) + (offsetStart + round(prevI)) * width;
        outs[offset] = outsPrev[offsetPrev];
    }
}

kernel void rotate2DBackwardHalf(
    const device half * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant float * pAngle,
    constant uint * pNbBatch,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    float angle;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pAngle && pNbBatch &&
        delta && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        angle = *pAngle;
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
    
    float centerI = float(height - 1) / 2.0;
    float centerJ = float(width - 1) / 2.0;
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offsetPrev = j + (offsetStart + i) * width;
    
    float rotJ =
        cos(angle) * (float(j) - centerJ) +
        sin(angle) * (float(i) - centerI) + centerJ;
    float rotI =
        cos(angle) * (float(i) - centerI) -
        sin(angle) * (float(j) - centerJ) + centerI;
    
    for (int k = floor(rotI); k <= ceil(rotI); k++) {
    for (int l = floor(rotJ); l <= ceil(rotJ); l++)
    {
        float prevL =
            cos(-angle) * (float(l) - centerJ) +
            sin(-angle) * (float(k) - centerI) + centerJ;
        float prevK =
            cos(-angle) * (float(k) - centerI) -
            sin(-angle) * (float(l) - centerJ) + centerI;
        
        if (round(prevL) == j && round(prevK) == i &&
            l >= 0 && l < (int)width && k >= 0 && k < (int)height)
        {
            uint offset = l + (offsetStart + k) * width;
            deltaPrev[offsetPrev] += delta[offset];
        }
    }}
}

kernel void resizeBilinearCropForwardHalf(
    const device half * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pDimensions2Resize,
    constant uint * pCropOffsets,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint height2Resize, width2Resize;
    uint offsetI, offsetJ;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pDimensions2Resize &&
        pCropOffsets && pNbBatch && outsPrev && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        width2Resize = pDimensions2Resize[0];
        height2Resize = pDimensions2Resize[1];
        offsetJ = pCropOffsets[0];
        offsetI = pCropOffsets[1];
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
    
    float ratioInOutI = float(height2Resize - 1) / float(height - 1);
    float ratioInOutJ = float(width2Resize - 1) / float(width - 1);
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    float iPrev = i * ratioInOutI;
    float jPrev = j * ratioInOutJ;
    
    uint iPrevInf = floor(iPrev);
    uint iPrevSup = ceil(iPrev);
    uint jPrevInf = floor(jPrev);
    uint jPrevSup = ceil(jPrev);
    
    float iWeight = ratioInOutI * i - iPrevInf;
    float jWeight = ratioInOutJ * j - jPrevInf;
    
    uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
    uint offsetPrev11 = jPrevInf+offsetJ +
        (offsetStartPrev + iPrevInf+offsetI) * widthPrev;
    uint offsetPrev12 = jPrevSup+offsetJ +
        (offsetStartPrev + iPrevInf+offsetI) * widthPrev;
    uint offsetPrev21 = jPrevInf+offsetJ +
        (offsetStartPrev + iPrevSup+offsetI) * widthPrev;
    uint offsetPrev22 = jPrevSup+offsetJ +
        (offsetStartPrev + iPrevSup+offsetI) * widthPrev;
    
    float out = outsPrev[offsetPrev11] * (1.0 - iWeight) * (1.0 - jWeight);
    out += outsPrev[offsetPrev12] * (1.0 - iWeight) * jWeight;
    out += outsPrev[offsetPrev21] * iWeight * (1.0 - jWeight);
    out += outsPrev[offsetPrev22] * iWeight * jWeight;
    
    outs[offset] = out;
}

kernel void resizeBilinearCropBackwardHalf(
    const device half * delta,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pDimensionsPrev,
    constant uint * pDimensions2Resize,
    constant uint * pCropOffsets,
    constant uint * pNbBatch,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint height2Resize, width2Resize;
    uint offsetI, offsetJ;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pDimensionsPrev && pDimensions2Resize &&
        pCropOffsets && pNbBatch && delta && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        width2Resize = pDimensions2Resize[0];
        height2Resize = pDimensions2Resize[1];
        offsetJ = pCropOffsets[0];
        offsetI = pCropOffsets[1];
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
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
    if (i < offsetI || i >= height2Resize + offsetI ||
        j < offsetJ || j >= width2Resize + offsetJ)
    {
        return ;
    }
    
    float ratioInOutI = float(height2Resize - 1) / float(height - 1);
    float ratioInOutJ = float(width2Resize - 1) / float(width - 1);
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offsetStartPrev = (depth + nbChannels * elem) * heightPrev;
    uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
    
    float I = i-offsetI;
    float J = j-offsetJ;
        
    float kLow = (I-1.0) / ratioInOutI;
    float kHigh = (I+1.0) / ratioInOutI;
    float lLow = (J-1.0) / ratioInOutJ;
    float lHigh = (J+1.0) / ratioInOutJ;
    
    int kStart = ceil(kLow);
    int kEnd = floor(kHigh);
    int lStart = ceil(lLow);
    int lEnd = floor(lHigh);
    
    for (int k = kStart; k <= kEnd; k++) {
    for (int l = lStart; l <= lEnd; l++)
    {
        if (k >= 0 && k < (int)height &&
            l >= 0 && l < (int)width)
        {
            float kPrev = k * ratioInOutI;
            float lPrev = l * ratioInOutJ;
            
            uint kPrevInf = floor(kPrev);
            uint kPrevSup = ceil(kPrev);
            uint lPrevInf = floor(lPrev);
            uint lPrevSup = ceil(lPrev);
            
            float kWeight = ratioInOutI * k - kPrevInf;
            float lWeight = ratioInOutJ * l - lPrevInf;
            
            if (kPrevInf == I && lPrevInf == J)
            {
                uint offset = l + (offsetStart + k) * width;
                float deltaCur = delta[offset];
                
                deltaPrev[offsetPrev] +=
                    deltaCur * (1.0 - kWeight) * (1.0 - lWeight);
            }
            else if (kPrevInf == I && lPrevSup == J)
            {
                uint offset = l + (offsetStart + k) * width;
                float deltaCur = delta[offset];
                
                deltaPrev[offsetPrev] += deltaCur * (1.0 - kWeight) * lWeight;
            }
            else if (kPrevSup == I && lPrevInf == J)
            {
                uint offset = l + (offsetStart + k) * width;
                float deltaCur = delta[offset];
                
                deltaPrev[offsetPrev] += deltaCur * kWeight * (1.0 - lWeight);
            }
            else if (kPrevSup == I && lPrevSup == J)
            {
                uint offset = l + (offsetStart + k) * width;
                float deltaCur = delta[offset];
                
                deltaPrev[offsetPrev] += deltaCur * kWeight * lWeight;
            }
        }
    }}
}

kernel void concat02DForwardHalf(
    const device half * outsPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    uint globalOffset;
    
    if (pGlobalOffset && pNbChannels && pDimensions &&
        pNbBatch && outsPrev && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        globalOffset = *pGlobalOffset;
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
    uint offsetStart = (depth + nbChannels * (globalOffset+elem)) * height;
    
    uint offsetPrev = j + (offsetStartPrev + i) * width;
    uint offset = j + (offsetStart + i) * width;
    
    outs[offset] = outsPrev[offsetPrev];
}

kernel void concat02DBackwardHalf(
    const device half * delta,
    constant uint * pGlobalOffset,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    uint globalOffset;
    uint dirty;
    
    if (pGlobalOffset && pNbChannels && pDimensions &&
        pNbBatch && pDirty && delta && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
        globalOffset = *pGlobalOffset;
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
    uint offsetStart = (depth + nbChannels * (globalOffset+elem)) * height;
    
    uint offsetPrev = j + (offsetStartPrev + i) * width;
    uint offset = j + (offsetStart + i) * width;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = delta[offset];
    }
    else
    {
        deltaPrev[offsetPrev] += delta[offset];
    }
}

kernel void concat12DForwardHalf(
    const device half * outsPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbChannels,
    constant uint * pNbChannelsPrev,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbChannelsPrev;
    uint nbBatch;
    uint globalOffset;
    
    if (pGlobalOffset && pNbChannels && pNbChannelsPrev && pDimensions &&
        pNbBatch && outsPrev && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        nbChannelsPrev = *pNbChannelsPrev;
        nbBatch = *pNbBatch;
        globalOffset = *pGlobalOffset;
    }
    else
        return ;
    
    uint depthPrev = id[0] / width;
    uint elem = id[1] / height;
    uint i = id[1] % height;
    uint j = id[0] % width;
    
    if (i * elem >= height * nbBatch ||
        j * depthPrev >= width * nbChannelsPrev)
    {
        return ;
    }
    
    uint offsetStartPrev = (depthPrev + nbChannelsPrev * elem) * height;
    uint offsetStart = (globalOffset+depthPrev + nbChannels * elem) * height;
    
    uint offsetPrev = j + (offsetStartPrev + i) * width;
    uint offset = j + (offsetStart + i) * width;
    
    outs[offset] = outsPrev[offsetPrev];
}

kernel void concat12DBackwardHalf(
    const device half * delta,
    constant uint * pGlobalOffset,
    constant uint * pNbChannels,
    constant uint * pNbChannelsPrev,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbChannelsPrev;
    uint nbBatch;
    uint globalOffset;
    uint dirty;
    
    if (pGlobalOffset && pNbChannels && pNbChannelsPrev && pDimensions &&
        pNbBatch && pDirty && delta && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        nbChannelsPrev = *pNbChannelsPrev;
        nbBatch = *pNbBatch;
        globalOffset = *pGlobalOffset;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depthPrev = id[0] / width;
    uint elem = id[1] / height;
    uint i = id[1] % height;
    uint j = id[0] % width;
    
    if (i * elem >= height * nbBatch ||
        j * depthPrev >= width * nbChannelsPrev)
    {
        return ;
    }
    
    uint offsetStartPrev = (depthPrev + nbChannelsPrev * elem) * height;
    uint offsetStart = (globalOffset+depthPrev + nbChannels * elem) * height;
    
    uint offsetPrev = j + (offsetStartPrev + i) * width;
    uint offset = j + (offsetStart + i) * width;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = delta[offset];
    }
    else
    {
        deltaPrev[offsetPrev] += delta[offset];
    }
}

kernel void constant2DForwardHalf(
    const device half * weights,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch && weights && outs)
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
    
    outs[offset] = weights[depth];
}

kernel void MSE2DLossHalf(
    const device half * outs,
    const device half * groundTruth,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * losses,
    uint id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch && outs && groundTruth && losses)
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
    
    half tmp = 0.0;
    for (uint depth=0; depth<nbChannels; depth++)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        
        for (uint i=0; i<height; i++) {
        for (uint j=0; j<width; j++)
        {
            uint offset = j + (offsetStart + i) * width;
            
            half out = outs[offset];
            half gt = groundTruth[offset];
            half diff = out - gt;
            
            tmp += diff * diff;
        }}
    }
    
    losses[elem] = tmp;
}

kernel void MSE2DLossDerivativeHalf(
    const device half * outs,
    const device half * groundTruth,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant float * pCoeff,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    half coeff;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensions && pNbBatch && pCoeff && pDirty &&
        outs && groundTruth && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        coeff = *pCoeff;
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
    
    half gt = groundTruth[offset];
    half out = outs[offset];
    half diff = out - gt;
    
    if (dirty)
    {
        deltaPrev[offset] = 2 * coeff * diff /
            half(nbBatch * nbChannels * height * width);
    }
    else
    {
        deltaPrev[offset] += 2 * coeff * diff /
            half(nbBatch * nbChannels * height * width);
    }
}

kernel void selfCorrelate2DForwardHalf(
    const device half * outsPrev,
    constant uint * pNbChannelsPrev,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint heightPrev, widthPrev;
    uint nbChannelsPrev;
    uint nbBatch;
    
    if (pNbChannelsPrev && pDimensionsPrev && pNbBatch &&
        outsPrev && outs)
    {
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        nbChannelsPrev = *pNbChannelsPrev;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint channel1 = id[0] / nbChannelsPrev;
    uint channel2 = id[0] % nbChannelsPrev;
    uint elem = id[1];
    
    if (channel1 * channel2 >= nbChannelsPrev * nbChannelsPrev ||
        elem >= nbBatch)
    {
        return ;
    }
        
    uint offsetStart1 = (channel1 + nbChannelsPrev * elem) * heightPrev;
    uint offsetStart2 = (channel2 + nbChannelsPrev * elem) * heightPrev;
    
    half correlation = 0.0;
    for (uint i=0; i<heightPrev; i++){
    for (uint j=0; j<widthPrev; j++)
    {
        uint offset1 = j + (offsetStart1 + i) * widthPrev;
        uint offset2 = j + (offsetStart2 + i) * widthPrev;
        
        correlation += outsPrev[offset1] * outsPrev[offset2];
    }}
    
    uint offset = channel2 +
        (elem * nbChannelsPrev + channel1) * nbChannelsPrev;
    outs[offset] = correlation;
}

kernel void selfCorrelate2DBackwardHalf(
    const device half * delta,
    const device half * outsPrev,
    constant uint * pNbChannelsPrev,
    constant uint * pDimensionsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint heightPrev, widthPrev;
    uint nbChannelsPrev;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannelsPrev && pDimensionsPrev && pNbBatch && pDirty &&
        delta && outsPrev && deltaPrev)
    {
        widthPrev = pDimensionsPrev[0];
        heightPrev = pDimensionsPrev[1];
        nbChannelsPrev = *pNbChannelsPrev;
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
        j * depthPrev >= widthPrev * nbChannelsPrev)
    {
        return ;
    }
    
    half correlation = 0.0;
    for (uint col=0; col<nbChannelsPrev; col++)
    {
        uint offsetStartPrev = (col + nbChannelsPrev * elem) * heightPrev;
        uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
        uint offset = col +
            (elem * nbChannelsPrev + depthPrev) * nbChannelsPrev;
        
        correlation += delta[offset] * outsPrev[offsetPrev];
    }
    for (uint row=0; row<nbChannelsPrev; row++)
    {
        uint offsetStartPrev = (row + nbChannelsPrev * elem) * heightPrev;
        uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
        uint offset = depthPrev +
            (elem * nbChannelsPrev + row) * nbChannelsPrev;
        
        correlation += delta[offset] * outsPrev[offsetPrev];
    }
    
    uint offsetStartPrev = (depthPrev + nbChannelsPrev * elem) * heightPrev;
    uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = correlation;
    }
    else
    {
        deltaPrev[offsetPrev] += correlation;
    }
}

kernel void normalize12DForwardHalf(
    const device half * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch &&
        outsPrev && outs)
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
    
    float norm = 0.0;
    for (uint depth1=0; depth1<nbChannels; depth1++)
    {
        uint offsetStart1 = (depth1 + nbChannels * elem) * height;
        uint offset1 = j + (offsetStart1 + i) * width;
        
        float outPrev1 = outsPrev[offset1];
        norm += outPrev1 * outPrev1;
    }
    norm = sqrt(norm);
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    float outPrev = outsPrev[offset];
    outs[offset] = outPrev / max(norm, 1e-12);
}

kernel void normalize12DBackwardHalf(
    const device half * delta,
    const device half * outsPrev,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensions && pNbBatch && pDirty &&
        delta && outsPrev && deltaPrev)
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
    
    float normTmp = 0.0;
    for (uint depth1=0; depth1<nbChannels; depth1++)
    {
        uint offsetStart1 = (depth1 + nbChannels * elem) * height;
        uint offset1 = j + (offsetStart1 + i) * width;
        
        float outPrev1 = outsPrev[offset1];
        normTmp += outPrev1 * outPrev1;
    }
    float norm = sqrt(normTmp);
    normTmp = pow(norm, 3);
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    float deltaCur = delta[offset];
    float outPrev = outsPrev[offset];
    
    float newValue = 0.0;
    if (norm > 1e-12)
    {
        for (uint depth1=0; depth1<nbChannels; depth1++)
        {
            uint offsetStart1 = (depth1 + nbChannels * elem) * height;
            uint offset1 = j + (offsetStart1 + i) * width;
            
            float deltaCur1 = delta[offset1];
            float outPrev1 = outsPrev[offset1];
            
            newValue -= outPrev1 * outPrev / normTmp * deltaCur1;
        }
        newValue += deltaCur / norm;
    }
    else
    {
        newValue = deltaCur / 1e-12;
    }
    
    if (dirty)
    {
        deltaPrev[offset] = newValue;
    }
    else
    {
        deltaPrev[offset] += newValue;
    }
}

kernel void computeSquaredNorm122DHalf(
     const device half * outsPrev,
     constant uint * pNbChannels,
     constant uint * pDimensions,
     constant uint * pNbThreadgroups,
     constant uint * pNbBatch,
     device half * squaredNorms,
     uint2 groupId [[ threadgroup_position_in_grid ]],
     uint2 threadId [[ thread_position_in_threadgroup ]],
     uint2 id [[ thread_position_in_grid ]])
{
    constexpr uint threadsPerThreadgroup = 64;
    threadgroup half normShared[threadsPerThreadgroup];
    
    uint height, width;
    uint nbChannels;
    uint nbThreadgroups;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbThreadgroups && pNbBatch &&
        outsPrev && squaredNorms)
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
    uint remains = id[0];
    uint depth = remains / (height * width);
    remains = remains % (height * width);
    uint i = remains / width;
    uint j = remains % width;
    
    if (depth * i * j >= nbChannels * height * width ||
        elem >= nbBatch)
    {
        return ;
    }
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    half outPrev = outsPrev[offset];
    normShared[threadId[0]] = outPrev * outPrev;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride=threadsPerThreadgroup/2; stride>0; stride>>=1)
    {
        uint index = threadId[0] + groupId[0] * threadsPerThreadgroup;
        if (threadId[0] < stride &&
            (index + stride) < nbChannels * height * width)
        {
            normShared[threadId[0]] += normShared[threadId[0] + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (threadId[0] == 0)
    {
        uint offset = elem * nbThreadgroups + groupId[0];
        squaredNorms[offset] = normShared[0];
    }
}

kernel void normalize122DForwardHalf(
    const device half * outsPrev,
    const device half * squaredNorms,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbThreadgroups,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbThreadgroups;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbThreadgroups && pNbBatch &&
        outsPrev && squaredNorms && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        nbThreadgroups = *pNbThreadgroups;
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
    
    float norm = sqrt(squaredNorms[elem]);
    float outPrev = outsPrev[offset];
    
    outs[offset] = outPrev / max(norm, 1e-12);
}

kernel void computeDeltaTmp122DHalf(
     const device half * delta,
     const device half * outsPrev,
     const device half * squaredNorms,
     constant uint * pNbChannels,
     constant uint * pDimensions,
     constant uint * pNbThreadgroups,
     constant uint * pNbBatch,
     device half * deltaTmp,
     uint2 groupId [[ threadgroup_position_in_grid ]],
     uint2 threadId [[ thread_position_in_threadgroup ]],
     uint2 id [[ thread_position_in_grid ]])
{
    constexpr uint threadsPerThreadgroup = 64;
    threadgroup half deltaShared[threadsPerThreadgroup];
    
    uint height, width;
    uint nbChannels;
    uint nbThreadgroups;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbThreadgroups && pNbBatch &&
        delta && outsPrev && squaredNorms && deltaTmp)
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
    uint remains = id[0];
    uint depth = remains / (height * width);
    remains = remains % (height * width);
    uint i = remains / width;
    uint j = remains % width;
    
    if (depth * i * j >= nbChannels * height * width ||
        elem >= nbBatch)
    {
        return ;
    }
    
    float norm = sqrt(squaredNorms[elem]);
    if (norm > 1e-12)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        uint offset = j + (offsetStart + i) * width;
        
        half deltaCur = delta[offset];
        half outPrev = outsPrev[offset];
        
        deltaShared[threadId[0]] = outPrev * deltaCur;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint stride=threadsPerThreadgroup/2; stride>0; stride>>=1)
        {
            uint index = threadId[0] + groupId[0] * threadsPerThreadgroup;
            if (threadId[0] < stride &&
                (index + stride) < nbChannels * height * width)
            {
                deltaShared[threadId[0]] += deltaShared[threadId[0] + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        if (threadId[0] == 0)
        {
            uint offset = elem * nbThreadgroups + groupId[0];
            deltaTmp[offset] = deltaShared[0];
        }
    }
}

kernel void normalize122DBackwardHalf(
    const device half * delta,
    const device half * outsPrev,
    const device half * squaredNorms,
    const device half * deltaTmp,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbThreadgroups,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbThreadgroups;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensions && pNbThreadgroups && pNbBatch && pDirty &&
        delta && outsPrev && squaredNorms && deltaTmp && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        nbThreadgroups = *pNbThreadgroups;
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
    
    float norm = sqrt(squaredNorms[elem]);
    float deltaCurTmp = deltaTmp[elem];
    float normTmp = pow(norm, 3);
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset = j + (offsetStart + i) * width;
    
    float outPrev = outsPrev[offset];
    float deltaCur = delta[offset];
    
    float newValue = 0.0;
    if (norm > 1e-12)
    {
        newValue = deltaCur / norm - deltaCurTmp * outPrev / normTmp;
    }
    else
    {
        newValue = deltaCur / 1e-12;
    }
    
    if (dirty)
    {
        deltaPrev[offset] = newValue;
    }
    else
    {
        deltaPrev[offset] += newValue;
    }
}

kernel void similarBatchError2DLossHalf(
    const device half * outs,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * losses,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch && outs && losses)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint elem1 = id[0];
    uint elem2 = id[1];
    
    if (elem1 >= nbBatch || elem2 >= nbBatch)
    {
        return ;
    }
    
    if (elem1 == elem2)
    {
        losses[elem2 + nbBatch * elem1] = 0.0;
    }
    else
    {
        float sum = 0.0;
        for (uint i=0; i<height; i++) {
        for (uint j=0; j<width; j++)
        {
            uint offset1 = j + (elem1 * height + i) * width;
            uint offset2 = j + (elem2 * height + i) * width;
        
            sum += outs[offset1] * outs[offset2];
        }}
        losses[elem2 + nbBatch * elem1] = sum;
    }
}

kernel void similarBatchError2DLossDerivativeHalf(
    const device half * outs,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant float * pCoeff,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    float coeff;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensions && pNbBatch && pCoeff && pDirty &&
        outs && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint i = id[0] / width;
    uint j = id[0] % width;
    uint elem = id[1];
    
    if (i * j >= width * height || elem >= nbBatch)
    {
        return ;
    }
    
    float sum = 0.0;
    for (uint elem1=0; elem1<nbBatch; elem1++)
    {
        if (elem1 == elem)
        {
            continue;
        }
        uint offset1 = j + (elem1 * height + i) * width;
        sum += 2 * outs[offset1];
    }
    
    uint offset = j + (elem * height + i) * width;
    
    if (dirty)
    {
        deltaPrev[offset] = coeff / nbBatch * sum;
    }
    else
    {
        deltaPrev[offset] += coeff / nbBatch * sum;
    }
}

kernel void similarError2DLossDerivativeHalf(
    const device half * outs,
    constant uint * pGlobalOffset,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant float * pCoeff,
    constant uint * pNbBatch,
    constant uint * pNbBatchPrev,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    half coeff;
    uint globalOffset;
    uint nbBatch, nbBatchPrev;
    uint dirty;
    
    if (pGlobalOffset && pNbChannels && pDimensions &&
        pNbBatch && pNbBatchPrev && pCoeff && pDirty &&
        outs && deltaPrev)
    {
        globalOffset = *pGlobalOffset;
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
        nbBatchPrev = *pNbBatchPrev;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint i = id[0] / width;
    uint j = id[0] % width;
    uint elem = id[1];
    
    if (i * j >= width * height || elem >= nbBatchPrev)
    {
        return ;
    }
    
    half sum = 0.0;
    for (uint elem1=0; elem1<nbBatch; elem1++)
    {
        if (elem1 == elem+globalOffset)
        {
            continue;
        }
        uint offset1 = j + (elem1 * height + i) * width;
        sum += 2 * outs[offset1];
    }
    
    uint offset = j + (elem * height + i) * width;
    
    if (dirty)
    {
        deltaPrev[offset] = coeff / nbBatch * sum;
    }
    else
    {
        deltaPrev[offset] += coeff / nbBatch * sum;
    }
}

kernel void flipHorizontal2DForwardHalf(
    const device half * outsPrev,
    constant uint * pDoFlip,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint doFlip;
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    
    if (pDoFlip && pNbChannels && pDimensions && pNbBatch &&
        outsPrev && outs)
    {
        doFlip = *pDoFlip;
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
    uint offset1 = j + (offsetStart + i) * width;
    uint offset2 = offset1;
    if (doFlip)
    {
        offset2 = width-1-j + (offsetStart + i) * width;
    }
    
    outs[offset1] = outsPrev[offset2];
}

kernel void flipHorizontal2DBackwardHalf(
    const device half * delta,
    constant uint * pDoFlip,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint doFlip;
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    uint dirty;
    
    if (pDoFlip && pNbChannels && pDimensions && pNbBatch && pDirty &&
        delta && deltaPrev)
    {
        doFlip = *pDoFlip;
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
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset1 = j + (offsetStart + i) * width;
    uint offset2 = offset1;
    if (doFlip)
    {
        offset2 = width-1-j + (offsetStart + i) * width;
    }
    
    if (dirty)
    {
        deltaPrev[offset1] = delta[offset2];
    }
    else
    {
        deltaPrev[offset1] += delta[offset2];
    }
}

kernel void flipVertical2DForwardHalf(
    const device half * outsPrev,
    constant uint * pDoFlip,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint doFlip;
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    
    if (pDoFlip && pNbChannels && pDimensions && pNbBatch &&
        outsPrev && outs)
    {
        doFlip = *pDoFlip;
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
    uint offset1 = j + (offsetStart + i) * width;
    uint offset2 = offset1;
    if (doFlip)
    {
        offset2 = j + (offsetStart + height-1-i) * width;
    }
    
    outs[offset1] = outsPrev[offset2];
}

kernel void flipVertical2DBackwardHalf(
    const device half * delta,
    constant uint * pDoFlip,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint doFlip;
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    uint dirty;
    
    if (pDoFlip && pNbChannels && pDimensions && pNbBatch && pDirty &&
        delta && deltaPrev)
    {
        doFlip = *pDoFlip;
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
    
    uint offsetStart = (depth + nbChannels * elem) * height;
    uint offset1 = j + (offsetStart + i) * width;
    uint offset2 = offset1;
    if (doFlip)
    {
        offset2 = j + (offsetStart + height-1-i) * width;
    }
    
    if (dirty)
    {
        deltaPrev[offset1] = delta[offset2];
    }
    else
    {
        deltaPrev[offset1] += delta[offset2];
    }
}

kernel void colorJitterHSVForwardHalf(
    const device half * outsPrev,
    constant float * pNoise,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    float noiseH, noiseS, noiseV;
    uint height, width;
    uint nbBatch;
    
    if (pNoise && pDimensions && pNbBatch && outsPrev && outs)
    {
        noiseH = pNoise[0];
        noiseS = pNoise[1];
        noiseV = pNoise[2];
        width = pDimensions[0];
        height = pDimensions[1];
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint elem = id[1];
    uint row = id[0] / width;
    uint col = id[0] % width;
    
    if (row * col >= height * width ||
        elem >= nbBatch)
    {
        return ;
    }
        
    uint offsetStartR = (0 + 3 * elem) * height;
    uint offsetStartG = (1 + 3 * elem) * height;
    uint offsetStartB = (2 + 3 * elem) * height;
    
    uint offsetR = col + (offsetStartR + row) * width;
    uint offsetG = col + (offsetStartG + row) * width;
    uint offsetB = col + (offsetStartB + row) * width;
    
    float r = outsPrev[offsetR];
    float g = outsPrev[offsetG];
    float b = outsPrev[offsetB];
    
    float maxValue = max(max(r, g), b);
    float minValue = min(min(r, g), b);
    float delta = maxValue - minValue;
    
    float h;
    if (delta == 0)
    {
        h = 0.0;
    }
    else if (maxValue == r)
    {
        h = (g - b) / delta;
    }
    else if (maxValue == g)
    {
        h = (g - b) / delta + 2.0;
    }
    else
    {
        h = (g - b) / delta + 4.0;
    }
    h *= 60.0;
    
    float s = 0.0;
    if (maxValue != 0)
    {
        s = delta / maxValue;
    }
    
    float v = maxValue;
    
    h += noiseH; h = max(h, 0.0); h = min(h, 360.0);
    s += noiseS; s = max(s, 0.0); s = min(s, 1.0);
    v += noiseV; v = max(v, 0.0); v = min(v, 1.0);
    
    if (s == 0.0)
    {
        r = v; g = v; b = v;
    }
    
    float angle = h;
    float sector = angle / 60; // Sector
    float i = floor(sector);
    float f = sector - i; // Factorial part of h
    
    float p = v * (1 - s);
    float q = v * (1 - (s * f));
    float t = v * (1 - (s * (1 - f)));
    
    if (i == 0)
    {
        r = v; g = t; b = p;
    }
    else if (i == 1)
    {
        r = q; g = v; b = p;
    }
    else if (i == 2)
    {
        r = p; g = v; b = t;
    }
    else if (i == 3)
    {
        r = p; g = q; b = v;
    }
    else if (i == 4)
    {
        r = t; g = p; b = v;
    }
    else
    {
        r = v; g = p; b = q;
    }
    
    outs[offsetR] = r;
    outs[offsetG] = g;
    outs[offsetB] = b;
}

kernel void BCE2DLossHalf(
    const device half * outs,
    const device half * groundTruth,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * losses,
    uint id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch && outs && groundTruth && losses)
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
    
    half tmp = 0.0;
    for (uint depth=0; depth<nbChannels; depth++)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        
        for (uint i=0; i<height; i++) {
        for (uint j=0; j<width; j++)
        {
            uint offset = j + (offsetStart + i) * width;
            
            half out = outs[offset];
            half gt = groundTruth[offset];
            half tmp1 = log(out);
            half tmp2 = log(1 - out);
            
            tmp -= (gt * tmp1 + (1 - gt) * tmp2);
        }}
    }
    
    losses[elem] = tmp;
}

kernel void BCE2DLossDerivativeHalf(
    const device half * outs,
    const device half * groundTruth,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant float * pCoeff,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    half coeff;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensions && pNbBatch && pCoeff && pDirty &&
        outs && groundTruth && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        coeff = *pCoeff;
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
    
    half gt = groundTruth[offset];
    half out = outs[offset];
    half derivative = 0.0;
    
    if (gt == 1.0)
    {
        derivative = -1 / out;
    }
    else if (gt == 0.0)
    {
        derivative = 1 / (1 - out);
    }
    
    if (dirty)
    {
        deltaPrev[offset] = coeff * derivative /
            half(nbBatch * nbChannels * height * width);
    }
    else
    {
        deltaPrev[offset] += coeff * derivative /
            half(nbBatch * nbChannels * height * width);
    }
}

kernel void BCESigmoid2DLossHalf(
    const device half * outs,
    const device half * groundTruth,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant uint * pNbBatch,
    device half * losses,
    uint id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    uint nbBatch;
    
    if (pNbChannels && pDimensions && pNbBatch && outs && groundTruth && losses)
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
    
    half tmp = 0.0;
    for (uint depth=0; depth<nbChannels; depth++)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        
        for (uint i=0; i<height; i++) {
        for (uint j=0; j<width; j++)
        {
            uint offset = j + (offsetStart + i) * width;
            
            half out = outs[offset];
            half gt = groundTruth[offset];
            half value;
            
            if (out > 0)
            {
                value = (1 - gt) * out;
                value += log(1 + exp(-out));
            }
            else
            {
                value = -out * gt;
                value += log(exp(out) + 1);
            }
            
            tmp += value;
        }}
    }
    
    losses[elem] = tmp;
}

kernel void BCESigmoid2DLossDerivativeHalf(
    const device half * outs,
    const device half * groundTruth,
    constant uint * pNbChannels,
    constant uint * pDimensions,
    constant float * pCoeff,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbChannels;
    half coeff;
    uint nbBatch;
    uint dirty;
    
    if (pNbChannels && pDimensions && pNbBatch && pCoeff && pDirty &&
        outs && groundTruth && deltaPrev)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannels = *pNbChannels;
        coeff = *pCoeff;
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
    
    half gt = groundTruth[offset];
    half out = outs[offset];
    half value;
    
    if (out >= 0)
    {
        value = 1.0 / (1.0 + exp(-out));
    }
    else
    {
        value = exp(out) / (1.0 + exp(out));
    }
    
    if (dirty)
    {
        deltaPrev[offset] = coeff * (value - gt) /
            half(nbBatch * nbChannels * height * width);
    }
    else
    {
        deltaPrev[offset] += coeff * (value - gt) /
            half(nbBatch * nbChannels * height * width);
    }
}

kernel void layerCAM2DForwardHalf(
    const device half * outsPrev,
    const device half * deltaPrev,
    constant uint * pNbChannelsPrev,
    constant uint * pDimensions,
    constant uint * pKeepPositive,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint nbBatch;
    uint nbChannelsPrev;
    uint keepPositive;
    
    if (pNbChannelsPrev && pDimensions && pKeepPositive && pNbBatch &&
        outsPrev && outs)
    {
        width = pDimensions[0];
        height = pDimensions[1];
        nbChannelsPrev = *pNbChannelsPrev;
        keepPositive = *pKeepPositive;
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
        
    float sum = 0.0;
    for (uint depthPrev=0; depthPrev<nbChannelsPrev; depthPrev++)
    {
        uint offsetStartPrev = (depthPrev + nbChannelsPrev * elem) * height;
        uint offsetPrev = j + (offsetStartPrev + i) * width;
        
        float deltaPrevTmp = deltaPrev[offsetPrev];
        if (!keepPositive)
        {
            deltaPrevTmp = -deltaPrevTmp;
        }
        if (deltaPrevTmp < 0)
        {
            deltaPrevTmp = 0.0;
        }
        
        sum += deltaPrevTmp * outsPrev[offsetPrev];
    }
    
    uint offset = j + (elem * height + i) * width;
    outs[offset] = sum;
}
