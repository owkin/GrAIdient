//
// Convolution.metal
// GrAIdient
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
    int offI, offJ;
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
        offI = pStart[4];
        offJ = pStart[5];
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
            if ((int)(stride*j)+l-offJ >= 0 &&
                (int)(stride*j)+l-offJ < (int)widthPrev &&
                (int)(stride*i)+k-offI >= 0 &&
                (int)(stride*i)+k-offI < (int)heightPrev)
            {
                uint offsetPrev = (int)(stride*j)+l-offJ +
                    (offsetStartPrev + (int)(stride*i)+k-offI)*widthPrev;
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

kernel void conv16Forward(
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
    int offI, offJ;
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
        offI = pStart[4];
        offJ = pStart[5];
        stride = pStride[0];
    }
    else
        return ;
    
    uint coeff = 16;
    uint depth = id[0] / width;
    uint elem = id[1] / height;
    uint i = id[1] % height;
    uint j = id[0] % width;
    
    if (i * elem >= height * nbBatch ||
        j * depth * coeff >= width * nbChannels)
    {
        return ;
    }
    
    float tmp[16] = {0};
    for (uint depthPrev=0; depthPrev<nbChannelsPrev; depthPrev++)
    {
        uint offsetStartPrev =
            (depthPrev + nbChannelsPrev*elem) * heightPrev;
        
        for (int k=startI; k<=endI; k++){
        for (int l=startJ; l<=endJ; l++)
        {
            if ((int)(stride*j)+l-offJ >= 0 &&
                (int)(stride*j)+l-offJ < (int)widthPrev &&
                (int)(stride*i)+k-offI >= 0 &&
                (int)(stride*i)+k-offI < (int)heightPrev)
            {
                uint offsetPrev = (int)(stride*j)+l-offJ +
                    (offsetStartPrev + (int)(stride*i)+k-offI)*widthPrev;
                float outPrev = outsPrev[offsetPrev];
                
                for (uint c=0; c<coeff; c++)
                {
                    uint offsetStartWeights = weightHeight *
                        (depthPrev + nbChannelsPrev * (depth*coeff+c));
                    uint offsetWeights = l-startJ +
                        (offsetStartWeights + k-startI) * weightWidth;
                    float w = weights[offsetWeights];
                    
                    tmp[c] += outPrev * w;
                }
            }
        }}
    }
    
    for (uint c=0; c<coeff; c++)
    {
        uint offsetStart = ((depth*coeff+c) + nbChannels * elem) * height;
        uint offset = j + (offsetStart + i) * width;
        outs[offset] = tmp[c] + biases[depth*coeff+c];
    }
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
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint weightHeight, weightWidth;
    uint nbChannels;
    uint nbChannelsPrev;
    int startI, startJ;
    int endI, endJ;
    int offI, offJ;
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
        offI = pStart[4];
        offJ = pStart[5];
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
            if ((i-k+offI) % stride == 0 && (j-l+offJ) % stride == 0)
            {
                int i1 = (i-k+offI) / stride;
                int j1 = (j-l+offJ) / stride;
                
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

kernel void conv16Backward(
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
    int offI, offJ;
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
        offI = pStart[4];
        offJ = pStart[5];
        stride = pStride[0];
        dirty = *pDirty;
    }
    else
        return ;
    
    uint coeff = 16;
    uint depthPrev = id[0] / widthPrev;
    uint elem = id[1] / heightPrev;
    uint i = id[1] % heightPrev;
    uint j = id[0] % widthPrev;
    
    if (i * elem >= heightPrev * nbBatch ||
        j * depthPrev * coeff >= widthPrev * nbChannelsPrev)
    {
        return ;
    }
    
    float tmp[16] = {0};
    for (uint depth=0; depth<nbChannels; depth++)
    {
        uint offsetStart = (depth + nbChannels * elem) * height;
        
        for (int k=startI; k<=endI; k++){
        for (int l=startJ; l<=endJ; l++)
        {
            if ((i-k+offI) % stride == 0 && (j-l+offJ) % stride == 0)
            {
                int i1 = (i-k+offI) / stride;
                int j1 = (j-l+offJ) / stride;
                
                if (j1 >= 0 && j1 < (int)width &&
                    i1 >= 0 && i1 < (int)height)
                {
                    uint offset = j1 + (offsetStart + i1) * width;
                    float deltaCur = delta[offset];
                    
                    for (uint c=0; c<coeff; c++)
                    {
                        uint offsetStartWeights = weightHeight *
                            ((depthPrev*coeff+c) + nbChannelsPrev * depth);
                        uint offsetWeights = l-startJ +
                            (offsetStartWeights + k-startI) * weightWidth;
                        float w = weights[offsetWeights];
                        
                        tmp[c] += deltaCur * w;
                    }
                }
            }
        }}
    }
    
    for (uint c=0; c<coeff; c++)
    {
        uint offsetStartPrev = heightPrev *
            ((depthPrev*coeff+c) + nbChannelsPrev * elem);
        uint offsetPrev = j + (offsetStartPrev + i) * widthPrev;
        
        if (dirty)
        {
            deltaPrev[offsetPrev] = tmp[c];
        }
        else
        {
            deltaPrev[offsetPrev] += tmp[c];
        }
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
    int offI, offJ;
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
        offI = pStart[4];
        offJ = pStart[5];
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
            if ((int)(stride*l)+j-offJ >= 0 &&
                (int)(stride*l)+j-offJ < (int)widthPrev &&
                (int)(stride*k)+i-offI >= 0 &&
                (int)(stride*k)+i-offI < (int)heightPrev)
            {
                uint offset = l + (offsetStart + k) * width;
                float deltaCur = delta[offset];
                
                uint offsetPrev = (int)(stride*l)+j-offJ +
                    (offsetStartPrev + (int)(stride*k)+i-offI)*widthPrev;
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

kernel void conv34BatchDerWeights(
    const device float4 * outsPrev,
    const device float4 * delta,
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
    int offI, offJ;
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
        offI = pStart[4];
        offJ = pStart[5];
        stride = pStride[0];
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint depth = id[0];
    uint depthPrev = id[1];
    
    if (id[0] >= nbChannels ||
        id[1] >= nbChannelsPrev)
    {
        return ;
    }
    
    float tmp[8] = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offsetStart =
            (depth + nbChannels * elem) * height;
        uint offsetStartPrev =
            (depthPrev + nbChannelsPrev * elem) * heightPrev;
        
        for (uint k=0; k<height/2; k++){
        for (uint l=0; l<width/4; l++)
        {
            uint offset4 = (l*4 + (offsetStart + k*2) * width) / 4;
            uint offset7 = (l*4 + (offsetStart + k*2+1) * width) / 4;
            float4 delta4 = delta[offset4];
            float4 delta7 = delta[offset7];
            
            if (k > 0 && l > 0)
            {
                uint offsetPrev0 =
                    ((l-1)*4 + (offsetStartPrev + (k-1)*2) * widthPrev) / 4;
                float outPrev0 = outsPrev[offsetPrev0][3];
                
                tmp[0] += outPrev0 * delta4[0];
            }
            if (k > 0)
            {
                uint offsetPrev1 =
                    (l*4 + (offsetStartPrev + (k-1)*2) * widthPrev) / 4;
                float4 outPrev1 = outsPrev[offsetPrev1];
                
                tmp[0] += outPrev1[0] * delta4[1];
                tmp[0] += outPrev1[1] * delta4[2];
                tmp[0] += outPrev1[2] * delta4[3];
                
                float4 sum = outPrev1 * delta4;
                tmp[1] += sum[0] + sum[1] + sum[2] + sum[3];
                
                tmp[2] += outPrev1[1] * delta4[0];
                tmp[2] += outPrev1[2] * delta4[1];
                tmp[2] += outPrev1[3] * delta4[2];
            }
            if (k > 0 && l < width/4 - 1)
            {
                uint offsetPrev2 =
                    ((l+1)*4 + (offsetStartPrev + (k-1)*2) * widthPrev) / 4;
                float outPrev2 = outsPrev[offsetPrev2][0];
                
                tmp[2] += outPrev2 * delta4[3];
            }
            
            if (l > 0)
            {
                uint offsetPrev3 =
                    ((l-1)*4 + (offsetStartPrev + k*2) * widthPrev) / 4;
                uint offsetPrev6 =
                    ((l-1)*4 + (offsetStartPrev + k*2+1) * widthPrev) / 4;
                float outPrev3 = outsPrev[offsetPrev3][3];
                float outPrev6 = outsPrev[offsetPrev6][3];
                
                tmp[0] += outPrev3 * delta7[0];
                tmp[3] += outPrev3 * delta4[0];
                tmp[3] += outPrev6 * delta7[0];
                tmp[6] += outPrev6 * delta4[0];
            }
            
            uint offsetPrev4 =
                (l*4 + (offsetStartPrev + k*2) * widthPrev) / 4;
            uint offsetPrev7 =
                (l*4 + (offsetStartPrev + k*2+1) * widthPrev) / 4;
            float4 outPrev4 = outsPrev[offsetPrev4];
            float4 outPrev7 = outsPrev[offsetPrev7];
            
            tmp[0] += outPrev4[0] * delta7[1];
            tmp[0] += outPrev4[1] * delta7[2];
            tmp[0] += outPrev4[2] * delta7[3];
            
            float4 sum = outPrev4 * delta7;
            tmp[1] += sum[0] + sum[1] + sum[2] + sum[3];
            
            tmp[2] += outPrev4[1] * delta7[0];
            tmp[2] += outPrev4[2] * delta7[1];
            tmp[2] += outPrev4[3] * delta7[2];
            
            tmp[3] += outPrev4[0] * delta4[1];
            tmp[3] += outPrev4[1] * delta4[2];
            tmp[3] += outPrev4[2] * delta4[3];
            tmp[3] += outPrev7[0] * delta7[1];
            tmp[3] += outPrev7[1] * delta7[2];
            tmp[3] += outPrev7[2] * delta7[3];
            
            sum = outPrev4 * delta4;
            tmp[4] += sum[0] + sum[1] + sum[2] + sum[3];
            sum = outPrev7 * delta7;
            tmp[4] += sum[0] + sum[1] + sum[2] + sum[3];
            
            tmp[5] += outPrev4[1] * delta4[0];
            tmp[5] += outPrev4[2] * delta4[1];
            tmp[5] += outPrev4[3] * detta4[2];
            tmp[5] += outPrev7[1] * delta7[0];
            tmp[5] += outPrev7[2] * delta7[1];
            tmp[5] += outPrev7[3] * delta7[2];
            
            tmp[6] += outPrev7[0] * delta4[1];
            tmp[6] += outPrev7[1] * delta4[2];
            tmp[6] += outPrev7[2] * delta4[3];
            
            sum = outPrev7 * delta4;
            tmp[7] += sum[0] + sum[1] + sum[2] + sum[3];
            
            tmp[8] += outPrev7[1] * delta4[0];
            tmp[8] += outPrev7[2] * delta4[1];
            tmp[8] += outPrev7[3] * delta4[2];
            
            if (l < width/4 - 1)
            {
                uint offsetPrev5 =
                    ((l+1)*4 + (offsetStartPrev + k*2) * widthPrev) / 4;
                uint offsetPrev8 =
                    ((l+1)*4 + (offsetStartPrev + k*2+1) * widthPrev) / 4;
                float outPrev5 = outsPrev[offsetPrev5][0];
                float outPrev8 = outsPrev[offsetPrev8][0];
                
                tmp[2] += outPrev5 * delta7[3];
                tmp[5] += outPrev5 * delta4[3];
                tmp[5] += outPrev8 * delta7[3];
                tmp[8] += outPrev8 * delta4[3];
            }
            
            if (k < height/2 - 1 && l > 0)
            {
                uint offsetPrev9 =
                    ((l-1)*4 + (offsetStartPrev + (k+1)*2) * widthPrev) / 4;
                float outPrev9 = outsPrev[offsetPrev9][3];
                
                tmp[6] += outPrev9 * delta7[0];
            }
            if (k < height/2 - 1)
            {
                uint offsetPrev10 =
                    (l*4 + (offsetStartPrev + (k+1)*2) * widthPrev) / 4;
                float4 outPrev10 = outsPrev[offsetPrev10];
                
                tmp[6] += outPrev10[0] * delta7[1];
                tmp[6] += outPrev10[1] * delta7[2];
                tmp[6] += outPrev10[2] * delta7[3];
                
                float4 sum = outPrev10 * delta7;
                tmp[7] += sum[0] + sum[1] + sum[2] + sum[3];
                
                tmp[8] += outPrev10[1] * delta7[0];
                tmp[8] += outPrev10[2] * delta7[1];
                tmp[8] += outPrev10[3] * delta7[2];
            }
            if (k < height/2 - 1 && l < width/4 - 1)
            {
                uint offsetPrev11 =
                    ((l+1)*4 + (offsetStartPrev + (k+1)*2) * widthPrev) / 4;
                float outPrev11 = outsPrev[offsetPrev11][0];
                
                tmp[9] += outPrev11 * delta7[3];
            }
        }}
    }
    
    uint offsetStartWeights =
        (depthPrev + nbChannelsPrev * depth) * weightHeight;
    uint offsetWeights0 = 0 +
        (offsetStartWeights + 0) * weightWidth;
    uint offsetWeights1 = 1 +
        (offsetStartWeights + 0) * weightWidth;
    uint offsetWeights2 = 2 +
        (offsetStartWeights + 0) * weightWidth;
    uint offsetWeights3 = 0 +
        (offsetStartWeights + 1) * weightWidth;
    uint offsetWeights4 = 1 +
        (offsetStartWeights + 1) * weightWidth;
    uint offsetWeights5 = 2 +
        (offsetStartWeights + 1) * weightWidth;
    uint offsetWeights6 = 0 +
        (offsetStartWeights + 2) * weightWidth;
    uint offsetWeights7 = 1 +
        (offsetStartWeights + 2) * weightWidth;
    uint offsetWeights8 = 2 +
        (offsetStartWeights + 2) * weightWidth;
    
    if (accumulate)
    {
        grads[offsetWeights0] += tmp[0];
        grads[offsetWeights1] += tmp[1];
        grads[offsetWeights2] += tmp[2];
        grads[offsetWeights3] += tmp[3];
        grads[offsetWeights4] += tmp[4];
        grads[offsetWeights5] += tmp[5];
        grads[offsetWeights6] += tmp[6];
        grads[offsetWeights7] += tmp[7];
        grads[offsetWeights8] += tmp[8];
    }
    else
    {
        grads[offsetWeights0] = tmp[0];
        grads[offsetWeights1] = tmp[1];
        grads[offsetWeights2] = tmp[2];
        grads[offsetWeights3] = tmp[3];
        grads[offsetWeights4] = tmp[4];
        grads[offsetWeights5] = tmp[5];
        grads[offsetWeights6] = tmp[6];
        grads[offsetWeights7] = tmp[7];
        grads[offsetWeights8] = tmp[8];
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
    uint2 id [[ thread_position_in_grid ]])
{
    uint height, width;
    uint heightPrev, widthPrev;
    uint weightHeight, weightWidth;
    uint nbChannels;
    uint nbChannelsPrev;
    int startI, startJ;
    int endI, endJ;
    int offI, offJ;
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
        offI = pStart[4];
        offJ = pStart[5];
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
        if ((int)(stride*l)+j-offJ >= 0 &&
            (int)(stride*l)+j-offJ < (int)widthPrev &&
            (int)(stride*k)+i-offI >= 0 &&
            (int)(stride*k)+i-offI < (int)heightPrev)
        {
            uint offset = l + (offsetStart + k) * width;
            float deltaCur = delta[offset];
            
            uint offsetPrev = (int)(stride*l)+j-offJ +
                (offsetStartPrev + (int)(stride*k)+i-offI)*widthPrev;
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
