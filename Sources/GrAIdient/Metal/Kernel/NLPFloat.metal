//
// NLPFloat.metal
// GrAIdient
//
// Created by Jean-François Reboud on 25/06/2024.
//

#include <metal_stdlib>
using namespace metal;

kernel void createRoPESeqMatrixFloat(
    constant int * seqPositions,
    constant uint & nbHeads,
    constant uint & nbNeurons,
    constant uint & sequence,
    device float * rotationMatrix,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeurons / nbHeads;
    uint nbBlocks = size / 2;
    
    uint block = id[0];
    uint seq = id[1];
    
    if (block >= nbBlocks || seq >= sequence)
    {
        return ;
    }
    
    float position = (float)seqPositions[seq];
    float theta = pow(
        10000.0,
        -2.0 * (float)block / (float)size
    );
    float mTheta = position * theta;
    float cosVal = cos(mTheta);
    float sinVal = sin(mTheta);
    
    uint offset = 2 * block + seq * size;
    rotationMatrix[offset] = cosVal;
    rotationMatrix[1 + offset] = sinVal;
}

kernel void RoPESeqForwardFloat(
    const device float * outsPrev,
    const device float * rotationMatrix,
    constant uint & nbHeads,
    constant uint & nbNeurons,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeurons / nbHeads;
    uint nbBlocks = size / 2;
    
    uint head = id[0] / nbBlocks;
    uint block = id[0] % nbBlocks;
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (head >= nbHeads || block >= nbBlocks ||
        elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint offset1 = 2 * block + seq * size;
    uint offset2 = 2 * block + head * size +
        nbNeurons * seq + sequence * nbNeurons * elem;
    
    float cosVal = rotationMatrix[offset1];
    float sinVal = rotationMatrix[1 + offset1];
    
    float in1 = outsPrev[offset2];
    float in2 = outsPrev[1 + offset2];
    
    float out1 = in1 * cosVal - in2 * sinVal;
    float out2 = in1 * sinVal + in2 * cosVal;
    
    outs[offset2] = out1;
    outs[1 + offset2] = out2;
}

kernel void RoPESeqSeqBackwardFloat(
    const device float * delta,
    const device float * rotationMatrix,
    constant uint & nbHeads,
    constant uint & nbNeurons,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device float * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeurons / nbHeads;
    uint nbBlocks = size / 2;
    
    uint head = id[0] / nbBlocks;
    uint block = id[0] % nbBlocks;
    uint elem = id[1] / sequence;
    uint seq = id[1] % sequence;
    
    if (head >= nbHeads || block >= nbBlocks ||
        elem >= nbBatch || seq >= sequence)
    {
        return ;
    }
    
    uint offset1 = 2 * block + seq * size;
    uint offset2 = 2 * block + head * size +
        nbNeurons * seq + sequence * nbNeurons * elem;
    
    float cosVal = rotationMatrix[offset1];
    float sinVal = rotationMatrix[1 + offset1];
    
    float out1 = delta[offset2];
    float out2 = delta[1 + offset2];
    
    float in1 = out1 * cosVal + out2 * sinVal;
    float in2 = -out1 * sinVal + out2 * cosVal;
    
    if (dirty)
    {
        deltaPrev[offset2] = in1;
        deltaPrev[1 + offset2] = in2;
    }
    else
    {
        deltaPrev[offset2] += in1;
        deltaPrev[1 + offset2] += in2;
    }
}

kernel void encodeCausalityFloat(
    constant uint & nbHeadsQuery,
    constant uint & nbNeurons,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint headQuery = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (headQuery >= nbHeadsQuery ||
        seqK >= sequence || seqK <= seqQ ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    uint offset = seqK + headQuery * sequence +
        nbNeurons * seqQ + sequence * nbNeurons * elem;
    outs[offset] = -1e9;
}

kernel void queryCausalSeqForwardFloat(
    const device float * query,
    const device float * key,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevQuery / nbHeadsQuery;
    uint nbBlocksHead = nbHeadsQuery / nbHeadsKey;
    
    uint headQuery = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (headQuery >= nbHeadsQuery || seqK > seqQ ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    uint headKey = headQuery / nbBlocksHead;
    float tmp = 0.0;
    
    for (uint j=0; j<size; j++)
    {
        uint depthPrevKey = j + headKey * size;
        uint depthPrevQuery = j + headQuery * size;
        
        uint offsetQuery = depthPrevQuery +
            nbNeuronsPrevQuery * seqQ + sequence * nbNeuronsPrevQuery * elem;
        uint offsetKey = depthPrevKey +
            nbNeuronsPrevKey * seqK + sequence * nbNeuronsPrevKey * elem;
        
        tmp += query[offsetQuery] * key[offsetKey];
    }
    tmp /= sqrt((float)size);
    
    uint offset = seqK + headQuery * sequence +
        nbNeurons * seqQ + sequence * nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void queryCausalSeq4ForwardFloat(
    const device float4 * query,
    const device float4 * key,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevQuery / nbHeadsQuery;
    uint nbBlocksHead = nbHeadsQuery / nbHeadsKey;
    
    uint headQuery = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (headQuery >= nbHeadsQuery || seqK > seqQ ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    uint headKey = headQuery / nbBlocksHead;
    float4 tmp = 0.0;
    
    for (uint j=0; j<size/4; j++)
    {
        uint depthPrevKey = j * 4 + headKey * size;
        uint depthPrevQuery = j * 4 + headQuery * size;
        
        uint offsetQuery = (depthPrevQuery +
            nbNeuronsPrevQuery * seqQ +
            sequence * nbNeuronsPrevQuery * elem) / 4;
        uint offsetKey = (depthPrevKey +
            nbNeuronsPrevKey * seqK + sequence * nbNeuronsPrevKey * elem) / 4;
        
        tmp += query[offsetQuery] * key[offsetKey];
    }
    tmp /= sqrt((float)size);
    
    uint offset = seqK + headQuery * sequence +
        nbNeurons * seqQ + sequence * nbNeurons * elem;
    outs[offset] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

kernel void queryCausalSeqGenerateFloat(
    const device float * query,
    const device float * key,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevQuery / nbHeadsQuery;
    uint nbBlocksHead = nbHeadsQuery / nbHeadsKey;
    
    uint headQuery = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1];
    
    if (headQuery >= nbHeadsQuery || seqK >= sequence ||
        elem >= nbBatch)
    {
        return ;
    }
    
    uint headKey = headQuery / nbBlocksHead;
    float tmp = 0.0;
    
    for (uint j=0; j<size; j++)
    {
        uint depthPrevKey = j + headKey * size;
        uint depthPrevQuery = j + headQuery * size;
        
        uint offsetQuery = depthPrevQuery + nbNeuronsPrevQuery * elem;
        uint offsetKey = depthPrevKey +
            nbNeuronsPrevKey * seqK + sequence * nbNeuronsPrevKey * elem;
        
        tmp += query[offsetQuery] * key[offsetKey];
    }
    tmp /= sqrt((float)size);
    
    uint offset = seqK + headQuery * sequence + nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void queryCausalSeq4GenerateFloat(
    const device float4 * query,
    const device float4 * key,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevQuery / nbHeadsQuery;
    uint nbBlocksHead = nbHeadsQuery / nbHeadsKey;
    
    uint headQuery = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1];
    
    if (headQuery >= nbHeadsQuery || seqK >= sequence ||
        elem >= nbBatch)
    {
        return ;
    }
    
    uint headKey = headQuery / nbBlocksHead;
    float4 tmp = 0.0;
    
    for (uint j=0; j<size/4; j++)
    {
        uint depthPrevKey = j * 4 + headKey * size;
        uint depthPrevQuery = j * 4 + headQuery * size;
        
        uint offsetQuery = (depthPrevQuery + nbNeuronsPrevQuery * elem) / 4;
        uint offsetKey = (depthPrevKey +
            nbNeuronsPrevKey * seqK + sequence * nbNeuronsPrevKey * elem) / 4;
        
        tmp += query[offsetQuery] * key[offsetKey];
    }
    tmp /= sqrt((float)size);
    
    uint offset = seqK + headQuery * sequence + nbNeurons * elem;
    outs[offset] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

kernel void queryCausalQuerySeqBackwardFloat(
    const device float * delta,
    const device float * key,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device float * query,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevQuery / nbHeadsQuery;
    uint nbBlocksHead = nbHeadsQuery / nbHeadsKey;
    
    uint headQuery = id[0] / size;
    uint j = id[0] % size;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (headQuery >= nbHeadsQuery || j >= size ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    uint headKey = headQuery / nbBlocksHead;
    uint depthPrevKey = j + headKey * size;
    uint depthPrevQuery = j + headQuery * size;
    
    float tmp = 0.0;
    for (uint seqK=0; seqK<=seqQ; seqK++)
    {
        uint offset = seqK + headQuery * sequence +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetKey = depthPrevKey +
            nbNeuronsPrevKey * seqK + sequence * nbNeuronsPrevKey * elem;
        
        tmp += delta[offset] * key[offsetKey];
    }
    tmp /= sqrt((float)size);
    
    uint offsetQuery = depthPrevQuery +
        nbNeuronsPrevQuery * seqQ + sequence * nbNeuronsPrevQuery * elem;
    
    if (dirty)
    {
        query[offsetQuery] = tmp;
    }
    else
    {
        query[offsetQuery] += tmp;
    }
}

kernel void queryCausalQuerySeq4BackwardFloat(
    const device float * delta,
    const device float4 * key,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device float4 * query,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevQuery / nbHeadsQuery;
    uint nbBlocksHead = nbHeadsQuery / nbHeadsKey;
    
    uint headQuery = id[0] / (size / 4);
    uint j = id[0] % (size / 4);
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (headQuery >= nbHeadsQuery || j * 4 >= size ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    uint headKey = headQuery / nbBlocksHead;
    uint depthPrevKey = j * 4 + headKey * size;
    uint depthPrevQuery = j * 4 + headQuery * size;
    
    float4 tmp = 0.0;
    for (uint seqK=0; seqK<=seqQ; seqK++)
    {
        uint offset = seqK + headQuery * sequence +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetKey = (depthPrevKey +
            nbNeuronsPrevKey * seqK + sequence * nbNeuronsPrevKey * elem) / 4;
        
        tmp += delta[offset] * key[offsetKey];
    }
    tmp /= sqrt((float)size);
    
    uint offsetQuery = (depthPrevQuery +
        nbNeuronsPrevQuery * seqQ + sequence * nbNeuronsPrevQuery * elem) / 4;
    
    if (dirty)
    {
        query[offsetQuery] = tmp;
    }
    else
    {
        query[offsetQuery] += tmp;
    }
}

kernel void queryCausalKeySeqBackwardFloat(
    const device float * delta,
    const device float * query,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device float * key,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevKey / nbHeadsKey;
    uint nbBlocksHead = nbHeadsQuery / nbHeadsKey;
    
    uint headKey = id[0] / size;
    uint j = id[0] % size;
    uint elem = id[1] / sequence;
    uint seqK = id[1] % sequence;
    
    if (headKey >= nbHeadsKey || j >= size ||
        elem >= nbBatch || seqK >= sequence)
    {
        return ;
    }
    
    uint depthPrevKey = j + headKey * size;
    
    float tmp = 0.0;
    for (uint blockHead=0; blockHead<nbBlocksHead; blockHead++)
    {
        uint headQuery = blockHead + nbBlocksHead * headKey;
        uint depthPrevQuery = j + headQuery * size;
        
        for (uint seqQ=seqK; seqQ<sequence; seqQ++)
        {
            uint offset = seqK + headQuery * sequence +
                nbNeurons * seqQ + sequence * nbNeurons * elem;
            uint offsetQuery = depthPrevQuery +
                nbNeuronsPrevQuery * seqQ +
                sequence * nbNeuronsPrevQuery * elem;
            
            tmp += delta[offset] * query[offsetQuery];
        }
    }
    tmp /= sqrt((float)size);
    
    uint offsetKey = depthPrevKey +
        nbNeuronsPrevKey * seqK + sequence * nbNeuronsPrevKey * elem;
    
    if (dirty)
    {
        key[offsetKey] = tmp;
    }
    else
    {
        key[offsetKey] += tmp;
    }
}

kernel void queryCausalKeySeq4BackwardFloat(
    const device float * delta,
    const device float4 * query,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device float4 * key,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevKey / nbHeadsKey;
    uint nbBlocksHead = nbHeadsQuery / nbHeadsKey;
    
    uint headKey = id[0] / (size / 4);
    uint j = id[0] % (size / 4);
    uint elem = id[1] / sequence;
    uint seqK = id[1] % sequence;
    
    if (headKey >= nbHeadsKey || j * 4 >= size ||
        elem >= nbBatch || seqK >= sequence)
    {
        return ;
    }
    
    uint depthPrevKey = j * 4 + headKey * size;
    
    float4 tmp = 0.0;
    for (uint blockHead=0; blockHead<nbBlocksHead; blockHead++)
    {
        uint headQuery = blockHead + nbBlocksHead * headKey;
        uint depthPrevQuery = j * 4 + headQuery * size;
        
        for (uint seqQ=seqK; seqQ<sequence; seqQ++)
        {
            uint offset = seqK + headQuery * sequence +
                nbNeurons * seqQ + sequence * nbNeurons * elem;
            uint offsetQuery = (depthPrevQuery +
                nbNeuronsPrevQuery * seqQ +
                sequence * nbNeuronsPrevQuery * elem) / 4;
            
            tmp += delta[offset] * query[offsetQuery];
        }
    }
    tmp /= sqrt((float)size);
    
    uint offsetKey = (depthPrevKey +
        nbNeuronsPrevKey * seqK + sequence * nbNeuronsPrevKey * elem) / 4;
    
    if (dirty)
    {
        key[offsetKey] = tmp;
    }
    else
    {
        key[offsetKey] += tmp;
    }
}

kernel void valueCausalSeqForwardFloat(
    const device float * value,
    const device float * score,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevValue / nbHeadsValue;
    uint nbBlocksHead = nbHeadsScore / nbHeadsValue;
    
    uint headScore = id[0] / size;
    uint j = id[0] % size;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (headScore >= nbHeadsScore || j >= size ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    uint headValue = headScore / nbBlocksHead;
    
    uint depthScore = j + headScore * size;
    uint depthValue = j + headValue * size;
    
    float tmp = 0.0;
    for (uint seqK=0; seqK<=seqQ; seqK++)
    {
        uint offsetValue = depthValue +
            nbNeuronsPrevValue * seqK + sequence * nbNeuronsPrevValue * elem;
        uint offsetScore = seqK + headScore * sequence +
            nbNeuronsPrevScore * seqQ + sequence * nbNeuronsPrevScore * elem;
        
        tmp += value[offsetValue] * score[offsetScore];
    }
    
    uint offset = depthScore + nbNeurons * seqQ + sequence * nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void valueCausalSeq4ForwardFloat(
    const device float4 * value,
    const device float * score,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    device float4 * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevValue / nbHeadsValue;
    uint nbBlocksHead = nbHeadsScore / nbHeadsValue;
    
    uint headScore = id[0] / (size / 4);
    uint j = id[0] % (size / 4);
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (headScore >= nbHeadsScore || j >= size ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    uint headValue = headScore / nbBlocksHead;
    
    uint depthScore = j * 4 + headScore * size;
    uint depthValue = j * 4 + headValue * size;
    
    float4 tmp = 0.0;
    for (uint seqK=0; seqK<=seqQ; seqK++)
    {
        uint offsetValue = (depthValue +
            nbNeuronsPrevValue * seqK +
            sequence * nbNeuronsPrevValue * elem) / 4;
        uint offsetScore = seqK + headScore * sequence +
            nbNeuronsPrevScore * seqQ + sequence * nbNeuronsPrevScore * elem;
        
        tmp += value[offsetValue] * score[offsetScore];
    }
    
    uint offset = (depthScore +
        nbNeurons * seqQ + sequence * nbNeurons * elem) / 4;
    outs[offset] = tmp;
}

kernel void valueCausalSeqGenerateFloat(
    const device float * value,
    const device float * score,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    device float * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevValue / nbHeadsValue;
    uint nbBlocksHead = nbHeadsScore / nbHeadsValue;
    
    uint headScore = id[0] / size;
    uint j = id[0] % size;
    uint elem = id[1];
    
    if (headScore >= nbHeadsScore || j >= size ||
        elem >= nbBatch)
    {
        return ;
    }
    
    uint headValue = headScore / nbBlocksHead;
    
    uint depthScore = j + headScore * size;
    uint depthValue = j + headValue * size;
    
    float tmp = 0.0;
    for (uint seqK=0; seqK<=sequence; seqK++)
    {
        uint offsetValue = depthValue +
            nbNeuronsPrevValue * seqK + sequence * nbNeuronsPrevValue * elem;
        uint offsetScore = seqK + headScore * sequence +
            nbNeuronsPrevScore * elem;
        
        tmp += value[offsetValue] * score[offsetScore];
    }
    
    uint offset = depthScore + nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void valueCausalSeq4GenerateFloat(
    const device float4 * value,
    const device float * score,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    device float4 * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevValue / nbHeadsValue;
    uint nbBlocksHead = nbHeadsScore / nbHeadsValue;
    
    uint headScore = id[0] / (size / 4);
    uint j = id[0] % (size / 4);
    uint elem = id[1];
    
    if (headScore >= nbHeadsScore || j >= size ||
        elem >= nbBatch)
    {
        return ;
    }
    
    uint headValue = headScore / nbBlocksHead;
    
    uint depthScore = j * 4 + headScore * size;
    uint depthValue = j * 4 + headValue * size;
    
    float4 tmp = 0.0;
    for (uint seqK=0; seqK<=sequence; seqK++)
    {
        uint offsetValue = (depthValue +
            nbNeuronsPrevValue * seqK +
            sequence * nbNeuronsPrevValue * elem) / 4;
        uint offsetScore = seqK + headScore * sequence +
            nbNeuronsPrevScore * elem;
        
        tmp += value[offsetValue] * score[offsetScore];
    }
    
    uint offset = (depthScore + nbNeurons * elem) / 4;
    outs[offset] = tmp;
}

kernel void valueCausalValueSeqBackwardFloat(
    const device float * delta,
    const device float * score,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device float * value,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevValue / nbHeadsValue;
    uint nbBlocksHead = nbHeadsScore / nbHeadsValue;
    
    uint headValue = id[0] / size;
    uint j = id[0] % size;
    uint elem = id[1] / sequence;
    uint seqK = id[1] % sequence;
    
    if (headValue >= nbHeadsValue || j >= size ||
        elem >= nbBatch || seqK >= sequence)
    {
        return ;
    }
    
    uint depthValue = j + headValue * size;
    
    float tmp = 0.0;
    for (uint blockHead=0; blockHead<nbBlocksHead; blockHead++)
    {
        uint headScore = blockHead + nbBlocksHead * headValue;
        uint depthScore = j + headScore * size;
        
        for (uint seqQ=seqK; seqQ<sequence; seqQ++)
        {
            uint offset = depthScore +
                nbNeurons * seqQ + sequence * nbNeurons * elem;
            uint offsetScore = seqK + headScore * sequence +
                nbNeuronsPrevScore * seqQ +
                sequence * nbNeuronsPrevScore * elem;
            
            tmp += delta[offset] * score[offsetScore];
        }
    }
    
    uint offsetValue = depthValue +
        nbNeuronsPrevValue * seqK + sequence * nbNeuronsPrevValue * elem;
    if (dirty)
    {
        value[offsetValue] = tmp;
    }
    else
    {
        value[offsetValue] += tmp;
    }
}

kernel void valueCausalValueSeq4BackwardFloat(
    const device float4 * delta,
    const device float * score,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device float4 * value,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevValue / nbHeadsValue;
    uint nbBlocksHead = nbHeadsScore / nbHeadsValue;
    
    uint headValue = id[0] / (size / 4);
    uint j = id[0] % (size / 4);
    uint elem = id[1] / sequence;
    uint seqK = id[1] % sequence;
    
    if (headValue >= nbHeadsValue || j * 4 >= size ||
        elem >= nbBatch || seqK >= sequence)
    {
        return ;
    }
    
    uint depthValue = j + headValue * size;
    
    float4 tmp = 0.0;
    for (uint blockHead=0; blockHead<nbBlocksHead; blockHead++)
    {
        uint headScore = blockHead + nbBlocksHead * headValue;
        uint depthScore = j * 4 + headScore * size;
        
        for (uint seqQ=seqK; seqQ<sequence; seqQ++)
        {
            uint offset = (depthScore +
                nbNeurons * seqQ + sequence * nbNeurons * elem) / 4;
            uint offsetScore = seqK + headScore * sequence +
                nbNeuronsPrevScore * seqQ +
                sequence * nbNeuronsPrevScore * elem;
            
            tmp += delta[offset] * score[offsetScore];
        }
    }
    
    uint offsetValue = (depthValue +
        nbNeuronsPrevValue * seqK + sequence * nbNeuronsPrevValue * elem) / 4;
    if (dirty)
    {
        value[offsetValue] = tmp;
    }
    else
    {
        value[offsetValue] += tmp;
    }
}

kernel void valueCausalScoreSeqBackwardFloat(
    const device float * delta,
    const device float * value,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device float * score,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevValue / nbHeadsValue;
    uint nbBlocksHead = nbHeadsScore / nbHeadsValue;
    
    uint headScore = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (headScore >= nbHeadsScore || seqK > seqQ ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    uint headValue = headScore / nbBlocksHead;
    
    float tmp = 0.0;
    for (uint j=0; j<size; j++)
    {
        uint depthScore = j + headScore * size;
        uint depthValue = j + headValue * size;
        
        uint offset = depthScore +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetValue = depthValue +
            nbNeuronsPrevValue * seqK + sequence * nbNeuronsPrevValue * elem;
        
        tmp += delta[offset] * value[offsetValue];
    }
    
    uint offsetScore = seqK + headScore * sequence +
        nbNeuronsPrevScore * seqQ + sequence * nbNeuronsPrevScore * elem;
    
    if (dirty)
    {
        score[offsetScore] = tmp;
    }
    else
    {
        score[offsetScore] += tmp;
    }
}

kernel void valueCausalScoreSeq4BackwardFloat(
    const device float4 * delta,
    const device float4 * value,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device float * score,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size = nbNeuronsPrevValue / nbHeadsValue;
    uint nbBlocksHead = nbHeadsScore / nbHeadsValue;
    
    uint headScore = id[0] / sequence;
    uint seqK = id[0] % sequence;
    uint elem = id[1] / sequence;
    uint seqQ = id[1] % sequence;
    
    if (headScore >= nbHeadsScore || seqK > seqQ ||
        elem >= nbBatch || seqQ >= sequence)
    {
        return ;
    }
    
    uint headValue = headScore / nbBlocksHead;
    
    float4 tmp = 0.0;
    for (uint j=0; j<size/4; j++)
    {
        uint depthScore = j * 4 + headScore * size;
        uint depthValue = j * 4 + headValue * size;
        
        uint offset = (depthScore +
            nbNeurons * seqQ + sequence * nbNeurons * elem) / 4;
        uint offsetValue = (depthValue +
            nbNeuronsPrevValue * seqK +
            sequence * nbNeuronsPrevValue * elem) / 4;
        
        tmp += delta[offset] * value[offsetValue];
    }
    
    uint offsetScore = seqK + headScore * sequence +
        nbNeuronsPrevScore * seqQ + sequence * nbNeuronsPrevScore * elem;
    
    if (dirty)
    {
        score[offsetScore] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
    else
    {
        score[offsetScore] += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
}
