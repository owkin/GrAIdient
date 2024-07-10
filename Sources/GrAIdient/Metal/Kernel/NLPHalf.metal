//
// NLPHalf.metal
// GrAIdient
//
// Created by Jean-François Reboud on 25/06/2024.
//

#include <metal_stdlib>
using namespace metal;

kernel void createRoPESeqMatrixHalf(
    constant int * seqPositions,
    constant uint & nbHeads,
    constant uint & nbNeurons,
    constant uint & sequence,
    device half * rotationMatrix,
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

kernel void RoPESeqForwardHalf(
    const device half * outsPrev,
    const device half * rotationMatrix,
    constant uint & nbHeads,
    constant uint & nbNeurons,
    constant uint & nbBatch,
    constant uint & sequence,
    device half * outs,
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
    
    half cosVal = rotationMatrix[offset1];
    half sinVal = rotationMatrix[1 + offset1];
    
    half in1 = outsPrev[offset2];
    half in2 = outsPrev[1 + offset2];
    
    half out1 = in1 * cosVal - in2 * sinVal;
    half out2 = in1 * sinVal + in2 * cosVal;
    
    outs[offset2] = out1;
    outs[1 + offset2] = out2;
}

kernel void RoPESeqSeqBackwardHalf(
    const device half * delta,
    const device half * rotationMatrix,
    constant uint & nbHeads,
    constant uint & nbNeurons,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device half * deltaPrev,
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
    
    half cosVal = rotationMatrix[offset1];
    half sinVal = rotationMatrix[1 + offset1];
    
    half out1 = delta[offset2];
    half out2 = delta[1 + offset2];
    
    half in1 = out1 * cosVal + out2 * sinVal;
    half in2 = -out1 * sinVal + out2 * cosVal;
    
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

kernel void encodeCausalityHalf(
    constant uint & nbHeadsQuery,
    constant uint & nbNeurons,
    constant uint & nbBatch,
    constant uint & sequence,
    device half * outs,
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
    outs[offset] = -1e4;
}

kernel void queryCausalSeqForwardHalf(
    const device half * query,
    const device half * key,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    device half * outs,
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
    half tmp = 0.0;
    
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
    tmp /= sqrt((half)size);
    
    uint offset = seqK + headQuery * sequence +
        nbNeurons * seqQ + sequence * nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void queryCausalSeq4ForwardHalf(
    const device half4 * query,
    const device half4 * key,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    device half * outs,
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
    half4 tmp = 0.0;
    
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
    tmp /= sqrt((half)size);
    
    uint offset = seqK + headQuery * sequence +
        nbNeurons * seqQ + sequence * nbNeurons * elem;
    outs[offset] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

kernel void queryCausalSeqGenerateHalf(
    const device half * query,
    const device half * key,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    device half * outs,
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
    half tmp = 0.0;
    
    for (uint j=0; j<size; j++)
    {
        uint depthPrevKey = j + headKey * size;
        uint depthPrevQuery = j + headQuery * size;
        
        uint offsetQuery = depthPrevQuery + nbNeuronsPrevQuery * elem;
        uint offsetKey = depthPrevKey +
            nbNeuronsPrevKey * seqK + sequence * nbNeuronsPrevKey * elem;
        
        tmp += query[offsetQuery] * key[offsetKey];
    }
    tmp /= sqrt((half)size);
    
    uint offset = seqK + headQuery * sequence + nbNeurons * elem;
    outs[offset] = tmp;
}

kernel void queryCausalSeq4GenerateHalf(
    const device half4 * query,
    const device half4 * key,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    device half * outs,
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
    half4 tmp = 0.0;
    
    for (uint j=0; j<size/4; j++)
    {
        uint depthPrevKey = j * 4 + headKey * size;
        uint depthPrevQuery = j * 4 + headQuery * size;
        
        uint offsetQuery = (depthPrevQuery + nbNeuronsPrevQuery * elem) / 4;
        uint offsetKey = (depthPrevKey +
            nbNeuronsPrevKey * seqK + sequence * nbNeuronsPrevKey * elem) / 4;
        
        tmp += query[offsetQuery] * key[offsetKey];
    }
    tmp /= sqrt((half)size);
    
    uint offset = seqK + headQuery * sequence + nbNeurons * elem;
    outs[offset] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

kernel void queryCausalQuerySeqBackwardHalf(
    const device half * delta,
    const device half * key,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device half * query,
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
    
    half tmp = 0.0;
    for (uint seqK=0; seqK<=seqQ; seqK++)
    {
        uint offset = seqK + headQuery * sequence +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetKey = depthPrevKey +
            nbNeuronsPrevKey * seqK + sequence * nbNeuronsPrevKey * elem;
        
        tmp += delta[offset] * key[offsetKey];
    }
    tmp /= sqrt((half)size);
    
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

kernel void queryCausalQuerySeq4BackwardHalf(
    const device half * delta,
    const device half4 * key,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device half4 * query,
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
    
    half4 tmp = 0.0;
    for (uint seqK=0; seqK<=seqQ; seqK++)
    {
        uint offset = seqK + headQuery * sequence +
            nbNeurons * seqQ + sequence * nbNeurons * elem;
        uint offsetKey = (depthPrevKey +
            nbNeuronsPrevKey * seqK + sequence * nbNeuronsPrevKey * elem) / 4;
        
        tmp += delta[offset] * key[offsetKey];
    }
    tmp /= sqrt((half)size);
    
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

kernel void queryCausalKeySeqBackwardHalf(
    const device half * delta,
    const device half * query,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device half * key,
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
    
    half tmp = 0.0;
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
    tmp /= sqrt((half)size);
    
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

kernel void queryCausalKeySeq4BackwardHalf(
    const device half * delta,
    const device half4 * query,
    constant uint & nbHeadsQuery,
    constant uint & nbHeadsKey,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevQuery,
    constant uint & nbNeuronsPrevKey,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device half4 * key,
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
    
    half4 tmp = 0.0;
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
    tmp /= sqrt((half)size);
    
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

kernel void valueCausalSeqForwardHalf(
    const device half * value,
    const device half * score,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    device half * outs,
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
    
    half tmp = 0.0;
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

kernel void valueCausalSeq4ForwardHalf(
    const device half4 * value,
    const device half * score,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    device half4 * outs,
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
    
    half4 tmp = 0.0;
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

kernel void valueCausalSeqGenerateHalf(
    const device half * value,
    const device half * score,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    device half * outs,
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
    
    half tmp = 0.0;
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

kernel void valueCausalSeq4GenerateHalf(
    const device half4 * value,
    const device half * score,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    device half4 * outs,
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
    
    half4 tmp = 0.0;
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

kernel void valueCausalValueSeqBackwardHalf(
    const device half * delta,
    const device half * score,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device half * value,
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
    
    half tmp = 0.0;
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

kernel void valueCausalValueSeq4BackwardHalf(
    const device half4 * delta,
    const device half * score,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device half4 * value,
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
    
    half4 tmp = 0.0;
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

kernel void valueCausalScoreSeqBackwardHalf(
    const device half * delta,
    const device half * value,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device half * score,
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
    
    half tmp = 0.0;
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

kernel void valueCausalScoreSeq4BackwardHalf(
    const device half4 * delta,
    const device half4 * value,
    constant uint & nbHeadsValue,
    constant uint & nbHeadsScore,
    constant uint & nbNeurons,
    constant uint & nbNeuronsPrevValue,
    constant uint & nbNeuronsPrevScore,
    constant uint & nbBatch,
    constant uint & sequence,
    constant uint & dirty,
    device half * score,
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
    
    half4 tmp = 0.0;
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
