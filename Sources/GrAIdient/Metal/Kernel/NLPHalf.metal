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
