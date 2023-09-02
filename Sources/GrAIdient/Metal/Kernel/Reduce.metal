//
// Reduce.metal
// GrAIdient
//
// Created by Jean-François Reboud on 17/05/2023.
//

#include <metal_stdlib>
using namespace metal;

kernel void reduceSum64(
     const device float * ins,
     constant uint * pDimensions,
     constant uint * pNbThreadgroups,
     device float * outs,
     uint2 groupId [[ threadgroup_position_in_grid ]],
     uint2 threadId [[ thread_position_in_threadgroup ]],
     uint2 id [[ thread_position_in_grid ]])
{
    constexpr uint threadsPerThreadgroup = 64;
    threadgroup float sumShared[threadsPerThreadgroup];
    
    uint dim1;
    uint dim2;
    uint nbThreadgroups;
    
    if (pDimensions && pNbThreadgroups && ins && outs)
    {
        dim1 = pDimensions[0];
        dim2 = pDimensions[1];
        nbThreadgroups = *pNbThreadgroups;
    }
    else
        return ;
    
    uint elem1 = id[0];
    uint elem2 = id[1];
    
    if (elem1 >= dim1 && elem2 >= dim2)
    {
        return ;
    }
    
    uint offset = elem2 * dim1 + elem1;
    sumShared[threadId[0]] = ins[offset];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride=threadsPerThreadgroup/2; stride>0; stride>>=1)
    {
        uint index = threadId[0] + groupId[0] * threadsPerThreadgroup;
        if (threadId[0] < stride && (index + stride) < dim1)
        {
            sumShared[threadId[0]] += sumShared[threadId[0] + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (threadId[0] == 0)
    {
        uint offset = elem2 * nbThreadgroups + groupId[0];
        outs[offset] = sumShared[0];
    }
}

kernel void reduceSum(
     const device float * ins,
     constant uint * pDimensions,
     device float * outs,
     uint id [[ thread_position_in_grid ]])
{
    uint dim1;
    uint dim2;
    
    if (pDimensions && ins && outs)
    {
        dim1 = pDimensions[0];
        dim2 = pDimensions[1];
    }
    else
        return ;
    
    uint elem2 = id;
    if (elem2 >= dim2)
    {
        return ;
    }
    
    float sum = 0.0;
    for (uint elem1=0; elem1<dim1; elem1++)
    {
        uint offset = elem2 * dim1 + elem1;
        sum += ins[offset];
    }
    outs[elem2] = sum;
}

kernel void reduceMax64(
     const device float * ins,
     constant uint * pDimensions,
     constant uint * pNbThreadgroups,
     device float * outs,
     uint2 groupId [[ threadgroup_position_in_grid ]],
     uint2 threadId [[ thread_position_in_threadgroup ]],
     uint2 id [[ thread_position_in_grid ]])
{
    constexpr uint threadsPerThreadgroup = 64;
    threadgroup float valShared[threadsPerThreadgroup];
    
    uint dim1;
    uint dim2;
    uint nbThreadgroups;
    
    if (pDimensions && pNbThreadgroups && ins && outs)
    {
        dim1 = pDimensions[0];
        dim2 = pDimensions[1];
        nbThreadgroups = *pNbThreadgroups;
    }
    else
        return ;
    
    uint elem1 = id[0];
    uint elem2 = id[1];
    
    if (elem1 >= dim1 && elem2 >= dim2)
    {
        return ;
    }
    
    uint offset = elem2 * dim1 + elem1;
    valShared[threadId[0]] = ins[offset];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride=threadsPerThreadgroup/2; stride>0; stride>>=1)
    {
        uint index = threadId[0] + groupId[0] * threadsPerThreadgroup;
        if (threadId[0] < stride && (index + stride) < dim1)
        {
            valShared[threadId[0]] = max(
                 valShared[threadId[0] + stride],
                 valShared[threadId[0]]
             );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (threadId[0] == 0)
    {
        uint offset = elem2 * nbThreadgroups + groupId[0];
        outs[offset] = valShared[0];
    }
}

kernel void reduceMax(
     const device float * ins,
     constant uint * pDimensions,
     device float * outs,
     uint id [[ thread_position_in_grid ]])
{
    uint dim1;
    uint dim2;
    
    if (pDimensions && ins && outs)
    {
        dim1 = pDimensions[0];
        dim2 = pDimensions[1];
    }
    else
        return ;
    
    uint elem2 = id;
    if (elem2 >= dim2)
    {
        return ;
    }
    
    float val = ins[elem2 * dim1];
    for (uint elem1=0; elem1<dim1; elem1++)
    {
        uint offset = elem2 * dim1 + elem1;
        val = max(ins[offset], val);
    }
    outs[elem2] = val;
}
