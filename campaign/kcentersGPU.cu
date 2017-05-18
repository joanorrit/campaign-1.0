/*
 * --------------------------------------------------------------------------- *
 *                                  CAMPAIGN                                   *
 * --------------------------------------------------------------------------- *
 * This is part of the CAMPAIGN data clustering library originating from       *
 * Simbios, the NIH National Center for Physics-Based Simulation of Biological *
 * Structures at Stanford, funded under the NIH Roadmap for Medical Research,  *
 * grant U54 GM072970 (See https://simtk.org), and the FEATURE Project at      *
 * Stanford, funded under the NIH grant LM05652                                *
 * (See http://feature.stanford.edu/index.php).                                *
 *                                                                             *
 * Portions copyright (c) 2010 Stanford University, Authors, and Contributors. *
 * Authors:  Kai J. Kolhoff                                                    *
 * Contributors: Marc Sosnick, William Hsu                                     *
 *                                                                             *
 * This program is free software: you can redistribute it and/or modify it     *
 * under the terms of the GNU Lesser General Public License as published by    *
 * the Free Software Foundation, either version 3 of the License, or (at your  *
 * option) any later version.                                                  *
 *                                                                             *
 * This program is distributed in the hope that it will be useful, but WITHOUT *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public        *
 * License for more details.                                                   *
 *                                                                             *
 * You should have received a copy of the GNU Lesser General Public License    *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.       *
 * --------------------------------------------------------------------------- *
 */

/* $Id: kcentersGPU.cu 158 2011-01-12 23:06:27Z msosnick $ */

/**
 * \file kcentersGPU.cu
 * \brief A CUDA K-centers implementation
 *
 * Implements parallel K-centers clustering on the GPU
 * 
 * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 12/2/2010
 * \version 1.0
 **/
#include <cstdio>
#include "kcentersGPU.h"
#include "timing.h"
#include <iostream>

#define SHARED_MEM 0
#define INTERCALATED_DATA 1

#ifdef SHARED_MEM
#define MEMBARRIER() __syncthreads()
#define WARPMEMBARRIER()
#else
#define MEMBARRIER() {__threadfence();__syncthreads();}
#define WARPMEMBARRIER() {__threadfence();__syncthreads();}
#endif


template <class T, class U>
__device__ static void parallelMax(int BLOCKSIZE, int tid, T *s_A, U *s_B)
{
    if (BLOCKSIZE >= 1024) { if (tid < 512 && s_A[tid + 512] > s_A[tid]) { s_A[tid] = s_A[tid + 512]; s_B[tid] = s_B[tid + 512]; } MEMBARRIER(); }
    if (BLOCKSIZE >=  512) { if (tid < 256 && s_A[tid + 256] > s_A[tid]) { s_A[tid] = s_A[tid + 256]; s_B[tid] = s_B[tid + 256]; } MEMBARRIER(); }
    if (BLOCKSIZE >=  256) { if (tid < 128 && s_A[tid + 128] > s_A[tid]) { s_A[tid] = s_A[tid + 128]; s_B[tid] = s_B[tid + 128]; } MEMBARRIER(); }
    if (BLOCKSIZE >=  128) { if (tid <  64 && s_A[tid +  64] > s_A[tid]) { s_A[tid] = s_A[tid +  64]; s_B[tid] = s_B[tid +  64]; } MEMBARRIER(); }
    
    if (tid < 32)
    {
        volatile T *vs_A = s_A;
        volatile U *vs_B = s_B;
        if (BLOCKSIZE >= 64) { if (vs_A[tid + 32] > vs_A[tid]) { vs_A[tid] = vs_A[tid + 32]; vs_B[tid] = vs_B[tid + 32]; } WARPMEMBARRIER();}
        if (BLOCKSIZE >= 32) { if (vs_A[tid + 16] > vs_A[tid]) { vs_A[tid] = vs_A[tid + 16]; vs_B[tid] = vs_B[tid + 16]; } WARPMEMBARRIER();}
        if (BLOCKSIZE >= 16) { if (vs_A[tid +  8] > vs_A[tid]) { vs_A[tid] = vs_A[tid +  8]; vs_B[tid] = vs_B[tid +  8]; } WARPMEMBARRIER();}
        if (BLOCKSIZE >=  8) { if (vs_A[tid +  4] > vs_A[tid]) { vs_A[tid] = vs_A[tid +  4]; vs_B[tid] = vs_B[tid +  4]; } WARPMEMBARRIER();}
        if (BLOCKSIZE >=  4) { if (vs_A[tid +  2] > vs_A[tid]) { vs_A[tid] = vs_A[tid +  2]; vs_B[tid] = vs_B[tid +  2]; } WARPMEMBARRIER();}
        if (BLOCKSIZE >=  2) { if (vs_A[tid +  1] > vs_A[tid]) { vs_A[tid] = vs_A[tid +  1]; vs_B[tid] = vs_B[tid +  1]; } WARPMEMBARRIER();}
    }
}


template <unsigned int BLOCKSIZE, class T, class U>
__device__ static void parallelMax(int tid, T *s_A, U *s_B)
{
    if (BLOCKSIZE >= 1024) { if (tid < 512 && s_A[tid + 512] > s_A[tid]) { s_A[tid] = s_A[tid + 512]; s_B[tid] = s_B[tid + 512]; } __syncthreads(); }
    if (BLOCKSIZE >=  512) { if (tid < 256 && s_A[tid + 256] > s_A[tid]) { s_A[tid] = s_A[tid + 256]; s_B[tid] = s_B[tid + 256]; } __syncthreads(); }
    if (BLOCKSIZE >=  256) { if (tid < 128 && s_A[tid + 128] > s_A[tid]) { s_A[tid] = s_A[tid + 128]; s_B[tid] = s_B[tid + 128]; } __syncthreads(); }
    if (BLOCKSIZE >=  128) { if (tid <  64 && s_A[tid +  64] > s_A[tid]) { s_A[tid] = s_A[tid +  64]; s_B[tid] = s_B[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        if (BLOCKSIZE >= 64) { if (s_A[tid + 32] > s_A[tid]) { s_A[tid] = s_A[tid + 32]; s_B[tid] = s_B[tid + 32]; } }
        if (BLOCKSIZE >= 32) { if (s_A[tid + 16] > s_A[tid]) { s_A[tid] = s_A[tid + 16]; s_B[tid] = s_B[tid + 16]; } }
        if (BLOCKSIZE >= 16) { if (s_A[tid +  8] > s_A[tid]) { s_A[tid] = s_A[tid +  8]; s_B[tid] = s_B[tid +  8]; } }
        if (BLOCKSIZE >=  8) { if (s_A[tid +  4] > s_A[tid]) { s_A[tid] = s_A[tid +  4]; s_B[tid] = s_B[tid +  4]; } }
        if (BLOCKSIZE >=  4) { if (s_A[tid +  2] > s_A[tid]) { s_A[tid] = s_A[tid +  2]; s_B[tid] = s_B[tid +  2]; } }
        if (BLOCKSIZE >=  2) { if (s_A[tid +  1] > s_A[tid]) { s_A[tid] = s_A[tid +  1]; s_B[tid] = s_B[tid +  1]; } }
    }
}



__global__ void checkCentroid_CUDA(int TPB, int N, int D, int iter, int centroid, FLOAT_TYPE *X, FLOAT_TYPE *CTR, FLOAT_TYPE *DIST, int *ASSIGN, FLOAT_TYPE *MAXDIST, int *MAXID)
{

unsigned int tid = threadIdx.x;                   // thread ID in block
unsigned int t   = blockIdx.x * blockDim.x + tid; // global thread ID

#ifdef SHARED_MEM
    extern __shared__ FLOAT_TYPE array[];                  // shared memory
    FLOAT_TYPE *s_dist    = (FLOAT_TYPE*) array;                // tpb distances
    int   *s_ID      = (int*)   &s_dist[blockDim.x];  // tpb IDs
    FLOAT_TYPE *s_ctr     = (FLOAT_TYPE*) &s_ID[blockDim.x];    // tpb centroid components
#else    
    FLOAT_TYPE *s_dist;
    int   *s_ID;
    extern __shared__ FLOAT_TYPE array[];                  // shared memory
    FLOAT_TYPE *s_ctr     = (FLOAT_TYPE*) array;    // tpb centroid components
    //FLOAT_TYPE s_ctr[THREADSPERBLOCK];
#endif

#ifndef SHARED_MEM
    int origAssign;
    FLOAT_TYPE origDist;
    s_ID = &ASSIGN[t-(t%TPB)];
    s_dist = &DIST[t-(t%TPB)];
    origDist = DIST[t];
    origAssign = ASSIGN[t];
#endif

    s_dist[tid] = 0.0;
    s_ID  [tid] = t;
   // if (t < N)
   // {
        // compute distance
        FLOAT_TYPE dist    = 0.0;
        int   offsetD = 0;
        // process each centroid component
        while (offsetD < D)
        {
            // read up to tpb components from global to shared memory
            if (offsetD + tid < D) s_ctr[tid] = CTR[offsetD + tid];
            //__syncthreads();
	    MEMBARRIER();
            // compute distance in each dimension separately and build sum
	if (t < N) {
            for (int d = 0; d < min(blockDim.x, D - offsetD); d++)
            {
                //dist += distanceComponentGPU(s_ctr + d - offsetD, X + (offsetD + d) * N + t);
#ifdef INTERCALATED_DATA
                dist += distanceComponentGPU(s_ctr + d, X +  t*D + offsetD + d);
#else
                dist += distanceComponentGPU(s_ctr + d, X + (offsetD + d) * N + t);
#endif
            }
	}
            offsetD += blockDim.x;
            // __syncthreads();
	    MEMBARRIER();
        }
        if (t < N) {
            dist = distanceFinalizeGPU<FLOAT_TYPE>(1, &dist);
        // if new distance smaller then update assignment
#ifdef SHARED_MEM
        FLOAT_TYPE currDist = DIST[t];
#else
        FLOAT_TYPE currDist = origDist;
#endif
        if (dist < currDist)
        {
#ifdef SHARED_MEM
            DIST  [t] = currDist = dist;
            ASSIGN[t] = iter;
#else       
	    origDist = currDist = dist;
            origAssign = iter;
#endif
        }
        s_dist[tid] = currDist;
       }
    //__syncthreads();
    MEMBARRIER();
    // find max distance of data point to centroid and its index in block
    //parallelMax<THREADSPERBLOCK>(tid, s_dist, s_ID);

    parallelMax(TPB, tid, s_dist, s_ID);
    MEMBARRIER();
    // write maximum distance and index to global mem
    if (t < N) {
    	if (tid == 0) 
    	{
        	/*MAXDIST[blockIdx.x] = s_dist[tid];
        	MAXID  [blockIdx.x] = s_ID[tid];*/
            int blockID = blockIdx.y*gridDim.x + blockIdx.x;
            MAXDIST[blockID] = s_dist[tid];
            MAXID  [blockID] = s_ID[tid];
		
    	}
#ifndef SHARED_MEM
        DIST[t] = origDist;
        ASSIGN[t] = origAssign;
#endif
    }
}

void kcentersGPU(int TPB, int N, int K, int D, FLOAT_TYPE *x, int *assign, FLOAT_TYPE *dist, int *centroids, int seed, DataIO *data)
{
    // CUDA kernel parameters
    int numBlock = (int) ceil((FLOAT_TYPE) N / (FLOAT_TYPE) THREADSPERBLOCK);
    //dim3 block(THREADSPERBLOCK);
    //dim3 grid(numBlock);
    dim3 grid;
    if (numBlock>65535) {
cout << "2D grid: " << sqrt(numBlock) + 1 << endl;
      grid = dim3(sqrt(numBlock) + 1, sqrt(numBlock) + 1);
    }
    else {
      grid = dim3(numBlock);
    }
    dim3 block(TPB);
    int sMem = (2 * sizeof(FLOAT_TYPE) + sizeof(int)) * TPB;//THREADSPERBLOCK;

    //Timing timer;
    Timing timer;

    timer.init("kcenterGPU kernel");
 
    // GPU memory pointers, allocate and initialize device memory
    /*FLOAT_TYPE *dist_d         = data->allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * N, dist);
    int   *assign_d       = data->allocDeviceMemory<int*>  (sizeof(int)   * N);
    FLOAT_TYPE *x_d            = data->allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * N * D, x);
    FLOAT_TYPE *ctr_d          = data->allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * D);
    FLOAT_TYPE *maxDistBlock_d = data->allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * numBlock);
    int   *maxIdBlock_d   = data->allocDeviceMemory<int*>  (sizeof(int)   * numBlock);*/

    int NMAX = N + ( ((N%TPB)==0) ? 0 : TPB-(N%TPB) ); // for non shared memory reductions 
    FLOAT_TYPE *dist_d         = data->allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * NMAX, dist);
    int   *assign_d       = data->allocDeviceMemory<int*>  (sizeof(int)   * NMAX);
    FLOAT_TYPE *x_d            = data->allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * N * D, x);
    FLOAT_TYPE *ctr_d          = data->allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * D);
    FLOAT_TYPE *maxDistBlock_d = data->allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * numBlock);
    int   *maxIdBlock_d   = data->allocDeviceMemory<int*>  (sizeof(int)   * numBlock);
    
    // Initialize host memory
    FLOAT_TYPE *maxDistBlock = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * numBlock);
    int   *maxID        = (int*)   malloc(sizeof(int));
    FLOAT_TYPE *ctr          = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * D);
    
    int centroid = seed;
    // for each cluster
    for (int k = 0; k < K; ++k)
    {
        // send centroid coordinates for current iteration to device memory
#ifdef INTERCALATED_DATA
        for (int d = 0; d < D; d++) ctr[d] = x[d + centroid*D];
#else
        for (int d = 0; d < D; d++) ctr[d] = x[d * N + centroid];
#endif
        cudaMemcpy(ctr_d, ctr, sizeof(FLOAT_TYPE) * D, cudaMemcpyHostToDevice);      
        centroids[k] = centroid; 

        // for each data point, check if new centroid closer than previous best and if, reassign
	timer.start("kcenterGPU kernel");
        checkCentroid_CUDA<<<grid, block, sMem>>>(TPB, N, D, k, centroid, x_d, ctr_d, dist_d, assign_d, maxDistBlock_d, maxIdBlock_d);
	timer.stop("kcenterGPU kernel");
        CUT_CHECK_ERROR("checkCentroid_CUDA() kernel execution failed");
        
        // get next max distance between data point and centroid
        int tempMax = 0;
        cudaMemcpy(maxDistBlock, maxDistBlock_d, sizeof(FLOAT_TYPE) * numBlock, cudaMemcpyDeviceToHost);        
        for (int i = 1; i < numBlock; i++) 
        {
            if (maxDistBlock[i] > maxDistBlock[tempMax]) tempMax = i;
        }
        cudaMemcpy(maxID, maxIdBlock_d + tempMax, sizeof(int), cudaMemcpyDeviceToHost);
        centroid = maxID[0];
    }
    // copy final assignments back to host
    cudaMemcpy(assign, assign_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
    
    // free up memory
    cudaFree(dist_d);
    cudaFree(assign_d);
    cudaFree(x_d);
    cudaFree(ctr_d);
    cudaFree(maxDistBlock_d);
    cudaFree(maxIdBlock_d);
    free(maxDistBlock);
    free(maxID);
    free(ctr);
    timer.report();
}


