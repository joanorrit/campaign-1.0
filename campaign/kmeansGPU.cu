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
 * Authors: Kai J. Kolhoff                                                     *
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

/* $Id: kmeansGPU.cu 158 2011-01-12 23:06:27Z msosnick $ */

/**
 * \File kmeansGPU.cu
 * \brief A basic CUDA K-means implementation
 *
 * A module of the CAMPAIGN data clustering library for parallel architectures
 * Implements parallel K-means clustering (base implementation) on the GPU
 *
 * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 12/2/2010
 * \version 1.0
 **/

#include "./kmeansGPU.h"

#define SHARED_MEM 1
//#define INTERCALATED_DATA 1

#ifdef SHARED_MEM
#define MEMBARRIER() __syncthreads()
#define WARPMEMBARRIER()
#else
#define MEMBARRIER() {__threadfence();__syncthreads();}
#define WARPMEMBARRIER() {__threadfence();__syncthreads();}
#endif

using namespace std;

/*template <unsigned int BLOCKSIZE, class T>
__device__ static void reduceOne(int tid, T *s_A)
{
    if (BLOCKSIZE >= 1024) { if (tid < 512) { s_A[tid] += s_A[tid + 512]; } __syncthreads(); }
    if (BLOCKSIZE >=  512) { if (tid < 256) { s_A[tid] += s_A[tid + 256]; } __syncthreads(); }
    if (BLOCKSIZE >=  256) { if (tid < 128) { s_A[tid] += s_A[tid + 128]; } __syncthreads(); }
    if (BLOCKSIZE >=  128) { if (tid <  64) { s_A[tid] += s_A[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        if (BLOCKSIZE >= 64) { s_A[tid] += s_A[tid + 32]; }
        if (BLOCKSIZE >= 32) { s_A[tid] += s_A[tid + 16]; }
        if (BLOCKSIZE >= 16) { s_A[tid] += s_A[tid +  8]; }
        if (BLOCKSIZE >=  8) { s_A[tid] += s_A[tid +  4]; }
        if (BLOCKSIZE >=  4) { s_A[tid] += s_A[tid +  2]; }
        if (BLOCKSIZE >=  2) { s_A[tid] += s_A[tid +  1]; }
    }
}


template <unsigned int BLOCKSIZE, class T, class U>
__device__ static void reduceTwo(int tid, T *s_A, U *s_B)
{
    if (BLOCKSIZE >= 1024) { if (tid < 512) { s_A[tid] += s_A[tid + 512]; s_B[tid] += s_B[tid + 512]; } __syncthreads(); }
    if (BLOCKSIZE >=  512) { if (tid < 256) { s_A[tid] += s_A[tid + 256]; s_B[tid] += s_B[tid + 256]; } __syncthreads(); }
    if (BLOCKSIZE >=  256) { if (tid < 128) { s_A[tid] += s_A[tid + 128]; s_B[tid] += s_B[tid + 128]; } __syncthreads(); }
    if (BLOCKSIZE >=  128) { if (tid <  64) { s_A[tid] += s_A[tid +  64]; s_B[tid] += s_B[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        if (BLOCKSIZE >= 64) { s_A[tid] += s_A[tid + 32]; s_B[tid] += s_B[tid + 32]; }
        if (BLOCKSIZE >= 32) { s_A[tid] += s_A[tid + 16]; s_B[tid] += s_B[tid + 16]; }
        if (BLOCKSIZE >= 16) { s_A[tid] += s_A[tid +  8]; s_B[tid] += s_B[tid +  8]; }
        if (BLOCKSIZE >=  8) { s_A[tid] += s_A[tid +  4]; s_B[tid] += s_B[tid +  4]; }
        if (BLOCKSIZE >=  4) { s_A[tid] += s_A[tid +  2]; s_B[tid] += s_B[tid +  2]; }
        if (BLOCKSIZE >=  2) { s_A[tid] += s_A[tid +  1]; s_B[tid] += s_B[tid +  1]; }
    }
}*/
//template <unsigned int BLOCKSIZE, class T>
template <class T>
__device__ static void reduceOne(int BLOCKSIZE, int tid, T *s_A)
{
    if (BLOCKSIZE >= 1024) { if (tid < 512) { s_A[tid] += s_A[tid + 512]; } MEMBARRIER(); }
    if (BLOCKSIZE >=  512) { if (tid < 256) { s_A[tid] += s_A[tid + 256]; } MEMBARRIER(); }
    if (BLOCKSIZE >=  256) { if (tid < 128) { s_A[tid] += s_A[tid + 128]; } MEMBARRIER(); }
    if (BLOCKSIZE >=  128) { if (tid <  64) { s_A[tid] += s_A[tid +  64]; } MEMBARRIER(); }
    
    if (tid < 32)
    {
        volatile T *vs_A=s_A;
        if (BLOCKSIZE >= 64) { vs_A[tid] += vs_A[tid + 32]; }
        if (BLOCKSIZE >= 32) { vs_A[tid] += vs_A[tid + 16]; }
        if (BLOCKSIZE >= 16) { vs_A[tid] += vs_A[tid +  8]; }
        if (BLOCKSIZE >=  8) { vs_A[tid] += vs_A[tid +  4]; }
        if (BLOCKSIZE >=  4) { vs_A[tid] += vs_A[tid +  2]; }
        if (BLOCKSIZE >=  2) { vs_A[tid] += vs_A[tid +  1]; }
    }
}


//template <unsigned int BLOCKSIZE, class T, class U>
template <class T, class U>
__device__ static void reduceTwo(int BLOCKSIZE, int tid, T *s_A, U *s_B)
{
    if (BLOCKSIZE >= 1024) { if (tid < 512) { s_A[tid] += s_A[tid + 512]; s_B[tid] += s_B[tid + 512]; } MEMBARRIER(); }
    if (BLOCKSIZE >=  512) { if (tid < 256) { s_A[tid] += s_A[tid + 256]; s_B[tid] += s_B[tid + 256]; } MEMBARRIER(); }
    if (BLOCKSIZE >=  256) { if (tid < 128) { s_A[tid] += s_A[tid + 128]; s_B[tid] += s_B[tid + 128]; } MEMBARRIER(); }
    if (BLOCKSIZE >=  128) { if (tid <  64) { s_A[tid] += s_A[tid +  64]; s_B[tid] += s_B[tid +  64]; } MEMBARRIER(); }
    
    if (tid < 32)
    {
        volatile T *vs_A=s_A;
        volatile U *vs_B=s_B;
        if (BLOCKSIZE >= 64) { vs_A[tid] += vs_A[tid + 32]; vs_B[tid] += vs_B[tid + 32]; }
        if (BLOCKSIZE >= 32) { vs_A[tid] += vs_A[tid + 16]; vs_B[tid] += vs_B[tid + 16]; }
        if (BLOCKSIZE >= 16) { vs_A[tid] += vs_A[tid +  8]; vs_B[tid] += vs_B[tid +  8]; }
        if (BLOCKSIZE >=  8) { vs_A[tid] += vs_A[tid +  4]; vs_B[tid] += vs_B[tid +  4]; }
        if (BLOCKSIZE >=  4) { vs_A[tid] += vs_A[tid +  2]; vs_B[tid] += vs_B[tid +  2]; }
        if (BLOCKSIZE >=  2) { vs_A[tid] += vs_A[tid +  1]; vs_B[tid] += vs_B[tid +  1]; }
    }
}


__global__ static void assignToClusters_KMCUDA(int TPB, int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN)
{
#ifdef SHARED_MEM
    extern __shared__ FLOAT_TYPE array[];                        // shared memory
    FLOAT_TYPE *s_center = (FLOAT_TYPE*) array;                       // tpb centroid components
#else
    extern __shared__ FLOAT_TYPE array[];                        // shared memory
    FLOAT_TYPE *s_center = CTR;
#endif 
    unsigned int t = blockDim.x * blockIdx.x + threadIdx.x; // global thread ID
    unsigned int tid = threadIdx.x;                         // thread ID in block
    
    // for each element
    if (t < N)
    {
        FLOAT_TYPE minDist  = 0.0;
        int   minIndex = 0;
        // for each centroid
        for (unsigned int k = 0; k < K; k++)
        {
            // compute distance
            FLOAT_TYPE dist = 0.0;
            unsigned int offsetD = 0;
            // loop over all dimensions in segments of size tpb
            while (offsetD < D)
            {
                // read up to tpb dimensions of centroid K (coalesced)
#ifdef SHARED_MEM
                if (offsetD + tid < D) s_center[tid] = CTR[k * D + offsetD + tid];
                //__syncthreads();
		MEMBARRIER();
#endif
                // for each of the following tpb (or D - offsetD) dimensions
                for (unsigned int d = offsetD; d < min(offsetD + blockDim.x, D); d++)
                {
                    // broadcast centroid position and compute distance to data
                    // point along dimension; reading of X is coalesced
#ifdef SHARED_MEM
	#ifdef INTERCALATED_DATA
                    dist += distanceComponentGPU(s_center + (d - offsetD), X + t*D + d);
	#else
                    dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + t));
	#endif

#else
	#ifdef INTERCALATED_DATA
                    dist += distanceComponentGPU(s_center + (k*D + d), X + t*D + d);
	#else
                    dist += distanceComponentGPU(s_center + (k*D + d), X + (d * N + t));
	#endif
#endif

                }
                offsetD += blockDim.x;
                MEMBARRIER();//__syncthreads();
            }
            dist = distanceFinalizeGPU<FLOAT_TYPE>(1, &dist);
            // if distance to centroid smaller than previous best, reassign
            if (dist < minDist || k == 0)
            {
                minDist = dist;
                minIndex = k;
            }
        }
        // now write index of closest centroid to global mem (coalesced)
        ASSIGN[t] = minIndex;
    }
}


__global__ static void calcScore_CUDA(int TPB, int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN, FLOAT_TYPE *SCORE, FLOAT_TYPE *BSCORE)
{
#ifdef SHARED_MEM
    extern __shared__ FLOAT_TYPE array[];                     // shared memory
    FLOAT_TYPE *s_scores = (FLOAT_TYPE*) array;                    // tpb partial scores
    FLOAT_TYPE *s_center = (FLOAT_TYPE*) &s_scores[blockDim.x];    // up to tpb centroid components
#else
    extern __shared__ FLOAT_TYPE array[];                     // shared memory
    FLOAT_TYPE *s_center = CTR;    // up to tpb centroid components
    FLOAT_TYPE *s_scores = &BSCORE[blockIdx.x*TPB];
#endif

    int k   = blockIdx.x;                                // cluster ID
    int tid = threadIdx.x;                               // in-block thread ID
    
    // initialize partial scores
    s_scores[tid] = 0.0;
    // loop over data points
    unsigned int offsetN = tid;
    while (offsetN < N)
    {
        FLOAT_TYPE dist = 0.0;
        unsigned int offsetD = 0;
        // loop over dimensions
        while (offsetD < D)
        {
            // at each iteration read up to tpb centroid components from global mem (coalesced)
#ifdef SHARED_MEM
            if (offsetD + tid < D) s_center[tid] = CTR[k * D + offsetD + tid];
            MEMBARRIER();//__syncthreads();
#endif
            // thread divergence likely
            if (ASSIGN[offsetN] == k)
            {
                // for each of the following tpb (or D - offsetD) dimensions
                for (unsigned int d = offsetD; d < min(offsetD + blockDim.x, D); d++)
                {
                    // broadcast centroid component and compute contribution to distance to data poi

                    //dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + offsetN));
                    //dist += distanceComponentGPU(s_center + (d - offsetD), X + (offsetN) * D + d);
#ifdef SHARED_MEM
	#ifdef INTERCALATED_DATA
                    dist += distanceComponentGPU(s_center + (d - offsetD), X + (offsetN) * D + d);
	#else
                    dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + offsetN));
	#endif
#else
	#ifdef INTERCALATED_DATA
                    dist += distanceComponentGPU(s_center + (k*D + d), X + offsetN*D + d);
	#else
                    dist += distanceComponentGPU(s_center + (k*D + d), X + (d * N + offsetN));
	#endif
#endif

                }
            }
            offsetD += blockDim.x;
            MEMBARRIER();//__syncthreads();
        }
        // update partial score
        s_scores[tid] += distanceFinalizeGPU(1, &dist);
        offsetN += blockDim.x;
    }
    MEMBARRIER();//__syncthreads();
    // compute score for block by reducing over threads
   reduceOne(TPB, tid, s_scores);
    if (tid == 0) SCORE[k] = s_scores[tid];
}



__global__ static void calcCentroids_CUDA(int TPB, int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN, FLOAT_TYPE *TMP1, int *TMP2)
{
#ifdef SHARED_MEM
    extern __shared__ FLOAT_TYPE array[];                            // shared memory
    int   *s_numElements = (int*)   array;                      // tpb partial sums of elements in cluster
    FLOAT_TYPE *s_centerParts = (FLOAT_TYPE*) &s_numElements[blockDim.x]; // tpb partial centroid components    
#else
    FLOAT_TYPE   *s_centerParts = &TMP1[blockIdx.x*TPB];       
    int *s_numElements = &TMP2[blockIdx.x*TPB]; 
#endif

    int k   = blockIdx.x;                                       // cluster ID
    int tid = threadIdx.x;                                      // in-block thread ID
    
    FLOAT_TYPE clusterSize = 0.0;                                    // only used by thread 0 in block
    
    // initialize partial cluster size
    s_numElements[tid] = 0;
    // for each dimension
    for (unsigned int d = 0; d < D; d++)
    {
        // initialize centroid parts
        s_centerParts[tid] = 0.0;
        unsigned int offset = tid;
        // loop over data points
        while (offset < N)
        {
            // thread divergence likely
            if (ASSIGN[offset] == k)
            {
                // update centroid parts
                //s_centerParts[tid] += X[d * N + offset];
                s_centerParts[tid] += X[offset*D + d];

                // increment number of elements in cluster
                if (d == 0) s_numElements[tid]++;
            }
            // move on to next segment
            offset += blockDim.x;
        }
        MEMBARRIER();//__syncthreads();
        
        // take sum over all tpb array elements
        // reduce number of cluster elements only once
        if (d == 0)
        {
            // reduce number of elements and centroid parts
            //reduceTwo<THREADSPERBLOCK>(tid, s_centerParts, s_numElements);
            reduceTwo(TPB, tid, s_centerParts, s_numElements);
            if (tid == 0) clusterSize = (FLOAT_TYPE) s_numElements[tid];
            // note: if clusterSize == 0 we can return here
        }
        else
        {
            // reduce centroid parts
            //reduceOne<THREADSPERBLOCK>(tid, s_centerParts);
            reduceOne(TPB, tid, s_centerParts);
        }
        // write result to global mem (non-coalesced)
        // could replace this by coalesced writes followed by matrix transposition
        if (tid == 0) if (clusterSize > 0) CTR[k * D + d] = s_centerParts[tid] / clusterSize;
    }
}


FLOAT_TYPE kmeansGPU(int TPB, int N, int K, int D, FLOAT_TYPE *x, FLOAT_TYPE *ctr, int *assign, unsigned int maxIter, DataIO *data)
{
    // CUDA kernel parameters
    /*dim3 block(THREADSPERBLOCK);
    dim3 gridK(K);
    dim3 gridN((int) ceil((FLOAT_TYPE) N / (FLOAT_TYPE) THREADSPERBLOCK));
    int sMemAssign  = (sizeof(FLOAT_TYPE) *     THREADSPERBLOCK);
    int sMemScore   = (sizeof(FLOAT_TYPE) * 2 * THREADSPERBLOCK);
    int sMemCenters = (sizeof(FLOAT_TYPE) *     THREADSPERBLOCK + sizeof(int) * THREADSPERBLOCK);*/
    // CUDA kernel parameters
Timing timer;
timer.init("Centroids Kernel");
timer.init("AssignToClusters Kernel");
timer.init("Score Kernel");
timer.init("Allocate Memory");
timer.init("CopyMem DeviceToHost");

    dim3 block(TPB);
    dim3 gridK(K);
    int  NB =  ( int ) ceil((FLOAT_TYPE) N / (FLOAT_TYPE) TPB);
    dim3 gridN(NB);
    int sMemAssign  = (sizeof(FLOAT_TYPE) *     TPB);
    int sMemScore   = (sizeof(FLOAT_TYPE) * 2 * TPB);
    int sMemCenters = (sizeof(FLOAT_TYPE) *     TPB + sizeof(int) * TPB);
    
    // GPU memory pointers, allocate and initialize device memory
    timer.start("Allocate Memory");
    FLOAT_TYPE *x_d         = data->allocDeviceMemory<FLOAT_TYPE*>      (sizeof(FLOAT_TYPE) * N * D, x);
    FLOAT_TYPE *ctr_d       = data->allocDeviceMemory<FLOAT_TYPE*>      (sizeof(FLOAT_TYPE) * K * D, ctr);
    int *assign_d      = data->allocDeviceMemory<int*>        (sizeof(int) * N);
    FLOAT_TYPE *s_d         = data->allocZeroedDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * K);
    
    FLOAT_TYPE *tmp_d         = data->allocZeroedDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) *TPB*K*2);
    FLOAT_TYPE *coord_d         = data->allocZeroedDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) *NB*D);
    int *count_d         = data->allocZeroedDeviceMemory<int*>(sizeof(int) *K);
    timer.stop("Allocate Memory");
    // Initialize host memory
    FLOAT_TYPE *s   = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * K);
    
    // initialize scores
    FLOAT_TYPE oldscore = -1000.0, score = 0.0;
    if (maxIter < 1) maxIter = INT_MAX;
    unsigned int iter = 0;
    // loop until defined number of iterations reached or converged
    //while (iter < maxIter && ((score - oldscore) * (score - oldscore)) > EPS)
    while (iter < maxIter)
    {
        oldscore = score;
        
        // skip at first iteration and use provided centroids instead
        if (iter > 0)
        {
            timer.start("Centroids Kernel");
            calcCentroids_CUDA<<<gridK, block, sMemCenters>>>(TPB, N, D, x_d, ctr_d, assign_d, tmp_d, (int*)(tmp_d + TPB*K));
            timer.stop("Centroids Kernel");
            CUT_CHECK_ERROR("calcCentroids_CUDA() kernel execution failed");
        }
        iter++;
        
        // update clusters and create backup of current cluster centers
        timer.start("AssignToClusters Kernel");
        assignToClusters_KMCUDA<<<gridN, block, sMemAssign>>>(TPB, N, K, D, x_d, ctr_d, assign_d);
        timer.stop("AssignToClusters Kernel");
        CUT_CHECK_ERROR("assignToClusters_KMCUDA() kernel execution failed");
        
        // get score per cluster for unsorted data
        timer.start("Score Kernel");
        calcScore_CUDA<<<gridK, block, sMemScore>>>(TPB, N, D, x_d, ctr_d, assign_d, s_d, tmp_d);
        timer.stop("Score Kernel");
        CUT_CHECK_ERROR("calcScore_CUDA() kernel execution failed");
        
        // copy scores per cluster back to the host and do reduction on CPU
        cudaMemcpy(s, s_d,    sizeof(FLOAT_TYPE) * K, cudaMemcpyDeviceToHost);
        score = 0.0;
        for (int i = 0; i < K; i++) score += s[i];
    }
    cout << "Number of iterations: " << iter << endl;
    timer.start("CopyMem DeviceToHost");
    // copy centroids back to host
    cudaMemcpy(ctr, ctr_d,         sizeof(FLOAT_TYPE) * K * D, cudaMemcpyDeviceToHost);
    // copy assignments back to host
    cudaMemcpy(assign, assign_d,    sizeof(int)   * N    , cudaMemcpyDeviceToHost); 
    timer.stop("CopyMem DeviceToHost");

    timer.report("Centroids Kernel");
    timer.report("AssignToClusters Kernel");
    timer.report("Score Kernel");
    timer.report("Allocate Memory");
    timer.report("CopyMem DeviceToHost");
    cout << endl;
    // free memory
    cudaFree(x_d);
    cudaFree(ctr_d);
    cudaFree(assign_d);
    cudaFree(s_d);
    free(s);
    
    return score;
}

