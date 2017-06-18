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

/* $Id: kmeansGPUMain.cc 158 2011-01-12 23:06:27Z msosnick $ */

/**
 * \file kmeansGPU.cc
 * \brief Test/example file for k-means clustering on the GPU
 *
 * Test/example file for k-means clustering on the GPU
 *
 * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 2/2/2010
 * \version 1.0
 **/

#ifdef HAVE_CONFIG_H
#include "../config.h"
#include "../campaign.h"
#else
#include "./kmeansGPU.h"
#endif

#define SHARED_MEM 1
#define INTERCALATED_DATA 1
using namespace std;


/**
 * \brief Main for testing
 */
int main(int argc, const char* argv[])
{
    // Initialization
    Timing timer;
#if 0  
    // Parse command line
    Defaults *defaults = new Defaults(argc, argv, "km");
    
    // select CUDA device based on command-line switches
    GpuDevices *systemGpuDevices = new GpuDevices(defaults);
#endif
    // get data and information about data from datafile
    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    int d = atoi(argv[3]);

    const int seed = 0;
    DataIO* data = new DataIO;

    timer.init("kmeansGPU");
    
    FLOAT_TYPE score = 0.0f;
    //FLOAT_TYPE* x = data->readData(defaults->getInputFileName().c_str()); // data point 
    //float* x = data->makeData(n,k,d);
    
    /*int N = data->getNumElements();
    int K = data->getNumClusters();
    int D = data->getDimensions();*/

    int N = n;
    int K = k;
    int D = d;
    
    
    /*FLOAT_TYPE* ctr = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * K * D); // array containing centroids
    memset(ctr, 0, sizeof(FLOAT_TYPE) * K * D);
    int* assign = (int*) malloc(sizeof(int) * N); // assignments
    memset(assign, 0, sizeof(int) * N);*/
    
    // initialize first centroids with first K data points  
    // for each cluster
    /*for (unsigned int k = 0; k < K; k++)
    {
        // for each dimension
        for (unsigned int d = 0; d < D; d++)
        {
            
            ctr[k * D + d] = x[d * N + k];
        }
    }
    
    // do clustering on GPU 
    timer.start("kmeansGPU");
    score = kmeansGPU(N, K, D, x, ctr, assign, (unsigned int)0, data);
    timer.stop("kmeansGPU");*/
#if 0
    int MaxN = 4;  
    int MaxD = 10;  
    int MaxK = 10;  
#else
    int MaxN = 1;
    int MaxD = 1;
    int MaxK = 1;
#endif

#if 1
    int MaxTPB = 1024;
    int TPB = 64;
#else
    int MaxTPB = 1024;
    int TPB = 512;
#endif
    int copyN=N;
    int copyK=K;
    int copyD=D;
    
    float *x;
    //timer.start("kmeansGPU");
    for (;TPB<=MaxTPB; TPB=TPB*2) {

      //int tmp = 10;
      N=copyN;
      for (int n=0; n<MaxN; n=n+1) {
        D=copyD;
        for (int di=0; di<MaxD; di=di+1) {
          D=copyD-di*10*2;
          K=copyK;
          for (int ki=0; ki<MaxK; ki=ki+1) {
            K=copyK-ki*10;

    cout << "Fent punts per N, K i D: " << N << " " << K << " " << D << endl;
    x = data->makeData(N,K,D);

    FLOAT_TYPE* ctr = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * K * D); // array containing centroids
    memset(ctr, 0, sizeof(FLOAT_TYPE) * K * D);
    int* assign = (int*) malloc(sizeof(int) * N); // assignments
    memset(assign, 0, sizeof(int) * N);

    // initialize first centroids with first K data points  
    // for each cluster
    for (unsigned int k = 0; k < K; k++)
    {
        // for each dimension
        for (unsigned int d = 0; d < D; d++)
        {
#ifdef INTERCALATED_DATA
	    ctr[k * D + d] = x[k * D + d];
#else    
            ctr[k * D + d] = x[d * N + k];
#endif
        }
    }
    
    cout << "kmeansGPU: BEGIN GPU (N, K, D, TPB) " << N << " " << K << " " << D << " " << TPB << endl;
    // do clustering on GPU 
    timer.reset("kmeansGPU");
    timer.start("kmeansGPU");
    score = kmeansGPU(TPB, N, K, D, x, ctr, assign, (unsigned int)10, data);
    timer.stop("kmeansGPU");
    
    //timer.report();
    cout << "kmeansGPU: END GPU (N, K, D, TPB) " << N << " " << K << " " << D << " " << TPB << endl;
    timer.report("kmeansGPU"); 

    // free memory
    free(x);
    free(ctr);
    free(assign);
          } /* K */
        } /* D */
        //N=N/tmp;
	N /= 10;
      } /* N */
    } /* TPB */
    //timer.stop("kmeansGPU");
    N=copyN; K=copyK; D=copyD;

#if 0
    // print results
    data->printClusters(N, K, D, x, ctr, assign);  
    cout << "Score: " << score << endl;
    if (defaults->getTimerOutput() == true) timer.report();
#endif
  	
    //////////////timer.report("kmeansGPU"); 
    // done
    cout << "Done clustering" << endl;
    
    return 0;
}
