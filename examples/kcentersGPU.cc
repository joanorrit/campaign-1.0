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

/* $Id: kcentersGPUMain.cc 158 2011-01-12 23:06:27Z msosnick $ */

/**
 * \file kcentersGPU.cc
 * \brief Test/example file for k-centers clustering on the GPU
 *
 * Test/example file for k-centers clustering on the GPU
 *
 * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 2/2/2010
 * \version 1.0
 **/


#ifdef HAVE_CONFIG_H
#include "../config.h"
#include "../campaign.h"
#else
#include "./kcentersGPU.h"
#endif

#include <time.h>

#define SHARED_MEM 1
//#define INTERCALATED_DATA 1

#include <iostream>
using namespace std;


/**
 * \brief Main for testing
 */
int main(int argc, const char* argv[])
{
    clock_t time_ini, time_end;
    // Initialization
    Timing timer;
    //Generator generator;
    
    // Parse command line
    //Defaults *defaults = new Defaults(argc, argv, "kc");
    
    // select device based on command-line switches
    //GpuDevices *systemGpuDevices = new GpuDevices(defaults);
    
    // place for data and information about data from datafile
    DataIO* data = new DataIO;

    timer.init("kcentersGPU");

    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    int d = atoi(argv[3]);

    const int seed = 0;
    data = new DataIO;
    //float* x = data->readData(defaults->getInputFileName().c_str());
    //float* x = data->makeData(n,k,d);

    /*int N = data->getNumElements();
    int K = data->getNumClusters();
    int D = data->getDimensions();*/

    int N = n;
    int K = k;
    int D = d;

    /*cout << "N,K,D: " << N << "," << K << "," << D << endl;
    FLOAT_TYPE* dist = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * (N+1024)); // cluster means
    for (int i = 0; i < N+1024; i++) dist[i] = FLT_MAX;
    int* centroids = (int*) malloc(sizeof(int) * K);  // centroid indices
    memset(centroids, 0, sizeof(int) * K);
    int* assign = (int*) malloc(sizeof(int) * (N+1024));     // assignments
    memset(assign, seed, sizeof(int) * (N+1024));*/
    
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
    int MaxTPB = 2048;
    int TPB = 64;
#else
    int MaxTPB = 1024;
    int TPB = 512;
#endif
    int copyN=N;
    int copyK=K;
    int copyD=D;

    float *x;
    //timer.start("kcentersGPU");

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
    
    cout << "N,K,D: " << N << "," << K << "," << D << endl;
    FLOAT_TYPE* dist = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * (N+1024)); // cluster means
    for (int i = 0; i < N+1024; i++) dist[i] = FLT_MAX;
    int* centroids = (int*) malloc(sizeof(int) * K);  // centroid indices
    memset(centroids, 0, sizeof(int) * K);
    int* assign = (int*) malloc(sizeof(int) * (N+1024));     // assignments
    memset(assign, seed, sizeof(int) * (N+1024));
    cout << "Punts fets";
    //time_ini = clock();

    for (int i = 0; i < N; i++) dist[i] = FLT_MAX;
    memset(centroids, 0, sizeof(int) * K);
    memset(assign, seed, sizeof(int) * N);

            // do clustering on GPU 
            cout << "kcentersGPU: BEGIN (N, K, D, TPB) " << N << " " << K << " " << D << " " << TPB << endl;
            timer.reset("kcentersGPU");
	    timer.start("kcentersGPU");
            kcentersGPU(TPB, N, K, D, x, assign, dist, centroids, seed, data);
	    timer.stop("kcentersGPU");

            //timer.report();
            cout << "kcentersGPU: END (N, K, D, TPB) " << N << " " << K << " " << D << " " << TPB << endl;
    	    timer.report("kcentersGPU"); 
    // free memory
    free(x);
    free(dist);
    free(centroids);
    free(assign);
          } /* K */
        } /* D */
        //N=N/tmp;
	N /= 10;
      } /* N */
    } /* TPB */

    //time_end = clock();

    //double secs = (double)(time_end - time_ini) / CLOCKS_PER_SEC;
    //printf("%.16g milisegundos\n", secs * 1000.0);
    //timer.stop("kcentersGPU");
    N=copyN; K=copyK; D=copyD;
    

    /*FLOAT_TYPE* dist = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * N); // cluster means
    for (int i = 0; i < N; i++) dist[i] = FLT_MAX;
    int* centroids = (int*) malloc(sizeof(int) * K);  // centroid indices
    memset(centroids, 0, sizeof(int) * K);
    int* assign = (int*) malloc(sizeof(int) * N);     // assignments
    memset(assign, seed, sizeof(int) * N);
    
    // do clustering on GPU 
    timer.start("kcentersGPU");
    kcentersGPU(TPB, N, K, D, x, assign, dist, centroids, seed, data);
    timer.stop("kcentersGPU");*/
   /* 
    // print results
    FLOAT_TYPE* ctr = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * K * D); // cluster centers
    // for each centroid
    for (int i = 0; i < K; i++)
        // for each dimension
        for (int d = 0; d < D; d++) 
            // collect centroid coordinates
#ifdef INTERCALATED_DATA
            ctr[i * D + d] = x[centroids[i] * D + d];
#else
            ctr[i * D + d] = x[d * N + centroids[i]];
#endif

    data->printClusters(N, K, D, x, ctr, assign);
    free(ctr); ctr = NULL;
   // if (data->doPrintTime()) timer.report(); 
   // if (defaults->getTimerOutput() == true) timer.report();*/
   //timer.report("kcentersGPU");
    
    
    // done
    cout << "Done clustering" << endl;
    
    return 0;
}
