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

/* $Id: kmeansCPU.cpp 158 2011-01-12 23:06:27Z msosnick $ */

/**
 * \file kmeansCPU.cpp
 * \brief A CUDA K-means implementation for the CPU
 *
 * Implements K-means clustering on the CPU
 * 
 * \author Author: Kai Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 12/2/2010
 * \version 1.0
 **/

#include "kmeansCPU.h"

using namespace std;

void assignToClustersKMCPU(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN)
{
    // for each element
    for (unsigned int n = 0; n < N; n++)
    {
        // find closest centroid
        int closestCenter = 0;
        FLOAT_TYPE dist, bestDistance = 0.0;
        // for each centroid
        for (unsigned int k = 0; k < K; k++)
        {
            dist = 0.0;
            for (unsigned int d = 0; d < D; d++)
            {
                // calculate distance component in dimension d
                dist += distanceComponentCPU(CTR + k * D + d, X + d * N + n);
            }
            dist = distanceFinalizeCPU(1, &dist);
            
            // if new centroid closer to element than previous best, reassign
            if (k == 0 || dist < bestDistance)
            {
                bestDistance  = dist;
                closestCenter = k;
            }     
        }
        ASSIGN[n] = closestCenter;
    }
}


FLOAT_TYPE calcScore(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN)
{
    FLOAT_TYPE score = 0.0;
    // for each element
    for (unsigned int n = 0; n < N; n++)
    {
        // read assignment
        unsigned int k = ASSIGN[n];
        FLOAT_TYPE scr = 0.0;
        // for each dimension 
        for (unsigned int d = 0; d < D; d++)
        {
            // compute score contribution in dimension d
            scr += distanceComponentCPU(X + d * N + n, CTR + k * D + d); 
        }
        // update score
        score += distanceFinalizeCPU(1, &scr);
    }
    return score;
}


void calcCentroids(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN)
{
    int* numElements = (int*)   malloc(sizeof(int)   * K); // number of elements in each cluster
    memset(numElements, 0, sizeof(int) * K);
    // for each data point
    for (unsigned int n = 0; n < N; n++)
    {
        // read assignment
        unsigned int k = ASSIGN[n]; // --> We take the cluster where n was assign.
        // only resets centroid if data is assigned to it
        // this avoids centering all empty clusters at origin
        if (numElements[k] == 0)
        {
            memset(CTR + k * D, 0, sizeof(FLOAT_TYPE) * D); 
        }
        // for each dimension
        for (unsigned int d = 0; d < D; d++)
        {
            // if element i assigned to current centroid, sum up location of
            // element at that index for computation of new centroid location
            CTR[k * D + d] += X[d * N + n];
        }
        numElements[k]++;
    }
    // finalize update of centroid position
    // for each cluster center
    for (unsigned int k = 0; k < K; k++)
    {
        // if any data points assigned to cluster
        if (numElements[k] > 0)
        {
            FLOAT_TYPE numEl = (FLOAT_TYPE) numElements[k];
            // compute average for each dimension
            for (unsigned int d = 0; d < D; d++) CTR[k * D + d] /= numEl; 
        }
    }
}


FLOAT_TYPE kmeansCPU(int N, int K, int D, FLOAT_TYPE *x, FLOAT_TYPE *ctr, int *assign, unsigned int maxIter)
{
    // initialize scores
    FLOAT_TYPE oldscore = -1000.0, score = 0.0;
    if (maxIter < 1) maxIter = INT_MAX;
    unsigned int iter = 0;
    // loop until defined number of iterations reached or converged
    while (iter < maxIter && ((score - oldscore) * (score - oldscore)) > EPS)
    {
        oldscore = score;
        
        // compute new centroids
        if (iter > 0) calcCentroids(N, K, D, x, ctr, assign);
        iter++;
        
        // update assignments of data points to clusters
        assignToClustersKMCPU(N, K, D, x, ctr, assign); 
        
        // compute new score for test of convergence
        score = calcScore(N, K, D, x, ctr, assign);
    }
    cout << "Number of iterations: " << iter << endl;
    return score;
}

