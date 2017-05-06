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

/* $Id: kmedoidsCPU.cpp 158 2011-01-12 23:06:27Z msosnick $ */

/**
 * \file kmedoidsCPU.cpp
 * \brief A CUDA K-medoids implementation for the CPU
 *
 * Implements K-medoids clustering on the CPU
 * 
 * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 15/2/2010
 * \version 1.0
 **/

#include "kmedoidsCPU.h"

using namespace std;

/**
 * \brief Assign data points to clusters
 * Runtime O(D*K*N)
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param CTR Medoid positions
 * \param ASSIGN Assignments of data points to clusters
 * \return Updated values of ASSIGN
 */
void assignToClustersKMDCPU(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN)
{
    // for each element
    for (unsigned int n = 0; n < N; n++)
    {
        // find closest medoid
        int closestCenter = 0;
        FLOAT_TYPE dist, bestDistance = 0.0;
        // for each medoid
        for (unsigned int k = 0; k < K; k++)
        {
            dist = 0.0;
            for (unsigned int d = 0; d < D; d++)
            {
                // calculate distance component in dimension d
                dist += distanceComponentCPU(CTR + k * D + d, X + d * N + n);
            }
            dist = distanceFinalizeCPU(1, &dist);
            
            // if new medoid closer to element than previous best, reassign
            if (k == 0 || dist < bestDistance)
            {
                bestDistance  = dist;
                closestCenter = k;
            }     
        }
        ASSIGN[n] = closestCenter;
    }
}

/**
 * \brief Compute score for a cluster
 * Runtime O(D*N)
 *
 * \param N Number of data points
 * \param num Number of data points in cluster
 * \param D Number of dimensions
 * \param MAP Data point indices for cluster
 * \param X Clustering input data
 * \param CTR Coordinates of medoid for given cluster
 * \return Score for selected cluster
 */ 
FLOAT_TYPE calcScore(int N, int num, int D, int *MAP, FLOAT_TYPE *X, FLOAT_TYPE *CTR)
{
    FLOAT_TYPE score = 0.0;
    // for each data point in cluster
    for (int n = 0; n < num; n++)
    {
        unsigned int index = MAP[n];
        FLOAT_TYPE scr = 0.0;
        // for each dimension
        for (unsigned int d = 0; d < D; d++)
        {
            // compute score contribution in dimension d    
            scr += distanceComponentCPU(X + d * N + index, CTR + d);
        }
        // update score
        score += distanceFinalizeCPU(1, &scr);
    }
    return score;
}

/**
 * \brief Runs K-medoids on the CPU
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param x Clustering input data
 * \param medoid Initial set of medoids
 * \param assign Assignments of data points to clusters
 * \param maxIter Maximum number of iterations
 * \return Score for assignment, new set of medoids in medoid and assignments in assign
 */ 
FLOAT_TYPE kmedoidsCPU(int N, int K, int D, FLOAT_TYPE *x, int *medoid, int *assign, unsigned int maxIter)
{
    // allocate memory
    int *map         = (int*)   malloc(sizeof(int)   * N);
    int *clusterSize = (int*)   malloc(sizeof(int)   * K);
    FLOAT_TYPE *ctr       = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * K * D);
    FLOAT_TYPE *newCtr    = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * D);
    
    // copy coordinates for initial set of medoids
    for (unsigned int k = 0; k < K; k++)
    {
        if (medoid[k] < 0 || medoid[k] >= N) 
        {
            cerr << "Error: medoid " << k << " (" << medoid[k] << ") does not map to data point" << endl;
            return 0.0;
        }
        for (unsigned int d = 0; d < D; d++) ctr[k * D + d] = x[d * N + medoid[k]];
    }
    
    // loop for defined number of iterations
    unsigned int iter = 0;
    while (iter < maxIter)
    {
        // assign data points to clusters
        assignToClustersKMDCPU(N, K, D, x, ctr, assign);
        
        // sort by assignment (O(K*N))
        unsigned int mapPos = 0;
        // for each cluster
        for (unsigned int k = 0; k < K; k++)
        {
            clusterSize[k] = 0;
            // for each data point
            for (unsigned int n = 0; n < N; n++)
            {
                // check assignment
                if (assign[n] == k) 
                {
                    // update cluster size and mapping
                    clusterSize[k]++;
                    map[mapPos] = n;
                    mapPos++;
                }
            }
        }
        
        // try new medoids
        unsigned int clusterOffset = 0;
        // for each cluster
        for (unsigned int k = 0; k < K; k++)
        {
            // swap medoid at random with one of its assigned data points
            unsigned int newMedoid = map[clusterOffset + rand() % clusterSize[k]];
            // cout << iter << ": Swapping medoid " << medoid[k] << " (cluster " << k << ") with data point " << newMedoid << endl;
            if (newMedoid != medoid[k])
            {    
                // copy coordinates of new medoid
                for (unsigned int d = 0; d < D; d++) newCtr[d] = x[newMedoid + d * N];
                
                // compute current score for cluster
                FLOAT_TYPE scoreBefore = calcScore(N, clusterSize[k], D, map + clusterOffset, x, ctr + k * D);
                
                // compute new score for cluster
                FLOAT_TYPE scoreAfter  = calcScore(N, clusterSize[k], D, map + clusterOffset, x, newCtr);
                
                // accept if better score; also accept equal score to improve sampling
                if (scoreAfter <= scoreBefore)
                {
                    // update medoid
                    medoid[k] = newMedoid;
                    for (unsigned int d = 0; d < D; d++) ctr[k * D + d] = newCtr[d];
                }
            }
            // move on to next cluster
            clusterOffset += clusterSize[k];
        }
        iter++;
    }
    // compute final score
    FLOAT_TYPE score = 0.0;
    unsigned int clusterOffset = 0;
    for (unsigned int k = 0; k < K; k++) 
    {
        score += calcScore(N, clusterSize[k], D, map + clusterOffset, x, ctr + k * D);
        clusterOffset += clusterSize[k];
    }
    
    // free memory
    free(map);
    free(clusterSize);
    free(ctr);
    free(newCtr);
    
    return score;
}

