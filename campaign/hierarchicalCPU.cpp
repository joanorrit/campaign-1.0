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
 * Portions copyright (c) 2010 Stanford University and the Authors.            *
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

/* $Id: hierarchicalCPU.cpp 158 2011-01-12 23:06:27Z msosnick $ */

/**
 * \file hierarchicalCPU.cpp
 * \brief A hierarchical clustering implementation for the CPU
 *
 * Implements hierarchical clustering on the CPU
 * 
 * \headerfile hierarchicalCPU.h 
 * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 2/2/2010
 * \version 1.0
 **/

#include "hierarchicalCPU.h"

using namespace std;



void computeFirstRound(int N, int D, FLOAT_TYPE* x, int* closestCtr, FLOAT_TYPE* closestDist)
{
    FLOAT_TYPE dist;
    // calculate lower diagonal matrix
    // for each data point
    for (unsigned int n = 1; n < N; n++) // no distance computed for data point with index 0
    {
        closestCtr[n] = -1;
        closestDist[n] = FLT_MAX;
        // for each data point (of smaller index)
        for (int j = 0; j < n; j++)
        {
            dist = 0.0;
            for (unsigned int d = 0; d < D; d++)
            {
                dist += distanceComponentCPU(x + n + d * N, x + j + d * N);
            }
            dist = distanceFinalizeCPU(1, &dist);
            // update closest neighbor info if necessary
            if (dist < closestDist[n]) 
            {
                closestCtr[n] = j;
                closestDist[n] = dist;
            }
        }
    }
}



int* hierarchicalCPU(int N, int D, FLOAT_TYPE *x)
{
    int*   seq         = (int*)   malloc(sizeof(int) * (N - 1) * 2); // sequential list of indices of merged pairs
    int*   map         = (int*)   malloc(sizeof(int) * N);           // maps indices to positions in x, faster update than moving data points in memory
    int*   clustID     = (int*)   malloc(sizeof(int) * N);           // indices of clusters
    int*   closestCtr  = (int*)   malloc(sizeof(int) * N);           // list of nearest neighbor indices
    FLOAT_TYPE* closestDist = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * N);         // list of nearest neighbor distances
    if (seq == NULL || map == NULL || clustID == NULL || closestCtr == NULL || closestDist == NULL)
    {
        cout << "Error in hierarchicalCPU(): Unable to allocate sufficient memory" << endl;
        exit(1);
    }
    
    int indexA, indexB, last, d, minA, minB, nextID = N - 1;
    FLOAT_TYPE dist;
    
    // implement neighbor heuristic
    // first step: calculate all N^2 distances (improve to N * (N - 1) / 2 distances)
    //             save closest neighbor index and distance: O(N^2)
    computeFirstRound(N, D, x, closestCtr, closestDist);
    
    // initialize first N clustID and map
    for (unsigned int n = 0; n < N; n++) { clustID[n] = n; map[n] = n; }
    // loop:
    // find shortest distance, insert new element (by replacing old), delete old element
    // indices for original data points from 0 to N - 1, for new data points from N to 2 * N - 2
    // last data point is at index N - 1
    last = N;
    
    // for each of N - 1 connections to be made: O(N), with inner loops becomes O(N^3) in worst case, in practice usually O(N^2)
    for (int n = 0; n < N - 1; n++)
    {
        // new clusters are given sequential IDs
        nextID++;
        // data points A and B are those at minimum distance
        minA = minB = 1;
        // last data point is stored at index N - n - 1, decreasing by one at each iteration
        last--;
        // Find minimum distance: O(N)
        // for each remaining data point (including new averages)
        for (int j = 2; j <= last; j++) // closestDist[0] unused, closestDist[1] is first reference
        {
            // if new minimum, update minB
            if (closestDist[j] < closestDist[minB]) minB = j;
        }
        // minB is element at larger index position, minA = closestCtr[minB] at smaller
        // calculate new average data point and store at position minA * D in x: O(1)
        minA = closestCtr[minB];
        indexA = map[minA];
        indexB = map[minB];
        
        // update sequence of merged clusters
        seq[2 * n] = clustID[minA]; seq[2 * n + 1] = clustID[minB];
        // for each dimension compute average
        for (unsigned int d = 0; d < D; d++)
        {
            x[indexA + d * N] = (x[indexA + d * N] + x[indexB + d * N]) / 2.0;
        }
        clustID[minA] = nextID;
        
        // try to reproduce output from GPU, requires additional distance calculations
        if (DOMATCHGPU)
        {
            // if element at minB not last, move cluster from last position to position minB
            if (minB != last)
            {
                map        [minB] = map        [last];
                closestDist[minB] = closestDist[last];
                clustID    [minB] = clustID    [last];
                closestCtr [minB] = closestCtr [last];
                // if closest cluster is at a position larger minB, trigger search for new nearest neighbor
                if (closestCtr[minB] > minB) closestCtr[minB] = minA;
            }
            int indexB = map[minB];
            // for all elements larger minB check if newly inserted data point is closest neighbor
            for (unsigned int j = minB + 1; j < last; j++)
            {
                int indexJ = map[j];
                dist = 0.0;
                for (unsigned int d = 0; d < D; d++)
                {
                    dist += distanceComponentCPU(x + indexJ + d * N, x + indexB + d * N);
                }
                dist = distanceFinalizeCPU(1, &dist);
                if (dist < closestDist[j])
                {
                    closestCtr [j] = minB;
                    closestDist[j] = dist;
                }
            }
        }
        // might change sequence of clustering when compared with GPU, but more efficient
        else
        {     
            // in map, shift all data points at positions > minB down by one, i.e. remove one position
            for (int j = minB; j < last; j++) map[j] = map[j + 1];
            
            // if minimum element not equal last in which case no shifting is necessary
            if (minB != last) 
            {
                for (int j = minB; j < last; j++)
                {
                    // shift
                    closestCtr[j] = closestCtr[j + 1];
                    if (closestCtr[j] > minB) closestCtr[j]--;
                    closestDist[j] = closestDist[j + 1];
                    clustID[j] = clustID[j + 1];
                }
            }
        }
        
        // for all data points at positions >= minA
        int indexJ, indexK;
        for (int j = minA; j < last; j++)
        {
            // if minA or minB was closest to data point at position j
            if (closestCtr[j] == minA || closestCtr[j] == minB || j == minA)
            {
                indexJ = map[j];
                closestCtr[j] = -1;
                closestDist[j] = FLT_MAX;
                // recalculate all minimum distances d(j, k) for data points at smaller clustID k < j
                for (int k = 0; k < j; k++)
                {
                    // if index unequal new data point (will be dealt with in next step)
                    if (k != minA)
                    {
                        indexK = map[k];
                        dist = 0.0;
                        for (unsigned int d = 0; d < D; d++)
                        {
                            dist += distanceComponentCPU(x + indexJ + d * N, x + indexK + d * N);
                        }
                        dist = distanceFinalizeCPU(1, &dist);
                        if (dist < closestDist[j])
                        {
                            closestCtr[j] = k;
                            closestDist[j] = dist;
                        }
                    }
                }
            }
        }      
        // for each data point with index > minA, check if new data point is closer than previous closest
        for (int j = minA + 1; j < last; j++)
        {
            indexJ = map[j];
            dist = 0.0;
            for (unsigned int d = 0; d < D; d++)
            {
                dist += distanceComponentCPU(x + indexA + d * N, x + indexJ + d * N);
            }
            dist = distanceFinalizeCPU(1, &dist);
            if (dist < closestDist[j])
            {
                closestCtr[j] = minA;
                closestDist[j] = dist;
            }
        }
    }
    // free memory
    free(map);
    free(clustID);
    free(closestCtr);
    free(closestDist);
    return seq;
}


