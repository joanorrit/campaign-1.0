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

/* $Id: somCPU.cpp 158 2011-01-12 23:06:27Z msosnick $ */
/**
 * \file somCPU.cpp
 * \brief A self-organizing map (som) implementation for the CPU
 *
 * Implements self-organizing map (som) clustering on the CPU
 * 
 * \author Author: Kai Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 2/2/2010
 * \version 1.0
 **/

#include "somCPU.h"

using namespace std;

void updateWeightVector(int D, FLOAT_TYPE scale, FLOAT_TYPE d0, FLOAT_TYPE *bmu, FLOAT_TYPE **pwv, FLOAT_TYPE *inputV)
{
    FLOAT_TYPE *wv = *pwv;
    FLOAT_TYPE dist = distanceCPU(D, bmu, wv);
    scale *= pow((FLOAT_TYPE) 0.25, dist / d0);
    for (unsigned int d = 0; d < D; d++)
    {
        wv[d] += scale * (inputV[d] - wv[d]);
    }
    *pwv = wv;
}


void somCPU(int N, int K, int D, int numIter, FLOAT_TYPE *x, FLOAT_TYPE **pwv)
{
    FLOAT_TYPE *wv = *pwv;
    
    // allocate host memory
    FLOAT_TYPE* inpV   = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * D);
    FLOAT_TYPE* vect   = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * D);
    FLOAT_TYPE* currWV = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * D);
    
    // for each iteration
    for (unsigned int iter = 0; iter < numIter; iter++)
    {
        // learning restraint, between 1.0 and 0.0 and continuously decreasing
        FLOAT_TYPE learningRestraint = 1.0f - (FLOAT_TYPE) iter / (FLOAT_TYPE) numIter;
        // for each input vector
        for (unsigned int v = 0; v < N; v++)
        {
            // set input vector
            for (unsigned int d = 0; d < D; d++) inpV[d] = x[d * N + v];
            
            // find best matching unit (bmu) among weight vectors
            FLOAT_TYPE dist, minDist = FLT_MAX;
            int bmuID = 0;
            
            // for each weight vector
            for (unsigned int k = 0; k < K; k++)
            {
                // for each dimension, copy weight vector
                for (unsigned int d = 0; d < D; d++) vect[d] = wv[d * K + k];
                
                // compute distance to find closest weight vector (best matching unit, bmu)
                dist = distanceCPU(D, inpV, vect);
                if (dist < minDist)
                {
                    bmuID = k;
                    minDist = dist;
                }
            }
            // set best matching unit
            for (unsigned int d = 0; d < D; d++) vect[d] = wv[d * K + bmuID];
            
            // for each weight vector
            for (unsigned int k = 0; k < K; k++)
            {
                // for each dimension, copy weight vector
                for (unsigned int d = 0; d < D; d++) currWV[d] = wv[d * K + k];
                
                // update weight vector (do this for all weight vectors)
                // uses scaling, but no cut-off to determine bmu neighborhood
                updateWeightVector(D, learningRestraint, minDist, vect, &currWV, inpV);
                // write back weight vector
                for (unsigned int d = 0; d < D; d++) wv[d * K + k] = currWV[d];
            }
        }
    }
    
    // free memory
    free(inpV);
    free(vect);
    free(currWV);
    
    *pwv = wv;
}


