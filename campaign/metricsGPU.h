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

/* $Id: metricsGPU.h 158 2011-01-12 23:06:27Z msosnick $ */
/**
 * \file metricsGPU.h
 * \brief Distance calculation on GPU.
 *
 * \author Author: Kai J. Kolhoff, Contributors: Marc Sosnick, William Hsu
 * \date 5/26/10
 * \version pre-0.1
 */

#ifndef __CAMPAIGN_METRICS_GPU_H__
#define __CAMPAIGN_METRICS_GPU_H__

/**
 * \brief Calculates distance contribution along one dimension
 * It is often more efficient to calculate distances component-wise 
 * (i.e. dimension by dimension) to make optimal use of fast memory
 * (e.g. coalesced memory reads, use of shared memory)
 * Using component-wise calculation requires, for some metrics, the
 * call to distanceFinalize() to complete the distance calculation
 * \param elementA Coordinate A
 * \param elementB Coordinate B
 * \return 1D component of distance between element A and element B
 */
template <class T>
__device__ static T distanceComponentGPU(T *elementA, T *elementB)
{
    T dist = 0.0f;
#ifdef CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED
    // Euclidean distance squared
    dist = elementA[0] - elementB[0];
    dist = dist * dist;
#elif defined(CAMPAIGN_DISTANCE_EUCLIDEAN) // slow because of square root, use of CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED preferred
    // Euclidean distance
    dist = elementA[0] - elementB[0];
    dist = dist * dist;
#elif defined(CAMPAIGN_DISTANCE_MANHATTAN)
    // Manhattan distance
    dist = fabs(elementA[0] - elementB[0]); 
#elif defined(CAMPAIGN_DISTANCE_CHEBYSHEV)
    // Chebyshev distance
    dist = fabs(elementA[0] - elementB[0]);
#else
#error "No distance defined, add #define x to program's header with x = CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED, CAMPAIGN_DISTANCE_EUCLIDEAN, CAMPAIGN_DISTANCE_MANHATTAN, or CAMPAIGN_DISTANCE_CHEBYSHEV");
#endif
    return dist; 
}

/**
 * \brief Finalizes component-wise distance calculation
 * \param D Number of dimensions
 * \param components Pre-computed distance component(s)
 * \return Final distance
 */
template <class T>
__device__ static T distanceFinalizeGPU(int D, T *components)
{
    T dist = 0.0f;
#ifdef CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED
    // Euclidean distance squared
    for (unsigned int cnt = 0; cnt < D; cnt++) dist += components[cnt];
#elif defined(CAMPAIGN_DISTANCE_EUCLIDEAN) // slow because of square root, use of CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED preferred
    // Euclidean distance
    for (unsigned int cnt = 0; cnt < D; cnt++) dist += components[cnt];
    dist = sqrt(dist);
#elif defined(CAMPAIGN_DISTANCE_MANHATTAN)
    // Manhattan distance
    for (unsigned int cnt = 0; cnt < D; cnt++) dist += components[cnt]; 
#elif defined(CAMPAIGN_DISTANCE_CHEBYSHEV)
    // Chebyshev distance
    for (unsigned int cnt = 0; cnt < D; cnt++) dist = max(dist, components[cnt]);
#else
#error "No distance defined, add #define x to program's header with x = CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED, CAMPAIGN_DISTANCE_EUCLIDEAN, CAMPAIGN_DISTANCE_MANHATTAN, or CAMPAIGN_DISTANCE_CHEBYSHEV");
#endif
    return dist;
}

/**
 * \brief Computes distance between element A and element B
 * For best performance elements A and B should be in shared memory
 * \param D Number of dimensions
 * \param elementA Position data of element A
 * \param elementB Position data of element B 
 * \return Distance between element A and element B
 */
template <class T>
__device__ static T distanceGPU(int D, T *elementA, T *elementB) 
{
    T dist = 0.0f;
#ifdef CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED
    // Euclidean distance squared
    for (unsigned int cnt = 0; cnt < D; cnt++)
    {
        T di = (elementA[cnt] - elementB[cnt]);
        dist += di * di;
    }
#elif defined(CAMPAIGN_DISTANCE_EUCLIDEAN) // slow because of square root, use of CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED preferred
    // Euclidean distance
    for (unsigned int cnt = 0; cnt < D; cnt++)
    {
        T di = (elementA[cnt] - elementB[cnt]);
        dist += di * di;
    }
    dist = sqrt(dist);
#elif defined(CAMPAIGN_DISTANCE_MANHATTAN)
    // Manhattan distance
    for (unsigned int cnt = 0; cnt < D; cnt++)
    {
        dist += fabs(elementA[cnt] - elementB[cnt]); 
    }
#elif defined(CAMPAIGN_DISTANCE_CHEBYSHEV)
    // Chebyshev distance
    for (unsigned int cnt = 0; cnt < D; cnt++)
    {
        dist = max(dist, fabs(elementA[cnt] - elementB[cnt]));
    }
#else
#error "No distance defined, add #define x to program's header with x = CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED, CAMPAIGN_DISTANCE_EUCLIDEAN, CAMPAIGN_DISTANCE_MANHATTAN, or CAMPAIGN_DISTANCE_CHEBYSHEV");
#endif
    return dist; 
    
}

#endif
