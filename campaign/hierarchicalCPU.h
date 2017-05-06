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
 * Portions copyright (c) 2010 Stanford University and the Authors.            * * Authors: Kai J. Kolhoff                                                     *
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

/* $Id: hierarchicalCPU.h 158 2011-01-12 23:06:27Z msosnick $ */

/**
 * \file hierarchicalCPU.h
 * \brief A hierarchical clustering implementation for the CPU
 *
 * Implements hierarchical clustering on the CPU
 * 
 * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 2/2/2010
 * \version 1.0
 **/


#ifdef HAVE_CONFIG_H
#include "../config.h"
#include "../campaign.h"
#else
// these are defined globally in library compilation
// define distance metric, e.g. CAMPAIGN_DISTANCE_MANHATTAN, CAMPAIGN_DISTANCE_EUCLIDEAN_, etc.
#define CAMPAIGN_DISTANCE_MANHATTAN         /** < Type of distance metric */
#define FLOAT_TYPE float         /** < Precision of floating point numbers */

#include <iostream>
#include <cfloat>
#include "../util/dataio.h"
#include "../util/timing.h"
#include "../util/metricsCPU.h"
#include "../util/defaults.h"

#endif

#define DOMATCHGPU false    /** < Try to reproduce output from GPU (not guaranteed due to hardware-dependent rounding errors; reduces performance notably) */


using namespace std;

/**
 * \brief Calculates N * (N - 1) / 2 distances and stores nearest neighbor for all N data points
 * \param N Number of data points
 * \param D Number of dimensions
 * \param x Clustering input data
 * \param closestCtr List of indices of nearest neighbors
 * \param closestDist List of distances to nearest neighbors
 * \return Returns updated closestCtr and closestDist
 */
void computeFirstRound(int N, int D, FLOAT_TYPE* x, int* closestCtr, FLOAT_TYPE* closestDist);


/**
 * \brief Performs hierarchical clustering on the CPU
 * \param N Number of data points
 * \param D Number of dimensions
 * \param x Clustering input data
 * \return List of pairs of cluster indices in order of merging
 */ 
int* hierarchicalCPU(int N, int D, FLOAT_TYPE *x);

