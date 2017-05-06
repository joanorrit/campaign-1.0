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

/* $Id: kmedoidsCPU.h 158 2011-01-12 23:06:27Z msosnick $ */

/**
 * \file kmedoidsCPU.h
 * \brief A CUDA K-medoids implementation for the CPU
 *
 * Implements K-medoids clustering on the CPU
 * 
 * \author Authors: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 15/2/2010
 * \version 1.0
 **/


#ifndef HAVE_CONFIG_H
// defined globally in campaign.h
// define distance metric, e.g. CAMPAIGN_DISTANCE_MANHATTAN, CAMPAIGN_DISTANCE_EUCLIDEAN_, etc.
#define CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED /** < Type of distance metric */
#define FLOAT_TYPE float         /** < Precision of floating point numbers */

#include <iostream>
#include <ctime>
#include "../util/dataio.h"
#include "../util/timing.h"
#include "../util/defaults.h"
#include "../util/metricsCPU.h"

#else 
#include "../config.h"
#include "../campaign.h"
#endif

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
void assignToClustersKMDCPU(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN);


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
FLOAT_TYPE calcScore(int N, int num, int D, int *MAP, FLOAT_TYPE *X, FLOAT_TYPE *CTR);


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
FLOAT_TYPE kmedoidsCPU(int N, int K, int D, FLOAT_TYPE *x, int *medoid, int *assign, unsigned int maxIter);

