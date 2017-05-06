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

/* $Id: kmeansCPU.h 158 2011-01-12 23:06:27Z msosnick $ */

/**
 * \file kmeansCPU.h
 * \brief A CUDA K-means implementation for the CPU
 *
 * Implements K-means clustering on the CPU
 * 
 * \author Author: Kai Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 12/2/2010
 * \version 1.0
 **/


#ifndef HAVE_CONFIG_H
// defined globally in campaign.h
// define distance metric, e.g. CAMPAIGN_DISTANCE_MANHATTAN, CAMPAIGN_DISTANCE_EUCLIDEAN_, etc.
#define CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED /** < Type of distance metric */
#define FLOAT_TYPE float         /** < Precision of floating point numbers */

#include <iostream>
#include "../util/dataio.h"
#include "../util/timing.h"
#include "../util/defaults.h"
#include "../util/metricsCPU.h"

#else
#include "../config.h"
#include "../campaign.h"
#endif

#define EPS 0.01            /** < Epsilon for convergence criteria */

using namespace std;

/**
 * \brief Assign data points to clusters
 * Runtime O(K*D*N)
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param CTR Centroid positions for next iteration
 * \param ASSIGN Assignments of data points to clusters
 * \return Updated values of ASSIGN
 */
void assignToClustersKMCPU(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN);

/**
 * \brief Compute score for current assignment
 * Runtime O(D*N)
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param CTR Centroid positions for next iteration
 * \param ASSIGN Assignments of data points to clusters
 * \return Score for current assignment
 */ 
FLOAT_TYPE calcScore(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN);



/**
 * \brief Compute new centroids
 * Runtime O(D*N)
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param CTR Cluster center positions for next iteration
 * \param ASSIGN Assignments of data points to clusters
 * \return Updated values of CTR
 */ 
void calcCentroids(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN);


/**
 * \brief Runs K-means on the CPU
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param x Clustering input data
 * \param ctr Centroid positions
 * \param assign Assignments of data points to clusters
 * \param maxIter Maximum number of iterations
 * \return Score for clustering after convergence and updated values of ctr and assign
 */ 
FLOAT_TYPE kmeansCPU(int N, int K, int D, FLOAT_TYPE *x, FLOAT_TYPE *ctr, int *assign, unsigned int maxIter);
