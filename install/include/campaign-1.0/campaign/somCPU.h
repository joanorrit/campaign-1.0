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

/* $Id: somCPU.h 158 2011-01-12 23:06:27Z msosnick $ */
/**
 * \file somCPU.h
 * \brief A self-organizing map (som) implementation for the CPU
 *
 * Implements self-organizing map (som) clustering on the CPU
 * 
 * \author Author: Kai Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 2/2/2010
 * \version 1.0
 **/

#ifndef HAVE_CONFIG_H 
// defined globally in campaign.h
// define distance metric, e.g. CAMPAIGN_DISTANCE_MANHATTAN, CAMPAIGN_DISTANCE_EUCLIDEAN_, etc.
#define CAMPAIGN_DISTANCE_MANHATTAN     /** < Type of distance metric */
#define FLOAT_TYPE float        /** < Precision of floating point numbers */

#include <iostream>
#include <cfloat>
#include <cmath>
#include "../util/dataio.h"
#include "../util/timing.h"
#include "../util/defaults.h"
#include "../util/metricsCPU.h"
#include "../util/defaults.h"

#else 
#include "../config.h"
#include "../campaign.h"
#endif

using namespace std;

/**
 * \brief Shifts weight vector closer towards input vector
 * \param D Number of dimensions
 * \param scale Scaling factor depending of distance between wv and best matching unit
 * \param d0 Distance from bmu to input vector
 * \param bmu Best matching unit
 * \param wv Weight vector
 * \param inputV Input vector
 * \return Updated weight vector wv
 */
void updateWeightVector(int D, FLOAT_TYPE scale, FLOAT_TYPE d0, FLOAT_TYPE *bmu, FLOAT_TYPE **pwv, FLOAT_TYPE *inputV);


/**
 * \brief Runs self organizing map on the CPU
 *
 * \param N Number of input vectors
 * \param K Number of weight vectors
 * \param D Number of dimensions
 * \param numIter Number of iterations to be carried out
 * \param x Input vectors
 * \param pwv Reference to weight vectors
 * \return Updated weight vectors after numIter iterations
 */ 
void somCPU(int N, int K, int D, int numIter, FLOAT_TYPE *x, FLOAT_TYPE **pwv);


