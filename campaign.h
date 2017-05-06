/* --------------------------------------------------------------------------*
*                                 CAMPAIGN(tm)                               *
* -------------------------------------------------------------------------- *
* This is part of the CAMPAIGN data clustering library originating from      *
* Simbios, the NIH National Center for Physics-Based Simulation of           *
* Biological Structures at Stanford, funded under the NIH Roadmap for        *
* Medical Research, grant U54 GM072970, and the FEATURE project at Stanford, *
* funded under the NIH grant LM05652.  See https://simtk.org and             *
* http://feature.stanford.edu/index.php                                      *
*                                                                            *
* Portions copyright (c) 2010 Stanford University, Authors and Contributors  *
* Authors:                                                                   *
* Contributors:                                                              *
*                                                                            *
* Permission is hereby granted, free of charge, to any person obtaining a    *
* copy of this software and associated documentation files (the "Software"), *
* to deal with the Software limited to the rights to use, copy, modify, merge*
* and publish. No redistribution, licensing or commercialization             *
* is allowed without written permission.                                     *
*                                                                            *
* Software is furnished subject to the following conditions:                 *
*                                                                            *
* The above copyright notice and this permission notice shall be included in *
* all copies or substantial portions of the Software.                        *
*                                                                            *
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
* THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
* DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
* USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
* -------------------------------------------------------------------------- */

#ifndef __CAMPAIGN_H__
#define __CAMPAIGN_H__

#define THREADSPERBLOCK 256
#define FLOAT_TYPE float

#include <iostream>
#include <cmath>
#include <cfloat>
#include <assert.h>
#include <sys/time.h>

#include <cuda.h>
//#include <cutil.h>
//#include <cutil_inline.h>

#include "campaign/dataio.h"
#include "campaign/defaults.h"
#include "campaign/gpudevices.h"
#include "campaign/hierarchicalCPU.h"
#include "campaign/hierarchicalGPU.h"
#include "campaign/kcentersCPU.h"
#include "campaign/kcentersGPU.h"
#include "campaign/kmeansCPU.h"
#include "campaign/kmeansGPU.h"
#include "campaign/kmedoidsCPU.h"
#include "campaign/kmedoidsGPU.h"
#include "campaign/kpsmeansGPU.h"
#include "campaign/metricsCPU.h"
#include "campaign/metricsGPU.h"
#include "campaign/min.h"
#include "campaign/rmsdGPU.h"
#include "campaign/somCPU.h"
#include "campaign/somGPU.h"
#include "campaign/timing.h"
#include "campaign/tokens.h"

#endif
