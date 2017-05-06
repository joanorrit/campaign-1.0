# --------------------------------------------------------------------------- #
#                                  CAMPAIGN                                   #
# --------------------------------------------------------------------------- #
# This is part of the CAMPAIGN data clustering library originating from       #
# Simbios, the NIH National Center for Physics-Based Simulation of Biological #
# Structures at Stanford, funded under the NIH Roadmap for Medical Research,  #
# grant U54 GM072970 (See https://simtk.org), and the FEATURE Project at      #
# Stanford, funded under the NIH grant LM05652                                #
# (See http://feature.stanford.edu/index.php).                                #
#                                                                             #
# Portions copyright (c) 2010 Stanford University, Authors and Contributors.  #
# Authors: Marc Sosnick                                                       #
# Contributors: Kai J. Kolhoff, Wiliam Hsu                                    #
#                                                                             #
# This program is free software: you can redistribute it and/or modify it     #
# under the terms of the GNU Lesser General Public License as published by    #
# the Free Software Foundation, either version 3 of the License, or (at your  #
# option) any later version.                                                  #
#                                                                             #
# This program is distributed in the hope that it will be useful, but WITHOUT #
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       #
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public        #
# License for more details.                                                   #
#                                                                             #
# You should have received a copy of the GNU Lesser General Public License    #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
# --------------------------------------------------------------------------- #

# make sure examples gets copied

ACLOCAL_AMFLAGS = -I .
AUTOMAKE_OPTIONS = subdir-objects
ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS}

# one library for each distance metric
lib_LTLIBRARIES = libcampaign-euclideansquared-@CAMPAIGN_API_VERSION@.la \
		  libcampaign-euclidean-@CAMPAIGN_API_VERSION@.la \
		  libcampaign-manhattan-@CAMPAIGN_API_VERSION@.la \
		  libcampaign-chebyshev-@CAMPAIGN_API_VERSION@.la


## library source files - not .cu files
CAMPAIGN_SOURCE_FILES = campaign/dataio.cpp \
			campaign/defaults.cpp \
			campaign/gpudevices.cpp \
			campaign/hierarchicalCPU.cpp \
			campaign/kcentersCPU.cpp \
			campaign/kmeansCPU.cpp \
			campaign/kmedoidsCPU.cpp \
			campaign/timing.cpp \
			campaign/somCPU.cpp \
			campaign/tokens.cpp

## each .cu file is compiled to a .cu.cpp file by using the nvcc -cuda
## flag.  The .cpp file is then compiled by the system compiler.  The
## name before the .cu.cpp extension flags the Makefile to include the
## correct -D flag for the particular distance metric.  At this point,
## one CAMPAIGN_CUSOURCE_ is required for every distance metric.


EXTRA_DIST = examples \
             campaign/hierarchicalGPU.cu \
             campaign/kcentersGPU.cu \
             campaign/kmeansGPU.cu \
             campaign/kmedoidsGPU.cu \
             campaign/kpsmeansGPU.cu \
             campaign/min.cu \
             campaign/rmsdGPU.cu \
             campaign/somGPU.cu 



CAMPAIGN_CUSOURCE_EUCLIDEANSQUARED = \
			campaign/hierarchicalGPU.euclideansquared.cu.cpp \
			campaign/kcentersGPU.euclideansquared.cu.cpp \
			campaign/kmeansGPU.euclideansquared.cu.cpp \
			campaign/kmedoidsGPU.euclideansquared.cu.cpp \
			campaign/kpsmeansGPU.euclideansquared.cu.cpp \
			campaign/min.euclideansquared.cu.cpp \
			campaign/rmsdGPU.euclideansquared.cu.cpp \
			campaign/somGPU.euclideansquared.cu.cpp

CAMPAIGN_CUSOURCE_EUCLIDEAN = \
			campaign/hierarchicalGPU.euclidean.cu.cpp \
			campaign/kcentersGPU.euclidean.cu.cpp \
			campaign/kmeansGPU.euclidean.cu.cpp \
			campaign/kmedoidsGPU.euclidean.cu.cpp \
			campaign/kpsmeansGPU.euclidean.cu.cpp \
			campaign/min.euclidean.cu.cpp \
			campaign/rmsdGPU.euclidean.cu.cpp \
			campaign/somGPU.euclidean.cu.cpp

CAMPAIGN_CUSOURCE_MANHATTAN = \
			campaign/hierarchicalGPU.manhattan.cu.cpp
			campaign/kcentersGPU.manhattan.cu.cpp \
			campaign/kmeansGPU.manhattan.cu.cpp \
			campaign/kmedoidsGPU.manhattan.cu.cpp \
			campaign/kpsmeansGPU.manhattan.cu.cpp \
			campaign/min.manhattan.cu.cpp \
			campaign/rmsdGPU.manhattan.cu.cpp \
			campaign/somGPU.manhattan.cu.cpp

CAMPAIGN_CUSOURCE_CHEBYSHEV = \
			campaign/hierarchicalGPU.chebyshev.cu.cpp
			campaign/kcentersGPU.chebyshev.cu.cpp \
			campaign/kmeansGPU.chebyshev.cu.cpp \
			campaign/kmedoidsGPU.chebyshev.cu.cpp \
			campaign/kpsmeansGPU.chebyshev.cu.cpp \
			campaign/min.chebyshev.cu.cpp \
			campaign/rmsdGPU.chebyshev.cu.cpp \
			campaign/somGPU.chebyshev.cu.cpp

CAMPAIGN_HEADER_FILES = campaign.h \
			campaign/dataio.h \
			campaign/defaults.h \
			campaign/gpudevices.h \
			campaign/hierarchicalCPU.h \
			campaign/hierarchicalGPU.h \
			campaign/kcentersCPU.h \
			campaign/kcentersGPU.h \
			campaign/kmeansCPU.h \
			campaign/kmeansGPU.h \
			campaign/kmedoidsCPU.h \
			campaign/kmedoidsGPU.h \
			campaign/kpsmeansGPU.h \
			campaign/metricsCPU.h \
			campaign/metricsGPU.h \
			campaign/min.h \
			campaign/rmsdGPU.h \
			campaign/somCPU.h \
			campaign/somGPU.h \
			campaign/timing.h \
			campaign/tokens.h 


## .cu.cpp files are generated with nvcc -cuda and linked with standard library
libcampaign_euclideansquared_@CAMPAIGN_API_VERSION@_la_SOURCES = $(CAMPAIGN_SOURCE_FILES)
# don't distribute intermediate files.
nodist_libcampaign_euclideansquared_@CAMPAIGN_API_VERSION@_la_SOURCES = $(CAMPAIGN_CUSOURCE_EUCLIDEANSQUARED)
libcampaign_euclideansquared_@CAMPAIGN_API_VERSION@_la_LDFLAGS = -version-info $(CAMPAIGN_SO_VERSION) -L@CUDA_LD_PATH@
libcampaign_euclideansquared_@CAMPAIGN_API_VERSION@_la_CPPFLAGS =  -I@CUDA_LD_PATH@/../include \
						  -I@CUDA_DEV_PATH@/C/common/inc/ \
						  $(AM_CFLAGS) \
						  -DCAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED


libcampaign_euclidean_@CAMPAIGN_API_VERSION@_la_SOURCES = $(CAMPAIGN_SOURCE_FILES)
nodist_libcampaign_euclidean_@CAMPAIGN_API_VERSION@_la_SOURCES = $(CAMPAIGN_CUSOURCE_EUCLIDEAN)
libcampaign_euclidean_@CAMPAIGN_API_VERSION@_la_LDFLAGS = -version-info $(CAMPAIGN_SO_VERSION) -L@CUDA_LD_PATH@
libcampaign_euclidean_@CAMPAIGN_API_VERSION@_la_CPPFLAGS =  -I@CUDA_LD_PATH@/../include \
						  -I@CUDA_DEV_PATH@/C/common/inc/ \
						  $(AM_CFLAGS) \
						  -DCAMPAIGN_DISTANCE_EUCLIDEAN


libcampaign_manhattan_@CAMPAIGN_API_VERSION@_la_SOURCES = $(CAMPAIGN_SOURCE_FILES) 
nodist_libcampaign_manhattan_@CAMPAIGN_API_VERSION@_la_SOURCES = $(CAMPAIGN_CUSOURCE_MANHATTAN)
libcampaign_manhattan_@CAMPAIGN_API_VERSION@_la_LDFLAGS = -version-info $(CAMPAIGN_SO_VERSION) -L@CUDA_LD_PATH@
libcampaign_manhattan_@CAMPAIGN_API_VERSION@_la_CPPFLAGS =  -I@CUDA_LD_PATH@/../include \
						  -I@CUDA_DEV_PATH@/C/common/inc/ \
						  $(AM_CFLAGS) \
						  -DCAMPAIGN_DISTANCE_MANHATTAN

libcampaign_chebyshev_@CAMPAIGN_API_VERSION@_la_SOURCES = $(CAMPAIGN_SOURCE_FILES) 
nodist_libcampaign_chebyshev_@CAMPAIGN_API_VERSION@_la_SOURCES = $(CAMPAIGN_CUSOURCE_CHEBYSHEV)
libcampaign_chebyshev_@CAMPAIGN_API_VERSION@_la_LDFLAGS = -version-info $(CAMPAIGN_SO_VERSION) -L@CUDA_LD_PATH@
libcampaign_chebyshev_@CAMPAIGN_API_VERSION@_la_CPPFLAGS =  -I@CUDA_LD_PATH@/../include \
						  -I@CUDA_DEV_PATH@/C/common/inc/ \
						  $(AM_CFLAGS) \
						  -DCAMPAIGN_DISTANCE_CHEBYSHEV

campaign_includedir = $(includedir)/campaign-$(CAMPAIGN_API_VERSION)
nobase_campaign_include_HEADERS =	${CAMPAIGN_HEADER_FILES}

campaign_libincludedir = $(libdir)/campaign-$(EXAMPLE_API_VERSION)/include

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = campaign-$(CAMPAIGN_API_VERSION).pc


## .cu.cpp files are not automatically taken care of by autotools
mostlyclean-local:
	rm -rf campaign/*.cu.cpp

clean-local: 
	rm -rf campaign/*.cu.cpp


# instead of worrying about creating .lo files from .cu files,
# use nvcc to cross-compile to .cpp code, and include the cpp
# file as a .cu.cpp file as the source
#%.cu.cpp:
#	@echo "#########################################"
#	@echo @CUDA_DYLD_PATH@
#	@echo aM_CPPFLAGS $(AM_CPPFLAGS)
#	@echo _cPPFLAGS $(_CPPFLAGS)
#	@echo aM_CLFAGS $(AM_CFLAGS)
#	@echo "#########################################"
#	nvcc -cuda -g -I/Developer/GPU\ Computing/C/common/inc/ $*.cu -o $@
#	vcc -cuda -g $(AM_CPPFLAGS) $*.cu -o $@

%.euclideansquared.cu.cpp:
	nvcc -cuda -g -I. -I@CUDA_DEV_PATH@/C/common/inc/ $*.cu -o $@ -DCAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED -DHAVE_CONFIG_H -DNVCC_COMPILE

%.euclidean.cu.cpp:
	nvcc -cuda -g -I@CUDA_DEV_PATH@/C/common/inc/ $*.cu -o $@ -DCAMPAIGN_DISTANCE_EUCLIDEAN -DHAVE_CONFIG_H -DNVCC_COMPILE


%.manhattan.cu.cpp:
	nvcc -cuda -g -I@CUDA_DEV_PATH@/C/common/inc/ $*.cu -o $@ -DCAMPAIGN_DISTANCE_MANHATTAN -DHAVE_CONFIG_H -DNVCC_COMPILE

%.chebyshev.cu.cpp:
	nvcc -cuda -g -I@CUDA_DEV_PATH@/C/common/inc/ $*.cu -o $@ -DCAMPAIGN_DISTANCE_CHEBYSHEV -DHAVE_CONFIG_H -DNVCC_COMPILE

