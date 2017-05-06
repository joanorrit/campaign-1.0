dnl ---------------------------------------------------------------------------
dnl                                  CAMPAIGN                                  
dnl ---------------------------------------------------------------------------
dnl This is part of the CAMPAIGN data clustering library originating from      
dnl Simbios, the NIH National Center for Physics-Based Simulation of Biological
dnl Structures at Stanford, funded under the NIH Roadmap for Medical Research, 
dnl grant U54 GM072970 (See https://simtk.org), and the FEATURE Project at     
dnl Stanford, funded under the NIH grant LM05652                               
dnl (See http://feature.stanford.edu/index.php).                               
dnl                                                                            
dnl Portions copyright (c) 2010 Stanford University, Authors and Contributors. 
dnl Authors: Marc Sosnick                                                      
dnl Contributors: Kai J. Kolhoff, William Hsu                                  
dnl                                                                            
dnl This program is free software: you can redistribute it and/or modify it    
dnl under the terms of the GNU Lesser General Public License as published by   
dnl the Free Software Foundation, either version 3 of the License, or (at your 
dnl option) any later version.                                                 
dnl                                                                            
dnl This program is distributed in the hope that it will be useful, but WITHOUT
dnl ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or      
dnl FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public       
dnl License for more details.                                                  
dnl                                                                            
dnl You should have received a copy of the GNU Lesser General Public License   
dnl along with this program.  If not, see <http://www.gnu.org/licenses/>.      
dnl ---------------------------------------------------------------------------

AC_INIT([campaign],[1.0],[marcsosnick@mac.com])
AC_PREREQ([2.59])

dnl suggested order
dnl AC_INIT(package, version, bug-report-address) 
dnl information on the package 
dnl checks for programs 
dnl checks for libraries
dnl checks for header files 
dnl checks for types 
dnl checks for structures 
dnl checks for compiler 
dnl characteristics checks for library functions
dnl checks for system services 
dnl AC_CONFIG_FILES([file...]) 
dnl AC_OUTPUT

dnl CHECKS I ----------------------------------------------------------
dnl n/a
dnl -------------------------------------------------------------------

dnl CHECKS II ---------------------------------------------------------
dnl The verification section verifies that the NVIDIA cuda toolkit is
dnl properly installed, and that the needed libraries are accessible
dnl -------------------------------------------------------------------

dnl initialize verification section
CAMP_CHECK_INIT()

AC_CANONICAL_BUILD()
AC_CANONICAL_HOST()
AC_CANONICAL_TARGET()


CAMP_GET_CUDA_INCLUDE_PATHS()


dnl verify that NVIDIA developer software has been installed
CAMP_CHECK_LIBRARY_PATHS_SET()	

dnl TODO:  Should explicitly check to see if nvcc program is working

CAMP_CHECK_OPTIONS()

dnl set library pathnames
CAMP_GET_CUDA_LIBRARY_PATHS()

dnl find CUDA installation directory 
AS_BOX([Checking include directory])





AS_BOX([checking distance metric])


AC_ARG_WITH([distance-metric],
		AS_HELP_STRING([--with-distance-metric],
			[Specify which metric to use (defaults to CAMPAIGN_DISTANCE_EUCLIDIAN)]),
			[
			
				AS_IF( 	[test "x$with_distance_metric" = xyes],
						[ AC_MSG_ERROR([--with-distance-metric must have a metric type defined])],
					[test "x$with_distance_metric" != xno], 
						[ AS_BOX([Distance metric NOT SET set to $with_distance_metric]) ],
					[AC_MSG_ERROR([--without-distance-metric is an invalid switch])])
			],
			[  AS_BOX([No --with-distance-metric specified ]) ]) 

dnl AC_DEFINE([_MANHATTAN_],[],[Manhattan distance])






AM_INIT_AUTOMAKE([1.10 -Wall no-define])

AC_CONFIG_HEADERS([config.h])
AC_PROG_CXX
AM_PROG_CC_C_O
AC_PROG_LIBTOOL
# LT_INIT([disable-static])
LT_INIT()

AC_SUBST([CAMPAIGN_SO_VERSION], [0:0:0])
AC_SUBST([CAMPAIGN_API_VERSION], [1.0])

AC_CONFIG_FILES([
		 Makefile
                 examples/Makefile
                 campaign-${CAMPAIGN_API_VERSION}.pc:campaign.pc.in
		])
AC_OUTPUT


