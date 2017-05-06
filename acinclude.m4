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

dnl $Id: acinclude.m4 159 2011-01-13 22:54:23Z msosnick $

dnl ## ----------------------------------------------------
dnl ## acinclude.m4
dnl ##
dnl ## Functions for autoconf for CAMPAIGN installation
dnl ##
dnl ## ----------------------------------------------------

AC_DEFUN([CAMP_CHECK_INIT],[
	AS_BOX(CAMPAGIN Verifying CUDA Installation)
])

dnl CAMP_CHECK_OPTIONS
dnl Checks to see if either 
dnl   --with-cuda-dev=dev_pathname 
dnl   --with-cuda-lib=lib_pathname
dnl   --with-cuda-dylib=dylib_pathname
dnl have been set, where 
dnl dev_pathname is the path to the installation directory of 
dnl              the NVIDIA CUDA development tools, 
dnl lib_pathname is the path to where the CUDA libraries 
dnl              are installed
dnl dylib_pathname is the path to where the CUDA dynamic
dnl                libraries are installed.
dnl If none are present, nothing happens.
dnl If any or all are present, the value is stored in
dnl CUDA_DEV_OPTION_PATHNAME, CUDA_LIB_OPTION_PATHNAME, and
dnl CUDA_DYLIB_OPTION_PATHNAME respectively.
dnl
dnl TODO: CAMP_CHECK_OPTIONS: Check to see that requisite files are located in these directories.
dnl
AC_DEFUN([CAMP_CHECK_OPTIONS],[

  AC_MSG_CHECKING([configure command line options])


  AC_ARG_WITH([cuda-dev],
    AS_HELP_STRING([--with-cuda-dev], [Specify root directory of CUDA development tools]),
    [ AS_IF( [test "x$with_cuda_dev" = xyes],
   	     	[ AC_MSG_ERROR([--with-cuda-dev must include a directory specification])],
	     [test "x$with_cuda_dev" != xno], 
		[ CUDA_DEV_OPTION_PATHNAME=$with_cuda_dev ],
	     [AC_MSG_ERROR([--without-cuda-dev is an invalid switch])])
    ],
    []) 


  AC_ARG_WITH([cuda-lib],
    AS_HELP_STRING([--with-cuda-lib], [Specify directory of CUDA libraries]),
    [ AS_IF( [test "x$with_cuda_lib" = xyes],
   	     	[ AC_MSG_ERROR([--with-cuda-lib must include a directory specification])],
	     [test "x$with_cuda_lib" != xno], 
		[ CUDA_LIB_OPTION_PATHNAME=$with_cuda_lib ],
	     [AC_MSG_ERROR([--without-cuda-lib is an invalid switch])])
    ],
    []) 


  AC_ARG_WITH([cuda-dylib],
    AS_HELP_STRING([--with-cuda-dylib], [Specify location of CUDA dytnamic libraries]),
    [ AS_IF( [test "x$with_cuda_dylib" = xyes],
   	     	[ AC_MSG_ERROR([--with-cuda-dylib must include a directory specification])],
	     [test "x$with_cuda_dylib" != xno], 
		[ CUDA_DYLIB_OPTION_PATHNAME=$with_cuda_dylib ],
	     [AC_MSG_ERROR([--without-cuda-dylib is an invalid switch])])
    ],
    []) 

    AC_MSG_RESULT([ok])
])


dnl
dnl  CAMP_CHECK_LIBRARY_PATHS_SET
dnl  
dnl  Checks that LD_LIBRARY_PATH and DYLD_LIBRARY_PATH are both set
dnl  Exits with error message if either not set
dnl
AC_DEFUN([CAMP_CHECK_LIBRARY_PATHS_SET],[

  AC_MSG_CHECKING([that DYLD_LIBRARY_PATH is set])
  AS_IF([test $DYLD_LIBRARY_PATH],
    [AC_MSG_RESULT([yes])],
    [AC_MSG_ERROR([DYLD_LIBRARY_PATH is not set.])
    ])

  AC_MSG_CHECKING([that LD_LIBRARY_PATH is set])
  AS_IF([test $LD_LIBRARY_PATH], 
    [AC_MSG_RESULT([yes])],
    [AC_MSG_ERROR([LD_LIBRARY_PATH is not set.])
    ])
])


dnl CAMP_GET_CUDA_INCLUDE_PATHS
dnl 
dnl Sets CAMP_
AC_DEFUN([CAMP_GET_CUDA_INCLUDE_PATHS],[

  AS_BOX([in camp get cuda include paths])

  AS_CASE( [$host],
           [*darwin*],[ echo "You're on a mac" 
			AS_IF([test $CUDA_DEV_OPTION_PATHNAME],
			      [CAMP_BASE_INCLUDE_PATH=$CUDA_DEV_OPTION_PATHNAME],
			      [CAMP_BASE_INCLUDE_PATH="/Developer/GPU\ Computing"]
			)
      	              ],
           [*linux*],[ echo "You're on a linux box"
			AS_IF([test $CUDA_DEV_OPTION_PATHNAME],
			      [CAMP_BASE_INCLUDE_PATH=$CUDA_DEV_OPTION_PATHNAME],
			      [CAMP_BASE_INCLUDE_PATH="$HOME/NVIDIA_GPU_Computing_SDK"]
		     	)
                     ],
           [echo "Unknown system type"]
         )
   AC_SUBST([CUDA_DEV_PATH],[$CAMP_BASE_INCLUDE_PATH])

   AS_BOX([ $CAMP_BASE_INCLUDE_PATH ])

])


dnl CAMP_GET_CUDA_LIBRARY_PATHS
dnl Checks to see if CUDA_LIB_OPTION_PATHNAME or 
dnl CUDA_DYLIB_OPTION_PATHNAME are set, and if so sets
dnl CAMP_CUDA_LD_PATH and CAMP_CUDA_DYLD_PATH respectively.
dnl If either which are not set, checks to see that LD_LIBRARY_PATH 
dnl or DYLD_LIBRARY_PATH (respectively) have a directory containing 
dnl the word cuda.  This is then directory CAMP_CUDA_LD_PATH and 
dnl CAMP_CUDA_DYLD_PATH, respectively.
dnl
AC_DEFUN([CAMP_GET_CUDA_LIBRARY_PATHS], [

  AC_MSG_CHECKING([DYLD_LIBRARY_PATH])
  AS_IF([test $CUDA_DYLIB_OPTION_PATHNAME],
        [AC_SUBST([CUDA_DYLD_PATH],[$CUDA_DYLIB_OPTION_PATHNAME])
	 AC_MSG_RESULT([yes (--with-cuda-dylib)])
	],
        [ pathRemainder=$DYLD_LIBRARY_PATH
          while test $pathRemainder
          do
            tempPathString=${pathRemainder%%$PATH_SEPARATOR*} 
            pathRemainder=${pathRemainder#*$PATH_SEPARATOR}
            if [[ "`expr "$tempPathString" : '.*cuda*'`" != 0 ]] 
            then  
              AC_SUBST([CUDA_DYLD_PATH],[$tempPathString])
              AC_MSG_RESULT([yes (DYLD_LIBRARY_PATH)])
              break
            else
              AC_MSG_ERROR([cannot find cuda subdirectory in DYLD_LIBRARY_PATH.])
            fi 
          done
       ])

  AC_MSG_CHECKING([LD_LIBRARY_PATH])
  AS_IF([test $CUDA_LIB_OPTION_PATHNAME],
        [AC_SUBST([CUDA_LD_PATH],[$CUDA_LIB_OPTION_PATHNAME])
	 AC_MSG_RESULT([yes (--with-cuda-lib)])
	],
        [ pathRemainder=$LD_LIBRARY_PATH
          while test $pathRemainder
          do
            tempPathString=${pathRemainder%%$PATH_SEPARATOR*} 
            pathRemainder=${pathRemainder#*$PATH_SEPARATOR}
            if [[ "`expr "$tempPathString" : '.*cuda*'`" != 0 ]] 
            then  
              AC_SUBST([CUDA_LD_PATH],[$tempPathString])
              AC_MSG_RESULT([yes (LD_LIBRARY_PATH)])
              break
            else
              AC_MSG_ERROR([cannot find cuda subdirectory in LD_LIBRARY_PATH])
            fi 
          done
       ])
 
  AS_UNSET(pathRemainder)
  AS_UNSET(tempPathString)

  AC_MSG_NOTICE([cuda LD path: $CUDA_LD_PATH])
  AC_MSG_NOTICE([cuda DYLD path: $CUDA_DYLD_PATH])

])

dnl
dnl  Checks to see if 64 is in the cuda LD or DYLD path.  If it is,
dnl  assunes that installation is 64 bit, otherwise assumes 32 bit.
dnl  CUDA_LD_BITWIDTH and CUDA_DYLD_BITWITDH are set to either 64 or
dnl  32 respectively.

AC_DEFUN([CAMP_CHECK_LINK_BITWIDTH],[

  AC_MSG_CHECKING([if DYLD is 32 or 64 bit installation])
  if [[ "`expr "$CUDA_DYLD_PATH" : '.*64*'`" != 0 ]] 
  then
    CUDA_DYLD_BITWIDTH=64
    AC_MSG_RESULT([64])
  else
    CUDA_DYLD_BITWIDTH=32
    AC_MSG_RESULT([32])
  fi

  AC_MSG_CHECKING([if LD is 32 or 64 bit installation])
  if [[ "`expr "$CUDA_LD_PATH" : '.*64*'`" != 0 ]] 
  then
    CUDA_LD_BITWIDTH=64
    AC_MSG_RESULT([64])
  else
    CUDA_LD_BITWIDTH=32
    AC_MSG_RESULT([32])
  fi

])


dnl
dnl  CAMP_CHECK_CUDA_INSTALL(message string, return value) 
dnl
dnl  Outputs message string as a check message.  Checks to see if cuda
dnl  is installed at the location contained in CUDA_BASE_DIR, by checking
dnl  for CUDA_BASE_DIR/include, CUDA_BASE_DIR/lib, and CUDA_BASE_DIR/bin.
dnl  If it is, displays yes and returns yes in return value.  
dnl  If it is not, displays no and returns no in return value.
dnl
AC_DEFUN([CAMP_CHECK_CUDA_INSTALL],[

	$2=yes

	dnl find out if these files exist
	AC_CHECK_FILE([$CUDA_BDIR/nvcc],[ $2=$$2 ],[ $2=no ])
	AC_CHECK_FILE([$CUDA_LDIR/libcuda.dylib],[ $2=$$2 ],[ $2=no ])
	AC_CHECK_FILE([$CUDA_LDIR/libcudart.dylib],[ $2=$$2 ],[ $2=no ])
	AC_CHECK_FILE([$CUDA_IDIR/cuda.h],[ $2=$$2 ],[ $2=no ])

	AC_MSG_CHECKING([$1])
	AS_IF([test $$2 = yes],[ AC_MSG_RESULT(yes) ],[ AC_MSG_RESULT(no)])
])


dnl
dnl  Tries to find cuda installation in some default locations.  Used
dnl  if cuda installation location is not specified with --with-cuda=
dnl
AC_DEFUN([CAMP_FIND_CUDA_INSTALL],[
	AC_MSG_NOTICE([No cuda install directory specified.  Auto finding cuda installation])
	CUDA_BASE_DIR="/usr/local/cuda"
        CUDA_IDIR="$CUDA_BASE_DIR/include"
        CUDA_LDIR="$CUDA_BASE_DIR/lib"
        CUDA_BDIR="$CUDA_BASE_DIR/bin"
	CAMP_CHECK_CUDA_INSTALL([in $CUDA_BASE_DIR])
])
