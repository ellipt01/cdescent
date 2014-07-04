#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.59])
AC_INIT([libcdescent], [0.1.1])
AC_CONFIG_SRCDIR([example/example.h])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CC
AM_PROG_CC_C_O
AC_PROG_CPP
AC_PROG_CXX
AC_PROG_AWK
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_RANLIB
AC_PROG_LIBTOOL

AM_INIT_AUTOMAKE

# Checks for libraries.
AC_CHECK_LIB([m],[floor],[])
AC_CHECK_LIB([gfortran],[main],[])

AC_ARG_WITH(
	[atlas-libdir],
	AC_HELP_STRING([--with-atlas-libdir=DIR],[ATLAS library directory]),
	[with_atlas_libdir="$withval"],
	[with_atlas=""]
)
if test "$with_atlas_libdir"; then
	LDFLAGS="$LDFLAGS -L$with_atlas_libdir"
fi
AC_ARG_WITH(
	[atlas-incdir],
	AC_HELP_STRING([--with-atlas-incdir=DIR],[ATLAS include directory]),
	[with_atlas_incdir="$withval"],
	[with_atlas=""]
)
if test "$with_atlas_incdir"; then
	CPPFLAGS="$CPPFLAGS -L$with_atlas_incdir"
fi

AC_ARG_WITH(
	[blas-libdir],
	AC_HELP_STRING([--with-blas-libdir=DIR],[BLAS library directory]),
	[with_blas_libdir="$withval"],
	[with_blas=""]
)
if test "$with_blas_libdir"; then
	LDFLAGS="$LDFLAGS -L$with_blas_libdir"
fi
AC_ARG_WITH(
	[blas-incdir],
	AC_HELP_STRING([--with-blas-incdir=DIR],[BLAS include directory]),
	[with_blas_incdir="$withval"],
	[with_blas=""]
)
if test "$with_blas_incdir"; then
	CPPFLAGS="$CPPFLAGS -L$with_blas_incdir"
fi

AC_ARG_WITH(
	[lapack-libdir],
	AC_HELP_STRING([--with-lapack-libdir=DIR],[LAPACK library directory]),
	[with_lapack_libdir="$withval"],
	[with_lapack=""]
)
if test "$with_lapack_libdir"; then
	LDFLAGS="$LDFLAGS -L$with_lapack_libdir"
fi
AC_ARG_WITH(
	[lapack-incdir],
	AC_HELP_STRING([--with-lapack-incdir=DIR],[LAPACK include directory]),
	[with_lapack_incdir="$withval"],
	[with_lapack=""]
)
if test "$with_lapack_incdir"; then
	CPPFLAGS="$CPPFLAGS -L$with_lapack_incdir"
fi

AC_CHECK_LIB([blas],[dcopy_],[])
AC_CHECK_LIB([lapack],[dlamch_],[])

AC_ARG_ENABLE(
	[openmp],
	AC_HELP_STRING([--enable-openmp],[enable openMP [[default=yes]]]),
	[],
	[enable_openmp="yes"]
)
if test "x$enable_openmp"="xyes"; then
	case "$CC" in
		gcc )
			CFLAGS="$CFLAGS -fopenmp"
			;;
		pgcc )
			CFLAGS="$CFLAGS -mp"
			;;
		icc )
			CFLAGS="$CFLAGS -openmp"
			;;
	esac
fi

# Checks for header files.
AC_CHECK_HEADER([stdlib.h])
AC_CHECK_HEADER([string.h])
AC_CHECK_HEADER([lapack.h])
AC_CHECK_HEADER([blas.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_CHECK_FUNCS([floor pow sqrt log strrchr])
AC_CHECK_FUNCS([dcopy_])
AC_CHECK_FUNCS([dlamch_])

# pkgconfig
CDESCENT_DESCRIPTION='C implementation of coordinate descent algorithm'
AC_SUBST([CDESCENT_DESCRIPTION])
AC_CONFIG_FILES([cdescent.pc:pkgconfig/cdescent.pc.in])

# make files
AC_CONFIG_FILES([Makefile
                 include/Makefile
                 src/Makefile
                 example/Makefile])
AC_OUTPUT
