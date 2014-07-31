#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.59])
AC_INIT([cdescent], [0.1.2])
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
	[AC_HELP_STRING([--with-atlas-libdir=DIR],
	[ATLAS library directory])],
	[with_atlas_libdir="$withval"],
	[with_atlas_libdir=""])
if test "$with_atlas_libdir"; then
	LDFLAGS="$LDFLAGS -L$with_atlas_libdir"
	use_atlas=yes
fi

AC_ARG_WITH(
	[atlas-incdir],
	[AC_HELP_STRING([--with-atlas-incdir=DIR],
	[ATLAS include directory])],
	[with_atlas_incdir="$withval"],
	[with_atlas_incdir=""])
if test "$with_atlas_incdir"; then
	CPPFLAGS="$CPPFLAGS -L$with_atlas_incdir"
	use_atlas=yes
fi

AC_ARG_WITH(
	[mkl-libdir],
	[AC_HELP_STRING([--with-mkl-libdir=DIR],
	[Intel math kernel library directory])],
	[with_mkl_libdir="$withval"],
	[with_mkl_libdir=""])
if test "$with_mkl_libdir"; then
	LDFLAGS="$LDFLAGS -L$with_mkl_libdir"
	use_mkl=yes
fi

AC_ARG_WITH(
	[mkl-incdir],
	[AC_HELP_STRING([--with-mkl-incdir=DIR],
	[Intel math kernel library include directory])],
	[with_mkl_incdir="$withval"],
	[with_mkl_incdir=""])
if test "$with_mkl_incdir"; then
	CPPFLAGS="$CPPFLAGS -L$with_mkl_incdir"
	use_mkl=yes
fi

AC_ARG_WITH(
	[blas-libdir],
	[AC_HELP_STRING([--with-blas-libdir=DIR],
	[BLAS library directory])],
	[with_blas_libdir="$withval"],
	[with_blas_libdir=""])
if test "$with_blas_libdir"; then
	LDFLAGS="$LDFLAGS -L$with_blas_libdir"
fi

AC_ARG_WITH(
	[blas-incdir],
	[AC_HELP_STRING([--with-blas-incdir=DIR],
	[BLAS include directory])],
	[with_blas_incdir="$withval"],
	[with_blas_incdir=""])
if test "$with_blas_incdir"; then
	CPPFLAGS="$CPPFLAGS -L$with_blas_incdir"
fi

if test "x$use_atlas" == xyes; then
	AC_CHECK_LIB([f77blas],[main],[])
elif test "x$use_mkl" == xyes; then
	AC_CHECK_LIB([mkl_intel_lp64],
	[main],
	[LIBS="-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm $LIBS"],
	[AC_MSG_ERROR([Cannot find Intel Math Kernel Library])],
	[-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm])
else
	AC_CHECK_LIB([blas],[main],[])
fi

AC_ARG_ENABLE(
	[openmp],
	[AC_HELP_STRING([--enable-openmp],
	[enable openMP support [[default=yes]]])],
	[],
	[enable_openmp=yes]
)

if test "x$enable_openmp" != xno; then
	case "$CC" in
		gcc )
			OMPFLAGS="-fopenmp"
			;;
		pgcc )
			OMPFLAGS="-mp"
			;;
		icc )
			OMPFLAGS="-openmp"
			;;
		* )
			OMPFLAGS=""
			;;
	esac
	CFLAGS="$CFLAGS $OMPFLAGS"
fi
AC_SUBST([OMPFLAGS])

# Checks for header files.
AC_CHECK_HEADER([stdlib.h])
AC_CHECK_HEADER([string.h])
AC_CHECK_HEADER([blas.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_CHECK_FUNCS([floor pow sqrt log strrchr])
AC_CHECK_FUNCS([dcopy_])

# pkgconfig
CDESCENT_DESCRIPTION="C implementation of coordinate descent algorithm"
AC_SUBST([CDESCENT_DESCRIPTION])
AC_CONFIG_FILES([cdescent.pc:pkgconfig/cdescent.pc.in])

# make files
AC_CONFIG_FILES([Makefile
                 include/Makefile
                 src/Makefile
                 example/Makefile])
AC_OUTPUT

