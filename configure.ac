#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.59])
AC_INIT([cdescent], [0.4.4])
AC_CONFIG_SRCDIR([example/example.h])
AC_CONFIG_HEADERS([config.h])

CDESCENT_CURRENT=6
CDESCENT_REVISION=1
CDESCENT_AGE=5
CDESCENT_LT_VERSION="${CDESCENT_CURRENT}:${CDESCENT_REVISION}:${CDESCENT_AGE}"
AC_SUBST(CDESCENT_LT_VERSION)

AM_INIT_AUTOMAKE([foreign subdir-objects -Wall])

AM_PROG_AR([ACT-IF-FAIL])
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

# checks gcc version
AC_DEFUN([AX_GCC_VERSION], [
	GCC_VERSION=""
	AS_IF([test "x$GCC" = "xyes"],[
		AC_CACHE_CHECK([gcc version],[ax_cv_gcc_version],[
			ax_cv_gcc_version="`$CC -dumpversion`"
			AS_IF([test "x$ax_cv_gcc_version" = "x"],[ax_cv_gcc_version=""])
		])
		GCC_VERSION=$ax_cv_gcc_version
	])
	AC_SUBST([GCC_VERSION])
])

# if CC=gcc, checks gcc version (>= 4.1)
if test "x$GCC" == xyes; then
	AX_GCC_VERSION
	AC_CACHE_CHECK([gcc major version],[ax_cv_gcc_version_major],[
		ax_cv_gcc_version_major="`echo $GCC_VERSION | cut -d'.' -f1`"
		AS_IF([test "x$ax_cv_gcc_version_major" = "x"],[ax_cv_gcc_version_major=""])
	])
	GCC_VERSION_MAJOR=$ax_cv_gcc_version_major
	AC_CACHE_CHECK([gcc minor version],[ax_cv_gcc_version_minor],[
		ax_cv_gcc_version_minor="`echo $GCC_VERSION | cut -d'.' -f2`"
		AS_IF([test "x$ax_cv_gcc_version_minor" = "x"],[ax_cv_gcc_version_minor=""])
	])
	GCC_VERSION_MINOR=$ax_cv_gcc_version_minor

	AS_IF([test "$GCC_VERSION_MAJOR" -lt 4],[
		AC_MSG_FAILURE([gcc version requires >= 4.1])
	])

	AS_IF([test "$GCC_VERSION_MAJOR" -eq 4],[
		AS_IF([test "$GCC_VERSION_MINOR" -lt 1],[
			AC_MSG_FAILURE([gcc version requires >= 4.1])
		])
	])
fi

# Checks for libraries.
AC_CHECK_LIB([m],[floor],[])
AC_CHECK_LIB([gfortran],[main],[])

AC_ARG_WITH(
	[atlas],
	[AC_HELP_STRING([--with-atlas[[=DIR]]],
	[ATLAS library directory])],
	[atlas_libdir="$withval"],
	[atlas_libdir=""])
if test "$atlas_libdir"; then
	LDFLAGS="$LDFLAGS -L$atlas_libdir"
	use_atlas=yes
fi

AC_ARG_WITH(
	[tatlas],
	[AC_HELP_STRING([--with-tatlas[[=DIR]]],
	[parallel ATLAS library directory])],
	[tatlas_libdir="$withval"],
	[tatlas_libdir=""])
if test "$tatlas_libdir"; then
	LDFLAGS="$LDFLAGS -L$tatlas_libdir"
	use_tatlas=yes
fi

AC_ARG_WITH(
	[satlas],
	[AC_HELP_STRING([--with-satlas[[=DIR]]],
	[serial ATLAS library directory])],
	[satlas_libdir="$withval"],
	[satlas_libdir=""])
if test "$satlas_libdir"; then
	LDFLAGS="$LDFLAGS -L$satlas_libdir"
	use_satlas=yes
fi

AC_ARG_WITH(
	[atlas-incdir],
	[AC_HELP_STRING([--with-atlas-incdir[[=DIR]]],
	[ATLAS include directory])],
	[atlas_incdir="$withval"],
	[atlas_incdir=""])
if test "$atlas_incdir"; then
	CPPFLAGS="$CPPFLAGS -I$atlas_incdir"
fi

AC_ARG_WITH(
	[mkl],
	[AC_HELP_STRING([--with-mkl[[=DIR]]],
	[Intel math kernel library directory])],
	[mkl_libdir="$withval"],
	[mkl_libdir=""])
if test "$mkl_libdir"; then
	LDFLAGS="$LDFLAGS -L$mkl_libdir"
	use_mkl=yes
fi

AC_ARG_WITH(
	[mkl-incdir],
	[AC_HELP_STRING([--with-mkl-incdir[[=DIR]]],
	[Intel math kernel library include directory])],
	[mkl_incdir="$withval"],
	[mkl_incdir=""])
if test "$mkl_incdir"; then
	CPPFLAGS="$CPPFLAGS -I$mkl_incdir"
fi

AC_ARG_WITH(
	[blas],
	[AC_HELP_STRING([--with-blas[[=DIR]]],
	[BLAS library directory])],
	[blas_libdir="$withval"],
	[blas_libdir=""])
if test "$blas_libdir"; then
	LDFLAGS="$LDFLAGS -L$blas_libdir"
fi

AC_ARG_WITH(
	[blas-incdir],
	[AC_HELP_STRING([--with-blas-incdir[[=DIR]]],
	[BLAS include directory])],
	[blas_incdir="$withval"],
	[blas_incdir=""])
if test "$blas_incdir"; then
	CPPFLAGS="$CPPFLAGS -I$blas_incdir"
fi

if test "x$use_atlas" == xyes; then
	AC_CHECK_LIB([f77blas],[main],[])
elif test "x$use_tatlas" == xyes; then
	AC_CHECK_LIB([tatlas],[main],[])
elif test "x$use_satlas" == xyes; then
	AC_CHECK_LIB([satlas],[main],[])
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
	[enable openMP support [default=yes]])],
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

AC_SUBST(AM_LDFLAGS, "$LDFLAGS")
AC_SUBST(AM_CPPFLAGS, "$CPPFLAGS")

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

