## Process this file with automake to produce Makefile.in

AM_CFLAGS =\
	 -Wall\
	 -g

lib_LTLIBRARIES = libcdescent.la

libcdescent_la_SOURCES =\
	private/private.c\
	private/atomic.c\
	mmio.c\
	mmreal.c\
	linregmodel.c\
	cdescent.c\
	stepsize.c\
	bic.c\
	regression.c

libcdescent_la_LIBADD = -L$(top_srcdir)/src
libcdescent_la_CPPFLAGS = -I$(top_srcdir)/include
libcdescent_la_LDFLAGS = -version-info $(CDESCENT_LT_VERSION)


