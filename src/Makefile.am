## Process this file with automake to produce Makefile.in

INCLUDES = -I$(top_srcdir)/include

AM_CFLAGS =\
	 -Wall\
	 -g

lib_LTLIBRARIES = libcdescent.la

libcdescent_la_SOURCES =\
	private.c\
	cas.c\
	mmio.c\
	mmreal.c\
	linregmodel.c\
	cdescent.c\
	stepsize.c\
	update.c

libcdescent_la_CPPFLAGS = -fopenmp -I$(top_srcdir)/include

libcdescent_la_LDFLAGS = -L$(top_srcdir)/src

libcdescent_la_LIBADD = -lgomp

