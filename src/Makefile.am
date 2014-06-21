## Process this file with automake to produce Makefile.in

INCLUDES = -I$(top_srcdir)/include

AM_CFLAGS =\
	 -Wall\
	 -g

lib_LTLIBRARIES = libcdescent.la

libcdescent_la_SOURCES =\
	mmio.c\
	mm_mtx.c\
	linreg.c\
	utils.c\
	update.c\
	cdescent.c

libcdescent_la_CPPFLAGS = -I$(top_srcdir)/include

libcdescent_la_LDFLAGS = -L$(top_srcdir)/src

libcdescent_la_LIBADD =

