## Process this file with automake to produce Makefile.in

# sample program
noinst_PROGRAMS = l1reg_example

noinst_LTLIBRARIES = libexample.la 

noinst_HEADERS = example.h

libexample_la_SOURCES =\
	example.c\
	example_cdescent.c

libexample_la_CPPFLAGS =\
	-I$(top_srcdir)/include\
	-I$(top_srcdir)/example

libexample_la_LDFLAGS = -L$(top_srcdir)/src

libexample_la_LIBADD = -lcdescent

# sample program
l1reg_example_SOURCES = main.c
l1reg_example_CPPFLAGS =\
	-I$(top_srcdir)/include\
	-I$(top_srcdir)/example

l1reg_example_LDADD =\
	$(top_srcdir)/example/libexample.la
