## Process this file with automake to produce Makefile.in

# sample program
noinst_PROGRAMS = elasticnet adaelasticnet elasticnet_rwt

noinst_LTLIBRARIES = libexample.la 

noinst_HEADERS = example.h

libexample_la_SOURCES = example.c

libexample_la_CPPFLAGS =\
	-I$(top_srcdir)/include\
	-I$(top_srcdir)/example

libexample_la_LDFLAGS = -L$(top_srcdir)/src

libexample_la_LIBADD = -lcdescent

# sample program
elasticnet_SOURCES = elasticnet.c
elasticnet_CPPFLAGS =\
	-I$(top_srcdir)/include\
	-I$(top_srcdir)/example

elasticnet_LDADD =\
	$(top_srcdir)/example/libexample.la

# sample program
adaelasticnet_SOURCES = adaelasticnet.c
adaelasticnet_CPPFLAGS =\
	-I$(top_srcdir)/include\
	-I$(top_srcdir)/example

adaelasticnet_LDADD =\
	$(top_srcdir)/example/libexample.la

# sample program
elasticnet_rwt_SOURCES = elasticnet_reweight.c
elasticnet_rwt_CPPFLAGS =\
	-I$(top_srcdir)/include\
	-I$(top_srcdir)/example

elasticnet_rwt_LDADD =\
	$(top_srcdir)/example/libexample.la
