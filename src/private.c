/*
 * private.c
 *
 *  Created on: 2014/06/25
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>

#include "private.h"

const int		ione  =  1;
const double	dzero =  0.;
const double	done  =  1.;
const double	dmone = -1.;

double	_cdescent_double_eps_ = -1.;

/* print an error message and exit */
void
cdescent_error (const char * function_name, const char *error_msg, const char *file, const int line)
{
	fprintf (stderr, "ERROR: %s: %s:%d: %s\n", function_name, file, line, error_msg);
	exit (1);
}

/* print warning message */
void
cdescent_warning (const char * function_name, const char *error_msg, const char *file, const int line)
{
	fprintf (stderr, "WARNING: %s: %s:%d: %s\n", function_name, file, line, error_msg);
	return;
}

/* double_eps of machine precision */
double
cdescent_double_eps (void)
{
	if (_cdescent_double_eps_ < 0.) _cdescent_double_eps_ = dlamch_ ("e");
	return _cdescent_double_eps_;
}
