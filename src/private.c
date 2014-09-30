/*
 * private.c
 *
 *  Created on: 2014/06/25
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "private.h"

const int		ione  =  1;
const double	dzero =  0.;
const double	done  =  1.;
const double	dmone = -1.;

double	_double_eps_ = -1.;

/* print an error message and exit */
void
error_and_exit (const char * function_name, const char *error_msg, const char *file, const int line)
{
	fprintf (stderr, "ERROR: %s: %s:%d: %s\n", function_name, file, line, error_msg);
	exit (1);
}

/* print warning message */
void
printf_warning (const char * function_name, const char *error_msg, const char *file, const int line)
{
	fprintf (stderr, "WARNING: %s: %s:%d: %s\n", function_name, file, line, error_msg);
	return;
}
