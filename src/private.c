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

bool	_compare_and_swap (long *ptr, long oldv, long newv);

#ifdef __GNUC__
#	if __GNUC_PREREQ (4, 1)	// gcc_version >= 4.1
#	else
#define __sync_bool_compare_and_swap(p, f, t) _compare_and_swap((p), (f), (t))
#	endif
#else
#  ifdef __ICC
#define __sync_bool_compare_and_swap(p, f, t) stm::iccsync::bool_compare_and_swap((p), (f), (t))
#  endif
#endif

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

union dlvar {
	double	dv;
	long	lv;
};

union dlptr {
	double	*dp;
	long	*lp;
};

bool
_compare_and_swap (long *ptr, long oldv, long newv)
{
	unsigned char	ret;
	__asm__ __volatile__ (
			"  lock\n"
			"  cmpxchgq %2,%1\n"
			"  sete %0\n"
			:  "=q" (ret), "=m" (*ptr)
			:  "r" (newv), "m" (*ptr), "a" (oldv)
			:  "memory");
	return ret;
}

void
cdescent_cas_add (double *data, double delta)
{
	union dlvar	oldval;
	union dlvar	newval;
	union dlptr	ptr;
	ptr.dp = data;
	while (1) {
		oldval.dv = *(volatile double *) ptr.dp;
		newval.dv = oldval.dv + delta;
		if (__sync_bool_compare_and_swap (ptr.lp, *(volatile long *) &oldval.lv, *(volatile long *) &newval.lv))
			break;
	}
	return;
}
