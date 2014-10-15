/*
 * atomic.c
 *
 *  Created on: 2014/07/04
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdbool.h>

#define atomic_bool_compare_and_swap(p, f, t) __sync_bool_compare_and_swap((p), (f), (t))

/* union with double and long */
union dlvar {
	double	dv;
	long	lv;
};

/* union with double and long pointer */
union dlptr {
	double	*dp;
	long	*lp;
};

/*** atomic add operation ***/
void
atomic_add (double *data, double delta)
{
	union dlvar	oldval;
	union dlvar	newval;
	union dlptr	ptr;
	ptr.dp = data;
	while (1) {
		oldval.dv = *data;
		newval.dv = oldval.dv + delta;
		if (atomic_bool_compare_and_swap (ptr.lp, *(volatile long *) &oldval.lv, *(volatile long *) &newval.lv))
			break;
	}
	return;
}

/*** atomic max function ***/
void
atomic_max (double *data, double val)
{
	union dlvar	oldval;
	union dlvar	newval;
	union dlptr	ptr;
	ptr.dp = data;
	while (1) {
		if (*data >= val) break;
		oldval.dv = *data;
		newval.dv = val;
		if (atomic_bool_compare_and_swap (ptr.lp, *(volatile long *) &oldval.lv, *(volatile long *) &newval.lv))
			break;
	}
	return;
}
