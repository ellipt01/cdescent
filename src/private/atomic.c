/*
 * atomic.c
 *
 *  Created on: 2014/07/04
 *      Author: utsugi
 */

#include <stdbool.h>

#define atomic_bool_compare_and_swap(p, f, t) __sync_bool_compare_and_swap((p), (f), (t))

/* union with double and long */
union dlvar {
	double	dv;
	long	lv;
};

/*** atomic add operation ***/
void
atomic_add (double *data, double delta)
{
	union dlvar	old;
	union dlvar	new;
	while (1) {
		old.dv = *data;
		new.dv = old.dv + delta;
		if (atomic_bool_compare_and_swap ((long *) data, *(volatile long *) &old.lv, *(volatile long *) &new.lv))
			break;
	}
	return;
}

/*** atomic max function ***/
void
atomic_max (double *data, double val)
{
	union dlvar	old;
	union dlvar	new;
	if (*data >= val) return;
	while (1) {
		old.dv = *data;
		new.dv = val;
		if (atomic_bool_compare_and_swap ((long *) data, *(volatile long *) &old.lv, *(volatile long *) &new.lv))
			break;
	}
	return;
}
