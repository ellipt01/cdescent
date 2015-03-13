/*
 * atomic.c
 *
 *  Created on: 2014/07/04
 *      Author: utsugi
 */

#include <stdbool.h>

#define atomic_bool_compare_and_swap(p, f, t) __sync_bool_compare_and_swap((p), (f), (t))

/* union with double and long */
union {
	double	dv;
	long	lv;
} dlvar;

/*** atomic add operation ***/
void
atomic_add (double *data, double delta)
{
	long	old_lv, new_lv;
	while (1) {
		dlvar.dv = *data;
		old_lv = dlvar.lv;
		dlvar.dv += delta;
		new_lv = dlvar.lv;
		if (atomic_bool_compare_and_swap ((volatile long *) data, old_lv, new_lv))
			break;
	}
	return;
}

/*** atomic max function ***/
void
atomic_max (double *data, double val)
{
	long	old_lv, new_lv;
	if (*data >= val) return;
	while (1) {
		dlvar.dv = *data;
		old_lv = dlvar.lv;
		dlvar.dv = val;
		new_lv = dlvar.lv;
		if (atomic_bool_compare_and_swap ((volatile long *) data, old_lv, new_lv))
			break;
	}
	return;
}
