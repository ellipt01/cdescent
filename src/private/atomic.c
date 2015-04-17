/*
 * atomic.c
 *
 *  Created on: 2014/07/04
 *      Author: utsugi
 */

#include <stdbool.h>

#define atomic_bool_compare_and_swap(p, f, t) __sync_bool_compare_and_swap((p), (f), (t))

/*** atomic add operation ***/
void
atomic_add (double *data, double delta)
{
	union {
		double	dv;
		long	lv;
	} old;

	union {
		double	dv;
		long	lv;
	} new;

	while (1) {
		old.lv = *(volatile long *) data;
		new.dv = old.dv + delta;
		if (atomic_bool_compare_and_swap ((volatile long *) data, old.lv, new.lv)) break;
	}
	return;
}

/*** atomic max function ***/
void
atomic_max (double *data, double val)
{
	if (*data >= val) return;

	union {
		double	dv;
		long	lv;
	} old;

	union {
		double	dv;
		long	lv;
	} new;

	while (1) {
		old.lv = *(volatile long *) data;
		new.dv = val;
		if (atomic_bool_compare_and_swap ((volatile long *) data, old.lv, new.lv)) break;
	}
	return;
}
