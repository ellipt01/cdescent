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

/*** atomic add operation ***/
void
atomic_add (double *data, double delta)
{
	volatile long	oldl;
	volatile long	newl;
	union dlvar	var;
	while (1) {
		var.dv = *data;
		oldl = var.lv;
		var.dv += delta;
		newl = var.lv;
		if (atomic_bool_compare_and_swap ((long *) data, *(volatile long *) &oldl, *(volatile long *) &newl)) break;
	}
	return;
}

/*** atomic max function ***/
void
atomic_max (double *data, double val)
{
	volatile long	oldl;
	volatile long	newl;
	union dlvar	var;
	if (*data >= val) return;
	while (1) {
		var.dv = *data;
		oldl = var.lv;
		var.dv = val;
		newl = var.lv;
		if (atomic_bool_compare_and_swap ((long *) data, *(volatile long *) &oldl, *(volatile long *) &newl)) break;
	}
	return;
}
