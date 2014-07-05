/*
 * atomic.c
 *
 *  Created on: 2014/07/04
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdbool.h>

bool	bool_compare_and_swap (long *ptr, long oldv, long newv);

#ifdef __GNUC__
#	if __GNUC_PREREQ (4, 1)	// gcc_version >= 4.1
#define atomic_bool_compare_and_swap(p, f, t) __sync_bool_compare_and_swap((p), (f), (t))
#	else
#define atomic_bool_compare_and_swap(p, f, t) bool_compare_and_swap((p), (f), (t))
#	endif
#else
#  ifdef __ICC
#define atomic_bool_compare_and_swap(p, f, t) stm::iccsync::bool_compare_and_swap((p), (f), (t))
#  endif
#endif

union dlvar {
	double	dv;
	long	lv;
};

union dlptr {
	double	*dp;
	long	*lp;
};

bool
bool_compare_and_swap (long *ptr, long oldv, long newv)
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

void
atomic_max (double *data, double val)
{
	union dlvar	oldval;
	union dlvar	newval;
	union dlptr	ptr;
	ptr.dp = data;
	while (1) {
		oldval.dv = *data;
		if (oldval.dv >= val) break;
		if (atomic_bool_compare_and_swap (ptr.lp, *(volatile long *) &oldval.lv, *(volatile long *) &newval.lv))
			break;
	}
	return;
}
