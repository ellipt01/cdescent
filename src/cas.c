/*
 * cas.c
 * compare and swap
 *
 *  Created on: 2014/07/01
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdbool.h>

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

static bool
compare_and_swap (long *ptr, long oldv, long newv)
{
	bool	status = false;
#ifdef __GNUC__
#	if __GNUC_PREREQ (4, 2)	// gcc_version >= 4.2
	status = __sync_bool_compare_and_swap (ptr, oldv, newv);
#	endif
#else
	status = _compare_and_swap (ptr, oldv, newv);
#endif
	return status;
}

void
cas_add (double *data, int idx, double delta)
{
	union dlvar	oldval;
	union dlvar	newval;
	union dlptr	ptr;
	ptr.dp = data;
	while (1) {
		oldval.dv = *(volatile double *) &ptr.dp[idx];
		newval.dv = oldval.dv + delta;
		if (compare_and_swap (ptr.lp + idx, *(volatile long *) &oldval.lv, *(volatile long *) &newval.lv))
			break;
	}
	return;
}
