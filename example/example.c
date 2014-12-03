/*
 * example.c
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#include <stdbool.h>
#include <math.h>

#include <cdescent.h>

/*** 1D derivation operator for the L2 penalty of s-lasso ***/

static mm_sparse *
penalty_ssmooth (const int n)
{
	int		i, j, k;
	mm_sparse	*s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n - 1, n, 2 * (n - 1));

	k = 0;
	for (j = 0; j < n; j++) {
		if (j > 0) {
			s->i[k] = j - 1;
			s->data[k++] = -1.;
		}
		if (j < n - 1) {
			s->i[k] = j;
			s->data[k++] = 1.;
		}
		s->p[j + 1] = k;
	}
	return s;
}

static mm_dense *
penalty_dsmooth (const int n)
{
	int		j;
	mm_dense	*d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n - 1, n, (n - 1) * n);
	mm_real_set_all (d, 0.);
	for (j = 0; j < n; j++) {
		if (j > 0) d->data[j - 1 + j * d->m] = -1.;
		if (j < n - 1) d->data[j + j * d->m] = 1.;
	}
	return d;
}

/*** sparse/dense 1D derivation operator ***/
mm_real *
penalty_smooth (MMRealFormat format, const int n)
{
	return (format == MM_REAL_SPARSE) ? penalty_ssmooth (n) : penalty_dsmooth (n);
}

