/*
 * example.c
 *
 *  Created on: 2014/03/17
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "example.h"

/* penalty term for s-lasso */

static mm_sparse *
mm_real_penalty_ssmooth (const int n)
{
	int		i, j, k;
	mm_sparse	*s = mm_real_new (MM_REAL_SPARSE, false, n - 1, n, 2 * (n - 1));
	s->i = (int *) malloc (s->nz * sizeof (int));
	s->p = (int *) malloc ((s->n + 1) * sizeof (int));
	s->data = (double *) malloc (s->nz * sizeof (double));

	k = 0;
	s->p[0] = 0;
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
mm_real_penalty_dsmooth (const int n)
{
	int		j;
	mm_dense	*d = mm_real_new (MM_REAL_DENSE, false, n - 1, n, (n - 1) * n);
	d->data = (double *) malloc (d->nz * sizeof (double));
	mm_real_set_all (d, 0.);
	for (j = 0; j < n; j++) {
		if (j > 0) d->data[j - 1 + j * d->m] = -1.;
		if (j < n - 1) d->data[j + j * d->m] = 1.;
	}
	return d;
}

/* s-lasso */
mm_real *
mm_real_penalty_smooth (MMRealFormat format, const int n)
{
	return (format == MM_REAL_SPARSE) ? mm_real_penalty_ssmooth (n) : mm_real_penalty_dsmooth (n);
}

