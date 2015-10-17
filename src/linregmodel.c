/*
 * linregmodel.c
 *
 *  Created on: 2014/05/19
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include <mmreal.h>
#include <linregmodel.h>

#include "private/private.h"

/* calculate sum x(:,j) */
static bool
calc_sum (const mm_real *x, double **sum)
{
	int		j;
	bool	centered;
	double	*_sum = (double *) malloc (x->n * sizeof (double));
#pragma omp parallel for
	for (j = 0; j < x->n; j++) _sum[j] = mm_real_xj_sum (x, j);
	// check whether mean is all 0 (x is already centered)
	centered = true;
	for (j = 0; j < x->n; j++) {
		if (fabs (_sum[j] / (double) x->m) > DBL_EPSILON) {
			centered = false;
			break;
		}
	}
	if (centered) {	// mean is all 0
		*sum = NULL;
		free (_sum);
	} else *sum = _sum;

	return centered;
}

/* calculate sum x(:,j)^2 */
static bool
calc_ssq (const mm_real *x, double **ssq)
{
	int		j;
	bool	normalized;
	double	*_ssq = (double *) malloc (x->n * sizeof (double));
#pragma omp parallel for
	for (j = 0; j < x->n; j++) _ssq[j] = mm_real_xj_ssq (x, j);
	// check whether norm is all 1 (x is already normalized)
	normalized = true;
	for (j = 0; j < x->n; j++) {
		if (fabs (_ssq[j] - 1.) > DBL_EPSILON) {
			normalized = false;
			break;
		}
	}
	if (normalized) {
		*ssq = NULL;	// norm is all 1
		free (_ssq);
	} else *ssq = _ssq;

	return normalized;
}

/* centering each column of matrix:
 * x(:,j) -> x(:,j) - mean(x(:,j)) */
static void
do_centering (mm_dense *x, const double *sum)
{
	int		j;
#pragma omp parallel for
	for (j = 0; j < x->n; j++) {
		double	meanj = sum[j] / (double) x->m;
		if (fabs (meanj) > DBL_EPSILON) mm_real_xj_add_const (x, j, - meanj);
	}
	return;
}

/* normalizing each column of matrix:
 * x(:,j) -> x(:,j) / norm(x(:,j)) */
static void
do_normalizing (mm_real *x, const double *ssq)
{
	int		j;
#pragma omp parallel for
	for (j = 0; j < x->n; j++) {
		double	nrm2j = sqrt (ssq[j]);
		if (fabs (nrm2j - 1.) > DBL_EPSILON) mm_real_xj_scale (x, j, 1. / nrm2j);
	}
	return;
}

/* allocate linregmodel object */
static linregmodel *
linregmodel_alloc (void)
{
	linregmodel	*lreg = (linregmodel *) malloc (sizeof (linregmodel));
	if (lreg == NULL) return NULL;

	lreg->y = NULL;
	lreg->x = NULL;
	lreg->d = NULL;

	lreg->c = NULL;
	lreg->camax = 0.;

	lreg->ycentered = false;
	lreg->xcentered = false;
	lreg->xnormalized = false;

	lreg->sy = NULL;
	lreg->sx = NULL;
	lreg->xtx = NULL;
	lreg->dtd = NULL;

	return lreg;
}

/*** create new linregmodel object
 * INPUT:
 * mm_dense			*y: dense vector
 * mm_real			*x: sparse or dense general / symmetric matrix
 * const mm_real	*d: general linear penalty operator
 * PreProc			proc: specify pre-processings for y and x
 * 						DO_CENTERING_Y: centering of y
 * 						DO_CENTERING_X: centering of each column of x
 * 						DO_NORMALIZING_X: normalizing of each column of x
 * 						DO_STANDARDIZING_X: centering and normalizing of each column of x ***/
linregmodel *
linregmodel_new (mm_real *y, mm_real *x, const mm_real *d, PreProc proc)
{
	int			j;
	linregmodel	*lreg;

	/* check whether x and y are not empty */
	if (!y) error_and_exit ("linregmodel_new", "y is empty.", __FILE__, __LINE__);
	if (!x) error_and_exit ("linregmodel_new", "x is empty.", __FILE__, __LINE__);

	/* check whether y is vector */
	if (y->n != 1) error_and_exit ("linregmodel_new", "y must be vector.", __FILE__, __LINE__);

	/* check dimensions of x and y */
	if (y->m != x->m) error_and_exit ("linregmodel_new", "dimensions of x and y do not match.", __FILE__, __LINE__);

	/* check dimensions of x and d */
	if (d && x->n != d->n) error_and_exit ("linregmodel_new", "dimensions of x and d do not match.", __FILE__, __LINE__);

	lreg = linregmodel_alloc ();
	if (lreg == NULL) error_and_exit ("linregmodel_new", "failed to allocate memory for linregmodel object.", __FILE__, __LINE__);

	lreg->y = y;
	lreg->ycentered = calc_sum (lreg->y, &lreg->sy);
	/* if DO_CENTERING_Y is set and y is not already centered */
	if ((proc & DO_CENTERING_Y) && !lreg->ycentered) {
		/* if lreg->y is sparse, convert to dense vector */
		if (mm_real_is_sparse (lreg->y)) mm_real_sparse_to_dense (lreg->y);
		do_centering (lreg->y, lreg->sy);
		lreg->ycentered = true;
	}

	lreg->x = x;
	lreg->xcentered = calc_sum (lreg->x, &lreg->sx);
	/* if DO_CENTERING_X is set and x is not already centered */
	if ((proc & DO_CENTERING_X) && !lreg->xcentered) {
		/* if lreg->x is sparse, convert to dense matrix */
		if (mm_real_is_sparse (lreg->x)) mm_real_sparse_to_dense (lreg->x);
		/* if lreg->x is symmetric, convert to general matrix */
		if (mm_real_is_symmetric (lreg->x)) mm_real_symmetric_to_general (lreg->x);
		do_centering (lreg->x, lreg->sx);
		lreg->xcentered = true;
	};

	lreg->xnormalized = calc_ssq (lreg->x, &lreg->xtx);
	/* if DO_NORMALIZING_X is set and x is not already normalized */
	if ((proc & DO_NORMALIZING_X) && !lreg->xnormalized) {
		/* if lreg->x is symmetric, convert to general matrix */
		if (mm_real_is_symmetric (lreg->x)) mm_real_symmetric_to_general (lreg->x);
		do_normalizing (lreg->x, lreg->xtx);
		lreg->xnormalized = true;
	}

	/* copy d */
	if (d) {
		lreg->d = d;
		lreg->dtd = (double *) malloc (lreg->d->n * sizeof (double));
#pragma omp parallel for
		for (j = 0; j < lreg->d->n; j++) {
			lreg->dtd[j] = mm_real_xj_ssq (lreg->d, j);
		}
	}

	// c = X' * y
	lreg->c = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, lreg->x->n, 1, lreg->x->n);
#pragma omp parallel for
	for (j = 0; j < lreg->x->n; j++) {
		lreg->c->data[j] = mm_real_xj_trans_dot_yk (lreg->x, j, lreg->y, 0);
	}

	// camax = max ( abs (c) )
	lreg->camax = fabs (lreg->c->data[idamax_ (&lreg->c->nnz, lreg->c->data, &ione) - 1]);

	return lreg;
}

/*** free linregmodel object ***/
void
linregmodel_free (linregmodel *lreg)
{
	if (lreg) {
		if (lreg->sy) free (lreg->sy);
		if (lreg->sx) free (lreg->sx);
		if (lreg->xtx) free (lreg->xtx);
		if (lreg->dtd) free (lreg->dtd);
		free (lreg);
	}
	return;
}
