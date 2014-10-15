/*
 * linregmodel.c
 *
 *  Created on: 2014/05/19
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <linregmodel.h>

#include "private.h"

/* centering each column of matrix:
 * x(:, j) -> x(:, j) - mean(x(:, j)) */
static void
do_centering (mm_dense *x)
{
	int		j;
	for (j = 0; j < x->n; j++) {
		int		i;
		double	meanj = 0.;
		for (i = 0; i < x->m; i++) meanj += x->data[i + j * x->m];
		meanj /= (double) x->m;
		for (i = 0; i < x->m; i++) x->data[i + j * x->m] -= meanj;
	}
	return;
}

/* normalizing each column of matrix:
 * x(:, j) -> x(:, j) / norm(x(:, j)) */
static void
do_normalizing (mm_real *x)
{
	int		j;
	for (j = 0; j < x->n; j++) {
		double	alpha;
		double	nrmj;
		int		size = (mm_real_is_sparse (x)) ? x->p[j + 1] - x->p[j] : x->m;
		double	*xj = x->data + ((mm_real_is_sparse (x)) ? x->p[j] : j * x->m);
		nrmj = dnrm2_ (&size, xj, &ione);
		alpha = 1. / nrmj;
		dscal_ (&size, &alpha, xj, &ione);
	}
	return;
}

/* allocate linregmodel object */
static linregmodel *
linregmodel_alloc (void)
{
	linregmodel	*lreg = (linregmodel *) malloc (sizeof (linregmodel));

	lreg->has_copy_y = false;
	lreg->has_copy_x = false;

	lreg->y = NULL;
	lreg->x = NULL;
	lreg->d = NULL;
	lreg->lambda2 = 0.;
	lreg->is_regtype_lasso = true;

	lreg->c = NULL;
	lreg->logcamax = 0.;

	lreg->ycentered = false;
	lreg->xcentered = false;
	lreg->xnormalized = false;

	lreg->sy = 0.;
	lreg->sx = NULL;
	lreg->xtx = NULL;
	lreg->dtd = NULL;

	return lreg;
}

/*** create new linregmodel object
 * INPUT:
 * mm_dense          *y: dense vector
 * bool      has_copy_y: whether linregmodel object has own copy of y
 * mm_sparse         *x: sparse general / symmetric matrix
 * bool      has_copy_x: whether linregmodel object has own copy of x
 * const double lambda2: regularization parameter
 * const mm_real     *d: general linear operator of penalty
 * PreProc         proc: specify pre-processing for y and x
 *                       DO_CENTERING_Y: do centering of y
 *                       DO_CENTERING_X: do centering of each column of x
 *                       DO_NORMALIZING_X: do normalizing of each column of x
 *                       DO_STANDARDIZING_X: do centering and normalizing of each column of x ***/
linregmodel *
linregmodel_new (mm_dense *y, bool has_copy_y, mm_real *x, bool has_copy_x, const double lambda2, const mm_real *d, PreProc proc)
{
	double			camax;
	linregmodel	*lreg;

	/* check whether x and y are not empty */
	if (!y) error_and_exit ("linregmodel_new", "y is empty.", __FILE__, __LINE__);
	if (!x) error_and_exit ("linregmodel_new", "x is empty.", __FILE__, __LINE__);

	/* check whether y is dense unsymmetric vector */
	if (!mm_real_is_dense (y)) error_and_exit ("linregmodel_new", "y must be dense.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (y)) error_and_exit ("linregmodel_new", "y must be general.", __FILE__, __LINE__);
	if (y->n != 1) error_and_exit ("linregmodel_new", "y must be vector.", __FILE__, __LINE__);

	/* if x or d is symmetric, check whether it is square */
	if (mm_real_is_symmetric (x) && x->m != x->n) error_and_exit ("linregmodel_new", "symmetric matrix must be square.", __FILE__, __LINE__);
	if (d && mm_real_is_symmetric (d) && d->m != d->n) error_and_exit ("linregmodel_new", "symmetric matrix must be square.", __FILE__, __LINE__);

	/* check dimensions */
	if (y->m != x->m) error_and_exit ("linregmodel_new", "dimensions of matrix x and vector y do not match.", __FILE__, __LINE__);
	if (d && x->n != d->n) error_and_exit ("linregmodel_new", "dimensions of matrix x and d do not match.", __FILE__, __LINE__);

	lreg = linregmodel_alloc ();

	/* has copy of y
	 * if DO_CENTERING_Y is set to proc, lreg->has_copy_y is set to true */
	if (proc & DO_CENTERING_Y) lreg->has_copy_y = true;
	else lreg->has_copy_y = has_copy_y;

	/* has copy of x
	 * if DO_CENTERING_X flag is set to proc, or DO_NORMALIZING_X flag is set to proc and x is symmetric,
	 * lreg->has_copy_x is set to true */
	if (proc & DO_CENTERING_X) lreg->has_copy_x = true;
	else if ((proc & DO_NORMALIZING_X) && mm_real_is_symmetric (x)) lreg->has_copy_x = true;
	else lreg->has_copy_x = has_copy_x;

	if (lreg->has_copy_y) lreg->y = mm_real_copy (y);
	else lreg->y = y;

	if (lreg->has_copy_x) lreg->x = mm_real_copy (x);
	else lreg->x = x;

	if (d) lreg->d = mm_real_copy (d);

	/* centering y */
	if (proc & DO_CENTERING_Y) {	// lreg->has_copy_y = true
		do_centering (lreg->y);
		lreg->ycentered = true;
	}
	/* centering x */
	if (proc & DO_CENTERING_X) {	// lreg->has_copy_x = true
		/* if lreg->x is sparse, convert to dense matrix */
		if (mm_real_is_sparse (lreg->x)) {
			mm_real	*tmp = lreg->x;
			lreg->x = mm_real_sparse_to_dense (tmp);
			mm_real_free (tmp);
		}
		/* if lreg->x is symmetric, convert to general matrix */
		if (mm_real_is_symmetric (lreg->x)) {
			mm_real	*tmp = lreg->x;
			lreg->x = mm_real_symmetric_to_general (tmp);
			mm_real_free (tmp);
		}
		do_centering (lreg->x);
		lreg->xcentered = true;
	}

	/* normalizing x */
	if (proc & DO_NORMALIZING_X) {
		/* if lreg->x is symmetric, convert to general matrix */
		if (mm_real_is_symmetric (lreg->x)) {	// lreg->has_copy_x = true
			mm_real	*tmp = lreg->x;
			lreg->x = mm_real_symmetric_to_general (tmp);
			mm_real_free (tmp);
		}
		do_normalizing (lreg->x);
		lreg->xnormalized = true;
	}

	/* lambda2 */
	if (lambda2 > DBL_EPSILON) lreg->lambda2 = lambda2;

	/* if lambda2 > 0 && d != NULL, regression type is NOT lasso: is_regtype_lasso = false */
	if (lreg->lambda2 > DBL_EPSILON && lreg->d) lreg->is_regtype_lasso = false;

	// c = X' * y
	lreg->c = mm_real_x_dot_y (true, 1., lreg->x, lreg->y, 0.);

	// camax = max ( abs (c) )
	camax = fabs (lreg->c->data[idamax_ (&lreg->c->nz, lreg->c->data, &ione) - 1]);
	lreg->logcamax = floor (log10 (camax)) + 1.;

	/* sum y */
	if (!lreg->ycentered) lreg->sy = mm_real_xj_sum (lreg->y, 0);

	/* sx(j) = sum X(:,j) */
	if (!lreg->xcentered) {
		int		j;
		lreg->sx = (double *) malloc (lreg->x->n * sizeof (double));
		for (j = 0; j < lreg->x->n; j++) lreg->sx[j] = mm_real_xj_sum (lreg->x, j);
	}

	/* xtx = diag (X' * X) */
	if (!lreg->xnormalized) {
		int		j;
		lreg->xtx = (double *) malloc (lreg->x->n * sizeof (double));
		for (j = 0; j < lreg->x->n; j++) lreg->xtx[j] = pow (mm_real_xj_nrm2 (lreg->x, j), 2.);
	}

	/* dtd = diag (D' * D) */
	if (!lreg->is_regtype_lasso) {
		int		j;
		lreg->dtd = (double *) malloc (lreg->d->n * sizeof (double));
		for (j = 0; j < lreg->d->n; j++) lreg->dtd[j] = pow (mm_real_xj_nrm2 (lreg->d, j), 2.);
	}

	return lreg;
}

/*** free linregmodel object ***/
void
linregmodel_free (linregmodel *lreg)
{
	if (lreg) {
		if (lreg->y && lreg->has_copy_y) mm_real_free (lreg->y);
		if (lreg->x && lreg->has_copy_x) mm_real_free (lreg->x);
		if (lreg->d) mm_real_free (lreg->d);
		if (lreg->sx) free (lreg->sx);
		if (lreg->xtx) free (lreg->xtx);
		if (lreg->dtd) free (lreg->dtd);
		free (lreg);
	}
	return;
}
