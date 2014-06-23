/*
 * linreg.c
 *
 *  Created on: 2014/05/19
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <linreg.h>

#include "linreg_private.h"

const int		ione  =  1;
const double	dzero =  0.;
const double	done  =  1.;
const double	dmone = -1.;

double	_linreg_double_eps_ = -1.;

double
linreg_double_eps (void)
{
	if (_linreg_double_eps_ < 0.) _linreg_double_eps_ = dlamch_ ("e");
	return _linreg_double_eps_;
}

/* print an error message and exit */
void
linreg_error (const char * function_name, const char *error_msg, const char *file, const int line)
{
	fprintf (stderr, "ERROR: %s: %s:%d: %s\n", function_name, file, line, error_msg);
	exit (1);
}

linreg *
linreg_alloc (mm_mtx *y, mm_mtx *x, const double lambda2, const mm_mtx *d)
{
	linreg		*lreg;

	if (!y) linreg_error ("linreg_alloc", "vector *y is empty.", __FILE__, __LINE__);
	if (!x) linreg_error ("linreg_alloc", "matrix *x is empty.", __FILE__, __LINE__);
	if (!mm_is_dense (y->typecode)) linreg_error ("linreg_alloc", "vector *y must be dense.", __FILE__, __LINE__);
	if (y->m != x->m) linreg_error ("linreg_alloc", "size of matrix *x and vector *y are incompatible.", __FILE__, __LINE__);
	if (d && x->n != d->n) linreg_error ("linreg_alloc", "size of matrix *x and *d incompatible.", __FILE__, __LINE__);

	lreg = (linreg *) malloc (sizeof (linreg));

	lreg->y = y;
	lreg->x = x;

	lreg->lambda2 = (lambda2 > linreg_double_eps ()) ? lambda2 : 0.;
	lreg->d = (d) ? d : NULL;

	/* By default, data is assumed to be not centered or standardized */
	lreg->meany = NULL;
	lreg->meanx = NULL;
	lreg->normx = NULL;
	lreg->ycentered = false;
	lreg->xcentered = false;
	lreg->xnormalized = false;

	return lreg;
}

void
linreg_free (linreg *lreg)
{
	if (lreg) {
		if (lreg->meany) free (lreg->meany);
		if (lreg->meanx) free (lreg->meanx);
		if (lreg->normx) free (lreg->normx);
		free (lreg);
	}
	return;
}

/* centering each column of matrix:
 * x(:, j) -> x(:, j) - mean(x(:, j)) */
static double *
centering (mm_mtx *x)
{
	int		i, j;
	double	*mean = (double *) malloc (x->n * sizeof (double));
	for (j = 0; j < x->n; j++) {
		double	meanj = mm_mtx_real_xj_sum (j, x) / (double) x->m;
		if (mm_is_sparse (x->typecode)) {
			for (i = x->p[j]; i < x->p[j + 1]; i++) x->data[i] -= meanj;
		} else {
			for (i = 0; i < x->m; i++) x->data[i + j * x->m] -= meanj;
		}
		mean[j] = meanj;
	}
	return mean;
}

/* normalizing each column of matrix:
 * x(:, j) -> x(:, j) / norm(x(:, j)) */
static double *
normalizing (mm_mtx *x)
{
	int		j;
	double	*nrm = (double *) malloc (x->n * sizeof (double));
	for (j = 0; j < x->n; j++) {
		double	alpha;
		double	nrmj;
		int		size = (mm_is_sparse (x->typecode)) ? x->p[j + 1] - x->p[j] : x->m;
		double	*xj = x->data + ((mm_is_sparse (x->typecode)) ? x->p[j] : j * x->m);
		nrmj = dnrm2_ (&size, xj, &ione);
		alpha = 1. / nrmj;
		dscal_ (&size, &alpha, xj, &ione);
		nrm[j] = nrmj;
	}
	return nrm;
}

/* centering lreg->y,
 * and set lreg->meany[0] = mean( y ) */
void
linreg_centering_y (linreg *lreg)
{
	lreg->meany = centering (lreg->y);
	lreg->ycentered = true;
	return;
}

/* centering each col of lreg->x,
 * and set lreg->meanx[j] = mean( X(:,j) ) */
void
linreg_centering_x (linreg *lreg)
{
	lreg->meanx = centering (lreg->x);
	lreg->xcentered = true;
	return;
}

/* normalizing each col of lreg->x,
 * and set lreg->normx[j] = mean( X(:,j) ) */
void
linreg_normalizing_x (linreg *lreg)
{
	lreg->normx = normalizing (lreg->x);
	lreg->xnormalized = true;
	return;
}

/* standardizing lreg->x */
void
linreg_standardizing_x (linreg *lreg)
{
	linreg_centering_x (lreg);
	linreg_normalizing_x (lreg);
	return;
}

/* set lreg->pen = pen, and set lreg->scale2 = 1 / (a + b * lambda2) */
void
linreg_set_penalty (linreg *lreg, const double lambda2, const mm_mtx *d)
{
	if (d && lreg->x->n != d->n)
		linreg_error ("linreg_set_penalty", "penalty *pen->p must be same as linreg *lreg->p.", __FILE__, __LINE__);
	if (lambda2 > linreg_double_eps ()) lreg->lambda2 = lambda2;
	lreg->d = d;
	return;
}
