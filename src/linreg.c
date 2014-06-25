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

#include "private.h"

linreg *
linreg_alloc (mm_mtx *y, mm_mtx *x, const double lambda2, mm_mtx *d)
{
	linreg		*lreg;

	if (!y) cdescent_error ("linreg_alloc", "vector *y is empty.", __FILE__, __LINE__);
	if (!x) cdescent_error ("linreg_alloc", "matrix *x is empty.", __FILE__, __LINE__);
	if (!mm_is_dense (y->typecode)) cdescent_error ("linreg_alloc", "vector *y must be dense.", __FILE__, __LINE__);
	if (y->m != x->m) cdescent_error ("linreg_alloc", "size of matrix *x and vector *y are incompatible.", __FILE__, __LINE__);
	if (d && x->n != d->n) cdescent_error ("linreg_alloc", "size of matrix *x and *d incompatible.", __FILE__, __LINE__);

	lreg = (linreg *) malloc (sizeof (linreg));

	lreg->y = mm_mtx_copy (y);
	lreg->x = mm_mtx_copy (x);

	lreg->lambda2 = (lambda2 > cdescent_double_eps ()) ? lambda2 : 0.;
	lreg->d = (d) ? mm_mtx_copy (d) : NULL;

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
		if (lreg->y) mm_mtx_free (lreg->y);
		if (lreg->x) mm_mtx_free (lreg->x);
		if (lreg->d) mm_mtx_free (lreg->d);

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
centering (mm_dense *x)
{
	int		i, j;
	double	*mean;
	if (mm_is_sparse (x->typecode)) return NULL;
	mean = (double *) malloc (x->n * sizeof (double));
	for (j = 0; j < x->n; j++) {
		double	meanj = 0.;
		for (i = 0; i < x->m; i++) meanj += x->data[i + j * x->m];
		meanj /= (double) x->m;
		for (i = 0; i < x->m; i++) x->data[i + j * x->m] -= meanj;
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
	if (mm_is_sparse (lreg->y->typecode)) {
		mm_dense	*d = mm_mtx_sparse_to_dense (lreg->y);
		mm_mtx_free (lreg->y);
		lreg->y = d;
	}
	lreg->meany = centering (lreg->y);
	lreg->ycentered = true;
	return;
}

/* centering each col of lreg->x,
 * and set lreg->meanx[j] = mean( X(:,j) ) */
void
linreg_centering_x (linreg *lreg)
{
	if (mm_is_sparse (lreg->x->typecode)) {
		mm_dense	*d = mm_mtx_sparse_to_dense (lreg->x);
		mm_mtx_free (lreg->x);
		lreg->x = d;
	}
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
