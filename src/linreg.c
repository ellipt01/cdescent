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
linreg_alloc (void)
{
	linreg	*lreg = (linreg *) malloc (sizeof (linreg));
	lreg->has_copy = false;
	lreg->y = NULL;
	lreg->x = NULL;
	lreg->d = NULL;
	lreg->lambda2 = 0.;
	/* By default, data is assumed to be not centered or standardized */
	lreg->meany = NULL;
	lreg->meanx = NULL;
	lreg->normx = NULL;
	lreg->ycentered = false;
	lreg->xcentered = false;
	lreg->xnormalized = false;
	return lreg;
}

linreg *
linreg_new (mm_real *y, mm_real *x, const double lambda2, mm_real *d, bool has_copy)
{
	linreg	*lreg;

	if (!y) cdescent_error ("linreg_alloc", "vector *y is empty.", __FILE__, __LINE__);
	if (!x) cdescent_error ("linreg_alloc", "matrix *x is empty.", __FILE__, __LINE__);
	if (!mm_is_dense (y->typecode)) cdescent_error ("linreg_alloc", "vector *y must be dense.", __FILE__, __LINE__);
	if (y->m != x->m) cdescent_error ("linreg_alloc", "size of matrix *x and vector *y are not match.", __FILE__, __LINE__);
	if (d && x->n != d->n) cdescent_error ("linreg_alloc", "size of matrix *x and *d are not match.", __FILE__, __LINE__);

	lreg = linreg_alloc ();

	lreg->has_copy = has_copy;
	if (has_copy) {
		lreg->y = mm_real_copy (y);
		lreg->x = mm_real_copy (x);
		lreg->d = (d) ? mm_real_copy (d) : NULL;
	} else {
		lreg->x = x;
		lreg->y = y;
		lreg->d = d;
	}
	lreg->lambda2 = (lambda2 > cdescent_double_eps ()) ? lambda2 : 0.;

	return lreg;
}

void
linreg_free (linreg *lreg)
{
	if (lreg) {
		if (lreg->has_copy) {
			if (lreg->y) mm_real_free (lreg->y);
			if (lreg->x) mm_real_free (lreg->x);
			if (lreg->d) mm_real_free (lreg->d);
		}
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
normalizing (mm_real *x)
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
	if (mm_is_sparse (lreg->x->typecode)) 	mm_real_replace_sparse_to_dense (lreg->x);
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
