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
linreg_alloc (const int n, const int p, double *y, double *x)
{
	int			nz;
	linreg		*lreg;

	if (!y) linreg_error ("lisys_alloc", "vector *y is empty.", __FILE__, __LINE__);
	if (!x) linreg_error ("lisys_alloc", "matrix *x is empty.", __FILE__, __LINE__);
	if (n <= 0) linreg_error ("lisys_alloc", "n must be > 0.", __FILE__, __LINE__);
	if (p <= 0) linreg_error ("lisys_alloc", "p must be > 0.", __FILE__, __LINE__);

	lreg = (linreg *) malloc (sizeof (linreg));

	lreg->n = n;
	lreg->p = p;
	nz = n * p;

	lreg->y = mm_mtx_real_new (false, false, n, 1, n);
	lreg->y->data = y;

	lreg->x = mm_mtx_real_new (true, false, n, p, nz);
	lreg->x->i = (int *) malloc (nz * sizeof (int));
	lreg->x->j = (int *) malloc (nz * sizeof (int));
	lreg->x->p = (int *) malloc ((p + 1) * sizeof (int));
	lreg->x->data = x;
	int		i, j, k = 0;
	lreg->x->p[0] = 0;
	for (j = 0; j < p; j++) {
		for (i = 0; i < n; i++) {
			lreg->x->i[k] = i;
			lreg->x->j[k] = j;
			k++;
		}
		lreg->x->p[j + 1] = k;
	}

	lreg->x1 = lreg->x->data;

	/* By default, data is assumed to be not centered or standardized */
	lreg->meany = NULL;
	lreg->meanx = NULL;
	lreg->normx = NULL;
	lreg->ycentered = false;
	lreg->xcentered = false;
	lreg->xnormalized = false;

	lreg->lambda2 = 0.;
	lreg->pen = NULL;

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
centering_mm_mtx_real (mm_mtx *x)
{
	int		i, j;
	double	*mean = (double *) malloc (x->n * sizeof (double));
	for (j = 0; j < x->n; j++) {
		double	meanj = 0.;
		if (mm_is_sparse (x->typecode)) {
			for (i = x->p[j]; i < x->p[j + 1]; i++) meanj += x->data[i];
		} else {
			for (i = 0; i < x->n; i++) meanj += x->data[i + j * x->m];
		}
		meanj /= (double) x->m;
		if (mm_is_sparse (x->typecode)) {
			for (i = x->p[j]; i < x->p[j + 1]; i++) x->data[i] -= meanj;
		} else {
			for (i = 0; i < x->n; i++) x->data[i + j * x->m] -= meanj;
		}
		mean[j] = meanj;
	}
	return mean;
}

/* normalizing each column of matrix:
 * x(:, j) -> x(:, j) / norm(x(:, j)) */
static double *
normalizing_mm_mtx_real (mm_mtx *x)
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
	lreg->meany = centering_mm_mtx_real (lreg->y);
	lreg->ycentered = true;
	return;
}

/* centering each col of lreg->x,
 * and set lreg->meanx[j] = mean( X(:,j) ) */
void
linreg_centering_x (linreg *lreg)
{
	lreg->meanx = centering_mm_mtx_real (lreg->x);
	lreg->xcentered = true;
	return;
}

/* normalizing each col of lreg->x,
 * and set lreg->normx[j] = mean( X(:,j) ) */
void
linreg_normalizing_x (linreg *lreg)
{
	lreg->normx = normalizing_mm_mtx_real (lreg->x);
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

penalty *
penalty_alloc (const int pj, const int p, const double *d)
{
	penalty	*pen;

	if (!d) linreg_error ("penalty_alloc", "matrix *d is empty.", __FILE__, __LINE__);
	pen = (penalty *) malloc (sizeof (penalty));
	pen->pj = pj;
	pen->p = p;
	pen->d = d;
	return pen;
}

void
penalty_free (penalty *pen)
{
	if (pen) free (pen);
	return;
}

/* set lreg->pen = pen, and set lreg->scale2 = 1 / (a + b * lambda2) */
void
linreg_set_penalty (linreg *lreg, const double lambda2, const penalty *pen)
{
	if (pen && lreg->p != pen->p)
		linreg_error ("linreg_set_penalty", "penalty *pen->p must be same as linreg *lreg->p.", __FILE__, __LINE__);

	if (lambda2 > linreg_double_eps ()) lreg->lambda2 = lambda2;
	lreg->pen = pen;
	return;
}
