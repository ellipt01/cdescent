/*
 * utils.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include <cdescent.h>

#include "linreg_private.h"

static void
array_set_all (int n, double *x, const double val)
{
	while (n--) *x++ = val;
	return;
}

cdescent *
cdescent_alloc (const linreg *lreg, const double lambda1, const double tol)
{
	double		camax;
	cdescent	*cd;

	if (!lreg) linreg_error ("cdescent_alloc", "linreg *lreg is empty.", __FILE__, __LINE__);

	cd = (cdescent *) malloc (sizeof (cdescent));

	cd->lreg = lreg;

	cd->tolerance = tol;

	// c = X' * y
	cd->c = mm_mtx_real_x_dot_y (true, 1., lreg->x, lreg->y, 0.);

	camax = fabs (cd->c->data[idamax_ (&cd->c->nz, cd->c->data, &ione) - 1]);
	cd->lambda1 = (camax < lambda1) ? floor (camax) + 1. : lambda1;

	cd->b = 0.;
	cd->nrm1 = 0.;
	cd->beta = (double *) malloc (lreg->p * sizeof (double));
	array_set_all (lreg->p, cd->beta, 0.);

	// mu = X * beta
	cd->mu = mm_mtx_real_new (false, false, lreg->x->m, 1, lreg->x->m);
	cd->mu->data = (double *) malloc (lreg->x->m * sizeof (double));
	mm_mtx_real_set_all (cd->mu, 0.);

	// nu = D * beta
	if (!cdescent_is_regtype_lasso (cd)) {
		cd->nu = mm_mtx_real_new (false, false, lreg->pen->pj, 1, lreg->pen->pj);
		cd->nu->data = (double *) malloc (lreg->pen->pj * sizeof (double));
		mm_mtx_real_set_all (cd->nu, 0.);
	} else cd->nu = NULL;

	/* sum y */
	cd->sy = 0.;
	if (!lreg->ycentered) {
		int		i;
		for (i = 0; i < lreg->y->m; i++) cd->sy += lreg->y->data[i];
		cd->b = cd->sy / (double) lreg->y->m;
	}

	/* sx(j) = sum X(:,j) */
	if (!lreg->xcentered) {
		int		j;
		cd->sx = (double *) malloc (lreg->p * sizeof (double));
		for (j = 0; j < lreg->p; j++) {
			int				i;
			cd->sx[j] = 0.;
			if (mm_is_sparse (lreg->x->typecode))
				for (i = lreg->x->p[j]; i < lreg->x->p[j + 1]; i++) cd->sx[j] += lreg->x->data[i];
			else
				for (i = 0; i < lreg->x->m; i++) cd->sx[j] += lreg->x->data[i + j * lreg->x->m];
		}
	} else cd->sx = NULL;


	/* xtx = diag (X' * X) */
	if (!lreg->xnormalized) {
		int				j;
		cd->xtx = (double *) malloc (lreg->x->n * sizeof (double));
		for (j = 0; j < lreg->x->n; j++) cd->xtx[j] = mm_mtx_real_xj_dot_xj (j, lreg->x);
	} else cd->xtx = NULL;

	/* dtd = diag (D' * D) */
	if (!cdescent_is_regtype_lasso (cd)) {
		int				j;
		int				p = lreg->p;
		int				pj = lreg->pen->pj;
		const double	*d = lreg->pen->d;
		cd->dtd = (double *) malloc (p * sizeof (double));
		for (j = 0; j < p; j++) {
			const double	*dj = d + LINREG_INDEX_OF_MATRIX (0, j, pj);
			cd->dtd[j] = ddot_ (&pj, dj, &ione, dj, &ione);
		}
	} else cd->dtd = NULL;

	return cd;
}

void
cdescent_free (cdescent *cd)
{
	if (cd) {
		if (cd->c) free (cd->c);
		if (cd->beta) free (cd->beta);
		if (cd->mu) free (cd->mu);
		if (cd->nu) free (cd->nu);

		if (cd->sx) free (cd->sx);
		if (cd->xtx) free (cd->xtx);
		if (cd->dtd) free (cd->dtd);

		free (cd);
	}
	return;
}

/* lambda2 <= eps, regression type = lasso */
bool
cdescent_is_regtype_lasso (const cdescent *cd)
{
	return (cd->lreg->lambda2 < linreg_double_eps () || cd->lreg->pen == NULL);
}
