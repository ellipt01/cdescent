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
	cd->beta = mm_mtx_real_new (MM_MTX_DENSE, MM_MTX_UNSYMMETRIC, lreg->x->n, 1, lreg->x->n);
	cd->beta->data = (double *) malloc (cd->beta->nz * sizeof (double));
	mm_mtx_real_set_all (cd->beta, 0.);

	// mu = X * beta
	cd->mu = mm_mtx_real_new (false, false, lreg->x->m, 1, lreg->x->m);
	cd->mu->data = (double *) malloc (lreg->x->m * sizeof (double));
	mm_mtx_real_set_all (cd->mu, 0.);

	// nu = D * beta
	if (!cdescent_is_regtype_lasso (cd)) {
		cd->nu = mm_mtx_real_new (false, false, lreg->d->m, 1, lreg->d->m);
		cd->nu->data = (double *) malloc (lreg->d->nz * sizeof (double));
		mm_mtx_real_set_all (cd->nu, 0.);
	} else cd->nu = NULL;

	/* sum y */
	cd->sy = 0.;
	if (!lreg->ycentered) {
		cd->sy = mm_mtx_real_sum (lreg->y);
		cd->b = cd->sy / (double) lreg->y->m;
	}

	/* sx(j) = sum X(:,j) */
	if (!lreg->xcentered) {
		int		j;
		cd->sx = (double *) malloc (lreg->x->n * sizeof (double));
		for (j = 0; j < lreg->x->n; j++) cd->sx[j] = mm_mtx_real_xj_sum (j, lreg->x);
	} else cd->sx = NULL;


	/* xtx = diag (X' * X) */
	if (!lreg->xnormalized) {
		int		j;
		cd->xtx = (double *) malloc (lreg->x->n * sizeof (double));
		for (j = 0; j < lreg->x->n; j++) cd->xtx[j] = pow (mm_mtx_real_xj_nrm2 (j, lreg->x), 2.);
	} else cd->xtx = NULL;

	/* dtd = diag (D' * D) */
	if (!cdescent_is_regtype_lasso (cd)) {
		int				j;
		cd->dtd = (double *) malloc (lreg->d->n * sizeof (double));
		for (j = 0; j < lreg->d->n; j++) cd->dtd[j] = pow (mm_mtx_real_xj_nrm2 (j, lreg->d), 2.);
	} else cd->dtd = NULL;

	return cd;
}

void
cdescent_free (cdescent *cd)
{
	if (cd) {
		if (cd->c) mm_mtx_free (cd->c);
		if (cd->beta) mm_mtx_free (cd->beta);
		if (cd->mu) mm_mtx_free (cd->mu);
		if (cd->nu) mm_mtx_free (cd->nu);

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
	return (cd->lreg->lambda2 < linreg_double_eps () || cd->lreg->d == NULL);
}
