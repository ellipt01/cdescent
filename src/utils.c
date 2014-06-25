/*
 * utils.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include <cdescent.h>

#include "private.h"

cdescent *
cdescent_alloc (void)
{
	cdescent	*cd = (cdescent *) malloc (sizeof (cdescent));

	cd->lreg = NULL;
	cd->tolerance = 0.;
	cd->c = NULL;
	cd->logcamax = 0.;
	cd->lambda1_max = 0.;
	cd->lambda1 = 0.;

	cd->b = 0.;
	cd->nrm1 = 0.;
	cd->beta = NULL;
	cd->mu = NULL;

	cd->nu = NULL;
	cd->sy = 0.;
	cd->sx = NULL;
	cd->xtx = NULL;
	cd->dtd = NULL;

	return cd;
}

cdescent *
cdescent_new (const linreg *lreg, const double tol)
{
	double		camax;
	cdescent	*cd;

	if (!lreg) cdescent_error ("cdescent_alloc", "linreg *lreg is empty.", __FILE__, __LINE__);

	cd = cdescent_alloc ();

	cd->lreg = lreg;

	cd->tolerance = tol;

	// c = X' * y
	cd->c = mm_real_x_dot_y (true, 1., lreg->x, lreg->y, 0.);

	// camax = max ( abs (c) )
	camax = fabs (cd->c->data[idamax_ (&cd->c->nz, cd->c->data, &ione) - 1]);
	cd->logcamax = floor (log10 (camax)) + 1.;
	cd->lambda1_max = pow (10., cd->logcamax);
	cd->lambda1 = cd->lambda1_max;

	cd->beta = mm_real_new (MM_REAL_DENSE, MM_REAL_UNSYMMETRIC, lreg->x->n, 1, lreg->x->n);
	cd->beta->data = (double *) malloc (cd->beta->nz * sizeof (double));
	mm_real_set_all (cd->beta, 0.);

	// mu = X * beta
	cd->mu = mm_real_new (false, false, lreg->x->m, 1, lreg->x->m);
	cd->mu->data = (double *) malloc (lreg->x->m * sizeof (double));
	mm_real_set_all (cd->mu, 0.);

	// nu = D * beta
	if (!cdescent_is_regtype_lasso (cd)) {
		cd->nu = mm_real_new (false, false, lreg->d->m, 1, lreg->d->m);
		cd->nu->data = (double *) malloc (lreg->d->nz * sizeof (double));
		mm_real_set_all (cd->nu, 0.);
	}

	/* sum y */
	if (!lreg->ycentered) {
		cd->sy = mm_real_sum (lreg->y);
		cd->b = cd->sy / (double) lreg->y->m;
	}

	/* sx(j) = sum X(:,j) */
	if (!lreg->xcentered) {
		int		j;
		cd->sx = (double *) malloc (lreg->x->n * sizeof (double));
		for (j = 0; j < lreg->x->n; j++) cd->sx[j] = mm_real_xj_sum (j, lreg->x);
	}

	/* xtx = diag (X' * X) */
	if (!lreg->xnormalized) {
		int		j;
		cd->xtx = (double *) malloc (lreg->x->n * sizeof (double));
		for (j = 0; j < lreg->x->n; j++) cd->xtx[j] = pow (mm_real_xj_nrm2 (j, lreg->x), 2.);
	}

	/* dtd = diag (D' * D) */
	if (!cdescent_is_regtype_lasso (cd)) {
		int				j;
		cd->dtd = (double *) malloc (lreg->d->n * sizeof (double));
		for (j = 0; j < lreg->d->n; j++) cd->dtd[j] = pow (mm_real_xj_nrm2 (j, lreg->d), 2.);
	}

	return cd;
}

void
cdescent_free (cdescent *cd)
{
	if (cd) {
		if (cd->c) mm_real_free (cd->c);
		if (cd->beta) mm_real_free (cd->beta);
		if (cd->mu) mm_real_free (cd->mu);
		if (cd->nu) mm_real_free (cd->nu);

		if (cd->sx) free (cd->sx);
		if (cd->xtx) free (cd->xtx);
		if (cd->dtd) free (cd->dtd);

		free (cd);
	}
	return;
}

void
cdescent_set_lambda1 (cdescent *cd, const double lambda1)
{
	cd->lambda1 = (cd->lambda1_max <= lambda1) ? cd->lambda1_max : lambda1;
	return;
}

void
cdescent_set_log10_lambda1 (cdescent *cd, const double log10_lambda1)
{
	cdescent_set_lambda1 (cd, pow (10., log10_lambda1));
	return;
}

/* if lambda2 < eps || d == NULL, regression type = lasso */
bool
cdescent_is_regtype_lasso (const cdescent *cd)
{
	return (cd->lreg->lambda2 < cdescent_double_eps () || cd->lreg->d == NULL);
}
