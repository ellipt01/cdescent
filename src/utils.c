/*
 * utils.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <stdlib.h>
#include <cdescent.h>

#include "linreg_private.h"

static void
array_set_all (const size_t n, double *x, const double val)
{
	int		i;
	for (i = 0; i < n; i++) x[i] = val;
	return;
}

cdescent *
cdescent_alloc (const linreg *lreg, const double lambda1, const double tol)
{
	cdescent	*cd;

	if (!lreg) linreg_error ("cdescent_alloc", "linreg *lreg is empty.", __FILE__, __LINE__);


	cd = (cdescent *) malloc (sizeof (cdescent));

	cd->lreg = lreg;

	cd->tolerance = tol;
	cd->lambda1 = lambda1;

	// c = X' * y
	cd->c = (double *) malloc (lreg->p * sizeof (double));
	{
		int		n = (int) lreg->n;
		int		p = (int) lreg->p;
		dgemv_ ("T", &n, &p, &done, lreg->x, &n, lreg->y, &ione, &dzero, cd->c, &ione);
	}
	cd->camax = cd->c[idamax_ (LINREG_CINTP (lreg->p), cd->c, &ione)];

	cd->h = 0.;
	cd->nrm1 = 0.;
	cd->beta = (double *) malloc (lreg->p * sizeof (double));
	array_set_all (lreg->p, cd->beta, 0.);

	cd->mu = (double *) malloc (lreg->n * sizeof (double));
	array_set_all (lreg->n, cd->mu, 0.);

	cd->beta_prev = (double *) malloc (lreg->p * sizeof (double));

	/* xtx = diag (X' * X) */
	if (!lreg->xnormalized) {
		int				j;
		size_t			n = lreg->n;
		size_t			p = lreg->p;
		const double	*x = lreg->x;
		cd->xtx = (double *) malloc (p * sizeof (double));
		for (j = 0; j < p; j++) {
			const double	*xj = x + LINREG_INDEX_OF_MATRIX (0, j, n);
			cd->xtx[j] = ddot_ (LINREG_CINTP (n), xj, &ione, xj, &ione);
		}
	} else cd->xtx = NULL;

	/* jtj = diag (J' * J) */
	if (lreg->pentype == PENALTY_USERDEF) {
		int				j;
		size_t			p = lreg->p;
		size_t			pj = lreg->pen->pj;
		const double	*jr = lreg->pen->r;
		cd->jtj = (double *) malloc (p * sizeof (double));
		for (j = 0; j < p; j++) {
			const double	*jrj = jr + LINREG_INDEX_OF_MATRIX (0, j, pj);
			cd->jtj[j] = ddot_ (LINREG_CINTP (pj), jrj, &ione, jrj, &ione);
		}
	} else cd->jtj = NULL;

	return cd;
}

void
cdescent_free (cdescent *cd)
{
	if (cd) {
		if (cd->c) free (cd->c);
		if (cd->beta) free (cd->beta);
		if (cd->mu) free (cd->mu);
		if (cd->beta_prev) free (cd->beta_prev);
		free (cd);
	}
	return;
}

void
cdescent_set_lambda1 (cdescent *cd, const double lambda1)
{
	cd->lambda1 = lambda1;
	return;
}

double *
cdescent_copy_beta (const cdescent *cd, bool scaling)
{
	size_t	p = cd->lreg->p;
	double	*beta = (double *) malloc (p * sizeof (double));
	dcopy_ (LINREG_CINTP (p), cd->beta, &ione, beta, &ione);
	if (scaling && !cdescent_is_regtype_lasso (cd)) {
		int		j;
		double	lambda2 = cd->lreg->lambda2;
		for (j = 0; j < p; j++) {
			double	xtx = (cd->lreg->xnormalized) ? 1. : cd->xtx[j];
			double	jtj = (cdescent_is_regtype_ridge (cd)) ? 1. : cd->jtj[j];
			beta[j] *= (xtx + jtj * lambda2);
		}
	}
	return beta;
}

/* lambda2 <= eps, regression type = lasso */
bool
cdescent_is_regtype_lasso (const cdescent *cd)
{
	return (cd->lreg->pentype == NO_PENALTY);
}

/* lambda2 > eps && l->lreg->pen == NULL, regression type = Ridge */
bool
cdescent_is_regtype_ridge (const cdescent *cd)
{
	return (cd->lreg->pentype == PENALTY_RIDGE);
}

bool
cdescent_is_regtype_userdef (const cdescent *cd)
{
	return (cd->lreg->pentype == PENALTY_USERDEF);
}
