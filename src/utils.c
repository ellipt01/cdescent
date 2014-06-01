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

extern double	cdescent_jth_scale2 (const int j, const cdescent *cd);

static void
array_set_all (const int n, double *x, const double val)
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

	// c = X' * y
	cd->c = (double *) malloc (lreg->p * sizeof (double));
	dgemv_ ("T", &lreg->n, &lreg->p, &done, lreg->x, &lreg->n, lreg->y, &ione, &dzero, cd->c, &ione);

	cd->camax = fabs (cd->c[idamax_ (&lreg->p, cd->c, &ione) - 1]);

	cd->lambda1 = (cd->camax < lambda1) ? floor (cd->camax) + 1. : lambda1;

	cd->b = 0.;
	cd->nrm1 = 0.;
	cd->beta = (double *) malloc (lreg->p * sizeof (double));
	array_set_all (lreg->p, cd->beta, 0.);

	cd->mu = (double *) malloc (lreg->n * sizeof (double));
	array_set_all (lreg->n, cd->mu, 0.);

	cd->nrm1_prev = 0.;
	cd->beta_prev = (double *) malloc (lreg->p * sizeof (double));

	/* sum y */
	cd->sy = 0.;
	if (!lreg->ycentered) {
		int		i;
		for (i = 0; i < cd->lreg->n; i++) cd->sy += cd->lreg->y[i];
		cd->b = cd->sy / (double) lreg->n;
	}

	/* sx(j) = sum X(:,j) */
	if (!lreg->xcentered) {
		int		j;
		cd->sx = (double *) malloc (lreg->p * sizeof (double));
		for (j = 0; j < lreg->p; j++) {
			int				i;
			const double	*xj = lreg->x + LINREG_INDEX_OF_MATRIX (0, j, lreg->n);
			cd->sx[j] = 0.;
			for (i = 0; i < lreg->n; i++) cd->sx[j] += xj[i];
		}
	} else cd->sx = NULL;


	/* xtx = diag (X' * X) */
	if (!lreg->xnormalized) {
		int				j;
		cd->xtx = (double *) malloc (lreg->p * sizeof (double));
		for (j = 0; j < lreg->p; j++) {
			const double	*xj = lreg->x + LINREG_INDEX_OF_MATRIX (0, j, lreg->n);
			cd->xtx[j] = ddot_ (&lreg->n, xj, &ione, xj, &ione);
		}
	} else cd->xtx = NULL;

	/* dtd = diag (D' * D) */
	if (lreg->pentype == PENALTY_USERDEF) {
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
		if (cd->beta_prev) free (cd->beta_prev);

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
