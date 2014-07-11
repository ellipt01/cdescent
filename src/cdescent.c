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
	cd->lambda1_max = 0.;
	cd->lambda1 = 0.;

	cd->b = 0.;
	cd->nrm1 = 0.;
	cd->beta = NULL;
	cd->mu = NULL;
	cd->nu = NULL;

	cd->parallel = false;
	cd->total_iter = 0;

	return cd;
}

cdescent *
cdescent_new (const linregmodel *lreg, const double tol, bool parallel)
{
	cdescent	*cd;

	if (!lreg) error_and_exit ("cdescent_alloc", "linreg *lreg is empty.", __FILE__, __LINE__);

	cd = cdescent_alloc ();

	cd->lreg = lreg;

	cd->tolerance = tol;

	cd->lambda1_max = pow (10., lreg->logcamax);
	cd->lambda1 = cd->lambda1_max;

	cd->b = lreg->sy / (double) lreg->y->m;

	cd->beta = mm_real_new (MM_REAL_DENSE, false, lreg->x->n, 1, lreg->x->n);
	cd->beta->data = (double *) malloc (cd->beta->nz * sizeof (double));
	mm_real_set_all (cd->beta, 0.);

	// mu = X * beta
	cd->mu = mm_real_new (MM_REAL_DENSE, false, lreg->x->m, 1, lreg->x->m);
	cd->mu->data = (double *) malloc (lreg->x->m * sizeof (double));
	mm_real_set_all (cd->mu, 0.);

	// nu = D * beta
	if (!cd->lreg->regtype_is_lasso) {
		cd->nu = mm_real_new (MM_REAL_DENSE, false, lreg->d->m, 1, lreg->d->m);
		cd->nu->data = (double *) malloc (lreg->d->nz * sizeof (double));
		mm_real_set_all (cd->nu, 0.);
	}

	cd->parallel = parallel;

	return cd;
}

void
cdescent_free (cdescent *cd)
{
	if (cd) {
		if (cd->beta) mm_real_free (cd->beta);
		if (cd->mu) mm_real_free (cd->mu);
		if (cd->nu) mm_real_free (cd->nu);
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
