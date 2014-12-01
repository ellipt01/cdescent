/*
 * cdescent.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include <cdescent.h>

#include "private/private.h"

/* allocate cdescent object */
static cdescent *
cdescent_alloc (void)
{
	cdescent	*cd = (cdescent *) malloc (sizeof (cdescent));
	if (cd == NULL) return NULL;

	cd->was_modified = false;

	cd->is_regtype_lasso = true;

	cd->lreg = NULL;

	cd->lambda1_max = 0.;
	cd->lambda1 = 0.;
	cd->w = NULL;

	cd->tolerance = 0.;

	cd->nrm1 = 0.;
	cd->b = 0.;
	cd->beta = NULL;
	cd->mu = NULL;
	cd->nu = NULL;

	cd->parallel = false;
	cd->total_iter = 0;

	return cd;
}

/*** create new cdescent object ***/
cdescent *
cdescent_new (const linregmodel *lreg, const double tol, const int maxiter, bool parallel)
{
	cdescent	*cd;

	if (!lreg) error_and_exit ("cdescent_new", "linregmodel *lreg is empty.", __FILE__, __LINE__);

	cd = cdescent_alloc ();
	if (cd == NULL) error_and_exit ("cdescent_new", "failed to allocate object.", __FILE__, __LINE__);

	cd->was_modified = false;

	cd->lreg = lreg;

	/* if lreg->lambda2 == 0 || lreg->d == NULL, regression type is Lasso */
	cd->is_regtype_lasso =  (cd->lreg->lambda2 < DBL_EPSILON || cd->lreg->d == NULL);

	cd->tolerance = tol;

	cd->lambda1_max = pow (10., lreg->log10camax);
	cd->lambda1 = cd->lambda1_max;

	if (!lreg->ycentered) cd->b = *(lreg->sy) / (double) lreg->y->m;

	cd->beta = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, lreg->x->n, 1, lreg->x->n);
	mm_real_set_all (cd->beta, 0.);	// in initial, set to 0

	// mu = X * beta
	cd->mu = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, lreg->x->m, 1, lreg->x->m);
	mm_real_set_all (cd->mu, 0.);	// in initial, set to 0

	// nu = D * beta
	if (!cd->is_regtype_lasso) {
		cd->nu = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, lreg->d->m, 1, lreg->d->m);
		mm_real_set_all (cd->nu, 0.);	// in initial, set to 0
	}

	cd->maxiter = maxiter;
	cd->parallel = parallel;

	return cd;
}

/*** free cdescent object ***/
void
cdescent_free (cdescent *cd)
{
	if (cd) {
		if (cd->w) mm_real_free (cd->w);
		if (cd->beta) mm_real_free (cd->beta);
		if (cd->mu) mm_real_free (cd->mu);
		if (cd->nu) mm_real_free (cd->nu);
		free (cd);
	}
	return;
}

/*** set penalty factor of adaptive L1 regression ***/
bool
cdescent_set_penalty_factor (cdescent *cd, const mm_dense *w)
{
	int		j;
	if (w == NULL) return false;

	if (cd == NULL) error_and_exit ("cdescent_set_penalty_factor", "cdescent *cd is empty.", __FILE__, __LINE__);
	/* check whether w is dense general */
	if (!mm_real_is_dense (w)) error_and_exit ("cdescent_set_penalty_factor", "w must be dense.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (w)) error_and_exit ("cdescent_set_penalty_factor", "w must be general.", __FILE__, __LINE__);
	/* check whether w is vector */
	if (w->n != 1) error_and_exit ("cdescent_set_penalty_factor", "w must be vector.", __FILE__, __LINE__);
	/* check dimensions of x and w */
	if (w->m != cd->lreg->x->n) error_and_exit ("cdescent_set_penalty_factor", "dimensions of w and cd->lreg->x do not match.", __FILE__, __LINE__);

	/* copy w */
	cd->w = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, w->m, 1, w->nnz);
	for (j = 0; j < w->nnz; j++) cd->w->data[j] = fabs (w->data[j]);
	return (cd->w != NULL);
}

/*** set cd->lambda1
 * if designated lambda1 >= cd->lambda1_max, cd->lambda1 is set to cd->lambda1_max and return false
 * else cd->lambda1 is set to lambda1 and return true ***/
bool
cdescent_set_lambda1 (cdescent *cd, const double lambda1)
{
	if (cd->lambda1_max <= lambda1) {
		cd->lambda1 = cd->lambda1_max;
		return false;
	}
	cd->lambda1 = lambda1;
	return true;
}

/*** set cd->lambda1 to 10^log10_lambda1 ***/
 bool
cdescent_set_log10_lambda1 (cdescent *cd, const double log10_lambda1)
{
	return cdescent_set_lambda1 (cd, pow (10., log10_lambda1));
}
