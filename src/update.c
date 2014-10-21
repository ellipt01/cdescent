/*
 * update.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <math.h>
#include <cdescent.h>

#include "private.h"
#include "atomic.h"

/* y += alpha * x(:,j) */
static void
axjpy (bool atomic, const double alpha, const mm_real *x, const int j, mm_dense *y)
{
	return (atomic) ? mm_real_axjpy_atomic (alpha, x, j, y) : mm_real_axjpy (alpha, x, j, y);
}

/* update intercept: (sum (y) - sum(X) * beta) / n */
static void
update_intercept (cdescent *cd)
{
	cd->b = 0.;
	// b += bar(y)
	if (!cd->lreg->ycentered) cd->b += cd->lreg->sy;
	// b -= bar(X) * beta
	if (!cd->lreg->xcentered) cd->b -= ddot_ (&cd->lreg->x->n, cd->lreg->sx, &ione, cd->beta->data, &ione);
	if (fabs (cd->b) > 0.) cd->b /= (double) cd->lreg->x->m;
	return;
}

/* update abs max: *amax = max (*amax, absval) */
static void
update_amax (bool atomic, double *amax, double absval)
{
	if (atomic) atomic_max (amax, absval);
	else if (*amax < absval) *amax = absval;
	return;
}

static void
cdescent_update (cdescent *cd, int j, bool atomic, double *amax_eta)
{
	// eta(j) = beta_new(j) - beta_prev(j)
	double	etaj = cdescent_beta_stepsize (cd, j);
	double	abs_etaj = fabs (etaj);

	if (abs_etaj > 0.) {
		// update beta: beta(j) += eta(j)
		cd->beta->data[j] += etaj;
		// update mu (= X * beta): mu += etaj * X(:,j)
		axjpy (atomic, etaj, cd->lreg->x, j, cd->mu);
		// update nu (= D * beta) if lambda2 != 0 && cd->nu != NULL: nu += etaj * D(:,j)
		if (!cd->lreg->is_regtype_lasso) axjpy (atomic, etaj, cd->lreg->d, j, cd->nu);
		// update max( |eta| )
		update_amax (atomic, amax_eta, abs_etaj);
	}
	return;
}

/*** progress coordinate descent update for one full cycle ***/
bool
cdescent_update_cyclic_once_cycle (cdescent *cd)
{
	int		j;
	bool	atomic = false;
	double	amax_eta;	// max of |eta(j)| = |beta_new(j) - beta_prev(j)|

	/* b = (sum(y) - sum(X) * beta) / n.
	 * so, if y or X are not centered,
	 * i.e. sum(y) != 0 or sum(X) != 0,
	 * b must be updated on each cycle. */
	if (!cd->lreg->xcentered) update_intercept (cd);

	amax_eta = 0.;

	/*** single "one-at-a-time" update of cyclic coordinate descent ***/
#ifdef _OPENMP

	if (cd->parallel) atomic = true;	// multiple threads

#pragma omp parallel for
	for (j = 0; j < cd->lreg->x->n; j++) {
#else
	for (j = 0; j < cd->lreg->x->n; j++) {
#endif
		cdescent_update (cd, j, atomic, &amax_eta);
	}
	cd->nrm1 = mm_real_xj_asum (cd->beta, 0);

	return (amax_eta < cd->tolerance);
}

/*** cyclic coordinate descent
 * repeat coordinate descent until solution is converged ***/
bool
cdescent_update_cyclic (cdescent *cd, const int maxiter)
{
	int		iter = 0;
	bool	converged = false;

	while (!converged) {

		converged = cdescent_update_cyclic_once_cycle (cd);

		if (++iter >= maxiter) {
			printf_warning ("cdescent_cyclic", "reaching max number of iterations.", __FILE__, __LINE__);
			break;
		}
	}
	cd->total_iter += iter;
	return converged;
}
