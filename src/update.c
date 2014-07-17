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

/*** update intercept: (sum (y) - sum(X) * beta) / n ***/
static void
update_intercept (cdescent *cd)
{
	if (cd->lreg->ycentered && cd->lreg->xcentered) cd->b = 0.;
	else {
		double	nb = 0.;	// n * b
		// b += bar(y)
		if (!cd->lreg->ycentered) nb += cd->lreg->sy;
		// b -= bar(X) * beta
		if (!cd->lreg->xcentered) nb -= ddot_ (&cd->lreg->x->n, cd->lreg->sx, &ione, cd->beta->data, &ione);
		cd->b = nb / (double) cd->lreg->x->m;
	}
	return;
}

/* update mm_dense *mm: mm += x(:,j) * val */
static void
update_mm_dense (bool atomic, mm_dense *mm, int j, mm_real *x, const double val)
{
	return (atomic) ? mm_real_axjpy_atomic (val, j, x, mm) : mm_real_axjpy (val, j, x, mm);
}

/* update abs max */
static void
update_amax (bool atomic, double *amax, double absval)
{
	if (atomic) atomic_max (amax, absval);
	else if (*amax < absval) *amax = absval;
	return;
}

/*** progress coordinate descent update for one full cycle ***/
bool
cdescent_update_cyclic_once_cycle (cdescent *cd)
{
	int		j;
	bool	atomic;
	double	amax_change;	// max of |eta(j)| = |beta_new(j) - beta_prev(j)|

	/* b = (sum(y) - sum(X) * beta) / n.
	 * so, if y or X are not centered,
	 * i.e. sum(y) != 0 or sum(X) != 0,
	 * b must be updated on each cycle. */
	if (!cd->lreg->xcentered) update_intercept (cd);

	amax_change = 0.;

	/*** single "one-at-a-time" update of cyclic coordinate descent ***/
#ifdef _OPENMP
	if (cd->parallel) {	// multiple threads
		atomic = true;

#pragma omp parallel for
		for (j = 0; j < cd->lreg->x->n; j++) {
			// eta(j) = beta_new(j) - beta_prev(j)
			double	etaj = cdescent_beta_stepsize (cd, j);
			double	abs_etaj = fabs (etaj);

			if (abs_etaj > 0.) {
				// update beta: beta(j) += eta(j)
				cd->beta->data[j] += etaj;
				// update mu (= X * beta): mu += X(:,j) * etaj
				update_mm_dense (atomic, cd->mu, j, cd->lreg->x, etaj);
				// update nu (= D * beta) if lambda2 != 0 && cd->nu != NULL: nu += D(:,j) * etaj
				if (!cd->lreg->is_regtype_lasso) update_mm_dense (atomic, cd->nu, j, cd->lreg->d, etaj);
				// update max( |eta| )
				update_amax (atomic, &amax_change, abs_etaj);
			}
		}

	} else {	// single thread
#endif
		atomic = false;

		for (j = 0; j < cd->lreg->x->n; j++) {
			double	etaj = cdescent_beta_stepsize (cd, j);
			double	abs_etaj = fabs (etaj);

			if (abs_etaj > 0.) {
				cd->beta->data[j] += etaj;
				update_mm_dense (atomic, cd->mu, j, cd->lreg->x, etaj);
				if (!cd->lreg->is_regtype_lasso) update_mm_dense (atomic, cd->nu, j, cd->lreg->d, etaj);
				update_amax (atomic, &amax_change, abs_etaj);
			}
		}
#ifdef _OPENMP
	}
#endif

	cd->nrm1 = mm_real_xj_asum (0, cd->beta);

	return (amax_change < cd->tolerance);
}

/*** cyclic coordinate descent ***/
/* repeat coordinate descent until solution is converged */
bool
cdescent_update_cyclic (cdescent *cd, const int maxiter)
{
	int		iter = 0;
	bool	converged = false;

	while (!converged) {

		converged = cdescent_update_cyclic_once_cycle (cd);

		if (++iter >= maxiter) {
			print_warning ("cdescent_cyclic", "reaching max number of iterations.", __FILE__, __LINE__);
			break;
		}
	}
	cd->total_iter += iter;
	return converged;
}
