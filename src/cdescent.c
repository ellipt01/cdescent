/*
 * cdescent.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <math.h>
#include <cdescent.h>

#include "private.h"

/*** progress coordinate descent update for one full cycle ***/
bool
cdescent_cyclic_once_cycle (cdescent *cd)
{
	int			j;
	double		nrm2;

	/* b = (sum(y) - sum(X) * beta) / n.
	 * so, if y or X are not centered,
	 * i.e. sum(y) != 0 or sum(X) != 0,
	 * b must be updated on each cycle. */
	if (!cd->lreg->xcentered) cd->b = cdescent_update_intercept (cd);

	nrm2 = 0.;

	/*** single "one-at-a-time" update of cyclic coordinate descent ***/
	for (j = 0; j < cd->lreg->x->n; j++) {
		// era[j] = beta[j] - beta_prev[j]
		double	etaj = cdescent_beta_stepsize (cd, j);

		if (fabs (etaj) > 0.) {
			// update beta[j]: beta[j] += eta[j]
			cd->beta->data[j] += etaj;

			// mu += eta[j] * X(:, j)
			mm_mtx_axjpy (etaj, j, cd->lreg->x, cd->mu);

			/* not lasso, nu += eta[j] * D(:,j) */
			if (!cdescent_is_regtype_lasso (cd)) {
				mm_mtx_axjpy (etaj, j, cd->lreg->d, cd->nu);
			}

			nrm2 += pow (etaj, 2.);
		}

	}
	cd->nrm1 = mm_mtx_asum (cd->beta);

	return (sqrt (nrm2) < cd->tolerance);
}

/*** cyclic coordinate descent ***/
/* repeat coordinate descent until solution is converged */
bool
cdescent_cyclic (cdescent *cd, const int maxiter)
{
	int		iter = 0;
	bool	converged = false;

	while (!converged) {

		converged = cdescent_cyclic_once_cycle (cd);

		if (++iter >= maxiter) break;
	}

	return converged;
}
