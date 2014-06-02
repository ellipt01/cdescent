/*
 * cdescent.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <math.h>
#include <cdescent.h>

#include "linreg_private.h"

/*** y += x(:, j) * delta ***/
static void
update_partially  (const int j, const double delta, int n, const double *x, double *y)
{
	const double	*xj = x + LINREG_INDEX_OF_MATRIX (0, j, n);	// X(:,j)
	daxpy_ (&n, &delta, xj, &ione, y, &ione);
	return;
}

/*** progress coordinate descent update for one full cycle ***/
bool
cdescent_cyclic_once_cycle (cdescent *cd)
{
	int			j;
	int			p = cd->lreg->p;
	double		nrm2;

	/* b = (sum(y) - sum(X) * beta) / n.
	 * so, if y or X are not centered,
	 * i.e. sum(y) != 0 or sum(X) != 0,
	 * b must be updated on each cycle. */
	if (!cd->lreg->xcentered) cd->b = cdescent_update_intercept (cd);

	nrm2 = 0.;

	/*** single "one-at-a-time" update of cyclic coordinate descent ***/
	for (j = 0; j < p; j++) {
		// deltaj = beta[j] - beta_prev[j]
		double	deltaj = cdescent_beta_stepsize (cd, j);

		// update beta[j]
		cd->beta[j] += deltaj;

		if (fabs (deltaj) > 0.) {
			int				n = cd->lreg->n;
			const double	*x = cd->lreg->x;
			// update mu : mu += X(:, j) * (beta[j] - beta_prev[j])
			update_partially (j, deltaj, n, x, cd->mu);

			/* user defined penalty (not lasso nor ridge) */
			if (cd->nu) {
				int			pj = cd->lreg->pen->pj;
				const double	*d = cd->lreg->pen->d;
				// update nu : nu += D(:,j) * (beta[j] - beta_prev[j])
				update_partially (j, deltaj, pj, d, cd->nu);
			}

			nrm2 += pow (deltaj, 2.);
		}

	}
	cd->nrm1 = dasum_ (&p, cd->beta, &ione);

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
