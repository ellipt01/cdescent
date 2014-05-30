/*
 * cdescent.c
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include <cdescent.h>

#include "linreg_private.h"

static double
soft_threshold (const double z, const double gamma)
{
	double	val = 0.;
	if (gamma < fabs (z)) val = (z > 0.) ? z - gamma : z + gamma;
	return val;
}

static double
get_jth_scale2 (const int j, const cdescent *cd)
{
	double		scale2;
	double		xtx = (cd->lreg->xnormalized) ? 1. : cd->xtx[j];
	if (cdescent_is_regtype_lasso (cd)) scale2 = 1. / xtx;
	else {
		double	jtj = (cdescent_is_regtype_ridge (cd)) ? 1. : cd->jtj[j];
		double	lambda2 = cd->lreg->lambda2;
		scale2 = 1. / (xtx + jtj * lambda2);
	}
	return scale2;
}

/*** y += x(:, j) * delta ***/
static void
update_partially  (const int j, const double delta, int n, const double *x, double *y)
{
	const double	*xj = x + LINREG_INDEX_OF_MATRIX (0, j, n);	// X(:,j)

	// y += x(:, j) * delta
	daxpy_ (&n, &delta, xj, &ione, y, &ione);

	return;
}

/* intercept = sum ( (r - X * beta) ) / n */
static double
eval_intercept (const cdescent *cd)
{
	int			n = cd->lreg->n;
	int			p = cd->lreg->p;
	double		nb = 0.;	// n * b

	if (cd->lreg->ycentered && cd->lreg->xcentered) return 0.;
	if (!cd->lreg->ycentered) nb += cd->sy;	// b = bar(y)
	if (!cd->lreg->xcentered) {	// b -= bar(X) * beta
		nb -= ddot_ (&p, cd->sx, &ione, cd->beta, &ione);
	}
	return nb / (double) n;	// return b
}

/*
 * z = X(:,j)' * y - X(:,j)' * X * beta + beta(j)
 *   - lambda2 * J(:,j)' * J * beta + lambda2 * J(:,j)' * J(:,j) * beta(j)
 */
static double
beta_j_updater (const int j, const cdescent *cd, double *jb)
{
	int				n = cd->lreg->n;
	double			cj = cd->c[j];	// X' * y
	const double	*x = cd->lreg->x;
	const double	*xj = x + LINREG_INDEX_OF_MATRIX (0, j, n);	// X(:,j)
	double			xjtm = ddot_ (&n, xj, &ione, cd->mu, &ione);	// X(:,j)' * mu

	double			xtx = (cd->lreg->xnormalized) ? 1. : cd->xtx[j];	// norm of X(:,j)

	double			z = cj - xjtm + xtx * cd->beta[j];

	// if y or X are not centered, need to consider intercept : z -= sum( X(:,j) ) * b
	if (!cd->lreg->xcentered) z -= cd->sx[j] * cd->b;

	/* user defined penalty (not lasso nor ridge) */
	if (cdescent_is_regtype_userdef (cd)) {
		/*
		 * z = z - lambda2 * J(:,j)' * J * beta
		 *   + lambda2 * J(:,j)' * J(:,j) * beta(j)
		 */
		int				pj = cd->lreg->pen->pj;
		double			lambda2 = cd->lreg->lambda2;
		const double	*jr = cd->lreg->pen->r;
		const double	*jrj = jr + LINREG_INDEX_OF_MATRIX (0, j, pj);	// J(:,j)

		// z -= lambda2 * J(:,j)' * (J * beta)
		z -= lambda2 * ddot_ (&pj, jrj, &ione, jb, &ione);

		// z += lambda2 * (J(:,j)' * J(:,j)) * beta(j)
		z += lambda2 * cd->jtj[j] * cd->beta[j];
	}
	return z;
}

/*
 * progress coordinate descent update for one cycle
 */
bool
cdescent_cyclic_once_cycle (cdescent *cd)
{
	int			j;
	int			p = cd->lreg->p;
	double		lambda1 = cd->lambda1;

	double		nrm;
	double		*delta = (double *) malloc (p * sizeof (double));
	bool		converged;

	double		*jb = NULL;	// J * beta
	if (cdescent_is_regtype_userdef (cd)) {
		// J * beta
		int				pj = cd->lreg->pen->pj;
		const double	*jr = cd->lreg->pen->r;
		jb = (double *) malloc (pj * sizeof (double));
		dgemv_ ("N", &pj, &p, &done, jr, &pj, cd->beta, &ione, &dzero, jb, &ione);
	}

	/* backup solution */
	cd->nrm1 = cd->nrm1_prev;
	dcopy_ (&p, cd->beta, &ione, cd->beta_prev, &ione);

	/* b = (sum(y) - sum(X) * beta) / n.
	 * so, if y or X are not centered,
	 * i.e. sum(y) != 0 or sum(X) != 0,
	 * b must be updated on each cycle. */
	if (!cd->lreg->xcentered) cd->b = eval_intercept (cd);

	/*** single "one-at-a-time" update of cyclic coordinate descent ***/
	for (j = 0; j < p; j++) {
		double		z;
		double		scale2 = get_jth_scale2 (j, cd);

		z = beta_j_updater (j, cd, jb);

		cd->beta[j] = scale2 * soft_threshold (z, lambda1);

		delta[j] = cd->beta[j] - cd->beta_prev[j];
		if (fabs (delta[j]) > 0.) {
			int			n = cd->lreg->n;
			const double	*x = cd->lreg->x;
			// mu += X(:, j) * (beta[j] - beta_prev[j])
			update_partially (j, delta[j], n, x, cd->mu);

			/* user defined penalty (not lasso nor ridge) */
			if (cdescent_is_regtype_userdef (cd)) {
				int			pj = cd->lreg->pen->pj;
				const double	*jr = cd->lreg->pen->r;
				// jb += J(:,j) * (beta[j] - beta_prev[j])
				update_partially (j, delta[j], pj, jr, jb);
			}
		}

	}
	if (jb) free (jb);

	cd->nrm1 = dasum_ (&p, cd->beta, &ione);

	nrm = dnrm2_ (&p, delta, &ione);
	free (delta);
	converged = (nrm < cd->tolerance);

	return converged;
}

/*
 * cyclic coordinate descent
 * repeat coordinate descent until solution is converged
 */
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
