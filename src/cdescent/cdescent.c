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
update_partially  (const int j, const double delta, size_t n, const double *x, double *y)
{
	const double	*xj = x + LINREG_INDEX_OF_MATRIX (0, j, n);	// X(:,j)

	// y += x(:, j) * delta
	daxpy_ (LINREG_CINTP (n), &delta, xj, &ione, y, &ione);

	return;
}

static double
update_intercept (const int j, const cdescent *cd)
{
	int				i;
	double			h;
	double			*r;
	size_t			n = cd->lreg->n;
	const double	*y = cd->lreg->y;

	/* h = y - mu */
	r = (double *) malloc (n * sizeof (double));
	dcopy_ (LINREG_CINTP (n), y, &ione, r, &ione);
	daxpy_ (LINREG_CINTP (n), &dmone, cd->mu, &ione, r, &ione);

	h = 0.;
	for (i = 0; i < n; i++) h += r[i];
	free (r);

	return h;
}

/*
 * z = X(:,j)' * y - X(:,j)' * X * beta + beta(j)
 *   - lambda2 * J(:,j)' * J * beta + lambda2 * J(:,j)' * J(:,j) * beta(j)
 */
static double
beta_j_updater (const int j, const cdescent *cd, double *jb)
{
	size_t			n = cd->lreg->n;
	double			cj = cd->c[j];	// X' * y
	const double	*x = cd->lreg->x;
	const double	*xj = x + LINREG_INDEX_OF_MATRIX (0, j, n);	// X(:,j)
	double			xjtm = ddot_ (LINREG_CINTP (n), xj, &ione, cd->mu, &ione);	// X(:,j)' * mu

	double			xtx = (cd->lreg->xnormalized) ? 1. : cd->xtx[j];

	double			z = cj - xjtm + xtx * cd->beta[j];

	/* z = X(:,j)' * y - h - X(:,j)' * X * beta + beta(j) */
	if (!cd->lreg->ycentered && !cd->lreg->xcentered) z -= update_intercept (j, cd);

	/* user defined penalty (not lasso nor ridge) */
	if (cdescent_is_regtype_userdef (cd)) {
		/*
		 * z = z - lambda2 * J(:,j)' * J * beta
		 *   + lambda2 * J(:,j)' * J(:,j) * beta(j)
		 */
		size_t			pj = cd->lreg->pen->pj;
		double			lambda2 = cd->lreg->lambda2;
		const double	*jr = cd->lreg->pen->r;
		const double	*jrj = jr + LINREG_INDEX_OF_MATRIX (0, j, pj);	// J(:,j)

		// z -= lambda2 * J(:,j)' * (J * beta)
		z -= lambda2 * ddot_ (LINREG_CINTP (pj), jrj, &ione, jb, &ione);

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
	int				j;
	size_t			p = cd->lreg->p;
	double			lambda1 = cd->lambda1;

	double			nrm;
	double			*delta = (double *) malloc (p * sizeof (double));
	bool			converged;

	double			*jb = NULL;	// J * beta
	if (cdescent_is_regtype_userdef (cd)) {
		// J * beta
		size_t			pj = cd->lreg->pen->pj;
		const double	*jr = cd->lreg->pen->r;
		jb = (double *) malloc (pj * sizeof (double));
		dgemv_ ("N", LINREG_CINTP (pj), LINREG_CINTP (p), &done, jr, LINREG_CINTP (pj), cd->beta, &ione, &dzero, jb, &ione);
	}

	/* backup solution */
	cd->nrm1 = cd->nrm1_prev;
	dcopy_ (LINREG_CINTP (p), cd->beta, &ione, cd->beta_prev, &ione);

	for (j = 0; j < p; j++) {
		double		scale2 = get_jth_scale2 (j, cd);
		double		z = beta_j_updater (j, cd, jb);

		cd->beta[j] = scale2 * soft_threshold (z, lambda1);

		delta[j] = cd->beta[j] - cd->beta_prev[j];
		if (fabs (delta[j]) > 0.) {
			size_t			n = cd->lreg->n;
			const double	*x = cd->lreg->x;
			// mu += X(:, j) * (beta[j] - beta_prev[j])
			update_partially (j, delta[j], n, x, cd->mu);

			/* user defined penalty (not lasso nor ridge) */
			if (cdescent_is_regtype_userdef (cd)) {
				size_t			pj = cd->lreg->pen->pj;
				const double	*jr = cd->lreg->pen->r;
				// jb += J(:,j) * (beta[j] - beta_prev[j])
				update_partially (j, delta[j], pj, jr, jb);
			}
		}

	}
	if (jb) free (jb);

	cd->nrm1 = dasum_ (LINREG_CINTP (p), cd->beta, &ione);

	nrm = dnrm2_ (LINREG_CINTP (p), delta, &ione);
	free (delta);
	converged = (nrm < cd->tolerance);

	return converged;
}

/*
 * cyclic coordinate descent
 * repeat coordinate descent cycle until solution is converged
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
