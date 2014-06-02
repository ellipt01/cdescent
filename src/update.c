/*
 * update.c
 *
 *  Created on: 2014/06/02
 *      Author: utsugi
 */

#include <math.h>
#include <cdescent.h>

#include "linreg_private.h"

/*** return S(z + beta, gamma) - beta ***/
static double
cdescent_soft_threshold (const double z, const double gamma, const double beta)
{
	double	val = - beta;
	if (gamma < fabs (z + beta)) val = (z + beta > 0.) ? z - gamma : z + gamma;
	return val;
}

/*** return X(:,j)' * X(:,j) + D(:,j)' * D(:,j) * lambda2 ***/
static double
cdescent_scale2 (const cdescent *cd, const int j)
{
	double		scale2;
	double		xtxj = (cd->xtx) ? cd->xtx[j] : 1.;
	if (cdescent_is_regtype_lasso (cd)) scale2 = xtxj;
	else {
		double	lambda2 = cd->lreg->lambda2;
		double	dtdj = (cd->dtd) ? cd->dtd[j] : 1.;
		scale2 = xtxj + dtdj * lambda2;
	}
	return scale2;
}

/*** return gradient of objective function with respect to beta_j ***/
/* z = d L_j - beta_j
 *   = c(j) - X(:,j)' * mu - lambda2 * D(:,j)' * D * beta */
static double
cdescent_gradient (const cdescent *cd, const int j)
{
	int				n = cd->lreg->n;
	double			cj = cd->c[j];	// X' * y
	const double	*x = cd->lreg->x;
	const double	*xj = x + LINREG_INDEX_OF_MATRIX (0, j, n);	// X(:,j)
	double			xjtmu = ddot_ (&n, xj, &ione, cd->mu, &ione);	// X(:,j)' * mu

	double			lambda2 = cd->lreg->lambda2;

	//	z = c(j) - X(:,j)' * mu
	double			z = cj - xjtmu;

	// if X is not centered, z -= sum(X(:,j)) * b
	if (!cd->lreg->xcentered) z -= cd->sx[j] * cd->b;

	if (cdescent_is_regtype_ridge (cd)) {	// ridge
		// z -= lambda2 * E(:,j)' * E * beta
		z -= lambda2 * cd->beta[j];
	} else if (cdescent_is_regtype_userdef (cd)) {	// not lasso nor ridge
		// z -= lambda2 * D(:,j)' * D * beta
		int				pj = cd->lreg->pen->pj;
		const double	*d = cd->lreg->pen->d;
		const double	*dj = d + LINREG_INDEX_OF_MATRIX (0, j, pj);
		z -= lambda2 * ddot_ (&pj, dj, &ione, cd->nu, &ione);
	}
	return z;
}

/*** updater of intercept: ( sum (y) - sum(X) * beta ) / n ***/
double
cdescent_update_intercept (const cdescent *cd)
{
	int			n = cd->lreg->n;
	int			p = cd->lreg->p;
	double		nb = 0.;	// n * b

	if (cd->lreg->ycentered && cd->lreg->xcentered) return 0.;
	// b = bar(y)
	if (!cd->lreg->ycentered) nb += cd->sy;
	// b -= bar(X) * beta
	if (!cd->lreg->xcentered) nb -= ddot_ (&p, cd->sx, &ione, cd->beta, &ione);
	return nb / (double) n;	// return b
}

/*** return step-size for updating beta ***/
double
cdescent_beta_stepsize (const cdescent *cd, const int j)
{
	double		scale2 = cdescent_scale2 (cd, j);
	double		z = cdescent_gradient (cd, j) / scale2;
	double		gamma = cd->lambda1 / scale2;
	/* eta = S(z / scale2 + beta, lambda1 / scale2) - beta */
	return cdescent_soft_threshold (z, gamma, cd->beta[j]);
}
