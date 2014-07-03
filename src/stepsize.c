/*
 * update.c
 *
 *  Created on: 2014/06/02
 *      Author: utsugi
 */

#include <math.h>
#include <cdescent.h>

#include "private.h"

/*** soft thresholding ***/
/* S(z, gamma) = sign(z)(|z| - gamma)+
 *             = 0, -gamma <= z <= gamma,
 *               z - gamma, z >  gamma (> 0)
 *               z + gamma, z < -gamma (< 0) */
static double
soft_threshold (const double z, const double gamma)
{
	double	val = 0.;
	if (gamma < fabs (z)) val = (z > 0.) ? z - gamma : z + gamma;
	return val;
}

/*** return X(:,j)' * X(:,j) + D(:,j)' * D(:,j) * lambda2 ***/
static double
cdescent_scale2 (const cdescent *cd, const int j)
{
	double	scale2 = (cd->lreg->xnormalized) ? 1. : cd->lreg->xtx[j];
	if (!linregmodel_is_regtype_lasso (cd->lreg)) scale2 += cd->lreg->dtd[j] * cd->lreg->lambda2;
	return scale2;
}

/*** return gradient of objective function with respect to beta_j ***/
/* z = d L_j
 *   = c(j) - X(:,j)' * mu - X(:,j)' * b - lambda2 * D(:,j)' * D * beta
 *     + scale2 * beta_j,
 * however, the last term scale2 * beta_j is omitted */
static double
cdescent_gradient (const cdescent *cd, const int j)
{
	double	cj = cd->lreg->c->data[j];	// X' * y
	double	xjmu = mm_real_xj_trans_dot_y (j, cd->lreg->x, cd->mu);	// X(:,j)' * mu
	double	lambda2 = cd->lreg->lambda2;

	//	z = c(j) - X(:,j)' * mu
	double	z = cj - xjmu;

	// if X is not centered, z -= sum(X(:,j)) * b
	if (!cd->lreg->xcentered) z -= cd->lreg->sx[j] * cd->b;

	// not lasso, z -= lambda2 * D(:,j)' * nu (nu = D * beta)
	if (!linregmodel_is_regtype_lasso (cd->lreg)) z -= lambda2 * mm_real_xj_trans_dot_y (j, cd->lreg->d, cd->nu);

	return z;
}

/*** return step-size for updating beta ***/
double
cdescent_beta_stepsize (const cdescent *cd, const int j)
{
	double	scale2 = cdescent_scale2 (cd, j);
	double	z = cdescent_gradient (cd, j) / scale2;
	double	gamma = cd->lambda1 / scale2;
	/* eta = S(z / scale2 + beta, lambda1 / scale2) - beta */
	return soft_threshold (z + cd->beta->data[j], gamma) - cd->beta->data[j];
}