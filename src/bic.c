/*
 * bic.c
 *
 *  Created on: 2014/07/15
 *      Author: utsugi
 */

#ifdef DEBUG
#include <stdio.h>
#endif

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cdescent.h>

#include "private.h"

/*
 *   Bayesian Information Criterion for L2 reguralized
 *   linear regression model b = Z * beta
 *   where b = [y ; 0], Z = [x ; sqrt(lambda2) * D]
 */

/* residual sum of squares
 * rss = | b - Z * beta |^2
 *     = | y - mu |^2 + | 0 - sqrt(lambda2) * nu |^2 */
static double
calc_rss (const cdescent *cd)
{
	int		m = cd->lreg->x->m;
	double	rss;
	double	*r = (double *) malloc (m * sizeof (double));
	dcopy_ (&m, cd->lreg->y->data, &ione, r, &ione);	// r = y
	daxpy_ (&m, &dmone, cd->mu->data, &ione, r, &ione);	// r = y - mu
	rss = pow (dnrm2_ (&m, r, &ione), 2.);	// rss = | y - mu |^2
	free (r);
	// rss += | 0 - sqrt(lambda2) * nu |^2
	if (!cd->lreg->is_regtype_lasso)
		rss += cd->lreg->lambda2 * pow (dnrm2_ (&cd->nu->m, cd->nu->data, &ione), 2.);
	return rss;
}

/* degree of freedom
 * it is equal to #{j ; j \in A i.e., beta_j != 0} (Efron et al., 2004) */
static double
calc_degree_of_freedom (const cdescent *cd)
{
	int		i;
	int		sizeA = 0;	// size of active set
	double	eps = DBL_EPSILON;
	for (i = 0; i < cd->beta->m; i++) if (fabs (cd->beta->data[i]) > eps) sizeA++;
	return (double) sizeA;
}

/* Extended Bayesian Information Criterion (Chen and Chen, 2008)
 * EBIC = m log(rss) + df * log(m) + 2 * gamma * df * log(n)
 * gamma	: tuning parameter for EBIC
 * rss		: residual sum of squares |b - Z * beta|^2
 * df		: degree of freedom of the system
 * m		: number of data (num rows of b and Z)
 * n		: number of variables (num cols of Z and num rows of beta)
 *
 * 	if gamma = 0, eBIC is identical with the classical BIC
*/
double
cdescent_eval_bic (const cdescent *cd, double gamma)
{
	double	rss = calc_rss (cd);
	double	df = calc_degree_of_freedom (cd);
	double	m = (double) cd->lreg->x->m;
	double	n = (double) cd->lreg->x->n;
	if (!cd->lreg->is_regtype_lasso) m += (double) cd->lreg->d->m;
#ifdef DEBUG
	fprintf (stdout, "rss %f df %f ", log (rss), df);
#endif
	return log (rss) + df * (log (m) + 2. * gamma * log (n)) / m;
}

