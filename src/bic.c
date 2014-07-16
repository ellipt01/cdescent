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
#include <cdescent.h>

#include "private.h"

/*
 *   Bayesian Information Criterion for L2 reguralized
 *   linear regression model b = Z * beta
 *   where b = [y ; 0], Z = [x ; sqrt(lambda2) * D]
 */

/* residual sum of squares | b - Z * beta |^2 */
static double
calc_rss (const cdescent *cd)
{
	int			m = cd->lreg->x->m;
	double		rss;
	double		*r = (double *) malloc (m * sizeof (double));
	dcopy_ (&m, cd->lreg->y->data, &ione, r, &ione);	// r = y
	daxpy_ (&m, &dmone, cd->mu->data, &ione, r, &ione);	// r = y - mu
	rss = pow (dnrm2_ (&m, r, &ione), 2.);
	if (!cd->lreg->is_regtype_lasso) {
		int			k = cd->lreg->d->m;
		mm_dense	*db = mm_real_x_dot_y (false, 1., cd->lreg->d, cd->beta, 0.);
		rss += cd->lreg->lambda2 * pow (dnrm2_ (&k, db->data, &ione), 2.);
		mm_real_free (db);
	}
	free (r);
	return rss;
}

/* degree of freedom
 * it is equal to #{j ; j \in A i.e., beta_j != 0} (Efron et al., 2004) */
static double
calc_degree_of_freedom (const cdescent *cd)
{
	int		i;
	int		size = 0;
	double	eps = double_eps ();
	for (i = 0; i < cd->beta->m; i++) if (fabs (cd->beta->data[i]) > eps) size++;
	return (double) size;
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
	if (!cd->lreg->is_regtype_lasso) m += n;
#ifdef DEBUG
	fprintf (stdout, "rss %f df %f ", log (rss), df);
#endif
	return log (rss) + df * (log (m) + 2. * gamma * log (n)) / m;
}

