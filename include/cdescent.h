/*
 * cdescent.h
 *
 *  Created on: 2014/05/27
 *      Author: utsugi
 */

#ifndef CDESCENT_H_
#define CDESCENT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <linregmodel.h>
#include <bic.h>
#include "pathwiseopt.h"

/*** object of coordinate descent regression for L1 regularized linear regression problem
 *       argmin_beta || b - Z * beta ||^2 + sum_j lambda1 * | beta_j |
 *   or
 *       argmin_beta || b - Z * beta ||^2 + sum_j lambda1 * w_j * | beta_j | ***/
typedef struct s_cdescent	cdescent;

struct s_cdescent {

	bool				was_modified;	// whether this object was modified after created

	/* whether regression type is Lasso */
	bool				is_regtype_lasso;	// = (d == NULL || lambda2 == 0)

	const linregmodel	*lreg;			// linear regression model

	double				lambda1;		// regularization parameter of L1 penalty
	double				lambda1_max;	// maximum value of lambda1
	mm_dense			*w;				// dense general: weight for L1 penalty (penalty factor)

	double				tolerance;		// tolerance of convergence

	double				nrm1;			// L1 norm of beta (= sum_j |beta_j|)
	double				b;				// intercept
	mm_dense			*beta;			// estimated regression coefficients
	mm_dense			*mu;			// mu = X * beta, estimate of y
	mm_dense			*nu;			// nu = D * beta

	int					total_iter;	// total number of iterations
	int					maxiter;		// maximum number of iterations

	bool				parallel;		// whether enable parallel calculation
};

/* cdescent.c */
cdescent	*cdescent_new (const linregmodel *lreg, const double tol, const int maxiter, bool parallel);
void		cdescent_free (cdescent *cd);
bool		cdescent_set_penalty_factor (cdescent *cd, const mm_dense *w, const double tau);
bool		cdescent_set_lambda1 (cdescent *cd, const double lambda1);
bool		cdescent_set_log10_lambda1 (cdescent *cd, const double log10_lambda1);

/* stepsize.c */
double		cdescent_beta_stepsize (const cdescent *cd, const int j);

/* regression.c */
bool		cdescent_cyclic_update_once_cycle (cdescent *cd);
bool		cdescent_cyclic_update (cdescent *cd);
bool		cdescent_cyclic_pathwise (cdescent *cd, pathwiseopt *path);

/* bic.c */
bic_info	*cdescent_eval_bic (const cdescent *cd, double gamma);

#ifdef __cplusplus
}
#endif

#endif /* CDESCENT_H_ */
