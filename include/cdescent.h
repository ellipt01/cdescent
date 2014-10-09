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

typedef struct s_cdescent	cdescent;

struct s_cdescent {

	const linregmodel	*lreg;			// linear regression model

	double				tolerance;		// tolerance of convergence

	double				lambda1;		// regularization parameter of L1 penalty
	double				lambda1_max;	// maximum value of lambda1

	double				b;				// intercept
	double				nrm1;			// = sum_j |beta_j|
	mm_real			*beta;			// dense : solution

	mm_real			*mu;			// dense : mu = X * beta, estimate of y
	mm_real			*nu;			// dense : nu = D * beta

	bool				parallel;		// enable parallel calculation
	int					total_iter;	// total number of iterations
};

/* cdescent.c */
cdescent	*cdescent_new (const linregmodel *lreg, const double tol, bool parallel);
void		cdescent_free (cdescent *cd);
bool		cdescent_set_lambda1 (cdescent *cd, const double lambda1);
bool		cdescent_set_log10_lambda1 (cdescent *cd, const double log10_lambda1);

/* update.c */
double		cdescent_beta_stepsize (const cdescent *cd, const int j);

/* cdescent.c */
bool		cdescent_update_cyclic_once_cycle (cdescent *cd);
bool		cdescent_update_cyclic (cdescent *cd, const int maxiter);

/* bic.c */
bic_info	*bic_info_new (void);
void		bic_info_free (bic_info *info);
bic_info	*cdescent_eval_bic (const cdescent *cd, double gamma);

#ifdef __cplusplus
}
#endif

#endif /* CDESCENT_H_ */
