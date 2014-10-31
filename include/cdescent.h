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

/*** coordinate descent object ***/
typedef struct s_cdescent	cdescent;

struct s_cdescent {

	const linregmodel	*lreg;			// linear regression model

	double				tolerance;		// tolerance of convergence

	double				lambda1;		// regularization parameter of L1 penalty
	double				lambda1_max;	// maximum value of lambda1

	double				nrm1;			// L1 norm of beta (= sum_j |beta_j|)
	double				b;				// intercept
	mm_dense			*beta;			// estimated regression coefficients
	mm_dense			*mu;			// mu = X * beta, estimate of y
	mm_dense			*nu;			// nu = D * beta

	double				lambda1_opt;	// optimal lambda1
	double				nrm1_opt;		// L1 norm of optimal beta
	mm_dense			*beta_opt;		// optimal beta

	int					total_iter;	// total number of iterations
	int					maxiter;		// maximum number of iterations

	bool				parallel;		// whether enable parallel calculation
};

/* cdescent.c */
cdescent	*cdescent_new (const linregmodel *lreg, const double tol, const int maxiter, bool parallel);
void		cdescent_free (cdescent *cd);
bool		cdescent_set_lambda1 (cdescent *cd, const double lambda1);
bool		cdescent_set_log10_lambda1 (cdescent *cd, const double log10_lambda1);

/* stepsize.c */
double		cdescent_beta_stepsize (const cdescent *cd, const int j);

/* update.c */
bool		cdescent_cyclic_update_once_cycle (cdescent *cd);
bool		cdescent_cyclic_update (cdescent *cd);

/* pathwise.c */
void		cdescent_cyclic_pathwise_set_conditions (bool is_output_solutionpath, bool is_output_bic_info, const double gamma_bic_val);
void		cdescent_cyclic_pathwise (cdescent *cd, const double log10_lambda1_lower, const double dlog10_lambda1);

/* bic.c */
bic_info	*bic_info_new (void);
void		bic_info_free (bic_info *info);
bic_info	*cdescent_eval_bic (const cdescent *cd, double gamma);

#ifdef __cplusplus
}
#endif

#endif /* CDESCENT_H_ */
