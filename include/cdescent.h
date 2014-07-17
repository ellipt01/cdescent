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

typedef struct s_cdescent	cdescent;

#define cdescent_get_m (cd)	((cd)->lreg->x->m)
#define cdescent_get_n (cd)	((cd)->lreg->x->n)

struct s_cdescent {

	const linregmodel	*lreg;			// linear regression model

	double				tolerance;		// tolerance of convergence

	double				lambda1;		// regularization parameter of L1 penalty
	double				lambda1_max;	// maximum value of lambda1

	double				b;				// intercept
	double				nrm1;			// = sum_j |beta_j|
	mm_real			*beta;			// solution

	mm_real			*mu;			// mu = X * beta, estimate of y
	mm_real			*nu;			// nu = D * beta

	bool				parallel;		// enable parallel calculation
	int					total_iter;	// total number of iterations
};

/* cdescent.c */
cdescent	*cdescent_alloc (void);
cdescent	*cdescent_new (const linregmodel *lreg, const double tol, bool parallel);
void		cdescent_free (cdescent *cd);
void		cdescent_set_lambda1 (cdescent *cd, const double lambda1);
void		cdescent_set_log10_lambda1 (cdescent *cd, const double log10_lambda1);

/* update.c */
double		cdescent_beta_stepsize (const cdescent *cd, const int j);

/* cdescent.c */
bool		cdescent_update_cyclic_once_cycle (cdescent *cd);
bool		cdescent_update_cyclic (cdescent *cd, const int maxiter);

/* bic.c */
double		cdescent_eval_bic (const cdescent *cd, double gamma);

#ifdef __cplusplus
}
#endif

#endif /* CDESCENT_H_ */
