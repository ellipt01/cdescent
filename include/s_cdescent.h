/*
 * s_cdescent.h
 *
 *  Created on: 2015/02/27
 *      Author: utsugi
 */

#ifndef S_CDESCENT_H_
#define S_CDESCENT_H_

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif /* S_CDESCENT_H_ */
