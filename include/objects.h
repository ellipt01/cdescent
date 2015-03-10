/*
 * s_cdescent.h
 *
 *  Created on: 2015/02/27
 *      Author: utsugi
 */

#ifndef OBJECTS_H_
#define OBJECTS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct s_cdescent	cdescent;
typedef struct s_pathwise	pathwise;

/*** object of coordinate descent regression for L1 regularized linear regression problem
 *       argmin_beta || b - Z * beta ||^2 + sum_j lambda1 * | beta_j |
 *   or
 *       argmin_beta || b - Z * beta ||^2 + sum_j lambda1 * w_j * | beta_j | ***/
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

	int					total_iter;		// total number of iterations
	int					maxiter;		// maximum number of iterations

	bool				parallel;		// whether enable parallel calculation

	pathwise			*path;			// pathwise CD optimization object

};


/*** reweighting function
 * the function weighting_func is called to calculate weight
 * for each iteration of pathwise optimization ***/
typedef struct s_reweighting_func reweighting_func;

/* weighting function */
typedef mm_dense* (*weight_func) (int i, cdescent *cd, void *data);


struct s_reweighting_func {
	double		tau;
	weight_func	function;
	void		*data;
};

/*** object of pathwise CD optimization ***/
struct s_pathwise {

	bool		was_modified;		// whether this object was modified after initialization

	char		fn_path[BUFSIZ];	// file to output solution path
	bool		output_fullpath;	// whether to outputs full solution path
	char		fn_bic[BUFSIZ];		// file to output BIC info
	bool		output_bic_info;	// whether to outputs BIC info
	double		gamma_bic;			// gamma for eBIC

	double		log10_lambda1_upper;// upper bound of lambda1 on log10 scale
	double		log10_lambda1_lower;// lower bound of lambda1 on log10 scale
	double		dlog10_lambda1;		// increment of lambda1 on log10 scale

	double		min_bic_val;		// minimum BIC
	mm_dense	*beta_opt;			// optimal beta corresponding to min_bic_val
	double		lambda1_opt;		// optimal lambda1
	double		nrm1_opt;			// | beta_opt |

	reweighting_func	*func;		// reweighting function

};

#ifdef __cplusplus
}
#endif

#endif /* OBJECTS_H_ */
