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

typedef struct s_cdescent		cdescent;
typedef struct s_linregmodel	linregmodel;
typedef struct s_pathwise		pathwise;
typedef struct s_bic_info		bic_info;

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

	bool				update_intercept;
	double				b;				// intercept
	mm_dense			*beta;			// estimated regression coefficients
	mm_dense			*mu;			// mu = X * beta, estimate of y
	mm_dense			*nu;			// nu = D * beta

	int					total_iter;		// total number of iterations
	int					maxiter;		// maximum number of iterations

	bool				parallel;		// whether enable parallel calculation

	pathwise			*path;			// pathwise CD optimization object

};

/*** Object of convex/nonconvex regularized linear regression problem
 *
 *   argmin_beta || b - Z * beta ||^2
 *
 *   where
 *   	b = [y; 0]
 *   	Z = scale * [X; sqrt(lambda2) * D] ***/

/* flag of data preprocessing */
typedef enum {
	DO_NOTHING       = 0x0,		// do nothing
	DO_CENTERING_Y   = 1 << 0,	// do centering y
	DO_CENTERING_X   = 1 << 1,	// do centering x
	DO_NORMALIZING_X = 1 << 2,	// do normalizing x
	// do standardizing x
	DO_STANDARDIZING_X = DO_CENTERING_X | DO_NORMALIZING_X
} PreProc;

struct s_linregmodel {
	mm_dense	*y;	// dense general: observed data vector y (must be dense)
	mm_real	*x;		// sparse/dense symmetric/general: matrix of predictors X
	mm_real	*d;		// sparse/dense symmetric/general: linear operator of penalty D

	double		lambda2;	// weight for penalty term

	mm_dense	*c;				// = x' * y: correlation (constant) vector
	double		log10camax;	// log10 ( amax(c) )

	bool		ycentered;		// y is centered?
	bool		xcentered;		// x is centered?
	bool		xnormalized;	// x is normalized?

	/* sum y. If y is centered, sy = NULL */
	double		*sy;

	/* sum X(:, j). If X is centered, sx = NULL. */
	double		*sx;

	/* xtx = diag(X' * X). If X is normalized, xtx = NULL. */
	double		*xtx;

	/* dtd = diag(D' * D), D = lreg->pen->d */
	double		*dtd;

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

	bool		output_fullpath;	// whether to outputs full solution path
	char		fn_path[BUFSIZ];	// file to output solution path
	bool		output_bic_info;	// whether to outputs BIC info
	char		fn_bic[BUFSIZ];		// file to output BIC info
	double		gamma_bic;			// gamma for eBIC

	double		log10_lambda1_upper;// upper bound of lambda1 on log10 scale
	double		log10_lambda1_lower;// lower bound of lambda1 on log10 scale
	double		dlog10_lambda1;		// increment of lambda1 on log10 scale

	double		min_bic_val;		// minimum BIC
	int			index_opt;			// index of optimal beta
	double		b_opt;				// optimal intercept
	mm_dense	*beta_opt;			// optimal beta corresponding to min_bic_val
	double		lambda1_opt;		// optimal lambda1
	double		nrm1_opt;			// | beta_opt |

	reweighting_func	*func;		// reweighting function

};

/*** Extended Bayesian Information Criterion (Chen and Chen, 2008)
 * eBIC = log(rss) + df * (log(m) + 2 * gamma * log(n)) / m
 * 	if gamma = 0, eBIC is identical with the classical BIC ***/
struct s_bic_info
{
	double		m;			// number of data
	double		n;			// number of variables
	double		rss;		// residual sum of squares
	double		df;			// degree of freedom
	double		gamma;		// tuning parameter for eBIC (0 <= gamma <= 1)
	double		bic_val;	// value of eBIC
};

#ifdef __cplusplus
}
#endif

#endif /* OBJECTS_H_ */
