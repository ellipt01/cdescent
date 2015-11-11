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
typedef struct s_bic_func		bic_func;
typedef struct s_bic_info		bic_info;

typedef enum {
	CDESCENT_SELECTION_RULE_CYCLIC,
	CDESCENT_SELECTION_RULE_STOCHASTIC
} CoordinateSelectionRule;

/*** function which decides whether the constraint condition is satisfied ***/
typedef bool (*constraint_func) (const double betaj, double *forced);

/*** object of coordinate descent regression for L1 regularized linear problem
 *       argmin_beta || b - Z * beta ||^2 + sum_j lambda1 * | beta_j |
 *   or
 *       argmin_beta || b - Z * beta ||^2 + sum_j lambda1 * w_j * | beta_j | ***/
struct s_cdescent {

	bool					was_modified;			// whether this object was modified after created

	/* whether regression type is Lasso */
	bool					is_regtype_lasso;		// = (d == NULL)
	bool					use_intercept;			// whether use intercept (default is true)
	bool					use_fixed_lambda2;		// use fixed lambda2 value (default is false)

	const int				*m;						// number of observations, points cd->lreg->y->m
	const int				*n;						// number of variables, points cd->lreg->x->n
	const linregmodel		*lreg;					// linear regression model


	double					alpha1;					// raito of weight for L1 and L2 norm penalty
	double					alpha2;					// = 1. - alpha

	double					lambda;					// regularization parameter of L1 and L2 penalty
	double					lambda1;				// regularization parameter of L1 penalty
	double					lambda2;				// regularization parameter of L2 penalty

	mm_dense				*w;						// dense general: weight for L1 penalty (penalty factor)

	double					tolerance;				// tolerance of convergence

	double					nrm1;					// L1 norm of beta (= sum_j |beta_j|)

	double					b0;						// intercept

	mm_dense				*beta;					// estimated regression coefficients
	mm_dense				*mu;					// mu = X * beta, estimate of y
	mm_dense				*nu;					// nu = D * beta

	int						total_iter;				// total number of iterations
	int						maxiter;				// maximum number of iterations

	bool					parallel;				// whether enable parallel calculation

	pathwise				*path;					// pathwise CD optimization object

	constraint_func			*cfunc;					// constraint function

	CoordinateSelectionRule	rule;
};

/* flag of data preprocessing */
typedef enum {
	DO_NOTHING       = 0x0,		// do nothing
	DO_CENTERING_Y   = 1 << 0,	// do centering y
	DO_CENTERING_X   = 1 << 1,	// do centering x
	DO_NORMALIZING_X = 1 << 2,	// do normalizing x
	// do standardizing x
	DO_STANDARDIZING_X = DO_CENTERING_X | DO_NORMALIZING_X
} PreProc;

/*** object of regularized linear regression problem
 *
 *   argmin_beta || b - Z * beta ||^2
 *
 *   where
 *   	b = [y; 0]
 *   	Z = scale * [X; sqrt(lambda2) * D]
 *
 * this object stores vector y, matrix x, d and their properties
 ***/
struct s_linregmodel {

	mm_dense		*y;				// dense general: observed data vector y (must be dense)
	mm_real			*x;				// sparse/dense symmetric/general: matrix of predictors X
	const mm_real	*d;				// sparse/dense symmetric/general: linear operator of penalty D

	mm_dense		*c;				// = x' * y: correlation (constant) vector
	double			camax;			// max ( abs (c) )

	bool			ycentered;		// y is centered?
	bool			xcentered;		// x is centered?
	bool			xnormalized;	// x is normalized?

	/* sum y. If y is centered, sy = NULL */
	double			*sy;

	/* sum X(:, j). If X is centered, sx = NULL. */
	double			*sx;

	/* xtx = diag(X' * X). If X is normalized, xtx = NULL. */
	double			*xtx;

	/* dtd = diag(D' * D), D = lreg->pen->d */
	double			*dtd;

};

/*** object of pathwise CD optimization ***/
struct s_pathwise {

	bool		was_modified;		// whether this object was modified after initialization

	bool		output_fullpath;	// whether to outputs full solution path
	char		fn_path[BUFSIZ];	// file to output solution path
	bool		output_bic_info;	// whether to outputs BIC info
	char		fn_bic[BUFSIZ];		// file to output BIC info

	double		log10_lambda_upper;	// upper bound of lambda1 on log10 scale
	double		log10_lambda_lower;	// lower bound of lambda1 on log10 scale
	double		dlog10_lambda;		// increment of lambda1 on log10 scale

	bic_func	*bicfunc;			// BIC evaluation function

	double		min_bic_val;		// minimum BIC
	int			index_opt;			// index of optimal beta
	double		b0_opt;				// optimal intercept
	mm_dense	*beta_opt;			// optimal beta corresponding to min_bic_val
	double		lambda_opt;			// optimal lambda1
	double		nrm1_opt;			// | beta_opt |

	bool		verbos;

};

/*** object for function which evaluates BIC ***/

/*** bic_eval_func
 * the function bic_eval_func is called to evaluate BIC using bic_info object ***/
typedef double (*bic_eval_func) (const cdescent *cd, bic_info *info, void *data);

struct s_bic_func {
	bic_eval_func	function;
	void			*data;
};

/*** Bayesian Information Criterion
 *  BIC = log(rss) + df * log(m) / m ***/
struct s_bic_info
{
	double		m;			// number of data
	double		n;			// number of variables
	double		rss;		// residual sum of squares
	double		df;			// degree of freedom
	double		bic_val;	// value of BIC
};

#ifdef __cplusplus
}
#endif

#endif /* OBJECTS_H_ */
