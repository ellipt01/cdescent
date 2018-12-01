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

typedef enum {
	CDESCENT_SELECTION_RULE_CYCLIC,		// use cyclic coordinate descent update
	CDESCENT_SELECTION_RULE_STOCHASTIC	// use stochastic coordinate descent update
} CoordinateSelectionRule;

/*** constraint_fun
 * pointer of the function which evaluates whether the constraint condition for beta is satisfied ***/
typedef bool (*constraint_func) (cdescent *cd, const int j, const double etaj, double *forced);
/* For example, the following function realizes non-negativity constraint for the solution double *cd->beta:
 *
 * bool
 * constraint_func0 (cdescent *cd, const int j, const double etaj, double *forced)
 * {
 *    double new_betaj = cd->beta[j] + etaj;
 *    if (new_betaj < 0.) {
 *       *forced = 0.;
 *       return false;
 *    }
 *    return true;
 * }
 *
 * If the next value of beta[j]; new_betaj = cd->beta[j] + etaj < 0. (non-negativity is violated),
 * this function returns false and double *forced is set to the forced value of beta[j] (i.e. in this
 * case, *forced = 0).
 *
 * The pointer of the above function is connected to cdescent *cd by calling
 *
 *     cdescent_set_constraint (cd, constraint_func0);
 *     (this function set to cd->cfunc = constraint_func0)
 *
 * This function is called by the function void update_betaj in update.c, which updates cd->beta[j]
 * on each coordinate descent updates.
 * If cd->cfunc (= constraint_func0) returns false, cd->beta[j] and etaj is replaced by the followings:
 *
 *     etaj -> - cd->beta[j] + *forced (so as to be the next beta[j] = cd->beta[j] + etaj = *forced)
 *     cd->beta[j] -> *forced
 */

/*** object of coordinate descent regression for L1 regularized linear problem
 *       argmin_beta || y - x * beta ||^2 + lambda2 * || d * beta ||^2 + sum_j lambda1 * | beta_j |
 *   or
 *       argmin_beta || y - x * beta ||^2 + lambda2 * || d * beta ||^2  + sum_j lambda1 * w_j * | beta_j |
 *   where vector y, matrix x and d are specified by linregmodel *lreg,
 *   e.g. y = lreg->y, x = lreg->x and d = lreg->d ***/
struct s_cdescent {

	bool					was_modified;			// whether this object was modified after created

	/* whether regression type is Lasso */
	bool					is_regtype_lasso;		// = (d == NULL)
	bool					use_penalty_factor;		// whether use penalty factor
	bool					use_intercept;			// whether use intercept (default is true)
	bool					use_fixed_lambda;		// use fixed lambda value (default is false)
	CoordinateSelectionRule	rule;					// decide cyclic or stochastic coordinate descent

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

	constraint_func			cfunc;					// constraint function

	bool					output_fullpath;		// whether to outputs full solution path
	char					fn_path[128];			// file to output solution path
	bool					output_info;			// whether to outputs regression info
	char					fn_info[128];			// file to output info

	double					log10_lambda_upper;		// upper bound of lambda1 on log10 scale
	double					log10_lambda_lower;		// lower bound of lambda1 on log10 scale
	double					log10_dlambda;			// increment of lambda1 on log10 scale

	bool					verbos;					// if this is set to true, detailed progress of calculation will be reported

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
 *   argmin_beta || b - z * beta ||^2
 *
 *   where
 *   	b = [y; 0]
 *   	z = scale * [x; sqrt(lambda2) * d]
 *
 * this object stores vector y, matrix x, d and their properties
 ***/
struct s_linregmodel {

	bool			ycentered;		// y is centered?
	bool			xcentered;		// x is centered?
	bool			xnormalized;	// x is normalized?

	mm_dense		*y;				// dense general: observed data vector y (must be dense)
	mm_real			*x;				// sparse/dense symmetric/general: matrix of predictors x
	const mm_real	*d;				// sparse/dense symmetric/general: linear operator of penalty d

	mm_dense		*c;				// = x' * y: correlation (constant) vector
	double			camax;			// max ( abs (c) )

	/* sum y. If y is centered, sy = NULL */
	double			*sy;

	/* sum X(:, j). If x is centered, sx = NULL. */
	double			*sx;

	/* xtx = diag(x' * x). If x is normalized, xtx = NULL. */
	double			*xtx;

	/* dtd = diag(d' * d) */
	double			*dtd;

};

#ifdef __cplusplus
}
#endif

#endif /* OBJECTS_H_ */
