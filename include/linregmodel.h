/*
 * linregmodel.h
 *
 *  linear regression model
 *
 *  Created on: 2014/04/08
 *      Author: utsugi
 */

#ifndef LINREGMODEL_H_
#define LINREGMODEL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <mmreal.h>

/* flag of pre-processing for the data */
enum {
	DO_CENTERING_Y   = 1 << 0,	// do centering of y
	DO_CENTERING_X   = 1 << 1,	// do centering of x
	DO_NORMALIZING_X = 1 << 2,	// do normalizing of x
	// do standardizing of x
	DO_STANDARDIZING_X = DO_CENTERING_X | DO_NORMALIZING_X,
	DO_NOTHING       = 0x0,	// do nothing
};

typedef struct s_linregmodel	linregmodel;

/* Object of L1 regularized linear regression problem
 *
 *   argmin(beta) || b - Z * beta ||^2 + lambda_1 sum |beta|
 *
 *   where
 *   	b = [y; 0]
 *   	Z = scale * [X; sqrt(lambda2) * D]
 */
struct s_linregmodel {
	bool		has_copy;	// has copy of x, y and d

	mm_real	*x;		// matrix of predictors X
	mm_dense	*y;		// observed data vector y (must be dense)
	/* penalty term. */
	mm_real	*d;		// linear operator matrix D

	/* weight of penalty term */
	double		lambda2;

	bool		is_regtype_lasso;	// = (d == NULL || lambda2 < eps)

	mm_dense	*c;				// = x' * y: correlation (constant) vector
	double		logcamax;		// log10 ( amax(c) )

	bool		ycentered;		// y is centered?
	bool		xcentered;		// x is centered?
	bool		xnormalized;	// x is normalized?

	/* sum of y. If y is centered, sy = 0. */
	double		sy;		// = sum_i y(i)

	/* sum of X(:, j). If X is centered, sx = NULL. */
	double		*sx;	// sx(j) = sum_i x(i,j)

	/* norm of X(:, j). If X is normalized, xtx = NULL. */
	double		*xtx;	// xtx(j) = x(:,j)' * x(:,j)

	/* dtd = diag(D' * D), D = lreg->pen->d */
	double		*dtd;	// dtd(j) = d(:,j)' * d(:,j)

};

/* linregmodel.c */
linregmodel	*linregmodel_new (mm_dense *y, mm_real *x, const double lambda2, mm_real *d, bool has_copy, int preproc);
void			linregmodel_free (linregmodel *l);

#ifdef __cplusplus
}
#endif

#endif /* LINREGMODEL_H_ */
