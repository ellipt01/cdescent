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

typedef struct s_linregmodel	linregmodel;

/* structure of linear regression model
 *   b = Z * beta,
 *   b = [y; 0]
 *   Z = scale * [X; sqrt(lambda2) * J]
 */
struct s_linregmodel {
	bool		has_copy;	// has copy of x, y and d

	mm_real	*x;
	mm_real	*y;
	/* penalty term. */
	mm_real	*d;

	/* threshold for L2 penalty */
	double		lambda2;

	bool		regtype_is_lasso;

	mm_real	*c;
	double		logcamax;		// log10 ( amax(c) )

	bool		ycentered;
	bool		xcentered;
	bool		xnormalized;

	/* sum of y. If y is centered, sy = 0. */
	double		sy;

	/* sum of X(:, j). If X is centered, sx = NULL. */
	double		*sx;

	/* norm of X(:, j). If X is normalized, xtx = NULL. */
	double		*xtx;

	/* dtd = diag(D' * D), D = lreg->pen->d */
	double		*dtd;

};

/* linregmodel.c */
linregmodel	*linregmodel_alloc (void);
linregmodel	*linregmodel_new (mm_real *y, mm_real *x, const double lambda2, mm_real *d,
		bool has_copy, bool do_ycentering, bool do_xcentering, bool do_xnormalizing);
void			linregmodel_free (linregmodel *l);

#ifdef __cplusplus
}
#endif

#endif /* LINREGMODEL_H_ */
