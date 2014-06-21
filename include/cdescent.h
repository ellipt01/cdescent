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

#include <linreg.h>

typedef struct s_cdescent	cdescent;

struct s_cdescent {

	const linreg	*lreg;		// linear regression equations

	double			lambda1;	// L-1 regularization parameter

	double			tolerance;	// tolerance of convergence

	mm_mtx			*c;

	double			b;			// intercept
	double			nrm1;
	double			*beta;		// solution

	double			*mu;		// mu = X * beta, estimate of y
	double			*nu;		// nu = D * beta.

	/* sum of y. If y is centered, sy = 0. */
	double			sy;

	/* sum of X(:, j). If X is centered, sx = NULL. */
	double			*sx;

	/* norm of X(:, j). If X is normalized, xtx = NULL. */
	double			*xtx;

	/* dtd = diag(D' * D), D = lreg->pen->d */
	double			*dtd;

};

/* utils.c */
cdescent	*cdescent_alloc (const linreg *lreg, const double lambda1, const double tol);
void		cdescent_free (cdescent *cd);

bool		cdescent_is_regtype_lasso (const cdescent *cd);

/* update.c */
double		cdescent_update_intercept (const cdescent *cd);
double		cdescent_beta_stepsize (const cdescent *cd, const int j);

/* cdescent.c */
bool		cdescent_cyclic_once_cycle (cdescent *cd);
bool		cdescent_cyclic (cdescent *cd, const int maxiter);

#ifdef __cplusplus
}
#endif

#endif /* CDESCENT_H_ */
