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

	double			camax;		// mas ( abs(c) )
	double			*c;			// correlation: c = X' * y


	double			nrm1;		// L-1 norm: sum ( abs(beta) )
	double			b;			// intercept
	double			*beta;		// solution
	double			*mu;		// estimation of y

	/* backup of solution */
	double			nrm1_prev;
	double			*beta_prev;


	/* sum of y. If y is centered, sy = 0. */
	double			sy;

	/* sum of X(:, j). If X is centered, sx[j] = 0. */
	double			*sx;

	/* norm of X(:, j). If X is normalized, xtx[j] = 1. */
	double			*xtx;

	/*
	 * jtj = diag(J' * J), J = lreg->pen->r
	 * This value is need to update the solution
	 * when the penalty is user defined
	 * (not lasso nor ridge penalty).
	 */
	double			*jtj;

};

/* utils.c */
cdescent	*cdescent_alloc (const linreg *lreg, const double lambda1, const double tol);
void		cdescent_free (cdescent *cd);

void		cdescent_set_lambda1 (cdescent *cd, const double lambda1);
double		cdescent_get_lambda2 (const cdescent *cd, bool scaling);

double		*cdescent_copy_beta (const cdescent *cd, bool scaling);
//double		*cdescent_copy_mu (const cdescent *cd, bool scaling);

bool		cdescent_is_regtype_lasso (const cdescent *cd);
bool		cdescent_is_regtype_ridge (const cdescent *cd);
bool		cdescent_is_regtype_userdef (const cdescent *cd);

/* cdescent.c */
bool		cdescent_cyclic_once_cycle (cdescent *cd);
bool		cdescent_cyclic (cdescent *cd, const int maxiter);

#ifdef __cplusplus
}
#endif

#endif /* CDESCENT_H_ */
