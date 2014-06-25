/*
 * linreg.h
 *
 *  Wrapper of lapack and qrupdate
 *
 *  Created on: 2014/04/08
 *      Author: utsugi
 */

#ifndef LINREG_H_
#define LINREG_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <mmreal.h>

typedef struct s_linreg	linreg;

/* structure of linear regression model
 *   b = Z * beta,
 *   b = [y; 0]
 *   Z = scale * [X; sqrt(lambda2) * J]
 */
struct s_linreg {
	bool		has_copy;	// has copy of x, y and d

	mm_real	*x;
	mm_real	*y;

	/* threshold for L2 penalty */
	double		lambda2;

	/* penalty term. */
	mm_real	*d;

	bool		ycentered;
	bool		xcentered;
	bool		xnormalized;

	double		*meany;	// mean(y)
	double		*meanx;	// meanx[j] = mean( X(:,j) )
	double		*normx;	// normx[j] = norm( X(:,j) )

};

/* linreg.c */
linreg		*linreg_alloc (void);
linreg		*linreg_new (mm_real *y, mm_real *x, const double lambda2, mm_real *d, bool has_copy);
void		linreg_free (linreg *l);

void		linreg_centering_y (linreg *lreg);
void		linreg_centering_x (linreg *lreg);
void		linreg_normalizing_x (linreg *lreg);
void		linreg_standardizing_x (linreg *lreg);

#ifdef __cplusplus
}
#endif

#endif /* LINREG_H_ */
