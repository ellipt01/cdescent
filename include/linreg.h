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
#include <mm_mtx.h>

#ifndef LINREG_INDEX_OF_MATRIX
#define LINREG_INDEX_OF_MATRIX(i, j, lda) ((i) + (j) * (lda))
#endif

typedef struct s_linreg		linreg;
typedef struct s_penalty		penalty;

/* structure of linear regression model
 *   b = Z * beta,
 *   b = [y; 0]
 *   Z = scale * [X; sqrt(lambda2) * J]
 */
struct s_linreg {

	mm_mtx				*x;
	mm_mtx				*y;

	/* threshold for L2 penalty */
	double				lambda2;

	/* penalty term. */
	const mm_mtx		*d;

	bool				ycentered;
	bool				xcentered;
	bool				xnormalized;

	double				*meany;	// mean(y)
	double				*meanx;	// meanx[j] = mean( X(:,j) )
	double				*normx;	// normx[j] = norm( X(:,j) )

};

/* linreg.c */
linreg			*linreg_alloc (mm_mtx *y, mm_mtx *x, const double lambda2, const mm_mtx *d);
void			linreg_free (linreg *l);

void			linreg_centering_y (linreg *lreg);
void			linreg_centering_x (linreg *lreg);
void			linreg_normalizing_x (linreg *lreg);
void			linreg_standardizing_x (linreg *lreg);

void			linreg_set_penalty (linreg *lreg, const double lambda2, const mm_mtx *d);

#ifdef __cplusplus
}
#endif

#endif /* LINREG_H_ */
