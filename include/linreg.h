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
	int					n;	// number of data
	int					p;	// number of variables

	mm_mtx				*x;
	mm_mtx				*y;

//	double				*y1;		// data
	double				*x1;		// variables

	bool				ycentered;
	bool				xcentered;
	bool				xnormalized;

	double				*meany;	// mean(y)
	double				*meanx;	// meanx[j] = mean( X(:,j) )
	double				*normx;	// normx[j] = norm( X(:,j) )

	/* threshold for L2 penalty */
	double				lambda2;

	/* penalty term.
	 * if pen == NULL && lambda2 > 0, ridge regression is assumed. */
	const mm_mtx		*d;
	const penalty		*pen;

};

/* penalty term */
struct s_penalty {
	int					pj;		// rows of D
	int					p;		// columns of D
	const double		*d;		// pj x p penalty matrix D
};

/* linreg.c */
linreg			*linreg_alloc (const int n, const int p, double *y, double *x);
void			linreg_free (linreg *l);

void			linreg_centering_y (linreg *lreg);
void			linreg_centering_x (linreg *lreg);
void			linreg_normalizing_x (linreg *lreg);
void			linreg_standardizing_x (linreg *lreg);

penalty		*penalty_alloc (const int p1, const int p, const double *d);
void			penalty_free (penalty *pen);

void			linreg_set_penalty (linreg *lreg, const double lambda2, const penalty *pen);

#ifdef __cplusplus
}
#endif

#endif /* LINREG_H_ */
