/*
 * linreg_private.h
 *
 *  Created on: 2014/05/14
 *      Author: utsugi
 */

#ifndef LINREG_PRIVATE_H_
#define LINREG_PRIVATE_H_

/* Private macros, constants and headers
 * which are only used internally */

/* blas */
#ifdef HAVE_BLAS_H
#include <blas.h>
#else
// Level1
extern double	dasum_  (const int *n, const double *x, const int *incx);
extern void	daxpy_  (const int *n, const double *alpha, const double *x, const int *incx, double *y, const int *incy);
extern void	dcopy_  (const int *n, const double *x, const int *incx, double *y, const int *incy);
extern double	ddot_   (const int *n, const double *x, const int *incx, const double *y, const int *incy);
extern double	dnrm2_  (const int *n, const double *x, const int *incx);
extern void	dscal_  (const int *n, const double *alpha, double *x, const int *incx);
extern int		idamax_ (const int *n, const double *x, const int *incx);
// Level2
extern void	dgemv_ (const char *trans, const int *m, const int *n, const double *alpha, const double *a, const int *lda,
		const double *x, const int *incx, const double *beta, double *y, const int *incy);
// Level3
extern void	dgemm_ (const char *transa, const char *transb, const int *m, const int *n, const int *k,
		const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
		const double *beta, double *c, const int *ldc);
#endif

#ifdef HAVE_LAPACK_H
#include <lapack.h>
#else
extern double	dlamch_ (const char *cmach);
#endif

/* cast pointer type to const int * */
#ifndef LINREG_CINTP
#define LINREG_CINTP(a)	((const int *) &(a))
#endif

/* min(a, b) */
#ifndef LINREG_MIN
#define LINREG_MIN(a, b)	(((a) <= (b)) ? (a) : (b))
#endif

/* following constants are set in linreg.c */
extern const int		ione;	//  1
extern const double	dzero;	//  0.
extern const double	done;	//  1.
extern const double	dmone;	// -1.

/* machine double epsilon */
double			linreg_double_eps (void);

/* print error message and terminate program */
void			linreg_error (const char * function_name, const char *error_msg, const char *file, const int line);

#endif /* LINREG_PRIVATE_H_ */
