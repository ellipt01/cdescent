/*
 * mm_real.h
 *
 *  Created on: 2014/06/20
 *      Author: utsugi
 */

#ifndef MM_REAL_H_
#define MM_REAL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <mmio.h>

typedef enum {
	MM_REAL_DENSE  = 0,
	MM_REAL_SPARSE = 1
} MMRealFormat;

typedef enum {
	MM_REAL_UNSYMMETRIC = 0,
	MM_REAL_SYMMETRIC   = 1
} MMRealSymmetry;

// matrix market format matrix
typedef struct s_mm_real	mm_real;
typedef struct s_mm_real	mm_dense;
typedef struct s_mm_real	mm_sparse;

struct s_mm_real {
	MM_typecode	typecode;

	int				m;
	int				n;
	int				nz;

	int				*i;
	int				*p;
	double			*data;
};

mm_real	*mm_real_alloc (void);
mm_real	*mm_real_new (MMRealFormat format, MMRealSymmetry symmetry, const int m, const int n, const int nz);
void		mm_real_free (mm_real *mm);
bool		mm_real_realloc (mm_real *mm, const int nz);
mm_real	*mm_real_copy (const mm_real *mm);

void		mm_real_array_set_all (int n, double *data, const double val);
void		mm_real_set_all (mm_real *mm, const double val);

void		mm_real_replace_sparse_to_dense (mm_real *x);
void		mm_real_replace_dense_to_sparse (mm_real *x, const double threshold);
mm_real	*mm_real_eye (MMRealFormat type, const int n);

double		mm_real_xj_asum (const int j, const mm_real *x);
double		mm_real_xj_sum (const int j, const mm_real *x);
double		mm_real_xj_nrm2 (const int j, const mm_real *x);

mm_real	*mm_real_x_dot_y (bool trans, const double alpha, const mm_real *x, const mm_dense *y, const double beta);
double		mm_real_xj_trans_dot_y (const int j, const mm_real *x, const mm_dense *y);
void		mm_real_axjpy (const double alpha, const int j, const mm_real *x, mm_dense *y);
void		mm_real_axjpy_atomic (const double alpha, const int j, const mm_real *x, mm_dense *y);

#ifdef __cplusplus
}
#endif

#endif /* MM_REAL_H_ */
