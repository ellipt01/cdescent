/*
 * mm_mtx.h
 *
 *  Created on: 2014/06/20
 *      Author: utsugi
 */

#ifndef MM_MTX_H_
#define MM_MTX_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <mmio.h>

typedef enum {
	MM_MTX_DENSE = 0,
	MM_MTX_SPARSE = 1
} MM_MtxType;

typedef enum {
	MM_MTX_UNSYMMETRIC = 0,
	MM_MTX_SYMMETRIC
} MM_MtxSymmetric;

// matrix market format matrix
typedef struct s_mm_mtx	mm_mtx;
typedef struct s_mm_mtx	mm_dense;
typedef struct s_mm_mtx	mm_sparse;

struct s_mm_mtx {
	MM_typecode	typecode;

	int				m;
	int				n;
	int				nz;

	int				*i;
	int				*j;
	int				*p;
	double			*data;
};

mm_mtx		*mm_mtx_alloc (void);
mm_mtx		*mm_mtx_new (MM_MtxType type, MM_MtxSymmetric symmetric, const int m, const int n, const int nz);
void		mm_mtx_free (mm_mtx *mm);
bool		mm_mtx_realloc (mm_mtx *mm, const int nz);
mm_mtx		*mm_mtx_copy (mm_mtx *mm);

void		mm_array_set_all (int n, double *data, const double val);
void		mm_mtx_set_all (mm_mtx *mm, const double val);

mm_dense	*mm_mtx_sparse_to_dense (mm_sparse *s);
mm_sparse	*mm_mtx_dense_to_sparse (mm_dense *d, const double threshold);
mm_mtx		*mm_mtx_eye (MM_MtxType type, const int n);

double		mm_mtx_sum (const mm_mtx *x);
double		mm_mtx_asum (const mm_mtx *x);
double		mm_mtx_xj_sum (const int j, const mm_mtx *x);
double		mm_mtx_nrm2 (const mm_mtx *x);
double		mm_mtx_xj_nrm2 (const int j, const mm_mtx *x);

mm_mtx		*mm_mtx_x_dot_y (bool trans, const double alpha, const mm_mtx *x, const mm_dense *y, const double beta);
double		mm_mtx_xj_trans_dot_y (const int j, const mm_mtx *x, const mm_dense *y);
void		mm_mtx_axjpy (const double alpha, const int j, const mm_mtx *x, mm_dense *y);

#ifdef __cplusplus
}
#endif

#endif /* MM_MTX_H_ */
