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

// matrix market format matrix
typedef struct s_mm_mtx mm_mtx;

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

mm_mtx		*mm_mtx_real_alloc (void);
mm_mtx		*mm_mtx_real_new (bool sparse, bool symmetric, const int m, const int n, const int nz);
void		mm_array_set_all (int n, double *data, const double val);
void		mm_mtx_real_set_all (mm_mtx *mm, const double val);
mm_mtx		*mm_mtx_real_eye (const int n);
mm_mtx		*mm_mtx_real_s_dot_d (bool trans, const double alpha, const mm_mtx *a, const mm_mtx *x, const double beta);
double		mm_mtx_real_sj_dot_d (const int j, const mm_mtx *s, const mm_mtx *d);
double		mm_mtx_real_sj_nrm (const int j, const mm_mtx *s);

mm_mtx		*mm_mtx_real_x_dot_y (bool trans, const double alpha, const mm_mtx *x, const mm_mtx *y, const double beta);
double		mm_mtx_real_xj_dot_y (const int j, const mm_mtx *x, const mm_mtx *y);
double		mm_mtx_real_xj_dot_xj (const int j, const mm_mtx *x);

#ifdef __cplusplus
}
#endif

#endif /* MM_MTX_H_ */
