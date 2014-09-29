/*
 * mm_real.h
 *
 *  Created on: 2014/06/20
 *      Author: utsugi
 */

#ifndef MMREAL_H_
#define MMREAL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <mmio.h>

#define mm_real_is_sparse(a)	mm_is_sparse((a)->typecode)
#define mm_real_is_dense(a)		mm_is_dense((a)->typecode)
#define mm_real_is_symmetric(a)	(mm_is_symmetric((a)->typecode) || mm_is_skew((a)->typecode))

typedef enum {
	MM_REAL_DENSE  = 0,	// dense matrix
	MM_REAL_SPARSE = 1	// sparse matrix
} MMRealFormat;

// matrix market format matrix
typedef struct s_mm_real	mm_real;
typedef struct s_mm_real	mm_dense;
typedef struct s_mm_real	mm_sparse;

// implementation of dense / sparse matrix
struct s_mm_real {
	MM_typecode	typecode;	// type of matrix. see mmio.h

	int				m;		// num of rows of matrix
	int				n;		// num of columns
	int				nz;		// num of nonzero matrix elements

	int				*i;		// row index of each nonzero elements: size = nz
	int				*p;		// p[0] = 0, p[j+1] = num of nonzeros in X(:,1:j): size = n + 1
	double			*data;	// nonzero matrix elements: size = nz
};

mm_real	*mm_real_new (MMRealFormat format, bool symmetric, const int m, const int n, const int nz);
void		mm_real_free (mm_real *mm);
bool		mm_real_realloc (mm_real *mm, const int nz);

mm_real	*mm_real_copy (const mm_real *mm);

void		mm_real_array_set_all (int n, double *data, const double val);
void		mm_real_set_all (mm_real *mm, const double val);

bool		mm_real_replace_sparse_to_dense (mm_real *x);
bool		mm_real_replace_dense_to_sparse (mm_real *x, const double threshold);

mm_real	*mm_real_eye (MMRealFormat type, const int n);

double		mm_real_xj_asum (const int j, const mm_real *x);
double		mm_real_xj_sum (const int j, const mm_real *x);
double		mm_real_xj_nrm2 (const int j, const mm_real *x);

mm_dense	*mm_real_x_dot_y (bool trans, const double alpha, const mm_real *x, const mm_dense *y, const double beta);
double		mm_real_xj_trans_dot_y (const int j, const mm_real *x, const mm_dense *y);
void		mm_real_axjpy (const double alpha, const int j, const mm_real *x, mm_dense *y);
void		mm_real_axjpy_atomic (const double alpha, const int j, const mm_real *x, mm_dense *y);

mm_real	*mm_real_fread (FILE *fp);
void		mm_real_fwrite (FILE *stream, mm_real *x, const char *format);

#ifdef __cplusplus
}
#endif

#endif /* MMREAL_H_ */
