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

/* dense / sparse */
typedef enum {
	MM_REAL_SPARSE = 0,	// sparse matrix
	MM_REAL_DENSE  = 1	// dense matrix
} MMRealFormat;

enum {
	MM_GENERAL   = 1 << 0,
	MM_SYMMETRIC = 1 << 1
};

enum {
	MM_UPPER = 1 << 2,
	MM_LOWER = 1 << 3
};

/* symmetric */
typedef enum {
	MM_REAL_GENERAL = MM_GENERAL,	// asymmetric
	MM_REAL_SYMMETRIC_UPPER = MM_SYMMETRIC | MM_UPPER,	// symmetric upper triangular
	MM_REAL_SYMMETRIC_LOWER = MM_SYMMETRIC | MM_LOWER	// symmetric lower triangular
} MMRealSymm;

#define mm_real_is_sparse(a)		mm_is_sparse((a)->typecode)
#define mm_real_is_dense(a)			mm_is_dense((a)->typecode)
#define mm_real_is_symmetric(a)		(mm_is_symmetric((a)->typecode) && ((a)->symm & MM_SYMMETRIC))
#define mm_real_is_upper(a) 		((a)->symm & MM_UPPER)
#define mm_real_is_lower(a) 		((a)->symm & MM_LOWER)

// MatrixMarket format matrix
typedef struct s_mm_real	mm_real;
typedef struct s_mm_real	mm_dense;
typedef struct s_mm_real	mm_sparse;

/*** implementation of dense / sparse matrix
 * In the case of dense matrix, nnz = m * n and
 * i = NULL, p = NULL. ***/
struct s_mm_real {
	MM_typecode	typecode;	// type of matrix. see mmio.h

	MMRealSymm	symm;

	int			m;			// num of rows of matrix
	int			n;			// num of columns
	int			nnz;		// num of nonzero matrix elements

	int			*i;			// row index of each nonzero elements: size = nnz
	int			*p;			// p[0] = 0, p[j+1] = num of nonzeros in X(:,1:j): size = n + 1
	double		*data;		// nonzero matrix elements: size = nnz
};

mm_real		*mm_real_new (MMRealFormat format, MMRealSymm symm, const int m, const int n, const int nnz);
void		mm_real_free (mm_real *mm);
bool		mm_real_realloc (mm_real *mm, const int nnz);

void		mm_real_set_sparse (mm_real *x);
void		mm_real_set_dense (mm_real *x);
void		mm_real_set_general (mm_real *x);
void		mm_real_set_symmetric (mm_real *x);
void		mm_real_set_upper (mm_real *x);
void		mm_real_set_lower (mm_real *x);

mm_real		*mm_real_copy (const mm_real *mm);
void		mm_real_set_all (mm_real *mm, const double val);

mm_dense	*mm_real_sparse_to_dense (const mm_sparse *s);
mm_sparse	*mm_real_dense_to_sparse (const mm_dense *x, const double threshold);
mm_real		*mm_real_symmetric_to_general (const mm_real *x);

mm_real		*mm_real_eye (MMRealFormat type, const int n);

mm_real		*mm_real_vertcat (const mm_real *x1, const mm_real *x2);
mm_real		*mm_real_holzcat (const mm_real *x1, const mm_real *x2);

void		mm_real_xj_add_const (mm_real *x, const int j, const double alpha);
void		mm_real_xj_scale (mm_real *x, const int j, const double alpha);

double		mm_real_xj_asum (const mm_real *x, const int j);
double		mm_real_xj_sum (const mm_real *x, const int j);
double		mm_real_xj_ssq (const mm_real *x, const int j);
double		mm_real_xj_nrm2 (const mm_real *x, const int j);

void		mm_real_x_dot_y (bool trans, const double alpha, const mm_real *x, const mm_dense *y, const double beta, mm_dense *z);
double		mm_real_xj_trans_dot_y (const mm_real *x, const int j, const mm_dense *y);
void		mm_real_axjpy (const double alpha, const mm_real *x, const int j, mm_dense *y);
void		mm_real_axjpy_atomic (const double alpha, const mm_real *x, const int j, mm_dense *y);

mm_real		*mm_real_fread (FILE *fp);
void		mm_real_fwrite (FILE *stream, const mm_real *x, const char *format);

#ifdef __cplusplus
}
#endif

#endif /* MMREAL_H_ */
