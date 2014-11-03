/*
 * mmreal.c
 *
 *  Created on: 2014/06/25
 *      Author: utsugi
 */

#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include <mmreal.h>

#include "private/atomic.h"
#include "private/private.h"

/* mm_real supports real symmetric or general sparse, and real general dense matrix */
static bool
is_type_supported (MM_typecode typecode)
{
	// invalid type
	if (!mm_is_valid (typecode)) return false;

	// pattern is not supported
	if (mm_is_pattern (typecode)) return false;

	// integer and complex matrix are not supported
	if (mm_is_integer (typecode) || mm_is_complex (typecode)) return false;

	// skew and hermitian are not supported
	if (mm_is_skew (typecode) || mm_is_hermitian (typecode)) return false;

	return true;
}

/* check format */
static bool
is_format_valid (MMRealFormat format) {
	return (format == MM_REAL_SPARSE || format == MM_REAL_DENSE);
}

/* check symmetric */
static bool
is_symm_valid (MMRealSymm symm)
{
	return (symm == MM_REAL_GENERAL || symm == MM_REAL_SYMMETRIC_UPPER
			|| symm == MM_REAL_SYMMETRIC_LOWER);
}

/* allocate mm_real */
static mm_real *
mm_real_alloc (void)
{
	mm_real	*mm = (mm_real *) malloc (sizeof (mm_real));
	mm->m = 0;
	mm->n = 0;
	mm->nz = 0;
	mm->i = NULL;
	mm->p = NULL;
	mm->data = NULL;

	mm->symm = MM_REAL_GENERAL;

	/* set typecode = "M_RG" : Matrix Real General */
	// typecode[3] = 'G' : General
	mm_initialize_typecode (&mm->typecode);
	// typecode[0] = 'M' : Matrix
	mm_set_matrix (&mm->typecode);
	// typecode[2] = 'R' : Real
	mm_set_real (&mm->typecode);

	return mm;
}

/*** create new mm_real object
 * MMRealFormat	format: MM_REAL_DENSE or MM_REAL_SPARSE
 * MMRealSymm		symm  : MM_REAL_GENERAL, MM_REAL_SYMMETRIC_UPPER or MM_REAL_SYMMETRIC_LOWER
 * int				m, n  : rows and columns of the matrix
 * int				nz    : number of nonzero elements of the matrix ***/
mm_real *
mm_real_new (MMRealFormat format, MMRealSymm symm, const int m, const int n, const int nz)
{
	mm_real	*mm;
	bool		symmetric;

	if (!is_format_valid (format))
		error_and_exit ("mm_real_new", "invalid format", __FILE__, __LINE__);
	if (!is_symm_valid (symm))
		error_and_exit ("mm_real_new", "invalid symm", __FILE__, __LINE__);

	symmetric = symm & MM_SYMMETRIC;
	if (symmetric && m != n)
		error_and_exit ("mm_real_new", "symmetric matrix must be square", __FILE__, __LINE__);

	mm = mm_real_alloc ();
	mm->m = m;
	mm->n = n;
	mm->nz = nz;

	// typecode[1] = 'C' or 'A'
	if (format == MM_REAL_SPARSE) mm_set_coordinate (&mm->typecode);
	else mm_set_array (&mm->typecode);

	mm->symm = symm;
	// typecode[3] = 'S'
	if (symmetric) mm_set_symmetric (&mm->typecode);

	if (!is_type_supported (mm->typecode)) {
		char	msg[128];
		sprintf (msg, "matrix type does not supported :[%s].", mm_typecode_to_str (mm->typecode));
		error_and_exit ("mm_real_new", msg, __FILE__, __LINE__);
	}

	return mm;
}

/*** free mm_real ***/
void
mm_real_free (mm_real *mm)
{
	if (mm) {
		if (mm_real_is_sparse (mm)) {
			if (mm->i) free (mm->i);
			if (mm->p) free (mm->p);
		}
		if (mm->data) free (mm->data);
		free (mm);
	}
	return;
}

/*** reallocate mm_real ***/
bool
mm_real_realloc (mm_real *mm, const int nz)
{
	if (mm->nz == nz) return true;
	mm->data = (double *) realloc (mm->data, nz * sizeof (double));
	if (mm->data == NULL) return false;
	if (mm_real_is_sparse (mm)) {
		mm->i = (int *) realloc (mm->i, nz * sizeof (int));
		if (mm->i == NULL) return false;
	}
	mm->nz = nz;
	return true;
}

/*** set sparse ***/
void
mm_real_set_sparse (mm_real *x)
{
	if (mm_real_is_sparse (x)) return;
	mm_set_sparse (&(x->typecode));
	return;
}

/*** set dense ***/
void
mm_real_set_dense (mm_real *x)
{
	if (mm_real_is_dense (x)) return;
	mm_set_dense (&(x->typecode));
	return;
}

/*** set general ***/
void
mm_real_set_general (mm_real *x)
{
	if (!mm_real_is_symmetric (x)) return;
	mm_set_general (&(x->typecode));
	x->symm = MM_REAL_GENERAL;
	return;
}

/*** set symmetric
 * by default, assume symmetric upper
 * i.e., x->symm is set to MM_SYMMETRIC | MM_UPPER ***/
void
mm_real_set_symmetric (mm_real *x)
{
	if (mm_real_is_symmetric (x)) return;
	if (x->m != x->n) error_and_exit ("mm_real_set_symmetric", "symmetric matrix must be square.", __FILE__, __LINE__);
	mm_set_symmetric (&(x->typecode));
	x->symm = MM_SYMMETRIC | MM_UPPER;	// by default, assume symmetric upper
	return;
}

/*** set upper ***/
void
mm_real_set_upper (mm_real *x)
{
	if (!mm_real_is_symmetric (x)) error_and_exit ("mm_real_set_upper", "matrix must be symmetric.", __FILE__, __LINE__);
	if (x->m != x->n) error_and_exit ("mm_real_set_upper", "symmetric matrix must be square.", __FILE__, __LINE__);
	if (mm_real_is_upper (x)) return;
	x->symm = MM_SYMMETRIC | MM_UPPER;
	return;
}

/*** set lower ***/
void
mm_real_set_lower (mm_real *x)
{
	if (!mm_real_is_symmetric (x)) error_and_exit ("mm_real_set_lower", "matrix must be symmetric.", __FILE__, __LINE__);
	if (x->m != x->n) error_and_exit ("mm_real_set_lower", "symmetric matrix must be square.", __FILE__, __LINE__);
	if (mm_real_is_lower (x)) return;
	x->symm = MM_SYMMETRIC | MM_LOWER;
	return;
}

/* copy sparse */
static mm_sparse *
mm_real_copy_sparse (const mm_sparse *src)
{
	int			k;
	mm_sparse	*dest = mm_real_new (MM_REAL_SPARSE, src->symm, src->m, src->n, src->nz);

	dest->i = (int *) malloc (src->nz * sizeof (int));
	dest->p = (int *) malloc ((src->n + 1) * sizeof (int));
	dest->data = (double *) malloc (src->nz * sizeof (double));

	for (k = 0; k < src->nz; k++) {
		dest->i[k] = src->i[k];
		dest->data[k] = src->data[k];
	}
	for (k = 0; k <= src->n; k++) dest->p[k] = src->p[k];

	return dest;
}

/* copy dense */
static mm_dense *
mm_real_copy_dense (const mm_dense *src)
{
	mm_dense	*dest = mm_real_new (MM_REAL_DENSE, src->symm, src->m, src->n, src->nz);
	dest->data = (double *) malloc (src->nz * sizeof (double));
	dcopy_ (&src->nz, src->data, &ione, dest->data, &ione);
	return dest;
}

/*** copy x ***/
mm_real *
mm_real_copy (const mm_real *x)
{
	return (mm_real_is_sparse (x)) ? mm_real_copy_sparse (x) : mm_real_copy_dense (x);
}

/* set all elements of array to val */
static void
mm_real_array_set_all (const int n, double *data, const double val)
{
	int		k;
	for (k = 0; k < n; k++) data[k] = val;
	return;
}

/*** set all data of mm to val ***/
void
mm_real_set_all (mm_real *mm, const double val)
{
	mm_real_array_set_all (mm->nz, mm->data, val);
	return;
}

/*** convert sparse -> dense ***/
mm_dense *
mm_real_sparse_to_dense (const mm_sparse *s)
{
	int			j;
	mm_dense	*d;

	if (!mm_real_is_sparse (s)) return false;
	d = mm_real_new (MM_REAL_DENSE, s->symm, s->m, s->n, s->m * s->n);
	d->data = (double *) malloc (d->nz * sizeof (double));
	mm_real_set_all (d, 0.);

	for (j = 0; j < s->n; j++) {
		int		k;
		for (k = s->p[j]; k < s->p[j + 1]; k++) {
			int		i = s->i[k];
			d->data[i + j * s->m] = s->data[k];
		}
	}
	return d;
}

/*** convert dense -> sparse ***/
/* fabs (x->data[j]) < threshold are set to 0 */
mm_sparse *
mm_real_dense_to_sparse (const mm_dense *d, const double threshold)
{
	int			j, k;
	mm_sparse	*s;

	if (!mm_real_is_dense (d)) return false;
	s = mm_real_new (MM_REAL_SPARSE, d->symm, d->m, d->n, d->nz);
	s->i = (int *) malloc (s->nz * sizeof (int));
	s->p = (int *) malloc ((s->n + 1) * sizeof (int));
	s->data = (double *) malloc (s->nz * sizeof (double));

	k = 0;
	s->p[0] = 0;
	for (j = 0; j < d->n; j++) {
		int		i;
		int		i0 = 0;
		int		i1 = d->m;
		if (mm_real_is_symmetric (d)) {
			if (mm_real_is_lower (d)) i0 = j;
			if (mm_real_is_upper (d)) i1 = j + 1;
		}
		for (i = i0; i < i1; i++) {
			double	dij = d->data[i + j * d->m];
			if (fabs (dij) >= threshold) {
				s->i[k] = i;
				s->data[k] = dij;
				k++;
			}
		}
		s->p[j + 1] = k;
	}
	if (s->nz != k) mm_real_realloc (s, k);
	return s;
}

/* convert sparse symmetric -> sparse general */
static mm_sparse *
mm_real_symmetric_to_general_sparse (const mm_sparse *x)
{
	int			j, m;
	mm_sparse	*s;
	if (!mm_real_is_symmetric (x)) return mm_real_copy (x);

	s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, x->m, x->n, 2 * x->nz);
	s->i = (int *) malloc (s->nz * sizeof (int));
	s->data = (double *) malloc (s->nz * sizeof (double));
	s->p = (int *) malloc ((s->n + 1) * sizeof (int));

	m = 0;
	s->p[0] = 0;
	for (j = 0; j < x->n; j++) {
		int		k, l;
		if (mm_real_is_upper (x)) {
			for (k = x->p[j]; k < x->p[j + 1]; k++) {
				s->i[m] = x->i[k];
				s->data[m++] = x->data[k];
			}
			for (l = j + 1; l < x->n; l++) {
				for (k = x->p[l]; k < x->p[l + 1]; k++) {
					if (x->i[k] == j) {
						s->i[m] = l;
						s->data[m++] = x->data[k];
						break;
					}
				}
			}
		} else if (mm_real_is_lower (x)) {
			for (l = 0; l < j; l++) {
				for (k = x->p[l]; k < x->p[l + 1]; k++) {
					if (x->i[k] == j) {
						s->i[m] = l;
						s->data[m++] = x->data[k];
						break;
					}
				}
			}
			for (k = x->p[j]; k < x->p[j + 1]; k++) {
				s->i[m] = x->i[k];
				s->data[m++] = x->data[k];
			}
		}
		s->p[j + 1] = m;
	}
	if (s->nz != m) mm_real_realloc (s, m);
	return s;
}

/* convert dense symmetric -> dense general */
static mm_dense *
mm_real_symmetric_to_general_dense (const mm_dense *x)
{
	int			i, j;
	mm_dense	*d = mm_real_copy (x);
	if (!mm_real_is_symmetric (x)) return d;

	for (j = 0; j < x->n; j++) {
		if (mm_real_is_upper (x)) {
			for (i = j + 1; i < x->m; i++) d->data[i + j * d->m] = x->data[j + i * x->m];
		} else if (mm_real_is_lower (x)) {
			for (i = 0; i < j; i++) d->data[i + j * d->m] = x->data[j + i * x->m];
		}
	}
	mm_real_set_general (d);
	return d;
}

/* convert symmetric -> general */
mm_real *
mm_real_symmetric_to_general (const mm_real *x)
{
	return (mm_real_is_sparse (x)) ? mm_real_symmetric_to_general_sparse (x) : mm_real_symmetric_to_general_dense (x);
}

/* identity sparse matrix */
static mm_sparse *
mm_real_seye (const int n)
{
	int			k;
	mm_sparse	*s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, n, n, n);
	s->i = (int *) malloc (n * sizeof (int));
	s->data = (double *) malloc (n * sizeof (double));
	s->p = (int *) malloc ((n + 1) * sizeof (int));

	s->p[0] = 0;
	for (k = 0; k < n; k++) {
		s->i[k] = k;
		s->data[k] = 1.;
		s->p[k + 1] = k + 1;
	}
	return s;
}

/* identity dense matrix */
static mm_dense *
mm_real_deye (const int n)
{
	int			k;
	mm_dense	*d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n, n, n * n);
	d->data = (double *) malloc (d->nz * sizeof (double));
	mm_real_set_all (d, 0.);
	for (k = 0; k < n; k++) d->data[k + k * n] = 1.;
	return d;
}

/*** n x n identity matrix ***/
mm_real *
mm_real_eye (MMRealFormat format, const int n)
{
	if (n <= 0) error_and_exit ("mm_real_eye", "index out of range.", __FILE__, __LINE__);
	return (format == MM_REAL_SPARSE) ? mm_real_seye (n) : mm_real_deye (n);
}

/* sum |s(:,j)| */
static double
mm_real_sj_asum (const mm_sparse *s, const int j)
{
	int		size = s->p[j + 1] - s->p[j];
	double	asum = dasum_ (&size, s->data + s->p[j], &ione);
	if (mm_real_is_symmetric (s)) {
		int		l;
		int		l0 = (mm_real_is_upper (s)) ? j + 1 : 0;
		int		l1 = (mm_real_is_upper (s)) ? s->n : j;
		for (l = l0; l < l1; l++) {
			int		k;
			for (k = s->p[l]; k < s->p[l + 1]; k++) {
				if (s->i[k] == j) {
					asum += fabs (s->data[k]);
					break;
				}
			}
		}
	}
	return asum;
}

/* sum |d(:,j)| */
static double
mm_real_dj_asum (const mm_dense *d, const int j)
{
	double	val = 0.;
	if (!mm_real_is_symmetric (d)) val = dasum_ (&d->m, d->data + j * d->m, &ione);
	else {
		int		len;
		if (mm_real_is_upper (d)) {
			len = j;
			val = dasum_ (&len, d->data + j * d->m, &ione);
			len = d->m - j;
			val += dasum_ (&len, d->data + j * d->m + j, &d->m);
		} else if (mm_real_is_lower (d)) {
			len = d->m - j;
			val = dasum_ (&len, d->data + j * d->m + j, &ione);
			len = j;
			val += dasum_ (&len, d->data + j, &d->m);
		}
	}
	return val;
}

/*** sum |x(:,j)| ***/
double
mm_real_xj_asum (const mm_real *x, const int j)
{
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_asum", "index out of range.", __FILE__, __LINE__);
	return (mm_real_is_sparse (x)) ? mm_real_sj_asum (x, j) : mm_real_dj_asum (x, j);
}

/* sum s(:,j) */
static double
mm_real_sj_sum (const mm_sparse *s, const int j)
{
	int		k;
	double	sum = 0.;
	for (k = s->p[j]; k < s->p[j + 1]; k++) sum += s->data[k];
	if (mm_real_is_symmetric (s)) {
		int		l;
		int		l0 = (mm_real_is_upper (s)) ? j + 1 : 0;
		int		l1 = (mm_real_is_upper (s)) ? s->n : j;
		for (l = l0; l < l1; l++) {
			for (k = s->p[l]; k < s->p[l + 1]; k++) {
				if (s->i[k] == j) {
					sum += s->data[k];
					break;
				}
			}
		}
	}
	return sum;
}

/* sum d(:,j) */
static double
mm_real_dj_sum (const mm_dense *d, const int j)
{
	int		k;
	double	sum = 0.;
	if (!mm_real_is_symmetric (d))
		for (k = 0; k < d->m; k++) sum += d->data[k + j * d->m];
	else {
		int		len;
		if (mm_real_is_upper (d)) {
			len = j;
			for (k = 0; k < len; k++) sum += d->data[k + j * d->m];
			len = d->m - j;
			for (k = 0; k < len; k++) sum += d->data[k * d->m + j * d->m + j];
		} else if (mm_real_is_lower (d)) {
			len = d->m - j;
			for (k = 0; k < len; k++) sum += d->data[k + j * d->m + j];
			len = j;
			for (k = 0; k < len; k++) sum += d->data[k * d->m + j];
		}
	}
	return sum;
}

/*** sum x(:,j) ***/
double
mm_real_xj_sum (const mm_real *x, const int j)
{
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_sum", "index out of range.", __FILE__, __LINE__);
	return (mm_real_is_sparse (x)) ? mm_real_sj_sum (x, j) : mm_real_dj_sum (x, j);
}

/* norm2 s(:,j) */
static double
mm_real_sj_nrm2 (const mm_sparse *s, const int j)
{
	int		size = s->p[j + 1] - s->p[j];
	double	nrm2 = dnrm2_ (&size, s->data + s->p[j], &ione);
	if (mm_real_is_symmetric (s)) {
		double	val = pow (nrm2, 2.);
		int		l;
		int		l0 = (mm_real_is_upper (s)) ? j + 1 : 0;
		int		l1 = (mm_real_is_upper (s)) ? s->n : j;
		for (l = l0; l < l1; l++) {
			int		k;
			for (k = s->p[l]; k < s->p[l + 1]; k++) {
				if (s->i[k] == j) {
					val += pow (s->data[k], 2.);
					break;
				}
			}
		}
		nrm2 = sqrt (val);
	}
	return nrm2;
}

/* norm2 d(:,j) */
static double
mm_real_dj_nrm2 (const mm_dense *d, const int j)
{
	double	nrm2;
	if (!mm_real_is_symmetric (d)) nrm2 = dnrm2_ (&d->m, d->data + j * d->m, &ione);
	else {
		int		len;
		double	val = 0.;
		if (mm_real_is_upper (d)) {
			len = j;
			val = ddot_ (&len, d->data + j * d->m, &ione, d->data + j * d->m, &ione);
			len = d->m - j;
			val += ddot_ (&len, d->data + j * d->m + j, &d->m, d->data + j * d->m + j, &d->m);
		} else if (mm_real_is_lower (d)) {
			len = d->m - j;
			val = ddot_ (&len, d->data + j * d->m + j, &ione, d->data + j * d->m + j, &ione);
			len = j;
			val += ddot_ (&len, d->data + j, &d->m, d->data + j, &d->m);
		}
		nrm2 = sqrt (val);
	}
	return nrm2;
}

/*** norm2 x(:,j) ***/
double
mm_real_xj_nrm2 (const mm_real *x, const int j)
{
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_nrm2", "index out of range.", __FILE__, __LINE__);
	return (mm_real_is_sparse (x)) ? mm_real_sj_nrm2 (x, j) : mm_real_dj_nrm2 (x, j);
}

/* s * y, where s is sparse matrix and y is dense vector */
static mm_dense *
mm_real_s_dot_y (bool trans, const double alpha, const mm_sparse *s, const mm_dense *y, const double beta)
{
	int			j;
	int			m;
	mm_dense	*d;
	m = (trans) ? s->n : s->m;
	
	d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m, 1, m);
	d->data = (double *) malloc (d->nz * sizeof (double));
	mm_real_set_all (d, beta);

	for (j = 0; j < s->n; j++) {
		int		k;
		for (k = s->p[j]; k < s->p[j + 1]; k++) {
			int		si = (trans) ? j : s->i[k];
			int		sj = (trans) ? s->i[k] : j;
			d->data[si] += s->data[k] * y->data[sj];
			if (mm_real_is_symmetric (s) && j != s->i[k]) d->data[sj] += s->data[k] * y->data[si];
		}		
	}
	return d;
}

/* d * y, where d is dense matrix and y is dense vector */
static mm_dense *
mm_real_d_dot_y (bool trans, const double alpha, const mm_dense *d, const mm_dense *y, const double beta)
{
	int			m;
	mm_dense	*c;
	m = (trans) ? d->n : d->m;

	c = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m, 1, m);
	c->data = (double *) malloc (c->nz * sizeof (double));

	if (!mm_real_is_symmetric (d))
		dgemv_ ((trans) ? "T" : "N", &d->m, &d->n, &alpha, d->data, &d->m, y->data, &ione, &beta, c->data, &ione);
	else {
		char	uplo = (mm_real_is_upper (d)) ? 'U' : 'L';
		dsymv_ (&uplo, &d->m, &alpha, d->data, &d->m, y->data, &ione, &beta, c->data, &ione);
	}
	return c;
}

/*** x * y, where x is matrix and y is dense vector ***/
mm_dense *
mm_real_x_dot_y (bool trans, const double alpha, const mm_real *x, const mm_dense *y, const double beta)
{
	if (!mm_real_is_dense (y)) error_and_exit ("mm_real_x_dot_y", "y must be dense.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (y)) error_and_exit ("mm_real_x_dot_y", "y must be general.", __FILE__, __LINE__);
	if (y->n != 1) error_and_exit ("mm_real_x_dot_y", "y must be vector.", __FILE__, __LINE__);
	if ((trans && x->m != y->m) || (!trans && x->n != y->m))
		error_and_exit ("mm_real_x_dot_y", "vector and matrix dimensions do not match.", __FILE__, __LINE__);

	return (mm_real_is_sparse (x)) ? mm_real_s_dot_y (trans, alpha, x, y, beta) : mm_real_d_dot_y (trans, alpha, x, y, beta);
}

/* s(:,j)' * y */
static double
mm_real_sj_trans_dot_y (const mm_sparse *s, const int j, const mm_dense *y)
{
	int	k;
	double	val = 0;
	for (k = s->p[j]; k < s->p[j + 1]; k++) val += s->data[k] * y->data[s->i[k]];
	if (mm_real_is_symmetric (s)) {
		int		l;
		int		l0 = (mm_real_is_upper (s)) ? j + 1 : 0;
		int		l1 = (mm_real_is_upper (s)) ? s->n : j;
		for (l = l0; l < l1; l++) {
			for (k = s->p[l]; k < s->p[l + 1]; k++) {
				if (s->i[k] == j) {
					val += s->data[k] * y->data[l];
					break;
				}
			}
		}
	}
	return val;
}

/* d(:,j)' * y */
static double
mm_real_dj_trans_dot_y (const mm_dense *d, const int j, const mm_dense *y)
{
	double	val = 0.;
	if (!mm_real_is_symmetric (d)) val = ddot_ (&d->m, d->data + j * d->m, &ione, y->data, &ione);
	else {
		int		len;
		if (mm_real_is_upper (d)) {
			len = j;
			val = ddot_ (&len, d->data + j * d->m, &ione, y->data, &ione);
			len = d->m - j;
			val += ddot_ (&len, d->data + j * d->m + j, &d->m, y->data + j, &ione);
		} else if (mm_real_is_lower (d)) {
			len = d->m - j;
			val = ddot_ (&len, d->data + j * d->m + j, &ione, y->data + j, &ione);
			len = j;
			val += ddot_ (&len, d->data + j, &d->m, y->data, &ione);
		}
	}
	return val;
}

/*** x(:,j)' * y ***/
double
mm_real_xj_trans_dot_y (const mm_real *x, const int j, const mm_dense *y)
{
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_trans_dot_y", "index out of range.", __FILE__, __LINE__);
	if (!mm_real_is_dense (y)) error_and_exit ("mm_real_xj_trans_dot_y", "y must be dense.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (y)) error_and_exit ("mm_real_xj_trans_dot_y", "y must be general.", __FILE__, __LINE__);
	if (y->n != 1) error_and_exit ("mm_real_xj_trans_dot_y", "y must be vector.", __FILE__, __LINE__);
	if (x->m != y->m) error_and_exit ("mm_real_xj_trans_dot_y", "vector and matrix dimensions do not match.", __FILE__, __LINE__);

	return (mm_real_is_sparse (x)) ? mm_real_sj_trans_dot_y (x, j, y) : mm_real_dj_trans_dot_y (x, j, y);
}

/* y = alpha * s(:,j) + y */
static void
mm_real_asjpy (const double alpha, const mm_sparse *s, const int j, mm_dense *y)
{
	int		k;
	for (k = s->p[j]; k < s->p[j + 1]; k++) y->data[s->i[k]] += alpha * s->data[k];
	if (mm_real_is_symmetric (s)) {
		int		l;
		int		l0 = (mm_real_is_upper (s)) ? j + 1 : 0;
		int		l1 = (mm_real_is_upper (s)) ? s->n : j;
		for (l = l0; l < l1; l++) {
			for (k = s->p[l]; k < s->p[l + 1]; k++) {
				if (s->i[k] == j) {
					y->data[l] += alpha * s->data[k];
					break;
				}
			}
		}
	}
	return;
}

/* y = alpha * d(:,j) + y */
static void
mm_real_adjpy (const double alpha, const mm_dense *d, const int j, mm_dense *y)
{
	if (!mm_real_is_symmetric (d)) daxpy_ (&d->m, &alpha, d->data + j * d->m, &ione, y->data, &ione);
	else {
		int		len;
		if (mm_real_is_upper (d)) {
			len = j;
			daxpy_ (&len, &alpha, d->data + j * d->m, &ione, y->data, &ione);
			len = d->m - j;
			daxpy_ (&len, &alpha, d->data + j * d->m + j, &d->m, y->data + j, &ione);
		} else if (mm_real_is_lower (d)) {
			len = d->m - j;
			daxpy_ (&len, &alpha, d->data + j * d->m + j, &ione, y->data + j, &ione);
			len = j;
			daxpy_ (&len, &alpha, d->data + j, &d->m, y->data, &ione);
		}
	}
	return;
}

/*** y = alpha * x(:,j) + y ***/
void
mm_real_axjpy (const double alpha, const mm_real *x, const int j, mm_dense *y)
{
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_axjpy", "index out of range.", __FILE__, __LINE__);
	if (!mm_real_is_dense (y)) error_and_exit ("mm_real_axjpy", "y must be dense.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (y)) error_and_exit ("mm_real_axjpy", "y must be general.", __FILE__, __LINE__);
	if (y->n != 1) error_and_exit ("mm_real_axjpy", "y must be vector.", __FILE__, __LINE__);
	if (x->m != y->m) error_and_exit ("mm_real_axjpy", "vector and matrix dimensions do not match.", __FILE__, __LINE__);

	return (mm_real_is_sparse (x)) ? mm_real_asjpy (alpha, x, j, y) : mm_real_adjpy (alpha, x, j, y);
}

/* y = alpha * s(:,j) + y, atomic */
static void
mm_real_asjpy_atomic (const double alpha, const mm_sparse *s, const int j, mm_dense *y)
{
	int		k;
	for (k = s->p[j]; k < s->p[j + 1]; k++) atomic_add (y->data + s->i[k], alpha * s->data[k]);
	if (mm_real_is_symmetric (s)) {
		int		l;
		int		l0 = (mm_real_is_upper (s)) ? j + 1 : 0;
		int		l1 = (mm_real_is_upper (s)) ? s->n : j;
		for (l = l0; l < l1; l++) {
			for (k = s->p[l]; k < s->p[l + 1]; k++) {
				if (s->i[k] == j) {
					atomic_add (y->data + l, alpha * s->data[k]);
					break;
				}
			}
		}
	}
	return;
}

/* y = alpha * d(:,j) + y, atomic */
static void
mm_real_adjpy_atomic (const double alpha, const mm_dense *d, const int j, mm_dense *y)
{
	int		k;
	double	*dj;
	if (!mm_real_is_symmetric (d)) {
		dj = d->data + j * d->m;
		for (k = 0; k < d->m; k++) atomic_add (y->data + k, alpha * dj[k]);
	} else {
		int		len;
		if (mm_real_is_upper (d)) {
			len = j;
			dj = d->data + j * d->m;
			for (k = 0; k < len; k++) atomic_add (y->data + k, alpha * dj[k]);
			len = d->m - j;
			dj = d->data + j * d->m + j;
			for (k = 0; k < len; k++) atomic_add (y->data + j + k, alpha * dj[k * d->m]);
		} else if (mm_real_is_lower (d)) {
			len = d->m - j;
			dj = d->data + j * d->m + j;
			for (k = 0; k < len; k++) atomic_add (y->data + j + k, alpha * dj[k]);
			len = j;
			dj = d->data + j;
			for (k = 0; k < len; k++) atomic_add (y->data + k, alpha * dj[k * d->m]);
		}
	}
	return;
}

/*** y = alpha * x(:,j) + y, atomic ***/
void
mm_real_axjpy_atomic (const double alpha, const mm_real *x, const int j, mm_dense *y)
{
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_axjpy_atomic", "index out of range.", __FILE__, __LINE__);
	if (!mm_real_is_dense (y)) error_and_exit ("mm_real_axjpy_atomic", "y must be dense.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (y)) error_and_exit ("mm_real_axjpy_atomic", "y must be general.", __FILE__, __LINE__);
	if (y->n != 1) error_and_exit ("mm_real_axjpy_atomic", "y must be vector.", __FILE__, __LINE__);
	if (x->m != y->m) error_and_exit ("mm_real_axjpy_atomic", "vector and matrix dimensions do not match.", __FILE__, __LINE__);

	return (mm_real_is_sparse (x)) ? mm_real_asjpy_atomic (alpha, x, j, y) : mm_real_adjpy_atomic (alpha, x, j, y);
}

/* fread sparse */
static mm_sparse *
mm_real_fread_sparse (FILE *fp, MM_typecode typecode)
{
	int			k, l;
	int			m, n, nz;
	int			*j;
	mm_sparse	*s;

	mm_read_mtx_crd_size (fp, &m, &n, &nz);
	s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, m, n, nz);
	s->i = (int *) malloc (s->nz * sizeof (int));
	s->data = (double *) malloc (s->nz * sizeof (double));
	s->p = (int *) malloc ((s->n + 1) * sizeof (int));

	j = (int *) malloc (s->nz * sizeof (int));
	mm_read_mtx_crd_data (fp, s->m, s->n, s->nz, s->i, j, s->data, typecode);

	l = 0;
	for (k = 0; k < s->nz; k++) {
		s->i[k]--;	// fortran -> c
		while (l < j[k]) s->p[l++] = k;
	}
	while (l <= n) s->p[l++] = k;

	if (mm_is_symmetric (typecode)) {
		mm_real_set_symmetric (s);
		for (k = 0; k < s->nz; k++) {
			if (s->i[k] == j[k] - 1) continue;
			(s->i[k] < j[k] - 1) ? mm_real_set_upper (s) : mm_real_set_lower (s);
			break;
		}
	}
	free (j);

	return s;
}

/* fread dense */
static mm_dense *
mm_real_fread_dense (FILE *fp, MM_typecode typecode)
{
	int			k;
	int			m, n;
	int			ret;
	mm_dense	*d;

	mm_read_mtx_array_size (fp, &m, &n);
	d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m, n, m * n);
	d->data = (double *) malloc (d->nz * sizeof (double));

	k = 0;
	do {
		ret = fscanf (fp, "%lf", &d->data[k]);
		if (ret > 0 && ++k >= d->nz) break;
	} while (ret != EOF);

	return d;
}

/*** fread MatrixMarket format file ***/
mm_real *
mm_real_fread (FILE *fp)
{
	MM_typecode	typecode;
	mm_real		*x;
	mm_read_banner (fp, &typecode);
	if (!is_type_supported (typecode)) {
		char	msg[128];
		sprintf (msg, "matrix type does not supported :[%s].", mm_typecode_to_str (typecode));
		error_and_exit ("mm_real_fread", msg, __FILE__, __LINE__);
	}
	x = (mm_is_sparse (typecode)) ? mm_real_fread_sparse (fp, typecode) : mm_real_fread_dense (fp, typecode);
	if (mm_real_is_symmetric (x) && x->m != x->n) {
		mm_real_free (x);
		error_and_exit ("mm_real_fread", "symmetric matrix must be square.", __FILE__, __LINE__);
	}
	return x;
}

/* fwrite sparse */
static void
mm_real_fwrite_sparse (FILE *stream, mm_sparse *s, const char *format)
{
	int		j, k;
	mm_write_banner (stream, s->typecode);
	mm_write_mtx_crd_size (stream, s->m, s->n, s->nz);
	for (j = 0; j < s->n; j++) {
		for (k = s->p[j]; k < s->p[j + 1]; k++) {
			fprintf (stream, "%d %d ", s->i[k] + 1, j + 1);	// c -> fortran
			fprintf (stream, format, s->data[k]);
			fprintf (stream, "\n");
		}
	}
	return;
}

/* fwrite dense */
static void
mm_real_fwrite_dense (FILE *stream, mm_dense *d, const char *format)
{
	int		k;
	mm_write_banner (stream, d->typecode);
	mm_write_mtx_array_size (stream, d->m, d->n);
	for (k = 0; k < d->nz; k++) {
		fprintf (stream, format, d->data[k]);
		fprintf (stream, "\n");
	}
	return;
}

/*** fwrite in MatrixMarket format ***/
void
mm_real_fwrite (FILE *stream, mm_real *x, const char *format)
{
	return (mm_real_is_sparse (x)) ? mm_real_fwrite_sparse (stream, x, format) : mm_real_fwrite_dense (stream, x, format);
}
