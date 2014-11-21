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

#include "private/private.h"
#include "private/atomic.h"

/* num of error codes */
static const int num_error_code = 6;

/* error code */
enum {
	MM_REAL_IS_VALID		= 100,	// x is valid
	MM_REAL_IS_NULL		= 101,	// x == NULL
	MM_REAL_IS_EMPTY		= 102,	// x->n == 0 || x == 0 || x->nz == 0
	MM_REAL_DATA_IS_NULL	= 103,	// x->data == NULL
	MM_REAL_I_IS_NULL		= 104,	// x->i == NULL
	MM_REAL_P_IS_NULL		= 105		// x->p == NULL
};

/* error message */
static const char	*error_msg[6] = {
	"mm_real is valid.",
	"mm_real is not allocated.",
	"mm_real is empty.",
	"x->data is not allocated.",
	"x->i is not allocated.",
	"x->p is not allocated."
};

/* mm_real supports real symmetric/general sparse/dense matrix */
static bool
is_type_supported (const MM_typecode typecode)
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

static int
mm_real_is_valid (const mm_real *x)
{
	if (x == NULL) return MM_REAL_IS_NULL;
	if (x->m <= 0 || x->n <= 0 || x->nz <= 0) return MM_REAL_IS_EMPTY;
	if (x->data == NULL) return MM_REAL_DATA_IS_NULL;
	if (mm_real_is_sparse (x)) {
		if (x->i == NULL) return MM_REAL_I_IS_NULL;
		if (x->p == NULL) return MM_REAL_P_IS_NULL;
	}
	return MM_REAL_IS_VALID;
}

/* print error message and exit */
static void
mm_real_error_and_exit (const char *funcname, const int error_code)
{
	int		id = error_code - 100;
	if (id < 0 || num_error_code <= id) return;
	if (id == 0) return;	// mm_real is valid
	error_and_exit (funcname, error_msg[id], __FILE__, __LINE__);
}

/* check format */
static bool
is_format_valid (const MMRealFormat format) {
	return (format == MM_REAL_SPARSE || format == MM_REAL_DENSE);
}

/* check symmetric */
static bool
is_symm_valid (const MMRealSymm symm)
{
	return (symm == MM_REAL_GENERAL || symm == MM_REAL_SYMMETRIC_UPPER
			|| symm == MM_REAL_SYMMETRIC_LOWER);
}

/* allocate mm_real */
static mm_real *
mm_real_alloc (void)
{
	mm_real	*x = (mm_real *) malloc (sizeof (mm_real));
	if (x == NULL) return NULL;

	x->m = 0;
	x->n = 0;
	x->nz = 0;
	x->i = NULL;
	x->p = NULL;
	x->data = NULL;

	x->symm = MM_REAL_GENERAL;

	/* set typecode = "M_RG" : Matrix Real General */
	// typecode[3] = 'G' : General
	mm_initialize_typecode (&x->typecode);
	// typecode[0] = 'M' : Matrix
	mm_set_matrix (&x->typecode);
	// typecode[2] = 'R' : Real
	mm_set_real (&x->typecode);

	return x;
}

/*** create new mm_real object
 * MMRealFormat	format: MM_REAL_DENSE or MM_REAL_SPARSE
 * MMRealSymm		symm  : MM_REAL_GENERAL, MM_REAL_SYMMETRIC_UPPER or MM_REAL_SYMMETRIC_LOWER
 * int				m, n  : rows and columns of the matrix
 * int				nz    : number of nonzero elements of the matrix ***/
mm_real *
mm_real_new (MMRealFormat format, MMRealSymm symm, const int m, const int n, const int nz)
{
	mm_real	*x;
	bool		symmetric;

	if (!is_format_valid (format))
		error_and_exit ("mm_real_new", "invalid MMRealFormat format.", __FILE__, __LINE__);
	if (!is_symm_valid (symm))
		error_and_exit ("mm_real_new", "invalid MMRealSymm symm.", __FILE__, __LINE__);

	symmetric = symm & MM_SYMMETRIC;
	if (symmetric && m != n)
		error_and_exit ("mm_real_new", "symmetric matrix must be square.", __FILE__, __LINE__);

	x = mm_real_alloc ();
	if (x == NULL) error_and_exit ("mm_real_new", "failed to allocate object.", __FILE__, __LINE__);
	x->m = m;
	x->n = n;
	x->nz = nz;

	// typecode[1] = 'C' or 'A'
	if (format == MM_REAL_SPARSE) mm_set_coordinate (&x->typecode);
	else mm_set_array (&x->typecode);

	x->symm = symm;
	// typecode[3] = 'S'
	if (symmetric) mm_set_symmetric (&x->typecode);

	if (!is_type_supported (x->typecode)) {
		char	msg[128];
		sprintf (msg, "matrix type does not supported :[%s].", mm_typecode_to_str (x->typecode));
		error_and_exit ("mm_real_new", msg, __FILE__, __LINE__);
	}

	return x;
}

/*** free mm_real ***/
void
mm_real_free (mm_real *x)
{
	if (x) {
		if (mm_real_is_sparse (x)) {
			if (x->i) free (x->i);
			if (x->p) free (x->p);
		}
		if (x->data) free (x->data);
		free (x);
	}
	return;
}

/*** reallocate mm_real ***/
bool
mm_real_realloc (mm_real *x, const int nz)
{
	if (x->nz == nz) return true;
	x->data = (double *) realloc (x->data, nz * sizeof (double));
	if (x->data == NULL) return false;
	if (mm_real_is_sparse (x)) {
		x->i = (int *) realloc (x->i, nz * sizeof (int));
		if (x->i == NULL) return false;
	}
	x->nz = nz;
	return true;
}

/*** set to sparse ***/
void
mm_real_set_sparse (mm_real *x)
{
	if (mm_real_is_sparse (x)) return;
	mm_set_sparse (&(x->typecode));
	return;
}

/*** set to dense ***/
void
mm_real_set_dense (mm_real *x)
{
	if (mm_real_is_dense (x)) return;
	mm_set_dense (&(x->typecode));
	return;
}

/*** set to general ***/
void
mm_real_set_general (mm_real *x)
{
	if (!mm_real_is_symmetric (x)) return;
	mm_set_general (&(x->typecode));
	x->symm = MM_REAL_GENERAL;
	return;
}

/*** set to symmetric
 * by default, assume symmetric upper
 * i.e., x->symm is set to MM_SYMMETRIC | MM_UPPER ***/
void
mm_real_set_symmetric (mm_real *x)
{
	if (x->m != x->n) error_and_exit ("mm_real_set_symmetric", "symmetric matrix must be square.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (x)) return;
	mm_set_symmetric (&(x->typecode));
	x->symm = MM_SYMMETRIC | MM_UPPER;	// by default, assume symmetric upper
	return;
}

/*** set to symmetric upper ***/
void
mm_real_set_upper (mm_real *x)
{
	if (x->m != x->n) error_and_exit ("mm_real_set_upper", "symmetric matrix must be square.", __FILE__, __LINE__);
	if (!mm_real_is_symmetric (x)) error_and_exit ("mm_real_set_upper", "matrix must be symmetric.", __FILE__, __LINE__);
	if (mm_real_is_upper (x)) return;
	x->symm = MM_SYMMETRIC | MM_UPPER;
	return;
}

/*** set to symmetric lower ***/
void
mm_real_set_lower (mm_real *x)
{
	if (x->m != x->n) error_and_exit ("mm_real_set_lower", "symmetric matrix must be square.", __FILE__, __LINE__);
	if (!mm_real_is_symmetric (x)) error_and_exit ("mm_real_set_lower", "matrix must be symmetric.", __FILE__, __LINE__);
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

	for (k = 0; k < src->nz; k++) dest->i[k] = src->i[k];
	for (k = 0; k <= src->n; k++) dest->p[k] = src->p[k];
	dcopy_ (&src->nz, src->data, &ione, dest->data, &ione);

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

/*** set x->data to val ***/
void
mm_real_set_all (mm_real *x, const double val)
{
	mm_real_array_set_all (x->nz, x->data, val);
	return;
}

/*** convert sparse -> dense ***/
mm_dense *
mm_real_sparse_to_dense (const mm_sparse *s)
{
	int			j;
	mm_dense	*d;

	if (mm_real_is_dense (s)) return mm_real_copy (s);
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

/*** convert dense -> sparse
 * if fabs (x->data[j]) < threshold, set to 0 ***/
mm_sparse *
mm_real_dense_to_sparse (const mm_dense *d, const double threshold)
{
	int			j, k;
	mm_sparse	*s;

	if (mm_real_is_sparse (d)) return mm_real_copy (d);
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

/* find element that s->i[l] = j in the k-th column of s and return its index l */
static int
find_jth_row_element_from_sk (const int j, const mm_sparse *s, const int k)
{
	int		l;
	for (l = s->p[k]; l < s->p[k + 1]; l++) {
		if (s->i[l] < j) continue;
		else if (s->i[l] == j) return l;	// found
		else break;
	}
	return -1;	// not found
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
		int		k;
		if (mm_real_is_upper (x)) {
			for (k = x->p[j]; k < x->p[j + 1]; k++) {
				s->i[m] = x->i[k];
				s->data[m++] = x->data[k];
			}
			for (k = j + 1; k < x->n; k++) {
				int		l = find_jth_row_element_from_sk (j, x, k);
				// if found
				if (l >= 0) {
					s->i[m] = k;
					s->data[m++] = x->data[l];
				}
			}
		} else if (mm_real_is_lower (x)) {
			for (k = 0; k < j; k++) {
				int		l = find_jth_row_element_from_sk (j, x, k);
				// if found
				if (l >= 0) {
					s->i[m] = k;
					s->data[m++] = x->data[l];
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
	int			j;
	mm_dense	*d = mm_real_copy (x);
	if (!mm_real_is_symmetric (x)) return d;

	for (j = 0; j < x->n; j++) {
		if (mm_real_is_upper (x)) {
			int		i0 = j + 1;
			int		len = x->m - i0;
			dcopy_ (&len, x->data + j + i0 * x->m, &x->m, d->data + i0 + j * d->m, &ione);
		} else if (j > 0 && mm_real_is_lower (x)) {
			dcopy_ (&j, x->data + j, &x->m, d->data + j * d->m, &ione);
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
	if (n <= 0) error_and_exit ("mm_real_eye", "invalid size.", __FILE__, __LINE__);
	return (format == MM_REAL_SPARSE) ? mm_real_seye (n) : mm_real_deye (n);
}

/* s = [s1; s2] */
static mm_sparse *
mm_real_vertcat_sparse (const mm_sparse *s1, const mm_sparse *s2)
{
	int			i, j, k;
	int			m = s1->m + s2->m;
	int			n = s1->n;
	int			nz = s1->nz + s2->nz;
	mm_sparse	*s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, m, n, nz);
	s->i = (int *) malloc (s->nz * sizeof (int));
	s->data = (double *) malloc (s->nz * sizeof (double));
	s->p = (int *) malloc ((s->n + 1) * sizeof (int));

	k = 0;
	s->p[0] = 0;
	for (j = 0; j < n; j++) {
		int		len1 = s1->p[j + 1] - s1->p[j];
		int		len2 = s2->p[j + 1] - s2->p[j];
		dcopy_ (&len1, s1->data + s1->p[j], &ione, s->data + k, &ione);
		for (i = s1->p[j]; i < s1->p[j + 1]; i++) s->i[k++] = s1->i[i];
		dcopy_ (&len2, s2->data + s2->p[j], &ione, s->data + k, &ione);
		for (i = s2->p[j]; i < s2->p[j + 1]; i++) s->i[k++] = s2->i[i] + s1->m;
		s->p[j + 1] = k;
	}
	if (s->nz != k) mm_real_realloc (s, k);
	return s;
}

/* d = [d1; d2] */
static mm_dense *
mm_real_vertcat_dense (const mm_dense *d1, const mm_dense *d2)
{
	int			j;
	int			m = d1->m + d2->m;
	int			n = d1->n;
	int			nz = d1->nz + d2->nz;
	mm_dense	*d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m, n, nz);
	d->data = (double *) malloc (d->nz * sizeof (double));

	for (j = 0; j < n; j++) {
		dcopy_ (&d1->m, d1->data + j * d1->m, &ione, d->data + j * d->m, &ione);
		dcopy_ (&d2->m, d2->data + j * d2->m, &ione, d->data + j * d->m + d1->m, &ione);
	}
	return d;
}

/*** x = [x1; x2] ***/
mm_real *
mm_real_vertcat (const mm_real *x1, const mm_real *x2)
{
	if ((mm_real_is_sparse (x1) && mm_real_is_dense (x1)) || (mm_real_is_dense (x1) && mm_real_is_sparse (x1)))
		error_and_exit ("mm_real_vertcat", "format of matrix x1 and x2 are incompatible.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (x1) || mm_real_is_symmetric (x2))
		error_and_exit ("mm_real_vertcat", "matrix must be general.", __FILE__, __LINE__);
	if (x1->n != x2->n) error_and_exit ("mm_real_vertcat", "matrix size is incompatible.", __FILE__, __LINE__);

	return (mm_real_is_sparse (x1)) ? mm_real_vertcat_sparse (x1, x2) : mm_real_vertcat_dense (x1, x2);
}

/* s = [s1, s2] */
static mm_sparse *
mm_real_holzcat_sparse (const mm_sparse *s1, const mm_sparse *s2)
{
	int			j, k;
	int			m = s1->m;
	int			n = s1->n + s2->n;
	int			nz = s1->nz + s2->nz;
	mm_sparse	*s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, m, n, nz);
	s->i = (int *) malloc (s->nz * sizeof (int));
	s->data = (double *) malloc (s->nz * sizeof (double));
	s->p = (int *) malloc ((s->n + 1) * sizeof (int));

	for (k = 0; k < s1->nz; k++) s->i[k] = s1->i[k];
	for (k = 0; k < s2->nz; k++) s->i[k + s1->nz] = s2->i[k];
	for (j = 0; j <= s1->n; j++) s->p[j] = s1->p[j];
	for (j = 0; j <= s2->n; j++) s->p[j + s1->n] = s2->p[j] + s1->nz;
	dcopy_ (&s1->nz, s1->data, &ione, s->data, &ione);
	dcopy_ (&s2->nz, s2->data, &ione, s->data + s1->nz, &ione);

	return s;
}

/* d = [d1, d2] */
static mm_dense *
mm_real_holzcat_dense (const mm_dense *d1, const mm_dense *d2)
{
	int			m = d1->m;
	int			n = d1->n + d2->n;
	int			nz = d1->nz + d2->nz;
	mm_dense	*d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m, n, nz);
	d->data = (double *) malloc (d->nz * sizeof (double));

	dcopy_ (&d1->nz, d1->data, &ione, d->data, &ione);
	dcopy_ (&d2->nz, d2->data, &ione, d->data + d1->nz, &ione);

	return d;
}

/*** x = [x1, x2] ***/
mm_real *
mm_real_holzcat (const mm_real *x1, const mm_real *x2)
{
	if ((mm_real_is_sparse (x1) && mm_real_is_dense (x1)) || (mm_real_is_dense (x1) && mm_real_is_sparse (x1)))
		error_and_exit ("mm_real_holzcat", "format of matrix x1 and x2 are incompatible.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (x1) || mm_real_is_symmetric (x2))
		error_and_exit ("mm_real_holzcat", "matrix must be general.", __FILE__, __LINE__);
	if (x1->m != x2->m) error_and_exit ("mm_real_holzcat", "matrix size is incompatible.", __FILE__, __LINE__);

	return (mm_real_is_sparse (x1)) ? mm_real_holzcat_sparse (x1, x2) : mm_real_holzcat_dense (x1, x2);
}

/*** x(:,j) += alpha ***/
void
mm_real_xj_add_const (mm_real *x, const int j, const double alpha)
{
	int		k;
	int		len;
	double	*data;
	if (mm_real_is_symmetric (x)) error_and_exit ("mm_real_xj_add_const", "matrix must be general.", __FILE__, __LINE__);
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_add_const", "index out of range.", __FILE__, __LINE__);

	len = (mm_real_is_sparse (x)) ? x->p[j + 1] - x->p[j] : x->m;
	data = x->data + ((mm_real_is_sparse (x)) ? x->p[j] : j * x->m);

	for (k = 0; k < len; k++) data[k] += alpha;

	return;
}

/*** x(:,j) *= alpha ***/
void
mm_real_xj_scale (mm_real *x, const int j, const double alpha)
{
	int		len;
	double	*data;
	if (mm_real_is_symmetric (x)) error_and_exit ("mm_real_xj_scale", "matrix must be general.", __FILE__, __LINE__);
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_scale", "index out of range.", __FILE__, __LINE__);

	len = (mm_real_is_sparse (x)) ? x->p[j + 1] - x->p[j] : x->m;
	data = x->data + ((mm_real_is_sparse (x)) ? x->p[j] : j * x->m);

	dscal_ (&len, &alpha, data, &ione);

	return;
}

/* sum |s(:,j)| */
static double
mm_real_sj_asum (const mm_sparse *s, const int j)
{
	int		size = s->p[j + 1] - s->p[j];
	double	asum = dasum_ (&size, s->data + s->p[j], &ione);
	if (mm_real_is_symmetric (s)) {
		int		k;
		int		k0 = (mm_real_is_upper (s)) ? j + 1 : 0;
		int		k1 = (mm_real_is_upper (s)) ? s->n : j;
		for (k = k0; k < k1; k++) {
			int		l = find_jth_row_element_from_sk (j, s, k);
			// if found
			if (l >= 0) asum += fabs (s->data[l]);
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
		int		k0 = (mm_real_is_upper (s)) ? j + 1 : 0;
		int		k1 = (mm_real_is_upper (s)) ? s->n : j;
		for (k = k0; k < k1; k++) {
			int		l = find_jth_row_element_from_sk (j, s, k);
			// if found
			if (l >= 0) sum += s->data[l];
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

/* sum_i s(i,j)^2 */
static double
mm_real_sj_ssq (const mm_sparse *s, const int j)
{
	int		size = s->p[j + 1] - s->p[j];
	double	ssq = ddot_ (&size, s->data + s->p[j], &ione, s->data + s->p[j], &ione);
	if (mm_real_is_symmetric (s)) {
		int		k;
		int		k0 = (mm_real_is_upper (s)) ? j + 1 : 0;
		int		k1 = (mm_real_is_upper (s)) ? s->n : j;
		for (k = k0; k < k1; k++) {
			int		l = find_jth_row_element_from_sk (j, s, k);
			// if found
			if (l >= 0) ssq += pow (s->data[l], 2.);
		}
	}
	return ssq;
}

/* sum_i d(i,j)^2 */
static double
mm_real_dj_ssq (const mm_dense *d, const int j)
{
	double	ssq;
	if (!mm_real_is_symmetric (d)) ssq = ddot_ (&d->m, d->data + j * d->m, &ione, d->data + j * d->m, &ione);
	else {
		int		len;
		ssq = 0.;
		if (mm_real_is_upper (d)) {
			len = j;
			ssq = ddot_ (&len, d->data + j * d->m, &ione, d->data + j * d->m, &ione);
			len = d->m - j;
			ssq += ddot_ (&len, d->data + j * d->m + j, &d->m, d->data + j * d->m + j, &d->m);
		} else if (mm_real_is_lower (d)) {
			len = d->m - j;
			ssq = ddot_ (&len, d->data + j * d->m + j, &ione, d->data + j * d->m + j, &ione);
			len = j;
			ssq += ddot_ (&len, d->data + j, &d->m, d->data + j, &d->m);
		}
	}
	return ssq;
}

/*** sum_i x(i,j)^2 ***/
double
mm_real_xj_ssq (const mm_real *x, const int j)
{
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_ssq", "index out of range.", __FILE__, __LINE__);
	return (mm_real_is_sparse (x)) ? mm_real_sj_ssq (x, j) : mm_real_dj_ssq (x, j);
}

/*** norm2 x(:,j) ***/
double
mm_real_xj_nrm2 (const mm_real *x, const int j)
{
	double	ssq;
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_nrm2", "index out of range.", __FILE__, __LINE__);
	ssq = mm_real_xj_ssq (x, j);
	return sqrt (ssq);
}

/* alpha * s * y, where s is sparse matrix and y is dense vector */
static mm_dense *
mm_real_s_dot_y (bool trans, const double alpha, const mm_sparse *s, const mm_dense *y)
{
	int			j;
	int			m;
	mm_dense	*c;
	m = (trans) ? s->n : s->m;
	
	c = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m, 1, m);
	c->data = (double *) malloc (c->nz * sizeof (double));
	mm_real_set_all (c, 0.);

	for (j = 0; j < s->n; j++) {
		int		k;
		for (k = s->p[j]; k < s->p[j + 1]; k++) {
			int		si = (trans) ? j : s->i[k];
			int		sj = (trans) ? s->i[k] : j;
			c->data[si] += alpha * s->data[k] * y->data[sj];
			if (mm_real_is_symmetric (s) && j != s->i[k]) c->data[sj] += alpha * s->data[k] * y->data[si];
		}		
	}
	return c;
}

/* alpha * d * y, where d is dense matrix and y is dense vector */
static mm_dense *
mm_real_d_dot_y (bool trans, const double alpha, const mm_dense *d, const mm_dense *y)
{
	int			m;
	mm_dense	*c;
	m = (trans) ? d->n : d->m;

	c = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m, 1, m);
	c->data = (double *) malloc (c->nz * sizeof (double));
	mm_real_set_all (c, 0.);

	if (!mm_real_is_symmetric (d)) {
		// c = alpha*d*y + 0*c
		dgemv_ ((trans) ? "T" : "N", &d->m, &d->n, &alpha, d->data, &d->m, y->data, &ione, &dzero, c->data, &ione);
	} else {
		char	uplo = (mm_real_is_upper (d)) ? 'U' : 'L';
		// c = alpha*d*y + 0*c
		dsymv_ (&uplo, &d->m, &alpha, d->data, &d->m, y->data, &ione, &dzero, c->data, &ione);
	}
	return c;
}

/*** alpha * x * y, where x is sparse/dense matrix and y is dense vector ***/
mm_dense *
mm_real_x_dot_y (bool trans, const double alpha, const mm_real *x, const mm_dense *y)
{
	if (!mm_real_is_dense (y)) error_and_exit ("mm_real_x_dot_y", "y must be dense.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (y)) error_and_exit ("mm_real_x_dot_y", "y must be general.", __FILE__, __LINE__);
	if (y->n != 1) error_and_exit ("mm_real_x_dot_y", "y must be vector.", __FILE__, __LINE__);
	if ((trans && x->m != y->m) || (!trans && x->n != y->m))
		error_and_exit ("mm_real_x_dot_y", "vector and matrix dimensions do not match.", __FILE__, __LINE__);

	return (mm_real_is_sparse (x)) ? mm_real_s_dot_y (trans, alpha, x, y) : mm_real_d_dot_y (trans, alpha, x, y);
}

/* s(:,j)' * y */
static double
mm_real_sj_trans_dot_y (const mm_sparse *s, const int j, const mm_dense *y)
{
	int		k;
	double	val = 0;
	for (k = s->p[j]; k < s->p[j + 1]; k++) val += s->data[k] * y->data[s->i[k]];
	if (mm_real_is_symmetric (s)) {
		int		k0 = (mm_real_is_upper (s)) ? j + 1 : 0;
		int		k1 = (mm_real_is_upper (s)) ? s->n : j;
		for (k = k0; k < k1; k++) {
			int		l = find_jth_row_element_from_sk (j, s, k);
			// if found
			if (l >= 0) val += s->data[l] * y->data[k];
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
		int		k0 = (mm_real_is_upper (s)) ? j + 1 : 0;
		int		k1 = (mm_real_is_upper (s)) ? s->n : j;
		for (k = k0; k < k1; k++) {
			int		l = find_jth_row_element_from_sk (j, s, k);
			// if found
			if (l >= 0) y->data[k] += alpha * s->data[l];
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
		int		k0 = (mm_real_is_upper (s)) ? j + 1 : 0;
		int		k1 = (mm_real_is_upper (s)) ? s->n : j;
		for (k = k0; k < k1; k++) {
			int		l = find_jth_row_element_from_sk (j, s, k);
			// if found
			if (l >= 0) atomic_add (y->data + k, alpha * s->data[l]);
		}
	}
	return;
}

/* y = alpha * d(:,j) + y, atomic */
static void
mm_real_adjpy_atomic (const double alpha, const mm_dense *d, const int j, mm_dense *y)
{
	int		k;
	if (!mm_real_is_symmetric (d)) {
		for (k = 0; k < d->m; k++) atomic_add (y->data + k, alpha * d->data[j * d->m + k]);
	} else {
		int		len;
		if (mm_real_is_upper (d)) {
			len = j;
			for (k = 0; k < len; k++) atomic_add (y->data + k, alpha * d->data[j * d->m + k]);
			len = d->m - j;
			for (k = 0; k < len; k++) atomic_add (y->data + j + k, alpha * d->data[j * d->m + j + k * d->m]);
		} else if (mm_real_is_lower (d)) {
			len = d->m - j;
			for (k = 0; k < len; k++) atomic_add (y->data + j + k, alpha * d->data[j * d->m + j + k]);
			len = j;
			for (k = 0; k < len; k++) atomic_add (y->data + k, alpha * d->data[j + k * d->m]);
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

	if (mm_read_mtx_crd_size (fp, &m, &n, &nz) != 0) return NULL;
	s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, m, n, nz);
	s->i = (int *) malloc (s->nz * sizeof (int));
	s->data = (double *) malloc (s->nz * sizeof (double));
	s->p = (int *) malloc ((s->n + 1) * sizeof (int));

	j = (int *) malloc (s->nz * sizeof (int));
	if (mm_read_mtx_crd_data (fp, s->m, s->n, s->nz, s->i, j, s->data, typecode) != 0) {
		free (j);
		mm_real_free (s);
		return NULL;
	}

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

	if (mm_read_mtx_array_size (fp, &m, &n) != 0) return NULL;
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
	if (mm_read_banner (fp, &typecode) != 0) error_and_exit ("mm_real_fread", "failed to read mm_real.", __FILE__, __LINE__);
	if (!is_type_supported (typecode)) {
		char	msg[128];
		sprintf (msg, "matrix type does not supported :[%s].", mm_typecode_to_str (typecode));
		error_and_exit ("mm_real_fread", msg, __FILE__, __LINE__);
	}
	x = (mm_is_sparse (typecode)) ? mm_real_fread_sparse (fp, typecode) : mm_real_fread_dense (fp, typecode);
	mm_real_error_and_exit ("mm_real_xj_axjpy_atomic", mm_real_is_valid (x));
	if (mm_real_is_symmetric (x) && x->m != x->n) error_and_exit ("mm_real_fread", "symmetric matrix must be square.", __FILE__, __LINE__);
	return x;
}

/* fwrite sparse */
static void
mm_real_fwrite_sparse (FILE *stream, const mm_sparse *s, const char *format)
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
mm_real_fwrite_dense (FILE *stream, const mm_dense *d, const char *format)
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
mm_real_fwrite (FILE *stream, const mm_real *x, const char *format)
{
	return (mm_real_is_sparse (x)) ? mm_real_fwrite_sparse (stream, x, format) : mm_real_fwrite_dense (stream, x, format);
}
