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
	x->nnz = 0;
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
 * int				nnz   : number of nonzero elements of the matrix ***/
mm_real *
mm_real_new (MMRealFormat format, MMRealSymm symm, const int m, const int n, const int nnz)
{
	mm_real	*x;
	bool	symmetric;

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
	x->nnz = nnz;

	// typecode[1] = 'C' or 'A'
	if (format == MM_REAL_SPARSE) {
		mm_set_coordinate (&x->typecode);
		x->i = (int *) malloc (x->nnz * sizeof (int));
		x->p = (int *) malloc ((x->n + 1) * sizeof (int));
		if (x->i == NULL || x->p == NULL) error_and_exit ("mm_real_new", "cannot allocate memory.", __FILE__, __LINE__);
		// initialize x->p[0]
		x->p[0] = 0;
	} else mm_set_array (&x->typecode);

	x->symm = symm;
	// typecode[3] = 'G' -> 'S'
	if (symmetric) mm_set_symmetric (&x->typecode);

	if (!is_type_supported (x->typecode)) {
		char	msg[128];
		sprintf (msg, "matrix type does not supported :[%s].", mm_typecode_to_str (x->typecode));
		error_and_exit ("mm_real_new", msg, __FILE__, __LINE__);
	}

	// allocate arrays
	x->data = (double *) malloc (x->nnz * sizeof (double));
	if (x->data == NULL) error_and_exit ("mm_real_new", "cannot allocate memory.", __FILE__, __LINE__);

	return x;
}

/*** free mm_real ***/
void
mm_real_free (mm_real *x)
{
	if (x) {
		if (x->i) free (x->i);
		if (x->p) free (x->p);
		if (x->data) free (x->data);
		free (x);
	}
	return;
}

/*** reallocate mm_real ***/
bool
mm_real_realloc (mm_real *x, const int nnz)
{
	if (x->nnz == nnz) return true;
	x->data = (double *) realloc (x->data, nnz * sizeof (double));
	if (x->data == NULL) return false;
	if (mm_real_is_sparse (x)) {
		x->i = (int *) realloc (x->i, nnz * sizeof (int));
		if (x->i == NULL) return false;
	}
	x->nnz = nnz;
	return true;
}

/*** set to general ***/
static void
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
static void
mm_real_set_symmetric (mm_real *x)
{
	if (mm_real_is_symmetric (x)) return;
	mm_set_symmetric (&(x->typecode));
	x->symm = MM_SYMMETRIC | MM_UPPER;	// by default, assume symmetric upper
	return;
}

/*** set to symmetric upper ***/
static void
mm_real_set_upper (mm_real *x)
{
	if (mm_real_is_upper (x)) return;
	x->symm = MM_SYMMETRIC | MM_UPPER;
	return;
}

/*** set to symmetric lower ***/
static void
mm_real_set_lower (mm_real *x)
{
	if (mm_real_is_lower (x)) return;
	x->symm = MM_SYMMETRIC | MM_LOWER;
	return;
}

/* copy sparse */
static mm_sparse *
mm_real_copy_sparse (const mm_sparse *src)
{
	int			k;
	mm_sparse	*dest = mm_real_new (MM_REAL_SPARSE, src->symm, src->m, src->n, src->nnz);

	int			*si = src->i;
	int			*sp = src->p;
	int			*di = dest->i;
	int			*dp = dest->p;

	for (k = 0; k < src->nnz; k++) di[k] = si[k];
	for (k = 0; k <= src->n; k++) dp[k] = sp[k];
	dcopy_ (&src->nnz, src->data, &ione, dest->data, &ione);

	return dest;
}

/* copy dense */
static mm_dense *
mm_real_copy_dense (const mm_dense *src)
{
	mm_dense	*dest = mm_real_new (MM_REAL_DENSE, src->symm, src->m, src->n, src->nnz);
	dcopy_ (&src->nnz, src->data, &ione, dest->data, &ione);
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
	mm_real_array_set_all (x->nnz, x->data, val);
	return;
}

/*** convert sparse -> dense ***/
mm_dense *
mm_real_sparse_to_dense (const mm_sparse *s)
{
	int			j, k;
	int			p, n;
	mm_dense	*d;

	int			*si;
	double		*sd;
	int			*sp = s->p;
	double		*dd;

	if (mm_real_is_dense (s)) return mm_real_copy (s);
	d = mm_real_new (MM_REAL_DENSE, s->symm, s->m, s->n, s->m * s->n);
	mm_real_set_all (d, 0.);

	dd = d->data;
	for (j = 0; j < s->n; j++) {
		p = sp[j];
		n = sp[j + 1] - p;
		si = s->i + p;
		sd = s->data + p;
		dd = d->data + j * s->m;
		for (k = 0; k < n; k++) dd[si[k]] = sd[k];
	}
	return d;
}

/*** convert dense -> sparse
 * if fabs (x->data[j]) < threshold, set to 0 ***/
mm_sparse *
mm_real_dense_to_sparse (const mm_dense *d, const double threshold)
{
	int			i, j, k;
	mm_sparse	*s;

	int			*si;
	int			*sp;
	double		*sd;
	double		*dd = d->data;
	double		dij;

	if (mm_real_is_sparse (d)) return mm_real_copy (d);
	s = mm_real_new (MM_REAL_SPARSE, d->symm, d->m, d->n, d->nnz);

	si = s->i;
	sp = s->p;
	sd = s->data;

	k = 0;
	if (!mm_real_is_symmetric (d)) {
		for (j = 0; j < d->n; j++) {
			for (i = 0; i < d->m; i++) {
				dij = dd[i + j * d->m];
				if (fabs (dij) >= threshold) {
					si[k] = i;
					sd[k] = dij;
					k++;
				}
			}
			sp[j + 1] = k;
		}
	} else {
		if (mm_real_is_upper (d)) {
			for (j = 0; j < d->n; j++) {
				for (i = 0; i < j + 1; i++) {
					dij = dd[i + j * d->m];
					if (fabs (dij) >= threshold) {
						si[k] = i;
						sd[k] = dij;
						k++;
					}
				}
				sp[j + 1] = k;
			}
		} else if (mm_real_is_lower (d)) {
			for (j = 0; j < d->n; j++) {
				for (i = j; i < d->m; i++) {
					dij = dd[i + j * d->m];
					if (fabs (dij) >= threshold) {
						si[k] = i;
						sd[k] = dij;
						k++;
					}
				}
				sp[j + 1] = k;
			}
		}
	}
	if (s->nnz != k) mm_real_realloc (s, k);
	return s;
}

/* find element that s->i[l] = j in the k-th column of s and return its index l */
static int
find_jth_row_element_of_sk (const int j, const mm_sparse *s, const int k)
{
	int		l;
	int		p = s->p[k];
	int		n = s->p[k + 1] - p;
	int		*si = s->i + p;
	for (l = 0; l < n; l++) {
		int		sil = si[l];
		if (sil < j) continue;
		if (sil == j) return l + p;	// found
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

	int			*si;
	int			*sp;
	double		*sd;
	int			*xi = x->i;
	int			*xp = x->p;
	double		*xd = x->data;

	if (!mm_real_is_symmetric (x)) return mm_real_copy (x);
	s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, x->m, x->n, 2 * x->nnz);

	si = s->i;
	sp = s->p;
	sd = s->data;

	m = 0;
	for (j = 0; j < x->n; j++) {
		int		k;
		int		pend = xp[j + 1];
		if (mm_real_is_upper (x)) {
			for (k = xp[j]; k < pend; k++) {
				si[m] = xi[k];
				sd[m++] = xd[k];
			}
			for (k = j + 1; k < x->n; k++) {
				int		l = find_jth_row_element_of_sk (j, x, k);
				// if found
				if (l >= 0) {
					si[m] = k;
					sd[m++] = xd[l];
				}
			}
		} else if (mm_real_is_lower (x)) {
			for (k = 0; k < j; k++) {
				int		l = find_jth_row_element_of_sk (j, x, k);
				// if found
				if (l >= 0) {
					si[m] = k;
					sd[m++] = xd[l];
				}
			}
			for (k = xp[j]; k < pend; k++) {
				si[m] = xi[k];
				sd[m++] = xd[k];
			}
		}
		sp[j + 1] = m;
	}
	if (s->nnz != m) mm_real_realloc (s, m);
	return s;
}

/* convert dense symmetric -> dense general */
static mm_dense *
mm_real_symmetric_to_general_dense (const mm_dense *x)
{
	int			j;
	mm_dense	*d = mm_real_copy (x);
	if (!mm_real_is_symmetric (x)) return d;

	if (mm_real_is_upper (x)) {
		for (j = 0; j < x->n; j++) {
			int		i0 = j + 1;
			int		n = x->m - i0;
			dcopy_ (&n, x->data + j + i0 * x->m, &x->m, d->data + i0 + j * d->m, &ione);
		}
	} else {
		for (j = 0; j < x->n; j++) dcopy_ (&j, x->data + j, &x->m, d->data + j * d->m, &ione);
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
	int			*si = s->i;
	int			*sp = s->p;
	double		*sd = s->data;
	for (k = 0; k < n; k++) {
		si[k] = k;
		sd[k] = 1.;
		sp[k + 1] = k + 1;
	}
	return s;
}

/* identity dense matrix */
static mm_dense *
mm_real_deye (const int n)
{
	int			k;
	mm_dense	*d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, n, n, n * n);
	double		*dd = d->data;
	mm_real_set_all (d, 0.);
	for (k = 0; k < n; k++) dd[k + k * n] = 1.;
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
	int			nnz = s1->nnz + s2->nnz;
	mm_sparse	*s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, m, n, nnz);

	int			*si = s->i;
	int			*sp = s->p;
	int			*si1 = s1->i;
	int			*sp1 = s1->p;
	int			*si2 = s2->i;
	int			*sp2 = s2->p;

	k = 0;
	for (j = 0; j < n; j++) {
		int		n1 = sp1[j + 1] - sp1[j];
		int		n2 = sp2[j + 1] - sp2[j];
		int		pend;
		dcopy_ (&n1, s1->data + sp1[j], &ione, s->data + k, &ione);
		pend = sp1[j + 1];
		for (i = sp1[j]; i < pend; i++) si[k++] = si1[i];
		dcopy_ (&n2, s2->data + sp2[j], &ione, s->data + k, &ione);
		pend = sp2[j + 1];
		for (i = sp2[j]; i < pend; i++) si[k++] = si2[i] + s1->m;
		sp[j + 1] = k;
	}
	return s;
}

/* d = [d1; d2] */
static mm_dense *
mm_real_vertcat_dense (const mm_dense *d1, const mm_dense *d2)
{
	int			j;
	int			m = d1->m + d2->m;
	int			n = d1->n;
	int			nnz = d1->nnz + d2->nnz;
	mm_dense	*d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, m, n, nnz);

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
	int			nnz = s1->nnz + s2->nnz;
	mm_sparse	*s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, m, n, nnz);

	int			*si = s->i;
	int			*sp = s->p;
	int			nnz1 = s1->nnz;
	int			*si1 = s1->i;
	int			*sp1 = s1->p;
	int			nnz2 = s2->nnz;
	int			*si2 = s2->i;
	int			*sp2 = s2->p;

	for (k = 0; k < nnz1; k++) si[k] = si1[k];
	for (k = 0; k < nnz2; k++) si[k + nnz1] = si2[k];
	for (j = 0; j <= s1->n; j++) sp[j] = sp1[j];
	for (j = 0; j <= s2->n; j++) sp[j + s1->n] = sp2[j] + nnz1;
	dcopy_ (&nnz1, s1->data, &ione, s->data, &ione);
	dcopy_ (&nnz2, s2->data, &ione, s->data + nnz1, &ione);

	return s;
}

/* d = [d1, d2] */
static mm_dense *
mm_real_holzcat_dense (const mm_dense *d1, const mm_dense *d2)
{
	mm_dense	*d = mm_real_new (MM_REAL_DENSE, MM_REAL_GENERAL, d1->m, d1->n + d2->n, d1->nnz + d2->nnz);

	dcopy_ (&d1->nnz, d1->data, &ione, d->data, &ione);
	dcopy_ (&d2->nnz, d2->data, &ione, d->data + d1->nnz, &ione);

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
	int		n;
	double	*data;
	if (mm_real_is_symmetric (x)) error_and_exit ("mm_real_xj_add_const", "matrix must be general.", __FILE__, __LINE__);
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_add_const", "index out of range.", __FILE__, __LINE__);

	if (mm_real_is_sparse (x)) {
		n = x->p[j + 1] - x->p[j];
		data = x->data + x->p[j];
	} else {
		n = x->m;
		data = x->data + j * x->m;
	}
	for (k = 0; k < n; k++) data[k] += alpha;

	return;
}

/*** x(:,j) *= alpha ***/
void
mm_real_xj_scale (mm_real *x, const int j, const double alpha)
{
	int		n;
	double	*data;
	if (mm_real_is_symmetric (x)) error_and_exit ("mm_real_xj_scale", "matrix must be general.", __FILE__, __LINE__);
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_scale", "index out of range.", __FILE__, __LINE__);

	if (mm_real_is_sparse (x)) {
		int		p = x->p[j];
		n = x->p[j + 1] - p;
		data = x->data + p;
	} else {
		n = x->m;
		data = x->data + j * x->m;
	}
	dscal_ (&n, &alpha, data, &ione);

	return;
}

/* sum |s(:,j)| */
static double
mm_real_sj_asum (const mm_sparse *s, const int j)
{
	int		p = s->p[j];
	int		n = s->p[j + 1] - p;
	double	*sd = s->data;
	double	asum = dasum_ (&n, sd + p, &ione);
	if (mm_real_is_symmetric (s)) {
		int		k;
		int		k0;
		int		k1;
		if (mm_real_is_upper (s)) {
			k0 = j + 1;
			k1 = s->n;
		} else {
			k0 = 0;
			k1 = j;
		}
		for (k = k0; k < k1; k++) {
			int		l = find_jth_row_element_of_sk (j, s, k);
			// if found
			if (l >= 0) asum += fabs (sd[l]);
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
		int		n;
		if (mm_real_is_upper (d)) {
			n = j;
			val = dasum_ (&n, d->data + j * d->m, &ione);
			n = d->m - j;
			val += dasum_ (&n, d->data + j * d->m + j, &d->m);
		} else if (mm_real_is_lower (d)) {
			n = d->m - j;
			val = dasum_ (&n, d->data + j * d->m + j, &ione);
			n = j;
			val += dasum_ (&n, d->data + j, &d->m);
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
	int		p = s->p[j];
	int		n = s->p[j + 1] - p;
	double	*sd = s->data + p;
	double	sum = 0.;
	for (k = 0; k < n; k++) sum += sd[k];
	if (mm_real_is_symmetric (s)) {
		int		k0;
		int		k1;
		if (mm_real_is_upper (s)) {
			k0 = j + 1;
			k1 = s->n;
		} else {
			k0 = 0;
			k1 = j;
		}
		sd = s->data;
		for (k = k0; k < k1; k++) {
			int		l = find_jth_row_element_of_sk (j, s, k);
			// if found
			if (l >= 0) sum += sd[l];
		}
	}
	return sum;
}

/* sum d(:,j) */
static double
mm_real_dj_sum (const mm_dense *d, const int j)
{
	int		k;
	double	*dd;
	double	sum = 0.;
	if (!mm_real_is_symmetric (d)) {
		dd = d->data + j * d->m;
		for (k = 0; k < d->m; k++) sum += dd[k];
	} else {
		int		n;
		if (mm_real_is_upper (d)) {
			n = j;
			dd = d->data + j * d->m;
			for (k = 0; k < n; k++) sum += dd[k];
			n = d->m - j;
			dd = d->data + j + j * d->m;
			for (k = 0; k < n; k++) sum += dd[k * d->m];
		} else if (mm_real_is_lower (d)) {
			n = d->m - j;
			dd = d->data + j + j * d->m;
			for (k = 0; k < n; k++) sum += dd[k];
			n = j;
			dd = d->data + j;
			for (k = 0; k < n; k++) sum += dd[k * d->m];
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
	int		p = s->p[j];
	int		n = s->p[j + 1] - p;
	double	*sd = s->data;
	double	ssq = ddot_ (&n, sd + p, &ione, sd + p, &ione);
	if (mm_real_is_symmetric (s)) {
		int		k;
		int		k0;
		int		k1;
		if (mm_real_is_upper (s)) {
			k0 = j + 1;
			k1 = s->n;
		} else {
			k0 = 0;
			k1 = j;
		}
		for (k = k0; k < k1; k++) {
			int		l = find_jth_row_element_of_sk (j, s, k);
			// if found
			if (l >= 0) ssq += pow (sd[l], 2.);
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
		int		n;
		ssq = 0.;
		if (mm_real_is_upper (d)) {
			n = j;
			ssq = ddot_ (&n, d->data + j * d->m, &ione, d->data + j * d->m, &ione);
			n = d->m - j;
			ssq += ddot_ (&n, d->data + j * d->m + j, &d->m, d->data + j * d->m + j, &d->m);
		} else if (mm_real_is_lower (d)) {
			n = d->m - j;
			ssq = ddot_ (&n, d->data + j * d->m + j, &ione, d->data + j * d->m + j, &ione);
			n = j;
			ssq += ddot_ (&n, d->data + j, &d->m, d->data + j, &d->m);
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

/* z = alpha * s * y + beta * z, where s is sparse matrix and y is dense vector */
static void
mm_real_s_dot_y (bool trans, const double alpha, const mm_sparse *s, const mm_dense *y, const double beta, mm_dense *z)
{
	int			j, k;
	int			p, n;

	int			*si;
	double		*sd;
	int			*sp = s->p;
	double		*yd = y->data;
	double		*zd = z->data;

	if (fabs (beta) > 0.) mm_real_xj_scale (z, 0, beta);
	else mm_real_set_all (z, 0.);

	if (trans) {
		if (!mm_real_is_symmetric (s)) {
			for (j = 0; j < s->n; j++) {
				p = sp[j];
				n = sp[j + 1] - p;
				if (n <= 0) continue;
				si = s->i + p;
				sd = s->data + p;
				for (k = 0; k < n; k++) zd[j] += alpha * sd[k] * yd[si[k]];
			}
		} else if (mm_real_is_upper (s)) {
			int		sik;
			double	alpha_sdk;
			for (j = 0; j < s->n; j++) {
				p = sp[j];
				n = sp[j + 1] - p;
				si = s->i + p;
				sd = s->data + p;
				for (k = 0; k < n - 1; k++) {
					sik = si[k];
					alpha_sdk = alpha * sd[k];
					zd[j] += alpha_sdk * yd[sik];
					zd[sik] += alpha_sdk * yd[j];
				}
				if (n <= 0) continue;
				sik = si[n - 1];
				alpha_sdk = alpha * sd[n - 1];
				zd[j] += alpha_sdk * yd[sik];
				// omit if diagonal element
				if (j != sik) zd[sik] += alpha_sdk * yd[j];
			}
		} else {
			int		sik;
			double	alpha_sdk;
			for (j = 0; j < s->n; j++) {
				p = sp[j];
				n = sp[j + 1] - p;
				if (n <= 0) continue;
				si = s->i + p;
				sd = s->data + p;
				sik = si[0];
				alpha_sdk = alpha * sd[0];
				zd[j] += alpha_sdk * yd[sik];
				// omit if diagonal element
				if (j != sik) zd[sik] += alpha_sdk * yd[j];
				for (k = 1; k < n; k++) {
					sik = si[k];
					alpha_sdk = alpha * sd[k];
					zd[j] += alpha_sdk * yd[sik];
					zd[sik] += alpha_sdk * yd[j];
				}
			}
		}
	} else {
		if (!mm_real_is_symmetric (s)) {
			for (j = 0; j < s->n; j++) {
				p = sp[j];
				n = sp[j + 1] - p;
				if (n <= 0) continue;
				si = s->i + p;
				sd = s->data + p;
				for (k = 0; k < n; k++) zd[si[k]] += alpha * sd[k] * yd[j];
			}
		} else if (mm_real_is_upper (s)) {
			int		sik;
			double	alpha_sdk;
			for (j = 0; j < s->n; j++) {
				p = sp[j];
				n = sp[j + 1] - p;
				si = s->i + p;
				sd = s->data + p;
				for (k = 0; k < n - 1; k++) {
					sik = si[k];
					alpha_sdk = alpha * sd[k];
					zd[sik] += alpha_sdk * yd[j];
					zd[j] += alpha_sdk * yd[sik];
				}
				if (n <= 0) continue;
				sik = si[n - 1];
				alpha_sdk = alpha * sd[n - 1];
				zd[sik] += alpha_sdk * yd[j];
				// omit if diagonal element
				if (j != sik) zd[j] += alpha_sdk * yd[sik];
			}
		} else {
			int		sik;
			double	alpha_sdk;
			for (j = 0; j < s->n; j++) {
				p = sp[j];
				n = sp[j + 1] - p;
				if (n <= 0) continue;
				si = s->i + p;
				sd = s->data + p;
				sik = si[0];
				alpha_sdk = alpha * sd[0];
				zd[sik] += alpha_sdk * yd[j];
				// omit if diagonal element
				if (j != sik) zd[j] += alpha_sdk * yd[sik];
				for (k = 1; k < n; k++) {
					sik = si[k];
					alpha_sdk = alpha * sd[k];
					zd[sik] += alpha_sdk * yd[j];
					zd[j] += alpha_sdk * yd[sik];
				}
			}
		}
	}	
	return;
}

/* z = alpha * d * y + beta * z, where d is dense matrix and y is dense vector */
static void
mm_real_d_dot_y (bool trans, const double alpha, const mm_dense *d, const mm_dense *y, const double beta, mm_dense *z)
{
	if (!mm_real_is_symmetric (d)) {
		// z = alpha * d * y + beta * z
		dgemv_ ((trans) ? "T" : "N", &d->m, &d->n, &alpha, d->data, &d->m, y->data, &ione, &beta, z->data, &ione);
	} else {
		char	uplo = (mm_real_is_upper (d)) ? 'U' : 'L';
		// z = alpha * d * y + beta * z
		dsymv_ (&uplo, &d->m, &alpha, d->data, &d->m, y->data, &ione, &beta, z->data, &ione);
	}
	return;
}

/*** alpha * x * y, where x is sparse/dense matrix and y is dense vector ***/
void
mm_real_x_dot_y (bool trans, const double alpha, const mm_real *x, const mm_dense *y, const double beta, mm_dense *z)
{
	if (!mm_real_is_dense (y)) error_and_exit ("mm_real_x_dot_y", "y must be dense.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (y)) error_and_exit ("mm_real_x_dot_y", "y must be general.", __FILE__, __LINE__);
	if (y->n != 1) error_and_exit ("mm_real_x_dot_y", "y must be vector.", __FILE__, __LINE__);
	if ((trans && x->m != y->m) || (!trans && x->n != y->m))
		error_and_exit ("mm_real_x_dot_y", "dimensions of x and y do not match.", __FILE__, __LINE__);
	if (!mm_real_is_dense (z)) error_and_exit ("mm_real_x_dot_y", "z must be dense.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (z)) error_and_exit ("mm_real_x_dot_y", "z must be general.", __FILE__, __LINE__);
	if (z->n != 1) error_and_exit ("mm_real_x_dot_y", "z must be vector.", __FILE__, __LINE__);
	if ((trans && x->n != z->m) || (!trans && x->m != z->m))
		error_and_exit ("mm_real_x_dot_y", "dimensions of x and z do not match.", __FILE__, __LINE__);

	return (mm_real_is_sparse (x)) ? mm_real_s_dot_y (trans, alpha, x, y, beta, z) : mm_real_d_dot_y (trans, alpha, x, y, beta, z);
}

/* s(:,j)' * y */
static double
mm_real_sj_trans_dot_y (const mm_sparse *s, const int j, const mm_dense *y)
{
	double	val = 0;

	int		*si;
	double	*sd;
	double	*yd = y->data;

	int		k;
	int		p = s->p[j];
	int		n = s->p[j + 1] - p;
	int		mod = n % 4;

	// unrolling
	si = s->i + p;
	sd = s->data + p;
	if (mod == 1) {
		val += sd[0] * yd[si[0]];
		si++;
		sd++;
	} else if (mod == 2) {
		val += sd[0] * yd[si[0]] + sd[1] * yd[si[1]];
		si += 2;
		sd += 2;
	} else if (mod == 3) {
		val += sd[0] * yd[si[0]] + sd[1] * yd[si[1]] + sd[2] * yd[si[2]];
		si += 3;
		sd += 3;
	}
	for (k = mod; k < n; k += 4) {
		val += sd[0] * yd[si[0]] + sd[1] * yd[si[1]] + sd[2] * yd[si[2]] + sd[3] * yd[si[3]];
		si += 4;
		sd += 4;
	}

	if (mm_real_is_symmetric (s)) {
		int		k0;
		int		k1;
		if (mm_real_is_upper (s)) {
			k0 = j + 1;
			k1 = s->n;
		} else {
			k0 = 0;
			k1 = j;
		}
		sd = s->data;
		for (k = k0; k < k1; k++) {
			int		l = find_jth_row_element_of_sk (j, s, k);
			// if found
			if (l >= 0) val += sd[l] * yd[k];
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
		int		n;
		if (mm_real_is_upper (d)) {
			n = j;
			val = ddot_ (&n, d->data + j * d->m, &ione, y->data, &ione);
			n = d->m - j;
			val += ddot_ (&n, d->data + j * d->m + j, &d->m, y->data + j, &ione);
		} else if (mm_real_is_lower (d)) {
			n = d->m - j;
			val = ddot_ (&n, d->data + j * d->m + j, &ione, y->data + j, &ione);
			n = j;
			val += ddot_ (&n, d->data + j, &d->m, y->data, &ione);
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
	int		*si;
	double	*sd;
	double	*yd = y->data;

	int		k;
	int		p = s->p[j];
	int		n = s->p[j + 1] - p;
	si = s->i + p;
	sd = s->data + p;
	for (k = 0; k < n; k++) yd[si[k]] += alpha * sd[k];
	if (mm_real_is_symmetric (s)) {
		int		k0;
		int		k1;
		if (mm_real_is_upper (s)) {
			k0 = j + 1;
			k1 = s->n;
		} else {
			k0 = 0;
			k1 = j;
		}
		sd = s->data;
		for (k = k0; k < k1; k++) {
			int		l = find_jth_row_element_of_sk (j, s, k);
			// if found
			if (l >= 0) yd[k] += alpha * sd[l];
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
		int		n;
		if (mm_real_is_upper (d)) {
			n = j;
			daxpy_ (&n, &alpha, d->data + j * d->m, &ione, y->data, &ione);
			n = d->m - j;
			daxpy_ (&n, &alpha, d->data + j * d->m + j, &d->m, y->data + j, &ione);
		} else if (mm_real_is_lower (d)) {
			n = d->m - j;
			daxpy_ (&n, &alpha, d->data + j * d->m + j, &ione, y->data + j, &ione);
			n = j;
			daxpy_ (&n, &alpha, d->data + j, &d->m, y->data, &ione);
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
	int		*si;
	double	*sd;
	double	*yd = y->data;

	int		k;
	int		p = s->p[j];
	int		n = s->p[j + 1] - p;
	si = s->i + p;
	sd = s->data + p;
	for (k = 0; k < n; k++) atomic_add (yd + si[k], alpha * sd[k]);
	if (mm_real_is_symmetric (s)) {
		int		k0;
		int		k1;
		if (mm_real_is_upper (s)) {
			k0 = j + 1;
			k1 = s->n;
		} else {
			k0 = 0;
			k1 = j;
		}
		sd = s->data;
		for (k = k0; k < k1; k++) {
			int		l = find_jth_row_element_of_sk (j, s, k);
			// if found
			if (l >= 0) atomic_add (yd + k, alpha * sd[l]);
		}
	}
	return;
}

/* y = alpha * d(:,j) + y, atomic */
static void
mm_real_adjpy_atomic (const double alpha, const mm_dense *d, const int j, mm_dense *y)
{
	double	*dd;
	double	*yd = y->data;

	int		k;
	if (!mm_real_is_symmetric (d)) {
		dd = d->data + j * d->m;
		for (k = 0; k < d->m; k++) atomic_add (yd + k, alpha * dd[k]);
	} else {
		int		n;
		if (mm_real_is_upper (d)) {
			n = j;
			dd = d->data + j * d->m;
			for (k = 0; k < n; k++) atomic_add (yd + k, alpha * dd[k]);
			n = d->m - j;
			dd = d->data + j * d->m + j;
			for (k = 0; k < n; k++) atomic_add (yd + j + k, alpha * dd[k * d->m]);
		} else if (mm_real_is_lower (d)) {
			n = d->m - j;
			dd = d->data + j * d->m + j;
			for (k = 0; k < n; k++) atomic_add (yd + j + k, alpha * dd[k]);
			n = j;
			dd = d->data + j;
			for (k = 0; k < n; k++) atomic_add (yd + k, alpha * dd[k * d->m]);
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
	int			m, n, nnz;
	int			*j;
	mm_sparse	*s;

	if (mm_read_mtx_crd_size (fp, &m, &n, &nnz) != 0) return NULL;
	s = mm_real_new (MM_REAL_SPARSE, MM_REAL_GENERAL, m, n, nnz);

	j = (int *) malloc (s->nnz * sizeof (int));
	if (mm_read_mtx_crd_data (fp, s->m, s->n, s->nnz, s->i, j, s->data, typecode) != 0) {
		free (j);
		mm_real_free (s);
		return NULL;
	}

	l = 0;
	for (k = 0; k < s->nnz; k++) {
		s->i[k]--;	// fortran -> c
		while (l < j[k]) s->p[l++] = k;
	}
	while (l <= n) s->p[l++] = k;

	if (mm_is_symmetric (typecode)) {
		mm_real_set_symmetric (s);
		for (k = 0; k < s->nnz; k++) {
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

	k = 0;
	do {
		ret = fscanf (fp, "%lf", &d->data[k]);
		if (ret > 0 && ++k >= d->nnz) break;
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
	if (!x) error_and_exit ("mm_real_fread", "failed to read mm_real.", __FILE__, __LINE__);
	if (mm_real_is_symmetric (x) && x->m != x->n) error_and_exit ("mm_real_fread", "symmetric matrix must be square.", __FILE__, __LINE__);
	return x;
}

/* fwrite sparse */
static void
mm_real_fwrite_sparse (FILE *stream, const mm_sparse *s, const char *format)
{
	int		j;
	mm_write_banner (stream, s->typecode);
	mm_write_mtx_crd_size (stream, s->m, s->n, s->nnz);
	for (j = 0; j < s->n; j++) {
		int		k = s->p[j];
		int		pend = s->p[j + 1];
		for (; k < pend; k++) {
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
	for (k = 0; k < d->nnz; k++) {
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
