/*
 * mmreal.c
 *
 *  Created on: 2014/06/25
 *      Author: utsugi
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include <mmreal.h>
#include "private.h"
#include "atomic.h"

/* mm_real supports real symmetric or general sparse, and real general dense matrix */
static bool
is_type_supported (MM_typecode typecode)
{
	// invalid type
	if (!mm_is_valid (typecode)) return false;

	// pattern is not supported
	if (mm_is_pattern (typecode)) return false;

	// integer and complex are not supported
	if (mm_is_integer (typecode) || mm_is_complex (typecode)) return false;

	// skew and hermitian are not supported
	if (mm_is_skew (typecode) || mm_is_hermitian (typecode)) return false;

	// for dense matrix, only general is supported
	if (mm_is_dense (typecode) && !mm_is_general (typecode)) return false;

	return true;
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

	mm->uplo = MM_REAL_FULL;

	/* set typecode = "M_RG" : Matrix Real General */
	// typdecode[3] = 'G' : General
	mm_initialize_typecode (&mm->typecode);
	// typecode[0] = 'M' : Matrix
	mm_set_matrix (&mm->typecode);
	// typecode[2] = 'R' : Real
	mm_set_real (&mm->typecode);

	return mm;
}

mm_real *
mm_real_new (MMRealFormat format, MMRealSymm symmetric, const int m, const int n, const int nz)
{
	mm_real	*mm;

	if (!(format == MM_REAL_SPARSE || format == MM_REAL_DENSE))
		error_and_exit ("mm_real_new", "format must be MM_REAL_SPARSE or MM_REAL_DENSE", __FILE__, __LINE__);
	if (format == MM_REAL_DENSE && (symmetric & MM_SYMMETRIC)) {
		printf_warning ("mm_real_new", "dense symmetric is not supported, assume dense unsymmetric.", __FILE__, __LINE__);
		symmetric = MM_REAL_GENERAL;
	}

	mm = mm_real_alloc ();
	mm->m = m;
	mm->n = n;
	mm->nz = nz;

	// typecode[1] = 'C' or 'A'
	if (format == MM_REAL_SPARSE) mm_set_coordinate (&mm->typecode);
	else mm_set_array (&mm->typecode);
	// typecode[3] = 'S'
	if (symmetric & MM_SYMMETRIC) {
		mm_set_symmetric (&mm->typecode);
		if (symmetric & MM_REAL_UPPER) mm_real_set_upper (mm);
		else if (symmetric & MM_REAL_LOWER) mm_real_set_lower (mm);
	}

	if (!is_type_supported (mm->typecode)) {
		char	msg[128];
		sprintf (msg, "matrix type does not supported :[%s].", mm_typecode_to_str (mm->typecode));
		error_and_exit ("mm_real_new", msg, __FILE__, __LINE__);
	}

	return mm;
}

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

/* reallocate mm_real */
bool
mm_real_realloc (mm_real *mm, const int nz)
{
	if (mm->nz == nz) return true;
	mm->nz = nz;
	mm->data = (double *) realloc (mm->data, nz * sizeof (double));
	if (mm->data == NULL) return false;
	if (mm_real_is_sparse (mm)) {
		mm->i = (int *) realloc (mm->i, nz * sizeof (int));
		if (mm->i == NULL) return false;
	}
	return true;
}

/* copy sparse */
static mm_sparse *
mm_real_copy_sparse (const mm_sparse *src)
{
	int			k;
	mm_sparse	*dest = mm_real_new (MM_REAL_SPARSE, mm_is_symmetric (src->typecode), src->m, src->n, src->nz);

	dest->i = (int *) malloc (src->nz * sizeof (int));
	dest->p = (int *) malloc ((src->n + 1) * sizeof (int));
	dest->data = (double *) malloc (src->nz * sizeof (double));
	dest->uplo = src->uplo;

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
	mm_dense	*dest = mm_real_new (MM_REAL_DENSE, false, src->m, src->n, src->nz);
	dest->data = (double *) malloc (src->nz * sizeof (double));
	dcopy_ (&src->nz, src->data, &ione, dest->data, &ione);
	return dest;
}

/*** copy x ***/
mm_real *
mm_real_copy (const mm_real *src)
{
	return (mm_real_is_sparse (src)) ? mm_real_copy_sparse (src) : mm_real_copy_dense (src);
}

/*** set all array to val ***/
void
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

/*** replace sparse -> dense ***/
bool
mm_real_replace_sparse_to_dense (mm_real *x)
{
	int		j;
	double	*data = x->data;
	if (!mm_is_sparse (x->typecode)) return false;

	mm_set_dense (&x->typecode);
	x->nz = x->m * x->n;
	x->data = (double *) malloc (x->nz * sizeof (double));
	mm_real_set_all (x, 0.);

	for (j = 0; j < x->n; j++) {
		int		k;
		for (k = x->p[j]; k < x->p[j + 1]; k++) {
			int		i = x->i[k];
			x->data[i + j * x->m] = data[k];
			if (mm_real_is_symmetric (x)) x->data[j + i * x->m] = data[k];
		}
	}
	free (data);

	if (x->i) free (x->i);
	x->i = NULL;
	if (x->p) free (x->p);
	x->p = NULL;
	mm_set_general (&x->typecode);
	mm_real_set_full (x);

	return true;
}

/*** replace dense -> sparse ***/
/* fabs (x->data[j]) < threshold are set to 0 */
bool
mm_real_replace_dense_to_sparse (mm_real *x, const double threshold)
{
	int		i, j, k;
	double	*data = x->data;
	if (!mm_is_dense (x->typecode)) return false;

	mm_set_sparse (&x->typecode);
	if (x->i) free (x->i);
	x->i = (int *) malloc (x->nz * sizeof (int));
	if (x->p) free (x->p);
	x->p = (int *) malloc ((x->n + 1) * sizeof (int));
	x->data = (double *) malloc (x->nz * sizeof (double));

	k = 0;
	x->p[0] = 0;
	for (j = 0; j < x->n; j++) {
		for (i = 0; i < x->m; i++) {
			double	dij = data[i + j * x->m];
			if (fabs (dij) >= threshold) {
				x->i[k] = i;
				x->data[k] = dij;
				k++;
			}
		}
		x->p[j + 1] = k;
	}
	free (data);
	if (x->nz != k) return mm_real_realloc (x, k);
	return true;
}

/* identity sparse matrix */
static mm_sparse *
mm_real_seye (const int n)
{
	int			k;
	mm_sparse	*s = mm_real_new (MM_REAL_SPARSE, false, n, n, n);
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
	mm_dense	*d = mm_real_new (MM_REAL_DENSE, false, n, n, n * n);
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
mm_real_sj_asum (const int j, const mm_sparse *s)
{
	int		size = s->p[j + 1] - s->p[j];
	double	asum = dasum_ (&size, s->data + s->p[j], &ione);
	if (mm_real_is_symmetric (s)) {
		int		l;
		int		l0 = (s->uplo == MM_REAL_UPPER) ? j + 1 : 0;
		int		l1 = (s->uplo == MM_REAL_UPPER) ? s->n : j;
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
mm_real_dj_asum (const int j, const mm_dense *d)
{
	return dasum_ (&d->m, d->data + j * d->m, &ione);
}

/*** sum |x(:,j)| ***/
double
mm_real_xj_asum (const int j, const mm_real *x)
{
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_asum", "index out of range.", __FILE__, __LINE__);
	return (mm_real_is_sparse (x)) ? mm_real_sj_asum (j, x) : mm_real_dj_asum (j, x);
}

/* sum s(:,j) */
static double
mm_real_sj_sum (const int j, const mm_sparse *s)
{
	int		k;
	double	sum = 0.;
	for (k = s->p[j]; k < s->p[j + 1]; k++) sum += s->data[k];
	if (mm_real_is_symmetric (s)) {
		int		l;
		int		l0 = (s->uplo == MM_REAL_UPPER) ? j + 1 : 0;
		int		l1 = (s->uplo == MM_REAL_UPPER) ? s->n : j;
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
mm_real_dj_sum (const int j, const mm_dense *d)
{
	int		k;
	double	sum = 0.;
	for (k = 0; k < d->m; k++) sum += d->data[k + j * d->m];
	return sum;
}

/*** sum x(:,j) ***/
double
mm_real_xj_sum (const int j, const mm_real *x)
{
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_sum", "index out of range.", __FILE__, __LINE__);
	return (mm_real_is_sparse (x)) ? mm_real_sj_sum (j, x) : mm_real_dj_sum (j, x);
}

/* norm2 s(:,j) */
static double
mm_real_sj_nrm2 (const int j, const mm_sparse *s)
{
	int		size = s->p[j + 1] - s->p[j];
	double	nrm2 = dnrm2_ (&size, s->data + s->p[j], &ione);
	if (mm_real_is_symmetric (s)) {
		double	sum = pow (nrm2, 2.);
		int		l;
		int		l0 = (s->uplo == MM_REAL_UPPER) ? j + 1 : 0;
		int		l1 = (s->uplo == MM_REAL_UPPER) ? s->n : j;
		for (l = l0; l < l1; l++) {
			int		k;
			for (k = s->p[l]; k < s->p[l + 1]; k++) {
				if (s->i[k] == j) {
					sum += pow (s->data[k], 2.);
					break;
				}
			}
		}
		nrm2 = sqrt (sum);
	}
	return nrm2;
}

/* norm2 d(:,j) */
static double
mm_real_dj_nrm2 (const int j, const mm_dense *d)
{
	return dnrm2_ (&d->m, d->data + j * d->m, &ione);
}

/*** norm2 x(:,j) ***/
double
mm_real_xj_nrm2 (const int j, const mm_real *x)
{
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_nrm2", "index out of range.", __FILE__, __LINE__);
	return (mm_real_is_sparse (x)) ? mm_real_sj_nrm2 (j, x) : mm_real_dj_nrm2 (j, x);
}

/* s * y, where s is sparse matrix and y is dense vector */
static mm_dense *
mm_real_s_dot_y (bool trans, const double alpha, const mm_sparse *s, const mm_dense *y, const double beta)
{
	int			j;
	int			m;
	mm_dense	*d;
	m = (trans) ? s->n : s->m;
	
	d = mm_real_new (MM_REAL_DENSE, false, m, 1, m);
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

	c = mm_real_new (MM_REAL_DENSE, false, m, 1, m);
	c->data = (double *) malloc (c->nz * sizeof (double));

	dgemv_ ((trans) ? "T" : "N", &d->m, &d->n, &alpha, d->data, &d->m, y->data, &ione, &beta, c->data, &ione);

	return c;
}

/*** x * y, where x is matrix and y is dense vector ***/
mm_dense *
mm_real_x_dot_y (bool trans, const double alpha, const mm_real *x, const mm_dense *y, const double beta)
{
	if (!mm_is_dense (y->typecode)) error_and_exit ("mm_real_x_dot_y", "y must be dense.", __FILE__, __LINE__);
	if (y->n != 1) error_and_exit ("mm_real_x_dot_y", "y must be vector.", __FILE__, __LINE__);
	if ((trans && x->m != y->m) || (!trans && x->n != y->m))
		error_and_exit ("mm_real_x_dot_y", "vector and matrix dimensions do not match.", __FILE__, __LINE__);

	return (mm_real_is_sparse (x)) ? mm_real_s_dot_y (trans, alpha, x, y, beta) : mm_real_d_dot_y (trans, alpha, x, y, beta);
}

/* s(:,j)' * y */
static double
mm_real_sj_trans_dot_y (const int j, const mm_sparse *s, const mm_dense *y)
{
	int		k;
	double	val = 0;
	for (k = s->p[j]; k < s->p[j + 1]; k++) val += s->data[k] * y->data[s->i[k]];
	if (mm_real_is_symmetric (s)) {
		int		l;
		int		l0 = (s->uplo == MM_REAL_UPPER) ? j + 1 : 0;
		int		l1 = (s->uplo == MM_REAL_UPPER) ? s->n : j;
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
mm_real_dj_trans_dot_y (const int j, const mm_dense *d, const mm_dense *y)
{
	return ddot_ (&d->m, d->data + j * d->m, &ione, y->data, &ione);
}

/*** x(:,j)' * y ***/
double
mm_real_xj_trans_dot_y (const int j, const mm_real *x, const mm_dense *y)
{
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_xj_trans_dot_y", "index out of range.", __FILE__, __LINE__);
	if (!mm_is_dense (y->typecode))
		error_and_exit ("mm_real_xj_trans_dot_y", "y must be dense.", __FILE__, __LINE__);
	if (y->n != 1) error_and_exit ("mm_real_xj_trans_dot_y", "y must be vector.", __FILE__, __LINE__);
	if (x->m != y->m) error_and_exit ("mm_real_xj_trans_dot_y", "vector and matrix dimensions do not match.", __FILE__, __LINE__);

	return (mm_real_is_sparse (x)) ? mm_real_sj_trans_dot_y (j, x, y) : mm_real_dj_trans_dot_y (j, x, y);
}

/* y = alpha * s(:,j) + y */
static void
mm_real_asjpy (const double alpha, const int j, const mm_sparse *s, mm_dense *y)
{
	int		k;
	for (k = s->p[j]; k < s->p[j + 1]; k++) y->data[s->i[k]] += alpha * s->data[k];
	if (mm_real_is_symmetric (s)) {
		int		l;
		int		l0 = (s->uplo == MM_REAL_UPPER) ? j + 1 : 0;
		int		l1 = (s->uplo == MM_REAL_UPPER) ? s->n : j;
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
mm_real_adjpy (const double alpha, const int j, const mm_dense *d, mm_dense *y)
{
	daxpy_ (&d->m, &alpha, d->data + j * d->m, &ione, y->data, &ione);
	return;
}

/*** y = alpha * x(:,j) + y ***/
void
mm_real_axjpy (const double alpha, const int j, const mm_real *x, mm_dense *y)
{
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_axjpy", "index out of range.", __FILE__, __LINE__);
	if (!mm_is_dense (y->typecode)) error_and_exit ("mm_real_axjpy", "y must be dense.", __FILE__, __LINE__);
	if (y->n != 1) error_and_exit ("mm_real_axjpy", "y must be vector.", __FILE__, __LINE__);
	if (x->m != y->m) error_and_exit ("mm_real_axjpy", "vector and matrix dimensions do not match.", __FILE__, __LINE__);

	return (mm_real_is_sparse (x)) ? mm_real_asjpy (alpha, j, x, y) : mm_real_adjpy (alpha, j, x, y);
}

/* y = alpha * s(:,j) + y, atomic */
static void
mm_real_asjpy_atomic (const double alpha, const int j, const mm_sparse *s, mm_dense *y)
{
	int		k;
	for (k = s->p[j]; k < s->p[j + 1]; k++) atomic_add (y->data + s->i[k], alpha * s->data[k]);
	if (mm_real_is_symmetric (s)) {
		int		l;
		int		l0 = (s->uplo == MM_REAL_UPPER) ? j + 1 : 0;
		int		l1 = (s->uplo == MM_REAL_UPPER) ? s->n : j;
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
mm_real_adjpy_atomic (const double alpha, const int j, const mm_dense *d, mm_dense *y)
{
	int		k;
	double	*dj = d->data + j * d->m;
	for (k = 0; k < d->m; k++) atomic_add (y->data + k, alpha * dj[k]);
	return;
}

/*** y = alpha * x(:,j) + y, atomic ***/
void
mm_real_axjpy_atomic (const double alpha, const int j, const mm_real *x, mm_dense *y)
{
	if (j < 0 || x->n <= j) error_and_exit ("mm_real_axjpy_atomic", "index out of range.", __FILE__, __LINE__);
	if (!mm_is_dense (y->typecode)) error_and_exit ("mm_real_axjpy_atomic", "y must be dense.", __FILE__, __LINE__);
	if (y->n != 1) error_and_exit ("mm_real_axjpy_atomic", "y must be vector.", __FILE__, __LINE__);
	if (x->m != y->m) error_and_exit ("mm_real_axjpy_atomic", "vector and matrix dimensions do not match.", __FILE__, __LINE__);

	return (mm_real_is_sparse (x)) ? mm_real_asjpy_atomic (alpha, j, x, y) : mm_real_adjpy_atomic (alpha, j, x, y);
}

/* fread sparse */
static mm_sparse *
mm_real_fread_sparse (FILE *fp, MM_typecode typecode)
{
	int			k;
	int			m, n, nz;
	int			*j;
	mm_sparse	*s;

	mm_read_mtx_crd_size (fp, &m, &n, &nz);
	s = mm_real_new (MM_REAL_SPARSE, mm_is_symmetric (typecode), m, n, nz);
	s->i = (int *) malloc (s->nz * sizeof (int));
	s->data = (double *) malloc (s->nz * sizeof (double));
	s->p = (int *) malloc ((s->n + 1) * sizeof (int));

	j = (int *) malloc (s->nz * sizeof (int));
	mm_read_mtx_crd_data (fp, s->m, s->n, s->nz, s->i, j, s->data, typecode);

	s->p[0] = 0;
	for (k = 0; k < s->nz; k++) {
		s->i[k]--;		// fortran -> c
		if (k > 0 && j[k] != j[k - 1]) s->p[j[k] - 1] = k;
	}
	s->p[n] = k;

	for (k = 0; k < s->nz; k++) {
		if (s->i[k] == j[k]) continue;
		(s->i[k] < j[k]) ? mm_real_set_upper (s) : mm_real_set_lower (s);
		break;
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
	d = mm_real_new (MM_REAL_DENSE, false, m, n, m * n);
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
	mm_read_banner (fp, &typecode);
	if (!is_type_supported (typecode)) {
		char	msg[128];
		sprintf (msg, "matrix type does not supported :[%s].", mm_typecode_to_str (typecode));
		error_and_exit ("mm_real_fread", msg, __FILE__, __LINE__);
	}
	return (mm_is_sparse (typecode)) ? mm_real_fread_sparse (fp, typecode) : mm_real_fread_dense (fp, typecode);
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
	return (mm_is_sparse (x->typecode)) ? mm_real_fwrite_sparse (stream, x, format) : mm_real_fwrite_dense (stream, x, format);
}
