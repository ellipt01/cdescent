#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include <mmreal.h>
#include "private.h"
#include "atomic.h"

/* allocate mm_real */
mm_real *
mm_real_alloc (void)
{
	mm_real	*mm = (mm_real *) malloc (sizeof (mm_real));
	mm->m = 0;
	mm->n = 0;
	mm->nz = 0;
	mm->i = NULL;
	mm->p = NULL;
	mm->data = NULL;

	// typdecode[3] = 'G'
	mm_initialize_typecode (&mm->typecode);
	// typecode[2] = 'R'
	mm_set_real (&mm->typecode);

	return mm;
}

mm_real *
mm_real_new (MMRealFormat format, bool symmetric, const int m, const int n, const int nz)
{
	mm_real	*mm = mm_real_alloc ();

	if (!(format == MM_REAL_SPARSE || format == MM_REAL_DENSE))
		error_and_exit ("mm_real_new", "format must be MM_REAL_SPARSE or DENSE.", __FILE__, __LINE__);

	mm->m = m;
	mm->n = n;
	mm->nz = nz;

	// typecode[3] = 'G'
	mm_initialize_typecode (&mm->typecode);

	// typecode[0] = 'M'
	mm_set_matrix (&mm->typecode);

	// typecode[1] = 'C' or 'A'
	if (format == MM_REAL_SPARSE) mm_set_coordinate (&mm->typecode);
	else mm_set_array (&mm->typecode);

	// typecode[2] = 'R'
	mm_set_real (&mm->typecode);

	// typecode[3] = 'S'
	if (symmetric) mm_set_symmetric (&mm->typecode);

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
void
mm_real_replace_sparse_to_dense (mm_real *x)
{
	int		j;
	double	*data;
	if (mm_is_dense (x->typecode)) return;

	data = (double *) malloc (x->nz * sizeof (double));
	dcopy_ (&x->nz, x->data, &ione, data, &ione);

	mm_set_dense (&x->typecode);
	x->nz = x->m * x->n;
	x->data = (double *) realloc (x->data, x->nz * sizeof (double));
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
	free (x->i);
	x->i = NULL;
	free (x->p);
	x->p = NULL;
	return;
}

/*** replace dense -> sparse ***/
/* the elements of x->data that fabs (x->data[j]) < threshold are set to 0 */
void
mm_real_replace_dense_to_sparse (mm_real *x, const double threshold)
{
	int		i, j, k;
	double	*data;
	if (!mm_is_dense (x->typecode)) return;

	data = (double *) malloc (x->nz * sizeof (double));
	dcopy_ (&x->nz, x->data, &ione, data, &ione);

	mm_set_sparse (&x->typecode);
	x->i = (int *) malloc (x->nz * sizeof (int));
	x->p = (int *) malloc ((x->n + 1) * sizeof (int));

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
	mm_real_realloc (x, k);
	return;
}

/* identical sparse matrix */
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

/* identical dense matrix */
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

/*** n x n identical matrix ***/
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
		int		k, l;
		for (l = 0; l < j; l++) {
			for (k = s->p[l]; k < s->p[l + 1]; k++) {
				if (s->i[k] < j) continue;
				if (s->i[k] == j) asum += fabs (s->data[k]);
				break;
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
		for (l = 0; l < j; l++) {
			for (k = s->p[l]; k < s->p[l + 1]; k++) {
				if (s->i[k] < j) continue;
				if (s->i[k] == j) sum += s->data[k];
				break;
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
	for (k = 0; k < d->m; k++) sum += d->data[k];
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
		int		k, l;
		double	sum = pow (nrm2, 2.);
		for (l = 0; l < j; l++) {
			for (k = s->p[l]; k < s->p[l + 1]; k++) {
				if (s->i[k] < j) continue;
				if (s->i[k] == j) sum += pow (s->data[k], 2.);
				break;
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
	int			j, k;
	int			m, n;
	mm_dense	*d;
	m = (trans) ? s->n : s->m;
	n = (trans) ? s->m : s->n;
	
	d = mm_real_new (MM_REAL_DENSE, false, m, 1, m);
	d->data = (double *) malloc (d->nz * sizeof (double));
	mm_real_set_all (d, beta);

	for (j = 0; j < s->n; j++) {
		for (k = s->p[j]; k < s->p[j + 1]; k++) {
			int	si = (trans) ? j : s->i[k];
			int	sj = (trans) ? s->i[k] : j;
			d->data[si] += s->data[k] * y->data[sj];
			if (mm_real_is_symmetric (s) && j < s->i[k]) {
				d->data[sj] += s->data[k] * y->data[si];
			}
		}		
	}
	return d;
}

/* d * y, where d is dense matrix and y is dense vector */
static mm_dense *
mm_real_d_dot_y (bool trans, const double alpha, const mm_dense *d, const mm_dense *y, const double beta)
{
	int			m, n;
	mm_dense	*c;
	m = (trans) ? d->n : d->m;
	n = (trans) ? d->m : d->n;

	c = mm_real_new (MM_REAL_DENSE, false, m, 1, m);
	c->data = (double *) malloc (c->nz * sizeof (double));

	dgemv_ ((trans) ? "T" : "N", &d->m, &d->n, &alpha, d->data, &d->m, y->data, &ione, &beta, c->data, &ione);

	return c;
}

/*** x * y, where x is matrix and y is dense vector ***/
mm_real *
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
		for (l = 0; l < j; l++) {
			for (k = s->p[l]; k < s->p[l + 1]; k++) {
				if (s->i[k] < j) continue;
				if (s->i[k] == j) val += s->data[k] * y->data[l];
				break;
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
		for (l = 0; l < j; l++) {
			for (k = s->p[l]; k < s->p[l + 1]; k++) {
				if (s->i[k] < j) continue;
				if (s->i[k] == j) y->data[l] += alpha * s->data[k];
				break;
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
		for (l = 0; l < j; l++) {
			for (k = s->p[l]; k < s->p[l + 1]; k++) {
				if (s->i[k] < j) continue;
				if (s->i[k] == j) atomic_add (y->data + l, alpha * s->data[k]);
				break;
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
