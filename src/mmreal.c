#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include <mmreal.h>
#include "private.h"

mm_real *
mm_real_alloc (void)
{
	mm_real	*mm = (mm_real *) malloc (sizeof (mm_real));
	mm->m = 0;
	mm->n = 0;
	mm->nz = 0;
	mm->i = NULL;
	mm->j = NULL;
	mm->p = NULL;
	mm->data = NULL;

	// typdecode[3] = 'G'
	mm_initialize_typecode (&mm->typecode);
	// typecode[2] = 'R'
	mm_set_real (&mm->typecode);

	return mm;
}

mm_real *
mm_real_new (MM_RealType type, MM_RealSymmetric symmetric, const int m, const int n, const int nz)
{
	mm_real	*mm = mm_real_alloc ();

	mm->m = m;
	mm->n = n;
	mm->nz = nz;

	// typecode[3] = 'G'
	mm_initialize_typecode (&mm->typecode);

	// typecode[0] = 'M'
	mm_set_matrix (&mm->typecode);

	// typecode[1] = 'C' or 'A'
	if (type == MM_REAL_SPARSE) mm_set_coordinate (&mm->typecode);
	else mm_set_array (&mm->typecode);

	// typecode[2] = 'R'
	mm_set_real (&mm->typecode);

	// typecode[3] = 'S'
	if (symmetric == MM_REAL_SYMMETRIC) mm_set_symmetric (&mm->typecode);

	return mm;
}

void
mm_real_free (mm_real *mm)
{
	if (mm) {
		if (mm->i) free (mm->i);
		if (mm->j) free (mm->j);
		if (mm->p) free (mm->p);
		if (mm->data) free (mm->data);
		free (mm);
	}
	return;
}

bool
mm_real_realloc (mm_real *mm, const int nz)
{
	if (!mm) cdescent_error ("mm_realloc", "input matrix is empty.", __FILE__, __LINE__);
	if (mm->nz == nz) return true;
	mm->nz = nz;
	mm->data = (double *) realloc (mm->data, mm->nz * sizeof (double));
	if (mm->data == NULL) return false;
	if (mm_is_sparse (mm->typecode)) {
		mm->i = (int *) realloc (mm->i, mm->nz * sizeof (int));
		mm->j = (int *) realloc (mm->j, mm->nz * sizeof (int));
		if (mm->i == NULL || mm->j == NULL) return false;
	}
	return true;
}

static mm_real *
mm_real_copy_sparse (const mm_real *src)
{
	int					k;
	MM_RealSymmetric	symmetric = (mm_is_symmetric (src->typecode)) ? MM_REAL_SYMMETRIC : MM_REAL_UNSYMMETRIC;
	mm_real			*dest = mm_real_new (MM_REAL_SPARSE, symmetric, src->m, src->n, src->nz);

	dest->i = (int *) malloc (src->nz * sizeof (int));
	dest->j = (int *) malloc (src->nz * sizeof (int));
	dest->p = (int *) malloc ((src->n + 1) * sizeof (int));
	dest->data = (double *) malloc (src->nz * sizeof (double));

	for (k = 0; k < src->nz; k++) {
		dest->i[k] = src->i[k];
		dest->j[k] = src->j[k];
		dest->data[k] = src->data[k];
	}
	for (k = 0; k <= src->n; k++)	dest->p[k] = src->p[k];

	return dest;
}

static mm_real *
mm_real_copy_dense (const mm_real *src)
{
	int		k;
	mm_real	*dest = mm_real_new (MM_REAL_DENSE, MM_REAL_UNSYMMETRIC, src->m, src->n, src->nz);
	dest->data = (double *) malloc (src->nz * sizeof (double));
	for (k = 0; k < src->nz; k++) dest->data[k] = src->data[k];
	return dest;
}

mm_real *
mm_real_copy (const mm_real *src)
{
	if (!src) cdescent_error ("mm_real_copy", "input matrix is empty.", __FILE__, __LINE__);
	return (mm_is_sparse (src->typecode)) ? mm_real_copy_sparse (src) : mm_real_copy_dense (src);
}

void
mm_array_set_all (int n, double *data, const double val)
{
	while (n-- >= 0) *data++ = val;
}

void
mm_real_set_all (mm_real *mm, const double val)
{
	mm_array_set_all (mm->nz, mm->data, val);
}

void
mm_real_replace_sparse_to_dense (mm_real *x)
{
	int		k;
	double	*data;
	if (!mm_is_sparse (x->typecode)) return;

	data = (double *) malloc (x->nz * sizeof (double));
	dcopy_ (&x->nz, x->data, &ione, data, &ione);

	mm_set_dense (&x->typecode);
	x->data = (double *) realloc (x->data, x->m * x->n * sizeof (double));
	mm_array_set_all (x->nz, x->data, 0.);

	for (k = 0; k < x->nz; k++) {
		int		i = x->i[k];
		int		j = x->j[k];
		x->data[i + j * x->m] = data[k];
	}
	free (data);
	free (x->i);
	x->i = NULL;
	free (x->j);
	x->j = NULL;
	free (x->p);
	x->p = NULL;
	return;
}

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
	x->j = (int *) malloc (x->nz * sizeof (int));
	x->p = (int *) malloc ((x->n + 1) * sizeof (int));
	x->data = (double *) malloc (x->nz * sizeof (double));

	k = 0;
	x->p[0] = 0;
	for (j = 0; j < x->n; j++) {
		for (i = 0; i < x->m; i++) {
			double	dij = data[i + j * x->m];
			if (fabs (dij) >= threshold) {
				x->i[k] = i;
				x->j[k] = j;
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

static mm_real *
mm_real_speye (const int n)
{
	int		k;
	mm_real	*mm = mm_real_new (MM_REAL_SPARSE, MM_REAL_SYMMETRIC, n, n, n);

	mm->i = (int *) malloc (n * sizeof (int));
	mm->j = (int *) malloc (n * sizeof (int));
	mm->data = (double *) malloc (n * sizeof (double));
	mm->p = (int *) malloc ((n + 1) * sizeof (int));

	mm->p[0] = 0;
	for (k = 0; k < n; k++) {
		mm->i[k] = k;
		mm->j[k] = k;
		mm->data[k] = 1.;
		mm->p[k + 1] = k + 1;
	}

	return mm;
}

mm_real *
mm_real_eye (MM_RealType type, const int n)
{
	mm_real	*mm;

	if (type == MM_REAL_SPARSE) mm = mm_real_speye (n);
	else {
		int		k;
		mm = mm_real_new (MM_REAL_DENSE, MM_REAL_UNSYMMETRIC, n, n, n * n);
		mm->data = (double *) malloc (mm->nz * sizeof (double));
		mm_real_set_all (mm, 0.);
		for (k = 0; k < n; k++) mm->data[k + k * n] = 1.;
	}

	return mm;
}

/* sum_i S(i) */
double
mm_real_sum (const mm_real *x)
{
	int		k;
	double	sum = 0.;
	for (k = 0; k < x->nz; k++) sum += x->data[k];
	return sum;
}

/* |S| */
double
mm_real_asum (const mm_real *x)
{
	return dasum_ (&x->nz, x->data, &ione);
}

/* sum_i S(i,j) */
static double
mm_real_sj_sum (const int j, const mm_sparse *s)
{
	int		k;
	double	sum = 0.;
	if (j < 0 || s->n <= j) cdescent_error ("mm_real_sj_sum", "specified index is invalid", __FILE__, __LINE__);
	for (k = s->p[j]; k < s->p[j + 1]; k++) sum += s->data[k];
	return sum;
}

/* sum_i X(i,j) */
double
mm_real_xj_sum (const int j, const mm_real *x)
{
	double	sum = 0.;
	if (mm_is_sparse (x->typecode)) sum = mm_real_sj_sum (j, x);
	else {
		int		k;
		for (k = 0; k < x->m; k++) sum += x->data[k + j * x->m];
	}
	return sum;
}

/* ||S|| */
double
mm_real_nrm2 (const mm_real *x)
{
	return dnrm2_ (&x->nz, x->data, &ione);
}

/* ||S(:,j)|| */
static double
mm_real_sj_nrm2 (const int j, const mm_sparse *s)
{
	int		size = s->p[j + 1] - s->p[j];
	return dnrm2_ (&size, s->data + s->p[j], &ione);
}

/* ||X(:,j)|| */
double
mm_real_xj_nrm2 (const int j, const mm_real *x)
{
	if (j < 0 || x->n <= j) cdescent_error ("mm_real_nrm2", "index out of range.", __FILE__, __LINE__);
	return (mm_is_sparse (x->typecode)) ? mm_real_sj_nrm2 (j, x) : dnrm2_ (&x->m, x->data + j * x->m, &ione);
}

/* S * D or S' * D */
static mm_real *
mm_real_s_dot_d (bool trans, const double alpha, const mm_sparse *s, const mm_dense *d, const double beta)
{
	int		j, k, l;
	int		m, n, lda;
	bool	symmetric;
	mm_real	*mm;

	m = (trans) ? s->n : s->m;
	n = (trans) ? s->m : s->n;
	if (n != d->m) cdescent_error ("mm_real_s_dot_d", "matrix size not match.", __FILE__, __LINE__);

	lda = (trans) ? s->n : s->m;

	mm = mm_real_new (false, false, m, d->n, m * d->n);

	symmetric = mm_is_symmetric (s->typecode);

	mm->data = (double *) malloc (mm->nz * sizeof (double));
	mm_real_set_all (mm, beta);

	for (l = 0; l < d->n; l++) {
		for (j = 0; j < s->n; j++) {
			for (k = s->p[j]; k < s->p[j + 1]; k++) {
				int		si = (trans) ? s->i[k] : s->j[k];
				int		sj = (trans) ? s->j[k] : s->i[k];
				mm->data[sj + l * lda] += alpha * s->data[k] * d->data[si + l * d->m];
				if (symmetric && j != sj) mm->data[si + l * lda] += alpha * s->data[k] * d->data[sj + l * d->m];
			}
		}
	}
	return mm;
}

/* X * y or X' * y */
mm_real *
mm_real_x_dot_y (bool trans, const double alpha, const mm_real *x, const mm_dense *y, const double beta)
{
	int		m = (trans) ? x->n : x->m;
	int		n = (trans) ? x->m : x->n;
	mm_real	*c;

	if (n != y->m) cdescent_error ("mm_real_x_dot_y", "matrix size not match.", __FILE__, __LINE__);
	if (!mm_is_dense (y->typecode)) cdescent_error ("mm_real_x_dot_y", "matrix *y must be dense.", __FILE__, __LINE__);

	c = mm_real_new (MM_REAL_DENSE, MM_REAL_UNSYMMETRIC, m, y->n, m * y->n);
	if (mm_is_sparse (x->typecode)) c = mm_real_s_dot_d (trans, alpha, x, y, beta);
	else {
		c->data = (double *) malloc (m * y->n * sizeof (double));
		dgemv_ ((trans) ? "T" : "N", &x->m, &x->n, &alpha, x->data, &x->m, y->data, &ione, &beta, c->data, &ione);
	}
	return c;
}

/* S(:,j)' * D */
static double
mm_real_sj_trans_dot_d (const int j, const mm_sparse *s, const mm_dense *d)
{
	int		k;
	double	val;

	val = 0;
	for (k = s->p[j]; k < s->p[j + 1]; k++) val += s->data[k] * d->data[s->i[k]];
	return val;
}

/* X(:,j)' * y */
double
mm_real_xj_trans_dot_y (const int j, const mm_real *x, const mm_dense *y)
{
	double	val;

	if (j < 0 || x->n <= j) cdescent_error ("mm_real_xj_trans_dot_y", "index out of range.", __FILE__, __LINE__);
	if (!mm_is_dense (y->typecode)) cdescent_error ("mm_real_xj_trans_dot_y", "matrix *y must be dense.", __FILE__, __LINE__);

	if (mm_is_sparse (x->typecode)) val = mm_real_sj_trans_dot_d (j, x, y);
	else val = ddot_ (&x->m, x->data + j * x->m, &ione, y->data, &ione);

	return val;
}

/* d += a * S(:,j) */
static void
mm_real_asjpd (const double alpha, const int j, const mm_sparse *s, mm_dense *d)
{
	int		k;
	for (k = s->p[j]; k < s->p[j + 1]; k++) d->data[s->i[k]] += alpha * s->data[k];
	return;
}

/* y += a * X(:,j) */
void
mm_real_axjpy (const double alpha, const int j, const mm_real *x, mm_dense *y)
{
	if (!mm_is_dense (y->typecode)) cdescent_error ("mm_real_axjpy", "matrix *y must be dense.", __FILE__, __LINE__);

	if (mm_is_sparse (x->typecode)) mm_real_asjpd (alpha, j, x, y);
	else daxpy_ (&y->m, &alpha, x->data + j * x->m, &ione, y->data, &ione);

	return;
}
