/*
 * bic.h
 *
 *  Created on: 2015/03/13
 *      Author: utsugi
 */

#ifndef BIC_H_
#define BIC_H_

#ifdef __cplusplus
extern "C" {
#endif

/* bic.c */
bic_func	*bic_function_new (const bic_eval_func function, void *data);

#ifdef __cplusplus
}
#endif

#endif /* BIC_H_ */
