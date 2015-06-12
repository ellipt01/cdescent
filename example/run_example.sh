#!/bin/sh

# sample script of elasticnet
#
# a demonstration of pathwise cyclic coordinate descent regression of l1 regularized problem.
#
# USAGE:
# elasticnet -x <input file of matrix X> -y <input file of vector y> -l <lambda2> 
# [optional] { -t <log10_lambda1_min>:<d_log10_lambda1>
#              -g <gamma of EBIC in [0, 1]> -m <maxiters> }
#
# -x <input file>    : specify MatrixMarket format file of matrix X
#
# -y <input file>    : specify MatrixMarket format file of vector y
#
# <optional>
#
# -l <lambda2>       : weight of L2 (elastic net) or other regralization penalty term.
#                      if lambda2 < machine_epsilon, it is regarded as 0.
#                      (default is 0)
#
# -r <log10l1_min>:<d_log10l1>
#				       : beta is calculated for each lambda1 (weight of L1 penalty)
#                      in the range of [10^log10l1_min : 10^d_log10l1 : 10^log10l1_max],
#                      where log10l1_max = cd->lreg->logcamax (see include linregmodel.h)
#                      (default is -2:0.1)
#
# -g <gamma>         : tunning parameter of extended BIC, in [0, 1] (see Chen and Chen, 2008).
#                      (default is 0)
#
# -t <tolerance>     : tolerance of convergence
#                      (default is 1.e-3)
# 
# -m <maxiters>      : max num of iterations
#                      (default is 100000)

# example for diabetes.data
# lambda2 = 0 (lasso), lambda1 = [0.01 : 10^0.1 : lambda1_max],
# where lambda1_max = 10^(cd->lreg->logcamax) 
./elasticnet -x ../share/diabetes_x.data -y ../share/diabetes_y.data -a 1 -r -4:0.1 -m 100000

# display resultant solution path
# USAGE: plot_path <dim of beta> <lambda2>
./plot_path 10 0

