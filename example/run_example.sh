#!/bin/sh

# sample script of elasticnet
#
# a demonstration of pathwise cyclic coordinate descent regression of l1 regularized problem.
#
# USAGE:
# elasticnet -x <input file of matrix X> -y <input file of vector y> -a <alpha> 
# [optional] { -t <log10_lambda_min>:<d_log10_lambda>
#              -g <gamma of EBIC in [0, 1]> -m <maxiters> }
#
# -x <input file>    : specify MatrixMarket format file of matrix X
#
# -y <input file>    : specify MatrixMarket format file of vector y
#
# <optional>
#
# -a <alpha>         : raito of the weight for L1(alpha) and L2(1-alpha) norm penalty.
#                      if alpha == 1, regression type is lasso
#
# -r <log10_lambda_min>:<d_log10_lambda>
#				       : beta is calculated for each lambda (weight of L1 / L2 penalty)
#                      in the range of [10^log10_lambda_min : 10^d_log10_lambda : 10^log10_lambda_max],
#                      lambda_max = max( |c| ) / cd->alpha, where c = X' * y
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
# alpha = 1 (lasso), lambda = [0.01 : 10^0.1 : max( |c| )],
./elasticnet -x ../share/diabetes_x.data -y ../share/diabetes_y.data -a 1 -r -4:0.1 -m 100000

# display resultant solution path
# USAGE: plot_path <dim of beta> <alpha>
./plot_path 10 1

