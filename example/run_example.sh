#!/bin/sh

# sample script of l1reg_example
#
# a demonstration of pathwise and cyclic coordinate descent regression of l1 regularized problem.
#
# USAGE:
# l1reg_example -f <input_file>{:num skipheaders} -l <lambda2> 
# [optional] { -t <log10_lambda1_min>:<d_log10_lambda1>:<log10_lambda1_max>
#              -g <gamma of EBIC in [0, 1]> -m <maxiter> }
#
# -f <input file>:xx : specify input data file. first xx rows are regarded as comments
#                      and not read. 
# <optional>
#
# -l <lambda2>       : weight of L2 (elastic net) or other regralization penalty term.
#                      if lambda2 < machine_double_prec_eps, it is regarded as 0.
#                      (default is 0)
#
# -t <log10l1_min>:<d_log10l1>:<log10l1_max>
#				       : beta is calculated for each lambda1 (weight of L1 penalty)
#                      in the range of [10^log10l1_min : 10^d_log10l1 : 10^log10l1_max]
#                      (default is -2:0.1:5)
#
# -g <gamma>         : tunning parameter of extended BIC, in [0, 1] (see Chen and Chen, 2008).
#                      (default is 0)
#
# -m <maxiter>       : max num of iterations
#                      (default is 100000)

# example for diabetes.data
# lasso : lambda2 = 0, lambda1 = [0.01 : 10^0.1 : 10000] 
./l1reg_example -f ../share/diabetes.data:1 -l 0 -t -2:0.1:4 -m 100000

# display resultant solution path
# USAGE: plot_path <dim of beta> <lambda2>
../bin/plot_path 9 0

