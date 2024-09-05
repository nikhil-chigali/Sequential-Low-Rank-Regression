# Sequential-Low-Rank-Regression

Results and Plots can be found under [this folder](https://github.com/nikhil-chigali/Sequential-Low-Rank-Regression/tree/main/results/exps)

Notebook [here](https://github.com/nikhil-chigali/Sequential-Low-Rank-Regression/blob/main/sequential_low_rank_regression.ipynb)


Nikhil,
1. enhance the overleaf with citations, with correct notation that follows the notation we use the rest of the paper (e.g., boldface notation for matrices and vectors);

2.  as far as I understand, it is evident that training LoRA is not as simple as a linear regression problem (to start, we do not have a least-squares objective), but I assume the goal here is to study ways to enforce orthogonality among se-
quential rank-1 updates of the LoRA matrices. I.e., **Bn, An** should be actually vectors if we want to do rank-1, also it should be a summation but a single rank-1 update, and we should also consider sequentially training by adding more rank updates. 

1.  These should be tested not only for multiple tasks but also for a single task (which is easier to start with): e.g., we can focus on the vanilla LoRA that assumes eg., rank-8 and then try our idea with rank-1 updates until we get to rank-8 (by adding orthogonal or somewhat orthogonal subspaces) and then study whether these rank-1 updates are approximately solved (decreasing the number of training iterations) lead to degradation or so.
