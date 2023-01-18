import math as mt

import numpy as np


def Plots(b_points=10000, l=2, l_sigma=0.6, L_star=2, beta_min=0.1, tol=0.000001):
    def Var(l_lambda, l, l_sigma, beta):
        delta = (l - l_lambda - 1) / beta - mt.floor((l - l_lambda - 1) / beta)
        return (-l_lambda - l_sigma - beta * min(delta, 1 - delta)) / l

    def Bias(l_lambda, l, beta, L_star):
        return -2.0 - 2.0 / l * (-l_lambda - 1.0 - beta * (L_star - 1))

    def Optimize(L_n, b, L_gt):
        lower_bound = -Bias(0, L_n, b, L_gt)
        L = lower_bound / 2
        jump = lower_bound / 4
        while jump >= tol * 2:
            obj = Bias(L, L_n, b, L_gt) - Var(L, L_n, l_sigma, b)
            L = L - jump * (2 * (obj > 0) - 1)
            jump = jump / 2
        return L

    bs = np.linspace(beta_min, 1, b_points + 1)[2:-1]
    var_s = np.zeros(bs.shape[0])
    bias_s = np.zeros(bs.shape[0])

    L_opt = np.zeros(bs.shape[0])
    var_s_opt = np.zeros(bs.shape[0])
    bias_s_opt = np.zeros(bs.shape[0])

    for i, b in enumerate(bs):
        var_s[i] = Var(l_lambda=0, l=l, l_sigma=l_sigma, beta=b)
        bias_s[i] = Bias(l_lambda=0, l=l, beta=b, L_star=L_star)

        L_opt[i] = Optimize(l, b, L_star)

        var_s_opt[i] = Var(L_opt[i], l, l_sigma, b)
        bias_s_opt[i] = Bias(L_opt[i], l, b, L_star)

    th = 0
    for (i, b) in enumerate(bs):
        if bias_s[i] >= var_s[i]:
            th = bs[i]
            break

    rate_s = np.max(np.vstack((var_s, bias_s)), axis=0)
    rate_s_opt = np.max(np.vstack((var_s_opt, bias_s_opt)), axis=0)

    return th, bs, rate_s, rate_s_opt, var_s, bias_s
