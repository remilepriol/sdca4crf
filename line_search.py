import numpy as np
import tensorboard_logger as tl


class LineSearch:

    def __init__(self, weights, primal_direction, dual_direction, alpha_i, beta_i,
                 divergence_gap, regu, ntrain, fixed_step_size, monitor_dual_objective, i):

        self.sample_id = i
        self.alpha_i = alpha_i
        self.beta_i = beta_i
        self.newmarg = None
        self.dual_direction = dual_direction  # TODO use log(d) instead of d.
        self.fixed_step_size = fixed_step_size

        # values for the derivative of the entropy
        self.divergence_gap = divergence_gap
        self.reverse_gap = self.beta_i.kullback_leibler(alpha_i)

        # values for the quadratic term
        self.primaldir_squared_norm = primal_direction.squared_norm()
        self.weights_dot_primaldir = weights.inner_product(primal_direction)
        self.quadratic_coeff = - regu * ntrain / 2 * self.primaldir_squared_norm
        self.linear_coeff = - regu * ntrain * self.weights_dot_primaldir

        # to update the value of the dual objective on the fly without redundant computations
        self.monitor_dual_objective = monitor_dual_objective

        # return values
        self.optimal_step_size = 0
        self.subobjectives = []

    def evaluator(self, step_size, return_f=False, return_df=False, return_newton=False):
        """Return the line search function f and its derivative df evaluated in the step
        size step_size. Because the second derivative can be insanely large, we return instead the
        Newton step f'(x)/f''(x). Newton can be returned only if df is returned."""
        ans = []

        self.update_new_marginal(step_size)

        if return_f:
            ans.append(self.update_dual_objective(step_size))

        if return_df:
            df = self._derivative(step_size)
            ans.append(df)

            if df != 0 and return_newton:
                ans.append(self._newton(df))

        return tuple(ans)

    def update_new_marginal(self, step_size):
        self.newmarg = self.alpha_i.convex_combination(self.beta_i, step_size)

    def update_dual_objective(self, step_size):
        """Update the true value of the dual objective on the fly through the object dual
        objective that is passed as an attribute of the class line search."""
        norm_update = step_size ** 2 * self.primaldir_squared_norm \
                      + 2 * step_size * self.weights_dot_primaldir
        self.monitor_dual_objective.update(self.sample_id, self.newmarg.entropy(), norm_update)
        return self.monitor_dual_objective.get_value()

    def _derivative(self, step_size):
        if step_size == 0:
            return self.divergence_gap + self.reverse_gap
        elif step_size == 1:
            return 2 * self.quadratic_coeff
        else:
            return self.divergence_gap \
                   + self.beta_i.kullback_leibler(self.newmarg) \
                   - self.alpha_i.kullback_leibler(self.newmarg) \
                   + 2 * step_size * self.quadratic_coeff

    def _newton(self, df):
        log_ddf = self.dual_direction \
            .absolute().log() \
            .multiply_scalar(2) \
            .subtract(self.newmarg) \
            .log_reduce_exp(- 2 * self.quadratic_coeff)  # stable log sum exp
        ans = np.log(np.absolute(df)) - log_ddf  # log(|f'(x)/f''(x)|)
        ans = - np.sign(df) * np.exp(ans)  # f'(x)/f''(x)
        return ans

    def run(self):
        if self.fixed_step_size is not None:
            self.optimal_step_size = self.fixed_step_size
        else:
            u0, = self.evaluator(0, return_df=True)
            u1, = self.evaluator(1, return_df=True)
            if u1 >= 0:  # 1 is optimal, the new marginal is already updated
                self.optimal_step_size = 1
                self.subobjectives = [u1]
            else:
                self.optimal_step_size, self.subobjectives = safe_newton(
                    self.evaluator, 0, 1, u0, u1, precision=1e-2)

        self.update_new_marginal(self.optimal_step_size)  # compute the new marginals
        self.update_dual_objective(self.optimal_step_size)  # update the dual objective
        return self.newmarg, self.optimal_step_size

    def log_tensorboard(self, step):
        tl.log_value("step size", self.optimal_step_size, step)
        tl.log_value("number of line search step", len(self.subobjectives), step)
        tl.log_value("log10 primal_direction_squared_norm", np.log10(self.primaldir_squared_norm),
                     step=step)


def safe_newton(evaluator, lowerbound, upperbound, u_lower, u_upper, precision):
    """Using a combination of Newton-Raphson and bisection, find the root of a function bracketed
    between lowerbound and upperbound.

    :param evaluator: user-supplied routine that returns both the function value u(x)
    and u(x)/u'(x)
    :param lowerbound: point smaller than the root
    :param upperbound: point larger than the root
    :param precision: accuracy on the root value rts
    :return: The root, returned as the value rts
    """
    MAX_ITER = 100

    if (u_lower > 0 and u_upper > 0) or (u_lower < 0 and u_upper < 0):
        raise ValueError("Root must be bracketed in [lower bound, upper bound]")
    if u_lower == 0:
        return lowerbound
    if u_upper == 0:
        return upperbound

    if u_lower < 0:  # Orient the search so that f(xl) < 0.
        xl, xh = lowerbound, upperbound
    else:
        xh, xl = lowerbound, upperbound

    rts = (xl + xh) / 2  # Initialize the guess for root
    dxold = abs(upperbound - lowerbound)  # the “stepsize before last"
    dx = dxold  # and the last step

    u, newton = evaluator(rts, return_df=True, return_newton=True)
    obj = [u]

    for _ in np.arange(MAX_ITER):  # Loop over allowed iterations.

        rts -= newton  # Newton step
        if (rts - xh) * (rts - xl) <= 0 and abs(newton) <= abs(dxold) / 2:
            # Keep the Newton step  if it remains in the bracket and
            # if it is converging fast enough.
            # This will be false if fdf is NaN.
            dxold = dx
            dx = newton

        else:  # Bisection otherwise
            dxold = dx
            dx = (xh - xl) / 2
            rts = xl + dx

        if abs(dx) < precision:  # Convergence criterion.
            return rts, obj
        u, newton = evaluator(rts, return_df=True, return_newton=True)
        # the one new function evaluation per iteration
        obj.append([u])

        if u < 0:  # maintain the bracket on the root
            xl = rts
        else:
            xh = rts

    raise RuntimeError("Maximum number of iterations exceeded in safe_newton")
