import numpy as np
import tensorboard_logger as tl

from scipy.optimize import minimize_scalar


class LineSearch:

    def __init__(self, weights, primal_direction, log_dual_direction,
                 alpha_i, beta_i, divergence_gap, regu, ntrain, subprecision=1e-3):

        # linse search direction
        self.alpha_i = alpha_i
        self.beta_i = beta_i
        self.log_dual_direction_squared = log_dual_direction.multiply_scalar(2)

        # values for the derivative of the entropy
        self.divergence_gap = divergence_gap
        self.reverse_gap = self.beta_i.kullback_leibler(alpha_i)

        # values for the quadratic term
        self.primaldir_squared_norm = primal_direction.squared_norm()
        self.weights_dot_primaldir = weights.inner_product(primal_direction)
        self.quadratic_coeff = - regu * ntrain / 2 * self.primaldir_squared_norm
        self.linear_coeff = - regu * ntrain * self.weights_dot_primaldir

        # hyperparameter
        self.subprecision = subprecision

        # return values
        self.optimal_step_size = 0
        self.subobjectives = []

    def evaluator(self, step_size, return_f=False, return_df=False, return_newton=False):
        """Return the line search function f and its derivative df evaluated in the step
        size step_size. Because the second derivative can be insanely large, we return instead the
        Newton step f'(x)/f''(x). Newton can be returned only if df is returned."""
        ans = []

        newmarg = self.new_marginal(step_size)

        if return_f:
            ans.append(self._function(newmarg, step_size))

        if return_df:
            df = self._derivative(newmarg, step_size)
            ans.append(df)

            if df != 0 and return_newton:
                ans.append(self._newton(newmarg, df))

        return tuple(ans) if len(ans) > 1 else ans[0]

    def new_marginal(self, step_size):
        return self.alpha_i.convex_combination(self.beta_i, step_size)

    def _function(self, newmarg, step_size):
        return newmarg.entropy() \
               + step_size ** 2 * self.quadratic_coeff \
               + 2 * step_size * self.linear_coeff

    def _derivative(self, newmarg, step_size):
        if step_size == 0:
            return self.divergence_gap + self.reverse_gap
        elif step_size == 1:
            return 2 * self.quadratic_coeff
        else:
            return self.divergence_gap \
                   + self.beta_i.kullback_leibler(newmarg) \
                   - self.alpha_i.kullback_leibler(newmarg) \
                   + 2 * step_size * self.quadratic_coeff

    def _newton(self, newmarg, df):
        log_ddf = self.log_dual_direction_squared \
            .subtract(newmarg) \
            .log_reduce_exp(- 2 * self.quadratic_coeff)  # stable log sum exp
        ans = np.log(np.absolute(df)) - log_ddf  # log(|f'(x)/f''(x)|)
        ans = - np.sign(df) * np.exp(ans)  # f'(x)/f''(x)
        return ans

    def run(self):
        u0 = self.evaluator(0, return_df=True)
        assert u0 > 0, u0

        u1 = self.evaluator(1, return_df=True)
        if u1 >= 0:
            self.optimal_step_size = 1
            self.subobjectives = [u1]
            return self.optimal_step_size

        self.optimal_step_size, self.subobjectives = safe_newton(
            lambda x: self.evaluator(x, return_df=True, return_newton=True),
            lowerbound=0, upperbound=1,
            u_lower=u0, u_upper=u1, precision=self.subprecision)
        return self.optimal_step_size

    def auto_run(self):
        result = minimize_scalar(lambda x: - self.evaluator(x, return_f=True),
                                 bounds=(0, 1), method='bounded',
                                 options={'xatol': self.subprecision})

        # compare = self.run()
        # step_diff = result.x - compare
        # dual_diff = - result.fun - self.evaluator(compare, return_f=True)

        # if dual_diff <0 or abs(step_diff) > 0.1:
        #   print(f"scipy - mine: step-size {step_diff:.4f} \t dual {dual_diff:.4f}")
        ##caution the function value of result is the negative dual : minimization

        self.optimal_step_size = result.x
        # hack to simulate a list of the right length
        self.subobjectives = [result.fun] * result.nfev

        return self.optimal_step_size

    def norm_update(self, step_size):
        return step_size ** 2 * self.primaldir_squared_norm \
               + 2 * step_size * self.weights_dot_primaldir

    def log_tensorboard(self, step):
        tl.log_value("optimal step size", self.optimal_step_size, step)
        tl.log_value("number of line search steps", len(self.subobjectives), step)
        tl.log_value("log10 primal_direction_squared_norm", np.log10(self.primaldir_squared_norm),
                     step=step)


def safe_newton(evaluator, lowerbound, upperbound, u_lower, u_upper, precision,
                max_iter=100):
    """Using a combination of Newton-Raphson and bisection, find the root of a function bracketed
    between lowerbound and upperbound.

    :param evaluator: user-supplied routine that returns both the function value u(x)
    and u(x)/u'(x)
    :param lowerbound: point smaller than the root
    :param upperbound: point larger than the root
    :param precision: accuracy on the root value rts
    :return: The root, returned as the value rts
    """

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

    u, newton = evaluator(rts)
    obj = [u]

    for _ in np.arange(max_iter):  # Loop over allowed iterations.

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
        u, newton = evaluator(rts)
        # the one new function evaluation per iteration
        obj.append(u)

        if u < 0:  # maintain the bracket on the root
            xl = rts
        else:
            xh = rts

    raise RuntimeError("Maximum number of iterations exceeded in safe_newton")
