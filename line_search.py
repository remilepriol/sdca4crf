import numpy as np


class LineSearch:

    def __init__(self, weights, primal_direction, dual_direction, alpha_i, beta_i,
                 divergence_gap, reverse_gap, regu, ntrain, step_size):
        self.primaldir_squared_norm = primal_direction.squared_norm()
        self.weights_dot_primaldir = weights.inner_product(primal_direction)
        self.divergence_gap = divergence_gap
        self.reverse_gap = reverse_gap
        self.quadratic_coeff = - regu * ntrain / 2 * self.primaldir_squared_norm
        self.linear_coeff = - regu * ntrain * self.weights_dot_primaldir

        self.alpha_i = alpha_i
        self.beta_i = beta_i
        self.dual_direction = dual_direction
        self.step_size = step_size

    def evaluator(self, gamma, return_f=False, return_df=False, return_newton=False):
        """Return the line search function f and its derivative df evaluated in the step
        size gamma. Because the second derivative can be insanely large, we return instead the
        Newton step f'(x)/f''(x). Newton can be returned only if df is returned."""
        # new marginals
        newmarg = self.alpha_i.convex_combination(self.beta_i, gamma)

        ans = []

        if return_f:
            ans.append(self._function(newmarg, gamma))

        if return_df:
            df = self._derivative(newmarg, gamma)
            ans.append(df)

            if df != 0 and return_newton:
                ans.append(self._newton(newmarg, df))

        return ans

    def _function(self, newmarg, gamma):
        return newmarg.entropy() + gamma ** 2 * self.quadratic_coeff + gamma * self.linear_coeff

    def _derivative(self, newmarg, gamma):
        if gamma == 0:
            return self.divergence_gap + self.reverse_gap
        elif gamma == 1:
            return 2 * self.quadratic_coeff
        else:
            return self.divergence_gap \
                   + self.beta_i.kullback_leibler(newmarg) \
                   - self.alpha_i.kullback_leibler(newmarg) \
                   + 2 * gamma * self.quadratic_coeff

    def _newton(self, newmarg, df):
        log_ddf = self.dual_direction \
            .absolute().log() \
            .multiply_scalar(2) \
            .subtract(newmarg) \
            .log_reduce_exp(- 2 * self.quadratic_coeff)  # stable log sum exp
        ans = np.log(np.absolute(df)) - log_ddf  # log(|f'(x)/f''(x)|)
        ans = - np.sign(df) * np.exp(ans)  # f'(x)/f''(x)
        return ans

    def run(self):
        if self.step_size is not None:
            return self.step_size, []
        else:
            return find_root_decreasing(evaluator=self.evaluator, precision=1e-2)


def find_root_decreasing(evaluator, precision):
    """Return the root x0 of a decreasing function u defined on [0,1] with given precision.
    The root can be smaller than 0, in which case, return 0.
    The root can be larger than 1, in which case, return 1.

    :param evaluator: function that return the values u(x) and u(x)/u'(x)
    :param precision: maximum value of |u(x)| so that x is returned
    :return: x an approximate root of u
    """

    u0, _ = evaluator(0)
    # if u0 <= precision:  # 0 is optimal
    #     return 0, [u0]

    u1, _ = evaluator(1)
    if u1 >= 0:  # 1 is optimal
        return 1, [u1]

    return safe_newton(evaluator, 0, 1, u0, u1, precision=precision)


def bounded_newton(evaluator, init, lowerbound, upperbound, precision=1e-12, max_iter=20):
    """Return the root x0 of a function u defined on [lowerbound, upperbound] with given precision,
    using Newton-Raphson method.

    :param evaluator: function that return the values u(x) and u(x)/u'(x)
    :param init: initial point x
    :param lowerbound:
    :param upperbound:
    :param precision: on the value of |u(x)|
    :param max_iter: maximum number of iterations
    :return: x an approximate root of u
    """
    x = init
    fx, gx = evaluator(x)
    obj = [fx]
    count = 0
    while np.absolute(fx) > precision and count < max_iter:
        # stop condition to avoid cycling over an extremity of the segment
        count += 1
        x -= gx
        # Make sure x is in (lower bound, upper bound)
        x = max(lowerbound, x)
        x = min(upperbound, x)
        fx, gx = evaluator(x)
        obj.append(fx)
    return x, obj


# define MAXIT 100 Maximum allowed number of iterations.
def safe_newton(evaluator, lowerbound, upperbound, flower, fupper, precision, max_iter=200):
    """Using a combination of Newton-Raphson and bisection, find the root of a function bracketed
    between lowerbound and upperbound.

    :param evaluator: user-supplied routine that returns both the function value u(x)
    and u(x)/u'(x)
    :param lowerbound: point smaller than the root
    :param upperbound: point larger than the root
    :param precision: accuracy on the root value rts
    :param max_iter: maximum number of iteration
    :return: The root, returned as the value rts
    """

    if (flower > 0 and fupper > 0) or (flower < 0 and fupper < 0):
        raise ValueError("Root must be bracketed in [lower bound, upper bound]")
    if flower == 0:
        return lowerbound
    if fupper == 0:
        return upperbound

    if flower < 0:  # Orient the search so that f(xl) < 0.
        xl, xh = lowerbound, upperbound
    else:
        xh, xl = lowerbound, upperbound

    rts = (xl + xh) / 2  # Initialize the guess for root
    dxold = abs(upperbound - lowerbound)  # the “stepsize before last"
    dx = dxold  # and the last step

    f, fdf = evaluator(rts)
    obj = [[f]]

    for _ in np.arange(max_iter):  # Loop over allowed iterations.
        rtsold = rts
        rts -= fdf  # Newton step

        if (rts - xh) * (rts - xl) <= 0 and abs(fdf) <= abs(dxold) / 2:
            # Keep the Newton step  if it remains in the bracket and
            # if it is converging fast enough.
            # This will be false if fdf is NaN.
            dxold = dx
            dx = fdf
            if rtsold == rts:  # change in root is negligible
                return rts, np.array(obj)

        else:  # Bisection otherwise
            dxold = dx
            dx = (xh - xl) / 2
            rts = xl + dx
            if xl == rts:  # change in root is negligible
                return rts, np.array(obj)

        if abs(dx) < precision:  # Convergence criterion.
            return rts, np.array(obj)
        f, fdf = evaluator(rts)  # the one new function evaluation per iteration
        obj.append([f])
        if f < 0:  # maintain the bracket on the root
            xl = rts
        else:
            xh = rts

    raise RuntimeError("Maximum number of iterations exceeded in safe_newton")
