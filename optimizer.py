import time

import numpy as np


def regularize(w, reg, bias, regularization_function='l2'):
    if bias:
        toreg = w[:-1]
    else:
        toreg = w

    if regularization_function == 'l2':
        return reg / 2 * np.sum(toreg ** 2)
    elif regularization_function == 'l1':
        return reg / 2 * np.sum(np.absolute(toreg))
    else:
        raise ValueError("unsupported regularization type. Either 'l1' or 'l2'.")


def stochastic_gradient_descent(objective, gradient, step_size, w0, x, y, reg, bias=True, npass=100, average=None):
    """ Return the optimal classifier w found by the stochastic gradient descent
    for a finite sum objective with an l2 regularization.
    step_size = float step size, or "auto" for a 1/t step size.
    w0 = initial point
    x = size n*d data points
    y = size n class in {-1,1}
    reg = regularization parameter in the logisitc regression
    bias = if true, do not regularize the last component of w.
    npass = number of passes over the data
    average = averaging scheme, in {None,"uniform","linear"}
    """
    n, d = x.shape
    z = y[:, np.newaxis] * x  # the real value of interest for the logistic regression

    if step_size == "auto":
        def step_sizes(t):
            return 1 / (t + 1) / reg
    else:
        def step_sizes(t):
            return step_size

    w = w0.copy()  # init
    wavg = w.copy()  # running average of the classifier.

    if not average:
        def rho(t):
            return 1
    elif average == "uniform":
        def rho(t):
            return 1 / (1 + t)
    elif average == "linear":
        def rho(t):
            return 2 / (2 + t)
    else:
        raise ValueError("Parameter average should be in {None,'uniform','linear'}")

    # List of the values taken by the objective along the path.
    # Extra cost of n*d at each computation.
    regularizations = [regularize(w, reg, bias)]
    obj = [objective(wavg, x, y)]

    regmask = np.ones_like(w)
    if bias:
        regmask[-1] = 0

    indices = np.random.randint(0, high=n, size=npass * n)
    for t in range(npass * n):
        w = (1 - reg * step_sizes(t) * regmask) * w - step_sizes(t) * gradient(w, x, y, indices[t])
        wavg = (1 - rho(t)) * wavg + rho(t) * w
        if (t + 1) % n == 0:  # compute the objective value after each full pass over the data
            obj.append(objective(wavg, x, y))
            regularizations.append(regularize(w, reg, bias))

    obj = np.array(obj)
    regularizations = np.array(regularizations)
    return w, obj, regularizations


def saga(objective, gradient, step_size, w0, x, y, reg, bias=True, precision=1e-7, npass=100, sag=False,
         uselinesearch=True, nonuniform=False):
    """ Return the optimal classifier w found by stochastic average gradient augmented.
    step_size = float step size, except if line search is used
    w0 = initial parameter
    x = size n*d datapoints
    y = size n classes
    reg = regularization parameter
    bias = if true, do not regularize the last component of w
    precision = stopping criterion for the norm of the estimate of the gradient.
    npass = maximum number of passes over the data
    sag = use sag instead of saga updata scheme
    uselinesearch = if True, use the line search to estimate the lipschitz constant and set the step size.
    nonuniform = use non uniform sampling. If true, line search will also be used.
    """
    n, d = x.shape

    ##################################################################################
    # INIT : quantities to track
    ##################################################################################
    m = 0  # number of examples seen so far
    seen = np.zeros([n, 1], dtype=bool)  # set to true for points already visited
    pastgrads = np.zeros([n, 1])  # past gradients, only a scalar because the model is linear
    sumgrad = 0  # sum of the last gradients seen so far for each data point (sum(pastgrads*x))

    ##################################################################################
    # SCORES : initialize the variables to be returned
    ##################################################################################
    w = w0.copy()
    obj = [objective(w, x, y)]
    gradlist = []
    regularizations = [regularize(w, reg, bias)]
    timing = [time.time()]

    ##################################################################################
    # LIPSCHITZ : lipschitz constant of the loglikelihood used in the step size if we use the line search.
    ##################################################################################
    global_lipschitz_constant = 1
    DECREASE_RATE = 2 ** (-1 / n)
    x_squared_norms = np.sum(x ** 2, axis=1)  # pre-processing of these values used in the line search.
    if nonuniform:
        lipschitz_constants = np.ones(n)
        sum_lipschitz_constants = np.sum(lipschitz_constants)

    # Initialize the stopping criterion
    updategrad = 2 * precision  # we will use the absolute norm of the gradient
    t = 0
    while t < npass * n and (t < n or np.max(np.absolute(updategrad)) > precision):
        t += 1

        ##################################################################################
        # DRAW : one sample at random.
        ##################################################################################
        if not nonuniform:
            i = np.random.randint(n)
        else:
            if np.random.rand() < 0.5:  # with probability 1/2 sample uniformly
                i = np.random.randint(n)
            else:  # and with the other 1/2 sample according to the lipschitz constants
                i = np.random.choice(n, p=lipschitz_constants / sum_lipschitz_constants)
            # we remove the i-th lip from the sum before updating it.
            sum_lipschitz_constants -= lipschitz_constants[i]

        ##################################################################################
        # FIRST TIME : initialize some things if i was never drawn before.
        ##################################################################################
        if not seen[i]:
            seen[i] = True
            if nonuniform:  # initialize the local lipschitz constant.
                lipschitz_constants[i] = 0.5 * sum_lipschitz_constants / n
            m += 1

        ##################################################################################
        # GRADIENT : compute the gradient update for example i.
        ##################################################################################
        # function i and gradient i at position w
        wxi = np.dot(x[i], w)
        newgrad = gradient(w, x, y, i, scalar=True)
        # difference between new gradient and old gradient
        diffgrad = (newgrad - pastgrads[i]) * x[i]
        # define the gradient to be used in the update
        if sag:
            updategrad = (diffgrad + sumgrad) / m
        else:  # saga
            updategrad = diffgrad + sumgrad / m
        # add the regularization for everything but the bias
        if bias:
            updategrad[:-1] += reg * w[:-1]
        else:
            updategrad += reg * w

        # update the gradient in memory
        pastgrads[i] = newgrad
        # update the sum of gradients
        sumgrad += diffgrad

        ##################################################################################
        # STEP SIZE : perform the line search and update the step size.
        ##################################################################################
        if uselinesearch or nonuniform:
            grad_squared_norm = newgrad ** 2 * x_squared_norms[i]
            if not nonuniform:
                if grad_squared_norm > 1e-8:
                    # decrease lipschitz in case the objective is smoother next to the optimum
                    global_lipschitz_constant *= DECREASE_RATE
                    # find an upper bound on the global Lipschitz constant
                    global_lipschitz_constant = linesearch(wxi, y[i], x_squared_norms[i], grad_squared_norm,
                                                           global_lipschitz_constant)
                step_size = 1 / (global_lipschitz_constant + reg)
            else:
                if grad_squared_norm > 1e-8:
                    # let the L_i decrease to be able to increase the step size.
                    lipschitz_constants[i] *= 0.9
                    lipschitz_constants[i] = linesearch(wxi, y[i], x_squared_norms[i], grad_squared_norm,
                                                        lipschitz_constants[i])
                    # decrease the global lipschitz in case the objective is smoother next to the optimum
                    global_lipschitz_constant *= DECREASE_RATE
                    # update the global lipschitz constant
                    global_lipschitz_constant = max(global_lipschitz_constant, lipschitz_constants[i])
                sum_lipschitz_constants += lipschitz_constants[i]  # we already removed the previous one above
                # the step size is defined as the average of the uniform step size and the non-uniform one.
                step_size = .5 / (global_lipschitz_constant + reg) + .5 / (sum_lipschitz_constants / n + reg)

        ##################################################################################
        # UPDATE : apply the update to the parameter w.
        ##################################################################################
        w -= step_size * updategrad

        ##################################################################################
        # SCORES : after each pass over the data, record the scores.
        ##################################################################################
        if t % n == 0:  # update the score after each pass over the data
            obj.append(objective(w, x, y))
            gradlist.append(np.max(np.absolute(updategrad)))
            regularizations.append(regularize(w, reg, bias))
            timing.append(time.time())

    ##################################################################################
    # FINISH : update the scores to simplify the after process.
    ##################################################################################
    # add the last score observed
    obj.append(objective(w, x, y))
    gradlist.append(np.max(np.absolute(updategrad)))
    regularizations.append(regularize(w, reg, bias))
    timing.append(time.time())
    # make the lists into arrays
    obj = np.array(obj)
    gradlist = np.array(gradlist)
    regularizations = np.array(regularizations)
    timing = np.array(timing)
    timing -= timing[0]

    return w, obj, gradlist, regularizations, timing  # TODO : line search only defined for the logistic regression


def softmax(z):
    if z > 0:
        return z + np.log(1 + np.exp(-z))
    else:
        return np.log(1 + np.exp(z))


def sigmoid(z):
    if z > 0:
        return 1 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))


def linesearch(wxi, yi, xnorm, gnorm, lip):
    zi = -yi * wxi
    fi = softmax(zi)
    fbis = softmax(zi - xnorm * sigmoid(zi) / lip)
    while fbis > fi - gnorm / 2 / lip:
        lip *= 2
        fbis = softmax(zi - xnorm * sigmoid(zi) / lip)
    return lip


def sdcalinesearch(x0, a, b, opt='newton', epsilon=1e-15, debug=False):
    """Find a solution of the equation ax + b + log(x) - log(1-x)=0 for x in (0,1)
    with precision epsilon, using Newton's method or Halley's method.
    """

    def u(x):
        return a * x + b + np.log(x / (1 - x))

    def gu(x):
        return a + 1 / x + 1 / (1 - x)

    def ggu(x):
        return -x ** (-2) + (1 - x) ** (-2)

    # def ggudgu(x):
    #     return (- (1 - x) ** 2 + x ** 2) / (a * (x * (1 - x)) ** 2 + x * (1 - x) ** 2 + x ** 2 * (1 - x))

    x = x0
    ux = u(x)
    obj = [ux]
    count = 0
    while np.absolute(ux) > epsilon and count < 100:
        count += 1
        gux = gu(x)
        if opt == 'newton':
            x -= ux / gux
        elif opt == 'halley':
            ggux = ggu(x)
            print(x, ux, gux, ggux)
            # udgux = ux/gux
            # x -= udgux/(1-udgux*ggux/gux/2)
            x -= 2 * ux * gux / (2 * gux ** 2 - ux * ggux)
        else:
            raise ValueError("Argument opt is invalid.")
        # Bring back x in the interior of (0,1)
        x = max(epsilon, x)
        x = min(1 - epsilon, x)
        ux = u(x)
        obj.append(ux)
    if debug:
        return x, obj
    else:
        return x


def sdca(alpha0, x, y, reg, npass=50, precision=1e-15):
    """no step size here as we are doing exact line search
    alpha0 instead of w0 as we are optimizing over the dual variable alpha
    """
    n, d = x.shape
    import binary_logistic_regression as logreg

    def four_objectives(w, x, y, alpha, reg):
        """Return the regularization in w,the negative log-likelihood in w,
        the entropy in alpha, and the duality gap in (w,alpha) dual points."""
        regu = regularize(w, reg, bias=False)
        negll = logreg.negloglikelihood(w, x, y)
        ent = np.mean(logreg.entropy_bernoulli(alpha))
        dual_gap = 2 * regu + negll - ent
        return regu, negll, ent, dual_gap

    ##################################################################################
    # INIT: initialize the dual and primal variables
    ##################################################################################
    alpha = alpha0.copy()
    w = logreg.dual_to_primal(alpha, x, y, reg)

    ##################################################################################
    # SCORES : initialize the variables to be returned
    ##################################################################################
    obj = [four_objectives(w, x, y, alpha, reg)]
    duality_gap = obj[-1][-1]  # stopping criterion
    timing = [time.time()]

    # linear coefficients of the functions for the line search
    linear_coeffs = np.sum(x ** 2, axis=1) / reg / n

    t = 0
    while t < npass * n and duality_gap > precision:
        t += 1

        ##################################################################################
        # DRAW : one sample at random.
        ##################################################################################
        i = np.random.randint(n)

        ##################################################################################
        # LINE SEARCH : find the optimal alphai
        ##################################################################################
        constant_coeff = np.dot(y[i] * x[i], w) - alpha[i] * linear_coeffs[i]
        alphai = sdcalinesearch(alpha[i], linear_coeffs[i], constant_coeff)

        ##################################################################################
        # UPDATE : the primal and dual coordinates
        ##################################################################################
        w += (alphai - alpha[i]) * y[i] * x[i] / reg / n
        alpha[i] = alphai

        ##################################################################################
        # SCORES : after each pass over the data, compute the scores
        ##################################################################################
        if t % n == 0:
            obj.append(four_objectives(w, x, y, alpha, reg))
            duality_gap = obj[-1][-1]
            timing.append(time.time())

    ##################################################################################
    # FINISH : update the scores to simplify the after process.
    ##################################################################################
    obj = np.array(obj)
    timing = np.array(timing)
    timing -= timing[0]
    return w, alpha, obj, timing
