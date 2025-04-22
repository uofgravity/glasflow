import torch
from torch.nn import functional as F

from glasflow.nflows.transforms.base import InputOutsideDomain


clip = 0.0


def bernstein_basis_polynomial(k, n):
    # returns a berstein basis polynomial b_{k,n}(x) = nCk x^k (1-x)^(n-k), nCk = n!/(k!(n-k)!), as a function
    # gamma(n) = (n-1)!
    binomial_coeff = torch.exp(
        torch.special.gammaln(n + 1)
        - (torch.special.gammaln(n - k + 1) + torch.special.gammaln(k + 1))
    )

    def basis_polynomial(x):
        if k == 0:
            return torch.where(
                x == 0,
                torch.tensor(1.0),
                binomial_coeff
                * torch.exp(k * torch.log(x) + (n - k) * torch.log(1 - x)),
            )
        if k == n:
            return torch.where(
                x == 1,
                torch.tensor(1.0),
                binomial_coeff
                * torch.exp(k * torch.log(x) + (n - k) * torch.log(1 - x)),
            )
        else:
            return binomial_coeff * torch.exp(
                k * torch.log(x) + (n - k) * torch.log(1 - x)
            )

    # def basis_polynomial(x):
    #    return binomial_coeff * torch.exp(
    #            k * torch.log(x) + (n - k) * torch.log(1 - x)
    #        )

    return basis_polynomial


def log_bernstein_basis_polynomial(k, n):
    # returns a berstein basis polynomial b_{k,n}(x) = nCk x^k (1-x)^(n-k), nCk = n!/(k!(n-k)!), as a function
    # gamma(n) = (n-1)!
    log_binomial_coeff = torch.special.gammaln(n + 1) - (
        torch.special.gammaln(n - k + 1) + torch.special.gammaln(k + 1)
    )

    def log_basis_polynomial(x):
        if k == 0:
            return torch.where(
                x == 0,
                torch.tensor(0.0),
                log_binomial_coeff
                + k * torch.log1p(x - 1)
                + (n - k) * torch.log1p(-x),
            )
        if k == n:
            return torch.where(
                x == 1,
                torch.tensor(0.0),
                log_binomial_coeff
                + k * torch.log1p(x - 1)
                + (n - k) * torch.log1p(-x),
            )
        else:
            return (
                log_binomial_coeff
                + k * torch.log1p(x - 1)
                + (n - k) * torch.log1p(-x)
            )

    # def log_basis_polynomial(x):
    #    return (
    #        log_binomial_coeff
    #        + k * torch.log1p(x - 1)
    #        + (n - k) * torch.log1p(-x)
    #    )

    return log_basis_polynomial


def bernstein_basis_polynomial_derivative(k, n):
    def db_k_n_dx(x):
        if k != 0:
            return n * (
                bernstein_basis_polynomial(k - 1, n - 1)(x)
                - bernstein_basis_polynomial(k, n - 1)(x)
            )
        else:
            return -n * torch.exp((n - 1) * torch.log(1 - x))

    return db_k_n_dx


def bernstein_transform_log_derivative(x, alphas):
    n = alphas.shape[-1] - 1  # degree of the bernstein polynomial

    # alphas[1:]  # alpha_1, alpha_2, ..., alpha_n
    # alphas[:-1] # alpha_0, alpha_1, ..., alpha_n-1

    log_bernstein_basis = [
        log_bernstein_basis_polynomial(torch.tensor(k), torch.tensor(n - 1))
        for k in range(n)
    ]
    log_bernstein_basis_at_x = torch.stack(
        [
            log_basis_polynomial(x)
            for log_basis_polynomial in log_bernstein_basis
        ]
    )
    coeffs = n * (
        alphas[:, 1:] - alphas[:, :-1]
    )  # (alpha_k - alpha_k-1) for k between 0 and n
    logdet = torch.logsumexp(
        log_bernstein_basis_at_x + torch.log(coeffs.T), dim=0
    )

    return logdet


def get_increasing_alphas_nd(unconstrained_params, range_min=0, range_max=1):
    # returns a strictly increasing sequence of alphas, alpha_0=range_min, alpha_n = range_max, n=len(unconstrained_params)

    unconstrained_params = torch.atleast_2d(unconstrained_params)
    cumsum = torch.cumsum(torch.abs(unconstrained_params) + 1e-5, axis=-1)
    scaled_params = range_min + cumsum / torch.atleast_2d(cumsum[:, -1]).T * (
        range_max - range_min
    )
    alphas = torch.concatenate(
        [
            torch.full((unconstrained_params.shape[0], 1), range_min),
            scaled_params,
        ],
        axis=-1,
    )

    return alphas


def bernstein_fwd(x, alphas):
    """Computes y = f(x)."""
    # takes x as 1D array of the shape (n_samps), alpha of the shape (n_samps, n_params)

    x = torch.clip(x, clip, 1.0 - clip)

    n = alphas.shape[-1] - 1  # degree of the bernstein polynomial
    bernstein_basis = [
        bernstein_basis_polynomial(torch.tensor(k), torch.tensor(n))
        for k in range(n + 1)
    ]
    basis_at_x = torch.stack(
        [basis_polynomial(x) for basis_polynomial in bernstein_basis]
    )

    y = torch.sum(torch.multiply(basis_at_x, alphas.T), axis=0)

    return y


def bernstein_fwd_log(x, alphas):
    """Computes y = f(x) and log|d/dx(f(x))|"""
    # takes in a scalar x

    n = alphas.shape[-1] - 1  # degree of the bernstein polynomial
    log_basis_at_x = torch.stack(
        [
            log_bernstein_basis_polynomial(torch.tensor(k), torch.tensor(n))(x)
            for k in range(n + 1)
        ]
    )
    # log_basis_at_x = torch.stack([log_basis_polynomial(x) for log_basis_polynomial in log_bernstein_basis])

    y = torch.exp(torch.logsumexp(log_basis_at_x + torch.log(alphas.T), dim=0))

    return y


def bernstein_log_det(x, alphas):
    # takes x as 1D array of the shape (n_samps), alpha of the shape (n_samps, n_params)

    n = alphas.shape[-1] - 1  # degree of the bernstein polynomial
    bernstein_basis_derivatives = [
        bernstein_basis_polynomial_derivative(torch.tensor(k), torch.tensor(n))
        for k in range(n + 1)
    ]

    basis_derivative_at_x = torch.stack(
        [
            basis_derivative(x)
            for basis_derivative in bernstein_basis_derivatives
        ]
    )  # (poly_degree,)

    derivative_in_basis = torch.sum(
        torch.multiply(basis_derivative_at_x, alphas.T), axis=0
    )

    logdet = torch.log(
        torch.abs(derivative_in_basis)
    )  # log of the abosolute value of the derivative

    return logdet


def bernstein_transform_fwd(x, alphas):
    """Computes y = f(x) and log|d/dx(f(x))|"""
    # takes in a scalar x and alphas 1D array
    x = torch.clip(x, clip, 1 - clip)
    y = bernstein_fwd(x, alphas)
    logabsdet = bernstein_log_det(x, alphas)

    return y, logabsdet


def bernstein_transform_fwd_log(x, alphas):
    """Computes y = f(x) and log|d/dx(f(x))|"""
    # takes in a scalar x and alphas 1D array
    x = torch.clip(x, clip, 1 - clip)
    y = bernstein_fwd_log(x, alphas)
    logabsdet = bernstein_transform_log_derivative(x, alphas)

    return y, logabsdet


def bernstein_transform_inv(y, alphas):
    """Computes x = f^{-1}(y) and  log|d/dy(f^-1(y))|."""
    n_points = 200
    n_samps = y.shape[0]

    x_fit = torch.tile(torch.linspace(clip, 1 - clip, n_points), (n_samps, 1))

    alphas_long = torch.tile(
        alphas, (n_points, 1, 1)
    )  # the same alpha for every sample

    y_fit = bernstein_fwd(x_fit, alphas_long)

    # interpolation function
    def inp(y, y_fit, x_fit):
        return interp(y, y_fit, x_fit)

    x = inp(torch.atleast_2d(y).T, y_fit, x_fit).flatten()

    logdet = -bernstein_log_det(x, alphas)

    return x, logdet


def bernstein_transform_inv_log(y, alphas):
    """Computes x = f^{-1}(y) and  log|d/dy(f^-1(y))|."""
    n_points = 200
    n_samps = y.shape[0]

    x_fit = torch.tile(torch.linspace(clip, 1 - clip, n_points), (n_samps, 1))

    alphas_long = torch.tile(
        alphas, (n_points, 1, 1)
    )  # the same alpha for every sample

    y_fit = bernstein_fwd_log(x_fit, alphas_long)

    # interpolation function
    def inp(y, y_fit, x_fit):
        return interp(y, y_fit, x_fit)

    x = inp(torch.atleast_2d(y).T, y_fit, x_fit).flatten()

    logdet = -bernstein_transform_log_derivative(x, alphas)

    return x, logdet


def interp(x, xp, fp):

    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[..., :-1] - m * xp[..., :-1]  # offset
    indices = torch.searchsorted(xp, x, right=False)

    # constant extrapolation
    m = torch.cat(
        [torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1
    )
    b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)

    values = m.gather(-1, indices) * x + b.gather(-1, indices)

    return values


def sigmoid(x):
    y = 1 / (1 + torch.exp(-x))
    log_dy_dx = torch.log(y) + torch.log1p(-y)

    return y, log_dy_dx


def logit(x):
    y = torch.log(x) - torch.log1p(-x)
    log_dy_dx = -(torch.log(x - x**2))
    return y, log_dy_dx


def bernstein_transform(
    inputs,
    unconstrained_alphas,
    inverse=False,
    left=0.0,
    right=1.0,
    top=1.0,
    bottom=0.0,
    base_distribution=None,
    log=False,
):

    domain_min = left
    domain_max = right
    range_min = bottom
    range_max = top

    initial_shape = inputs.shape
    inputs = inputs.flatten()

    # unconstrained_alphas = unconstrained_alphas.reshape((unconstrained_alphas.shape[0], unconstrained_alphas.shape[-1]))
    unconstrained_alphas = unconstrained_alphas.reshape(
        (inputs.shape[0], unconstrained_alphas.shape[-1])
    )

    # constrain alphas to an increasing sequence
    alphas = get_increasing_alphas_nd(
        unconstrained_alphas, range_min=range_min, range_max=range_max
    )

    # print(torch.min(inputs))
    # print(torch.max(inputs))

    if base_distribution == None:  # None = gaussian latent
        # sigmoid the data fit within the [0,1] domain of the Bernstein transforms
        inputs, log_j_1 = sigmoid(inputs)

        # check if input is in the domain
        if torch.min(inputs) < domain_min or torch.max(inputs) > domain_max:
            raise InputOutsideDomain()

        if log:
            if inverse:
                outputs, logabsdet = bernstein_transform_inv_log(
                    inputs, alphas
                )
            else:
                outputs, logabsdet = bernstein_transform_fwd_log(
                    inputs, alphas
                )

        else:
            if inverse:
                outputs, logabsdet = bernstein_transform_inv(inputs, alphas)
            else:
                outputs, logabsdet = bernstein_transform_fwd(inputs, alphas)

        # logit the data (bring back to infinite range)
        outputs, log_j_2 = logit(outputs)
        logabsdet = logabsdet + log_j_1 + log_j_2
        # print(log_j_2)

    else:
        # check if input is in the domain
        if torch.min(inputs) < domain_min or torch.max(inputs) > domain_max:
            raise InputOutsideDomain()

        if log:
            if inverse:
                outputs, logabsdet = bernstein_transform_inv_log(
                    inputs, alphas
                )
            else:
                outputs, logabsdet = bernstein_transform_fwd_log(
                    inputs, alphas
                )

        else:
            if inverse:
                outputs, logabsdet = bernstein_transform_inv(inputs, alphas)
            else:
                outputs, logabsdet = bernstein_transform_fwd(inputs, alphas)

    return outputs.reshape(initial_shape), logabsdet.reshape(initial_shape)
