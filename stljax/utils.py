import jax
import jax.numpy as jnp
import functools

def cond(pred, true_fun, false_fun, *operands):
    if pred:
        return true_fun(*operands)
    else:
        return false_fun(*operands)

def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)


def smooth_mask(T, t_start, t_end, scale):
    xs = jnp.arange(T) * 1.
    return jax.nn.sigmoid(scale * (xs - t_start * T)) - jax.nn.sigmoid(scale * (xs - t_end * T))

def anneal(i):
    return jax.nn.sigmoid(15 * (i - 0.5))


@jax.jit
def bar_plus(signal, p=2):
    '''max(0,signal)**p'''
    return jax.nn.relu(signal) ** p


@jax.jit
def bar_minus(signal, p=2):
    '''min(0,signal)**p'''
    return (-jax.nn.relu(-signal)) ** p

@functools.partial(jax.jit, static_argnames=("axis", "keepdims"))
def M0(signal, eps, weights=None, axis=1, keepdims=True):
    '''Used in gsmr approx method, Eq 4(a) in https://arxiv.org/abs/2405.10996'''
    if weights is None:
        weights = jnp.ones_like(signal)
    sum_w = weights.sum(axis, keepdims=keepdims)
    return (
        eps**sum_w + jnp.prod(signal**weights, axis=axis, keepdims=keepdims)
    ) ** (1 / sum_w)

@functools.partial(jax.jit, static_argnames=("axis", "keepdims"))
def Mp(signal, eps, p, weights=None, axis=1, keepdims=True):
    '''Used in gsmr approx method, Eq 4(b) in https://arxiv.org/abs/2405.10996'''
    if weights is None:
        weights = jnp.ones_like(signal)
    sum_w = weights.sum(axis, keepdims=keepdims)
    return (
        eps**p + 1 / sum_w * jnp.sum(weights * signal**p, axis=axis, keepdims=keepdims)
    ) ** (1 / p)

@functools.partial(jax.jit, static_argnames=("axis", "keepdims"))
def gmsr_min(signal, eps, p, weights=None, axis=1, keepdims=True):
    '''Used in gsmr approx method, Eq 3 in https://arxiv.org/abs/2405.10996'''

    return (
        M0(bar_plus(signal, 2), eps, weights=weights, axis=axis, keepdims=keepdims)
        ** 0.5
        - Mp(
            bar_minus(signal, 2), eps, p, weights=weights, axis=axis, keepdims=keepdims
        )
        ** 0.5
    )

@functools.partial(jax.jit, static_argnames=("axis", "keepdims"))
def gmsr_max(signal, eps, p, weights=None, axis=1, keepdims=True):
    '''Used in gsmr approx method, Eq 4(a) but for max in https://arxiv.org/abs/2405.10996'''

    return -gmsr_min(-signal, eps, p, weights=weights, axis=axis, keepdims=keepdims)

@functools.partial(jax.jit, static_argnames=("axis", "keepdims"))
def gmsr_min_turbo(signal, eps, p, weights=None, axis=1, keepdims=True):
    # TODO: (norrisg) make actually turbo (faster than normal `gmsr_min`)
    pos_idx = signal > 0.0
    neg_idx = ~pos_idx

    return jnp.where(
        neg_idx.sum(axis, keepdims=keepdims) > 0,
        eps**0.5
        - Mp(
            bar_minus(signal, 2),
            eps,
            p,
            weights=weights,
            axis=axis,
            keepdims=keepdims,
        )
        ** 0.5,
        M0(bar_plus(signal, 2), eps, weights=weights, axis=axis, keepdims=keepdims)
        ** 0.5
        - eps**0.5,
    )

@functools.partial(jax.jit, static_argnames=("axis", "keepdims"))
def gmsr_max_turbo(signal, eps, p, weights=None, axis=1, keepdims=True):
    return -gmsr_min_turbo(
        -signal, eps, p, weights=weights, axis=axis, keepdims=keepdims
    )


def gmsr_min_fast(signal, eps, p):
    # TODO: (norrisg) allow `axis` specification

    # Split indices into positive and non-positive values
    pos_idx = signal > 0.0
    neg_idx = ~pos_idx

    weights = jnp.ones_like(signal)

    # Sum of all weights
    sum_w = weights.sum()

    # If there exists a negative element
    if signal[neg_idx].size > 0:
        sums = 0.0
        sums = jnp.sum(weights[neg_idx] * (signal[neg_idx] ** (2 * p)))
        Mp = (eps**p + (sums / sum_w)) ** (1 / p)
        h_min = eps**0.5 - Mp**0.5

    # If all values are positive
    else:
        mult = 1.0
        mult = jnp.prod(signal[pos_idx] ** (2 * weights[pos_idx]))
        M0 = (eps**sum_w + mult) ** (1 / sum_w)
        h_min = M0**0.5 - eps**0.5

    return jnp.reshape(h_min, (1, 1, 1))


def gmsr_max_fast(signal, eps, p):
    return -gmsr_min_fast(-signal, eps, p)

@functools.partial(jax.jit, static_argnames=("axis", "keepdims", "approx_method", "padding"))
def maxish(signal, axis, keepdims=True, approx_method="true", temperature=None, **kwargs):
    """
    Function to compute max(ish) along an axis.

    Args:
        signal: A jnp.array or an Expression
        axis: (Int) axis along to compute max(ish)
        keepdims: (Bool) whether to keep the original array size. Defaults to True
        approx_method: (String) argument to choose the type of max(ish) approximation. possible choices are "true", "logsumexp", "softmax", "gmsr" (https://arxiv.org/abs/2405.10996).
        temperature: Optional, required for approx_method not True.

    Returns:
        jnp.array corresponding to the maxish

    Raises:
        If Expression does not have a value, or invalid approx_method

    """

    # if isinstance(signal, Expression):
    #     assert (
    #         signal.value is not None
    #     ), "Input Expression does not have numerical values"
    #     signal = signal.value

    match approx_method:
        case "true":
            """jax keeps track of multiple max values and will distribute the gradients across all max values!
            e.g., jax.grad(jnp.max)(jnp.array([0.01, 0.0, 0.01])) # --> Array([0.5, 0. , 0.5], dtype=float32)
            """
            return jnp.max(signal, axis, keepdims=keepdims)

        case "logsumexp":
            """https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.special.logsumexp.html"""
            assert temperature is not None, "need a temperature value"
            return (
                jax.scipy.special.logsumexp(
                    temperature * signal, axis=axis, keepdims=keepdims
                )
                / temperature
            )

        case "softmax":
            assert temperature is not None, "need a temperature value"
            return (jax.nn.softmax(temperature * signal, axis) * signal).sum(
                axis, keepdims=keepdims
            )

        case "gmsr":
            assert (
                temperature is not None
            ), "temperature tuple containing (eps, p) is required"
            (eps, p) = temperature
            return gmsr_max(signal, eps, p, axis=axis, keepdims=keepdims)

        case _:
            raise ValueError("Invalid approx_method")

@functools.partial(jax.jit, static_argnames=("axis", "keepdims", "approx_method", "padding"))
def minish(signal, axis, keepdims=True, approx_method="true", temperature=None, **kwargs):
    '''
    Same as maxish
    '''
    return -maxish(-signal, axis, keepdims, approx_method, temperature, **kwargs)
