
import jax
import jax.numpy as jnp
import numpy as np
from stljax.formula import *
from stljax.viz import *
import functools

jax.config.update("jax_enable_x64", True)

def test_always(signal, interval, verbose=True, **kwargs):
    def true_robustness_trace(signal, interval, **kwargs):
        T = len(signal)
        if (interval is None):
            return jnp.stack([minish(signal[i:], axis=0, keepdims=False, **kwargs) for i in range(len(signal))])
        else:
            if interval[1] == jnp.inf:
                large_number = 1E9
                start = interval[0]
                signal_padded = jnp.concat([signal, jnp.ones(T-1) * large_number, jnp.ones(start) * -large_number])
                return jnp.stack([minish(signal_padded[i+start:i+start+T], axis=0, keepdims=False, **kwargs) for i in range(len(signal))])
            else:
                large_number = 1E9
                signal_padded = jnp.concat([signal, jnp.ones(interval[1]+1) * -large_number])
                return jnp.stack([minish(signal_padded[i:][interval[0]:interval[1]+1], axis=0, keepdims=False, **kwargs) for i in range(len(signal))])

    def true_robustness(signal, interval, **kwargs):
        return true_robustness_trace(signal, interval, **kwargs)[0]

    pred = Predicate("identity", lambda x: x)
    phi = Always(pred > 0., interval=interval)
    rob = phi.robustness(signal, **kwargs)
    rob_trace = phi(signal, **kwargs)
    rob_grad = jax.grad(phi.robustness)(signal, **kwargs)

    true_trace = true_robustness_trace(signal, interval, **kwargs)
    true_rob = true_robustness(signal, interval, **kwargs)
    true_grad = jax.grad(true_robustness)(signal, interval, **kwargs)

    rob_correct = jnp.isclose(rob, true_rob)
    trace_correct = jnp.all(jnp.isclose(rob_trace, true_trace))
    grad_correct = jnp.all(jnp.isclose(rob_grad, true_grad))

    pass_n = 0

    if rob_correct:
        if verbose: print("\u2713 Always robustness value match expected answer")
        pass_n += 1
    else:
        print("\u274c Always robustness value does not match expected answer")

    if trace_correct:
        if verbose: print("\u2713 Always robustness trace match expected answer")
        pass_n += 1
    else:
        print("\u274c Always robustness trace does not match expected answer")

    if grad_correct:
        if verbose: print("\u2713 Always robustness gradient match expected answer")
        pass_n += 1
    else:
        print("\u274c Always robustness gradient does not match expected answer")

    print("%i/3 test passed for Always formula\n"%pass_n)


def test_eventually(signal, interval, verbose=True, **kwargs):
    def true_robustness_trace(signal, interval, **kwargs):
        T = len(signal)
        if (interval is None):
            return jnp.stack([maxish(signal[i:], axis=0, keepdims=False, **kwargs) for i in range(len(signal))])
        else:
            if interval[1] == jnp.inf:
                large_number = 1E9
                start = interval[0]
                signal_padded = jnp.concat([signal, jnp.ones(T) * -large_number])
                return jnp.stack([maxish(signal_padded[i+start:i+T], axis=0, keepdims=False, **kwargs) for i in range(len(signal))])
            else:
                large_number = 1E9
                signal_padded = jnp.concat([signal, jnp.ones(interval[1]+1) * -large_number])
                return jnp.stack([maxish(signal_padded[i:][interval[0]:interval[1]+1], axis=0, keepdims=False, **kwargs) for i in range(len(signal))])

    def true_robustness(signal, interval, **kwargs):
        return true_robustness_trace(signal, interval, **kwargs)[0]

    pred = Predicate("identity", lambda x: x)
    phi = Eventually(pred > 0., interval=interval)
    rob = phi.robustness(signal, **kwargs)
    rob_trace = phi(signal, **kwargs)
    rob_grad = jax.grad(phi.robustness)(signal, **kwargs)

    true_trace = true_robustness_trace(signal, interval, **kwargs)
    true_rob = true_robustness(signal, interval, **kwargs)
    true_grad = jax.grad(true_robustness)(signal, interval, **kwargs)

    rob_correct = jnp.isclose(rob, true_rob)
    trace_correct = jnp.all(jnp.isclose(rob_trace, true_trace))
    grad_correct = jnp.all(jnp.isclose(rob_grad, true_grad))

    pass_n = 0

    if rob_correct:
        if verbose: print("\u2713 Eventually robustness value match expected answer")
        pass_n += 1
    else:
        print("\u274c Eventually robustness value does not match expected answer")

    if trace_correct:
        if verbose: print("\u2713 Eventually robustness trace match expected answer")
        pass_n += 1
    else:
        print("\u274c Eventually robustness trace does not match expected answer")

    if grad_correct:
        if verbose: print("\u2713 Eventually robustness gradient match expected answer")
        pass_n += 1
    else:
        print("\u274c Eventually robustness gradient does not match expected answer")

    print("%i/3 test passed for Eventually formula\n"%pass_n)


def test_until(signal, interval, verbose=True, **kwargs):
    pred = Predicate("identity", lambda x: x)
    phi = Until(pred > 0., pred < 0., interval=interval)

    def true_robustness_trace(signal, interval, **kwargs):
        signal1, signal2 = phi.subformula1(signal), phi.subformula2(signal)
        T = len(signal1)
        if interval is None:
            interval = [0,T-1]
        elif interval[1] == jnp.inf:
            interval = [interval[0], T-1]
        large_number = 1E9
        signal1_padded = jnp.concat([signal1, jnp.ones_like(signal1) * -large_number])
        signal2_padded = jnp.concat([signal2, jnp.ones_like(signal2) * -large_number])
        return jnp.stack([maxish(jnp.stack([minish(jnp.stack([minish(signal1_padded[i:][:t+1], axis=0, keepdims=False, **kwargs), signal2_padded[i:][t]]), axis=0, keepdims=False, **kwargs) for t in range(interval[0],interval[-1]+1)]), axis=0, keepdims=False, **kwargs) for i in range(T)])

    def true_robustness(signal, interval, **kwargs):
        return true_robustness_trace(signal, interval, **kwargs)[0]


    pred = Predicate("identity", lambda x: x)
    phi = Until(pred > 0., pred < 0, interval=interval)
    rob = phi.robustness(signal, **kwargs)
    rob_trace = phi(signal, **kwargs)
    rob_grad = jax.grad(phi.robustness)(signal, **kwargs)

    true_trace = true_robustness_trace(signal, interval, **kwargs)
    true_rob = true_robustness(signal, interval, **kwargs)
    true_grad = jax.grad(true_robustness)(signal, interval, **kwargs)

    rob_correct = jnp.isclose(rob, true_rob, atol=1E-5)
    trace_correct = jnp.all(jnp.isclose(rob_trace, true_trace, atol=1E-5))
    grad_diff = jnp.linalg.norm(rob_grad - true_grad)
    grad_correct = jnp.all(jnp.isclose(rob_grad, true_grad, atol=1E-5))

    pass_n = 0

    if rob_correct:
        if verbose: print("\u2713 Until robustness value match expected answer")
        pass_n += 1
    else:
        print("\u274c Until robustness value does not match expected answer")

    if trace_correct:
        if verbose: print("\u2713 Until robustness trace match expected answer")
        pass_n += 1
    else:
        print("\u274c Until robustness trace does not match expected answer")

    if grad_correct:
        if verbose: print("\u2713 Until robustness gradient match expected answer")
        pass_n += 1
    else:
        print("\u274c Until robustness gradient does not match expected answer")
        print(grad_diff)
        print(rob_grad, true_grad)

    print("%i/3 test passed for Until formula\n"%pass_n)


def test_always_mask_recurrent(signal, interval, verbose=True, **kwargs):
    signal_flip = jnp.flip(signal)
    pred = Predicate("identity", lambda x: x)
    mask = Always(pred > 0., interval)
    rec = AlwaysRecurrent(pred > 0., interval)

    rob_correct = jnp.isclose(mask.robustness(signal, **kwargs), rec.robustness(signal_flip, **kwargs))
    trace_correct = jnp.all(jnp.isclose(mask(signal, **kwargs), jnp.flip(rec(signal_flip, **kwargs))))
    grad_correct = jnp.all(jnp.isclose(jax.grad(mask.robustness)(signal, **kwargs), jnp.flip(jax.grad(rec.robustness)(signal_flip, **kwargs))))

    pass_n = 0

    if rob_correct:
        if verbose: print("\u2713 Always mask vs recurrent robustness value match expected answer")
        pass_n += 1
    else:
        print("\u274c Always mask vs recurrent robustness value does not match expected answer")

    if trace_correct:
        if verbose: print("\u2713 Always mask vs recurrent robustness trace match expected answer")
        pass_n += 1
    else:
        print("\u274c Always mask vs recurrent robustness trace does not match expected answer")

    if grad_correct:
        if verbose: print("\u2713 Always mask vs recurrent robustness gradient match expected answer")
        pass_n += 1
    else:
        print("\u274c Always mask vs recurrent robustness gradient does not match expected answer")


    print("%i/3 test passed for Always mask vs recurrent formula\n"%pass_n)


def test_eventually_mask_recurrent(signal, interval, verbose=True, **kwargs):
    signal_flip = jnp.flip(signal)
    pred = Predicate("identity", lambda x: x)
    mask = Eventually(pred > 0., interval)
    rec = EventuallyRecurrent(pred > 0., interval)

    rob_correct = jnp.isclose(mask.robustness(signal, **kwargs), rec.robustness(signal_flip, **kwargs))
    trace_correct = jnp.all(jnp.isclose(mask(signal, **kwargs), jnp.flip(rec(signal_flip, **kwargs))))
    grad_correct = jnp.all(jnp.isclose(jax.grad(mask.robustness)(signal, **kwargs), jnp.flip(jax.grad(rec.robustness)(signal_flip, **kwargs))))

    pass_n = 0

    if rob_correct:
        if verbose: print("\u2713 Eventually mask vs recurrent robustness value match expected answer")
        pass_n += 1
    else:
        print("\u274c Eventually mask vs recurrent robustness value does not match expected answer")

    if trace_correct:
        if verbose: print("\u2713 Eventually mask vs recurrent robustness trace match expected answer")
        pass_n += 1
    else:
        print("\u274c Eventually mask vs recurrent robustness trace does not match expected answer")

    if grad_correct:
        if verbose: print("\u2713 Eventually mask vs recurrent robustness gradient match expected answer")
        pass_n += 1
    else:
        print("\u274c Eventually mask vs recurrent robustness gradient does not match expected answer")


    print("%i/3 test passed for Eventually mask vs recurrent formula\n"%pass_n)


def test_until_mask_recurrent(signal, interval, verbose=True, **kwargs):
    signal_flip = jnp.flip(signal)
    pred = Predicate("identity", lambda x: x)
    mask = Until(pred > 0., pred < 0., interval)
    rec = UntilRecurrent(pred > 0., pred < 0., interval)

    rob_correct = jnp.isclose(mask.robustness(signal, **kwargs), rec.robustness(signal_flip, **kwargs))
    trace_correct = jnp.all(jnp.isclose(mask(signal, **kwargs), jnp.flip(rec(signal_flip, **kwargs))))
    grad_correct = jnp.all(jnp.isclose(jax.grad(mask.robustness)(signal, **kwargs), jnp.flip(jax.grad(rec.robustness)(signal_flip, **kwargs))))

    pass_n = 0

    if rob_correct:
        if verbose: print("\u2713 Until mask vs recurrent robustness value match expected answer")
        pass_n += 1
    else:
        print("\u274c Until mask vs recurrent robustness value does not match expected answer")

    if trace_correct:
        if verbose: print("\u2713 Until mask vs recurrent robustness trace match expected answer")
        pass_n += 1
    else:
        print("\u274c Until mask vs recurrent robustness trace does not match expected answer")

    if grad_correct:
        if verbose: print("\u2713 Until mask vs recurrent robustness gradient match expected answer")
        pass_n += 1
    else:
        print("\u274c Until mask vs recurrent robustness gradient does not match expected answer")


    print("%i/3 test passed for Until mask vs recurrent formula\n"%pass_n)

def test_all_settings(test_func, verbose=True, T=10):
    assert T > 0, "Pick a T larger than 10"
    signal = jnp.array(np.random.randn(T)) * 1.0
    # None, [0,b], [a,b], [0, inf], [a, inf]
    interval_list = [None, [0, T//3], [T//4, T//2], [0, jnp.inf], [T//4, jnp.inf]]
    approx_method_list = ["true", "logsumexp"]
    temperature_list = [1., 10., 20., 100.]

    for interval in interval_list:
        for approx_method in approx_method_list:
            for temperature in temperature_list:
                kwargs = {"approx_method": approx_method,
                          "temperature": temperature
                         }
                print(f"int={interval}\t temp={temperature} \t approx={approx_method} ")
                test_func(signal, interval, verbose, **kwargs)

if __name__ == "__main__":

    test_all_settings(test_always, verbose=False, T=10)
    test_all_settings(test_eventually, verbose=False, T=10)
    test_all_settings(test_until, verbose=False, T=10)
    test_all_settings(test_always_mask_recurrent, verbose=False, T=10)
    test_all_settings(test_eventually_mask_recurrent, verbose=False, T=10)
    test_all_settings(test_until_mask_recurrent, verbose=False, T=10)


