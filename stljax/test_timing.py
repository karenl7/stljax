import jax
import jax.numpy as jnp
import numpy as np
from stljax.formula import *
from stljax.viz import *
import matplotlib.pyplot as plt
import timeit
import statistics
import pickle
import sys

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

if __name__ == "__main__":

    args = sys.argv[1:]
    bs = int(args[0])
    max_T = int(args[1])
    device = str(args[2])
    filename = "../results/timing_test_jax_bs_%i_maxT_%i_%s"%(bs, max_T, device)

    pred = Predicate('x', lambda x: x)
    interval = None
    mask = Always(pred > 4, interval=interval)
    recurrent = AlwaysRecurrent(pred > 4, interval=interval)



    def grad_mask(signal):
        return jax.vmap(jax.grad(lambda x: mask(x).mean()))(signal)
    def grad_recurrent(signal):
        return jax.vmap(jax.grad(lambda x: recurrent(x).mean()))(signal)

    def mask_(signal):
        return jax.vmap(lambda x: mask(x).mean())(signal)
    def recurrent_(signal):
        return jax.vmap(lambda x: recurrent(x).mean())(signal)


    @jax.jit
    def grad_mask_jit(signal):
        return jax.vmap(jax.grad(lambda x: mask(x).mean()))(signal)
    @jax.jit
    def grad_recurrent_jit(signal):
        return jax.vmap(jax.grad(lambda x: recurrent(x).mean()))(signal)
    @jax.jit
    def mask_jit(signal):
        return jax.vmap(lambda x: mask(x).mean())(signal)
    @jax.jit
    def recurrent_jit(signal):
        return jax.vmap(lambda x: recurrent(x).mean())(signal)

    # Number of loops per run
    loops = 100
    # Number of runs
    runs = 25
    T = 2
    data = {}
    Ts = []
    functions = ["mask_", "recurrent_", "grad_mask", "grad_recurrent", "mask_jit", "recurrent_jit", "grad_mask_jit", "grad_recurrent_jit"]


    data["functions"] = functions
    data["runs"] = runs
    data["loops"] = loops

    while T <= max_T:
        Ts.append(T)
        data['Ts'] = Ts
        print("running ", T)
        signal = jnp.array(np.random.random([bs, T]))
        times_list = []
        data[str(T)] = {}
        for f in functions:
            print("timing ", f)
            timeit.repeat(f + "(signal)", globals=globals(), repeat=1, number=1)
            times = timeit.repeat(f + "(signal)", globals=globals(), repeat=runs, number=loops)
            times_list.append(times)
            print("timing: ", statistics.mean(times), statistics.stdev(times))
            data[str(T)][f] = times
            with open(filename + '.pkl', 'wb') as f:
                pickle.dump(data, f)

        T *= 2


    means = []
    stds = []
    for k in data.keys():
        if k in ["Ts", "functions", "loops", "runs"]:
            continue
        mus = []
        sts = []
        for f in data[k].keys():
            mus.append(statistics.mean(data[k][f])/data["loops"])
            sts.append(statistics.stdev(data[k][f])/data["loops"])

        means.append(mus)
        stds.append(sts)
    means = np.array(means)
    stds = np.array(stds)

    fontsize = 14

    plt.figure(figsize=(10,5))
    plt.plot(data["Ts"], means * 1E3)
    for (m,s) in zip(means.T, stds.T):
        plt.fill_between(data["Ts"], (m - s) * 1E3, (m + s) * 1E3, alpha=0.3)
    plt.yscale("log")
    plt.legend(functions, fontsize=fontsize-2)
    plt.grid()
    plt.xlabel("Signal length", fontsize=fontsize)
    plt.ylabel("Computation time [ms]", fontsize=fontsize)
    plt.title("JAX " + str(device), fontsize=fontsize+2)
    plt.tight_layout()

    plt.savefig(filename + ".png", dpi=200, transparent=True)
