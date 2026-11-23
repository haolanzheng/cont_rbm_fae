"""Independent direct-draw audit for the rare-event cube pool probabilities."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/continuous_cube_matplotlib")
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from threadpoolctl import threadpool_limits, threadpool_info
from .cube_geometry_cpu import values


def make_figure(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    direct = json.loads((output/"direct_probability_large_audit.json").read_text())["records"]
    audits = json.loads((output/"probability_audits.json").read_text())
    dims = [r["dimension"] for r in direct]
    gd = [next(a for a in audits if a["dimension"]==d and a["rank"]==d//2)["trajectories"]["relaxed_pg100"]["snapshots"]["100"] for d in dims]
    fig, ax = plt.subplots(figsize=(7.8,4.5),constrained_layout=True)
    ax.plot(dims,[a["q_hat"] for a in gd],"o-",color="#265B7D",label="Projected GD, T=100")
    ax.fill_between(dims,[a["q_ci95"][0] for a in gd],[a["q_ci95"][1] for a in gd],color="#265B7D",alpha=.16)
    for m,color in ((101,"#777777"),(256,"#B48442"),(4096,"#55806C"),(10000,"#9B6267")):
        entries = [r["fresh_iid_pools"][str(m)] for r in direct]
        # Draw upper CI endpoints rather than falsely depicting zero-hit pools
        # as an exact zero success probability.
        ax.plot(dims,[v["ci95"][1] for v in entries],"s--",ms=4,color=color,label=f"Random pool M={m}: 95% upper endpoint")
        ax.fill_between(dims,[v["ci95"][0] for v in entries],[v["ci95"][1] for v in entries],color=color,alpha=.08)
    ax.set(xlabel="Parameter dimension d",ylabel="Success probability",ylim=(-.025,1.025),xticks=dims,
           title="Cube: fixed exact-vertex reference space, N=d/2")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8,loc="center right")
    for ext in ("png","pdf"):
        fig.savefig(output/f"cube_probability_dimension.{ext}",dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(12,6.5),constrained_layout=True)
    for ax,d,record in zip(axes.flat,dims,direct):
        entries = next(a for a in audits if a["dimension"]==d and a["rank"]==d//2)["trajectories"]["relaxed_pg100"]["snapshots"]
        times = np.array([int(t) for t in entries])
        gd_values = list(entries.values())
        ax.plot(times,[v["q_hat"] for v in gd_values],".-",color="#265B7D",label="Projected GD: estimate")
        ax.fill_between(times,[v["q_ci95"][0] for v in gd_values],[v["q_ci95"][1] for v in gd_values],color="#265B7D",alpha=.18)
        pool_lo = -np.expm1((times+1)*np.log1p(-record["ci95"][0]))
        pool_hi = -np.expm1((times+1)*np.log1p(-record["ci95"][1]))
        ax.plot(times,pool_hi,"--",color="#B48442",label="Pool M=T+1: 95% upper endpoint")
        ax.fill_between(times,pool_lo,pool_hi,color="#B48442",alpha=.16)
        if record["hits"]:
            ax.plot(times,-np.expm1((times+1)*np.log1p(-record["p_hat"])),":",color="#B48442",label="Pool M=T+1: estimate")
        ax.set(title=f"d={d}, N={d//2}",xlabel="Updates T",ylabel="Success probability",ylim=(-.02,1.02))
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.18)
    axes.flat[-1].axis("off")
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc="center",frameon=False,fontsize=9)
    fig.suptitle("Cube: equal H-action budget, one GD start versus T+1 random candidates\nPointwise 95% binomial intervals; 16,384 trajectories and 2,000,000 independent direct draws")
    for ext in ("png","pdf"):
        fig.savefig(output/f"cube_success.{ext}",dpi=180)
    plt.close(fig)


def main():
    output = Path("runs/cube_cpu_20261015")
    audits = json.loads((output/"probability_audits.json").read_text())
    result = []
    with threadpool_limits(limits=1):
        for record in audits:
            d, n = record["dimension"], record["rank"]
            if n != d//2:
                continue
            H = np.load(output/f"frozen_cube_d{d}_n{n}.npz")["H"]
            count, hits = 2_000_000, 0
            seed = 2026101500+d
            rng = np.random.default_rng(seed)
            for start in range(0,count,20000):
                z = rng.uniform(-1,1,(min(20000,count-start),d))
                hits += int(np.sum(values(H,z)>=record["gamma"]**2*record["J_max"]))
            lo = float(beta.ppf(.025,hits,count-hits+1)) if hits else 0.0
            hi = float(beta.ppf(.975,hits+1,count-hits)) if hits<count else 1.0
            p = hits/count
            pools = {str(m): {"estimate":float(-np.expm1(m*np.log1p(-p))),
                              "ci95":[float(-np.expm1(m*np.log1p(-lo))),float(-np.expm1(m*np.log1p(-hi)))]}
                     for m in (101,256,4096,10000)}
            result.append({"dimension":d,"rank":n,"seed":seed,"draws":count,"hits":hits,
                           "p_hat":p,"ci95":[lo,hi],"fresh_iid_pools":pools})
            print(f"d={d}: {hits}/{count}; pool101={pools['101']}; pool10000={pools['10000']}",flush=True)
        (output/"direct_probability_large_audit.json").write_text(json.dumps({"device":"CPU","dtype":"float64","threadpools":threadpool_info(),"records":result},indent=2)+"\n")
        make_figure(output)


if __name__=="__main__":
    main()
