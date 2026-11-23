"""Update only slide probability figures: T GD updates versus T+1 iid points.

Existing GD bounds and greedy histories are reused without alteration. Sphere
and ball already have fresh-pool-41 histories. The missing cube comparator is
run with 81 fresh uniform points per round, using the existing model and seeds.
Empirical plots retain their original meaning: the final selection in eight
method-specific greedy histories, rather than fixed-space probability trials.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matched_probability_matplotlib')
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from threadpoolctl import threadpool_limits
from .geometry_cpu import CoordinateROM, rom_options
from .cube.search import scores
from .common.box_bounds import maximum

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/analytic_hf_a06_t40_20261022'
CUBE = ROOT / 'runs/cube_slides_a06_t80_20261102'
OUT = ROOT / 'runs/matched_probability_figures_20261105'
FIG = ROOT / 'paper/slides_short/figs'
NAMES = ['step_size_probability_bounds', 'iteration_probability_bounds',
         'probability_bounds_dimension', 'empirical_probability_dimension',
         'cube_empirical_probability_dimension']
COLORS = ['#27647B', '#B1813D', '#527B61', '#A55359', '#64717D']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.titlesize': 14, 'axes.labelsize': 12, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.facecolor': '#FAFAF8',
    'axes.facecolor': '#FAFAF8', 'savefig.facecolor': '#FAFAF8',
    'axes.edgecolor': '#68727A', 'text.color': '#263039',
    'axes.labelcolor': '#263039', 'pdf.fonttype': 42, 'ps.fonttype': 42})


def load(path):
    return json.loads(path.read_text())


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def pool_bound(p, count):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = -np.expm1(np.asarray(count) * np.log1p(-p))
    return np.where(p >= 1, 1., result)


def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(FIG / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def theory():
    curves = load(SOURCE / 'probability_curves.json')
    dims = load(SOURCE / 'results.json')['config']['dimensions']
    rows = {(r['domain'], r['dimension']): r for r in curves
            if r['dimension'] in dims and r['rank'] == r['dimension'] // 2}
    assert all(r['lambda2_ratio'] < r['gamma']**2 for r in rows.values())
    exported = []
    for name, source, xkey, xlabel in [
        ('step_size_probability_bounds', 'step_size_bounds.json', 'steps', r'Step scale $\alpha$'),
        ('iteration_probability_bounds', 'iteration_bounds.json', 'T', r'Number of updates $T$')]:
        records = load(SOURCE / source)
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0), sharey=True)
        for dom, ax in zip(['sphere', 'ball'], axes):
            for d, color in zip(dims, COLORS):
                record = next(r for r in records if r['domain'] == dom and r['d'] == d)
                x = np.asarray(record[xkey])
                lower = np.asarray(record['lower_bound'])
                count = np.full(x.shape, 41) if xkey == 'steps' else x + 1
                upper = pool_bound(rows[dom, d]['direct_analytic_upper'], count)
                ax.plot(x, lower, color=color, lw=2.4, ls='-')
                ax.plot(x, upper, color=color, lw=2.0, ls=(0, (4, 2)))
                exported.append(dict(figure=name, domain=dom, d=d, x=x.tolist(),
                    gd_lower=lower.tolist(), pool_count=count.tolist(), pool_upper=upper.tolist()))
                assert np.all((upper >= 0) & (upper <= 1))
                if xkey == 'T':
                    assert np.all(np.diff(upper) >= -1e-15)
            ax.set(title=dom.capitalize(), xlabel=xlabel, ylim=(0, 1.025), xlim=(x[0], x[-1]))
            ax.grid(alpha=.17)
            if xkey == 'T':
                ax.axvline(40, color='#68727A', ls=':', lw=1, alpha=.65)
        axes[0].set_ylabel('Probability bound')
        method_handles = [Line2D([], [], color='#263039', lw=2.4, label='GD: lower bound'),
                          Line2D([], [], color='#263039', lw=2, ls=(0, (4, 2)),
                                     label=r'Pool: upper bound, $M=T+1$')]
        fig.legend(handles=method_handles, loc='lower center', bbox_to_anchor=(.52, .067),
                   frameon=False, ncol=2, fontsize=11)
        fig.legend(handles=[Line2D([], [], color=c, lw=2.5, label=f'd = {d}')
                            for d, c in zip(dims, COLORS)], loc='lower center',
                   bbox_to_anchor=(.52, -.005), frameon=False, ncol=len(dims), fontsize=11)
        fig.subplots_adjust(left=.08, right=.985, top=.91, bottom=.27, wspace=.18)
        save(fig, name)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    for dom, method, ax in zip(['sphere', 'ball'], ['exp_rgd', 'normalized_pg'], axes):
        rr = [rows[dom, d] for d in dims]
        snapshots = [next(s for s in r['methods'][method]['snapshots'] if s['T'] == 40) for r in rr]
        lower = [s['analytic_lower'] for s in snapshots]
        upper = pool_bound([r['direct_analytic_upper'] for r in rr], 41)
        ax.plot(dims, lower, color='#245E91', lw=2.8, ls='-', marker='o', mfc='white', ms=7,
                label=r'GD: lower bound for $q_{40}(\tau)$')
        ax.plot(dims, upper, color='#BC762C', lw=2.3, ls='--', marker='^', mfc='white', ms=7,
                label=r'Pool: upper bound for $P_{41}(\tau)$')
        ax.set(title=dom.capitalize(), xlabel=r'Parameter dimension $d$', xticks=dims,
               yscale='log', ylim=(None, 1.5))
        ax.grid(alpha=.16, which='major')
        exported.append(dict(figure='probability_bounds_dimension', domain=dom,
            dimensions=dims, T=40, M=41, gd_lower=lower, pool_upper=upper.tolist()))
    axes[0].set_ylabel('Probability bound')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(.52, .01), fontsize=11)
    fig.subplots_adjust(left=.085, right=.985, top=.91, bottom=.19, wspace=.2)
    save(fig, 'probability_bounds_dimension')
    dump('theoretical_comparisons.json', exported)


def empirical_sphere_ball():
    data = load(SOURCE / 'results.json')
    dims = data['config']['dimensions']
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    for dom, ax in zip(['sphere', 'ball'], axes):
        methods = [('exp_rgd40' if dom == 'sphere' else 'normalized_pg40',
                    'GD, $T=40$', '#245E91', '-', 'o'),
                   ('fresh_pool41', 'Pool, $M=41$', '#BC762C', '--', '^')]
        for method, label, color, style, marker in methods:
            ys = []
            for d in dims:
                rr = sorted([r for r in data['runs'] if (r['domain'], r['dimension'], r['method'])
                             == (dom, d, method)], key=lambda r: r['seed'])
                assert len(rr) == 8
                hits = [int(r['rounds'][-1]['indicator_quality'] >= .9) for r in rr]
                ys.append(float(np.mean(hits)))
                records.append(dict(domain=dom, d=d, method=method, T=40, M=41,
                                    seeds=[r['seed'] for r in rr], hits=hits, frequency=ys[-1]))
            ax.plot(dims, np.maximum(ys, np.finfo(float).eps), color=color, lw=2.3,
                    ls=style, marker=marker, ms=7, mfc='white', mew=1.5, label=label)
        ax.set(title=dom.capitalize(), xlabel=r'Parameter dimension $d$', xticks=dims,
               yscale='log', ylim=(np.finfo(float).eps/3, 2), yticks=[1, 1e-4, 1e-8, 1e-12, 1e-16])
        ax.grid(alpha=.16, which='major')
    axes[0].set_ylabel('Empirical attainment frequency')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(.52, .01), fontsize=11)
    fig.subplots_adjust(left=.10, right=.985, top=.91, bottom=.20, wspace=.20)
    save(fig, 'empirical_probability_dimension')
    dump('sphere_ball_empirical.json', records)


def cube_pool(data):
    path = OUT / 'cube_fresh_pool81.json'
    if path.exists():
        return load(path)
    records = []
    for d in [d for d in data['config']['dimensions'] if d >= 8]:
        model = np.load(SOURCE / f'model_d{d}.npz')
        for seed in data['config']['seeds']:
            rom = CoordinateROM(model['G0'], model['H0'], **rom_options(model))
            rng = np.random.default_rng(seed + 1000003)
            rounds = []
            for n in range(d//2):
                H, _ = rom.matrices()
                candidates = rng.uniform(-1, 1, (81, d))
                values = scores(H, candidates)
                z = candidates[int(np.argmax(values))].copy()
                value = float(z @ H @ z)
                rounds.append(dict(n=n, parameter=z.tolist(), selected_value=value))
                rom.append(z)
            lo, hi = maximum(H)
            record = dict(d=d, N=d//2, method='fresh_pool81', seed=seed, T=80, M=81,
                rounds=rounds, indicator_max_interval=[lo, hi],
                indicator_quality_interval=np.sqrt([value/hi, value/lo]).tolist())
            records.append(record)
            np.savez_compressed(OUT / f'cube_pool81_d{d}_s{seed}.npz', Q=rom.Q, final_search_H=H)
        print(f'Cube pool81: d={d}, eight histories complete', flush=True)
    dump(path.name, records)
    return records


def empirical_cube():
    data = load(CUBE / 'results.json')
    assert data['config']['updates'] == 80
    new = cube_pool(data)
    dims = [d for d in data['config']['dimensions'] if d >= 8]
    fig, ax = plt.subplots(figsize=(9, 4.3))
    records = []
    for method, source, label, color, style, marker in [
        ('gd', data['runs'], 'GD, $T=80$', '#245E91', '-', 'o'),
        ('fresh_pool81', new, 'Pool, $M=81$', '#BC762C', '--', '^')]:
        values = []
        for d in dims:
            rr = sorted([r for r in source if r['d'] == d and r['method'] == method], key=lambda r: r['seed'])
            assert len(rr) == 8
            bounds = np.mean(np.asarray([r['indicator_quality_interval'] for r in rr]) >= .9, axis=0)
            values.append(bounds)
            records.append(dict(d=d, method=method, T=80, M=81, frequency_interval=bounds.tolist(), runs=len(rr)))
        values = np.asarray(values)
        floor = np.finfo(float).eps
        ax.plot(dims, np.maximum(values[:, 0], floor), label=label, color=color,
                ls=style, marker=marker, ms=7, lw=2.3, mfc='white')
        if np.any(values[:, 0] != values[:, 1]):
            ax.fill_between(dims, np.maximum(values[:, 0], floor), np.maximum(values[:, 1], floor), color=color, alpha=.2)
    ax.set(xlabel=r'Parameter dimension $d$', ylabel='Empirical attainment frequency',
           xticks=dims, yscale='log', ylim=(np.finfo(float).eps/3, 2), yticks=[1, 1e-4, 1e-8, 1e-12, 1e-16])
    ax.grid(alpha=.16)
    ax.legend(frameon=False, fontsize=11, loc='center right')
    fig.tight_layout()
    save(fig, 'cube_empirical_probability_dimension')
    dump('cube_empirical.json', records)


def main():
    OUT.mkdir(exist_ok=True)
    tex = ROOT / 'paper/slides_short/slides.tex'
    before = hashlib.sha256(tex.read_bytes()).hexdigest()
    backup = OUT / 'figures_before'
    backup.mkdir(exist_ok=True)
    for name in NAMES:
        for ext in ['pdf', 'png']:
            source = FIG / f'{name}.{ext}'
            if source.exists() and not (backup / source.name).exists():
                shutil.copy2(source, backup / source.name)
    with threadpool_limits(limits=1):
        theory()
        empirical_sphere_ball()
        empirical_cube()
    assert hashlib.sha256(tex.read_bytes()).hexdigest() == before
    dump('manifest.json', dict(slides_tex_sha256=before, figures=NAMES,
        comparison='T GD updates versus T+1 independent uniform points',
        empirical_scope='Final selection producing N=d/2, eight method-specific greedy histories; fresh iid pool per round',
        source_runs=[str(SOURCE), str(CUBE)], device='CPU float64, one BLAS thread',
        original_logs_modified=False, tex_modified=False))
    print('Five probability figures updated; slides.tex unchanged.', flush=True)


if __name__ == '__main__':
    main()
