"""Reproduce the existing d64,N8 fixed-space audit and display source patterns."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/piecewise_forcing_2d_matplotlib')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import model, residual_gram, ascend

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'runs/piecewise_forcing_2d_20261112'
OUT = BASE / 'fixed_space_illustration'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, N, seed = 64, 8, 1000
    B, G, checks = model(8)
    saved = np.load(BASE / 'd64_snapshots.npz')
    Q = np.linalg.qr(B @ saved['z_s1000_pool10000_n8'].T)[0]
    H = residual_gram(B, G, Q)
    initial = np.random.default_rng(2650008).uniform(-1, 1, d)
    gd = ascend(H, initial, 80)
    pool = np.random.default_rng(1650008).uniform(-1, 1, (10000, d))
    values = np.sum((pool @ H) * pool, axis=1)
    selected_pool = pool[np.argmax(values)]
    rows = []
    for name, z in [('Initial source', initial), ('GD: 80 steps', gd), ('Best of pool 10000', selected_pool)]:
        J = float(z @ H @ z)
        gain = float(z @ H @ H @ z / (3 * J))
        norm2 = float(z @ z / d)
        rows.append(dict(label=name, J=J, mean_squared_energy_error_drop=gain,
                         source_L2_squared=norm2, J_per_source_L2_squared=J/norm2))
    old = json.loads((BASE / 'same_space_gain.json').read_text())
    old = next(r for r in old['results'] if r['d'] == d and r['N'] == N and r['basis_seed'] == seed)
    assert np.isclose(rows[1]['J'], old['J_gd'], rtol=1e-12)
    assert np.isclose(rows[2]['J'], old['J_pool'], rtol=1e-12)
    report = dict(d=d, N=N, basis_method='pool10000', basis_seed=seed,
                  uniform_mean_squared_energy_error_before=float(np.trace(H)/3),
                  results=rows, checks=checks)
    (OUT / 'd64_N8.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(OUT / 'd64_N8_parameters.npz', initial=initial, gd=gd, pool=selected_pool)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4), layout='constrained')
    for ax, z, row in zip(axes, [initial, gd, selected_pool], rows):
        im = ax.imshow(z.reshape(8, 8).T, origin='lower', extent=[0,1,0,1],
                       vmin=-1, vmax=1, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                           'blue_white_orange', ['#0072B2', '#F7F7F7', '#D55E00']), interpolation='nearest')
        ax.set_title(row['label'], fontsize=12)
        ax.set_xlabel(r'$x_1$'); ax.set_xticks([0,.5,1]); ax.set_yticks([0,.5,1])
    axes[0].set_ylabel(r'$x_2$')
    fig.colorbar(im, ax=axes, shrink=.83, label='Source amplitude')
    fig.savefig(OUT / 'd64_N8_sources.png', dpi=190)
    fig.savefig(OUT / 'd64_N8_sources.pdf')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':main()
