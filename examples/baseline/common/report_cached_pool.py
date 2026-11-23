"""Update only cached-pool timing tables and directly affected cost statements."""
from pathlib import Path
import json,re

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/energy_cached_pool_20261120'
METHODS=['gd','pool256','pool4096','pool10000','grid1','grid0.5','grid0.25']

def update_rows(block,times,domains):
    domain=domains[0];multi=len(domains)>1;result=[]
    for line in block.splitlines():
        cells=line.removesuffix(r'\\').split('&');di=1 if multi else 0
        if len(cells)>di+1 and cells[di].strip().isdigit():
            d=int(cells[di]);domain=cells[0].strip().lower() if multi and cells[0].strip() else domain
            vals=[times.get((domain,d,m)) for m in METHODS]
            line=' & '.join(cells[:di+1])+' & '+' & '.join('---' if v is None else f'${v:.2f}$' for v in vals)+r'\\'
        result.append(line)
    return '\n'.join(result)


def update_frame(s,title,times,domains):
    a=s.index(r'\begin{frame}{'+title+'}');b=s.index(r'\end{frame}',a)
    return s[:a]+update_rows(s[a:b],times,domains)+'\n'+s[b:]


def update_table(s,label,times,domains):
    end=s.index(r'\label{'+label+'}');a=s.rfind(r'\begin{table}',0,end)
    return s[:a]+update_rows(s[a:end],times,domains)+s[end:]


def main():
    rows=json.loads((RUN/'timings.json').read_text());times={(r['domain'],r['d'],r['method']):r['wall_ms'] for r in rows}
    backup=RUN/'documents_before';backup.mkdir(exist_ok=True)
    for name in ['slides_short','slides_meeting_20261113']:
        p=ROOT/'paper'/name/'slides.tex';s=p.read_text();(backup/f'{name}.tex').write_text(s)
        s=update_frame(s,'Sphere and ball: wall time',times,['sphere','ball'])
        s=update_frame(s,'Cube: wall time',times,['cube'])
        a=s.index(r'\begin{frame}{Search cost and parameter dimension}');b=s.index(r'\end{frame}',a)+len(r'\end{frame}')
        s=s[:a]+r'''\begin{frame}{Search cost and parameter dimension}
\small
\begin{center}
\renewcommand{\arraystretch}{1.4}\setlength{\tabcolsep}{8pt}
\begin{tabular}{ll}\toprule
Operation & Dominant cost\\\midrule
Evaluate $J_n$ and its gradient & $2d^2$\\
GD, $T$ updates & $2(T+1)d^2$\\
Fixed pool: precompute $M$ responses once & $2Md^2$\\
Pool: rank-one score update after enrichment & $2Md$\\\bottomrule
\end{tabular}
\end{center}
For a cube grid with $k$ points per coordinate, $M=k^d$.

Wall times include pool precomputation and stable residual refreshes.
\end{frame}'''+s[b:]
        s=s.replace('Shared HF assembly and solves are excluded.', 'Five repeats per seed; pool precomputation and score updates included.\\\\\nShared HF assembly, solves and test evaluation are excluded.')
        p.write_text(s)
    p=ROOT/'paper/writeup/writeup.tex';s=p.read_text();(backup/'writeup.tex').write_text(s)
    for label,doms in [('tab:sphere-time',['sphere','ball']),('tab:cube-time',['cube'])]:s=update_table(s,label,times,doms)
    old='Wall times include greedy training and exclude shared HF construction and test evaluation.'
    new='Fixed pools and meshes cache the HF solutions in energy coordinates and update squared errors after each basis enrichment. Residuals are recomputed from the cache when needed to avoid cancellation. Timings use five repeats per seed and include sampling, cache preparation, score updates, search, and enrichment; shared HF construction and test evaluation are excluded.'
    assert old in s;s=s.replace(old,new,1);p.write_text(s)
    p=ROOT/'paper/continuous_greedy_manifold_attainment.tex';s=p.read_text();(backup/'notebook.tex').write_text(s)
    end=s.index(r'\label{tab:baseline-error-time}');start=s.rfind(r'\begin{table}',0,end);block=s[start:end]
    labels=['GD, $T=40$','Pool $256$','Pool $4096$','Pool $10000$','Grid $h=1$','Grid $h=1/2$','Grid $h=1/4$']
    for label,m in zip(labels,METHODS):
        match=re.search('^'+re.escape(label)+r' &.*$',block,re.M);assert match
        cells=match[0].removesuffix(r'\\').split('&');new=[cells[0].strip()]
        for cell,(d,dom) in zip(cells[1:],[(6,'sphere'),(6,'ball'),(32,'sphere'),(32,'ball')]):
            new.append(cell.strip() if ' / ' not in cell else cell.split(' / ')[0].strip()+f' / {times[dom,d,m]:.2f}')
        block=block[:match.start()]+' & '.join(new)+r'\\'+block[match.end():]
    block=block.replace('Time includes candidate generation, matrix formation, search, and enrichment.', 'Time includes candidate generation, pool precomputation and score updates, search, and enrichment; five repeats per seed.')
    s=s[:start]+block+s[end:];p.write_text(s)
    print('Updated first-baseline timing tables and cached-pool descriptions; error results unchanged.')

if __name__=='__main__':main()
