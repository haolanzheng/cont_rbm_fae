"""Update only meeting result/timing tables from archived and added runs."""
import json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'runs/high_dimension_tables_20261113'
SLIDES=ROOT/'paper/slides_meeting_20261113/slides.tex'
DIMS=[4,8,16,32,64,128]
METHODS=['gd','pool256','pool4096','pool10000','grid1','grid0.5','grid0.25']

def main():
    high=json.loads((OUT/'results.json').read_text())['runs']
    times=json.loads((ROOT/'runs/search_timing_20261113/timings.json').read_text())+json.loads((OUT/'timings.json').read_text())
    time_map={(r['domain'],r['d'],r['method']):r['wall_ms_median'] for r in times}
    source=SLIDES.read_text();original=source
    (OUT/'slides_before_table_extension.tex').write_text(source)
    summaries=[]
    for title,domains,timing in [
        ('Sphere and ball: final test errors across dimensions',['sphere','ball'],False),
        ('Sphere and ball: wall time',['sphere','ball'],True),
        ('Cube: final test errors across dimensions',['cube'],False),
        ('Cube: wall time',['cube'],True)]:
        a=source.index('\\begin{frame}{'+title+'}');b=source.index('\\end{frame}',a)+len('\\end{frame}')
        frame=source[a:b];old={};domain=domains[0]
        # Retain existing low-dimensional error cells, including their emphasis.
        for line in frame.splitlines():
            if '&' not in line:continue
            cells=[v.strip() for v in line.removesuffix(r'\\').split('&')]
            if len(domains)>1:
                if cells[0].lower() in domains:domain=cells[0].lower()
                if len(cells)<2 or not cells[1].isdigit():continue
                d=int(cells[1]);old[domain,d]=cells[2:]
            else:
                if not cells[0].isdigit():continue
                old[domain,int(cells[0])]=cells[1:]
        multi=len(domains)>1;T=40 if multi else 80
        lines=['\\begin{frame}{'+title+'}',r'\small',
               r'Target dimension $N=d/2$. '+('Wall time in milliseconds.' if timing else r'Test-set maximum $A$-norm error.'),
               r'\begin{center}',r'\scriptsize',r'\renewcommand{\arraystretch}{1.15}\setlength{\tabcolsep}{'+'3pt'+'}',
               r'\begin{tabular}{'+('l' if multi else '')+r'rrrrrrrr}\toprule',
               ('Domain & ' if multi else '')+r'$d$ & GD & Pool $256$ & Pool $4096$ & Pool $10000$ & \multicolumn{3}{c}{Uniform mesh}\\',
               (' & ' if multi else '')+r' & $T='+str(T)+r'$ & & & & $h=1$ & $h=1/2$ & $h=1/4$\\\midrule']
        for j,domain in enumerate(domains):
            if j:lines.append(r'\midrule')
            for i,d in enumerate(DIMS):
                cells=[]
                for k,method in enumerate(METHODS):
                    if d>=128:
                        rr=[r for r in high if (r['domain'],r['d'],r['method'])==(domain,d,method)]
                        if not rr:cells.append('---');continue
                        value=time_map[domain,d,method] if timing else float(np.median([r['test_max_error'] for r in rr]))
                        stopped=not all(r['reached_target'] for r in rr)
                        label=f'{value:.2f}' if timing else f'{value:.2e}'
                        if not timing:
                            mantissa,exponent=label.split('e');label=mantissa+r'\!\times\!10^{'+str(int(exponent))+'}'
                        annotation=''
                        if timing and stopped:
                            ranks=[r['rank'] for r in rr]
                            annotation=r'\,('+str(min(ranks))+r'\text{--}'+str(max(ranks))+')'
                        cells.append('$'+('{'+label+'}'+r'^{\dagger}' if stopped else label)+annotation+'$')
                        if not timing:summaries.append(dict(domain=domain,d=d,method=method,median_error=value,
                            rank_min=min(r['rank'] for r in rr),rank_max=max(r['rank'] for r in rr),complete_runs=sum(r['reached_target'] for r in rr)))
                    else:
                        if timing:
                            cells.append(f'{time_map[domain,d,method]:.2f}' if (domain,d,method) in time_map else '---')
                        else:cells.append(old[domain,d][k])
                prefix=((domain.capitalize() if i==0 else '')+' & ') if multi else ''
                lines.append(prefix+str(d)+' & '+' & '.join(cells)+r'\\')
        lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{center}',r'\footnotesize',
            r'Eight-seed medians for GD and pools; uniform mesh shown at $d=4$.\\',
            r'$\dagger$: stopped before $N=d/2$; '+('time' if timing else 'error')+r' at the attained dimension.',r'\end{frame}'])
        new_frame='\n'.join(lines)
        if timing:
            new_frame=new_frame.replace(r'$\dagger$: stopped before $N=d/2$; time at the attained dimension.',r'$\dagger$: stopped early; parentheses give the attained $N$ range over eight seeds.')
        if multi or timing:
            new_frame=new_frame.replace(r'\begin{tabular}', r'\resizebox{\linewidth}{!}{%'+'\n'+r'\begin{tabular}',1).replace(r'\end{tabular}',r'\end{tabular}}',1)
        source=source[:a]+new_frame+source[b:]
    assert source!=original
    SLIDES.write_text(source)
    (OUT/'table_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print('Updated four tables; boundary-update comparison unchanged.')

if __name__=='__main__':main()
