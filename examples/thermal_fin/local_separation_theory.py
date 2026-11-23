"""Exact coercive Poisson/RB separation example, not a thermal-fin surrogate.

Energy-orthonormal phi_k=sqrt(2)/(k*pi)*sin(k*pi*x), k=1,2,3.
u(t,z)=phi_1+t*phi_2+t*g(z)*phi_3 solves -u_xx=f(t,z), u(0)=u(1)=0.
Initial V1=span(phi1), from t=0. One selected snapshot produces rank two.
All reported worst relative energy errors are analytic suprema, not test maxima.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,brentq

A=np.sqrt(3.)-1
B=.8

def g_all(z):
    z=np.asarray(z);left=z<=B;w=np.where(left,B+1,1-B);start=np.where(left,-1.,B);amp=np.where(left,A,2.)
    q=np.pi*(z-start)/w;s=np.sin(q);c=np.cos(q)
    g=amp*s**4;gp=4*amp*np.pi/w*s**3*c;gpp=4*amp*(np.pi/w)**2*(3*s*s*c*c-s**4)
    return g,gp,gpp

def fg(x):
    t,z=x;g,gp,_=g_all(z)
    return -np.log(max(t*t*(1+g*g),1e-290)), -np.array([2/t,2*g*gp/(1+g*g)])

def error(a):
    a=np.asarray(a)
    return np.maximum(a/np.sqrt(2),(2-a)/np.sqrt(6))/np.sqrt(1+a*a)

EOPT=float(error(A))

def success_mass(delta):
    eps=EOPT+delta
    lo=0. if eps>=error(0) else brentq(lambda a:error(a)-eps,0,A,xtol=1e-14)
    hi=2. if eps>=error(2) else brentq(lambda a:error(a)-eps,A,2,xtol=1e-14)
    left=(B+1)*.5*(1-2/np.pi*np.arcsin((lo/A)**.25))
    right=(1-B)/np.pi*(np.arcsin((hi/2)**.25)-np.arcsin((lo/2)**.25))
    return float(left+right),float(lo),float(hi)

def numerical_study():
    rng=np.random.default_rng(12039);anchors=rng.uniform(-1,1,(2000,2));local=[]
    for steps in [5,10,20,1000]:
        errs=[];types=[];counts=[];start=time.perf_counter()
        for x in anchors:
            r=minimize(fg,x,jac=True,method='L-BFGS-B',bounds=[(-1,1)]*2,
                       options={'maxiter':steps,'gtol':1e-10,'ftol':0.,'maxls':50})
            gg,gp,gpp=g_all(r.x[1]);errs.append(float(error(gg)));counts.append(int(r.nit))
            h=2*((gp*gp+gg*gpp)/(1+gg*gg)-2*gg*gg*gp*gp/(1+gg*gg)**2)
            stationary=abs(2*gg*gp/(1+gg*gg))<1e-7 and abs(abs(r.x[0])-1)<1e-7 and h< -1e-6
            types.append('nonglobal_local' if stationary and abs(gg-A)<1e-6 else 'global' if stationary and abs(gg-2)<1e-6 else 'unresolved')
        e=np.array(errs);local.append({'maxiter':steps,'trials':len(e),'median_error':float(np.median(e)),'mean_error':float(e.mean()),'success_within_1e-4':float(np.mean(e<=EOPT+1e-4)),
                                     'nonglobal_local_count':types.count('nonglobal_local'),'global_count':types.count('global'),'unresolved_count':types.count('unresolved'),
                                     'median_iterations':float(np.median(counts)),'max_iterations':max(counts),'seconds':time.perf_counter()-start})
    pools=[]
    for m in [16,64,256,2000,4000,20000]:
        actual=[];oracle=[];start=time.perf_counter()
        for batch in range(20):
            t=rng.uniform(-1,1,(100,m));z=rng.uniform(-1,1,(100,m));gg=g_all(z)[0];j=t*t*(1+gg*gg);indices=j.argmax(axis=1)
            actual.extend(error(gg[np.arange(100),indices]));oracle.extend(np.min(error(gg),axis=1))
        e=np.array(actual);best=np.array(oracle);pools.append({'size':m,'trials':len(e),'median_actual_greedy_error':float(np.median(e)),
                    'mean_actual_greedy_error':float(e.mean()),'actual_success_within_1e-4':float(np.mean(e<=EOPT+1e-4)),
                    'median_best_possible_pool_error':float(np.median(best)),'best_possible_success_within_1e-4':float(np.mean(best<=EOPT+1e-4)),
                    'seconds':time.perf_counter()-start})
    bounds=[]
    for delta in [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]:
        p,lo,hi=success_mass(delta)
        bounds.append({'delta':delta,'success_mass':p,'required_M_for_90pct':int(np.ceil(np.log(.1)/np.log1p(-p))),'a_lower':lo,'a_upper':hi})
    result={'model':'exact coercive Poisson, two parameters, rank 1 -> 2','a_local':float(A),'relative_error_local':EOPT,'relative_error_global':float(error(2)),
            'gradient_flow_nonglobal_basin_mass':(B+1)/2,'local_search':local,'random_pool':pools,'best_possible_pool_lower_bound':bounds}
    Path('local_separation_theory_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':numerical_study()
