"""Indicator-driven continuous greedy on three natural-structure screens.

Search never evaluates truth or a projection error.  Truth is called only to
enrich at the selected parameter and to evaluate a separate deployment test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import linalg

from examples.flow.natural_continuous_greedy_bench import (
    LiftUpModel, KolmogorovModel, MicrowaveModel, fixed_sphere_starts,
    orth_append, radam_ascent, rlbfgs_ascent,
)


OUT=Path(__file__).resolve().parent/"indicator_driven_results"


def pod_append(V,X):
    E=X-V@(V.T@X) if V.size else X
    q=np.linalg.svd(E,full_matrices=False)[0][:,0]
    return orth_append(V,q)


def summarize(rows):
    ans={}
    for method in sorted({r["method"] for r in rows}):
        rr=[r for r in rows if r["method"]==method]
        x=np.array([r["max_error"] for r in rr])
        ans[method]={
            "runs":len(rr),"max_error_median":float(np.median(x)),
            "max_error_mean":float(x.mean()),"max_error_q90":float(np.quantile(x,.9)),
            "max_error_max":float(x.max()),
            "mean_error_median":float(np.median([r["mean_error"] for r in rr])),
            "search_seconds_mean":float(np.mean([r["search_seconds"] for r in rr])),
            "indicator_calls_mean":float(np.mean([r["indicator_calls"] for r in rr])),
        }
    return ans


def trap_weights(times):
    w=np.ones(len(times));w[[0,-1]]=.5;return w*(times[-1]-times[0])/(len(times)-1)


# ---------------------------------------------------------------------------
# Linear lift-up: reduced-dynamics residual indicator


class LiftUpIndicator:
    def __init__(self,model):
        self.m=model;self.w=trap_weights(model.times)

    def matrix(self,V):
        m=self.m;Bn=V.T@m.B;E0=m.B-V@Bn
        H=E0.T@E0
        Ar=V.T@m.A@V;Rop=m.A@V-V@Ar
        for wt,t in zip(self.w,m.times):
            C=linalg.expm(Ar*t)@Bn
            R=Rop@C;H+=wt*(R.T@R)
        return .5*(H+H.T)

    def trajectory(self,z):
        return np.einsum("tnd,d->nt",self.m.Xmaps,z)

    def deployment(self,V,Zmean):
        m=self.m;Bn=V.T@m.B;Ar=V.T@m.A@V;Herr=np.zeros((m.d,m.d))
        mean_sq=np.zeros(len(Zmean))
        for wt,t,X in zip(self.w,m.times,m.Xmaps):
            R=V@(linalg.expm(Ar*t)@Bn);E=X-R
            Herr+=wt*(E.T@E);EZ=E@Zmean.T;mean_sq+=wt*np.sum(EZ*EZ,axis=0)
        return float(np.sqrt(np.linalg.eigvalsh(Herr)[-1])),float(np.mean(np.sqrt(mean_sq)))


def train_liftup(model,method,seed,add_rank,K=4,steps=8,lr=.1,pool_size=100):
    ind=LiftUpIndicator(model)
    # Exact representation of the parameterized initial-condition space.
    V=np.linalg.qr(model.B,mode="reduced")[0]
    starts=fixed_sphere_starts(seed,add_rank,K,model.d)
    rng=np.random.default_rng(seed+1801);pool=rng.normal(size=(pool_size,model.d));pool/=np.linalg.norm(pool,axis=1)[:,None]
    calls=0;seconds=0.;chosen=[]
    for k in range(add_rank):
        H=ind.matrix(V)
        if method=="pool":
            t0=time.perf_counter();vals=np.einsum("bi,ij,bj->b",pool,H,pool);seconds+=time.perf_counter()-t0
            z=pool[int(np.argmax(vals))];calls+=pool_size
        else:
            cand=[]
            fg=lambda x:(float(x@H@x),2*H@x)
            for j in range(K):
                t0=time.perf_counter()
                if method=="radam":zj,fj,info=radam_ascent(fg,starts[k,j],steps,lr)
                else:
                    zj,fj,raw=rlbfgs_ascent(fg,lambda x:fg(x)[0],starts[k,j],1.,steps)
                    info={"calls":raw["gradient_evaluations"]+raw["value_evaluations"]}
                seconds+=time.perf_counter()-t0;calls+=info["calls"];cand.append((fj,zj))
            z=max(cand,key=lambda q:q[0])[1]
        V,ok=pod_append(V,ind.trajectory(z));chosen.append(z.tolist())
        if not ok:break
    return V,{"indicator_calls":calls,"search_seconds":seconds,"chosen":chosen}


def run_liftup(quick=False):
    m=LiftUpModel(ny=36 if quick else 48);ind=LiftUpIndicator(m);add_rank=8
    seeds=[11] if quick else [11,22,33,44,55]
    rng=np.random.default_rng(914);Z=rng.normal(size=(3000 if quick else 20000,m.d));Z/=np.linalg.norm(Z,axis=1)[:,None]
    cfg=[]
    for lr in (.01,.03,.1,.3):cfg.append((f"radam_k4_s8_lr{lr}","radam",4,8,lr,None))
    cfg += [("radam_k4_s25_lr0.1","radam",4,25,.1,None),
            ("rlbfgs_k1_s8","rlbfgs",1,8,.1,None),("rlbfgs_k4_s8","rlbfgs",4,8,.1,None)]
    for M in (50,200,1000,10000):cfg.append((f"pool_{M}","pool",1,0,.1,M))
    rows=[]
    for seed in seeds:
        for name,method,K,steps,lr,M in cfg:
            V,info=train_liftup(m,method,seed,add_rank,K,steps,lr,M or 100)
            mx,me=ind.deployment(V,Z)
            row={"case":"liftup","method":name,"seed":seed,"initial_rank":m.d,
                 "final_rank":V.shape[1],"max_error":mx,"mean_error":me,**info}
            rows.append(row);print("LIFT-IND",name,seed,mx,flush=True)
    data={"case":"lift-up residual indicator","search_indicator":"initial projection error squared plus time-integrated Galerkin residual squared",
          "truth_during_search":False,"d":m.d,"initial_rank":m.d,"added_rank":add_rank,
          "final_rank":m.d+add_rank,"seeds":seeds,"rows":rows,"summary":summarize(rows)}
    (OUT/"liftup_indicator.json").write_text(json.dumps(data,indent=2));return data


# ---------------------------------------------------------------------------
# Nonlinear Kolmogorov flow: Galerkin trajectory residual


class KolmogorovIndicator:
    def __init__(self,m):self.m=m;self.P=m.Phi.reshape(m.d,-1)

    def initial_basis(self):return np.linalg.qr(self.P.T,mode="reduced")[0]

    def hf_trajectory(self,z):
        m=self.m;w=m.amplitude*np.einsum("d,dxy->xy",z,m.Phi);X=[w.reshape(-1).copy()]
        for _ in range(m.steps):w=m._step(w,m.dt);X.append(w.reshape(-1).copy())
        return np.column_stack(X)

    def _rom_rhs(self,V,a):
        W=(a@V.T).reshape(len(a),self.m.ngrid,self.m.ngrid)
        F=self.m.rhs(W).reshape(len(a),-1);da=F@V
        R=F-da@V.T
        return da,np.sum(R*R,axis=1)

    def values(self,V,Z):
        Z=np.atleast_2d(Z);W0=self.m.amplitude*(Z@self.P);a=W0@V
        E0=W0-a@V.T;eta=np.sum(E0*E0,axis=1);h=self.m.dt
        for _ in range(self.m.steps):
            k1,r2=self._rom_rhs(V,a);k2,_=self._rom_rhs(V,a+.5*h*k1)
            k3,_=self._rom_rhs(V,a+.5*h*k2);k4,_=self._rom_rhs(V,a+h*k3)
            eta+=h*r2;a=a+h*(k1+2*k2+2*k3+k4)/6
        return eta

    def fg(self,V,z,eps=2e-5):
        Z=np.repeat(np.asarray(z)[None,:],2*self.m.d+1,axis=0)
        for j in range(self.m.d):Z[1+2*j,j]+=eps;Z[2+2*j,j]-=eps
        f=self.values(V,Z);g=np.array([(f[1+2*j]-f[2+2*j])/(2*eps) for j in range(self.m.d)])
        return float(f[0]),g

    def deployment(self,V,Z):
        Z=np.atleast_2d(Z);m=self.m;W=m.amplitude*np.einsum("bd,dxy->bxy",Z,m.Phi)
        a=W.reshape(len(Z),-1)@V;err=np.sum((W.reshape(len(Z),-1)-a@V.T)**2,axis=1);h=m.dt
        for _ in range(m.steps):
            W=m._step(W,h)
            k1,_=self._rom_rhs(V,a);k2,_=self._rom_rhs(V,a+.5*h*k1)
            k3,_=self._rom_rhs(V,a+.5*h*k2);k4,_=self._rom_rhs(V,a+h*k3)
            a=a+h*(k1+2*k2+2*k3+k4)/6
            E=W.reshape(len(Z),-1)-a@V.T;err+=h*np.sum(E*E,axis=1)
        e=np.sqrt(err);return float(e.max()),float(e.mean())


def train_kol(m,method,seed,add_rank,K=4,steps=8,lr=.1,pool_size=100):
    ind=KolmogorovIndicator(m);V=ind.initial_basis();starts=fixed_sphere_starts(seed,add_rank,K,m.d)
    rng=np.random.default_rng(seed+373);pool=rng.normal(size=(pool_size,m.d));pool/=np.linalg.norm(pool,axis=1)[:,None]
    calls=0;seconds=0.;chosen=[]
    for k in range(add_rank):
        if method=="pool":
            t=time.perf_counter();vals=ind.values(V,pool);seconds+=time.perf_counter()-t
            z=pool[int(np.argmax(vals))];calls+=pool_size
        else:
            cand=[];fg=lambda x:ind.fg(V,x)
            for j in range(K):
                t=time.perf_counter()
                if method=="radam":zj,fj,info=radam_ascent(fg,starts[k,j],steps,lr)
                else:
                    zj,fj,raw=rlbfgs_ascent(fg,lambda x:fg(x)[0],starts[k,j],1.,steps)
                    info={"calls":raw["gradient_evaluations"]+raw["value_evaluations"]}
                seconds+=time.perf_counter()-t;calls+=info["calls"];cand.append((fj,zj))
            z=max(cand,key=lambda q:q[0])[1]
        V,ok=pod_append(V,ind.hf_trajectory(z));chosen.append(z.tolist())
        if not ok:break
    return V,{"indicator_calls":calls,"search_seconds":seconds,"chosen":chosen}


def run_kol(quick=False):
    amplitudes=(2.5,) if quick else (.5,2.5);seeds=[11] if quick else [11,22,33];rows=[];checks={}
    for amp in amplitudes:
        m=KolmogorovModel(n=12,re=40,T=2,dt=.05,amplitude=amp);ind=KolmogorovIndicator(m);add_rank=6
        rng=np.random.default_rng(618);z=rng.normal(size=m.d);z/=np.linalg.norm(z);V=ind.initial_basis()
        f,g=ind.fg(V,z,2e-5);_,g2=ind.fg(V,z,1e-5)
        checks[str(amp)]={"fd_step_halving_relative_difference":float(np.linalg.norm(g-g2)/max(np.linalg.norm(g2),1e-14)),"indicator":f}
        rt=np.random.default_rng(732);Z=rt.normal(size=(500 if quick else 1500,m.d));Z/=np.linalg.norm(Z,axis=1)[:,None]
        cfg=[("radam_k4_s8_lr0.01","radam",4,8,.01,None),("radam_k4_s8_lr0.1","radam",4,8,.1,None),
             ("radam_k4_s8_lr0.3","radam",4,8,.3,None)]
        for M in (50,200,1000):cfg.append((f"pool_{M}","pool",1,0,.1,M))
        for seed in seeds:
            for name,method,K,steps,lr,M in cfg:
                V,info=train_kol(m,method,seed,add_rank,K,steps,lr,M or 100)
                mx,me=ind.deployment(V,Z)
                row={"case":"kolmogorov","amplitude":amp,"method":name,"seed":seed,
                     "initial_rank":m.d,"final_rank":V.shape[1],"max_error":mx,"mean_error":me,**info}
                rows.append(row);print("KOL-IND",amp,name,seed,mx,flush=True)
    data={"case":"Kolmogorov Galerkin residual indicator","search_indicator":"initial projection error squared plus time-integrated full residual squared",
          "truth_during_search":False,"d":8,"initial_rank":8,"added_rank":6,"final_rank":14,
          "gradient_checks":checks,"rows":rows,
          "summary":{str(a):summarize([r for r in rows if r["amplitude"]==a]) for a in amplitudes}}
    (OUT/"kolmogorov_indicator.json").write_text(json.dumps(data,indent=2));return data


# ---------------------------------------------------------------------------
# Microwave: stability-scaled RB residual


class MicrowaveIndicator:
    def __init__(self,m):self.m=m

    def rb(self,V,p):
        A,_=self.m._data(p);b=self.m.b.astype(complex)
        if V.size:
            c=np.linalg.solve(V.conj().T@A@V,V.conj().T@b);u=V@c
        else:u=np.zeros_like(b)
        r=b-A@u;beta=np.linalg.svd(A,compute_uv=False)[-1]
        return u,r,float(beta)

    def value(self,V,p):
        _,r,beta=self.rb(V,p);return float(np.vdot(r,r).real/max(beta*beta,1e-30))

    def fg_q(self,V,q,eps=2e-5):
        f=self.value(V,self.m.from_q(q));g=np.zeros(3)
        for j in range(3):
            e=np.zeros(3);e[j]=eps
            g[j]=(self.value(V,self.m.from_q(q+e))-self.value(V,self.m.from_q(q-e)))/(2*eps)
        return f,g

    def errors(self,V,P):
        e=[]
        for p in P:
            ur,_,_=self.rb(V,p);u=self.m.state(p);e.append(np.linalg.norm(u-ur)/np.linalg.norm(u))
        a=np.asarray(e);return float(a.max()),float(a.mean())


def adam_q(ind,V,p0,steps,lr):
    q=ind.m.to_q(p0);m=np.zeros(3);v=np.zeros(3);best=(-np.inf,q.copy());calls=0
    for it in range(1,steps+1):
        f,g=ind.fg_q(V,q);calls+=7
        if f>best[0]:best=(f,q.copy())
        m=.9*m+.1*g;v=.999*v+.001*g*g;q+=lr*(m/(1-.9**it))/(np.sqrt(v/(1-.999**it))+1e-8)
    f,_=ind.fg_q(V,q);calls+=7
    if f>best[0]:best=(f,q.copy())
    return ind.m.from_q(best[1]),best[0],{"calls":calls}


def train_mw(m,method,seed,rank,K=4,steps=25,lr=.1,pool_size=100):
    ind=MicrowaveIndicator(m);V=np.empty((m.n,0),complex)
    V,_=orth_append(V,m.state(np.array([2.17,0,0.])))
    starts=np.empty((rank-1,K,3))
    for k in range(rank-1):
        for j in range(K):
            rng=np.random.default_rng(np.random.SeedSequence([seed,k,j,1409]));starts[k,j]=rng.uniform(m.lo,m.hi)
    rng=np.random.default_rng(seed+883);pool=rng.uniform(m.lo,m.hi,size=(pool_size,3));calls=0;seconds=0.;chosen=[]
    for k in range(rank-1):
        if method=="pool":
            t=time.perf_counter();vals=np.array([ind.value(V,p) for p in pool]);seconds+=time.perf_counter()-t
            p=pool[int(np.argmax(vals))];calls+=pool_size
        else:
            cand=[]
            for j in range(K):
                t=time.perf_counter();p1,f1,info=adam_q(ind,V,starts[k,j],steps,lr)
                seconds+=time.perf_counter()-t;calls+=info["calls"];cand.append((f1,p1))
            p=max(cand,key=lambda q:q[0])[1]
        V,ok=orth_append(V,m.state(p));chosen.append(p.tolist())
        if not ok:break
    return V,{"indicator_calls":calls,"search_seconds":seconds,"chosen":chosen}


def run_mw(quick=False):
    m=MicrowaveModel();ind=MicrowaveIndicator(m);rank=8;seeds=[11] if quick else [11,22,33,44,55]
    rng=np.random.default_rng(887);P=rng.uniform(m.lo,m.hi,size=(3000 if quick else 20000,3));w=np.linspace(m.lo[0],m.hi[0],401 if quick else 1501)
    lines=[np.column_stack((w,np.full_like(w,a),np.full_like(w,b))) for a,b in [(0,0),(-.004,-.004),(-.004,.004),(.004,-.004),(.004,.004)]]
    tr=np.random.default_rng(991).uniform(m.lo[1:],m.hi[1:],size=(80 if quick else 300,2));P=np.vstack([P,*lines,m.resonance_aware_points(tr)])
    # Gradient audit in smooth coordinates.
    V=np.empty((m.n,0),complex);V,_=orth_append(V,m.state(np.array([2.17,0,0.])));q=m.to_q(np.array([2.171,.0012,-.0017]))
    _,g=ind.fg_q(V,q,2e-5);_,g2=ind.fg_q(V,q,1e-5);check=float(np.linalg.norm(g-g2)/max(np.linalg.norm(g2),1e-14))
    cfg=[]
    if quick:
        for K in (4,16):
            for steps in (8,25):
                for lr in (.03,.1,.3):cfg.append((f"adam_k{K}_s{steps}_lr{lr}","adam",K,steps,lr,None))
    else:
        cfg += [("adam_k4_s8_lr0.03","adam",4,8,.03,None),
                ("adam_k4_s8_lr0.1","adam",4,8,.1,None),
                ("adam_k4_s8_lr0.3","adam",4,8,.3,None),
                ("adam_k16_s8_lr0.1","adam",16,8,.1,None),
                ("adam_k16_s25_lr0.1","adam",16,25,.1,None)]
    for M in (200,800,1600):cfg.append((f"pool_{M}","pool",1,0,.1,M))
    rows=[]
    for seed in seeds:
        for name,method,K,steps,lr,M in cfg:
            V,info=train_mw(m,method,seed,rank,K,steps,lr,M or 100);mx,me=ind.errors(V,P)
            row={"case":"microwave","method":name,"seed":seed,"rank":V.shape[1],"max_error":mx,"mean_error":me,**info}
            rows.append(row);print("MW-IND",name,seed,mx,flush=True)
    data={"case":"microwave stability-scaled residual indicator","search_indicator":"RB residual norm divided by exact smallest singular value of operator",
          "truth_during_search":False,"rank":rank,"Q":m.Q,"test_size":len(P),"gradient_check":check,
          "rows":rows,"summary":summarize(rows)}
    (OUT/"microwave_indicator.json").write_text(json.dumps(data,indent=2));return data


def main():
    ap=argparse.ArgumentParser();ap.add_argument("case",choices=["liftup","kolmogorov","microwave","all"]);ap.add_argument("--quick",action="store_true");a=ap.parse_args();OUT.mkdir(exist_ok=True)
    if a.case in ("liftup","all"):run_liftup(a.quick)
    if a.case in ("kolmogorov","all"):run_kol(a.quick)
    if a.case in ("microwave","all"):run_mw(a.quick)


if __name__=="__main__":main()
