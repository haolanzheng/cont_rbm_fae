"""Natural-structure screens for continuous parameter greedy search.

The search objective is the exact squared projection error.  This deliberately
separates optimization geometry from error-estimator quality.  It is therefore
an oracle screening study, not an offline-cost claim for a production RBM.

Cases
-----
1. Linear streamwise-constant lift-up dynamics in plane Poiseuille flow.
2. Finite-amplitude perturbations of two-dimensional Kolmogorov shear flow.
3. A six-resonator high-Q coupled-mode microwave model with two tuning knobs.

Continuous search uses either Riemannian Adam/RL-BFGS on a sphere or Adam/BFGS
in smooth tanh coordinates for a box.  No component-wise clipping is used.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from contgreedy.sampling.host import BestObserved
from scipy import linalg, optimize
from scipy.sparse.linalg import expm_multiply

from contgreedy.sampling import rlbfgs_ascent, sphere_exp, tangent


ROOT = Path(__file__).resolve().parent / "natural_continuous_results"


def orth_append(V, x, tol=1e-11):
    x = np.asarray(x)
    dtype = np.result_type(x.dtype, V.dtype if V.size else x.dtype)
    q = x.astype(dtype, copy=True)
    for _ in range(2):
        if V.size:
            q -= V @ (V.conj().T @ q)
    nq = float(np.linalg.norm(q))
    if nq <= tol:
        return V, False
    return np.column_stack((V, q / nq)), True


def proj_error_sq(V, U):
    """Column-wise projection errors squared."""
    E = U - V @ (V.conj().T @ U) if V.size else U
    return np.sum(np.abs(E) ** 2, axis=0).real


def fixed_sphere_starts(seed, rank, nstarts, d):
    out = np.empty((rank, nstarts, d))
    for k in range(rank):
        for j in range(nstarts):
            rng = np.random.default_rng(np.random.SeedSequence([seed, k, j, 611]))
            z = rng.normal(size=d)
            out[k, j] = z / np.linalg.norm(z)
    return out


def radam_ascent(fg, z0, steps, lr, radius=1.0):
    """Riemannian Adam using exponential-map steps and no radial projection."""
    z = np.asarray(z0, float).copy()
    z *= radius / np.linalg.norm(z)
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    beta1, beta2, eps = .9, .999, 1e-8
    best_z = z.copy(); best_f = -np.inf
    trace = []
    calls = 0
    for it in range(1, steps + 1):
        old_z = z.copy()
        f, g = fg(z); calls += 1
        g = tangent(z, np.asarray(g, float), radius)
        # Moments are kept in the current ambient coordinates and re-tangented.
        # This is a first-order vector transport approximation; the exponential
        # map keeps every iterate exactly on the sphere.
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mh = m / (1 - beta1 ** it)
        vh = v / (1 - beta2 ** it)
        direction = tangent(z, mh / (np.sqrt(vh) + eps), radius)
        # lr is an angular/geodesic scale; cap avoids crossing the cut locus.
        dn = np.linalg.norm(direction)
        if dn > 0:
            step = direction * min(lr, .45 * np.pi * radius / dn)
            znew = sphere_exp(z, step, radius)
            m = tangent(znew, m, radius)
            z = znew
        if f > best_f:
            best_f, best_z = float(f), old_z
        trace.append(float(f))
    f, _ = fg(z); calls += 1
    if f > best_f:
        best_f, best_z = float(f), z.copy()
    return best_z, best_f, {"calls": calls, "trace": trace}


def summarize(rows):
    result = {}
    for method in sorted({r["method"] for r in rows}):
        rr = [r for r in rows if r["method"] == method]
        x = np.array([r["max_error"] for r in rr])
        result[method] = {
            "runs": len(rr),
            "max_error_median": float(np.median(x)),
            "max_error_mean": float(x.mean()),
            "max_error_q90": float(np.quantile(x, .9)),
            "max_error_max": float(x.max()),
            "mean_error_median": float(np.median([r["mean_error"] for r in rr])),
            "search_seconds_mean": float(np.mean([r["search_seconds"] for r in rr])),
            "objective_calls_mean": float(np.mean([r["objective_calls"] for r in rr])),
        }
    return result


# ---------------------------------------------------------------------------
# 1. Linear lift-up model


class LiftUpModel:
    """Finite-difference streamwise-constant lift-up subsystem.

    For each spanwise wavenumber beta,
        v_t = Re^{-1}(Dyy-beta^2)v,
        u_t = -U'(y)v + Re^{-1}(Dyy-beta^2)u,
    with U(y)=1-y^2 and homogeneous Dirichlet walls.  Parameters are unit-
    energy coefficients of smooth roll disturbances in v.
    """
    def __init__(self, ny=48, re=1000.0, T=20.0, betas=(1., 1.5, 2., 2.5),
                 wall_modes=4):
        self.ny = int(ny); self.re = float(re); self.T = float(T)
        self.betas = tuple(map(float, betas)); self.wall_modes = int(wall_modes)
        self.d = len(self.betas) * self.wall_modes
        h = 2.0 / (ny + 1)
        y = -1 + h * np.arange(1, ny + 1)
        D2 = (np.diag(-2 * np.ones(ny)) + np.diag(np.ones(ny - 1), 1)
              + np.diag(np.ones(ny - 1), -1)) / h**2
        D1 = (np.diag(np.ones(ny - 1), 1)-np.diag(np.ones(ny - 1), -1))/(2*h)
        Up = -2 * y
        blocks = []
        Ablocks = []; Bblocks = []
        for beta in self.betas:
            L = (D2 - beta**2 * np.eye(ny)) / re
            A = np.block([[L, -np.diag(Up)], [np.zeros_like(L), L]])
            # For a streamwise-constant Fourier mode, incompressibility gives
            # w=i*D_y v/beta.  Hence kinetic energy is
            # ||u||^2+||v||^2+||D_y v/beta||^2.
            Wv = np.eye(ny) + (D1.T @ D1) / beta**2
            Rv = linalg.cholesky(Wv, lower=False)
            energy_map = linalg.block_diag(np.eye(ny), Rv)
            B = np.zeros((2 * ny, self.wall_modes))
            for j in range(self.wall_modes):
                phi = np.sin((j + 1) * np.pi * (y + 1) / 2)
                phi /= np.sqrt(phi @ Wv @ phi)
                B[ny:, j] = phi
            inv_energy_map=linalg.solve_triangular(energy_map,np.eye(2*ny),lower=False)
            Ae=energy_map@A@inv_energy_map;Be=energy_map@B
            blocks.append(linalg.expm(T*Ae)@Be)
            Ablocks.append(Ae);Bblocks.append(Be)
        self.S = linalg.block_diag(*blocks)
        self.A=linalg.block_diag(*Ablocks);self.B=linalg.block_diag(*Bblocks)
        self.times=np.linspace(0,T,41)
        self.Xmaps=np.asarray(expm_multiply(self.A,self.B,start=0.,stop=T,
                                            num=len(self.times),endpoint=True))
        # Normalize input modes at t=0; columns of Ball already have unit norm.
        self.n = self.S.shape[0]
        self.singular_values = np.linalg.svd(self.S, compute_uv=False)

    def state(self, z):
        return self.S @ np.asarray(z)

    def states(self, Z):
        return self.S @ np.asarray(Z).T

    def fg(self, V, z):
        u = self.S @ z
        e = u - V @ (V.T @ u) if V.size else u
        f = float(e @ e)
        g = 2 * (self.S.T @ e)
        return f, g

    def exact_max_error(self, V):
        E = self.S - V @ (V.T @ self.S) if V.size else self.S
        return float(np.linalg.svd(E, compute_uv=False)[0])


def sphere_train(model, method, seed, rank, nstarts=4, lr=.1, steps=25,
                 pool_size=100):
    starts = fixed_sphere_starts(seed, rank, nstarts, model.d)
    rng = np.random.default_rng(seed + 1701)
    pool = rng.normal(size=(pool_size, model.d)); pool /= np.linalg.norm(pool, axis=1)[:, None]
    V = np.empty((model.n, 0)); selected=[]; calls=0; seconds=0.; traces=[]
    for k in range(rank):
        if k == 0:
            z = np.ones(model.d); z /= np.linalg.norm(z)
            trace = {"initial": True}
        elif method == "pool":
            t0=time.perf_counter(); vals=proj_error_sq(V, model.states(pool)); seconds+=time.perf_counter()-t0
            i=int(np.argmax(vals)); z=pool[i].copy(); calls += pool_size
            trace={"pool_best":float(vals[i])}
        else:
            cand=[]
            for j in range(nstarts):
                fg=lambda x: model.fg(V,x)
                t0=time.perf_counter()
                if method == "radam":
                    zj,fj,info=radam_ascent(fg,starts[k,j],steps,lr)
                elif method == "rlbfgs":
                    zj,fj,info=rlbfgs_ascent(fg,lambda x:fg(x)[0],starts[k,j],1.0,steps)
                    info={"calls":info["gradient_evaluations"]+info["value_evaluations"],
                          "trace":info["trace"]}
                else: raise ValueError(method)
                seconds+=time.perf_counter()-t0; calls+=info["calls"]
                cand.append((fj,zj,info))
            cand.sort(key=lambda x:x[0],reverse=True); _,z,win=cand[0]
            trace={"candidate_objectives":[float(x[0]) for x in cand],"winner":win}
        V,ok=orth_append(V,model.state(z)); selected.append(z.tolist()); traces.append(trace)
        if not ok: break
    return V,{"calls":calls,"seconds":seconds,"selected":selected,"traces":traces}


def run_liftup(out, quick=False):
    model=LiftUpModel(ny=36 if quick else 48)
    rank=8; seeds=[11] if quick else [11,22,33,44,55]
    # Hyperparameter sweep and strong line-search baseline.
    configs=[]
    for s in (8,25,100):
        for lr in (.01,.03,.1,.3): configs.append((f"radam_k4_s{s}_lr{lr}","radam",4,lr,s,None))
    configs += [("rlbfgs_k1_s8","rlbfgs",1,None,8,None),
                ("rlbfgs_k4_s8","rlbfgs",4,None,8,None),
                ("rlbfgs_k4_s50","rlbfgs",4,None,50,None)]
    for M in (50,200,1000,10000): configs.append((f"pool_{M}","pool",1,None,0,M))
    rng=np.random.default_rng(882); Ztest=rng.normal(size=(5000 if quick else 40000,model.d));Ztest/=np.linalg.norm(Ztest,axis=1)[:,None]
    Utest=model.states(Ztest)
    rows=[]
    for seed in seeds:
        for name,method,K,lr,steps,M in configs:
            V,info=sphere_train(model,method,seed,rank,K,lr or .1,steps,pool_size=M or 100)
            err=np.sqrt(proj_error_sq(V,Utest)); exact=model.exact_max_error(V)
            row={"case":"liftup","method":name,"seed":seed,"rank":V.shape[1],
                 "max_error":exact,"sample_max_error":float(err.max()),"mean_error":float(err.mean()),
                 "search_seconds":info["seconds"],"objective_calls":info["calls"]}
            rows.append(row);print("LIFTUP",name,seed,row["max_error"],flush=True)
    data={"case":"linear streamwise-constant Poiseuille lift-up","d":model.d,
          "state_dimension":model.n,"rank":rank,"Re":model.re,"T":model.T,
          "singular_values":model.singular_values.tolist(),"test_size":len(Ztest),
          "unconstrained_optimal_rank_error":float(model.singular_values[rank]),
          "search_objective":"exact final-time projection error squared",
          "rows":rows,"summary":summarize(rows)}
    (out/"liftup.json").write_text(json.dumps(data,indent=2)); return data


# ---------------------------------------------------------------------------
# 2. Nonlinear Kolmogorov shear flow


class KolmogorovModel:
    """2-D periodic perturbation vorticity around U(y)=sin(y)."""
    def __init__(self, n=12, re=40., T=2., dt=.025, amplitude=.15):
        self.ngrid=int(n); self.re=float(re); self.nu=1/re; self.T=float(T); self.dt=float(dt)
        self.steps=int(round(T/dt)); self.dt=T/self.steps; self.amplitude=float(amplitude)
        x=2*np.pi*np.arange(n)/n; y=x.copy(); X,Y=np.meshgrid(x,y,indexing="ij")
        self.X=X;self.Y=Y;self.U=np.sin(Y)
        kk=np.fft.fftfreq(n,1/n); self.kx=kk[:,None];self.ky=kk[None,:]
        self.k2=self.kx**2+self.ky**2; self.inv_k2=np.zeros_like(self.k2);self.inv_k2[self.k2>0]=1/self.k2[self.k2>0]
        # 3/2-style sharp 2/3 filter on all nonlinear products.
        self.mask=(np.abs(self.kx)<=n//3)&(np.abs(self.ky)<=n//3)
        modes=[(1,0,"c"),(1,0,"s"),(1,1,"c"),(1,1,"s"),
               (1,2,"c"),(1,2,"s"),(2,1,"c"),(2,1,"s")]
        phis=[]
        for kx,ky,typ in modes:
            a=kx*X+ky*Y; q=np.cos(a) if typ=="c" else np.sin(a);q-=q.mean();q/=np.linalg.norm(q)
            phis.append(q)
        self.Phi=np.array(phis);self.d=len(phis);self.n=n*n

    def _derivs(self,w):
        wh=np.fft.fft2(w,axes=(-2,-1));wh*=self.mask
        psi_h=wh*self.inv_k2
        ux=np.fft.ifft2(1j*self.ky*psi_h,axes=(-2,-1)).real
        vy=np.fft.ifft2(-1j*self.kx*psi_h,axes=(-2,-1)).real
        wx=np.fft.ifft2(1j*self.kx*wh,axes=(-2,-1)).real
        wy=np.fft.ifft2(1j*self.ky*wh,axes=(-2,-1)).real
        lap=np.fft.ifft2(-self.k2*wh,axes=(-2,-1)).real
        return ux,vy,wx,wy,lap

    def rhs(self,w):
        ux,vy,wx,wy,lap=self._derivs(w)
        nonlinear=ux*wx+vy*wy
        nh=np.fft.fft2(nonlinear,axes=(-2,-1));nh*=self.mask
        nonlinear=np.fft.ifft2(nh,axes=(-2,-1)).real
        return -self.U*wx-vy*self.U-nonlinear+self.nu*lap

    def rhs_with_tangent(self,w,dw):
        """Primal RHS and exact tangent-linear RHS for a batch of directions."""
        ux,vy,wx,wy,lap=self._derivs(w)
        dux,dvy,dwx,dwy,dlap=self._derivs(dw)
        nonlinear=ux*wx+vy*wy
        dnonlinear=dux*wx+ux*dwx+dvy*wy+vy*dwy
        nh=np.fft.fft2(nonlinear,axes=(-2,-1));nh*=self.mask
        dnh=np.fft.fft2(dnonlinear,axes=(-2,-1));dnh*=self.mask
        nonlinear=np.fft.ifft2(nh,axes=(-2,-1)).real
        dnonlinear=np.fft.ifft2(dnh,axes=(-2,-1)).real
        fw=-self.U*wx-vy*self.U-nonlinear+self.nu*lap
        fdw=-self.U*dwx-dvy*self.U-dnonlinear+self.nu*dlap
        return fw,fdw

    def _step(self,w,h):
        k1=self.rhs(w);k2=self.rhs(w+.5*h*k1);k3=self.rhs(w+.5*h*k2);k4=self.rhs(w+h*k3)
        q=w+h*(k1+2*k2+2*k3+k4)/6
        qh=np.fft.fft2(q,axes=(-2,-1));qh*=self.mask
        return np.fft.ifft2(qh,axes=(-2,-1)).real

    def solve_batch(self,Z):
        Z=np.atleast_2d(Z);w=self.amplitude*np.einsum("bd,dxy->bxy",Z,self.Phi)
        for _ in range(self.steps):w=self._step(w,self.dt)
        return w.reshape(len(Z),-1).T

    def solve_sensitivity(self,z):
        w=self.amplitude*np.einsum("d,dxy->xy",z,self.Phi)
        dw=self.amplitude*self.Phi.copy();h=self.dt
        for _ in range(self.steps):
            k1,dk1=self.rhs_with_tangent(w,dw)
            k2,dk2=self.rhs_with_tangent(w+.5*h*k1,dw+.5*h*dk1)
            k3,dk3=self.rhs_with_tangent(w+.5*h*k2,dw+.5*h*dk2)
            k4,dk4=self.rhs_with_tangent(w+h*k3,dw+h*dk3)
            w=w+h*(k1+2*k2+2*k3+k4)/6
            dw=dw+h*(dk1+2*dk2+2*dk3+dk4)/6
            wh=np.fft.fft2(w);wh*=self.mask;w=np.fft.ifft2(wh).real
            dwh=np.fft.fft2(dw,axes=(-2,-1));dwh*=self.mask;dw=np.fft.ifft2(dwh,axes=(-2,-1)).real
        return w.reshape(-1),dw.reshape(self.d,-1).T

    def fg(self,V,z):
        u,du=self.solve_sensitivity(z);e=u-V@(V.T@u) if V.size else u
        return float(e@e),2*(e@du)

    def fg_fd(self,V,z,eps=2e-5):
        # Central differences are used only for the optimizer gradient audit.
        # All d perturbations are advanced as one vectorized batch.
        Z=np.repeat(z[None,:],2*self.d+1,axis=0)
        for j in range(self.d):Z[1+2*j,j]+=eps;Z[2+2*j,j]-=eps
        U=self.solve_batch(Z);f=proj_error_sq(V,U)
        g=np.array([(f[1+2*j]-f[2+2*j])/(2*eps) for j in range(self.d)])
        return float(f[0]),g

    def state(self,z):return self.solve_batch(np.asarray(z)[None,:])[:,0]


def nonlinear_sphere_train(model,method,seed,rank,nstarts=4,lr=.1,steps=8,pool_size=50):
    starts=fixed_sphere_starts(seed,rank,nstarts,model.d);rng=np.random.default_rng(seed+333)
    pool=rng.normal(size=(pool_size,model.d));pool/=np.linalg.norm(pool,axis=1)[:,None]
    V=np.empty((model.n,0));calls=0;seconds=0.; selected=[]
    for k in range(rank):
        if k==0:z=np.ones(model.d);z/=np.linalg.norm(z)
        elif method=="pool":
            t=time.perf_counter();vals=proj_error_sq(V,model.solve_batch(pool));seconds+=time.perf_counter()-t
            z=pool[int(np.argmax(vals))].copy();calls+=pool_size
        else:
            cand=[]
            for j in range(nstarts):
                fg=lambda x:model.fg(V,x)
                t=time.perf_counter()
                if method=="radam":zj,fj,info=radam_ascent(fg,starts[k,j],steps,lr)
                else:
                    zj,fj,raw=rlbfgs_ascent(fg,lambda x:fg(x)[0],starts[k,j],1.,steps)
                    info={"calls":raw["gradient_evaluations"]+raw["value_evaluations"]}
                seconds+=time.perf_counter()-t;calls+=info["calls"];cand.append((fj,zj))
            cand.sort(key=lambda x:x[0],reverse=True);z=cand[0][1]
        V,ok=orth_append(V,model.state(z));selected.append(z.tolist())
        if not ok:break
    return V,{"calls":calls,"seconds":seconds,"selected":selected}


def run_kolmogorov(out,quick=False):
    rows=[];checks={}
    amplitudes=(.5,2.5) if not quick else (2.5,)
    seeds=[11] if quick else [11,22,33]
    for amp in amplitudes:
        model=KolmogorovModel(n=12,re=40,T=2,dt=.025,amplitude=amp);rank=5
        # Independent finite-difference consistency check with a smaller step.
        rng=np.random.default_rng(7);z=rng.normal(size=model.d);z/=np.linalg.norm(z);V=np.empty((model.n,0))
        f,g=model.fg(V,z);_,gfd=model.fg_fd(V,z,1e-5)
        zp=-z;up=model.state(z);um=model.state(zp)
        checks[str(amp)]={"tangent_vs_fd_gradient_relative_error":float(np.linalg.norm(g-gfd)/max(np.linalg.norm(gfd),1e-14)),
                          "objective":f,"nonlinear_even_fraction":float(np.linalg.norm(up+um)/(np.linalg.norm(up)+np.linalg.norm(um)))}
        if quick:
            configs=[]
            for s in (8,25):
                for lr in (.01,.03,.1,.3):configs.append((f"radam_k4_s{s}_lr{lr}","radam",4,lr,s,None))
            configs += [("rlbfgs_k4_s8","rlbfgs",4,None,8,None),("rlbfgs_k4_s25","rlbfgs",4,None,25,None)]
            for M in (20,50,200,1000):configs.append((f"pool_{M}","pool",1,None,0,M))
        else:
            # Confirmation set: preserve a deliberately bad learning rate,
            # two good Adam budgets, a line-search optimizer, and nested costs.
            configs=[("radam_k4_s8_lr0.01","radam",4,.01,8,None),
                     ("radam_k4_s8_lr0.1","radam",4,.1,8,None),
                     ("radam_k4_s8_lr0.3","radam",4,.3,8,None),
                     ("radam_k4_s25_lr0.1","radam",4,.1,25,None),
                     ("rlbfgs_k4_s8","rlbfgs",4,None,8,None),
                     ("pool_50","pool",1,None,0,50),
                     ("pool_200","pool",1,None,0,200),
                     ("pool_1000","pool",1,None,0,1000)]
        rt=np.random.default_rng(981);Ztest=rt.normal(size=(1000 if quick else 6000,model.d));Ztest/=np.linalg.norm(Ztest,axis=1)[:,None]
        Utest=model.solve_batch(Ztest)
        for seed in seeds:
            for name,method,K,lr,steps,M in configs:
                V,info=nonlinear_sphere_train(model,method,seed,rank,K,lr or .1,steps,M or 50)
                err=np.sqrt(proj_error_sq(V,Utest))
                row={"case":"kolmogorov","amplitude":amp,"method":name,"seed":seed,"rank":V.shape[1],
                     "max_error":float(err.max()),"mean_error":float(err.mean()),
                     "search_seconds":info["seconds"],"objective_calls":info["calls"]}
                rows.append(row);print("KOLMOGOROV",amp,name,seed,row["max_error"],flush=True)
    data={"case":"finite-amplitude 2D Kolmogorov shear perturbations","d":8,"rank":5,
          "Re":40,"T":2,"search_objective":"exact final-vorticity projection error squared",
          "gradient_checks":checks,"rows":rows,
          "summary":{str(a):summarize([r for r in rows if r["amplitude"]==a]) for a in amplitudes}}
    (out/"kolmogorov.json").write_text(json.dumps(data,indent=2));return data


# ---------------------------------------------------------------------------
# 3. Coupled-resonator microwave model


class MicrowaveModel:
    """Six high-Q resonators coupled to weak parasitic cavity modes."""
    def __init__(self,Q=3500.,npar=18):
        rng=np.random.default_rng(410)
        main=np.array([2.151,2.159,2.166,2.174,2.183,2.192])
        paras=np.linspace(2.125,2.215,npar)+rng.normal(scale=7e-4,size=npar)
        self.freq0=np.r_[main,paras];self.n=len(self.freq0);self.Q=float(Q)
        C=np.zeros((self.n,self.n))
        # Inline main-resonator coupling plus weak coupling to cavity modes.
        for i in range(5):C[i,i+1]=C[i+1,i]=2.1e-3
        W=rng.normal(scale=2.2e-4,size=(6,npar));C[:6,6:]=W;C[6:,:6]=W.T
        self.Coupling=C
        self.b=np.r_[np.array([1.,.35,.15,.08,.04,.02]),rng.uniform(.002,.02,npar)]
        self.lo=np.array([2.14,-.004,-.004]);self.hi=np.array([2.20,.004,.004]);self.d=3

    def _data(self,p):
        w,t1,t2=map(float,p);freq=self.freq0.copy()
        # Two tuning groups perturb overlapping resonator subsets.
        a=np.array([1.,.7,.2,0,0,0]);b=np.array([0,0,.2,.7,1.,.45])
        freq[:6]+=a*t1+b*t2
        damp=np.diag(freq**2/self.Q)
        A=np.diag(freq**2)-self.Coupling-w*w*np.eye(self.n)+1j*w*damp
        deriv=[]
        deriv.append(-2*w*np.eye(self.n)+1j*damp)
        for q in (a,b):
            qq=np.r_[q,np.zeros(self.n-6)]
            deriv.append(np.diag(2*freq*qq)+1j*w*np.diag(2*freq*qq/self.Q))
        return A,deriv

    def state_derivs(self,p):
        A,Ap=self._data(p);u=np.linalg.solve(A,self.b.astype(complex))
        du=np.column_stack([-np.linalg.solve(A,B@u) for B in Ap])
        return u,du

    def state(self,p):return self.state_derivs(p)[0]

    def fg(self,V,p):
        u,du=self.state_derivs(p);e=u-V@(V.conj().T@u) if V.size else u
        return float(np.vdot(e,e).real),2*np.real(e.conj()@du)

    def from_q(self,q):
        s=np.tanh(q);return .5*(self.lo+self.hi)+.5*(self.hi-self.lo)*s

    def to_q(self,p):
        s=(2*np.asarray(p)-self.lo-self.hi)/(self.hi-self.lo);return np.arctanh(np.clip(s,-.999999,.999999))

    def fg_q(self,V,q):
        s=np.tanh(q);p=.5*(self.lo+self.hi)+.5*(self.hi-self.lo)*s
        f,g=self.fg(V,p);jac=.5*(self.hi-self.lo)*(1-s*s)
        return f,g*jac

    def states(self,P):return np.column_stack([self.state(p) for p in P])

    def resonance_aware_points(self,tunings,offsets=(-2.,0.,2.)):
        pts=[]
        a=np.array([1.,.7,.2,0,0,0]);b=np.array([0,0,.2,.7,1.,.45])
        for t1,t2 in np.asarray(tunings):
            freq=self.freq0.copy();freq[:6]+=a*t1+b*t2
            K=np.diag(freq**2)-self.Coupling
            roots=np.sqrt(np.maximum(np.linalg.eigvalsh(K),0))
            roots=roots[(roots>=self.lo[0])&(roots<=self.hi[0])]
            for root in roots:
                width=root/self.Q
                for c in offsets:
                    w=float(np.clip(root+c*width,self.lo[0],self.hi[0]))
                    pts.append((w,float(t1),float(t2)))
        return np.asarray(pts)


def adam_box(model,V,p0,steps,lr):
    q=model.to_q(p0);m=np.zeros(3);v=np.zeros(3);best=(-np.inf,q.copy());trace=[]
    for it in range(1,steps+1):
        f,g=model.fg_q(V,q);trace.append(float(f))
        if f>best[0]:best=(float(f),q.copy())
        m=.9*m+.1*g;v=.999*v+.001*g*g;mh=m/(1-.9**it);vh=v/(1-.999**it)
        q=q+lr*mh/(np.sqrt(vh)+1e-8)
    f,_=model.fg_q(V,q)
    if f>best[0]:best=(float(f),q.copy())
    return model.from_q(best[1]),best[0],{"calls":steps+1,"trace":trace}



def box_bfgs(model,V,p0,steps):
    q0 = model.to_q(p0)
    observed = BestObserved(lambda q: model.fg_q(V, q))

    def fg(q):
        f, g = observed.evaluate(q)
        den = max(f, 1e-300)
        return -np.log(den), -g / den

    res = optimize.minimize(
        fg, q0, jac=True, method="BFGS",
        options={"maxiter": steps, "gtol": 1e-9},
    )
    observed.ensure(q0)
    observed.ensure(res.x)
    q, value = observed.result()
    return model.from_q(q), value, {
        "calls": observed.calls, "nit": int(res.nit), "success": bool(res.success),
        "search_return_policy": "best_feasible_evaluation",
    }

def microwave_train(model,method,seed,rank,nstarts=4,lr=.1,steps=25,pool_size=100):
    starts=np.empty((rank,nstarts,3))
    for k in range(rank):
        for j in range(nstarts):
            rng=np.random.default_rng(np.random.SeedSequence([seed,k,j,903]));starts[k,j]=rng.uniform(model.lo,model.hi)
    rng=np.random.default_rng(seed+502);pool=rng.uniform(model.lo,model.hi,size=(pool_size,3))
    # Always include band endpoints at neutral tuning in the pool.
    if pool_size>=2:pool[:2]=np.array([[model.lo[0],0,0],[model.hi[0],0,0]])
    V=np.empty((model.n,0),complex);calls=0;seconds=0.;selected=[]
    for k in range(rank):
        if k==0:p=np.array([2.17,0.,0.])
        elif method=="pool":
            t=time.perf_counter();vals=proj_error_sq(V,model.states(pool));seconds+=time.perf_counter()-t
            p=pool[int(np.argmax(vals))].copy();calls+=pool_size
        else:
            cand=[]
            for j in range(nstarts):
                t=time.perf_counter()
                if method=="adam":pj,fj,info=adam_box(model,V,starts[k,j],steps,lr)
                else:pj,fj,info=box_bfgs(model,V,starts[k,j],steps)
                seconds+=time.perf_counter()-t;calls+=info["calls"];cand.append((fj,pj))
            cand.sort(key=lambda x:x[0],reverse=True);p=cand[0][1]
        V,ok=orth_append(V,model.state(p));selected.append(p.tolist())
        if not ok:break
    return V,{"calls":calls,"seconds":seconds,"selected":selected}


def run_microwave(out,quick=False):
    model=MicrowaveModel();rank=8;seeds=[11] if quick else [11,22,33,44,55]
    rng=np.random.default_rng(887);Ptest=rng.uniform(model.lo,model.hi,size=(3000 if quick else 30000,3))
    # Add dense frequency lines through neutral and four tuning corners.
    w=np.linspace(model.lo[0],model.hi[0],401 if quick else 2001)
    lines=[]
    for t1,t2 in [(0,0),(-.004,-.004),(-.004,.004),(.004,-.004),(.004,.004)]:
        lines.append(np.column_stack((w,np.full_like(w,t1),np.full_like(w,t2))))
    tune_rng=np.random.default_rng(991)
    tunings=tune_rng.uniform(model.lo[1:],model.hi[1:],size=(80 if quick else 400,2))
    Pres=model.resonance_aware_points(tunings)
    Ptest=np.vstack([Ptest,*lines,Pres]);Utest=model.states(Ptest)
    # Analytic derivative check at a generic point.
    p=np.array([2.171,.0012,-.0017]);V=np.empty((model.n,0),complex);f,g=model.fg(V,p);gfd=[]
    for j in range(3):
        h=1e-6 if j==0 else 1e-7;e=np.zeros(3);e[j]=h
        gfd.append((model.fg(V,p+e)[0]-model.fg(V,p-e)[0])/(2*h))
    check=float(np.linalg.norm(g-np.array(gfd))/np.linalg.norm(gfd))
    configs=[]
    for s in (8,25,100):
        for lr in (.01,.03,.1,.3):configs.append((f"adam_k4_s{s}_lr{lr}","adam",4,lr,s,None))
    configs += [("bfgs_k1_s8","bfgs",1,None,8,None),("bfgs_k4_s8","bfgs",4,None,8,None),
                ("bfgs_k4_s50","bfgs",4,None,50,None)]
    for M in (50,200,1000,10000):configs.append((f"pool_{M}","pool",1,None,0,M))
    rows=[]
    for seed in seeds:
        for name,method,K,lr,steps,M in configs:
            V,info=microwave_train(model,method,seed,rank,K,lr or .1,steps,M or 100)
            err=np.sqrt(proj_error_sq(V,Utest));wi=int(np.argmax(err))
            row={"case":"microwave","method":name,"seed":seed,"rank":V.shape[1],
                 "max_error":float(err[wi]),"mean_error":float(err.mean()),
                 "worst_parameter":Ptest[wi].tolist(),"search_seconds":info["seconds"],
                 "objective_calls":info["calls"]}
            rows.append(row);print("MICROWAVE",name,seed,row["max_error"],flush=True)
    data={"case":"six-resonator high-Q coupled-mode filter","d":3,"state_dimension":model.n,
          "rank":rank,"Q":model.Q,"test_size":len(Ptest),"gradient_relative_error":check,
          "resonance_aware_test_points":len(Pres),
          "smooth_box_map":"p=mid+halfwidth*tanh(q)",
          "search_objective":"exact complex-state projection error squared",
          "rows":rows,"summary":summarize(rows)}
    (out/"microwave.json").write_text(json.dumps(data,indent=2));return data


def main():
    ap=argparse.ArgumentParser();ap.add_argument("case",choices=["liftup","kolmogorov","microwave","all"])
    ap.add_argument("--quick",action="store_true");args=ap.parse_args();ROOT.mkdir(exist_ok=True)
    data={}
    if args.case in ("liftup","all"):data["liftup"]=run_liftup(ROOT,args.quick)
    if args.case in ("kolmogorov","all"):data["kolmogorov"]=run_kolmogorov(ROOT,args.quick)
    if args.case in ("microwave","all"):data["microwave"]=run_microwave(ROOT,args.quick)
    (ROOT/"all_summary.json").write_text(json.dumps({k:v["summary"] for k,v in data.items()},indent=2))


if __name__=="__main__":main()
