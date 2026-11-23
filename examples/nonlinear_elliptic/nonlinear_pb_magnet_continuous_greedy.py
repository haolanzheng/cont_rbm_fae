#!/usr/bin/env python3
"""Residual-only continuous greedy tests for two nonlinear elliptic PDEs.

Cases
-----
1. A compact finite-volume reproduction of the two-dimensional nonlinear
   Poisson--Boltzmann problem in Ji, Chen, and Xu (2018).  It uses their
   parameter box, fixed charge, and boundary conditions.
2. A reproducible graph finite-element analogue of nonlinear magnetostatics
   with a saturating ferromagnetic core and four independent coil currents.
   It follows the quasilinear operator structure in Hinze and Korolev (2020),
   but is not a reproduction of their proprietary motor mesh/B-H data.

No truth solution or projection error is used by any greedy selection rule.
Continuous selection uses one random start, 100 Adam updates, and a smooth
tanh chart for the box.  It therefore never clips parameters componentwise.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve
from scipy.stats import qmc


EPS = 1.0e-30


def atanh_safe(z):
    return np.arctanh(np.clip(np.asarray(z, float), -0.999999, 0.999999))


def smooth_box_adam(fg_z, z0, steps=100, lr=0.1):
    """Maximize f(z), z in (-1,1)^d, through z=tanh(y)."""
    y = atanh_safe(z0)
    m = np.zeros_like(y)
    v = np.zeros_like(y)
    best_f = -np.inf
    best_z = np.tanh(y).copy()
    trace = []
    for k in range(1, steps + 1):
        z = np.tanh(y)
        f, gz = fg_z(z)
        gy = np.asarray(gz) * (1.0 - z * z)
        m = 0.9 * m + 0.1 * gy
        v = 0.999 * v + 0.001 * gy * gy
        mh = m / (1.0 - 0.9**k)
        vh = v / (1.0 - 0.999**k)
        if f > best_f:
            best_f, best_z = float(f), z.copy()
        trace.append(float(f))
        y += lr * mh / (np.sqrt(vh) + 1.0e-8)
        # This guards floating-point overflow in exp used internally by tanh;
        # it does not project or clip the physical parameter z.
        y = np.maximum(np.minimum(y, 18.0), -18.0)
    z = np.tanh(y)
    f, _ = fg_z(z)
    if f > best_f:
        best_f, best_z = float(f), z.copy()
    return best_z, best_f, {"trace": trace, "calls": steps + 1}


def fixed_boundary_pool(rng, count, d):
    """A fixed strong baseline: 1/2 boundary faces, 1/4 corners, 1/4 interior."""
    nf = count // 2
    nc = count // 4
    ni = count - nf - nc
    faces = rng.uniform(-1.0, 1.0, (nf, d))
    face_axis = rng.integers(0, d, nf)
    faces[np.arange(nf), face_axis] = rng.choice([-1.0, 1.0], nf)
    corners = rng.choice([-1.0, 1.0], (nc, d))
    interior = rng.uniform(-1.0, 1.0, (ni, d))
    return np.vstack([faces, corners, interior])


def x_orth_append(V, u, X, tol=1.0e-11):
    q = np.asarray(u, float).copy()
    for _ in range(2):
        if V.size:
            q -= V @ (V.T @ (X @ q))
    nq = math.sqrt(max(float(q @ (X @ q)), 0.0))
    if nq <= tol:
        return V, False
    return np.column_stack([V, q / nq]), True


def fd_laplacian_dirichlet_x_neumann_y(n):
    """Cell-centred -Laplacian on [-1,1]^2.

    Dirichlet data are imposed at x=+-1 and zero flux at y=+-1.  Boundary
    face distance is h/2, giving the factor two in the x boundary diagonal.
    """
    h = 2.0 / n
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(n):
            k = i * n + j
            diag = 0.0
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < n:
                    ell = ii * n + jj
                    rows.append(k); cols.append(ell); vals.append(-1.0 / h**2)
                    diag += 1.0 / h**2
                elif di != 0:  # homogeneous Dirichlet for the lifted unknown
                    diag += 2.0 / h**2
                # missing y neighbour is homogeneous Neumann: no flux
            rows.append(k); cols.append(k); vals.append(diag)
    return sparse.csc_matrix((vals, (rows, cols)), shape=(n*n, n*n))


class PoissonBoltzmann:
    name = "poisson_boltzmann_2d"
    d = 2

    def __init__(self, grid=36):
        self.grid = int(grid)
        self.n = grid * grid
        h = 2.0 / grid
        x = -1.0 + (np.arange(grid) + 0.5) * h
        xx, yy = np.meshgrid(x, x, indexing="ij")
        self.xcoord = xx.ravel()
        self.g = np.exp(-50.0 * (xx*xx + yy*yy)).ravel()
        self.K = fd_laplacian_dirichlet_x_neumann_y(grid)
        self.Dmin, self.Dmax = 0.0064, 0.16
        self.Vmin, self.Vmax = 0.0, 5.0
        self.Dref = math.sqrt(self.Dmin * self.Dmax)
        self.X = sparse.csc_matrix(self.Dref * self.K + sparse.eye(self.n))
        self.Xlu = splu(self.X)

    def physical(self, z):
        z = np.asarray(z)
        logD = 0.5 * ((1-z[0])*math.log(self.Dmin) +
                      (1+z[0])*math.log(self.Dmax))
        D = math.exp(logD)
        voltage = self.Vmin + 0.5*(z[1]+1.0)*(self.Vmax-self.Vmin)
        dpdz = np.array([0.5*(math.log(self.Dmax)-math.log(self.Dmin))*D,
                         0.5*(self.Vmax-self.Vmin)])
        return np.array([D, voltage]), dpdz

    def _terms(self, w, p):
        D, voltage = p
        phi = w + voltage * self.xcoord
        s = np.sinh(np.clip(phi, -30.0, 30.0))
        c = np.cosh(np.clip(phi, -30.0, 30.0))
        F = D*(self.K @ w) + s + self.g
        J = sparse.csc_matrix(D*self.K + sparse.diags(c))
        return F, J, phi

    def fom(self, z, tol=2.0e-10, maxit=40):
        p, _ = self.physical(z)
        D, voltage = p
        A0 = sparse.csc_matrix(D*self.K + sparse.eye(self.n))
        w = spsolve(A0, -self.g - voltage*self.xcoord)
        for _ in range(maxit):
            F, J, _ = self._terms(w, p)
            nf = float(np.linalg.norm(F) / math.sqrt(self.n))
            if nf < tol:
                return w
            delta = spsolve(J, -F)
            merit = 0.5*float(F@F)
            step = 1.0
            for _ in range(18):
                trial = w + step*delta
                Ft, _, _ = self._terms(trial, p)
                if 0.5*float(Ft@Ft) <= merit*(1.0-1.0e-4*step):
                    w = trial; break
                step *= 0.5
            else:
                w += step*delta
        raise RuntimeError(f"PB Newton did not converge at {p}, residual {nf}")

    def reduced(self, z, V, tol=1.0e-11, maxit=30, sensitivity=False):
        p, dpdz = self.physical(z)
        D, voltage = p
        KV = self.K @ V
        Ar = D*(V.T@KV) + np.eye(V.shape[1])
        ccoef = np.linalg.solve(Ar, V.T@(-self.g-voltage*self.xcoord))
        for _ in range(maxit):
            w = V@ccoef
            F, J, phi = self._terms(w, p)
            R = V.T@F
            if np.linalg.norm(R) < tol:
                break
            Jr = V.T@(J@V)
            delta = np.linalg.solve(Jr, -R)
            merit = 0.5*float(R@R)
            step = 1.0
            for _ in range(14):
                ct = ccoef + step*delta
                Ft, _, _ = self._terms(V@ct, p)
                Rt = V.T@Ft
                if 0.5*float(Rt@Rt) <= merit*(1.0-1.0e-4*step):
                    ccoef = ct; break
                step *= 0.5
            else:
                ccoef += step*delta
        w = V@ccoef
        F, J, phi = self._terms(w, p)
        if not sensitivity:
            return w, F, phi
        Jr = V.T@(J@V)
        direct = [self.K@w, np.cosh(np.clip(phi,-30,30))*self.xcoord]
        dw_dp, dr_dp = [], []
        for a in direct:
            dc = np.linalg.solve(Jr, -(V.T@a))
            dw = V@dc
            dw_dp.append(dw)
            dr_dp.append(J@dw+a)
        return w, F, phi, np.column_stack(dw_dp), np.column_stack(dr_dp), dpdz

    def indicator(self, z, V, gradient=False):
        if gradient:
            w, r, _, dw_dp, dr_dp, dpdz = self.reduced(z,V,sensitivity=True)
        else:
            w, r, _ = self.reduced(z,V)
        rw = self.Xlu.solve(r)
        num = max(float(r@rw), EPS)
        den = max(float(w@(self.X@w)), 1.0e-20)
        p, _ = self.physical(z)
        D = p[0]
        value = math.log(num) - math.log(den) - 2.0*math.log(D)
        if not gradient:
            return value
        g_p = np.empty(2)
        Xw = self.X@w
        for j in range(2):
            g_p[j] = (2.0*float(dr_dp[:,j]@rw)/num
                      -2.0*float(dw_dp[:,j]@Xw)/den)
        g_p[0] -= 2.0/D
        return value, g_p*dpdz

    def truth_error(self, z, V, truth=None):
        truth = self.fom(z) if truth is None else truth
        wr, _, _ = self.reduced(z,V)
        e = truth-wr
        return math.sqrt(max(float(e@(self.X@e)),0.0)/
                         max(float(truth@(self.X@truth)),EPS))


def incidence_dirichlet_square(n):
    """Incidence/gradient matrix for n by n interior nodes, zero boundary."""
    h = 1.0/(n+1)
    rows=[]; cols=[]; vals=[]; mids=[]
    edge=0
    def node(i,j): return i*n+j
    # horizontal interior edges and two boundary edges per row
    for j in range(n):
        y=(j+1)*h
        rows.append(edge);cols.append(node(0,j));vals.append(1/h);mids.append((.5*h,y));edge+=1
        for i in range(n-1):
            rows += [edge,edge];cols += [node(i+1,j),node(i,j)];vals += [1/h,-1/h]
            mids.append(((i+1.5)*h,y));edge+=1
        rows.append(edge);cols.append(node(n-1,j));vals.append(-1/h);mids.append((1-.5*h,y));edge+=1
    # vertical interior edges and boundary edges
    for i in range(n):
        x=(i+1)*h
        rows.append(edge);cols.append(node(i,0));vals.append(1/h);mids.append((x,.5*h));edge+=1
        for j in range(n-1):
            rows += [edge,edge];cols += [node(i,j+1),node(i,j)];vals += [1/h,-1/h]
            mids.append((x,(j+1.5)*h));edge+=1
        rows.append(edge);cols.append(node(i,n-1));vals.append(-1/h);mids.append((x,1-.5*h));edge+=1
    B=sparse.csc_matrix((vals,(rows,cols)),shape=(edge,n*n))
    return B,np.asarray(mids),h


class SaturatingMagnet:
    name = "saturating_magnetostatic_circuit"
    d = 4

    def __init__(self, grid=28):
        self.grid=int(grid);self.n=grid*grid
        self.B,self.edge_mid,self.h=incidence_dirichlet_square(grid)
        x=self.edge_mid[:,0];y=self.edge_mid[:,1]
        # A square ferromagnetic yoke with a small right-hand gap.
        outer=(x>.12)&(x<.88)&(y>.12)&(y<.88)
        inner=(x>.30)&(x<.70)&(y>.30)&(y<.70)
        ring=outer&(~inner)
        gap=(x>.69)&(x<.90)&(y>.475)&(y<.525)
        self.iron=ring&(~gap)
        self.nu_low=0.055
        self.nu_high=1.0
        # Calibrated so that roughly ten percent of the iron edges cross the
        # saturation scale at maximum load on the reference grid.
        self.bsat=0.5
        xx,yy=np.meshgrid((np.arange(grid)+1)/(grid+1),
                          (np.arange(grid)+1)/(grid+1),indexing='ij')
        modes=[]
        # Four localized current patterns; each is a source/sink coil pair.
        specs=[((.38,.42),(.38,.58)),((.62,.42),(.62,.58)),
               ((.42,.38),(.58,.38)),((.42,.62),(.58,.62))]
        sig=.055
        for a,b in specs:
            q=np.exp(-((xx-a[0])**2+(yy-a[1])**2)/(2*sig**2))
            q-=np.exp(-((xx-b[0])**2+(yy-b[1])**2)/(2*sig**2))
            q-=q.mean();q/=np.linalg.norm(q)
            modes.append(q.ravel())
        self.loads=np.column_stack(modes)*42.0
        self.Kair=sparse.csc_matrix(self.B.T@self.B)
        # Reference norm uses the unsaturated low-field material.
        nue=np.where(self.iron,self.nu_low,1.0)
        self.X=sparse.csc_matrix(self.B.T@sparse.diags(nue)@self.B)
        self.Xlu=splu(self.X)

    def physical(self,z):
        # Positive, independently variable coil multipliers.
        return 0.25+1.25*0.5*(np.asarray(z)+1.0), np.full(self.d,0.625)

    def _edge_law(self, grad):
        s2=grad*grad
        frac=s2/(self.bsat*self.bsat+s2)
        nu=np.ones_like(grad)
        nu[self.iron]=self.nu_low+(self.nu_high-self.nu_low)*frac[self.iron]
        # derivative of q(g)=nu(|g|)g
        tangent=nu.copy()
        tangent[self.iron] += ((self.nu_high-self.nu_low)*2*self.bsat**2*s2[self.iron]/
                               (self.bsat*self.bsat+s2[self.iron])**2)
        return nu,tangent

    def _terms(self,u,p):
        grad=self.B@u;nu,tan=self._edge_law(grad)
        F=self.B.T@(nu*grad)-self.loads@p
        J=sparse.csc_matrix(self.B.T@sparse.diags(tan)@self.B)
        return np.asarray(F),J

    def fom(self,z,tol=2e-10,maxit=45):
        p,_=self.physical(z)
        u=spsolve(self.X,self.loads@p)
        for _ in range(maxit):
            F,J=self._terms(u,p)
            nf=np.linalg.norm(F)/math.sqrt(self.n)
            if nf<tol:return u
            delta=spsolve(J,-F);merit=.5*float(F@F);step=1.
            for _ in range(18):
                trial=u+step*delta;Ft,_=self._terms(trial,p)
                if .5*float(Ft@Ft)<=merit*(1-1e-4*step):u=trial;break
                step*=.5
            else:u+=step*delta
        raise RuntimeError(f"magnet Newton failed at {p}, residual {nf}")

    def reduced(self,z,V,tol=1e-11,maxit=35,sensitivity=False):
        p,dpdz=self.physical(z)
        c=np.linalg.solve(V.T@(self.X@V),V.T@(self.loads@p))
        for _ in range(maxit):
            u=V@c;F,J=self._terms(u,p);R=V.T@F
            if np.linalg.norm(R)<tol:break
            Jr=V.T@(J@V);dc=np.linalg.solve(Jr,-R)
            merit=.5*float(R@R);step=1.
            for _ in range(14):
                ct=c+step*dc;Ft,_=self._terms(V@ct,p);Rt=V.T@Ft
                if .5*float(Rt@Rt)<=merit*(1-1e-4*step):c=ct;break
                step*=.5
            else:c+=step*dc
        u=V@c;F,J=self._terms(u,p)
        if not sensitivity:return u,F
        Jr=V.T@(J@V);du=[];dr=[]
        for j in range(self.d):
            dc=np.linalg.solve(Jr,V.T@self.loads[:,j])
            uj=V@dc;du.append(uj);dr.append(J@uj-self.loads[:,j])
        return u,F,np.column_stack(du),np.column_stack(dr),dpdz

    def indicator(self,z,V,gradient=False):
        if gradient:u,r,du,dr,dpdz=self.reduced(z,V,sensitivity=True)
        else:u,r=self.reduced(z,V)
        rw=self.Xlu.solve(r);num=max(float(r@rw),EPS)
        den=max(float(u@(self.X@u)),1e-20)
        value=math.log(num)-math.log(den)-2*math.log(self.nu_low)
        if not gradient:return value
        Xu=self.X@u
        gp=np.array([2*float(dr[:,j]@rw)/num-2*float(du[:,j]@Xu)/den
                     for j in range(self.d)])
        return value,gp*dpdz

    def truth_error(self,z,V,truth=None):
        truth=self.fom(z) if truth is None else truth
        ur,_=self.reduced(z,V);e=truth-ur
        return math.sqrt(max(float(e@(self.X@e)),0.)/
                         max(float(truth@(self.X@truth)),EPS))


def pool_points(kind,rng,count,d):
    if kind=="pool_uniform":return rng.uniform(-1,1,(count,d))
    if kind=="pool_boundary":return fixed_boundary_pool(rng,count,d)
    raise ValueError(kind)


@dataclass
class TrainResult:
    V: np.ndarray
    records: list
    seconds: float
    bases: list


def train(model,method,rank,seed,lr=.1,pool_size=100):
    rng=np.random.default_rng(seed)
    V=np.empty((model.n,0))
    # A common, deterministic initial snapshot.
    V,_=x_orth_append(V,model.fom(np.zeros(model.d)),model.X)
    fixed_pool=(pool_points(method,rng,pool_size,model.d)
                if method.startswith("pool_") else None)
    records=[];bases=[V.copy()];tic=time.perf_counter()
    for k in range(1,rank):
        if method=="continuous":
            z0=rng.uniform(-.9,.9,model.d)
            z,val,info=smooth_box_adam(lambda zz:model.indicator(zz,V,True),z0,100,lr)
            rec={"round":k+1,"z0":z0.tolist(),"z":z.tolist(),"log_eta2":val,
                 "calls":info["calls"],"trace":info["trace"]}
        else:
            vals=np.array([model.indicator(z,V,False) for z in fixed_pool])
            idx=int(vals.argmax());z=fixed_pool[idx].copy();val=float(vals[idx])
            rec={"round":k+1,"z":z.tolist(),"log_eta2":val,"calls":len(vals)}
        Vnew,ok=x_orth_append(V,model.fom(z),model.X)
        rec["basis_added"]=bool(ok);records.append(rec);V=Vnew
        bases.append(V.copy())
        if not ok:break
    return TrainResult(V,records,time.perf_counter()-tic,bases)


def test_design(model,seed=20260919):
    if model.d==2:
        a=np.linspace(-1,1,25)
        grid=np.stack(np.meshgrid(a,a,indexing='ij'),axis=-1).reshape(-1,2)
        sob=2*qmc.Sobol(2,scramble=True,seed=seed).random_base2(8)-1
        return np.vstack([grid,sob]),{"tensor_grid":len(grid),"sobol":len(sob)}
    sob=2*qmc.Sobol(model.d,scramble=True,seed=seed).random_base2(9)-1
    idx=np.arange(2**model.d)[:,None]
    corners=2*((idx>>np.arange(model.d))&1)-1
    rng=np.random.default_rng(seed)
    faces=rng.uniform(-1,1,(256,model.d));axis=rng.integers(0,model.d,len(faces))
    faces[np.arange(len(faces)),axis]=rng.choice([-1.,1.],len(faces))
    return np.vstack([sob,corners,faces]),{"sobol":len(sob),"corners":len(corners),"faces":len(faces)}


def precompute_truth(model,Z):
    truths=[];tic=time.perf_counter()
    for i,z in enumerate(Z):
        truths.append(model.fom(z))
        if (i+1)%500==0:print(model.name,"truth",i+1,"/",len(Z),flush=True)
    return np.asarray(truths),time.perf_counter()-tic


def audit(model,result,Z,truths):
    err=np.array([model.truth_error(z,result.V,u) for z,u in zip(Z,truths)])
    idx=int(err.argmax())
    # Curves use a fixed, method-independent subset to keep the nonlinear
    # reduced solves tractable; the final-rank statistics above use all tests.
    stride=8 if model.d==2 else 7
    curve_idx=np.arange(0,len(Z),stride)
    curve=[]
    keep={1,2,4,6,8,10,12,len(result.bases)}
    for rank,Vr in enumerate(result.bases,start=1):
        if rank not in keep:continue
        ee=np.array([model.truth_error(Z[i],Vr,truths[i]) for i in curve_idx])
        curve.append({"rank":rank,"max_relative_error":float(ee.max()),
                      "mean_relative_error":float(ee.mean()),"test_size":len(ee)})
    return {"rank":int(result.V.shape[1]),"max_relative_error":float(err[idx]),
            "mean_relative_error":float(err.mean()),"q99_relative_error":float(np.quantile(err,.99)),
            "worst_z":Z[idx].tolist(),"training_seconds":result.seconds,
            "objective_calls":int(sum(x["calls"] for x in result.records)),
            "convergence_curve":curve,"selected":result.records}


def physics_audit(model):
    if isinstance(model,PoissonBoltzmann):
        out={}
        for name,z in [('center',np.zeros(2)),('small_D_high_V',np.array([-1.,1.]))]:
            w=model.fom(z);p,_=model.physical(z);F,_,phi=model._terms(w,p)
            out[name]={"physical_parameters":p.tolist(),"phi_min":float(phi.min()),
                       "phi_max":float(phi.max()),"rms_residual":float(np.linalg.norm(F)/math.sqrt(model.n))}
        return out
    z=np.ones(model.d);u=model.fom(z);p,_=model.physical(z);grad=model.B@u
    _,tan=model._edge_law(grad);iron_abs=np.abs(grad[model.iron])
    F,_=model._terms(u,p)
    return {"maximum_load":{"coil_multipliers":p.tolist(),
            "iron_abs_gradient_quantiles":np.quantile(iron_abs,[0,.5,.9,.99,1]).tolist(),
            "iron_fraction_above_bsat":float(np.mean(iron_abs>model.bsat)),
            "minimum_tangent_reluctivity":float(tan.min()),
            "rms_residual":float(np.linalg.norm(F)/math.sqrt(model.n))}}


def gradient_check(model,V,seed=1729):
    rng=np.random.default_rng(seed);z=rng.uniform(-.7,.7,model.d)
    value,g=model.indicator(z,V,True);h=2e-6;fd=np.zeros(model.d)
    for j in range(model.d):
        zp=z.copy();zm=z.copy();zp[j]+=h;zm[j]-=h
        fd[j]=(model.indicator(zp,V,False)-model.indicator(zm,V,False))/(2*h)
    return {"z":z.tolist(),"value":value,"analytic":g.tolist(),"finite_difference":fd.tolist(),
            "relative_discrepancy":float(np.linalg.norm(g-fd)/max(np.linalg.norm(fd),1e-14))}


def run_case(model,rank,seeds,lrs,pool_sizes,outdir):
    Z,design=test_design(model);truths,truth_seconds=precompute_truth(model,Z)
    # Validate the differentiated objective on a nontrivial rank-two space.
    V=np.empty((model.n,0))
    for z in [np.zeros(model.d),np.full(model.d,.37)]:
        V,_=x_orth_append(V,model.fom(z),model.X)
    check=gradient_check(model,V)
    rows=[]
    for seed in seeds:
        for lr in lrs:
            name=f"continuous_lr{lr:g}"
            print(model.name,name,"seed",seed,flush=True)
            tr=train(model,"continuous",rank,seed,lr=lr)
            rows.append({"method":name,"seed":seed,**audit(model,tr,Z,truths)})
        for ps in pool_sizes:
            for kind in ["pool_uniform","pool_boundary"]:
                name=f"{kind}_{ps}"
                print(model.name,name,"seed",seed,flush=True)
                tr=train(model,kind,rank,seed,pool_size=ps)
                rows.append({"method":name,"seed":seed,**audit(model,tr,Z,truths)})
    summary={}
    for method in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==method]
        vals=np.array([r['max_relative_error'] for r in rr])
        summary[method]={"runs":len(rr),"max_error_median":float(np.median(vals)),
                         "max_error_min":float(vals.min()),"max_error_max":float(vals.max()),
                         "mean_error_median":float(np.median([r['mean_relative_error'] for r in rr])),
                         "training_seconds_median":float(np.median([r['training_seconds'] for r in rr])),
                         "objective_calls":rr[0]['objective_calls']}
    data={"case":model.name,"parameter_dimension":model.d,"state_dimension":model.n,
          "rank":rank,"seeds":seeds,"adam_steps":100,"single_start":True,
          "box_handling":"z=tanh(y); no physical-parameter clipping",
          "selection":"stability-scaled relative dual residual; truth never used",
          "test_design":design,"test_size":len(Z),"truth_seconds":truth_seconds,
          "gradient_check":check,"physics_audit":physics_audit(model),
          "rows":rows,"summary":summary}
    path=outdir/f"{model.name}.json";path.write_text(json.dumps(data,indent=2))
    return data


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--case',choices=['pb','magnet','both'],default='both')
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--outdir',type=Path,default=Path('nonlinear_pde_results'))
    args=ap.parse_args();args.outdir.mkdir(exist_ok=True)
    seeds=[11] if args.quick else [11,22,33]
    lrs=[.03,.1,.3]
    pool_sizes=[100] if args.quick else [100,1000]
    all_data={}
    if args.case in ['pb','both']:
        m=PoissonBoltzmann(grid=24 if args.quick else 36)
        all_data['pb']=run_case(m,8 if args.quick else 12,seeds,lrs,pool_sizes,args.outdir)
    if args.case in ['magnet','both']:
        m=SaturatingMagnet(grid=20 if args.quick else 28)
        all_data['magnet']=run_case(m,7 if args.quick else 10,seeds,lrs,pool_sizes,args.outdir)
    manifest={"created_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              "python":platform.python_version(),"numpy":np.__version__,
              "cases":{k:{"summary":v['summary'],"gradient_check":v['gradient_check']}
                       for k,v in all_data.items()}}
    (args.outdir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
