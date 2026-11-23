#!/usr/bin/env python3
"""Continuous greedy on a Taylor--Hood backward-facing-step flow model.

The base flow is obtained by Picard continuation for the steady incompressible
Navier--Stokes equations.  The benchmark then evolves the linearized
perturbation equation.  Parameters are coefficients of localized, physically
specified disturbance forces projected onto the discretely divergence-free
space.

Greedy selection uses a time-discrete residual norm.
The velocity residual is projected away from pressure gradients.  Full-order
response maps are used only for independent deployment-error auditing and to
retrieve the selected high-fidelity trajectory.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from skfem import (Basis, BilinearForm, ElementTriP1, ElementTriP2,
                   ElementVector, MeshTri, asm, condense, solve)
from skfem.helpers import ddot, div, dot, grad, mul
from skfem.models import mass
from skfem.models.general import divergence
from skfem.models.poisson import vector_laplace

from examples.flow_elasticity.rigorous_candidate_bench import random_pool, sphere_adam, sphere_power


@BilinearForm
def convection(u, v, w):
    return dot(mul(grad(u), w['U']), v)


@BilinearForm
def base_gradient(u, v, w):
    # (u . grad) U
    return dot(mul(w['gradU'], u), v)


def make_step_mesh(nx=32, ny=12):
    # Upstream channel: [-1,0]x[0,1]; downstream: [0,6]x[-1,1].
    xs=np.linspace(-1.0,6.0,nx+1)
    ys=np.linspace(-1.0,1.0,ny+1)
    m=MeshTri.init_tensor(xs,ys)
    cent=m.p[:,m.t].mean(axis=1)
    solid=np.where((cent[0] < -1e-12) & (cent[1] < -1e-12))[0]
    m=m.remove_elements(solid).remove_unused_nodes()
    bf=m.boundary_facets(); mid=m.p[:,m.facets[:,bf]].mean(axis=1)
    inlet=bf[np.isclose(mid[0],-1.0)]
    outlet=bf[np.isclose(mid[0],6.0)]
    io=np.union1d(inlet,outlet)
    walls=np.setdiff1d(bf,io)
    return m, inlet, outlet, walls


@dataclass
class StepModel:
    mesh: MeshTri
    ub: Basis
    pb: Basis
    Iu: np.ndarray
    B: sp.csc_matrix
    M: sp.csc_matrix
    A: sp.csc_matrix
    Z: np.ndarray
    Y: list
    forcing_envelope: np.ndarray
    dt: float
    nt: int
    base_u: np.ndarray
    metadata: dict
    proj_pressure_lu: object
    diagM: np.ndarray


def _boundary_values(ub, inlet_dofs):
    vals=np.zeros(ub.N)
    dv=ub.get_dofs(facets=inlet_dofs)
    ix=dv.all('u^1')
    y=ub.doflocs[1,ix]
    vals[ix]=6.0*y*(1.0-y)  # unit mean velocity on y in [0,1]
    return vals


def solve_base_flow(mesh, inlet, walls, Re=80.0, maxit=160):
    ub=Basis(mesh,ElementVector(ElementTriP2()),intorder=4)
    pb=Basis(mesh,ElementTriP1(),intorder=4)
    K=asm(vector_laplace,ub).tocsc()
    B=asm(divergence,ub,pb).tocsc()
    Du=np.union1d(ub.get_dofs(facets=inlet).all(),ub.get_dofs(facets=walls).all())
    bc=_boundary_values(ub,inlet)
    # One pressure value fixes the nullspace.
    D=np.r_[Du,ub.N]
    xbc=np.zeros(ub.N+pb.N); xbc[:ub.N]=bc
    u=bc.copy(); p=np.zeros(pb.N)
    history=[]
    # Continuation is substantially more reliable than jumping directly to Re.
    schedule=[10.,20.,40.,Re] if Re>40 else [10.,20.,Re]
    schedule=sorted(set(min(Re,r) for r in schedule))
    for rr in schedule:
        nu=1.0/rr
        for it in range(maxit):
            Ui=ub.interpolate(u)
            C=asm(convection,ub,U=Ui).tocsc()
            S=sp.bmat([[nu*K+C,-B.T],[B,None]],format='csc')
            rhs=np.zeros(ub.N+pb.N)
            Sc,bcnd,x,I=condense(S,rhs,x=xbc,D=D)
            sol=x.copy(); sol[I]=spla.spsolve(Sc,bcnd)
            unew,pnew=sol[:ub.N],sol[ub.N:]
            # Under-relax high-Re Picard iterations.
            omega=0.72 if rr<=40 else 0.45
            rel=np.linalg.norm(unew-u)/max(np.linalg.norm(unew),1e-14)
            u=(1-omega)*u+omega*unew; p=(1-omega)*p+omega*pnew
            history.append((rr,it+1,float(rel)))
            if rel<2e-9: break
        if history[-1][2] > 1e-8:
            raise RuntimeError(f'base Picard did not converge at Re={rr}: {history[-1]}')
    return ub,pb,K,B,u,p,history


def _mass_vector(ub):
    @BilinearForm
    def vmass(u,v,w): return dot(u,v)
    return asm(vmass,ub).tocsc()


def _project_initial_modes(ub, Iu, B, M, d, seed=9182):
    """Project fixed localized Gaussian vortices into ker(B).

    Columns are normalized individually.  We deliberately do not whiten their
    Gram matrix: whitening would use a global algebraic transformation and
    destroy the meaning of an isotropic distribution over physical forcing
    coefficients.
    """
    xy=ub.doflocs[:,Iu]
    n=len(Iu); W=np.zeros((n,d))
    # Deterministic physical centers; seed only rotates signs slightly.
    rng=np.random.default_rng(seed)
    centers=[]
    ncol=4; nrow=math.ceil(d/ncol)
    for k in range(d):
        x=-0.85+2.15*((k%ncol)+0.5)/ncol
        y=0.08+0.84*((k//ncol)+0.5)/nrow
        centers.append((x,y))
    # P2 vector dofs have component-specific global indices.  Build nodal
    # streamfunction-inspired fields at the component dof locations.
    fullx=np.zeros((ub.N,d))
    allx=np.arange(ub.N)
    # ElementVector interlaces components in current scikit-fem.
    compx=(allx%2)==0; compy=~compx
    X,Y=ub.doflocs
    for k,(cx,cy) in enumerate(centers):
        sx,sy=0.34,0.18
        g=np.exp(-0.5*((X-cx)/sx)**2-0.5*((Y-cy)/sy)**2)
        # curl of a Gaussian streamfunction
        fullx[compx,k]=-(Y[compx]-cy)/(sy*sy)*g[compx]
        fullx[compy,k]=(X[compy]-cx)/(sx*sx)*g[compy]
        fullx[:,k]*=rng.choice([-1.,1.])
    W=fullx[Iu]
    Mii=M[Iu][:,Iu].tocsc(); Bi=B[:,Iu].tocsc()
    # Remove one dependent pressure constraint.
    Br=Bi[1:]
    saddle=sp.bmat([[Mii,-Br.T],[Br,None]],format='csc')
    lu=spla.splu(saddle)
    rhs=np.vstack((Mii@W,np.zeros((Br.shape[0],d))))
    Z=lu.solve(rhs)[:n]
    G=Z.T@(Mii@Z); ew=la.eigvalsh((G+G.T)/2)
    if ew[0] < 1e-10*ew[-1]:
        raise RuntimeError('initial-mode dictionary is rank deficient')
    Z/=np.sqrt(np.maximum(np.diag(G),1e-30))[None,:]
    return Z


def build_step_model(nx=32,ny=12,d=16,Re=80.,dt=0.08,nt=35):
    mesh,inlet,outlet,walls=make_step_mesh(nx,ny)
    ub,pb,Kfull,Bfull,base_u,base_p,bhist=solve_base_flow(mesh,inlet,walls,Re)
    Mfull=_mass_vector(ub)
    Dpert=np.union1d(ub.get_dofs(facets=inlet).all(),ub.get_dofs(facets=walls).all())
    Iu=np.setdiff1d(np.arange(ub.N),Dpert)
    M=Mfull[Iu][:,Iu].tocsc(); B=Bfull[:,Iu].tocsc()
    Ui=ub.interpolate(base_u)
    C=asm(convection,ub,U=Ui).tocsc()
    G=asm(base_gradient,ub,gradU=Ui.grad).tocsc()
    A=(Kfull/Re+C+G)[Iu][:,Iu].tocsc()
    Z=_project_initial_modes(ub,Iu,Bfull,Mfull,d)

    # Remove one pressure equation and factor the implicit Euler saddle system.
    Br=B[1:].tocsc()
    H=M/dt+A
    St=sp.bmat([[H,-Br.T],[Br,None]],format='csc')
    lu=spla.splu(St)
    envelope=np.sin(np.pi*np.arange(1,nt+1)/(nt+1))**2
    Y=[np.zeros_like(Z)]
    cur=np.zeros_like(Z)
    for k in range(nt):
        rhs=np.vstack(((M@cur)/dt+(M@Z)*envelope[k],
                       np.zeros((Br.shape[0],d))))
        cur=lu.solve(rhs)[:len(Iu)]
        Y.append(cur)

    # Diagonal mass dual projector removes pressure-gradient components.
    diagM=np.maximum(M.diagonal(),1e-14)
    Sch=(Br@sp.diags(1.0/diagM)@Br.T).tocsc()
    # A small gauge regularization protects against extra nearly dependent rows.
    Sch=Sch+1e-12*sp.eye(Sch.shape[0],format='csc')
    plu=spla.splu(Sch)
    div0=np.linalg.norm(Br@Z)/max(np.linalg.norm(Z),1e-14)
    meta={'nx':nx,'ny':ny,'d':d,'Re':Re,'dt':dt,'nt':nt,
          'triangles':int(mesh.nelements),'velocity_dofs_total':int(ub.N),
          'velocity_dofs_free':int(len(Iu)),'pressure_dofs':int(pb.N),
          'base_picard_iterations':len(bhist),'base_last_relative_update':bhist[-1][2],
          'initial_dictionary_relative_divergence':float(div0)}
    return StepModel(mesh,ub,pb,Iu,B,M,A,Z,Y,envelope,dt,nt,base_u,meta,plu,diagM)


def _m_append(V,u,M,tol=1e-10):
    w=u.copy()
    if V.shape[1]:
        for _ in range(2): w-=V@(V.T@(M@w))
    n2=float(w@(M@w))
    if n2<tol*tol:return V,False
    return np.column_stack((V,w/math.sqrt(n2))),True


def _projected_dual_gram(model,R):
    """Return R' P_D' D^{-1} P_D R without forming the projector."""
    Br=model.B[1:]
    DR=R/model.diagM[:,None]
    p=model.proj_pressure_lu.solve(Br@DR)
    X=DR-(Br.T@p)/model.diagM[:,None]
    return R.T@X


def indicator_matrix(model,V):
    """Quadratic residual-indicator matrix for all parameter directions."""
    d=model.Z.shape[1]; M,A=model.M,model.A
    if V.shape[1]:
        Mr=V.T@(M@V); Ar=V.T@(A@V)
        L=la.solve(Mr/model.dt+Ar,Mr/model.dt)
        G=la.solve(Mr/model.dt+Ar,V.T@(M@model.Z))
        states=[np.zeros((V.shape[1],d))]
        for k in range(model.nt):
            states.append(L@states[-1]+G*model.forcing_envelope[k])
    else:
        states=[np.zeros((0,d)) for _ in range(model.nt+1)]
    Q=np.zeros((d,d))
    for k in range(model.nt):
        if V.shape[1]:
            R=(M@(V@(states[k+1]-states[k]))/model.dt
               +A@(V@states[k+1])
               -(M@model.Z)*model.forcing_envelope[k])
        else:
            R=-(M@model.Z)*model.forcing_envelope[k]
        Q+=model.dt*_projected_dual_gram(model,R)
    return (Q+Q.T)/2,states


def exact_deployment_error(model,V):
    """Exact max over coefficient sphere and all saved times."""
    M=model.M
    denom=max(float(la.eigvalsh(Y.T@(M@Y))[-1]) for Y in model.Y)
    worst=0.; worstk=0
    for k,Y in enumerate(model.Y):
        E=Y-V@(V.T@(M@Y)) if V.shape[1] else Y
        val=float(la.eigvalsh(E.T@(M@E))[-1])
        if val>worst:worst,worstk=val,k
    return math.sqrt(max(worst,0)/max(denom,1e-300)),worstk


def add_worst_time_snapshot(model,V,z):
    M=model.M; best=-1.; bestu=None; bestk=0
    for k,Y in enumerate(model.Y):
        u=Y@z; e=u-V@(V.T@(M@u)) if V.shape[1] else u
        val=float(e@(M@e))
        if val>best:best,bestu,bestk=val,u,k
    V,added=_m_append(V,bestu,M)
    return V,added,bestk


def run_greedy(model,method,seed,N=10,steps=100,lr=.06,pool_M=101):
    rng=np.random.default_rng(seed); V=np.zeros((model.M.shape[0],0)); hist=[]
    t0=time.perf_counter()
    for n in range(N):
        Q,_=indicator_matrix(model,V); oracle=float(la.eigvalsh(Q)[-1])
        if method=='adam':z,val,tr=sphere_adam(Q,rng.normal(size=Q.shape[0]),steps,lr)
        elif method=='power':z,val,tr=sphere_power(Q,rng.normal(size=Q.shape[0]),steps)
        elif method=='pool':z,val,tr=random_pool(Q,rng,pool_M)
        elif method=='pool4096':z,val,tr=random_pool(Q,rng,4096)
        elif method=='oracle':
            ew,EV=la.eigh(Q);z=EV[:,-1];val=float(ew[-1]);tr=[val]
        else:raise ValueError(method)
        V,added,tk=add_worst_time_snapshot(model,V,z)
        err,wk=exact_deployment_error(model,V)
        hist.append({'round':n+1,'indicator_ratio':math.sqrt(max(val,0)/max(oracle,1e-300)),
                     'deployment_normalized_worst_velocity_error':err,
                     'selected_snapshot_time_index':int(tk),'worst_deployment_time_index':int(wk),
                     'added':bool(added),'evaluations':len(tr)})
    return {'method':method,'seed':seed,'runtime_s':time.perf_counter()-t0,'history':hist}


def experiment(out,quick=False):
    configs=[(20,8),(28,10)] if quick else [(28,10),(40,14)]
    d=12 if quick else 16; N=6 if quick else 10
    seeds=list(range(3 if quick else 10)); grids={}
    for nx,ny in configs:
        model=build_step_model(nx,ny,d=d,Re=60. if quick else 80.,dt=.10 if quick else .08,nt=24 if quick else 35)
        runs=[]
        for method in ('adam','power','pool','pool4096','oracle'):
            ss=seeds if method!='oracle' else [0]
            for seed in ss:runs.append(run_greedy(model,method,seed,N,100,.06,101))
        grids[f'{nx}x{ny}']={'metadata':model.metadata,'runs':runs}
    result={'benchmark':'Taylor-Hood linearized backward-facing-step flow',
            'selection':'pressure-projected time-discrete residual indicator',
            'deployment':'exact worst-case velocity error over coefficient sphere and saved times',
            'budget':{'continuous_steps':100,'matched_pool_size':101,'large_pool':4096},
            'd':d,'N':N,'grids':grids}
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(result,indent=2))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true')
    ap.add_argument('--out',default='candidate_bench_results/step_flow.json');a=ap.parse_args()
    r=experiment(a.out,a.quick);print(json.dumps({'written':a.out,'benchmark':r['benchmark']},indent=2))


if __name__=='__main__':main()
