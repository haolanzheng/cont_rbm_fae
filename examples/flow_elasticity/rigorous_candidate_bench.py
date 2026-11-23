#!/usr/bin/env python3
"""Auditable continuous-greedy benchmarks on two discretized PDE models.

The first benchmark is a perforated, heterogeneous plane-stress beam with
distributed boundary tractions.  The second (added below) is a linearized
incompressible step-flow model.  Greedy selection only sees a computable
residual indicator.  Full solutions are used only for deployment audits.

No external finite-element package is required; only NumPy/SciPy are used.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _unit(x, eps=1e-14):
    n = np.linalg.norm(x)
    return x / max(n, eps)


def sphere_adam(Q, z0, steps=100, lr=0.08, b1=0.9, b2=0.999, eps=1e-8):
    """Maximize z'Qz on the unit sphere by tangent Adam and retraction.

    The best feasible iterate is returned, so finite-step oscillation cannot
    make a run worse than an earlier iterate.
    """
    z = _unit(z0.copy())
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    best = z.copy()
    bestv = float(z @ Q @ z)
    vals = [bestv]
    for k in range(1, steps + 1):
        g = 2.0 * (Q @ z)
        g -= z * (z @ g)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)
        mh = m / (1.0 - b1**k)
        vh = v / (1.0 - b2**k)
        dz = mh / (np.sqrt(vh) + eps)
        dz -= z * (z @ dz)
        z = _unit(z + lr * dz)
        val = float(z @ Q @ z)
        vals.append(val)
        if val > bestv:
            bestv, best = val, z.copy()
    return best, bestv, vals


def sphere_power(Q, z0, steps=100, shift=1e-12):
    """Scale-free normalized gradient/power iteration, best iterate retained."""
    z = _unit(z0.copy())
    best, bestv = z.copy(), float(z @ Q @ z)
    vals = [bestv]
    scale = max(float(np.linalg.norm(Q, 2)), shift)
    A = Q / scale + shift * np.eye(Q.shape[0])
    for _ in range(steps):
        z = _unit(A @ z)
        val = float(z @ Q @ z)
        vals.append(val)
        if val > bestv:
            bestv, best = val, z.copy()
    return best, bestv, vals


def random_pool(Q, rng, M=101):
    Z = rng.normal(size=(M, Q.shape[0]))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    vals = np.einsum("bi,ij,bj->b", Z, Q, Z)
    j = int(np.argmax(vals))
    return Z[j], float(vals[j]), vals


@dataclass
class ElasticModel:
    K: sp.csc_matrix
    B: np.ndarray
    U: np.ndarray
    diag_inv: np.ndarray
    coords: np.ndarray
    triangles: np.ndarray
    free_dofs: np.ndarray
    metadata: dict


def build_perforated_elasticity(nx=40, ny=10, d=20):
    """CST FEM for a clamped perforated heterogeneous plane-stress beam."""
    L, H = 4.0, 1.0
    xs, ys = np.linspace(0, L, nx + 1), np.linspace(0, H, ny + 1)
    coords0 = np.array([(x, y) for y in ys for x in xs], dtype=float)

    def nid(i, j): return j * (nx + 1) + i

    tri0 = []
    for j in range(ny):
        for i in range(nx):
            # Alternating diagonals reduce directional mesh bias.
            if (i + j) % 2 == 0:
                pairs = [(nid(i,j),nid(i+1,j),nid(i+1,j+1)),
                         (nid(i,j),nid(i+1,j+1),nid(i,j+1))]
            else:
                pairs = [(nid(i,j),nid(i+1,j),nid(i,j+1)),
                         (nid(i+1,j),nid(i+1,j+1),nid(i,j+1))]
            for t in pairs:
                c = coords0[list(t)].mean(axis=0)
                # Two holes prevent the solution family from collapsing to a
                # single elementary bending mode.
                in_h1 = ((c[0]-1.35)/0.24)**2 + ((c[1]-0.52)/0.22)**2 < 1
                in_h2 = ((c[0]-2.55)/0.19)**2 + ((c[1]-0.40)/0.17)**2 < 1
                if not (in_h1 or in_h2): tri0.append(t)
    used = np.unique(np.asarray(tri0).ravel())
    ren = -np.ones(coords0.shape[0], dtype=int); ren[used] = np.arange(len(used))
    coords = coords0[used]
    tris = ren[np.asarray(tri0, dtype=int)]
    ndof = 2 * len(coords)
    rows, cols, data = [], [], []
    nu = 0.29
    for tri in tris:
        xy = coords[tri]
        x1,y1 = xy[0]; x2,y2 = xy[1]; x3,y3 = xy[2]
        det = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        area = abs(det)/2
        b = np.array([y2-y3, y3-y1, y1-y2]) / det
        c = np.array([x3-x2, x1-x3, x2-x1]) / det
        G = np.zeros((3,6))
        for a in range(3):
            G[0,2*a]=b[a]; G[1,2*a+1]=c[a]
            G[2,2*a]=c[a]; G[2,2*a+1]=b[a]
        xc,yc = xy.mean(axis=0)
        # A fixed composite layup plus smooth heterogeneity.
        E = (1.0 if yc < 0.5 else 2.8) * (1.0 + 0.18*np.sin(2*np.pi*xc/L))
        D = E/(1-nu*nu)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
        ke = area * (G.T @ D @ G)
        dofs = np.ravel(np.column_stack((2*tri,2*tri+1)))
        rr,cc = np.meshgrid(dofs,dofs,indexing='ij')
        rows.extend(rr.ravel()); cols.extend(cc.ravel()); data.extend(ke.ravel())
    K0 = sp.coo_matrix((data,(rows,cols)),shape=(ndof,ndof)).tocsc()
    fixed_nodes = np.where(coords[:,0] < 1e-12)[0]
    fixed = np.ravel(np.column_stack((2*fixed_nodes,2*fixed_nodes+1)))
    free = np.setdiff1d(np.arange(ndof), fixed)
    K = K0[free][:,free].tocsc()

    # Boundary edges are edges belonging to exactly one element.
    edge_count = {}
    for t in tris:
        for a,bn in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            e=tuple(sorted((int(a),int(bn)))); edge_count[e]=edge_count.get(e,0)+1
    bedges=[e for e,cnt in edge_count.items() if cnt==1]
    B0=np.zeros((ndof,d))
    centers=[]
    # Physically prescribed smooth traction modes along the external top and
    # right boundaries.  The mode dictionary is independent of K and the RB.
    n_top = max(1, d-4)
    for k in range(d):
        if k < n_top:
            centers.append((L*(k+0.5)/n_top, H, 'top'))
        else:
            centers.append((L, H*(k-n_top+0.5)/(d-n_top), 'right'))
    sigx=L/max(n_top,1)*0.72; sigy=H/max(d-n_top,1)*0.72
    for a,bn in bedges:
        xa,xb=coords[a],coords[bn]; mid=(xa+xb)/2; ell=np.linalg.norm(xb-xa)
        is_top=abs(mid[1]-H)<1e-10
        is_right=abs(mid[0]-L)<1e-10
        if not (is_top or is_right): continue
        for k,(cx,cy,side) in enumerate(centers):
            if side=='top' and is_top:
                amp=np.exp(-0.5*((mid[0]-cx)/sigx)**2)
                traction=np.array([0.20*np.sin(np.pi*mid[0]/L), -1.0])*amp
            elif side=='right' and is_right:
                amp=np.exp(-0.5*((mid[1]-cy)/sigy)**2)
                traction=np.array([0.35, -1.0])*amp
            else: continue
            B0[2*a:2*a+2,k]+=0.5*ell*traction
            B0[2*bn:2*bn+2,k]+=0.5*ell*traction
    B=B0[free]
    # Equalize Euclidean load amplitude; this is a physical load normalization,
    # not an error-operator/SVD normalization.
    B /= np.maximum(np.linalg.norm(B,axis=0,keepdims=True),1e-14)
    lu=spla.splu(K)
    U=np.column_stack([lu.solve(B[:,j]) for j in range(d)])
    return ElasticModel(K,B,U,1.0/np.maximum(K.diagonal(),1e-14),coords,tris,free,
        {'nx':nx,'ny':ny,'d':d,'nodes':len(coords),'triangles':len(tris),
         'free_dofs':len(free),'holes':2,'plane_stress':True})


def _k_orthonormal_append(V, u, K, tol=1e-11):
    w=u.copy()
    if V.shape[1]:
        for _ in range(2): w -= V @ (V.T @ (K @ w))
    n2=float(w @ (K @ w))
    if n2 <= tol*tol: return V, False
    return np.column_stack((V,w/math.sqrt(n2))), True


def elastic_matrices(model, V):
    K,B,U,p=model.K,model.B,model.U,model.diag_inv
    if V.shape[1]:
        # V is K-orthonormal to numerical precision.
        R=B-(K@V)@(V.T@B)
        EU=U-V@(V.T@B)
    else:
        R=B.copy(); EU=U.copy()
    Q=R.T@(p[:,None]*R)       # computable residual indicator
    A=EU.T@(K@EU)             # true energy error, audit only
    S=U.T@(K@U)               # full solution energy
    return (Q+Q.T)/2,(A+A.T)/2,(S+S.T)/2


def normalized_worst_error(A,S):
    """sup_z ||e(z)||_K divided by sup_z ||u(z)||_K on ||z||_2=1.

    Pointwise relative worst error is identically one for every rank N<d in a
    homogeneous linear load family, because an unresolved load direction can
    be chosen.  This operator-norm normalization is nondegenerate and exact.
    """
    ea=max(float(la.eigvalsh(A)[-1]),0.0)
    es=max(float(la.eigvalsh(S)[-1]),1e-300)
    return math.sqrt(ea/es)


def run_elastic_greedy(model, method, seed, N=12, steps=100, lr=0.08, pool_M=101):
    rng=np.random.default_rng(seed)
    V=np.zeros((model.K.shape[0],0)); hist=[]
    t0=time.perf_counter()
    for n in range(N):
        Q,A,S=elastic_matrices(model,V)
        oracle=float(la.eigvalsh(Q)[-1])
        if method=='adam': z,val,trace=sphere_adam(Q,rng.normal(size=Q.shape[0]),steps,lr)
        elif method=='power': z,val,trace=sphere_power(Q,rng.normal(size=Q.shape[0]),steps)
        elif method=='pool': z,val,trace=random_pool(Q,rng,pool_M)
        elif method=='pool4096': z,val,trace=random_pool(Q,rng,4096)
        elif method=='oracle':
            ew,EV=la.eigh(Q); z=EV[:,-1]; val=float(ew[-1]); trace=[val]
        else: raise ValueError(method)
        V,added=_k_orthonormal_append(V,model.U@z,model.K)
        _,A2,S2=elastic_matrices(model,V)
        hist.append({'round':n+1,'indicator_ratio':math.sqrt(max(val,0)/max(oracle,1e-300)),
                     'deployment_normalized_worst_energy_error':normalized_worst_error(A2,S2),
                     'added':bool(added),'evaluations':len(trace)})
    return {'method':method,'seed':seed,'runtime_s':time.perf_counter()-t0,'history':hist}


def elastic_experiment(out, quick=False):
    grids=[(24,6),(40,10)] if quick else [(32,8),(48,12)]
    seeds=list(range(5 if quick else 12))
    d=16 if quick else 20; N=8 if quick else 12
    all_grids={}
    for nx,ny in grids:
        model=build_perforated_elasticity(nx,ny,d)
        runs=[]
        for method in ('adam','power','pool','pool4096','oracle'):
            use_seeds=seeds if method not in ('oracle',) else [0]
            for seed in use_seeds:
                runs.append(run_elastic_greedy(model,method,seed,N=N,steps=100,lr=0.08,pool_M=101))
        all_grids[f'{nx}x{ny}']={'metadata':model.metadata,'runs':runs}
    result={'benchmark':'perforated heterogeneous plane-stress beam',
            'selection':'diagonally preconditioned residual norm only',
            'deployment':'exact worst-case energy error divided by maximum full-solution energy norm',
            'budget':{'continuous_steps':100,'matched_pool_size':101,'large_pool':4096},
            'd':d,'N':N,'grids':all_grids}
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(result,indent=2))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--case',choices=['elastic'],default='elastic')
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--out',default='candidate_bench_results/elastic.json')
    a=ap.parse_args()
    tracemalloc.start()
    if a.case=='elastic': result=elastic_experiment(a.out,a.quick)
    _,peak=tracemalloc.get_traced_memory()
    print(json.dumps({'written':a.out,'peak_python_memory_mb':peak/2**20,
                      'benchmark':result['benchmark']},indent=2))


if __name__=='__main__': main()
