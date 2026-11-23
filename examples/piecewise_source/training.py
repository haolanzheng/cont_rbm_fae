"""Greedy training with GD, a fixed random pool, or a uniform mesh."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import itertools
from time import perf_counter, process_time
import numpy as np
from run import ascend, append, residual_gram


def train(B,G,seed,M,*,steps=80,max_n=None):
    cpu0=process_time();tick=perf_counter();d=len(B);N=d//2 if max_n is None else max_n
    Q=np.zeros((d,0));selected=[]
    rng=np.random.default_rng(seed if M else seed+10000)
    if M:
        pool=rng.uniform(-1,1,(M,d));U=pool@B.T;scores=np.sum(U**2,axis=1)
    setup=perf_counter()-tick
    for n in range(N):
        z=pool[np.argmax(scores)] if M else ascend(residual_gram(B,G,Q),rng.uniform(-1,1,d),steps)
        Q,q=append(B,Q,z)
        selected.append(z.copy())
        if M and n<N-1:scores-=(U@q)**2
    measured=dict(wall_seconds=perf_counter()-tick,cpu_seconds=process_time()-cpu0,setup_seconds=setup)
    return Q,np.asarray(selected),measured


def train_method(B,G,seed,method,steps,max_n=None):
    if not method.startswith('grid'):
        return train(B,G,seed,0 if method=='gd' else int(method[4:]),steps=steps,max_n=max_n)
    tick=perf_counter();d=len(B);N=d//2 if max_n is None else max_n
    h=float(method[4:]);points=np.array(list(itertools.product(np.linspace(-1,1,round(2/h)+1),repeat=d)))
    U=points@B.T;scores=np.sum(U**2,axis=1);Q=np.zeros((d,0));Z=[]
    for n in range(N):
        z=points[np.argmax(scores)];Q,q=append(B,Q,z);Z.append(z.copy())
        if n<N-1:scores-=(U@q)**2
    return Q,np.asarray(Z),dict(wall_seconds=perf_counter()-tick)
