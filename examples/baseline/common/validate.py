"""Small checks of the shared model algebra, projection and evaluation bounds."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from .box_bounds import maximum,width,vertex_bounds
from ..cube.search import search as cube_search
from ..geometry_cpu import search

def main():
 rng=np.random.default_rng(42)
 for d in [3,6]:
  B=rng.normal(size=(d,d));H=B.T@B
  lo,hi=maximum(H);exact=vertex_bounds(H)[0]
  assert lo<=exact+1e-10 and hi>=exact-1e-10
 H=np.diag(np.geomspace(1,.001,20));lo,hi=maximum(H)
 assert abs(lo-np.trace(H))<1e-12 and hi>=lo
 wl,wu=width(H,10);assert np.allclose([wl,wu],np.sqrt(np.diag(H)[10:].sum()))
 for domain in ['sphere','ball','cube']:
  H=np.diag([1.,.5,.2]);z=np.array([.2,.5,-.3]);z=z/np.linalg.norm(z) if domain=='sphere' else z
  if domain=='cube':v,_=cube_search(H,z,alpha=.6,T=40);assert np.max(np.abs(v))<=1
  else:v,_=search(H,z,steps=40,domain=domain,alpha=.6,ball_step_rule='normalized');assert np.linalg.norm(v)<=1+1e-12
  assert v@H@v>=z@H@z-1e-12
 print('Projection, diagonal width and exhaustive small-box maximum checks passed.')
if __name__=='__main__':main()
