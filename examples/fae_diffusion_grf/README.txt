Diffusion coefficient fields for the FAE
=======================================

On (0,1)^2, solve -div(a grad u)=1 with zero Dirichlet data.
The input is a=exp(log(10)*tanh(g)), where g is a centered unit-variance
Matérn field with nu=1 and correlation length 0.2. Thus 0.1<a<10.

run.py generates a PDE pilot using circulant embedding and P1 finite
elements. prepare_fae.py generates the larger coefficient-only FAE dataset;
no PDE solves are required for the additional reconstruction samples.

From the repository root:
    python examples/fae_diffusion_grf/run.py --output runs/fae_pde_new
    python run.py fae-data --output runs/fae_new/data

The pilot uses 256/64/64 train/validation/test fields on 129x129 nodes.
The FAE dataset uses 2048/256/256 fields on the same grid. See README_FAE.txt
for training, initialization, checkpoint export and backend alignment.

Data, numerical checks, logs and checkpoints are saved under runs/ and
excluded from Git. The current model is a reconstruction experiment;
no FAE-based greedy-search result is claimed.
