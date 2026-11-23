"""Explicit alternatives for box-constrained quadratic ascent.

All methods maximize J=z^T H z/||H||_F. The returned point is best visited.
Counts distinguish dense H-vector actions from cheap coordinate endpoint formulas.
"""
import numpy as np

METHODS=['clip','first_hit','first_hit_relaxed','relaxed_clip','normal_scaled',
         'projected_linesearch','active_face','clip_coordinates',
         'linesearch_coordinates','active_face_coordinates','coordinates_only']


def first_hit(z,step):
    nonzero=step!=0
    if not np.any(nonzero):return z.copy()
    distances=np.where(step>0,1-z,1+z)
    fraction=min(1.,max(0.,float(np.min(distances[nonzero]/np.abs(step[nonzero])))))
    return np.clip(z+fraction*step,-1,1)


def search(H,initial,method,steps=80,alpha=.6,relaxation=.5,
           multipliers=(.25,.5,1.,2.,4.,8.,16.),coordinate_sweeps=10):
    if method not in METHODS:raise ValueError(method)
    scale=np.linalg.norm(H);A=H/scale if scale else np.zeros_like(H)
    z=np.asarray(initial,dtype=float).copy();hz=A@z;value=float(z@hz)
    counts={'h_actions':1,'objective_values':1,'gradient_evaluations':0,
            'coordinate_endpoint_scores':0,'coordinate_sweeps':0,'coordinate_changes':0,
            'updates':0,'min_increment':0.,'max_feasibility_defect':0.}
    best=z.copy();best_value=value
    core={'clip_coordinates':'clip','linesearch_coordinates':'projected_linesearch',
          'active_face_coordinates':'active_face'}.get(method,method)
    for _ in range(0 if core=='coordinates_only' else steps):
        grad=2*hz;counts['gradient_evaluations']+=1
        eta=alpha/np.sqrt(1+grad@grad)
        if core in ['first_hit','first_hit_relaxed']:
            trial=first_hit(z,eta*grad)
            if core=='first_hit_relaxed':trial=z+relaxation*(trial-z)
        elif core=='active_face':
            direction=grad.copy();blocked=((z>=1-1e-14)&(grad>0))|((z<=-1+1e-14)&(grad<0))
            direction[blocked]=0
            eta=alpha/np.sqrt(1+direction@direction)
            trial=first_hit(z,eta*direction)
        elif core=='normal_scaled':
            # Only outward-directed motion is damped near its destination face.
            distance=np.where(grad>=0,1-z,1+z)
            trial=z+eta*np.minimum(1,np.maximum(0,distance))*grad
            trial=np.clip(trial,-1,1)  # roundoff repair, no deliberate crossing
        elif core=='projected_linesearch':
            candidates=np.clip(z+np.array(multipliers)[:,None]*eta*grad,-1,1)
            products=candidates@A;values=np.sum(candidates*products,axis=1)
            counts['h_actions']+=len(candidates);counts['objective_values']+=len(candidates)
            k=int(np.argmax(values));trial=candidates[k];new_hz=products[k];new_value=float(values[k])
        else:
            trial=np.clip(z+eta*grad,-1,1)
            if core=='relaxed_clip':trial=z+relaxation*(trial-z)
        if core!='projected_linesearch':
            new_hz=A@trial;new_value=float(trial@new_hz)
            counts['h_actions']+=1;counts['objective_values']+=1
        counts['updates']+=1;counts['min_increment']=min(counts['min_increment'],new_value-value)
        counts['max_feasibility_defect']=max(counts['max_feasibility_defect'],float(np.max(np.maximum(np.abs(trial)-1,0))))
        changed=np.max(np.abs(trial-z));z,hz,value=trial,new_hz,new_value
        if value>best_value:best,best_value=z.copy(),value
        if changed==0:break
    if method.endswith('_coordinates') or method=='coordinates_only':
        z=best.copy();hz=A@z;value=float(z@hz);counts['h_actions']+=1;counts['objective_values']+=1
        for sweep in range(coordinate_sweeps):
            changes=0;counts['coordinate_sweeps']+=1
            for i in range(len(z)):
                delta=np.array([-1-z[i],1-z[i]])
                gains=2*delta*hz[i]+A[i,i]*delta**2
                counts['coordinate_endpoint_scores']+=2
                k=int(np.argmax(gains));move=float(delta[k]);gain=float(gains[k])
                # At interior points choose a maximizing endpoint; at vertices
                # only strict improvements are accepted, preventing tie cycles.
                interior=abs(z[i])<1
                if move!=0 and (gain>1e-13 or (interior and gain>=-1e-13)):
                    z[i]+=move;hz+=move*A[:,i];value+=gain;changes+=1
            counts['coordinate_changes']+=changes
            value=float(z@(A@z));hz=A@z;counts['h_actions']+=2;counts['objective_values']+=1
            if value>best_value:best,best_value=z.copy(),value
            if changes==0:break
    counts['boundary_coordinates']=int(np.sum(np.abs(best)>=1-1e-12))
    counts['returned_value']=float(best@A@best)
    return best,counts
