#include <cmath>
#include <vector>
// Exact dense BGK residual and reduced Jacobian: no velocity truncation or
// hyper-reduction. Arrays use x,velocity,coefficient contiguous ordering.
extern "C" int bgk_rj(int nx,int nv,int k,double vw,double weight,double a,double b,
 const double* vx,const double* vy,const double* base,const double* dirs,
 const double* bt,const double* dt,const double* y,double* r,double* jac) {
 std::vector<double> f(nv),g(nv),cx(nv),cy(nv),c2(nv);
 std::vector<double> dr(k),dmx(k),dmy(k),de(k),dux(k),duy(k),dtemp(k);
 const double scale=std::sqrt(weight);
 const double factor=std::tgamma(1+.5*a)/(2*std::acos(-1.0));
 for(int x=0;x<nx;x++) {
  double rho=0,mx=0,my=0,e=0;
  for(int h=0;h<k;h++)dr[h]=dmx[h]=dmy[h]=de[h]=0;
  for(int v=0;v<nv;v++) {
   int p=x*nv+v; double ff=base[p],vv=vx[v]*vx[v]+vy[v]*vy[v];
   for(int h=0;h<k;h++) {
    double d=dirs[p*k+h];ff+=d*y[h];
    dr[h]+=d*vw;dmx[h]+=d*vx[v]*vw;dmy[h]+=d*vy[v]*vw;de[h]+=d*vv*vw;
   }
   f[v]=ff;rho+=ff*vw;mx+=ff*vx[v]*vw;my+=ff*vy[v]*vw;e+=ff*vv*vw;
  }
  if(rho<=1e-8)return 1;
  double ux=mx/rho,uy=my/rho,e2=e/rho,T=.5*(e2-ux*ux-uy*uy);
  if(T<=1e-4)return 2;
  for(int h=0;h<k;h++) {
   dux[h]=(dmx[h]-ux*dr[h])/rho;duy[h]=(dmy[h]-uy*dr[h])/rho;
   dtemp[h]=.5*(de[h]-e2*dr[h])/rho-ux*dux[h]-uy*duy[h];
  }
  double sumg=0,meanx=0,meany=0,meansq=0;
  for(int v=0;v<nv;v++) {
   cx[v]=vx[v]-ux;cy[v]=vy[v]-uy;c2[v]=cx[v]*cx[v]+cy[v]*cy[v];
   g[v]=std::exp(-c2[v]/(2*T));sumg+=g[v];
   meanx+=g[v]*cx[v];meany+=g[v]*cy[v];meansq+=g[v]*c2[v];
  }
  meanx/=sumg;meany/=sumg;meansq/=sumg;
  double nu=b*rho*factor*std::pow(4*T,.5*a);
  for(int v=0;v<nv;v++) {
   int p=x*nv+v;double maxwell=rho*g[v]/(sumg*vw),diff=maxwell-f[v],tr=bt[p];
   for(int h=0;h<k;h++)tr+=dt[p*k+h]*y[h];
   r[p]=(tr-nu*diff)*scale;
   for(int h=0;h<k;h++) {
    double dnu=nu*(dr[h]/rho+.5*a*dtemp[h]/T);
    double dM=maxwell*(dr[h]/rho+((cx[v]-meanx)*dux[h]+(cy[v]-meany)*duy[h])/T
                    +(c2[v]-meansq)*dtemp[h]/(2*T*T));
    jac[p*k+h]=(dt[p*k+h]-dnu*diff-nu*(dM-dirs[p*k+h]))*scale;
   }
  }
 }
 return 0;
}
