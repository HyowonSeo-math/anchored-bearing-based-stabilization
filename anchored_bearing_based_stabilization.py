from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from scipy.linalg import cho_factor, cho_solve

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator

plt.rcParams.update({
    'font.size': 7.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
})


def set_time_ticks(ax, horizon):
    horizon = float(horizon)
    step = 100 if horizon > 150 else 50 if horizon > 50 else 10
    ax.set_xlim(0, horizon)
    ax.set_xticks(np.arange(0, horizon + step / 2, step))
    ax.xaxis.set_minor_locator(NullLocator())


def set_sparse_log_ticks(ax, axis='y'):
    lo, hi = ax.get_ylim() if axis == 'y' else ax.get_xlim()
    if lo <= 0 or hi <= 0:
        return
    first = int(np.ceil(np.log10(lo) - 1e-10))
    last = int(np.floor(np.log10(hi) + 1e-10))
    spacing = max(1, int(np.ceil(max(0, last - first) / 4)))
    exponents = np.arange(int(np.ceil(first / spacing)) * spacing, last + 1, spacing)
    ticks = 10.0 ** exponents
    target = ax.yaxis if axis == 'y' else ax.xaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(LogFormatterMathtext(base=10))
    target.set_minor_locator(NullLocator())

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / 'figures'
RESULTS = ROOT / 'results'

@dataclass
class System:
    target: np.ndarray
    edges: np.ndarray
    weights: np.ndarray
    damping: np.ndarray
    anchors: np.ndarray

    def __post_init__(self):
        self.target = np.asarray(self.target, float)
        self.edges = np.asarray(self.edges, int)
        self.weights = np.asarray(self.weights, float)
        self.damping = np.asarray(self.damping, float)
        self.anchors = np.asarray(self.anchors, float)
        self.n = len(self.target)
        self.m = len(self.edges)
        self.B = np.zeros((self.m, self.n))
        self.B[np.arange(self.m), self.edges[:,0]] = 1.
        self.B[np.arange(self.m), self.edges[:,1]] = -1.
        target_differences = self.B @ self.target
        self.gstar = target_differences / norm(target_differences, axis=1)[:, None]

    def potential(self, x, eps=None):
        delta = self.B @ x
        lengths = norm(delta,axis=1)
        if eps is None:
            phi = lengths
        else:
            phi = np.where(lengths < eps,
                           (lengths**2 + eps**2) / (2*eps), lengths)
        return float(np.dot(self.weights, phi - np.sum(self.gstar*delta, axis=1))
                     + .5 * np.sum(self.anchors[:,None]*(x-self.target)**2))

    def energy(self,x,v,eps=None):
        return .5*float(np.sum(v**2)) + self.potential(x,eps)

    def error(self,x,v):
        return float(np.sum((x-self.target)**2) + np.sum(v**2))

    def grad(self, x, eps):
        delta = self.B @ x
        lengths = norm(delta,axis=1)
        q = delta/np.maximum(lengths[:,None],eps)
        return self.B.T @ (self.weights[:,None] * (q-self.gstar)) + self.anchors[:,None]*(x-self.target)

    def matrices(self):
        rows = []
        for (i,j),w,g,ell in zip(self.edges,self.weights,self.gstar,
                                  norm(self.B @ self.target,axis=1)):
            P=np.eye(3)-np.outer(g,g)
            R=np.zeros((3,3*self.n))
            R[:,3*i:3*i+3]=P/ell*np.sqrt(2*w)
            R[:,3*j:3*j+3]=-P/ell*np.sqrt(2*w)
            rows.append(R)
        for i,b in enumerate(self.anchors):
            if b>0:
                R=np.zeros((3,3*self.n));R[:,3*i:3*i+3]=np.eye(3)
                rows.append(R)
        M=np.concatenate(rows,axis=0)
        K=np.zeros((3*self.n,3*self.n))
        for (i,j),w,g,ell in zip(self.edges,self.weights,self.gstar,
                                  norm(self.B @ self.target,axis=1)):
            block = w/ell*(np.eye(3)-np.outer(g,g))
            u=slice(3*i,3*i+3);v=slice(3*j,3*j+3)
            K[u,u]+=block;K[v,v]+=block;K[u,v]-=block;K[v,u]-=block
        for i,b in enumerate(self.anchors):
            K[3*i:3*i+3,3*i:3*i+3]+=b*np.eye(3)
        return M,K

def make_eight():
    target=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],
                     [-1,-1,2],[1,-1,2],[1,1,2],[-1,1,2]],float)
    edges=np.array([(1,2),(1,4),(1,5),(2,3),(2,6),(2,8),(3,4),
                    (3,7),(4,8),(5,6),(5,8),(6,7),(7,8)],int)-1
    return System(target,edges,np.ones(13),np.ones(8),np.array([2,2,2,0,0,0,0,0]))

def make_four(restored=False):
    target=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],float)
    edges=[(1,2),(1,3),(3,4)]
    if restored: edges.append((2,4))
    return System(target,np.asarray(edges,int)-1,np.ones(len(edges)),
                  np.ones(4),np.array([2,2,2,0]))

def initial_eight(kind):
    rng=np.random.default_rng(2026)
    x=rng.uniform(-5,5,(8,3));v=rng.uniform(-1,1,(8,3))
    if kind == 'collision_equal':
        x[2]=[-1.65,-1.45,1.35]; x[3]=x[2]
        v[2]=[.35,.05,-.1];v[3]=v[2]
    elif kind == 'collision_transversal':
        x[2]=[-1.65,-1.45,1.35];x[3]=x[2]
        v[2]=[.35,.05,-.1];v[3]=-v[2]
    elif kind != 'nominal':
        raise ValueError(kind)
    return x,v

def integrate_smooth(s,x0,v0,eps=1e-6,T=400.0,step_out=0.5,
                     rtol=1e-9,atol=1e-11,max_step=0.2):
    def rhs(_t,y):
        x=y[:3*s.n].reshape(s.n,3);v=y[3*s.n:6*s.n].reshape(s.n,3)
        p=s.grad(x,eps)
        return np.r_[v.flatten(),(-s.damping[:,None]*v-p).flatten(),
                     np.sum(s.damping[:,None]*v*v)]
    output=np.arange(0,T+0.5*step_out,step_out)
    y0=np.r_[x0.flatten(),v0.flatten(),0.]
    sol=solve_ivp(rhs,(0,float(T)),y0,t_eval=output,method='DOP853',
                  rtol=rtol,atol=atol,max_step=max_step)
    if not sol.success: raise RuntimeError(sol.message)
    x=sol.y[:3*s.n,:].T.reshape(-1,s.n,3)
    v=sol.y[3*s.n:6*s.n,:].T.reshape(-1,s.n,3)
    dissipation=sol.y[-1]
    eta=np.sum((x-s.target)**2,axis=(1,2))+np.sum(v**2,axis=(1,2))
    

    energy=np.array([s.energy(a,b,eps) for a,b in zip(x,v)])
    residual=energy-energy[0]+dissipation
    return dict(t=output,x=x,v=v,eta=eta,energy=energy,
                dissipation=dissipation,residual=residual)

class ImplicitStepper:
    def __init__(self,s,dt,rho=None,tol=2e-9,max_admm=4000):
        self.s=s;self.h=dt
        self.rho=rho if rho is not None else 4./dt 

        self.tol=tol;self.max_admm=max_admm
        self.q=1./dt**2+s.damping/dt+s.anchors
        self.cho=cho_factor(np.diag(self.q)+self.rho*s.B.T@s.B)
        self.z=None;self.u=None
        self.solved_iterations=[]
        self.max_primal=0.;self.max_dual=0.

    def step(self,x0,v0):
        s=self.s;h=self.h;rho=self.rho
        f=((x0+h*v0)/h**2 +s.damping[:,None]/h*x0
           +s.anchors[:,None]*s.target
           +s.B.T@(s.weights[:,None]*s.gstar))
        z=self.z.copy() if self.z is not None else s.B@x0
        u=self.u.copy() if self.u is not None else np.zeros((s.m,3))
        for iteration in range(1,self.max_admm+1):
            x=cho_solve(self.cho,f+rho*s.B.T@(z-u))
            bx=s.B@x
            oldz=z
            zarg=bx+u
            mag=norm(zarg,axis=1)
            fac=np.maximum(0.,1.-s.weights/(rho*np.maximum(mag,1e-300)))
            z=zarg*fac[:,None]
            u+=bx-z
            rp=norm(bx-z);rd=rho*norm(s.B.T@(z-oldz))
            eps_p=self.tol*np.sqrt(3*s.m)+self.tol*max(norm(bx),norm(z))
            eps_d=self.tol*np.sqrt(3*s.n)+self.tol*norm(rho*s.B.T@u)
            if rp<=eps_p and rd<=eps_d:
                break
        else:
            raise RuntimeError(f'ADMM failed to converge: h={h}, rp={rp}, rd={rd}, rtol={eps_p}, dualtol={eps_d}')
        self.z=z;self.u=u
        self.solved_iterations.append(iteration)
        self.max_primal=max(self.max_primal,rp)
        self.max_dual=max(self.max_dual,rd)
        return x,(x-x0)/h

def integrate_implicit(s,x0,v0,T=400,dt=0.2,out_interval=0.5,tol=2e-9):
    steps=round(T/dt)
    if abs(steps*dt-T)>1e-10: raise ValueError('T/dt should be integer')
    sample_steps=max(1,round(out_interval/dt))
    solver=ImplicitStepper(s,dt,tol=tol)
    x=x0.copy();v=v0.copy()
    ts=[0.];xs=[x.copy()];vs=[v.copy()];energies=[s.energy(x,v)]
    max_energy_increase=0.
    for k in range(1,steps+1):
        xn,vn=solver.step(x,v)
        old_en=s.energy(x,v);new_en=s.energy(xn,vn)
        max_energy_increase=max(max_energy_increase,new_en-old_en)
        x,v=xn,vn
        if k%sample_steps==0 or k==steps:
            ts.append(k*dt);xs.append(x.copy());vs.append(v.copy());energies.append(new_en)
    x=np.stack(xs);v=np.stack(vs)
    eta=np.sum((x-s.target)**2,axis=(1,2))+np.sum(v**2,axis=(1,2))
    return dict(t=np.asarray(ts),x=x,v=v,eta=eta,energy=np.asarray(energies),
                iterations=np.asarray(solver.solved_iterations),
                max_energy_increase=max_energy_increase,
                max_primal_residual=solver.max_primal,max_dual_residual=solver.max_dual)

def fig_convergence(data):
    fig,ax=plt.subplots(figsize=(3.42,2.17))
    curves=[('nominal','Noncollision','-'),
            ('collision_equal','Equal velocity (ODE)','--'),
            ('collision_transversal','Transversal (ODE)',':'),
            ('collision_implicit','Equal velocity (implicit)','-.')]
    colors={}
    for key,label,style in curves:
        d=data[key]
        line,=ax.semilogy(d['t'],np.maximum(d['eta'],1e-15),style,lw=1.2,label=label)
        colors[key]=line.get_color()
    ax.set_xlabel('Time')
    ax.set_ylabel(r'Squared state error $\eta(t)$')
    ax.set_ylim(1e-13,1e3)
    set_time_ticks(ax, data['nominal']['t'][-1])
    set_sparse_log_ticks(ax)
    ax.grid(alpha=.23)
    ax.legend(frameon=False,loc='upper right')
    inset=ax.inset_axes([.13,.14,.32,.24])
    for key,label,style in [('collision_equal','Equal velocity','--'),
                            ('collision_transversal','Transversal',':')]:
        d=data[key+'_early']
        sep=norm(d['x'][:,2]-d['x'][:,3],axis=1)
        inset.loglog(d['t'][1:],sep[1:],style,lw=1.0,color=colors[key])
    inset.set_xlim(.005,.5)
    set_sparse_log_ticks(inset, 'x')
    set_sparse_log_ticks(inset)
    inset.set_xlabel('Time',fontsize=6,labelpad=0)
    inset.set_ylabel('Separation',fontsize=6,labelpad=0)
    inset.tick_params(labelsize=5,length=2,pad=1)
    inset.grid(alpha=.17)
    fig.tight_layout(pad=.35)
    fig.savefig(FIGURES/'convergence.pdf',bbox_inches='tight')
    plt.close(fig)

def fig_rank(data):
    fig,ax=plt.subplots(figsize=(3.42,2.10))
    for k,label,style in [('rank_deficient','Rank deficient (11/12)','--'),
                          ('rank_restored','Full rank (12/12)','-')]:
        d=data[k]
        ax.semilogy(d['t'],np.maximum(d['eta'],1e-13),style,lw=1.4,label=label)
    ax.set_xlabel('Time');ax.set_ylabel(r'Squared state error $\eta(t)$')
    ax.grid(alpha=.24);ax.set_ylim(1e-13,1e1)
    set_time_ticks(ax, data['rank_deficient']['t'][-1])
    set_sparse_log_ticks(ax)
    ax.legend(frameon=False,loc='lower left')
    fig.tight_layout(pad=.35)
    fig.savefig(FIGURES/'rank_comparison.pdf',bbox_inches='tight')
    plt.close(fig)

def fig_energy(data):
    fig,ax=plt.subplots(figsize=(3.42,2.10))
    for k,label in [('collision_equal','Collision (regularized ODE)'),
                    ('nominal','Noncollision (regularized ODE)')]:
        d=data[k]
        ax.semilogy(d['t'],np.maximum(np.abs(d['residual'])/d['energy'][0],1e-17),
                    lw=1.2,label=label)
    ax.grid(alpha=.24);ax.set_xlabel('Time')
    ax.set_ylabel(r'Normalized energy-balance residual')
    set_time_ticks(ax, data['nominal']['t'][-1])
    set_sparse_log_ticks(ax)
    ax.legend(frameon=False)
    fig.tight_layout(pad=.35)
    fig.savefig(FIGURES/'supplementary_energy_balance.pdf',bbox_inches='tight')
    plt.close(fig)

def fig_bounded_set(data):
    fig,ax=plt.subplots(figsize=(3.42,2.25))
    for r,d in data.items():
        ax.semilogy(d['t'],np.maximum(d['eta']/d['eta'][0],1e-16),
                    lw=1.3,label=fr'$r={r:g}$')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'Normalized squared error $\eta(t)/\eta(0)$')
    ax.set_ylim(1e-13,1e1)
    set_time_ticks(ax, next(iter(data.values()))['t'][-1])
    set_sparse_log_ticks(ax)
    ax.grid(alpha=.23)
    ax.legend(frameon=False,ncol=2,loc='upper right')
    fig.tight_layout(pad=.35)
    fig.savefig(FIGURES/'supplementary_initial_scales.pdf',bbox_inches='tight')
    plt.close(fig)


def local_spectral_decay(s):
    kappa=float(np.linalg.eigvalsh(s.matrices()[1])[0])
    d=float(s.damping[0])
    rate=d if d*d<=4*kappa else d-np.sqrt(d*d-4*kappa)
    return kappa,float(rate)


def gain_sensitivity_experiment(T=200.):
    x0,v0=initial_eight('nominal')
    damping={}
    anchors={}
    rates={'damping':{},'anchors':{}}
    for d in (0.2,0.35,1.0,2.0):
        s=make_eight()
        s.damping[:]=d
        damping[d]=integrate_smooth(s,x0,v0,T=T,step_out=.5)
        kappa,rate=local_spectral_decay(s)
        rates['damping'][str(d)]={'kappa_min':kappa,'squared_error_spectral_exponent':rate}
    for b in (0.2,1.0,2.0,8.0):
        s=make_eight()
        s.anchors[:3]=b
        anchors[b]=integrate_smooth(s,x0,v0,T=T,step_out=.5)
        kappa,rate=local_spectral_decay(s)
        rates['anchors'][str(b)]={'kappa_min':kappa,'squared_error_spectral_exponent':rate}
    return dict(damping=damping,anchors=anchors,rates=rates)


def fig_gain_sensitivity(data):
    fig,axs=plt.subplots(1,2,figsize=(3.44,2.22),sharex=True,sharey=True)
    for ax,key,parameter in [(axs[0],'damping','d'),(axs[1],'anchors','b')]:
        for val,run in data[key].items():
            ax.semilogy(run['t'],np.maximum(run['eta']/run['eta'][0],1e-17),
                        lw=1.15,label=fr'${parameter}={val:g}$')
        ax.set_xlim(0,200)
        ax.set_ylim(1e-15,1e3)
        ax.set_xticks([0,100,200])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Time')
        ax.grid(alpha=.22)
        ax.legend(frameon=False,loc='upper right')
    axs[0].set_ylabel(r'$\eta(t)/\eta(0)$')
    set_sparse_log_ticks(axs[0])
    axs[0].text(.04,.06,'(a)',transform=axs[0].transAxes,fontsize=8,fontweight='bold')
    axs[1].text(.04,.06,'(b)',transform=axs[1].transAxes,fontsize=8,fontweight='bold')
    fig.tight_layout(pad=.45,w_pad=.5)
    fig.savefig(FIGURES/'gain_sensitivity.pdf',bbox_inches='tight')
    plt.close(fig)

def fig_refinement(refinement):
    fig,axs=plt.subplots(1,2,figsize=(3.42,2.25))
    lhs=refinement['implicit']
    rhs=refinement['regularization']
    axs[0].plot([a for a,_ in lhs],[b for _,b in lhs],'o-',lw=1.2,ms=3.5)
    axs[1].loglog([a for a,_ in rhs],[b for _,b in rhs],'s-',lw=1.2,ms=3.5)
    axs[0].set_xlabel(r'Step $h$')
    axs[1].set_xlabel(r'Regularization $\varepsilon$')
    axs[0].set_ylabel(r'$\Delta_h$')
    axs[1].set_ylabel(r'$\Delta_\varepsilon$')
    axs[0].set_xticks([.025,.1,.2])
    axs[0].set_xticklabels(['.025','.10','.20'])
    axs[0].set_xlim(.01,.22)
    axs[0].set_ylim(0,1.05)
    axs[0].set_yticks([0,0.5,1.0])
    axs[0].minorticks_off()
    axs[1].set_xticks([1e-6,1e-4,1e-3])
    axs[1].set_xticklabels([r'$10^{-6}$',r'$10^{-4}$',r'$10^{-3}$'])
    axs[1].minorticks_off()
    axs[1].set_ylim(1e-4, 1e-2)
    set_sparse_log_ticks(axs[1], 'x')
    set_sparse_log_ticks(axs[1])
    for i, ax in enumerate(axs):
        ax.grid(alpha=.23,which='major')
        ax.text(0.055,0.95,('(a)','(b)')[i],transform=ax.transAxes,
                ha='left',va='top',fontsize=8,fontweight='bold')
    fig.tight_layout(pad=.35,w_pad=.8)
    fig.savefig(FIGURES/'collision_refinement.pdf',bbox_inches='tight')
    plt.close(fig)

def bounded_set_experiment(s):
    x_base,v_base=initial_eight('nominal')
    runs={}
    slopes={}
    for r in (0.5,1.,2.,4.):
        x0=s.target+r*(x_base-s.target)
        v0=r*v_base
        d=integrate_smooth(s,x0,v0,T=400.,step_out=.5)
        mask=(d['t']>=250.) & (d['t']<=350.)
        slopes[str(r)]=float(np.polyfit(d['t'][mask],np.log(d['eta'][mask]),1)[0])
        runs[r]=d
    return runs,slopes

def collision_refinement_experiment(s):
    x0,v0=initial_eight('collision_equal')
    horizon=20.
    h_values=(0.2,0.1,0.05,0.025,0.0125)
    discrete={h:integrate_implicit(s,x0,v0,T=horizon,dt=h,out_interval=h)
              for h in h_values}
    implicit=[]
    for h in h_values[:-1]:
        coarse=discrete[h]
        finer=discrete[h/2]
        fine_times=np.rint(coarse['t']/(h/2)).astype(int)
        assert np.allclose(coarse['t'],finer['t'][fine_times])
        differences=(coarse['x']-finer['x'][fine_times]).reshape(len(coarse['t']),-1)
        implicit.append((h,float(np.max(norm(differences,axis=1)))))
    eps_values=(1e-3,1e-4,1e-5,1e-6,1e-7)
    smooth={eps:integrate_smooth(s,x0,v0,eps=eps,T=horizon,step_out=.025)
            for eps in eps_values}
    reference=smooth[1e-7]['x']
    regularization=[]
    for eps in eps_values[:-1]:
        differences=(smooth[eps]['x']-reference).reshape(len(reference),-1)
        regularization.append((eps,float(np.max(norm(differences,axis=1)))))
    return dict(implicit=implicit,regularization=regularization)

def fig_collision_energy(data):
    fig,ax=plt.subplots(figsize=(3.42,2.06))
    for key,label,style in [('collision_implicit','Implicit: original energy','-'),
                            ('collision_equal','Regularized ODE','--')]:
        d=data[key]
        ax.semilogy(d['t'],np.maximum(d['energy'],1e-14),style,
                    lw=1.3,label=label)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total energy')
    set_time_ticks(ax, data['collision_equal']['t'][-1])
    set_sparse_log_ticks(ax)
    ax.grid(alpha=.23)
    ax.legend(frameon=False)
    fig.tight_layout(pad=.35)
    fig.savefig(FIGURES/'collision_energy.pdf',bbox_inches='tight')
    plt.close(fig)

def sample_and_run(args):
    FIGURES.mkdir(exist_ok=True);RESULTS.mkdir(exist_ok=True)
    s=make_eight();x0,v0=initial_eight('nominal');xc,vc=initial_eight('collision_equal')
    M,K=s.matrices();spec=np.linalg.eigvalsh(K)
    A=np.block([[np.zeros((24,24)),np.eye(24)],[-K,-np.diag(np.repeat(s.damping,3))]])
    abscissa=float(np.max(np.linalg.eigvals(A).real))
    print('Running eight-agent noncollision ODE',flush=True)
    nominal=integrate_smooth(s,x0,v0,T=args.horizon)
    print('Running equal-velocity collision ODE',flush=True)
    collision=integrate_smooth(s,xc,vc,T=args.horizon)
    print('Running transversal collision ODE',flush=True)
    xt,vt=initial_eight('collision_transversal')
    transversal=integrate_smooth(s,xt,vt,T=args.horizon)
    collision_equal_early=integrate_smooth(s,xc,vc,T=.5,step_out=.005)
    collision_transversal_early=integrate_smooth(s,xt,vt,T=.5,step_out=.005)
    print('Running original subgradient implicit discretization',flush=True)
    implicit=integrate_implicit(s,xc,vc,T=args.horizon,dt=args.implicit_step,
                                 out_interval=args.implicit_step)
    s_bad=make_four(restored=False);s_good=make_four(restored=True)
    xr=s_bad.target.copy();xr[3]=s_bad.target[2]+1.6*(s_bad.target[3]-s_bad.target[2])
    vr=np.zeros_like(xr)
    print('Running rank-deficient/full-rank comparison',flush=True)
    bad=integrate_smooth(s_bad,xr,vr,T=150,step_out=0.5)
    good=integrate_smooth(s_good,xr,vr,T=150,step_out=0.5)
    data=dict(nominal=nominal,collision_equal=collision,
              collision_transversal=transversal,
              collision_equal_early=collision_equal_early,
              collision_transversal_early=collision_transversal_early,
              collision_implicit=implicit,rank_deficient=bad,rank_restored=good)
    fig_convergence(data);fig_rank(data);fig_energy(data)
    fig_collision_energy(data)
    print('Running gain sensitivity comparison',flush=True)
    gains=gain_sensitivity_experiment(T=200.)
    fig_gain_sensitivity(gains)
    print('Running collision refinement diagnostics',flush=True)
    refinement=collision_refinement_experiment(s)
    fig_refinement(refinement)
    

    np.savez_compressed(RESULTS/'trajectories.npz',
        **{f'{k}_{part}':d[part] for k,d in data.items() for part in ('t','x','v','eta','energy')},
        **{f'gain_{kind}_{val:g}_{part}':d[part]
           for kind in ('damping','anchors') for val,d in gains[kind].items()
           for part in ('t','x','v','eta','energy')})
    singular_bad=np.linalg.svd(s_bad.matrices()[0],compute_uv=False)
    singular_good=np.linalg.svd(s_good.matrices()[0],compute_uv=False)
    sv=np.linalg.svd(M,compute_uv=False)
    mask=(nominal['t']>=250)&(nominal['t']<=args.horizon)
    slope=float(np.polyfit(nominal['t'][mask],np.log(nominal['eta'][mask]),1)[0]) if mask.sum()>3 else None
    metrics={
       'numpy_version':np.__version__,
       'random_seed':2026,
       'eight_rank':int(np.linalg.matrix_rank(M,tol=1e-10)),
       'eight_sigma_min':float(sv[-1]),
       'eight_hessian_lambda_min':float(spec[0]),
       'linearization_spectral_abscissa':abscissa,
       'twice_spectral_abscissa':2*abscissa,
       'nominal_late_log_eta_slope':slope,
       'nominal_final_eta':float(nominal['eta'][-1]),
       'collision_regularized_final_eta':float(collision['eta'][-1]),
       'collision_transversal_final_eta':float(transversal['eta'][-1]),
       'collision_equal_separation_at_0_5':float(norm(collision_equal_early['x'][-1,2]-collision_equal_early['x'][-1,3])),
       'collision_transversal_separation_at_0_5':float(norm(collision_transversal_early['x'][-1,2]-collision_transversal_early['x'][-1,3])),
       'collision_implicit_final_eta':float(implicit['eta'][-1]),
       'collision_regularized_max_normalized_energy_residual':float(np.max(np.abs(collision['residual']))/collision['energy'][0]),
       'nominal_regularized_max_normalized_energy_residual':float(np.max(np.abs(nominal['residual']))/nominal['energy'][0]),
       'collision_implicit_max_positive_energy_increment':float(implicit['max_energy_increase']),
       'collision_implicit_max_discrete_dissipation_inequality_lhs':float(np.max(
          np.diff(implicit['energy']) + args.implicit_step *
          np.sum(s.damping[None,:,None]*implicit['v'][1:]**2,axis=(1,2)))),
       'collision_implicit_average_ADMM_iterations':float(np.mean(implicit['iterations'])),
       'collision_implicit_max_ADMM_iterations':int(np.max(implicit['iterations'])),
       'collision_implicit_max_ADMM_primal_residual':float(implicit['max_primal_residual']),
       'collision_implicit_max_ADMM_dual_residual':float(implicit['max_dual_residual']),
       'rank_deficient_rank':int(np.linalg.matrix_rank(s_bad.matrices()[0],tol=1e-10)),
       'rank_restored_rank':int(np.linalg.matrix_rank(s_good.matrices()[0],tol=1e-10)),
       'rank_deficient_final_eta':float(bad['eta'][-1]),
       'rank_restored_final_eta':float(good['eta'][-1]),
       'rank_deficient_sigma_smallest':float(singular_bad[-1]),
       'rank_restored_sigma_smallest':float(singular_good[-1]),
       'implicit_dt':args.implicit_step,
       'horizon':args.horizon,
       'regularization_epsilon':1e-6,
       'gain_sensitivity_rates':gains['rates'],
       'gain_sensitivity_initial_seed':2026,
       'd_opt_b2':float(2*np.sqrt(np.linalg.eigvalsh(K)[0])),
       'collision_refinement_implicit_h_vs_h_over_2_max_position_error_0_20':{str(h):delta for h,delta in refinement['implicit']},
       'collision_refinement_regularized_eps_vs_1e-7_max_position_error_0_20':{str(eps):delta for eps,delta in refinement['regularization']},
    }
    if args.supplementary:
        print('Running supplementary solver and regularization checks',flush=True)
        bounded,slopes=bounded_set_experiment(s)
        fig_bounded_set(bounded)
        metrics['supplementary_bounded_set_slopes_250_350']=slopes
        s_free=make_eight()
        s_free.anchors[:]=0.
        free=integrate_smooth(s_free,x0,v0,T=args.horizon)
        metrics['supplementary_anchor_free_final_eta']=float(free['eta'][-1])
        metrics['supplementary_transversal_final_eta']=float(transversal['eta'][-1])
        

        fig,ax=plt.subplots(figsize=(3.42,2.10))
        ax.semilogy(nominal['t'],nominal['eta'],label='Anchored')
        ax.semilogy(free['t'],free['eta'],label='Anchor-free')
        ax.set_xlabel('Time');ax.set_ylabel(r'Squared state error $\eta(t)$')
        set_time_ticks(ax, nominal['t'][-1]);set_sparse_log_ticks(ax)
        ax.grid(alpha=.24);ax.legend(frameon=False)
        fig.tight_layout(pad=.35)
        fig.savefig(FIGURES/'supplementary_anchor_free.pdf',bbox_inches='tight');plt.close(fig)
        fig,ax=plt.subplots(figsize=(3.42,2.10))
        for label,d in [('Transversal',transversal),('Equal velocity',collision)]:
            separation=norm(d['x'][:,2]-d['x'][:,3],axis=1)
            ax.loglog(d['t'][1:],separation[1:],label=label)
        ax.set_xlabel('Time');ax.set_ylabel('Edge (3,4) separation')
        set_sparse_log_ticks(ax, 'x');set_sparse_log_ticks(ax)
        ax.grid(alpha=.24);ax.legend(frameon=False)
        fig.tight_layout(pad=.35)
        fig.savefig(FIGURES/'supplementary_collision_separation.pdf',bbox_inches='tight');plt.close(fig)
        sensitivity={}
        for eps in (1e-4,1e-5,1e-6,1e-7):
            d=integrate_smooth(s,xc,vc,eps=eps,T=40,step_out=.1)
            sensitivity[eps]=d
        sensitivity_metrics={}
        ref=sensitivity[1e-7]
        for eps,d in sensitivity.items():
            sensitivity_metrics[str(eps)] = float(np.max(np.abs(d['x']-ref['x'])))
        metrics['supplementary_eps_trajectory_max_position_differences_vs_1e-7']=sensitivity_metrics
        fig,ax=plt.subplots(figsize=(3.42,2.10))
        for eps,d in sensitivity.items():
            sep=norm(d['x'][:,2]-d['x'][:,3],axis=1)
            ax.loglog(d['t'][1:],sep[1:],lw=1.2,label=fr'$\varepsilon={eps:.0e}$')
        ax.legend(frameon=False);ax.grid(alpha=.24)
        ax.set_xlabel('Time');ax.set_ylabel('Edge (3,4) separation')
        set_sparse_log_ticks(ax, 'x');set_sparse_log_ticks(ax)
        fig.tight_layout(pad=.35)
        fig.savefig(FIGURES/'supplementary_regularization.pdf',bbox_inches='tight');plt.close(fig)
        dt_small=max(.025,args.implicit_step/2)
        implicit_ref=integrate_implicit(s,xc,vc,T=20,dt=dt_small,out_interval=dt_small)
        coarse=integrate_implicit(s,xc,vc,T=20,dt=args.implicit_step,out_interval=args.implicit_step)
        from scipy.interpolate import interp1d
        xf=interp1d(implicit_ref['t'],implicit_ref['x'],axis=0)
        errors=norm((coarse['x']-xf(coarse['t'])).reshape(len(coarse['t']),-1),axis=1)
        metrics['supplementary_implicit_step_refinement_max_position_difference_0_20']=float(np.max(errors))
        fig,ax=plt.subplots(figsize=(3.42,2.10))
        ax.semilogy(coarse['t'],np.maximum(errors,1e-15))
        ax.grid(alpha=.24);ax.set_xlabel('Time')
        ax.set_ylabel('Implicit $h$ versus $h/2$ position difference')
        set_time_ticks(ax, coarse['t'][-1]);set_sparse_log_ticks(ax)
        fig.tight_layout(pad=.35)
        fig.savefig(FIGURES/'supplementary_implicit_steps.pdf',bbox_inches='tight');plt.close(fig)
    with open(RESULTS/'metrics.json','w') as out:
        json.dump(metrics,out,indent=2)
    print(json.dumps(metrics,indent=2),flush=True)

def cli():
    p=argparse.ArgumentParser(description='Anchored bearing-based stabilization simulations')
    p.add_argument('--horizon',type=float,default=400.0)
    p.add_argument('--implicit-step',type=float,default=.05)
    p.add_argument('--supplementary',action='store_true')
    p.add_argument('--gain-only',action='store_true')
    return p.parse_args()

if __name__=='__main__':
    args=cli()
    if args.gain_only:
        FIGURES.mkdir(exist_ok=True)
        RESULTS.mkdir(exist_ok=True)
        gain_data=gain_sensitivity_experiment(T=200.)
        fig_gain_sensitivity(gain_data)
        with open(RESULTS/'gain_sensitivity.json','w') as out:
            json.dump(gain_data['rates'],out,indent=2)
    else:
        sample_and_run(args)
