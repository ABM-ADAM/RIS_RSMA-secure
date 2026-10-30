"""Reproducible, conservative RIS-RSMA experiment for Sections II-III.

Numerical signal normalization: BS transmit power is one and all receiver
noise variances are one. A channel is scaled by sqrt(P_PRB/sigma2_PRB).
Channel path loss uses the 3GPP TR 38.901 UMi street-canyon equations for
BS/user and RIS/terminal links (the latter is an explicit adaptation).
The clear BS/RIS link uses ITU-R P.525 free-space loss. Small-scale fading,
RIS aperture efficiency and independent placement are documented assumptions,
not a full spatially consistent 3GPP channel implementation.

Author: Abuzar B. M. Adam
October 2026
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import time

import casadi as ca
import numpy as np


@dataclass(frozen=True)
class Config:
    fc_ghz: float = 3.5
    carrier_bandwidth_hz: float = 20e6
    prb_bandwidth_hz: float = 360e3
    carrier_power_dbm: float = 30.0
    noise_figure_db: float = 7.0
    n: int = 8
    k: int = 4
    e: int = 2
    m: int = 64
    b: int = 2
    ris_efficiency: float = 0.8
    public_floor: float = 0.1
    secret_floor: float = 0.0
    lambda_public: float = 0.2
    epsilon_relative: float = 0.01
    scenario: str = 'umi'
    bs: tuple = (0.0, 0.0, 10.0)
    ris: tuple = (40.0, 0.0, 10.0)
    # BS/RIS antenna gains are not included as separate gains; the BS array
    # steering and RIS elements are modeled explicitly in the channel.

    @property
    def prb_power_watt(self):
        return 10 ** ((self.carrier_power_dbm - 30) / 10) * (
            self.prb_bandwidth_hz / self.carrier_bandwidth_hz
        )

    @property
    def noise_watt(self):
        return 10 ** ((-174 + 10 * np.log10(self.prb_bandwidth_hz)
                       + self.noise_figure_db - 30) / 10)


def _db_loss_umi(d2, d3, fc, los):
    """TR 38.901 UMi LOS/NLOS; h_BS=10 m, h_UT=1.5 m.

    For these geometries d2 < d_BP = 4*(10-1)*(1.5-1)*fc/c.
    """
    bp = 4 * 9 * 0.5 * (fc * 1e9) / 299792458
    if d2 > bp:
        los_loss = (32.4 + 40 * np.log10(d3)
                    - 9.5 * np.log10(bp * bp + 8.5 * 8.5)
                    + 20 * np.log10(fc))
    else:
        los_loss = 32.4 + 21 * np.log10(d3) + 20 * np.log10(fc)
    nlos_loss = max(los_loss, 35.3 * np.log10(d3) + 22.4
                    + 21.3 * np.log10(fc))
    return los_loss if los else nlos_loss


def _los_probability(d2):
    if d2 <= 18: return 1.0
    return 18 / d2 + (1 - 18 / d2) * np.exp(-d2 / 36)


def _array(n, azimuth):
    return np.exp(1j * np.pi * np.arange(n) * np.sin(azimuth))


def _ris_steering(m, displacement):
    """Half-wavelength square planar surface in the y-z plane."""
    side=int(np.sqrt(m))
    if side*side!=m:raise ValueError('square RIS needed')
    direction=np.asarray(displacement)/np.linalg.norm(displacement)
    yy,zz=np.meshgrid(np.arange(side),np.arange(side),indexing='ij')
    return np.exp(1j*np.pi*(yy*direction[1]+zz*direction[2])).reshape(m)


def _rician(rng, shape, los_vector, factor):
    diffuse = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
    return (np.sqrt(factor / (1 + factor)) * los_vector
            + np.sqrt(1 / (1 + factor)) * diffuse)


@dataclass
class Draw:
    hd: np.ndarray       # (K,N) row channels, with power/noise normalization
    bu: np.ndarray       # (K,M,N) cascaded legitimate channels
    he: np.ndarray       # (E,N) estimated direct Eve channels
    be: np.ndarray       # (E,M,N) estimated cascaded Eve channels
    eps: np.ndarray      # enclosing effective error-ball radii for any phase
    user_xy: np.ndarray
    eve_xy: np.ndarray
    los_flags: np.ndarray

    def effective(self, theta, use_ris=True):
        if use_ris:
            return self.hd + np.einsum('m,kmn->kn',theta,self.bu), self.he + np.einsum('m,emn->en',theta,self.be)
        return self.hd.copy(),self.he.copy()


def channel_draw(seed: int, cfg: Config) -> Draw:
    rng = np.random.default_rng(seed)
    if cfg.m != int(np.sqrt(cfg.m)) ** 2:
        raise ValueError('M must be a square planar RIS')
    bs = np.asarray(cfg.bs); ris = np.asarray(cfg.ris)
    if cfg.scenario in ('blocked','unconditioned_corridor'):
        # An explicitly conditional reflective corridor: a street-corner
        # obstruction blocks direct user links, whereas the elevated RIS
        # sees the user corridor. Eavesdroppers occupy the exposed side street.
        positions = np.concatenate([
            np.c_[rng.uniform(70,120,cfg.k), rng.uniform(-10,10,cfg.k), np.full(cfg.k,1.5)],
            np.c_[rng.uniform(60,130,cfg.e),
                  rng.choice((-1,1),cfg.e)*rng.uniform(22,35,cfg.e),np.full(cfg.e,1.5)]
        ])
    elif cfg.scenario == 'umi':
        positions = np.concatenate([
            np.c_[rng.uniform(70,120,cfg.k),rng.uniform(-30,30,cfg.k),np.full(cfg.k,1.5)],
            np.c_[rng.uniform(60,130,cfg.e),rng.uniform(-35,35,cfg.e),np.full(cfg.e,1.5)]
        ])
    else:raise ValueError('unknown scenario')
    norm = np.sqrt(cfg.prb_power_watt / cfg.noise_watt)
    dbr = np.linalg.norm(bs-ris)
    loss_br = 32.4 + 20 * np.log10(dbr) + 20 * np.log10(cfg.fc_ghz)
    az_br = np.arctan2(ris[1]-bs[1],ris[0]-bs[0])
    # Unit-magnitude steering elements; a planar RIS has M physical elements.
    a_bs = _array(cfg.n,az_br)
    a_ris = _ris_steering(cfg.m,bs-ris)
    G = np.sqrt(10**(-loss_br/10) * cfg.ris_efficiency) * _rician(
        rng,(cfg.m,cfg.n),np.outer(a_ris,np.conj(a_bs)),10.0)
    direct=[];cascade=[];los_flags=[]
    for idx,pos in enumerate(positions):
        d2=np.linalg.norm((pos-bs)[:2]);d3=np.linalg.norm(pos-bs)
        d2r=np.linalg.norm((pos-ris)[:2]);d3r=np.linalg.norm(pos-ris)
        if cfg.scenario=='blocked':
            direct_los,ris_los=(False,True) if idx<cfg.k else (True,False)
        else:
            direct_los=rng.random()<_los_probability(d2)
            ris_los=rng.random()<_los_probability(d2r)
        los_flags.append((direct_los,ris_los))
        # 3GPP UMi shadow-fading standard deviations (LOS/NLOS): 4/7.82 dB.
        # Independent links here; no spatially consistent map is asserted.
        ld=_db_loss_umi(d2,d3,cfg.fc_ghz,direct_los)+rng.normal(0,4.0 if direct_los else 7.82)
        lr=_db_loss_umi(d2r,d3r,cfg.fc_ghz,ris_los)+rng.normal(0,4.0 if ris_los else 7.82)
        az_d=np.arctan2(pos[1]-bs[1],pos[0]-bs[0])
        az_r=np.arctan2(pos[1]-ris[1],pos[0]-ris[0])
        h_d=np.sqrt(10**(-ld/10))*_rician(
            rng,(cfg.n,),_array(cfg.n,az_d),5.0 if direct_los else 0.0)
        h_r=np.sqrt(10**(-lr/10))*_rician(
            rng,(cfg.m,),_ris_steering(cfg.m,pos-ris),5.0 if ris_los else 0.0)
        direct.append(norm*np.conj(h_d))
        cascade.append(norm*np.conj(h_r)[:,None]*G)
    direct=np.stack(direct);cascade=np.stack(cascade)
    he=direct[cfg.k:];be=cascade[cfg.k:]
    eps=cfg.epsilon_relative*(np.linalg.norm(he,axis=1)
          +np.sqrt(cfg.m)*np.linalg.norm(be,axis=(1,2)))
    return Draw(direct[:cfg.k],cascade[:cfg.k],he,be,eps,
                positions[:cfg.k],positions[cfg.k:],np.asarray(los_flags))


def aperture_subset(draw: Draw, old_m: int, new_m: int, epsilon_relative: float):
    """Nested central square apertures for paired M sweeps."""
    side=int(np.sqrt(old_m));small=int(np.sqrt(new_m))
    if side*side!=old_m or small*small!=new_m or small>side:
        raise ValueError('use nested square apertures')
    start=(side-small)//2
    index=np.arange(old_m).reshape(side,side)[start:start+small,start:start+small].ravel()
    bu=draw.bu[:,index,:].copy();be=draw.be[:,index,:].copy()
    eps=epsilon_relative*(np.linalg.norm(draw.he,axis=1)
         +np.sqrt(new_m)*np.linalg.norm(be,axis=(1,2)))
    return Draw(draw.hd.copy(),bu,draw.he.copy(),be,eps,
                draw.user_xy.copy(),draw.eve_xy.copy(),draw.los_flags.copy())


def physical_rates(H, EH, eps, wc, W, C, t, q, public_floor=0.0, secret_floor=0.0):
    """Directly check the physical conservative constraints of Section III."""
    K,N=H.shape;E=EH.shape[0]
    power=float(np.vdot(wc,wc).real+np.vdot(W,W).real+N*q)
    hc=H@wc; hp=H@W.T
    pwr=np.abs(hp)**2
    thermal=1+q*np.sum(np.abs(H)**2,axis=1)
    rc=np.log2(1+np.abs(hc)**2/(np.sum(pwr,axis=1)+thermal))
    rp=np.log2(1+np.diag(pwr)/(np.sum(pwr,axis=1)-np.diag(pwr)+thermal))
    ae=np.maximum(0,np.linalg.norm(EH,axis=1)-eps)**2
    leak=np.empty((E,K))
    for e in range(E):
        for k in range(K):
            amp=np.abs(EH[e]@W[k])+eps[e]*np.linalg.norm(W[k])
            leak[e,k]=np.log2(1+amp**2/(1+q*ae[e]))
    sec=rp-np.max(leak,axis=0)
    residual=float(min(1-power,np.min(rc)-np.sum(C),
               np.min(sec-t),np.min(C-public_floor),
               np.min(t-secret_floor),q))
    return dict(common=rc,private=rp,leakage=leak,secret=sec,
                public=float(np.sum(C)),confidential=float(np.sum(t)),
                power=power,residual=residual)


def _normalize(v):
    n=np.linalg.norm(v)
    return v/n if n>1e-12 else v


def _seed_beams(H,EH,eps,q,P0,common=True,private_fraction=.35,
                suppress_user_interference=False):
    K,N=H.shape
    # Project Eve estimates while accounting for their nonzero uncertainty.
    null=np.eye(N)-np.conj(EH).T@np.linalg.pinv(EH@np.conj(EH).T)@EH
    W=[]
    for k,row in enumerate(H):
        if suppress_user_interference:
            # With N >= K+E, a private beam can null the other served users
            # as well as the nominal Eve channels. This deterministic start
            # prevents the high-SNR solver from relying on interference-limited
            # Eve-only nulling as its sole candidate.
            other=np.vstack((EH,H[np.arange(K)!=k]))
            projection=np.eye(N)-np.conj(other).T@np.linalg.pinv(other@np.conj(other).T)@other
        else:
            projection=null
        W.append(_normalize(projection@np.conj(row)))
    W=np.stack(W)
    ae=np.maximum(0,np.linalg.norm(EH,axis=1)-eps)**2
    for k in range(K):
        legitimate=abs(H[k]@W[k])**2/(1+q*np.linalg.norm(H[k])**2)
        leakage=max((abs(EH[e]@W[k])+eps[e])**2/(1+q*ae[e])
                    for e in range(len(EH)))
        if legitimate<=leakage*(1+1e-5):W[k]=0
    active=sum(np.linalg.norm(row)>0 for row in W)
    if active:W=W*np.sqrt(private_fraction*P0/active)
    if common:
        direction=sum(np.conj(row)/max(np.linalg.norm(row),1e-12) for row in H)
        wc=_normalize(direction)*np.sqrt((1-private_fraction)*P0)
    else:
        wc=np.zeros(N,complex)
        if active and private_fraction>0:W=W/np.sqrt(private_fraction)
    return wc,W


def _auxiliaries(H,EH,eps,wc,W,q):
    K=H.shape[0];E=EH.shape[0]
    Hp=H@W.T;Hc=H@wc
    tp=np.sum(abs(Hp)**2,axis=1)+q*np.sum(abs(H)**2,axis=1)+1
    tc=tp+abs(Hc)**2
    gp=np.diag(Hp)/tp;gc=Hc/tc
    up=1/np.maximum(1-abs(np.diag(Hp))**2/tp,1e-10)
    uc=1/np.maximum(1-abs(Hc)**2/tc,1e-10)
    ae=np.maximum(0,np.linalg.norm(EH,axis=1)-eps)**2
    d=1+q*ae
    beta0=np.array([[(abs(EH[e]@W[k])+eps[e]*np.linalg.norm(W[k]))**2/d[e]
        for k in range(K)] for e in range(E)])
    return gp,gc,up,uc,d,beta0


def _subproblem(H,EH,eps,q,P0,wc,W,C,t,cfg,phase_one=False,common=True,rho=0.02):
    """Smooth-conic CasADi/IPOPT realization of the Section III surrogate.

    Norms use sqrt(sum_squares+1e-16), a conservative 1e-8 amplitude
    regularization. Mathematical constraints are convex; IPOPT solves their
    smooth nonlinear representation and every candidate is checked directly.
    """
    K,N=H.shape;E=EH.shape[0]
    gp,gc,up,uc,d,b0=_auxiliaries(H,EH,eps,wc,W,q)
    opt=ca.Opti()
    wr=opt.variable(K+1,N);wi=opt.variable(K+1,N)
    cv=opt.variable(K);tv=opt.variable(K)
    beta=opt.variable(E,K)
    slack=opt.variable() if phase_one else 0
    opt.subject_to(opt.bounded(0,ca.sumsqr(wr)+ca.sumsqr(wi),P0))
    opt.subject_to(cv >= (cfg.public_floor if common else 0))
    if not common:opt.subject_to(cv==0)
    opt.subject_to(tv >= cfg.secret_floor)
    opt.subject_to(ca.vec(beta) >= 0)
    if phase_one:opt.subject_to(slack>=0)
    if not common:opt.subject_to(wr[0,:]==0);opt.subject_to(wi[0,:]==0)

    def projection(row,z):
        return (ca.dot(row.real,z[0])-ca.dot(row.imag,z[1]),
                ca.dot(row.real,z[1])+ca.dot(row.imag,z[0]))

    def squared(z):return z[0]**2+z[1]**2

    old=np.vstack((wc,W))
    opt.set_initial(wr,old.real);opt.set_initial(wi,old.imag)
    opt.set_initial(cv,C);opt.set_initial(tv,t)
    opt.set_initial(beta,np.maximum(b0,0))
    if phase_one:
        current=physical_rates(H,EH,eps,wc,W,C,t,q,
            cfg.public_floor if common else 0,cfg.secret_floor)
        opt.set_initial(slack,max(0.1,-current['residual']+0.1))
    for k in range(K):
        row=H[k]
        ps=[projection(row,(wr[j,:].T,wi[j,:].T)) for j in range(K+1)]
        totalp=sum(squared(ps[j]) for j in range(1,K+1))+q*np.linalg.norm(row)**2+1
        totalc=totalp+squared(ps[0])
        ep=abs(gp[k])**2*totalp-2*(gp[k].real*ps[k+1][0]+gp[k].imag*ps[k+1][1])+1
        ec=abs(gc[k])**2*totalc-2*(gc[k].real*ps[0][0]+gc[k].imag*ps[0][1])+1
        lowerp=(np.log(up[k])-up[k]*ep+1)/np.log(2)
        lowerc=(np.log(uc[k])-uc[k]*ec+1)/np.log(2)
        if common:opt.subject_to(ca.sum1(cv)<=lowerc+slack)
        for e in range(E):
            major=np.log2(1+b0[e,k])+(beta[e,k]-b0[e,k])/((1+b0[e,k])*np.log(2))
            proj=projection(EH[e],(wr[k+1,:].T,wi[k+1,:].T))
            amp=ca.sqrt(squared(proj)+1e-16)+eps[e]*ca.sqrt(
                ca.sumsqr(wr[k+1,:])+ca.sumsqr(wi[k+1,:])+1e-16)
            opt.subject_to(amp**2<=d[e]*beta[e,k])
            opt.subject_to(tv[k]+major<=lowerp+slack)
    if phase_one:
        prox=ca.sumsqr(wr-old.real)+ca.sumsqr(wi-old.imag)+ca.sumsqr(cv-C)+ca.sumsqr(tv-t)
        opt.minimize(slack+1e-4*prox)
    else:
        prox=ca.sumsqr(wr-old.real)+ca.sumsqr(wi-old.imag)+ca.sumsqr(cv-C)+ca.sumsqr(tv-t)
        opt.minimize(-(ca.sum1(tv)+cfg.lambda_public*ca.sum1(cv)-rho/2*prox))
    opts={'ipopt.print_level':0,'print_time':False,'ipopt.max_iter':80,
          'ipopt.tol':1e-7,'ipopt.acceptable_tol':1e-6,
          'ipopt.sb':'yes','error_on_fail':False}
    opt.solver('ipopt',opts)
    try:sol=opt.solve()
    except RuntimeError as exc:
        if os.environ.get('RSMA_DEBUG'):
            print('inner-solve failure:',str(exc)[:900],flush=True)
            print('solver statistics:',opt.stats().get('iter_count'),opt.stats().get('return_status'),flush=True)
            try:
                print('debug rates',opt.debug.value(tv),opt.debug.value(cv),flush=True)
            except RuntimeError:pass
        if opt.stats().get('return_status') not in ('Maximum_Iterations_Exceeded','Search_Direction_Becomes_Too_Small'):
            return None
        sol=opt.debug
    wnew=np.asarray(sol.value(wr))+1j*np.asarray(sol.value(wi))
    state=(wnew[0],wnew[1:],np.asarray(sol.value(cv)).reshape(K),
           np.asarray(sol.value(tv)).reshape(K))
    return state,float(sol.value(slack)) if phase_one else 0.0


def optimize_fixed(draw,cfg,theta,q_fraction=0.0,use_ris=True,common=True,
                   phase_budget=10,inner_budget=10,seed_state=None):
    """Phase I + Section III convex updates, retaining only verified iterates.

    q_fraction is the total AN fraction Nq/P_B (P_B normalized to one).
    Failed Phase I is recorded as 'not found', never called infeasible.
    """
    H,EH=draw.effective(theta,use_ris)
    eps=draw.eps if use_ris else cfg.epsilon_relative*np.linalg.norm(draw.he,axis=1)
    q=q_fraction/cfg.n;P0=1-q_fraction
    if P0<0:raise ValueError('invalid AN fraction')
    start=time.perf_counter();solves=0
    wc,W=_seed_beams(H,EH,eps,q,P0,common) if seed_state is None else seed_state[:2]
    C=np.full(cfg.k,cfg.public_floor) if common else np.zeros(cfg.k)
    t=np.full(cfg.k,cfg.secret_floor)
    initial=physical_rates(H,EH,eps,wc,W,C,t,q,
       cfg.public_floor if common else 0,cfg.secret_floor)
    if seed_state is None:
        # Choose the best verified deterministic seed, including the all-public
        # fallback. This prevents a zero confidential-rate seed from looking
        # artificially weak when robust confidential beams already exist.
        best_seed=None
        for strategy in (False,True):
            for fraction in (.05,.15,.25,.35,.5,.7,.0):
                if fraction==0 and not common:continue
                wc2,W2=_seed_beams(H,EH,eps,q,P0,common,private_fraction=fraction,
                                   suppress_user_interference=strategy)
                trial=physical_rates(H,EH,eps,wc2,W2,C,t,q,
                    cfg.public_floor if common else 0,cfg.secret_floor)
                if trial['residual']>=-2e-6:
                    merit=float(np.sum(np.maximum(trial['secret'],cfg.secret_floor)))
                    if best_seed is None or merit>best_seed[0]:
                        best_seed=(merit,wc2,W2,trial)
        if best_seed is not None:
            _,wc,W,initial=best_seed
    direct=initial
    if initial['residual']<-2e-6:
        for _ in range(phase_budget):
            res=_subproblem(H,EH,eps,q,P0,wc,W,C,t,cfg,True,common)
            solves+=1
            if res is None:break
            (wc,W,C,t),slack=res
            direct=physical_rates(H,EH,eps,wc,W,C,t,q,
                cfg.public_floor if common else 0,cfg.secret_floor)
            if direct['residual']>=-2e-6:break
        if direct['residual']<-2e-6:return dict(status='not_found',solves=solves,seconds=time.perf_counter()-start)
    direct=physical_rates(H,EH,eps,wc,W,C,t,q,
        cfg.public_floor if common else 0,cfg.secret_floor)
    t=np.maximum(cfg.secret_floor,direct['secret']-1e-6)
    if common and cfg.lambda_public>0:
        room=max(0.,np.min(direct['common'])-np.sum(C)-1e-6)
        C=C+room/cfg.k
    trace=[];last_verified=None
    for iteration in range(inner_budget+1):
        direct=physical_rates(H,EH,eps,wc,W,C,t,q,
            cfg.public_floor if common else 0,cfg.secret_floor)
        if direct['residual'] < -2e-6:break
        obj=direct['confidential']+cfg.lambda_public*direct['public']
        last_verified=dict(status='verified',objective=obj,theta=theta.copy(),
             q_fraction=q_fraction,secret=direct['confidential'],
             public=direct['public'],residual=direct['residual'],
             min_secret=float(np.min(direct['secret'])),
             wc=wc.copy(),W=W.copy(),C=C.copy(),t=t.copy())
        trace.append(obj)
        if iteration==inner_budget:break
        res=_subproblem(H,EH,eps,q,P0,wc,W,C,t,cfg,False,common)
        solves+=1
        if res is None:break
        (wcn,Wn,Cn,tn),_=res
        check=physical_rates(H,EH,eps,wcn,Wn,Cn,tn,q,
            cfg.public_floor if common else 0,cfg.secret_floor)
        next_obj=check['confidential']+cfg.lambda_public*check['public']
        if check['residual'] < -2e-6 or next_obj < obj-1e-5:break
        wc,W,C,t=wcn,Wn,Cn,tn
        if next_obj-obj<1e-5:
            # Record terminal iterate in the next loop without another solve.
            inner_budget=iteration+1
    if last_verified is None:return dict(status='not_found',solves=solves,seconds=time.perf_counter()-start)
    last_verified.update(trace=np.asarray(trace),solves=solves,seconds=time.perf_counter()-start)
    return last_verified



def screened_search(draw,cfg,seed,mode='joint',candidate_budget=16,
                    phase_budget=6,inner_budget=4,refine_budget=2):
    """Screen finite AN/phase proposals with verified seeds, refine the top two.

    Every screened score is itself a directly verified feasible design. The
    nonlinear transmit subproblem is reserved for the top candidates, and
    even an IPOPT failure cannot make a reported design infeasible.
    """
    start=time.perf_counter()
    theta0=np.ones(cfg.m,complex)
    use_ris=mode!='no_ris'
    alphabet=np.exp(2j*np.pi*np.arange(2**cfg.b)/2**cfg.b)
    grid=(0.,.05,.1,.2,.3) if mode!='no_an' else (0.,)
    screened=[];visited=set();tested=0;solves=0

    def key(th,qf):
        return (round(qf,5),tuple(np.mod(np.rint(np.angle(th)*2**cfg.b/(2*np.pi)).astype(int),2**cfg.b)))

    def screen(th,qf):
        nonlocal tested
        if tested>=candidate_budget or key(th,qf) in visited:return None
        visited.add(key(th,qf));tested+=1
        h,e=draw.effective(th,use_ris)
        eps=draw.eps if use_ris else cfg.epsilon_relative*np.linalg.norm(draw.he,axis=1)
        q=qf/cfg.n;budget=1-qf
        c=np.full(cfg.k,cfg.public_floor)
        best=None
        for strategy in (False,True):
            for fraction in (.05,.15,.25,.35,.5,.7,.9,0.):
                wc,w=_seed_beams(h,e,eps,q,budget,True,fraction,strategy)
                metrics=physical_rates(h,e,eps,wc,w,c,np.full(cfg.k,cfg.secret_floor),q,
                                       cfg.public_floor,cfg.secret_floor)
                if metrics['residual'] < -2e-6:continue
                t=np.maximum(cfg.secret_floor,metrics['secret']-1e-6)
                allocations=c.copy()
                if cfg.lambda_public>0:
                    allocations+=max(0,np.min(metrics['common'])-np.sum(c)-1e-6)/cfg.k
                verified=physical_rates(h,e,eps,wc,w,allocations,t,q,
                                        cfg.public_floor,cfg.secret_floor)
                if verified['residual'] < -2e-6:continue
                objective=verified['confidential']+cfg.lambda_public*verified['public']
                if best is None or objective>best['objective']:
                    best=dict(status='verified',objective=objective,theta=th.copy(),
                        q_fraction=qf,secret=verified['confidential'],
                        public=verified['public'],residual=verified['residual'],
                        min_secret=float(np.min(verified['secret'])),
                        wc=wc,W=w,C=allocations,t=t)
        if best is not None:screened.append(best)
        return best

    initial=screen(theta0,0.)
    if initial is None:
        # A failed analytic seed is not an infeasibility certificate.
        backup=optimize_fixed(draw,cfg,theta0,0.,use_ris,True,
                              phase_budget,inner_budget)
        solves+=backup['solves']
        if backup['status']=='verified':
            initial=backup;screened.append(backup)
    for qf in grid[1:]:screen(theta0,qf)

    if use_ris and mode in ('joint','no_an') and initial is not None:
        basew=initial['W']
        for assignment in (np.arange(cfg.m)%cfg.k,
                           np.minimum(cfg.k-1,np.arange(cfg.m)*cfg.k//cfg.m),
                           *(np.full(cfg.m,k,int) for k in range(cfg.k))):
            th=np.empty(cfg.m,complex)
            for k in range(cfg.k):
                idx=np.flatnonzero(assignment==k)
                if not len(idx):continue
                w=basew[k]
                if np.linalg.norm(w)<1e-12:
                    # Use a direct-channel beam when this private stream is
                    # off in the incumbent, so the proposal remains defined.
                    w=_normalize(np.conj(draw.hd[k]))
                z=draw.bu[k,idx,:]@w
                d=draw.hd[k]@w
                desired=np.exp(1j*((np.angle(d) if abs(d)>1e-12 else 0.)-np.angle(z)))
                th[idx]=alphabet[np.argmin(abs(desired[:,None]-alphabet[None,:]),axis=1)]
            screen(th,0.)

        if tested<candidate_budget:
            # Rank one-element candidates using the incumbent precoders,
            # then screen only the strongest remaining alternatives.
            h0,e0=draw.effective(theta0)
            scored=[]
            for m in range(cfg.m):
                for val in alphabet[1:]:
                    delta=val-theta0[m]
                    h=h0+delta*draw.bu[:,m,:]
                    e=e0+delta*draw.be[:,m,:]
                    a=physical_rates(h,e,draw.eps,initial['wc'],basew,
                         initial['C'],initial['t'],initial['q_fraction']/cfg.n,
                         cfg.public_floor,cfg.secret_floor)
                    merit=np.sum(a['secret'])+cfg.lambda_public*np.min(a['common'])
                    scored.append((merit,m,val))
            for _,m,val in sorted(scored,key=lambda z:-z[0]):
                if tested>=candidate_budget:break
                th=theta0.copy();th[m]=val
                screen(th,0.)

    if not screened:
        return dict(status='not_found',tested=tested,feasible=0,solves=solves,
                    seconds=time.perf_counter()-start)
    ranked=sorted(screened,key=lambda r:-r['objective'])
    best=ranked[0]
    for candidate in ranked[:refine_budget]:
        refined=optimize_fixed(draw,cfg,candidate['theta'],candidate['q_fraction'],
                  use_ris,True,phase_budget,inner_budget,
                  seed_state=(candidate['wc'],candidate['W'],candidate['C'],candidate['t']))
        solves+=refined['solves']
        if refined['status']=='verified' and refined['objective']>best['objective']:
            best=refined
    best.update(tested=tested,feasible=len(screened),solves=solves,
                seconds=time.perf_counter()-start)
    return best
