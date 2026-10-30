"""Model and numerical verification independent of the plot code.
Author: Abuzar B. M. Adam
October 2026
"""
import numpy as np
from dataclasses import replace
from .core import Config, channel_draw, optimize_fixed, physical_rates, screened_search


def run():
    cfg=Config()
    assert abs(10*np.log10(cfg.prb_power_watt)+30-12.552725051)<1e-6
    assert abs(10*np.log10(cfg.noise_watt)+30+111.436974992)<1e-6
    theta=np.ones(cfg.m,complex)
    for seed in (0,1,10,18):
        draw=channel_draw(seed,cfg)
        for _ in range(10):
            rng=np.random.default_rng(seed*100+_)
            theta=np.exp(2j*np.pi*rng.integers(0,2**cfg.b,cfg.m)/2**cfg.b)
            direct=(rng.standard_normal((cfg.e,cfg.n))+1j*rng.standard_normal((cfg.e,cfg.n)))
            casc=(rng.standard_normal((cfg.e,cfg.m,cfg.n))
                  +1j*rng.standard_normal((cfg.e,cfg.m,cfg.n)))
            direct*=cfg.epsilon_relative*np.linalg.norm(draw.he,axis=1)[:,None]/np.linalg.norm(direct,axis=1)[:,None]
            casc*=cfg.epsilon_relative*np.linalg.norm(draw.be,axis=(1,2))[:,None,None]/np.linalg.norm(casc,axis=(1,2))[:,None,None]
            error=direct+np.einsum('m,emn->en',theta,casc)
            assert np.all(np.linalg.norm(error,axis=1)<draw.eps+1e-10)
        fit=optimize_fixed(draw,cfg,theta,q_fraction=.05,phase_budget=5,inner_budget=5)
        assert fit['status']=='verified',f'{seed}: failed start'
        H,E=draw.effective(theta)
        verified=physical_rates(H,E,draw.eps,fit['wc'],fit['W'],fit['C'],fit['t'],
            fit['q_fraction']/cfg.n,cfg.public_floor,cfg.secret_floor)
        assert verified['residual']>=-2e-6
        assert np.all(np.diff(fit['trace'])>=-1e-5)
        print(f'seed={seed}, objective={fit["objective"]:.3f}, '
              f'minimum residual={verified["residual"]:.2e}')
    cfg=replace(cfg,scenario='blocked',m=256)
    draw=channel_draw(1000,cfg)
    for mode in ('joint','no_an','fixed_ris','no_ris'):
        out=screened_search(draw,cfg,1000,mode,12,6,4)
        assert out['status']=='verified'
        H,E=draw.effective(out['theta'],use_ris=(mode!='no_ris'))
        eps=draw.eps if mode!='no_ris' else cfg.epsilon_relative*np.linalg.norm(draw.he,axis=1)
        check=physical_rates(H,E,eps,out['wc'],out['W'],out['C'],out['t'],
            out['q_fraction']/cfg.n,cfg.public_floor,cfg.secret_floor)
        assert check['residual']>=-2e-6
        print(f'blocked {mode}: objective={out["objective"]:.3f}, '
              f'minimum residual={check["residual"]:.2e}')


if __name__=='__main__':run()
