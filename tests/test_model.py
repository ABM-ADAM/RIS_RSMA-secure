"""Small checks that exercise the actual channel and conservative-rate model.
Author: Abuzar B. M. Adam
October 2026
"""

import unittest

import numpy as np

from ris_rsma.core import Config, channel_draw, physical_rates


class PhysicalModelTests(unittest.TestCase):
    def test_primitive_errors_fit_effective_uncertainty_ball(self):
        cfg=Config(scenario='blocked',m=64)
        draw=channel_draw(42,cfg)
        rng=np.random.default_rng(43)
        for _ in range(8):
            theta=np.exp(2j*np.pi*rng.integers(0,2**cfg.b,cfg.m)/2**cfg.b)
            direct=rng.normal(size=draw.he.shape)+1j*rng.normal(size=draw.he.shape)
            casc=rng.normal(size=draw.be.shape)+1j*rng.normal(size=draw.be.shape)
            direct*=cfg.epsilon_relative*np.linalg.norm(draw.he,axis=1)[:,None]/np.linalg.norm(direct,axis=1)[:,None]
            casc*=cfg.epsilon_relative*np.linalg.norm(draw.be,axis=(1,2))[:,None,None]/np.linalg.norm(casc,axis=(1,2))[:,None,None]
            effective=direct+np.einsum('m,emn->en',theta,casc)
            self.assertTrue(np.all(np.linalg.norm(effective,axis=1)<=draw.eps+1e-10))

    def test_public_rate_is_not_confidential_rate(self):
        cfg=Config(scenario='blocked',m=64)
        draw=channel_draw(7,cfg)
        h,e=draw.effective(np.ones(cfg.m,complex))
        wc=np.zeros(cfg.n,complex);private=np.zeros((cfg.k,cfg.n),complex)
        common=np.zeros(cfg.k);secret=np.zeros(cfg.k)
        zero=physical_rates(h,e,draw.eps,wc,private,common,secret,0.,0.,0.)
        self.assertEqual(zero['confidential'],0.)
        self.assertEqual(zero['public'],0.)
        self.assertGreaterEqual(zero['residual'],-1e-12)
        demand=physical_rates(h,e,draw.eps,wc,private,common,secret,0.,.1,0.)
        self.assertLess(demand['residual'],0.)


if __name__=='__main__':unittest.main()
