import unittest
import numpy as np
from numpy.linalg import norm
from anchored_bearing_based_stabilization import make_eight, make_four, initial_eight, ImplicitStepper, local_spectral_decay


class NumericalChecks(unittest.TestCase):
    def test_initial_seed(self):
        x,v=initial_eight('nominal')
        rng=np.random.default_rng(2026)
        np.testing.assert_array_equal(x,rng.uniform(-5,5,(8,3)))
        np.testing.assert_array_equal(v,rng.uniform(-1,1,(8,3)))

    def test_eight_target_rank_and_hessian(self):
        s=make_eight()
        M,K=s.matrices()
        self.assertEqual(np.linalg.matrix_rank(M,tol=1e-10),24)
        self.assertAlmostEqual(np.linalg.eigvalsh(K)[0],.03179028220724405,places=10)
        self.assertAlmostEqual(np.linalg.svd(M,compute_uv=False)[-1],.14240877226762516,places=9)

    def test_regularized_gradient_finite_difference(self):
        s=make_eight();x,_=initial_eight('collision_equal')
        rng=np.random.default_rng(12);u=rng.normal(size=x.shape);u/=norm(u)
        epsilon=1e-4;dh=1e-7
        d1=(s.potential(x+dh*u,epsilon)-s.potential(x-dh*u,epsilon))/(2*dh)
        d2=np.sum(s.grad(x,epsilon)*u)
        self.assertAlmostEqual(d1,d2,delta=4e-7)

    def test_implicit_initial_collision_energy(self):
        s=make_eight();x,v=initial_eight('collision_equal')
        stepper=ImplicitStepper(s,dt=.05)
        Eprev=s.energy(x,v)
        for _ in range(5):
            x,v=stepper.step(x,v)
            E=s.energy(x,v)
            self.assertLessEqual(E,Eprev+1e-8)
            self.assertLessEqual(E-Eprev+.05*np.sum(s.damping[:,None]*v*v),1e-8)
            Eprev=E

    def test_gain_spectral_rates(self):
        s=make_eight()
        self.assertAlmostEqual(local_spectral_decay(s)[1],.06574153941694,places=9)
        s.damping[:]=.35
        self.assertAlmostEqual(local_spectral_decay(s)[1],.35,places=12)
        s.damping[:]=1.
        s.anchors[:3]=8.
        self.assertAlmostEqual(local_spectral_decay(s)[1],.07188407286697,places=9)

    def test_rank_obstruction(self):
        bad=make_four();good=make_four(True)
        self.assertEqual(np.linalg.matrix_rank(bad.matrices()[0],tol=1e-10),11)
        self.assertEqual(np.linalg.matrix_rank(good.matrices()[0],tol=1e-10),12)
        x=bad.target.copy();x[3]=bad.target[2]+1.6*(bad.target[3]-bad.target[2])
        self.assertLess(norm(bad.grad(x,1e-6)),1e-12)
        self.assertGreater(norm(good.grad(x,1e-6)),1e-2)


if __name__=='__main__':
    unittest.main()
