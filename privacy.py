from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition 
from autodp import mechanism_zoo, transformer_zoo
import jax
import jax.numpy as jnp

class NoisySGD_mech(Mechanism):
    def __init__(self,prob,sigma,niter,name='NoisySGD'):
        Mechanism.__init__(self)
        self.name=name
        self.params={'prob':prob,'sigma':sigma,'niter':niter}
        
        # create such a mechanism as in previously
        subsample = transformer_zoo.AmplificationBySampling() # by default this is using poisson sampling
        mech = mechanism_zoo.GaussianMechanism(sigma=sigma)
        prob = prob
        # Create subsampled Gaussian mechanism
        SubsampledGaussian_mech = subsample(mech,prob,improved_bound_flag=True)

        # Now run this for niter iterations
        compose = transformer_zoo.Composition()
        mech = compose([SubsampledGaussian_mech],[niter])

        # Now we get it and let's extract the RDP function and assign it to the current mech being constructed
        rdp_total = mech.RenyiDP
        self.propagate_updates(rdp_total,type_of_update='RDP')

def tree_map_add_normal_noise(
    tree,
    noise_std,
    rng_key):
  """Add iid gaussian noise with std 'noise_std' to all leaves of 'tree'."""
  rng_keys = jax.random.split(rng_key, len(jax.tree_util.tree_leaves(tree)))
  rng_tree = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(tree), rng_keys
  )

  def with_noise(rng, x):
    scale = jnp.asarray(noise_std, dtype=x.dtype)
    return x + scale * jax.random.normal(rng, shape=x.shape, dtype=x.dtype)

  return jax.tree.map(with_noise, rng_tree, tree)

# import numpy as np
# gamma = 128 / 60000
# epoch = 0
# niter = int(np.ceil((epoch+1)/gamma))
# sigma = 1.0
        
# noisysgd = NoisySGD_mech(prob=gamma,sigma=sigma,niter=niter)


# # compute epsilon, as a function of delta
# print(noisysgd.get_approxDP(delta=1e-6))
