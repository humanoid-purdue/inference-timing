import pickle
from brax.training import networks
import jax.numpy as jnp
import jax
import jax.tree_util
import time
import numpy as np
import lstm


with open("lstm_walk_policy", "rb") as f:
    loaded_data = pickle.load(f)
    
params = loaded_data[1]

print(jax.tree_util.tree_map(lambda x: x.shape, loaded_data[1]))


obs_size = 256 + 74

action_size = 48

policy = lstm.make_policy_network(
    obs_size=obs_size, param_size=action_size
)

rng_key = jax.random.PRNGKey(0)

@jax.jit
def get_action(processer_params, params, obs):
    return policy.apply(processer_params, params, obs)

obs = jax.random.uniform(rng_key, (obs_size,))


print(get_action(None, params, obs))

num_trials = 1000
start = time.time()

for _ in range(num_trials):
    action = get_action(None, params, obs)

end = time.time()

avg = (end - start) / num_trials
print(f"{avg * 1000:.4f} ms")

