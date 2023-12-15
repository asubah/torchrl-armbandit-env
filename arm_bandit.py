from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
import random

from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn
from torchrl.modules.tensordict_module import Actor

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

NUM_BANDITS = 10
BANDIT_SIZE = 5
bandits = torch.randint(low=0, high=10, size=(NUM_BANDITS, BANDIT_SIZE), dtype=torch.int32)
print(bandits, "\n", bandits.sum(dim=1).unsqueeze(1))


def _step(tensordict):
    selected_arm = tensordict["action"].squeeze(-1)
    # print(selected_arm)`S
    reward = bandits[selected_arm.int(), random.randint(0, bandits.shape[1]-1)].float()
    done = torch.zeros_like(reward, dtype=torch.bool)

    out = TensorDict(
        {
            "selected_arm": selected_arm.int().float(),
            "reward": reward,
            "done": done,
        },
        tensordict.shape,
    )
    return out


def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty():
        tensordict = self.gen_params(batch_size=self.batch_size)
        
    selected_arm = torch.randint(low=0, high=NUM_BANDITS-1, size=tensordict.shape, generator=self.rng)
    
    out = TensorDict({
        "selected_arm": selected_arm.float()
        }, batch_size=tensordict.shape)

    # print ("THIS IS THE ONE", out)
    return out


def _make_spec(self, tensordict):
    self.observation_spec = CompositeSpec(
        selected_arm = BoundedTensorSpec(
            low=0,
            high=NUM_BANDITS,
            shape=(),
            dtype=torch.float32,
        ),
        shape=()
    )
    self.state_spec = self.observation_spec.clone()
    self.action_spec = BoundedTensorSpec(
        low=0,
        high=NUM_BANDITS,
        shape=(1,),
        dtype=torch.float32,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(*tensordict.shape, 1))


def gen_params(batch_size=None) -> TensorDictBase:
    if batch_size is None:
        batch_size = []

    selected_arm = torch.tensor(random.randint(0, bandits.shape[1]-1))
    # print(selected_arm)
    td = TensorDict(
        {
            "selected_arm": selected_arm.int().float()
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td


def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng


class ArmBanditEnv(EnvBase):
    batch_locked = False

    def __init__(self, tensordict=None, seed=None, device="cpu"):
        if tensordict is None:
            tensordict = TensorDict({}, [])

        super().__init__(device=device, batch_size=[])
        self._make_spec(tensordict)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed

env = ArmBanditEnv()
check_env_specs(env)
# print("observation_spec:", env.observation_spec)
# print("state_spec:", env.state_spec)
# print("reward_spec:", env.reward_spec)
# td = env.reset()
# print("reset tensordict", td)
# td = env.rand_step(td)
# print("random step tensordict", td)

env = TransformedEnv(
    env,
    # ``Unsqueeze`` the observations that we will concatenate
    UnsqueezeTransform(
        unsqueeze_dim=-1,
        in_keys=["selected_arm"],
        in_keys_inv=["selected_arm"],
    ),
)

cat_transform = CatTensors(
    in_keys=["selected_arm"], dim=-1, out_key="observation", del_keys=False
)
env.append_transform(cat_transform)
check_env_specs(env)

def simple_rollout(steps=100):
    # preallocate:
    data = TensorDict({}, [steps])
    # reset
    _data = env.reset()
    for i in range(steps):
        _data["action"] = env.action_spec.rand()
        _data = env.step(_data)
        data[i] = _data
        _data = step_mdp(_data, keep_other=True)
    return data


# print("data from rollout:", simple_rollout(100))

batch_size = 10  # number of environments to be executed in batch
# print("TEST TEST TEST")
# td = env.reset(env.gen_params(batch_size=[batch_size]))
# print("reset (batch size of 10)", td)
# td = env.rand_step(td)
# print("rand step (batch size of 10)", td)

rollout = env.rollout(
    3,
    auto_reset=False,  # we're executing the reset out of the ``rollout`` call
    tensordict=env.reset(env.gen_params(batch_size=[batch_size])),
)
print("rollout of len 3 (batch size of 10):", rollout)
