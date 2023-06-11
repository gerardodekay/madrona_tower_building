import torch
from torch import nn
import torch.nn.functional as F
import torch._dynamo
from torch import optim
from torch.func import vmap
from time import time
from os import environ as env_vars
from typing import Callable

from .cfg import TrainConfig, SimData
from .rollouts import RolloutManager, RolloutMiniBatch

def _ppo_update(cfg : TrainConfig,
                policy : nn.Module,
                policy_train_fwd_fn : Callable,
                mb : RolloutMiniBatch,
                optimizer,
                scaler,
                hindsight_fn=None):

    if mb.rnn_hidden_starts is not None:
        new_log_probs, entropies, new_values = policy_train_fwd_fn(
            mb.rnn_hidden_starts, mb.dones, mb.actions, *mb.obs)
    else:
        new_log_probs, entropies, new_values = policy_train_fwd_fn(
            mb.actions, *mb.obs)

    ratio = torch.exp(new_log_probs - mb.log_probs)

    surr1 = mb.advantages * ratio
    surr2 = mb.advantages * (
        torch.clamp(ratio, 1.0 - cfg.ppo.clip_coef, 1.0 + cfg.ppo.clip_coef))

    action_loss = -torch.min(surr1, surr2)

    returns = mb.advantages + mb.values

    if hindsight_fn is not None:
        hindsight_returns = hindsight_fn(mb)
        returns = torch.max(returns, hindsight_returns)

    if cfg.ppo.clip_value_loss:
        with torch.no_grad():
            low = mb.values - cfg.ppo.clip_coef
            high = mb.values + cfg.ppo.clip_coef

        new_values = torch.clamp(low, high)

    value_loss = 0.5 * F.mse_loss(new_values, returns, reduction='none')

    loss = (
        torch.mean(action_loss) +
        cfg.ppo.value_loss_coef * torch.mean(value_loss) +
        cfg.ppo.entropy_coef * torch.mean(entropies)
    )

    print("Loss:")
    print(loss)

    if scaler is None:
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.ppo.max_grad_norm)
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.ppo.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

    optimizer.zero_grad()

def _update_loop(cfg : TrainConfig,
                 sim : SimData,
                 rollouts : RolloutManager,
                 policy : nn.Module,
                 policy_train_fwd_fn : Callable,
                 policy_rollout_infer_fn : Callable,
                 policy_rollout_infer_values_fn : Callable,
                 optimizer,
                 scaler,
                 scheduler,
                 hindsight_fn=None):
    num_agents = rollouts.actions.shape[1]

    assert(num_agents % cfg.ppo.num_mini_batches == 0)

    for update_idx in range(cfg.num_updates):
        update_start_time = time()

        if update_idx % 1 == 0:
            print(f'Update: {update_idx}')

        with torch.no_grad():
            rollouts.collect(sim, policy_rollout_infer_fn,
                             policy_rollout_infer_values_fn)

        for epoch in range(cfg.ppo.num_epochs):
            for inds in torch.randperm(num_agents).chunk(
                    cfg.ppo.num_mini_batches):
                mb = rollouts.gather_minibatch(inds)
                _ppo_update(cfg, policy, policy_train_fwd_fn, mb,
                            optimizer, scaler, hindsight_fn)

        update_end_time = time()

        print("Time:")
        print(update_end_time - update_start_time)

def train(sim, cfg, policy, dev):
    print(cfg)

    num_agents = sim.actions.shape[0]

    policy = policy.to(dev)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    if dev.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if dev.type == 'cuda':
        def autocast_wrapper(fn):
            def autocast_wrapped_fn(*args, **kwargs):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    return fn(*args, **kwargs)

            return autocast_wrapped_fn
    else:
        def autocast_wrapper(fn):
            return fn

    def policy_rollout_infer(*args, **kwargs):
        return policy.rollout_infer(*args, **kwargs)

    def policy_rollout_infer_values(*args, **kwargs):
        return policy.rollout_infer_values(*args, **kwargs)

    def policy_train_fwd(*args, **kwargs):
        return policy.train(*args, **kwargs)

    policy_rollout_infer = autocast_wrapper(policy_rollout_infer)
    policy_rollout_infer_values = autocast_wrapper(policy_rollout_infer_values)
    policy_train_fwd = autocast_wrapper(policy_train_fwd)

    rollouts = RolloutManager(dev, sim, cfg.steps_per_update, cfg.gamma,
                              cfg.gae_lambda, policy.rnn_hidden_shape)

    if 'MADRONA_TRAIN_NO_TORCH_COMPILE' in env_vars and \
            env_vars['MADRONA_TRAIN_NO_TORCH_COMPILE'] == '1':
        update_loop = _update_loop
    else:
        if 'MADRONA_TRAIN_TORCH_COMPILE_DEBUG' in env_vars and \
                env_vars['MADRONA_TRAIN_TORCH_COMPILE_DEBUG'] == '1':
            torch._dynamo.config.verbose=True

        update_loop = torch.compile(_update_loop, dynamic=False)

    def hindsight_fn(mb: RolloutMiniBatch):
        # Get the number of agents and the length of the rollouts
        rollout_length = mb.actions.shape[0]

        # Perform hindsight relabeling for each agent
        for agent_idx in range(3):
            # Select a random goal from the rollouts
            random_goal_idx = random.randint(0, rollout_length - 1)
            random_goal = mb.obs[random_goal_idx]

            # Update the rewards for the selected goal
            new_rewards = compute_rewards(mb.obs, random_goal)

            # Update the rewards in the minibatch
            mb.rewards[:, agent_idx] = new_rewards

    def compute_rewards(obs, goal):
        # Extract the relevant observation tensors
        prep_counter_tensor = obs[0]
        agent_type_tensor = obs[1]
        agent_data_tensor = obs[2]
        visible_agents_mask_tensor = obs[5]

        # print(visible_agents_mask_tensor)
        # print(visible_agents_mask_tensor.shape)
        print(agent_type_tensor)
        print(agent_type_tensor.shape)

        # Check if all hiders are hidden
        all_hiders_hidden = torch.all(visible_agents_mask_tensor == 0)
        
        # Check if any hider is seen by a seeker
        any_hider_seen = torch.any(visible_agents_mask_tensor)
        
        rewards = []
        for agent_type in agent_type_tensor:
            # Compute the team-based reward for hiders
            print(agent_type)
            print(agent_type.shape)
            if agent_type == 1:
                if all_hiders_hidden:
                    reward = 1.0
                else:
                    reward = -1.0
            
            # Compute the team-based reward for seekers
            elif agent_type == 0:
                if any_hider_seen:
                    reward = 1.0
                else:
                    reward = -1.0
            
            # Invalid agent type
            else:
                raise ValueError("Invalid agent type. Must be 'hider' or 'seeker'.")
            rewards.append(reward)
        
        # Penalize agents going outside the play area
        # for pos in positions:
        #     if abs(pos[0]) >= 18.0 or abs(pos[1]) >= 18.0:
        #         reward -= 10.0
        
        rewards.to(dev)
        return rewards

    update_loop(
        cfg=cfg,
        sim=sim,
        rollouts=rollouts,
        policy=policy,
        policy_rollout_infer_fn=policy_rollout_infer,
        policy_rollout_infer_values_fn=policy_rollout_infer_values,
        policy_train_fwd_fn=policy_train_fwd,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=None,
        hindsight_fn=hindsight_fn)
