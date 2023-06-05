import madrona_learn # TODO: Unsure where to find this code at the moment
from madrona_learn.model import (ActorCritic, RecurrentActorCritic,
                                 SmallMLPBackbone, LSTMRecurrentPolicy) # Look into where these are defined I cannot find them 
import madrona_simple_example_python # replace
import torch
import argparse
import math

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-agents', type=int, default=1)
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--lr', type=float, default=3e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--steps-per-update', type=int, default=100)
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--cpu-sim', action='store_true')
# TODO: add more arguments if needed
# Example of some megaverse training arguments:
# number_agents_per_env = 1
# rnn_num_layers = 2
# num_epochs = 1
# batch_size = 4096
# num_policies = 1
# exploration loss type and coefficient
# num_envs_per_instance = 48


args = arg_parser.parse_args()

if torch.cuda.is_available():
    dev = torch.device(f'cuda:{args.gpu_id}')
else:
    dev = torch.device('cpu')

sim = madrona_simple_example_python.TowerBuilding(
        exec_mode = madrona_simple_example_python.ExecMode.CPU if args.cpu_sim else madrona_simple_example_python.ExecMode.CUDA,
        gpu_id = args.gpu_id,
        num_worlds = args.num_worlds,
        auto_reset = True
        ### TODO: more arguments if needed
)

### Just taken from gpu_hideseek
def setup_obs(num_agents):
    '''Prepares and processes observations for a simulation with multiple agents.'''

    prep_counter_tensor = sim.prep_counter_tensor().to_torch()[0:args.num_agents]
    agent_data_tensor = sim.agent_data_tensor().to_torch()[0:args.num_agents]
    block_data_tensor = sim.box_data_tensor().to_torch()[0:args.num_agents] # TODO: Replace with other tensors, like for example the blocks
    # ramp_data_tensor = sim.ramp_data_tensor().to_torch()[0:args.num_agents]
    
    obs_tensors = [
        prep_counter_tensor,
        # agent_type_tensor, All agents are the same
        agent_data_tensor,
        block_data_tensor,
        # ramp_data_tensor, TODO: Add more tensors here if needed
        # sim.visible_agents_mask_tensor().to_torch()[0:args.num_agents], I do not think visibility is a concern here
        # sim.visible_boxes_mask_tensor().to_torch()[0:args.num_agents],
        # sim.visible_ramps_mask_tensor().to_torch()[0:args.num_agents],
    ]
    
    num_agent_data_features = math.prod(agent_data_tensor.shape[1:])
    num_block_data_features = math.prod(block_data_tensor.shape[1:])
    # num_ramp_data_features = math.prod(ramp_data_tensor.shape[1:]) TODO: Add more tensors here if needed

    if dev.type == 'cuda':
        conv_args = {
                'dtype': torch.float16,
                'non_blocking': True,
            }
    else:
        conv_args = {
                'dtype': torch.float32,
            }
    
    def process_obs(prep_counter, agent_type, agent_data, block_data,
                    agent_mask, block_mask):
        return torch.cat([
                prep_counter.to(**conv_args),
                agent_type.to(**conv_args),
                (agent_data * agent_mask).to(**conv_args).view(
                    -1, num_agent_data_features),
                (block_data * block_mask).to(**conv_args).view(
                    -1, num_block_data_features),
                # (ramp_data * ramp_mask).to(**conv_args).view( # TODO: Add more tensors here if needed
                #     -1, num_ramp_data_features)
            ], dim=1)

    num_obs_features = prep_counter_tensor.shape[1] + \
        num_agent_data_features + num_block_data_features
        # agent_type_tensor.shape[1] + \
        # num_ramp_data_features TODO: Add more features here if needed

    return obs_tensors, process_obs, num_obs_features


### From gpu_hideseek train.py
# Hack to fill out observations: Reset envs and take step to populate envs
# FIXME: just make it possible to populate observations after init
# (IE run subset of task graph after init)
resets = sim.reset_tensor().to_torch()
actions = sim.action_tensor().to_torch()[0:args.num_agents]
dones = sim.done_tensor().to_torch()[0:args.num_agents]
rewards = sim.reward_tensor().to_torch()[0:args.num_agents]

actions.zero_()
resets[:, 0] = 1 # Unsure what this is doing
resets[:, 1] = args.num_agents
sim.step()

obs_tensors, process_obs_cb, num_obs_features = setup_obs(args.num_agents)

policy = RecurrentActorCritic( # Initializes an object which represents th epolicy for the agent. Combines an actor and critic to evaluate state-action pairs
    backbone = SmallMLPBackbone( # "Backbone" network of th epolicy which processes observations and extracts features that are relevant
        process_obs_cb,
        num_obs_features, 512),
    rnn = LSTMRecurrentPolicy(512, 512, 1),         # Represents the RNN used in the policy. LSTM-based with 512 hidden units and a single layer
    actor = ActorCritic.DefaultDiscreteActor(512,   # Actor component responsible for selecting actions. 512 hidden units, 5 discrete action dimensions. 
        [10, 10, 10, 2, 2]),                        # Can take 10 possible actions in first 3, 2 possible actions in last 2 dimensions
    critic = ActorCritic.DefaultCritic(512))        # Critic component responsible for evaluating state-action pairs.

madrona_learn.train(madrona_learn.SimData(
                        step = lambda: sim.step(),
                        obs = obs_tensors,
                        actions = actions,
                        dones = dones,
                        rewards = rewards,
                    ),
                    madrona_learn.TrainConfig(
                        num_updates = args.num_updates,
                        gamma = args.gamma,
                        gae_lambda = 0.95,
                        lr = args.lr,
                        steps_per_update = args.steps_per_update,
                        # TowerBuilding uses APPO so this should be okay right now. 
                        # I think asynchronous has some slight modifications to how the policy updates and data collection works compared to PPO
                        ppo = madrona_learn.PPOConfig( 
                            num_mini_batches=1,
                            clip_coef=0.2,
                            value_loss_coef=1.0,
                            entropy_coef=0.01,
                            max_grad_norm=5,
                            num_epochs=1,
                            clip_value_loss=True,
                        ),
                    ),
                    policy,
                    dev = dev)

del sim