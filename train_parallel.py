import os
os.environ['MUJOCO_GL'] = 'egl'

import random

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from jaxrl.agents import SACLearner, DACLearner
from jaxrl.datasets import ParallelReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env
import copy
import pickle

import wandb
os.environ["WANDB_MODE"] = "online"

FLAGS = flags.FLAGS

flags.DEFINE_string('exp', '', 'Experiment description (not actually used).')
flags.DEFINE_string('env_name', 'cheetah-run', 'Environment name.')
flags.DEFINE_string('algo_name', 'SAC', '')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 5, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 10000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('checkpoint_freq', int(1e7), 'Frequency at which to save agent and buffer.')
flags.DEFINE_integer('max_steps', int(3e6), 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', int(3e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(25000),'Number of training steps to start training.')
flags.DEFINE_integer('action_repeat', int(2),'')

flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_boolean('anneal_target_beta', False, 'Use tqdm bar.')
flags.DEFINE_boolean('start_from_scratch', True, 'Avoid loading checkpoints.')
flags.DEFINE_integer('updates_per_step', 15, 'Number of updates per step.')
flags.DEFINE_integer('num_seeds', 3, 'Number of parallel seeds to run.')
flags.DEFINE_integer('reset_interval', int(2560000), 'Number of agent updates before a reset.')
flags.DEFINE_boolean('resets', False, 'Whether to reset the agent.')

flags.DEFINE_integer('copy_interval', int(125000), '')
flags.DEFINE_integer('init_copy', int(25000), '')
flags.DEFINE_float('beta_lb', 1.0, '')
flags.DEFINE_float('std_multiplier', 1.25, '')
flags.DEFINE_float('target_kl', 0.5, '')
flags.DEFINE_float('adjustment_lr', 0.00003, '')
flags.DEFINE_float('adjustment_beta', 0.5, '')

config_flags.DEFINE_config_file(
    'config',
    'configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def log_multiple_seeds_to_wandb(step, infos):
    dict_to_log = {}
    for info_key in infos:
        for seed, value in enumerate(infos[info_key]):
            dict_to_log[f'seed{seed}/{info_key}'] = value
    wandb.log(dict_to_log, step=step)


def evaluate_if_time_to(i, agent, eval_env, eval_returns, info, seeds):
    if i % FLAGS.eval_interval == 0:
        if i < FLAGS.start_training:
            eval_stats = {'return': np.zeros(eval_env.num_envs)}
        else:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes, episode_length=1000//FLAGS.action_repeat)
        for j, seed in enumerate(seeds):
            eval_returns[j].append(
                (i, eval_stats['return'][j]))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{seed}.txt'),
                       eval_returns[j],
                       fmt=['%d', '%.1f'])
        log_multiple_seeds_to_wandb(i, eval_stats)


def restore_checkpoint_if_existing(path, agent, replay_buffer):
    if FLAGS.start_from_scratch:
        return 1, agent, replay_buffer, [[] for _ in range(FLAGS.num_seeds)], 0
    else:
        try:
            # Just to protect against agent/replay buffer failure.
            checkpoint_agent = copy.deepcopy(agent)
            checkpoint_agent.load_state(path)
            replay_buffer.load(path)
            with open(os.path.join(path, 'step'), 'r') as f:
                start_step = int(f.read())
            with open(os.path.join(path, 'update_count'), 'r') as f:
                update_count = int(f.read())
            # Load eval returns with pickle
            with open(os.path.join(path, 'eval_returns.pkl'), 'rb') as f:
                eval_returns = pickle.load(f)
            print(f'Loaded checkpoint from {path} at step {start_step}.')
            return start_step, checkpoint_agent, replay_buffer, eval_returns, update_count
        except:
            print("No valid checkpoint found. Starting from scratch.")
            return 1, agent, replay_buffer, [[] for _ in range(FLAGS.num_seeds)], 0


def save_checkpoint(path, step, agent, replay_buffer, eval_returns, update_count):
    agent.save_state(path)
    replay_buffer.save(path)
    with open(os.path.join(path, 'step'), 'w') as f:
        f.write(str(step))
    with open(os.path.join(path, 'eval_returns.pkl'), 'wb') as f:
        pickle.dump(eval_returns, f)
    with open(os.path.join(path, 'update_count'), 'w') as f:
        f.write(str(update_count))
    print("Saved checkpoint to {} at step {}".format(path, step))
    

def main(_):
    wandb.init(config=FLAGS, entity='naumix', project='rebuttal1',group=f"{FLAGS.env_name}_{FLAGS.updates_per_step}", name=f"{FLAGS.algo_name}1_big")
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env = make_env(FLAGS.env_name, FLAGS.seed, None, num_envs=FLAGS.num_seeds, action_repeat=FLAGS.action_repeat)
    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42, eval_episodes=FLAGS.eval_episodes, num_envs=FLAGS.num_seeds, action_repeat=FLAGS.action_repeat)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    reset_list_ = [50000] + [250000 * i for i in range(1,7)] + [500000 * i for i in range(4,6)]
    # Kwargs setup
    all_kwargs = FLAGS.flag_values_dict()
    all_kwargs.update(all_kwargs.pop('config'))
    kwargs = dict(FLAGS.config)
    if FLAGS.algo_name == 'SAC':
        Agent = SACLearner
    if FLAGS.algo_name == 'DAC':
        Agent = DACLearner
        kwargs['copy_interval'] = FLAGS.copy_interval
        kwargs['init_copy'] = FLAGS.init_copy
        kwargs['beta_lb'] = FLAGS.beta_lb
        kwargs['std_multiplier'] = FLAGS.std_multiplier
        kwargs['target_kl'] = FLAGS.target_kl
        kwargs['adjustment_lr'] = FLAGS.adjustment_lr
        kwargs['adjustment_beta'] = FLAGS.adjustment_beta
    _ = kwargs.pop('algo')

    agent = Agent(FLAGS.seed,
                  env.observation_space.sample()[0, np.newaxis],
                  env.action_space.sample()[0, np.newaxis], num_seeds=FLAGS.num_seeds,
                  **kwargs)
        
    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1],
                                         FLAGS.replay_buffer_size,
                                         num_seeds=FLAGS.num_seeds)
    
    observations, dones, rewards, infos = env.reset(), False, 0.0, {}
    idx = 0
    online_rewards = 0.0
    start_step, agent, replay_buffer, eval_returns, update_count = restore_checkpoint_if_existing(FLAGS.save_dir,
                                                                                                  agent,replay_buffer)
    checkpointing_due = False
    for i in tqdm.tqdm(range(start_step, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            actions = env.action_space.sample()
        else:
            if FLAGS.algo_name in ['DAC']:
                actions = agent.sample_actions_o(observations)
            else:
                actions = agent.sample_actions(observations)
                        
        next_observations, rewards, dones, infos = env.step(actions)
        online_rewards += rewards
        idx += 1 
        masks = env.generate_masks(dones, infos)

        replay_buffer.insert(observations, actions, rewards, masks, dones,
                             next_observations)
        observations = next_observations
            
        if i % FLAGS.checkpoint_freq == 0:
            checkpointing_due = True
        checkpointing_due_now = checkpointing_due and dones[0]

        observations, dones = env.reset_where_done(observations, dones)

        if i >= FLAGS.start_training:
            batches = replay_buffer.sample_parallel_multibatch(FLAGS.batch_size, FLAGS.updates_per_step)
            infos = agent.update(batches, int(i-1), num_updates=FLAGS.updates_per_step)
            update_count += FLAGS.updates_per_step
            if i % FLAGS.log_interval == 0:
                log_multiple_seeds_to_wandb(i, infos)

        evaluate_if_time_to(i, agent, eval_env, eval_returns, infos, list(range(FLAGS.seed, FLAGS.seed+FLAGS.num_seeds)))

        if i in reset_list_:
            print(i, 'reset!')
            agent.reset()
            update_count = 0

        if checkpointing_due_now:
            save_checkpoint(FLAGS.save_dir, i, agent, replay_buffer, eval_returns, update_count)
            checkpointing_due = False
            
if __name__ == '__main__':
    app.run(main)
