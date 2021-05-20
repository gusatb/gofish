import abc
import tensorflow as tf
import numpy as np

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

import matplotlib
import matplotlib.pyplot as plt

import gofish_env
import bots


replay_buffer_size = 20000
fc_layer_params = (256,64)
learning_rate = 5e-5
init_collection_steps = 10000
num_iterations = 100000
collect_steps_per_iteration = 1
log_interval = 100
eval_interval = 10000
num_eval_episodes = 100
train_reward_run_amount = 0.99

bot = bots.RandomBot()

memory_features = True
lose_on_illegal_move = False
drawless = False
conv_features = True

train_py_environment = gofish_env.GoFishEnv(bot, max_visible_opponent_hand_size=10, max_visible_deck_size=10, drawless=drawless, lose_on_illegal_move=lose_on_illegal_move, memory_features=memory_features, conv_features=conv_features)

print('Validating env.')
utils.validate_py_environment(train_py_environment, episodes=5)
print('Validation complete.')

eval_py_environment = gofish_env.GoFishEnv(bot, max_visible_opponent_hand_size=10, max_visible_deck_size=10, drawless=drawless, lose_on_illegal_move=lose_on_illegal_move, memory_features=memory_features, conv_features=conv_features)
train_env = tf_py_environment.TFPyEnvironment(train_py_environment)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_environment)

action_tensor_spec = tensor_spec.from_spec(train_py_environment.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

if conv_features:

else:
    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(num_units):
      return tf.keras.layers.Dense(
          num_units,
          activation=tf.keras.activations.relu,
          kernel_initializer=tf.keras.initializers.VarianceScaling(
              scale=2.0, mode='fan_in', distribution='truncated_normal'))

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # it's output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy



def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]



replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_size)



random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

def collect_step(environment, policy, buffer):
  """Returns reward and termination."""
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)
  return next_time_step.reward[0], next_time_step.is_last()


def collect_data(env, policy, buffer, steps):
  games_completed = 0
  total_reward = 0.0
  for _ in range(steps):
    r, t = collect_step(env, policy, buffer)
    total_reward += r
    games_completed += 1 if t else 0
  return total_reward, games_completed



collect_data(train_env, random_policy, replay_buffer, init_collection_steps)


dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=16, 
    num_steps=2).prefetch(3)

iterator = iter(dataset)


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)



# Evaluate the agent's policy once before training.
train_games = 0
train_rewards = 0

avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]
for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  rs, gs = collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)
  if gs:
    train_rewards = rs*(1-train_reward_run_amount) + train_rewards*train_reward_run_amount
    train_games = gs*(1-train_reward_run_amount) + train_games*train_reward_run_amount

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}, rough winrate: {2}'.format(step, train_loss, (train_rewards/train_games + 1)/2))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    win_rate = (avg_return+1)/2
    print('step = {0}: Average Return = {1}, Win Rate = {2}'.format(step, avg_return, win_rate))
    returns.append(avg_return)


final_eval = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
final_win_rate = (avg_return+1)/2
print('Final Avg Return = {1}, Final Win Rate = {2}'.format(step, final_eval, final_win_rate))

iterations = range(0, num_iterations + 1, eval_interval)

def plot_labeled_horizontal(label, x, y_value, color):
  horiz_line_data = np.array([y_value]*len(iterations))
  #plt.axvline()
  plt.text(x, y_value+0.01,label,rotation=0, color=color)
  plt.plot(iterations, horiz_line_data, f'{color}--') 

plot_labeled_horizontal('good_bot_best', 0, 0.567, color='g')
plot_labeled_horizontal('agent_final', iterations[len(iterations)//2], final_win_rate, color='r')
plt.plot(iterations, [(r+1)/2 for r in returns])
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(bottom=-1, top=1)
plt.show()










