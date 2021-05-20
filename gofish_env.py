import abc
import tensorflow as tf
import numpy as np
import random

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import game

class ActionPlayerController(game.PlayerController):
    """PlayerController which plays the most recently set action."""
    def __init__(self):
        self._action = None
        self.opponent_cards = set()

    def set_action(self, action):
        """Sets next action to take."""
        self._action = action

    def move(self, visible_state):
        """Returns move to make."""
        hand = visible_state.hand

        # Update opponent_cards
        for card in list(self.opponent_cards):
            if card in visible_state.sets:
                self.opponent_cards.remove(card)
            elif card in visible_state.opponent_sets:
                self.opponent_cards.remove(card)

        move = self._action
        if move in self.opponent_cards:
          self.opponent_cards.remove(move)
        return self._action

    def new_game(self, player):
        self.opponent_cards = set()

    def opponent_asks(self, card):
        self.opponent_cards.add(card)


class GoFishEnv(py_environment.PyEnvironment):
  """Environment for playing GoFish against a bot.

  Observations:
    hand: one-of-4 for each of the 13 ranks. 
    opponent_hand_size: one-of-(max_visible_opponent_hand_size) or more_than
    deck_size: one_of-(max_visible_deck_size) or more than

    Total observations: 52+(max_opponent_hand)+(max_deck_size)+4 (one for 0s one for more than)

  Actions:
    One-of-13: illegal to ask for a card you don't have.

  Attributes:
    _action_spec: ActionSpec.
    _observation_spec: ObservationSpec.
    _game_state: GoFish GameState.
    _bot: PlayerController to control opponent.
    _action_player_controller: ActionPlayerController as medium for agents actions.
    _max_visible_opponent_hand_size: Max observable opponent hand size.
    _max_visible_deck_size: Max observable deck size.
    _drawless: Whether the drawless variant of the game is being played.
    _agent_player: Which player (0 or 1) the agent is controlling.
    _lose_on_illegal_move: Whether to lose on illegal move or choose random move.
    _memory_features: Whether to supply known opponent cards as part of features.
    _conv_features: Whether to return features in a format for conv layers
  """
  def __init__(self, bot, max_visible_opponent_hand_size=10, max_visible_deck_size=10, drawless=False, lose_on_illegal_move=True, memory_features=False, conv_features=False):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=12, name='action')

    self._conv_features = conv_features

    self._observation_spec = array_spec.BoundedArraySpec(
        shape=self._calculate_obs_shape, dtype=np.int32, minimum=0, maximum=1, name='observation')


    self._game_state = None
    self._player = bot
    self._bot = bot
    self._action_player_controller = ActionPlayerController()
    self._max_visible_opponent_hand_size = max_visible_opponent_hand_size
    self._max_visible_deck_size = max_visible_deck_size
    self._drawless = drawless
    self._agent_player = None
    self._lose_on_illegal_move = lose_on_illegal_move
    self._memory_features = memory_features

  def _calculate_obs_shape(self):
    if self._conv_features:
      memory_depth = 1 if self._memory_features else 0
      depth = 4 + self._max_visible_hand_size + self._max_visible_deck_size + 4 + memory_depth
      return (13, depth)
    else:
      memory_size = 13 if self._memory_features else 0
      return (52 + self._max_visible_opponent_hand_size + self._max_visible_deck_size + 4 + memory_size,)

  def _get_observation(self):
    player = self._game_state.players[self._game_state.player_to_move]
    assert self._game_state.player_controllers[self._game_state.player_to_move] == self._action_player_controller or self._game_state.winner != -1
    other_player = self._game_state.players[1 - self._game_state.player_to_move]

    hand = player.hand
    opponent_hand_size = sum(other_player.hand.values())
    deck_size = len(self._game_state.deck)

    if self._conv_features:
      obs = np.empty(self._calculate_obs_shape(), dtype=np.int32)
      for rank in range(13):
          in_hand = 0 if rank not in hand else hand[rank]
          for quantity in range(4):
              obs[rank, quantity] = 1 if in_hand == quantity else 0

      depth = 4
      for size in range(self._max_visible_opponent_hand_size+1):
          obs[:,depth] = 1 if opponent_hand_size == size else 0
          depth += 1
      obs[:,depth] = 1 if opponent_hand_size > self._max_visible_opponent_hand_size else 0
      depth += 1
  
      for size in range(self._max_visible_deck_size+1):
          obs[:,depth] = 1 if deck_size == size else 0
          depth += 1
      obs[:,depth] = 1 if deck_size > self._max_visible_deck_size else 0
      
      if self_memory_features:
        depth += 1
        for rank in range(13):
              obs[rank, depth] = 1 if rank in self._action_player_controller.opponent_cards else 0

      assert depth == obs.shape[1]
      return obs
      
    else:
      obs = []
      for rank in range(13):
          in_hand = 0 if rank not in hand else hand[rank]
          for quantity in range(4):
              obs.append(1 if in_hand == quantity else 0)
 
      for size in range(self._max_visible_opponent_hand_size+1):
          obs.append(1 if opponent_hand_size == size else 0)
      obs.append(1 if opponent_hand_size > self._max_visible_opponent_hand_size else 0)
  
      for size in range(self._max_visible_deck_size+1):
          obs.append(1 if deck_size == size else 0)
      obs.append(1 if deck_size > self._max_visible_deck_size else 0)
  
      if self._memory_features:
          for rank in range(13):
              obs.append(1 if rank in self._action_player_controller.opponent_cards else 0)

      obs = np.array(obs, dtype=np.int32)
    return obs


  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._agent_player = random.randint(0,1)
    player_controllers = [self._bot]
    player_controllers.insert(self._agent_player, self._action_player_controller)
    self._game_state = game.GameState(player_controllers, self._drawless)
    game_over = -1
    while self._game_state.player_to_move != self._agent_player and game_over == -1:
        game_over = self._game_state.step()
    assert game_over == -1

    next_obs = self._get_observation()
    return ts.transition(next_obs, reward=0.0)

  def _step(self, action):
    if self._game_state.winner != -1:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()


    action = int(action)
    player = self._game_state.players[self._game_state.player_to_move]
    
    if self._lose_on_illegal_move:
      if action not in player.hand:
        next_obs = self._get_observation()
        return ts.termination(next_obs, reward=-1.0)
    elif action not in player.hand:
      action = random.choice(list(player.hand.keys()))
    
    self._action_player_controller.set_action(action)
    game_over = self._game_state.step()
    while self._game_state.player_to_move != self._agent_player and game_over == -1:
        game_over = self._game_state.step()
    next_obs = self._get_observation()
    if game_over != -1:
        if game_over == 0:
            reward = 0.0
        elif game_over - 1 == self._agent_player:
            reward = 1.0
        elif game_over - 1 == 1 - self._agent_player:
            reward = -1.0
        else:
            assert False
        return ts.termination(next_obs, reward=reward)

    return ts.transition(next_obs, reward=0.0)
