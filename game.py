import random
from collections import namedtuple

VisibleState = namedtuple('VisibleState', 'hand opponent_hand_size sets opponent_sets deck_size')
"""Info available to make a move.
hand: Dict of card types in hand to count.
opponent_hand_size: Number of cards in opponents hand.
sets: Set of sets scored.
opponent_sets: Set of sets scored by opponent.
deck_size: Number of cards left in the deck.
"""

class PlayerController:
    def __init__(self):
        self.wins = 0

    def new_game(self, player):
        """Informs new game is starting.

        Args:
            player: Which player 1 or 2.
        """
        pass

    def get_name(self):
        pass

    def opponent_asks(self, card):
        pass

    def move(self, visible_state):
        """Returns move to make."""
        pass

class HumanController(PlayerController):
    def __init__(self):
        super().__init__()
        self.player = None
    
    def new_game(self, player):
        self.player = player

    def get_name(self):
        return "Human"

    def opponent_asks(self, card):
        print(f'Info for Player {self.player}: Opponent has asked for {card}')

    def move(self, visible_state):
        p_hand = []
        for k, v in visible_state.hand.items():
            p_hand.extend([k]*v)
        p_hand.sort()
        p_hand = [str(n) for n in p_hand]
        hand_string = ' '.join(p_hand)
        print(f'Player {self.player}\'s turn:')
        print(f'\tYou scored sets: {list(visible_state.sets)}')
        print(f'\tOpponent scored sets: {list(visible_state.opponent_sets)}')
        print(f'\tOpponent hand size: {visible_state.opponent_hand_size}')
        print(f'\tCards remaining in deck: {visible_state.deck_size}')
        print(f'\tYour hand: {hand_string}')
        move = -1
        while move == -1:
            m = input('Card to ask for: ')
            try:
                m = int(m)
                if m in visible_state.hand:
                    move = m
            except:
                pass
        return move

class PlayerState:
    def __init__(self):
        self.hand = {}
        self.sets = set()

    def add_to_hand(self, card, n=1):
        """Returns set was completed or None."""
        if card in self.hand:
            self.hand[card] += n
            if self.hand[card] == 4:
                del self.hand[card]
                self.sets.add(card)
                return card
        else:
            self.hand[card] = n
        assert self.hand[card] > 0 and self.hand[card] < 4
        return None

class GameState:
    def __init__(self, player_controllers, drawless):
        assert len(player_controllers) == 2
        self.deck = [n%13 for n in range(52)]
        random.shuffle(self.deck)
        self.players = [PlayerState(), PlayerState()]
        self.player_controllers = player_controllers
        for i, player in enumerate(self.players):
            self.player_controllers[i].new_game(i+1)
            for _ in range(7):
                player.add_to_hand(self.deck.pop(-1))
        self.player_to_move = 0
        self.winner = -1
        self.drawless = drawless

    def step(self):
        """Returns whether game is over."""
        player = self.players[self.player_to_move]
        player_controller = self.player_controllers[self.player_to_move]
        other_player = self.players[1 - self.player_to_move]
        other_player_controller = self.player_controllers[1 - self.player_to_move]

        opponent_hand_size = sum(other_player.hand.values())

        visible_state = VisibleState(player.hand, opponent_hand_size, player.sets, other_player.sets, len(self.deck))

        if self.drawless and len(player.hand) == 0:
            # Compulsory draw and ask
            card = self.deck.pop(-1)
            player.add_to_hand(card)
            if isinstance(player_controller, HumanController):
                print(f'Player {self.player_to_move} must draw and ask.')
        else:
            card = player_controller.move(visible_state)
    
        other_player_controller.opponent_asks(card)

        assert card in player.hand
        if card in other_player.hand:
            n = other_player.hand[card]
            del other_player.hand[card]
            set_made = player.add_to_hand(card, n)
        else:
            self.player_to_move = 1 - self.player_to_move
            if len(self.deck) > 0:
                new_card = self.deck.pop(-1)
                player.add_to_hand(new_card)
        if not self.drawless:
            if not player.hand or not other_player.hand:
                if len(self.players[0].sets) > len(self.players[1].sets):
                    self.winner = 1
                elif len(self.players[0].sets) < len(self.players[1].sets):
                    self.winner = 2
                else:
                    self.winner = 0
        else:
            if len(player.sets) + len(other_player.sets) == 13:
                self.winner = 1 if len(self.players[0].sets) >= 7 else 2

