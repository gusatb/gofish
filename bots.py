import random
import game

class RandomBot(game.PlayerController):
    """Randomly chooses between types of cards in hand.
    """
    def __init__(self):
        super().__init__()

    def get_name(self):
        return 'RandomBot'

    def move(self, visible_state):
        types = list(visible_state.hand.keys())
        return random.choice(types)


class BadBot(game.PlayerController):
    """Orders card types by count, then sorts within count, chooses lowest.
    """
    def __init__(self):
        super().__init__()

    def get_name(self):
        return 'BadBot'

    def move(self, visible_state):
        hand = visible_state.hand
        min_count = min(hand.values())
        lowest_types = [k for k, v in hand.items() if v == min_count]
        lowest_types.sort()
        return lowest_types[0]


class GoodBot(game.PlayerController):
    """Keeps track of opponents cards, asks for a random from highest count.
    """
    def __init__(self):
        super().__init__()
        self.opponent_cards = set()

    def opponent_asks(self, card):
        self.opponent_cards.add(card)

    def get_name(self):
        return 'GoodBot'

    def move(self, visible_state):
        hand = visible_state.hand

        # Update opponent_cards
        for card in list(self.opponent_cards):
            if card in visible_state.sets:
                self.opponent_cards.remove(card)
            elif card in visible_state.opponent_sets:
                self.opponent_cards.remove(card)

        #print(f'DEBUG:\n\t{self.opponent_cards}\n\t{hand}')

        for card in hand.keys():
            if card in self.opponent_cards:
                self.opponent_cards.remove(card)
                return card

        max_count = max(hand.values())
        highest_types = [k for k, v in hand.items() if v == max_count]
        move = random.choice(highest_types)
        if move in self.opponent_cards:
            self.opponent_cards.remove(move)
        return move
