import game as gofish
import bots
import random


def get_int_input(message='>> '):
    while True:
        m = input(message)
        try:
            m = int(m)
        except:
            continue
        return m

def get_players_string():
    return 'Choose Player:\n\t1: Human\n\t2: RandomBot\n\t3: BadBot\n\t4: GoodBotHigh3\n\t5: GoodBotSame3\n\t6: GoodBotRandom'

def player_selection_to_bot(selection):
    if selection == 1:
        return gofish.HumanController()
    elif selection == 2:
        return bots.RandomBot()
    elif selection == 3:
        return bots.BadBot()
    elif selection == 4:
        return bots.GoodBot(default_ask='random_high')
    elif selection == 5:
        return bots.GoodBot(default_ask='same_high')
    elif selection == 6:
        return bots.GoodBot(default_ask='random_all')
    raise ValueError('Incorrect Selection')

def play_game(drawless, player_controllers):
    game = gofish.GameState(player_controllers, drawless=drawless)

    player_1_name = player_controllers[0].get_name()
    player_2_name = player_controllers[1].get_name()
    if has_human:
        print(f'Player 1 played by {player_1_name}')
        print(f'Player 2 played by {player_2_name}')

    while game.winner == -1:
        game.step()
    if game.winner != 0:
        if has_human:
            print(f'Winner: Player {game.winner}, {player_controllers[game.winner-1].get_name()}')
        player_controllers[game.winner-1].wins += 1
    elif has_human:
        print('Draw!')
    return game.winner



print('Which variant:\n\t1: Allow Draws\n\t2: No Draws')
drawless = get_int_input() == 2

print('Series:\n\t1: Tournament\n\t2: Head to Head')
tournament = get_int_input() == 1

n_games = get_int_input('How many games?\n>> ')

if not tournament:
    global_players = []
    has_human = False
    for _ in range(2):
        print(get_players_string())
        selection = get_int_input()
        if selection == 1:
            has_human = True
        global_players.append(player_selection_to_bot(selection))

    for i_game in range(n_games):
        player_controllers = [global_players[i_game % 2], global_players[(i_game+1)%2]]
        play_game(drawless, player_controllers)

    print(f'Drawless variant: {drawless}')
    print(f'Total Wins:')
    print(f'\t{global_players[0].get_name()}: {global_players[0].wins/n_games}, {global_players[0].wins}')
    print(f'\t{global_players[1].get_name()}: {global_players[1].wins/n_games}, {global_players[1].wins}')
    print(f'\tDraws: {(n_games - (global_players[0].wins + global_players[1].wins))/n_games}, {n_games - (global_players[0].wins + global_players[1].wins)}')
elif tournament:
    has_human = False
    all_players = [player_selection_to_bot(n) for n in range(2,7)]
    for i in range(n_games):
        player_controllers = list(random.sample(all_players, 2))
        winner = play_game(drawless, player_controllers)

        # Elo calculations
        elo_a = player_controllers[0].elo
        elo_b = player_controllers[1].elo
        qa = 10.0**(elo_a/400.0)
        qb = 10.0**(elo_b/400.0)
        ea = qa/(qa+qb)
        eb = qb/(qa+qb)
        k = 32.0 if i < n_games//2 else 16.0

        if winner == 1:
            sa = 1.0
            sb = 0.0
        elif winner == 2:
            sa = 0.0
            sb = 1.0
        elif winner == 0:
            sa = 0.5
            sb = 0.5

        player_controllers[0].elo += k * (sa-ea)
        player_controllers[1].elo += k * (sb-eb)
    print(f'Results from Drawless={drawless} tournament:')
    all_players.sort(key=lambda x: x.elo)
    for player in all_players:
        print(f'\t{player.get_name()}: {player.elo}')






