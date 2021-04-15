import game as gofish
import bots


def get_int_input(message='>> '):
    while True:
        m = input(message)
        try:
            m = int(m)
        except:
            continue
        return m

def get_players_string():
    return 'Choose Player:\n\t1: Human\n\t2: RandomBot\n\t3: BadBot\n\t4: GoodBot\n\t5: GoodBotNoRand'

def player_selection_to_bot(selection):
    if selection == 1:
        return gofish.HumanController()
    elif selection == 2:
        return bots.RandomBot()
    elif selection == 3:
        return bots.BadBot()
    elif selection == 4:
        return bots.GoodBot()
    elif selection == 5:
        return bots.GoodBot(no_random=True)
    raise ValueError('Incorrect Selection')

print('Which variant:\n\t1: Allow Draws\n\t2: No Draws')
drawless = get_int_input() == 2

global_players = []
has_human = False
for _ in range(2):
    print(get_players_string())
    selection = get_int_input()
    if selection == 1:
        has_human = True
    global_players.append(player_selection_to_bot(selection))

n_games = get_int_input('How many games?\n>> ')

for i_game in range(n_games):
    # Global player indices in game indices list: player_map[1 == first_move] = 1==human_player
    player_controllers = [global_players[i_game % 2], global_players[(i_game+1)%2]]
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

print(f'Drawless variant: {drawless}')
print(f'Total Wins:')
print(f'\t{player_controllers[0].get_name()}: {player_controllers[0].wins/n_games}, {player_controllers[0].wins}')
print(f'\t{player_controllers[1].get_name()}: {player_controllers[1].wins/n_games}, {player_controllers[1].wins}')
print(f'\tDraws: {(n_games - (player_controllers[0].wins + player_controllers[1].wins))/n_games}, {n_games - (player_controllers[0].wins + player_controllers[1].wins)}')
