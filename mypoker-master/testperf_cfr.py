import sys

sys.path.insert(0, './pypokerengine/api/')
import game

setup_config = game.setup_config
start_poker = game.start_poker
import time

""" =========== *Remember to import your agent!!! =========== """
from randomplayer import RandomPlayer
from honest_player import HonestPlayer
from cfr_player import CfrPlayer

""" ========================================================= """

""" Example---To run testperf.py with random warrior AI against itself. 

$ python testperf.py -n1 "Random Warrior 1" -a1 RandomPlayer -n2 "Random Warrior 2" -a2 RandomPlayer
$ python testperf.py -n1 "CFR Player 1" -a1 CfrPlayer -n2 "Random Warrior 1" -a2 RandomPlayer
"""


def testperf_cfr():
    # Init to play 500 games of 1000 rounds
    num_game = 1
    max_round = 1000
    initial_stack = 10000
    smallblind_amount = 20

    agent1_name = "cfr_player"
    honest_name = "honest_player"
    random_name = "random_player"
    latest_strat = 12
    for i in range(latest_strat):
        # Init pot of players
        agent1_pot = 0
        agent2_pot = 0
        start = time.time()

        print("testing cfr performance with strategy file " + str(i)+ " against honest player")
        config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=smallblind_amount)
        config.register_player(name=agent1_name, algorithm=CfrPlayer(i))
        config.register_player(name=honest_name, algorithm=HonestPlayer())

        # Start playing num_game games
        for game in range(1, num_game + 1):
            print("Game number: ", game)
            game_result = start_poker(config, verbose=0)
            agent1_pot = agent1_pot + game_result['players'][0]['stack']
            agent2_pot = agent2_pot + game_result['players'][1]['stack']

        print("\n After playing {} games of {} rounds, the results are: ".format(num_game, max_round))
        # print("\n Agent 1's final pot: ", agent1_pot)
        print("\n " + agent1_name + "'s final pot: ", agent1_pot)
        print("\n " + honest_name + "'s final pot: ", agent2_pot)

        if (agent1_pot < agent2_pot):
            print("\n Congratulations! " + honest_name + " has won.")
        elif (agent1_pot > agent2_pot):
            print("\n Congratulations! " + agent1_name + " has won.")
        else:
            print("\n It's a draw!")

        end = time.time()
        print("\n Time taken to play: %.4f seconds" % (end - start))

    for i in range(latest_strat):
        # Init pot of players
        agent1_pot = 0
        agent2_pot = 0
        start = time.time()

        print("testing cfr performance with strategy file " + str(i) + " against random player")
        config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=smallblind_amount)
        config.register_player(name=agent1_name, algorithm=CfrPlayer(i))
        config.register_player(name=random_name, algorithm=RandomPlayer())

        # Start playing num_game games
        for game in range(1, num_game + 1):
            print("Game number: ", game)
            game_result = start_poker(config, verbose=0)
            agent1_pot = agent1_pot + game_result['players'][0]['stack']
            agent2_pot = agent2_pot + game_result['players'][1]['stack']

        print("\n After playing {} games of {} rounds, the results are: ".format(num_game, max_round))
        # print("\n Agent 1's final pot: ", agent1_pot)
        print("\n " + agent1_name + "'s final pot: ", agent1_pot)
        print("\n " + random_name + "'s final pot: ", agent2_pot)

        if (agent1_pot < agent2_pot):
            print("\n Congratulations! " + random_name + " has won.")
        elif (agent1_pot > agent2_pot):
            print("\n Congratulations! " + agent1_name + " has won.")
        else:
            print("\n It's a draw!")
        end = time.time()
        print("\n Time taken to play: %.4f seconds" % (end - start))


if __name__ == '__main__':
    testperf_cfr()
