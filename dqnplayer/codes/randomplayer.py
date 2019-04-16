from pypokerengine.players import BasePokerPlayer
import random as rand


class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        r = rand.random()
        if r <= 0.5:
            call_action_info = valid_actions[1]
        elif r <= 0.9 and len(valid_actions) == 3:
            call_action_info = valid_actions[2]
        else:
            call_action_info = valid_actions[0]
        action = call_action_info["action"]
        return action  # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return RandomPlayer()
