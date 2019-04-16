from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import random as rand


class HeuristicPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        self.nb_player = 2
        call_action_info = valid_actions[1]
        fold_action_info = valid_actions[0]

        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(nb_simulation=100, nb_player=self.nb_player,
                                               hole_card=gen_cards(hole_card),
                                               community_card=gen_cards(community_card))
        if win_rate > 1 / float(self.nb_player) + 0.1:
            action = call_action_info["action"]
        else:
            action = fold_action_info["action"]
        return action

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


class RandomPlayer(BasePokerPlayer):
    def __init__(self):
        self.fold_ratio, self.call_ratio, raise_ratio = 1.0 / 5, 3.0 / 5, 1.0 / 5

    def set_action_ratio(self, fold_ratio, call_ratio, raise_ratio):
        ratio = [fold_ratio, call_ratio, raise_ratio]
        scaled_ratio = [1.0 * num / sum(ratio) for num in ratio]
        self.fold_ratio, self.call_ratio, self.raise_ratio = scaled_ratio

    def declare_action(self, valid_actions, hole_card, round_state):
        choice = self.__choice_action(valid_actions)
        action = choice["action"]
        return action

    def __choice_action(self, valid_actions):
        r = rand.random()
        if r <= self.fold_ratio:
            return valid_actions[0]
        elif r <= self.call_ratio:
            return valid_actions[1]
        elif len(valid_actions) == 3:
            return valid_actions[2]
        else:
            return valid_actions[0]

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
