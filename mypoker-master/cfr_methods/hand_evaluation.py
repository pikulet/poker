from functools import reduce
import random
from pypokerengine.engine.hand_evaluator import HandEvaluator


def get_winner(hole_card_1, hole_card_2, community_cards):
    p1_score = HandEvaluator.eval_hand(hole_card_1,community_cards)
    p2_score = HandEvaluator.eval_hand(hole_card_2,community_cards)
    if p1_score > p2_score:
        return 0
    else:
        return 1

def get_hand_bucket(hand):
    '''Arg: a list of card objects, possible length of the list is 2 to 7'''
    if len(hand) > 7 or len(hand) < 2:
        raise RuntimeError('hand size out of bounds')
    return random.randint(1, 5)
