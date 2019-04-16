from functools import reduce
import random
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate as est
from pypokerengine.engine.card import Card

def get_winner(hole_card_1, hole_card_2, community_cards):
    p1_score = HandEvaluator.eval_hand(hole_card_1,community_cards)
    p2_score = HandEvaluator.eval_hand(hole_card_2,community_cards)
    if p1_score > p2_score:
        return 0
    else:
        return 1

def hand_win_rate(state, rounds=500):
    hole_cards = state[:2]
    community_cards = state[2:]
    return est(rounds, 2, hole_cards, community_cards)

def get_hand_bucket(hand):
    '''Arg: a list of card objects, possible length of the list is 2 to 7'''
    if len(hand) not in [2,5,6,7]:
        raise RuntimeError('hand size out of bounds')

    buckets = [
        [0],[1],
        [0.406, 0.482, 0.534, 0.588, 1],
        [3], [4],
        [0.308, 0.43, 0.54, 0.684, 1],
        [0.262, 0.412, 0.562, 0.736, 1],
        [0.198, 0.416, 0.626, 0.832, 1]
    ]

    win_rate = hand_win_rate(hand)
    bucket = buckets[len(hand)]

    b = 1
    while b <= 5:
        if win_rate <= bucket[b-1]:
            break
        b = b + 1

    return b
