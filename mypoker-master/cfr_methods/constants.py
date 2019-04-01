NUM_PLAYERS = 2
NUM_ACTIONS = 3
SMALL_BLIND_AMOUNT = 10
BIG_BLIND_AMOUNT = 20
TOTAL_ROUNDS = 4
FIRST_PLAYER = 0
NUM_HOLECARDS = 2
NUM_BUCKETS = 5
RAISE_SIZE = 10


def get_num_board_cards(round_num):
    """:type round_num: number"""
    switcher = {
        0: 0,
        1: 3,
        2: 4,
        3: 5
    }
    return switcher.get(round_num, -1)

# TODO: check that small blind speaks first at the begining of each street cfr.112
# TODO: check that the get_num_board_cards switch numbers are set correctly