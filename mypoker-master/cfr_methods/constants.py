NUM_PLAYERS = 2
NUM_ACTIONS = 3
SMALL_BLIND_AMOUNT = 10
BIG_BLIND_AMOUNT = 20
TOTAL_ROUNDS = 4
FIRST_PLAYER = 0
NUM_HOLECARDS = 2
NUM_BUCKETS = 5
RAISE_SIZE = 10
MAX_RAISES = [2, 1]


def get_num_board_cards(round_num):
    """:type round_num: number"""
    switcher = {
        0: 0,
        1: 3,
        2: 4,
        3: 5
    }
    return switcher.get(round_num, -1)

def get_max_street_raises(street_num):
    if street_num == 0:
        return 3
    else:
        return 4

def get_street_raise_size(round_index):
    switcher = {
        0: BIG_BLIND_AMOUNT,
        1: BIG_BLIND_AMOUNT,
        2: 2 * BIG_BLIND_AMOUNT,
        3: 2 * BIG_BLIND_AMOUNT
    }
    return switcher.get(round_index, -1)

# TODO: check that small blind speaks first at the begining of each street cfr.112
# TODO: check that the get_num_board_cards switch numbers are set correctly