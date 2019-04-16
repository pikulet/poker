import random
from cfr_methods.hand_evaluation import get_hand_bucket
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards
import pprint

class CfrPlayer(BasePokerPlayer):

  def __init__(self):
    super(CfrPlayer, self).__init__()
    self.info_set = ''
    self.bucket_seq = []
    self.num_hole_cards = 0
    self.num_com_cards = 0
    strategy_num=14
    strategy = {}
    strategy_file_path = './outputs/training_output' + str(strategy_num)
    with open(strategy_file_path, 'r') as strategy_file:
        for line in strategy_file:
            if not line.strip() or line.strip().startswith('#'):
                continue
            line_split = line.split(' ')
            strategy[line_split[0]] = [float(probStr) for probStr in line_split[1:4]]
    self.strategy = strategy

  def convert_street_index_to_name(self, street_index):
      switcher = {
          0: 'preflop',
          1: 'flop',
          2: 'turn',
          3: 'river'
      }
      return switcher.get(street_index, -1)

  def convert_action_to_str(self, action):
    if action == 'FOLD':
        return 'f'
    elif action == 'CALL':
        return 'c'
    elif action == 'RAISE':
        return 'r'
    else:
        raise RuntimeError('Invalid action: %s' % action)

  def _get_info_set(self, hole_card, round_state):
    """Set up info_set according to the current round_state.

    info_set is used as a node key in strategy.
    """
    if len(hole_card) != self.num_hole_cards:
        self.num_hole_cards = len(hole_card)
        current_hand_bucket = get_hand_bucket(hole_card)
        self.bucket_seq.append(current_hand_bucket)


    if len(round_state['community_card']) != self.num_com_cards:
        self.num_com_cards = len(round_state['community_card'])
        com_cards = gen_cards(round_state['community_card'])
        current_hand_bucket = get_hand_bucket(hole_card + com_cards)
        self.bucket_seq.append(current_hand_bucket)

    cur_infoset = ''
    for i in range(len(round_state['action_histories'])):
        if i != 0:
            cur_infoset += ':'
        cur_infoset = cur_infoset + str(self.bucket_seq[i]) + ':'
        current_street_history = round_state['action_histories'][self.convert_street_index_to_name(i)]
        for j in range(len(current_street_history)):
            action = current_street_history[j]['action']
            if action == 'SMALLBLIND' or action == 'BIGBLIND':
                continue
            else:
                cur_infoset += self.convert_action_to_str(action)
    self.info_set = cur_infoset

  def select_action(self, strategy, valid_actions):
    """Randomly select action from node strategy.

    Args:
        strategy (list(float)): Strategy of the node

    Returns:
        acpc.ActionType: Selected action.
    """
    call_action_info = valid_actions[0]
    if len(valid_actions) == 2:
        choice = random.random()
        if choice > strategy[0] + strategy[2] / 2:
            call_action_info = valid_actions[1]
    elif len(valid_actions) == 3:
      choice = random.random()
      probability_sum = 0
      for i in range(3):
          action_probability = strategy[i]
          if action_probability == 0:
              continue
          probability_sum += action_probability
          if choice < probability_sum:
              call_action_info = valid_actions[1]
      # Return the last action since it could have not been selected due to floating point error
      call_action_info = valid_actions[2]
    return call_action_info["action"]

  def declare_action(self, valid_actions, hole_card, round_state):
    # valid_actions format => [raise_action_pp = pprint.PrettyPrinter(indent=2)
    #pp = pprint.PrettyPrinter(indent=2)
    #print("------------ROUND_STATE(RANDOM)--------")
    #pp.pprint(round_state)
    #print("------------HOLE_CARD----------")
    #pp.pprint(hole_card)
    #print("------------VALID_ACTIONS----------")
    #pp.pprint(valid_actions)
    #print("-------------------------------")

    self._get_info_set(gen_cards(hole_card), round_state)
    # if the training output does not include strategy for this information set, just call
    try:
        node_strategy = self.strategy[self.info_set]
    except KeyError:
        # print(self.info_set)
        return valid_actions[1]['action']
    action = self.select_action(node_strategy, valid_actions)
    return action  # action returned here is sent to the poker engine

  def receive_game_start_message(self, game_info):
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    self.info_set = ''
    self.num_hole_cards = 0
    self.num_com_cards = 0
    self.bucket_seq =[]


  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass

def setup_ai():
  return CfrPlayer()
