import sys
from pypokerengine.api.emulator import Action
# from tqdm import tqdm
# for i in tqdm(range(10000)):
try:
    from tqdm import tqdm
except ImportError:
    print('!!! Install tqdm library for better progress information !!!\n')

from cfr_methods.cfr import Cfr
from cfr_methods.game_tree import HoleCardsNode, ActionNode, BoardCardsNode

"""Trains strategy for poker agent using CFR algorithm and writes it to specified file.

Usage:
python train.py {iterations} {strategy_output_path}

  iterations: Number of iterations for which the CFR algorithm will run.
  strategy_output_path: Path to file into which the result strategy will be written. 
"""


# def _action_to_shorthand(action):
#     if action == Action.FOLD:
#         return 'f'
#     elif action == Action.CALL:
#         return 'c'
#     else:
#         return 'r'

def _action_to_shorthand(action):
    if action == 0:
        return 'f'
    elif action == 1:
        return 'c'
    else:
        return 'r'


def _get_strategy_lines(lines, node, prefix=''):
    node_type = type(node)
    if node_type == HoleCardsNode or node_type == BoardCardsNode:
        for key, child_node in node.children.items():
            new_prefix = prefix
            if new_prefix and not new_prefix.endswith(':'):
                new_prefix += ':'
            # new_prefix += ':'.join([str(card) for card in key]) + ':'
            new_prefix += str(key) + ':'
            _get_strategy_lines(lines, child_node, new_prefix)
    elif node_type == ActionNode:
        node_strategy_str = ' '.join([str(prob) for prob in node.average_strategy])
        lines.append('%s %s\n' % (prefix, node_strategy_str))

        for action, child_node in node.children.items():
            _get_strategy_lines(lines, child_node, prefix + _action_to_shorthand(action))


def _write_to_output_file(output_path, lines):
    with open(output_path, 'w') as file:
        for line in lines:
            file.write(line)


def _write_strategy(game_tree, iterations, output_path):
    strategy_file_lines = []

    try:
        with tqdm(total=1) as progress:
            progress.set_description('Obtaining strategy entries')
            _get_strategy_lines(strategy_file_lines, game_tree)
            progress.update(1)
    except NameError:
        _get_strategy_lines(strategy_file_lines, game_tree)

    try:
        with tqdm(total=1) as progress:
            progress.set_description('Sorting strategy file')
            strategy_file_lines_sorted = sorted(strategy_file_lines)
            progress.update(1)
    except NameError:
        strategy_file_lines_sorted = sorted(strategy_file_lines)

    strategy_file_lines_sorted = ['#  Training iterations: %s\n' % iterations] + strategy_file_lines_sorted

    try:
        with tqdm(total=1) as progress:
            progress.set_description('Writing strategy file')
            _write_to_output_file(output_path, strategy_file_lines_sorted)
            progress.update(1)
    except NameError:
        _write_to_output_file(output_path, strategy_file_lines_sorted)


# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage {iterations} {strategy_output_path}")
#         sys.exit(1)
#
#     iterations = int(sys.argv[1])
#     output_path = sys.argv[2]
#
#     cfr = Cfr()
#     cfr.train(iterations)
#
#     _write_strategy(cfr.game_tree, iterations, output_path)

if __name__ == "__main__":
    iterations = 10
    output_path = './training_output'

    cfr = Cfr()
    cfr.train(iterations)

    _write_strategy(cfr.game_tree, iterations, output_path)