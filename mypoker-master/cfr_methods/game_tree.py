import constants as const


def _tree_str_rec(root, offset):
    result = '%s%s\n' % (' ' * offset, root)
    for key, item in root.children.items():
        result += _tree_str_rec(item, offset + 1)
    return result


def tree_str(root):
    result = ''
    for key, item in root.children.items():
        result += _tree_str_rec(item, 0)
    return result


class Node(object):
    def __init__(self, parent):
        # super(object).__init__()
        self.parent = parent
        self.children = {}

    def set_child(self, key, child):
        self.children[key] = child

    def __str__(self):
        if not self.parent:
            return ''
        parent_str = str(self.parent)
        parents_children = list(filter(lambda item: item[1] == self, self.parent.children.items()))
        if len(parents_children) == 0:
            raise RuntimeError('Parent does have this node as a child')
        child_key = parents_children[0][0]
        parent_type = type(self.parent)
        if parent_type == HoleCardsNode or parent_type == BoardCardsNode:
            # child_key = ':'.join([str(card) for card in child_key]) + ':'
            child_key = str(child_key)  # convert the child key which is the bucket number to string
            if parent_str and not parent_str.endswith(':'):
                child_key = ':' + child_key
        elif parent_type == ActionNode:
            if child_key == 0:
                child_key = 'f'
            elif child_key == 1:
                child_key = 'c'
            elif child_key == 2:
                child_key = 'r'
        return parent_str + child_key


class TerminalNode(Node):
    def __init__(self, parent, pot_commitment):
        # print('a terminal node is constructed')
        super(TerminalNode, self).__init__(parent)  # Add args to super method to adapt to python 2
        self.pot_commitment = pot_commitment


class HoleCardsNode(Node):
    def __init__(self, parent, card_count):
        # print('a hole node is constructed')
        super(HoleCardsNode, self).__init__(parent)
        self.card_count = card_count


class BoardCardsNode(Node):
    def __init__(self, parent, card_count):
        # print('a board node is constructed')
        super(BoardCardsNode, self).__init__(parent)
        self.card_count = card_count


class ActionNode(Node):
    def __init__(self, parent, player):
        # print('an action node is constructed')
        super(ActionNode, self).__init__(parent)
        self.player = player
        self.regret_sum = [0] * const.NUM_ACTIONS
        self.strategy = [0] * const.NUM_ACTIONS
        self.strategy_sum = [0] * const.NUM_ACTIONS
        self.average_strategy = None
