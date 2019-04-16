from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import Card

import tensorflow as tf
import numpy as np
import random

# import sys
# sys.path.insert(0, '../scripts/')

suits = list(Card.SUIT_MAP.keys())
ranks = list(Card.RANK_MAP.keys())

card_pairs_prob = {}
DQN_weights = {}


class DQNPlayer(BasePokerPlayer):
    def __init__(self, h_size=128, lr=0.0001, gradient_clip_norm=500, total_num_actions=3, is_double=True,
                 is_main=False, is_restore=True, is_train=False, debug=False):
        self.h_size = h_size
        self.lr = lr
        self.gradient_clip_norm = gradient_clip_norm
        self.total_num_actions = total_num_actions
        self.is_double = is_double
        self.is_main = is_main
        self.is_restore = is_restore
        self.is_train = is_train
        self.debug = debug

        self.hole_card_est = card_pairs_prob

        if not is_train:
            tf.reset_default_graph()

        self.scalar_input = tf.placeholder(tf.float32, [None, 17 * 17 * 1])
        self.features_input = tf.placeholder(tf.float32, [None, 13])

        xavier_init = tf.contrib.layers.xavier_initializer()

        self.img_in = tf.reshape(self.scalar_input, [-1, 17, 17, 1])
        self.conv1 = tf.layers.conv2d(self.img_in, 32, 5, 2, activation=tf.nn.elu, kernel_initializer=xavier_init)
        self.conv2 = tf.layers.conv2d(self.conv1, 64, 3, activation=tf.nn.elu, kernel_initializer=xavier_init)
        self.conv3 = tf.layers.conv2d(self.conv2, self.h_size, 5, activation=tf.nn.elu,
                                      kernel_initializer=xavier_init)
        self.conv3_flat = tf.contrib.layers.flatten(self.conv3)
        self.conv3_flat = tf.layers.dropout(self.conv3_flat)

        self.d1 = tf.layers.dense(self.features_input, 64, activation=tf.nn.elu, kernel_initializer=xavier_init)
        self.d1 = tf.layers.dropout(self.d1)
        self.d2 = tf.layers.dense(self.d1, 128, activation=tf.nn.elu, kernel_initializer=xavier_init)
        self.d2 = tf.layers.dropout(self.d2)

        self.merge = tf.concat([self.conv3_flat, self.d2], axis=1)
        self.d3 = tf.layers.dense(self.merge, 256, activation=tf.nn.elu, kernel_initializer=xavier_init)
        self.d3 = tf.layers.dropout(self.d3)
        self.d4 = tf.layers.dense(self.d3, self.h_size, activation=tf.nn.elu, kernel_initializer=xavier_init)

        if is_double:
            self.stream_A, self.stream_V = tf.split(self.d4, 2, 1)
            self.AW = tf.Variable(xavier_init([self.h_size // 2, total_num_actions]))
            self.VW = tf.Variable(xavier_init([self.h_size // 2, 1]))

            self.advantage = tf.matmul(self.stream_A, self.AW)
            self.value = tf.matmul(self.stream_V, self.VW)

            self.Q_out = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, 1, True))
        else:
            self.Q_out = tf.layers.dense(self.d4, 3, kernel_initializer=xavier_init)

        self.predict = tf.argmax(self.Q_out, 1)

        self.target_Q = tf.placeholder(tf.float32, [None])
        self.actions = tf.placeholder(tf.int32, [None])
        self.actions_onehot = tf.one_hot(self.actions, total_num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.Q - self.target_Q)
        self.loss = tf.reduce_mean(self.td_error)

        if is_main:
            variables = tf.trainable_variables()  # [:len(tf.trainable_variables()) // 2]
            if is_train:
                self._print(len(variables))
                self._print(variables)
            self.gradients = tf.gradients(self.loss, variables)
            #             self.grad_norms = tf.global_norm(self.gradients)
            self.var_norms = tf.global_norm(variables)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, gradient_clip_norm)
            self.grad_norms = tf.global_norm(grads)
            self.trainer = tf.train.AdamOptimizer(lr)
            #             self.update_model = self.trainer.minimize(self.loss)
            self.update_model = self.trainer.apply_gradients(zip(grads, variables))

            self.summary_writer = tf.summary.FileWriter('../log/DQN/')

        if not is_train:
            self.init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(self.init)

        if is_restore:

            vars = tf.global_variables()
            for var in vars:
                op = tf.assign(var, np.array(DQN_weights[var.name]))
                self.sess.run(op)
                print(var.name)
                if var.name == "Variable_1:0":
                    print(DQN_weights[var.name])
                    print(self.sess.run(var.read_value()).tolist())

    def _print(self, *msg):
        if self.debug:
            print(msg)

    def declare_action(self, valid_actions, hole_card, round_state):
        street = round_state['street']
        bank = round_state['pot']['main']['amount']
        stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]
        other_stacks = [s['stack'] for s in round_state['seats'] if s['uuid'] != self.uuid]
        dealer_btn = round_state['dealer_btn']
        small_blind_pos = round_state['small_blind_pos']
        big_blind_pos = round_state['big_blind_pos']
        next_player = round_state['next_player']
        round_count = round_state['round_count']
        estimation = self.hole_card_est[(hole_card[0], hole_card[1])]

        self.features = get_street(street)
        self.features.extend([bank, stack, dealer_btn, small_blind_pos, big_blind_pos, next_player, round_count])
        self.features.extend(other_stacks)
        self.features.append(estimation)

        img_state = img_from_state(hole_card, round_state)
        img_state = process_img(img_state)
        action_num = self.sess.run(self.predict, feed_dict={self.scalar_input: [img_state],
                                                            self.features_input: [self.features]})[0]
        qs = self.sess.run(self.Q_out, feed_dict={self.scalar_input: [img_state],
                                                  self.features_input: [self.features]})[0]
        self._print(qs)
        action = get_action_by_num(action_num, valid_actions)

        return action

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self._print(['Hole:', hole_card])
        self.start_stack = [s['stack'] for s in seats if s['uuid'] == self.uuid][0]
        self._print(['Start stack:', self.start_stack])
        estimation = self.hole_card_est[(hole_card[0], hole_card[1])]
        self._print(['Estimation:', estimation])

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        end_stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]
        self._print(['End stack:', end_stack])


def gen_card_im(card):
    a = np.zeros((4, 13))
    s = suits.index(card.suit)
    r = ranks.index(card.rank)
    a[s, r] = 1
    return np.pad(a, ((6, 7), (2, 2)), 'constant', constant_values=0)


streep_map = {
    'preflop': 0,
    'flop': 1,
    'turn': 2,
    'river': 3
}


def get_street(s):
    val = [0, 0, 0, 0]
    val[streep_map[s]] = 1
    return val


def process_img(img):
    return np.reshape(img, [17 * 17 * 1])


class ExperienceBuffer():
    def __init__(self, buffer_size=5000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:len(self.buffer) + len(experience) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 7])


def get_action_by_num(action_num, valid_actions, is_train=True):
    if action_num == 0:
        action = valid_actions[0]['action']  # , valid_actions[0]['amount']
    elif action_num == 1:
        action = valid_actions[1]['action']  # , valid_actions[1]['amount']
    elif action_num == 2:
        if len(valid_actions) == 3:
            action = valid_actions[2]['action']  # , valid_actions[2]['amount']['min']
        else:
            action = valid_actions[0]['action']  # , valid_actions[1]['amount']

    # elif action_num == 3:
    #     action = valid_actions[2]['action']  # , valid_actions[2]['amount']['max']
    # elif action_num == 4:
    #     action = valid_actions[2]['action']  # , int(valid_actions[2]['amount']['max'] // 2)

    if not is_train:
        # print(action, amount)
        action = valid_actions[1]['action']  # , valid_actions[1]['amount']
    return action


def img_from_state(hole_card, round_state):
    imgs = np.zeros((8, 17, 17))
    for i, c in enumerate(hole_card):
        imgs[i] = gen_card_im(Card.from_str(c))

    for i, c in enumerate(round_state['community_card']):
        imgs[i + 2] = gen_card_im(Card.from_str(c))

    imgs[7] = imgs[:7].sum(axis=0)
    #     return imgs
    return np.swapaxes(imgs, 0, 2)[:, :, -1:]
