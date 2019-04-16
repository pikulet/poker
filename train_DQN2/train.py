import numpy as np
from pypokerengine.utils.card_utils import Card
from MyEmulator import MyEmulator
from DeepQNetworkPlayer import DeepQNetworkPlayer
import tensorflow as tf
import random, os, sys
import PlayerModels as pm
from DeepQNetworkPlayer1 import Group47Player

sys.path.insert(0, './')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

suits = list(Card.SUIT_MAP.keys())
ranks = list(Card.RANK_MAP.keys())


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
        action = valid_actions[0]['action']
    elif action_num == 1:
        action = valid_actions[1]['action']
    elif action_num == 2:
        if len(valid_actions) == 3:
            action = valid_actions[2]['action']
        else:
            action = valid_actions[0]['action']

    if not is_train:
        action = valid_actions[1]['action']
    return action


def img_from_state(hole_card, round_state):
    imgs = np.zeros((8, 17, 17))
    for i, c in enumerate(hole_card):
        imgs[i] = gen_card_im(Card.from_str(c))
    for i, c in enumerate(round_state['community_card']):
        imgs[i + 2] = gen_card_im(Card.from_str(c))
    imgs[7] = imgs[:7].sum(axis=0)
    return np.swapaxes(imgs, 0, 2)[:, :, -1:]


def update_target_graph(tf_vars, tau):
    total_vars = len(tf_vars)
    ops = []
    for i, var in enumerate(tf_vars[0:total_vars // 2]):
        ops.append(tf_vars[i + total_vars // 2].assign((var.value() * tau) +
                                                       (tf_vars[i + total_vars // 2].value() * (1 - tau))))
    return ops


def update_target(ops, sess):
    for op in ops:
        sess.run(op)


batch_size = 128
update_freq = 50  # how often to update model
y = 0.99  # discount
start_E = 1  # starting chance of random action
end_E = 0.2  # final chance of random action
annealings_steps = 100000  # how many steps to reduce start_E to end_E
num_episodes = 5000
pre_train_steps = 5000  # how many steps of random action before training begin
load_model = False
path = './checkpoint/'
h_size = 128
tau = 0.01  # rate to update target network toward primary network
is_dueling = True  # whether or not to use dueling architecture

emul = MyEmulator()
emul.set_game_rule(2, 1000, 20, 0)
my_uuid = '2'
players_info = {
    "1": {"name": "f1", "stack": 10000},
    "2": {"name": "f2", "stack": 10000},
}


def init_emul(my_uuid_):
    global my_uuid
    my_uuid = my_uuid_
    emul.register_player("1", Group47Player())
    emul.register_player("2", pm.HeuristicPlayer())


tf.reset_default_graph()
main_wp = DeepQNetworkPlayer(h_size, is_double=True)
target_wp = DeepQNetworkPlayer(h_size, is_main=False, is_double=True)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=3)
trainables = tf.trainable_variables()
target_ops = update_target_graph(trainables, tau)
my_buffer = ExperienceBuffer()

e = start_E
step_drop = (start_E - end_E) / annealings_steps

j_list = []
r_list = []
action_list = []
total_steps = 0

with tf.Session() as sess:
    sess.run(init)
    if load_model:
        print('Loading model')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    for i in range(num_episodes):
        episode_buffer = ExperienceBuffer()
        init_emul(str(np.random.randint(1, 3)))

        initial_state = emul.generate_initial_game_state(players_info)
        msgs = []
        game_state, events = emul.start_new_round(initial_state)
        is_last_round = False
        r_all = 0
        j = 0

        last_img_state = None
        last_features = None
        last_action_num = None

        while not is_last_round:
            j += 1
            a = emul.run_until_my_next_action(game_state, my_uuid, msgs)

            if len(a) == 4:
                game_state, valid_actions, hole_card, round_state = a
                img_state = img_from_state(hole_card, round_state)
                img_state = process_img(img_state)

                street = round_state['street']
                bank = round_state['pot']['main']['amount']
                stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == my_uuid][0]
                other_stacks = [s['stack'] for s in round_state['seats'] if s['uuid'] != my_uuid]
                dealer_btn = round_state['dealer_btn']
                small_blind_pos = round_state['small_blind_pos']
                big_blind_pos = round_state['big_blind_pos']
                next_player = round_state['next_player']
                round_count = round_state['round_count']
                estimation = main_wp.card_pairs_prob[(hole_card[0], hole_card[1])]

                features = get_street(street)
                features.extend([bank, stack, dealer_btn, small_blind_pos, big_blind_pos, next_player, round_count])
                features.extend(other_stacks)
                features.append(estimation)

                # add to buffer last hand
                if last_img_state is not None:
                    episode_buffer.add(np.reshape(np.array([last_img_state, last_features, last_action_num,
                                                            0, img_state, features, 0]), [1, 7]))

                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    action_num = np.random.randint(0, main_wp.total_num_actions)
                else:
                    action_num = sess.run(main_wp.predict, feed_dict={main_wp.scalar_input: [img_state],
                                                                      main_wp.features_input: [features]})[0]

                action_list.append(action_num)
                action = get_action_by_num(action_num, valid_actions)

                game_state, msgs = emul.apply_my_action(game_state, action)
                total_steps += 1

                last_img_state = img_state.copy()
                last_features = features.copy()
                last_action_num = action_num

                if total_steps > pre_train_steps:
                    if e > end_E:
                        e -= step_drop

                    if total_steps % (update_freq) == 0:
                        train_batch = my_buffer.sample(batch_size)

                        Q1 = sess.run(main_wp.predict,
                                      feed_dict={main_wp.scalar_input: np.vstack(train_batch[:, 4]),
                                                 main_wp.features_input: np.vstack(train_batch[:, 5])})
                        Q1_ = sess.run(main_wp.Q_out,
                                       feed_dict={main_wp.scalar_input: np.vstack(train_batch[:, 4]),
                                                  main_wp.features_input: np.vstack(train_batch[:, 5])})

                        Q2 = sess.run(target_wp.Q_out,
                                      feed_dict={target_wp.scalar_input: np.vstack(train_batch[:, 4]),
                                                 target_wp.features_input: np.vstack(train_batch[:, 5])})
                        end_multiplier = -(train_batch[:, 6] - 1)
                        double_Q = Q2[range(batch_size), Q1]
                        double_Q_ = Q1_[range(batch_size), Q1]

                        if is_dueling:
                            target_Q = train_batch[:, 3] + (y * double_Q * end_multiplier)
                        else:
                            target_Q = train_batch[:, 3] + (y * double_Q_ * end_multiplier)

                        _, er, g, v = sess.run([main_wp.update_model,
                                                main_wp.loss, main_wp.grad_norms, main_wp.var_norms],
                                               feed_dict={
                                                   main_wp.scalar_input: np.vstack(train_batch[:, 0]),
                                                   main_wp.features_input: np.vstack(train_batch[:, 1]),
                                                   main_wp.target_Q: target_Q,
                                                   main_wp.actions: train_batch[:, 2]
                                               })
                        update_target(target_ops, sess)

                        r = np.mean(r_list[-2:])
                        j = np.mean(j_list[-2:])
                        q1 = double_Q_[0]
                        q2 = double_Q[0]
                        al = np.mean(action_list[-10:])

            else:
                game_state, reward = a
                if reward >= 0:
                    reward = np.log(1 + reward)
                else:
                    reward = -np.log(1 - reward)
                r_all += reward

                if last_img_state is not None:
                    episode_buffer.add(np.reshape(np.array([last_img_state, last_features, last_action_num,
                                                            reward, last_img_state, last_features, 1]), [1, 7]))

                is_last_round = emul._is_last_round(game_state, emul.game_rule)
                game_state, events = emul.start_new_round(game_state)

                last_img_state = None
                last_action_num = None

        my_buffer.add(episode_buffer.buffer)
        r_list.append(r_all)
        j_list.append(j)

        if i % 1000 == 0:
            saver.save(sess, path + '/model_' + str(i) + '.ckpt')
            saver.save(sess, path, i)

            print('Saved model')
        if i % 100 == 0:
            print(i, total_steps, np.mean(r_list[-10:]), e, np.median(action_list[-200:]))
    saver.save(sess, path + '/model_' + str(i) + '.ckpt')
print('Mean reward: {}'.format(sum(r_list) / num_episodes))
