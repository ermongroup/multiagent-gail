import os.path as osp
import random
import time

import joblib
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from rl.acktr.utils import Scheduler, find_trainable_variables, discount_with_dones
from rl.acktr.utils import mse

from rl import logger
from rl.acktr import kfac
from rl.common import set_global_seeds, explained_variance
# from irl.mack.discriminator import Discriminator
from irl.mack.kfac_discriminator_wgan import Discriminator
from irl.dataset import Dset

from tqdm import tqdm


class Model(object):
    def __init__(self, policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=2, nsteps=200,
                 nstack=1, ent_coef=0.00, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear', identical=None):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        nbatch = nenvs * nsteps
        self.num_agents = num_agents = len(ob_space)
        if identical is None:
            identical = [False for _ in range(self.num_agents)]

        scale = [1 for _ in range(num_agents)]
        pointer = [i for i in range(num_agents)]
        h = 0
        for k in range(num_agents):
            if identical[k]:
                scale[h] += 1
            else:
                pointer[h] = k
                h = k
        pointer[h] = num_agents

        A, ADV, R, PG_LR = [], [], [], []
        for k in range(num_agents):
            if identical[k]:
                A.append(A[-1])
                ADV.append(ADV[-1])
                R.append(R[-1])
                PG_LR.append(PG_LR[-1])
            else:
                A.append(tf.placeholder(tf.float32, [nbatch * scale[k], ac_space[k].shape[0]]))
                ADV.append(tf.placeholder(tf.float32, [nbatch * scale[k]]))
                R.append(tf.placeholder(tf.float32, [nbatch * scale[k]]))
                PG_LR.append(tf.placeholder(tf.float32, []))

        # A = [tf.placeholder(tf.int32, [nbatch]) for _ in range(num_agents)]
        # ADV = [tf.placeholder(tf.float32, [nbatch]) for _ in range(num_agents)]
        # R = [tf.placeholder(tf.float32, [nbatch]) for _ in range(num_agents)]
        # PG_LR = [tf.placeholder(tf.float32, []) for _ in range(num_agents)]
        # VF_LR = [tf.placeholder(tf.float32, []) for _ in range(num_agents)]
        pg_loss, entropy, vf_loss, train_loss = [], [], [], []
        self.model = step_model = []
        self.model2 = train_model = []
        self.pg_fisher = pg_fisher_loss = []
        sample_net = []
        self.vf_fisher = vf_fisher_loss = []
        self.joint_fisher = joint_fisher_loss = []
        self.lld = lld = []

        for k in range(num_agents):
            if identical[k]:
                step_model.append(step_model[-1])
                train_model.append(train_model[-1])
            else:
                step_model.append(policy(sess, ob_space[k], ac_space[k], ob_space, ac_space,
                                         nenvs, 1, nstack, reuse=False, name='%d' % k))
                train_model.append(policy(sess, ob_space[k], ac_space[k], ob_space, ac_space,
                                          nenvs * scale[k], nsteps, nstack, reuse=True, name='%d' % k))

            stats = train_model[k].mean_std
            ac = ac_space[k].shape[0]
            logpac = 0.5 * tf.reduce_sum(tf.square((A[k] - stats[:, :ac]) / stats[:, ac:]), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(A[k])[-1]) \
               + tf.reduce_sum(tf.log(stats[:, ac:]), axis=-1)
            lld.append(tf.reduce_mean(logpac))

            pg_loss.append(tf.reduce_mean(ADV[k] * logpac))
            entropy.append(tf.reduce_sum(
                train_model[k].logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1))
            pg_loss[k] = pg_loss[k] - ent_coef * entropy[k]
            vf_loss.append(tf.reduce_mean(mse(tf.squeeze(train_model[k].vf), R[k])))
            train_loss.append(pg_loss[k] + vf_coef * vf_loss[k])

            pg_fisher_loss.append(-tf.reduce_mean(logpac))
            sample_net.append(train_model[k].vf + tf.random_normal(tf.shape(train_model[k].vf)))
            vf_fisher_loss.append(-vf_fisher_coef * tf.reduce_mean(
                tf.pow(train_model[k].vf - tf.stop_gradient(sample_net[k]), 2)))
            joint_fisher_loss.append(pg_fisher_loss[k] + vf_fisher_loss[k])

        self.policy_params = []
        self.value_params = []
        self.sigma_params = []

        for k in range(num_agents):
            if identical[k]:
                self.policy_params.append(self.policy_params[-1])
                self.value_params.append(self.value_params[-1])
                self.sigma_params.append(self.sigma_params[-1])
            else:
                self.policy_params.append(find_trainable_variables("policy_%d" % k))
                self.value_params.append(find_trainable_variables("value_%d" % k))
                self.sigma_params.append(find_trainable_variables("sigma_%d" % k))
        self.params = params = [a + b for a, b in zip(self.policy_params, self.value_params)]
        params_flat = []
        for k in range(num_agents):
            params_flat.extend(params[k])

        self.grads_check = grads = [
            tf.gradients(train_loss[k], params[k]) for k in range(num_agents)
        ]
        clone_grads = [
            tf.gradients(lld[k], params[k]) for k in range(num_agents)
        ]

        self.optim = optim = []
        self.vf_optim = vf_optim = []
        # self.adam = adam = []
        self.clones = clones = []
        # self.clone_adam = clone_adam = []
        update_stats_op = []
        train_op, vf_op, clone_op, q_runner = [], [], [], []

        for k in range(num_agents):
            if identical[k]:
                optim.append(optim[-1])
                train_op.append(train_op[-1])
                q_runner.append(q_runner[-1])
                clones.append(clones[-1])
                clone_op.append(clone_op[-1])
            else:
                with tf.variable_scope('optim_%d' % k):
                    optim.append(kfac.KfacOptimizer(
                        learning_rate=PG_LR[k], clip_kl=kfac_clip,
                        momentum=0.9, kfac_update=1, epsilon=0.01,
                        stats_decay=0.99, async=0, cold_iter=10,
                        max_grad_norm=max_grad_norm)
                    )
                    train_op_, q_runner_ = optim[k].minimize(pg_loss[k], pg_fisher_loss[k], var_list=self.policy_params[k])
                    q_runner.append(q_runner_)

                with tf.variable_scope('vf_%d' % k):
                    vf_optim.append(kfac.KfacOptimizer(
                        learning_rate=PG_LR[k], clip_kl=kfac_clip,
                        momentum=0.9, kfac_update=1, epsilon=0.01,
                        stats_decay=0.99, async=0, cold_iter=10,
                        max_grad_norm=max_grad_norm)
                    )
                    vf_op_, q_runner_ = vf_optim[k].minimize(vf_loss[k], vf_fisher_loss[k], var_list=self.value_params[k])
                    vf_op.append(vf_op_)
                    # update_stats_op.append(optim[k].compute_and_apply_stats(joint_fisher_loss, var_list=params[k]))
                    # train_op_, q_runner_ = optim[k].apply_gradients(list(zip(grads[k], params[k])))
                    train_op.append(train_op_)
                    # adam.append(tf.train.AdamOptimizer(learning_rate=0.001).minimize(
                    q_runner.append(q_runner_)
                    #     loss=train_loss[k], var_list=self.sigma_params[k]))

                with tf.variable_scope('clone_%d' % k):
                    clones.append(kfac.KfacOptimizer(
                        learning_rate=PG_LR[k], clip_kl=kfac_clip,
                        momentum=0.9, kfac_update=1, epsilon=0.01,
                        stats_decay=0.99, async=0, cold_iter=10,
                        max_grad_norm=max_grad_norm)
                    )
                    update_stats_op.append(clones[k].compute_and_apply_stats(
                        pg_fisher_loss[k], var_list=self.policy_params[k]))
                    clone_op_, q_runner_ = clones[k].apply_gradients(list(zip(clone_grads[k], self.policy_params[k])))
                    clone_op.append(clone_op_)
                    # clone_adam.append(tf.train.AdamOptimizer(learning_rate=0.001).minimize(
                    #     loss=lld[k], var_list=self.sigma_params[k]))

        update_stats_op = tf.group(*update_stats_op)
        train_ops = train_op
        vf_ops = vf_op
        clone_ops = clone_op
        train_op = tf.group(*train_op)
        clone_op = tf.group(*clone_op)

        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        self.clone_lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values, value_op_only=False):
            advs = [rewards[k] - values[k] for k in range(num_agents)]
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            ob = np.concatenate(obs, axis=1)

            td_map = {}
            for k in range(num_agents):
                if identical[k]:
                    continue
                new_map = {}
                if num_agents > 1:
                    action_v = []
                    for j in range(k, pointer[k]):
                        action_v.append(np.concatenate([actions[i] for i in range(num_agents) if i != k], axis=1))
                    action_v = np.concatenate(action_v, axis=0)
                    new_map.update({train_model[k].A_v: action_v})
                    td_map.update({train_model[k].A_v: action_v})

                new_map.update({
                    train_model[k].X: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    train_model[k].X_v: np.concatenate([ob.copy() for j in range(k, pointer[k])], axis=0),
                    A[k]: np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0),
                    ADV[k]: np.concatenate([advs[j] for j in range(k, pointer[k])], axis=0),
                    R[k]: np.concatenate([rewards[j] for j in range(k, pointer[k])], axis=0),
                    PG_LR[k]: cur_lr / float(scale[k])
                })
                if k == 0:
                    sess.run(vf_op[k], feed_dict=new_map)
                    sess.run(train_ops[k], feed_dict=new_map)
                td_map.update(new_map)

                if states[k] != []:
                    td_map[train_model[k].S] = states
                    td_map[train_model[k].M] = masks

            policy_loss, value_loss, policy_entropy = sess.run(
                [pg_loss, vf_loss, entropy],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def clone(obs, actions):
            td_map = {}
            cur_lr = self.clone_lr.value()
            for k in range(num_agents):
                if identical[k]:
                    continue
                new_map = {}
                new_map.update({
                    train_model[k].X: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
                    A[k]: np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0),
                    PG_LR[k]: cur_lr / float(scale[k])
                })
                sess.run(clone_ops[k], feed_dict=new_map)
                # sess.run(clone_adam[k], feed_dict=new_map)
                td_map.update(new_map)
            lld_loss = sess.run([lld], td_map)
            return lld_loss

        def save(save_path):
            ps = sess.run(params_flat)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params_flat, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.clone = clone
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model

        def step(ob, av, *_args, **_kwargs):
            a, v, s = [], [], []
            obs = np.concatenate(ob, axis=1)
            for k in range(num_agents):
                a_v = np.concatenate([av[i] for i in range(num_agents) if i != k], axis=1)
                a_, v_, s_ = step_model[k].step(ob[k], obs, a_v)
                a.append(a_)
                v.append(v_)
                s.append(s_)
            return a, v, s

        self.step = step

        def value(obs, av):
            v = []
            ob = np.concatenate(obs, axis=1)
            for k in range(num_agents):
                a_v = np.concatenate([av[i] for i in range(num_agents) if i != k], axis=1)
                v_ = step_model[k].value(ob, a_v)
                v.append(v_)
            return v

        self.value = value
        self.initial_state = [step_model[k].initial_state for k in range(num_agents)]


class Runner(object):
    def __init__(self, env, model, discriminator, nsteps, nstack, gamma, lam, disc_type):
        self.env = env
        self.model = model
        self.discriminator = discriminator
        self.disc_type = disc_type
        self.num_agents = len(env.observation_space.spaces)
        self.nenv = nenv = env.num_envs
        self.batch_ob_shape = [
            (nenv * nsteps, nstack * env.observation_space.spaces[k].shape[0]) for k in range(self.num_agents)
        ]
        self.obs = [
            np.zeros((nenv, nstack * env.observation_space.spaces[k].shape[0])) for k in range(self.num_agents)
        ]
        self.actions = [np.zeros((nenv, env.action_space.spaces[k].shape[0])) for k in range(self.num_agents)]
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [np.array([False for _ in range(nenv)]) for k in range(self.num_agents)]

    def update_obs(self, obs):
        # TODO: Potentially useful for stacking.
        self.obs = obs
        # for k in range(self.num_agents):
        #     ob = np.roll(self.obs[k], shift=-1, axis=1)
        #     ob[:, -1] = obs[:, 0]
        #     self.obs[k] = ob

        # self.obs = [np.roll(ob, shift=-1, axis=3) for ob in self.obs]
        # self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def run(self):
        mb_obs = [[] for _ in range(self.num_agents)]
        mb_true_rewards = [[] for _ in range(self.num_agents)]
        mb_rewards = [[] for _ in range(self.num_agents)]
        mb_actions = [[] for _ in range(self.num_agents)]
        mb_values = [[] for _ in range(self.num_agents)]
        mb_dones = [[] for _ in range(self.num_agents)]
        mb_masks = [[] for _ in range(self.num_agents)]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.actions)
            if self.disc_type == 'decentralized':
                rewards = [np.reshape(self.discriminator[k].get_reward(self.obs[k], self.actions[k]), [-1])
                    for k in range(self.num_agents)]
            elif self.disc_type == 'centralized':
                mul = [self.actions[k] for k in range(self.num_agents)]
                rewards = self.discriminator.get_reward(np.concatenate(self.obs, axis=1), np.concatenate(mul, axis=1))
                rewards = rewards.swapaxes(1, 0)
            elif self.disc_type == 'single':
                mul = [self.actions[k] for k in range(self.num_agents)]
                rewards = self.discriminator.get_reward(np.concatenate(self.obs, axis=1), np.concatenate(mul, axis=1))
                rewards = np.repeat(rewards, self.num_agents).reshape(len(rewards), self.num_agents)
                rewards = rewards.swapaxes(1, 0)
            else:
                assert False

            self.actions = actions
            for k in range(self.num_agents):
                mb_obs[k].append(np.copy(self.obs[k]))
                mb_actions[k].append(actions[k])
                mb_values[k].append(values[k])
                mb_dones[k].append(self.dones[k])
                mb_rewards[k].append(rewards[k])
            actions_list = []
            for i in range(self.nenv):
                actions_list.append([actions[k][i] for k in range(self.num_agents)])
            obs, true_rewards, dones, info = self.env.step(actions_list)
            self.states = states
            self.dones = dones
            for k in range(self.num_agents):
                for ni, done in enumerate(dones[k]):
                    if done:
                        self.obs[k][ni] = self.obs[k][ni] * 0.0
            self.update_obs(obs)
            for k in range(self.num_agents):
                mb_true_rewards[k].append(true_rewards[k])
        for k in range(self.num_agents):
            mb_dones[k].append(self.dones[k])

        # batch of steps to batch of rollouts
        # import ipdb; ipdb.set_trace()
        for k in range(self.num_agents):
            mb_obs[k] = np.asarray(mb_obs[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])
            mb_true_rewards[k] = np.asarray(mb_true_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_rewards[k] = mb_true_rewards[k] #np.asarray(mb_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_actions[k] = np.asarray(mb_actions[k], dtype=np.int32).swapaxes(1, 0)
            mb_values[k] = np.asarray(mb_values[k], dtype=np.float32).swapaxes(1, 0)
            mb_dones[k] = np.asarray(mb_dones[k], dtype=np.bool).swapaxes(1, 0)
            mb_masks[k] = mb_dones[k][:, :-1]
            mb_dones[k] = mb_dones[k][:, 1:]

        # GAE
        last_values = self.model.value(self.obs, self.actions)  # self.states, self.dones)
        mb_advs = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        mb_returns = [[] for _ in range(self.num_agents)]

        lastgaelam = 0.0
        for k in range(self.num_agents):
            for t in reversed(range(self.nsteps)):
                if t == self.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones[k]
                    nextvalues = last_values[k]
                else:
                    nextnonterminal = 1.0 - mb_dones[k][:, t + 1]
                    nextvalues = mb_values[k][:, t + 1]
                delta = mb_rewards[k][:, t] + self.gamma * nextvalues * nextnonterminal - mb_values[k][:, t]
                mb_advs[k][:, t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            mb_returns[k] = mb_advs[k] + mb_values[k]
            mb_returns[k] = mb_returns[k].flatten()
            mb_masks[k] = mb_masks[k].flatten()
            mb_values[k] = mb_values[k].flatten()
            mb_actions[k] = np.reshape(mb_actions[k], [-1, mb_actions[k].shape[-1]])

        # discount
        # mb_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        # last_values = self.model.value(self.obs, self.actions)
        # # discount/bootstrap off value fn
        # for k in range(self.num_agents):
        #     for n, (rewards, dones, value) in enumerate(zip(mb_rewards[k], mb_dones[k], last_values[k].tolist())):
        #         rewards = rewards.tolist()
        #         dones = dones.tolist()
        #         if dones[-1] == 0:
        #             rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
        #         else:
        #             rewards = discount_with_dones(rewards, dones, self.gamma)
        #         mb_returns[k][n] = rewards
        #
        # for k in range(self.num_agents):
        #     mb_returns[k] = mb_returns[k].flatten()
        #     mb_masks[k] = mb_masks[k].flatten()
        #     mb_values[k] = mb_values[k].flatten()
        #     mb_actions[k] = np.reshape(mb_actions[k], [-1, mb_actions[k].shape[-1]])

        mh_actions = [mb_actions[k] for k in range(self.num_agents)]
        mb_all_obs = np.concatenate(mb_obs, axis=1)
        mh_all_actions = np.concatenate(mh_actions, axis=1)
        return mb_obs, mb_states, mb_returns, mb_masks, mb_actions,\
               mb_values, mb_all_obs, mh_actions, mh_all_actions, mb_rewards, mb_true_rewards


def learn(policy, expert, env, seed, total_timesteps=int(40e6), gamma=0.995, lam=0.95, log_interval=1, nprocs=32,
          nsteps=20, nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=100, lrschedule='linear', dis_lr=0.001, disc_type='decentralized',
          bc_iters=500, identical=None):
    tf.reset_default_graph()
    set_global_seeds(seed)
    buffer = None

    nenvs = env.num_envs
    ob_space = env.observation_space.spaces
    ac_space = env.action_space.spaces
    num_agents = len(ob_space)
    make_model = lambda: Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps=nsteps,
                               nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
                               lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                               lrschedule=lrschedule, identical=identical)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if disc_type == 'decentralized':
        discriminator = [
            Discriminator(model.sess, ob_space, ac_space, nstack, k, disc_type=disc_type,
                          scope="Discriminator_%d" % k, # gp_coef=gp_coef,
                          total_steps=total_timesteps // (nprocs * nsteps),
                          lr_rate=dis_lr) for k in range(num_agents)
        ]
    elif disc_type == 'centralized':
        discriminator = Discriminator(model.sess, ob_space, ac_space, nstack, 0, disc_type=disc_type,
                                      total_steps=total_timesteps // (nprocs * nsteps),
                                      scope='discriminator', # gp_coef=gp_coef,
                                      lr_rate=dis_lr)
    elif disc_type == 'single':
        discriminator = Discriminator(model.sess, ob_space, ac_space, nstack, 0, disc_type=disc_type,
                                      total_steps=total_timesteps // (nprocs * nsteps),
                                      scope='discriminator', # gp_coef=gp_coef,
                                      lr_rate=dis_lr)
    else:
        assert False
    tf.global_variables_initializer().run(session=model.sess)
    runner = Runner(env, model, discriminator, nsteps=nsteps, nstack=nstack, gamma=gamma, lam=lam, disc_type=disc_type)
    nbatch = nenvs * nsteps
    tstart = time.time()
    coord = tf.train.Coordinator()
    # enqueue_threads = [q_runner.create_threads(model.sess, coord=coord, start=True) for q_runner in model.q_runner]
        # print(model.sess.run(model.train_model[0].std))
        # print(lld_loss)
    for _ in tqdm(range(bc_iters)):
        e_obs, e_actions, _, _ = expert.get_next_batch(nenvs * nsteps)
        lld_loss = model.clone(e_obs, e_actions)

    for update in range(1, total_timesteps // nbatch + 1):
        obs, states, rewards, masks, actions, values, all_obs,\
        mh_actions, mh_all_actions, mh_rewards, mh_true_rewards = runner.run()

        d_iters = 1
        g_loss, e_loss = np.zeros((num_agents, d_iters)), np.zeros((num_agents, d_iters))
        idx = 0
        idxs = np.arange(len(all_obs))
        random.shuffle(idxs)
        all_obs = all_obs[idxs]
        mh_actions = [mh_actions[k][idxs] for k in range(num_agents)]
        mh_obs = [obs[k][idxs] for k in range(num_agents)]
        mh_all_actions = mh_all_actions[idxs]

        if buffer:
            buffer.update(mh_actions, obs, all_obs, values)
        else:
            buffer = Dset(mh_actions, obs, all_obs, values, randomize=True, num_agents=num_agents)

        d_minibatch = nenvs * nsteps
            # print(lld_loss)

        for d_iter in range(0): # range(d_iters):
            e_obs, e_actions, e_all_obs, _ = expert.get_next_batch(d_minibatch)
            acts, bobs, aobs, _ = buffer.get_next_batch(batch_size=d_minibatch)
            if disc_type == 'decentralized':
                for k in range(num_agents):
                    g_loss[k, d_iter], e_loss[k, d_iter], _, _ = discriminator[k].train(
                        bobs[k],
                        acts[k],
                        e_obs[k],
                        e_actions[k]
                    )
            elif disc_type == 'centralized':
                g_loss_t, e_loss_t, _, _ = discriminator.train(
                    aobs,
                    np.concatenate(acts, axis=1),
                    e_all_obs, np.concatenate(e_actions, axis=1))
                g_loss[:, d_iter] = g_loss_t
                e_loss[:, d_iter] = e_loss_t
            elif disc_type == 'single':
                g_loss_t, e_loss_t, _, _ = discriminator.train(
                    aobs,
                    np.concatenate(acts, axis=1),
                    e_all_obs, np.concatenate(e_actions, axis=1))
                g_loss[:, d_iter] = g_loss_t
                e_loss[:, d_iter] = e_loss_t
            else:
                assert False
            idx += 1

        # if update < 10:
        #     for _ in range(bc_iters // 10):
        #         e_obs, e_actions, _, _ = expert.get_next_batch(nenvs * nsteps)
        #         lld_loss = model.clone(e_obs, e_actions)
        # else:
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        model.old_obs = obs
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = [explained_variance(values[k], rewards[k]) for k in range(model.num_agents)]
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)

            for k in range(model.num_agents):
                logger.record_tabular("explained_variance %d" % k, float(ev[k]))
                # if update > 10:
                logger.record_tabular("policy_entropy %d" % k, float(policy_entropy[k]))
                logger.record_tabular("policy_loss %d" % k, float(policy_loss[k]))
                logger.record_tabular("value_loss %d" % k, float(value_loss[k]))
                logger.record_tabular('pearson %d' % k, float(
                    pearsonr(mh_rewards[k].flatten(), mh_true_rewards[k].flatten())[0]))
                logger.record_tabular('spearman %d' % k, float(
                    spearmanr(mh_rewards[k].flatten(), mh_true_rewards[k].flatten())[0]))
            g_loss_m = np.mean(g_loss, axis=1)
            e_loss_m = np.mean(e_loss, axis=1)
            # g_loss_gp_m = np.mean(g_loss_gp, axis=1)
            # e_loss_gp_m = np.mean(e_loss_gp, axis=1)
            for k in range(num_agents):
                logger.record_tabular("g_loss %d" % k, g_loss_m[k])
                logger.record_tabular("e_loss %d" % k, e_loss_m[k])
                # logger.record_tabular("g_loss_gp %d" % k, g_loss_gp_m[k])
                # logger.record_tabular("e_loss_gp %d" % k, e_loss_gp_m[k])

            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'm_%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
            if disc_type == 'decentralized':
                for k in range(num_agents):
                    savepath = osp.join(logger.get_dir(), 'd_%d_%.5i' % (k, update))
                    discriminator[k].save(savepath)
            elif disc_type == 'centralized':
                savepath = osp.join(logger.get_dir(), 'd_%.5i' % update)
                discriminator.save(savepath)
            elif disc_type == 'single':
                savepath = osp.join(logger.get_dir(), 'd_%.5i' % update)
                discriminator.save(savepath)
    coord.request_stop()
    # coord.join(enqueue_threads)
    env.close()
