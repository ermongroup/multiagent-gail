import tensorflow as tf
import numpy as np
from rl.acktr.utils import Scheduler
from rl.acktr.utils import fc

disc_types = ['decentralized', 'centralized', 'single']


class Discriminator(object):
    def __init__(self, sess, ob_spaces, ac_spaces,
                 nstack, index, disc_type='decentralized', hidden_size=128, gp_coef=5,
                 lr_rate=5e-4, total_steps=50000, scope="discriminator"):
        self.lr = Scheduler(v=lr_rate, nvalues=total_steps * 20, schedule='linear')
        self.disc_type = disc_type
        if disc_type not in disc_types:
            assert False
        self.scope = scope
        self.index = index
        self.sess = sess
        ob_space = ob_spaces[index]
        ac_space = ac_spaces[index]
        self.ob_shape = ob_space.shape[0] * nstack
        nact = ac_space.n
        self.ac_shape = nact * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        self.all_ac_shape = sum([ac.n for ac in ac_spaces]) * nstack
        self.hidden_size = hidden_size

        if disc_type == 'decentralized':
            input_shape = self.all_ob_shape + self.ac_shape
        elif disc_type == 'centralized':
            input_shape = self.all_ob_shape + self.all_ac_shape
        elif disc_type == 'single':
            input_shape = self.all_ob_shape + self.all_ac_shape
        else:
            assert False

        self.g = tf.placeholder(tf.float32, (None, input_shape))
        self.e = tf.placeholder(tf.float32, (None, input_shape))
        self.lr_rate = tf.placeholder(tf.float32, ())

        num_outputs = len(ob_spaces) if disc_type == 'centralized' else 1
        self.bias = tf.get_variable(name=scope + '_bias', shape=(num_outputs,),
                                    initializer=tf.zeros_initializer, trainable=False)
        self.bias_ph = tf.placeholder(tf.float32, (num_outputs,))
        self.update_bias = tf.assign(self.bias, self.bias_ph * 0.01 + self.bias * 0.99)

        generator_logits = self.build_graph(self.g, num_outputs, reuse=False)
        expert_logits = self.build_graph(self.e, num_outputs, reuse=True)

        self.generator_loss = tf.reduce_mean(generator_logits, axis=0)
        self.expert_loss = tf.reduce_mean(expert_logits, axis=0)

        ddg = tf.gradients(generator_logits, [self.g])
        ddg = tf.sqrt(tf.reduce_sum(tf.square(ddg[0]), axis=1))
        self.ddg = tf.reduce_mean(tf.square(ddg - 1.))

        dde = tf.gradients(expert_logits, [self.e])
        dde = tf.sqrt(tf.reduce_sum(tf.square(dde[0]), axis=1))
        self.dde = tf.reduce_mean(tf.square(dde - 1.))

        epsilon = tf.random_uniform([], 0.0, 1.0)
        ge = self.g * epsilon + self.e * (1 - epsilon)
        gel = self.build_graph(ge, num_outputs, reuse=True)
        ddd = tf.gradients(gel, [ge])
        ddd = tf.norm(ddd, axis=1)
        self.ddd = tf.reduce_mean(tf.square(ddd - 1.))

        self.total_loss = self.generator_loss - self.expert_loss + gp_coef * self.ddd #(self.ddg + self.dde)
        self.reward_op = generator_logits

        self.var_list = self.get_trainable_variables()
        self.d_optim = tf.train.AdamOptimizer(self.lr_rate, beta1=0.5, beta2=0.9).minimize(self.total_loss, var_list=self.var_list)
        self.saver = tf.train.Saver(self.get_variables())

    def build_graph(self, x, num_outputs=1, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            p_h1 = fc(x, 'fc1', nh=self.hidden_size)
            p_h2 = fc(p_h1, 'fc2', nh=self.hidden_size)
            p_h3 = fc(p_h2, 'fc3', nh=self.hidden_size)
            logits = fc(p_h3, 'out', nh=num_outputs, act=lambda x: x)
            logits -= self.bias
        return logits

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, all_obs, acs):
        if len(all_obs.shape) == 1:
            all_obs = np.expand_dims(all_obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
        feed_dict = {self.g: np.concatenate([all_obs, acs], axis=1)}
        return self.sess.run(self.reward_op, feed_dict)

    def train(self, g_all_obs, g_acs, e_all_obs, e_acs):
        feed_dict = {self.g: np.concatenate([g_all_obs, g_acs], axis=1),
                     self.e: np.concatenate([e_all_obs, e_acs], axis=1), self.lr_rate: self.lr.value()}
        gl, el, _ = self.sess.run([self.generator_loss, self.expert_loss, self.d_optim], feed_dict)
        # self.sess.run(self.update_bias, feed_dict={self.bias_ph: (gl + el) / 2.0})
        return self.sess.run([self.generator_loss, self.expert_loss, self.ddg, self.dde], feed_dict)

    def restore(self, path):
        print('restoring from:' + path)
        self.saver.restore(self.sess, path)


