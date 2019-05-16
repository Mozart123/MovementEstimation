import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# Part of the code is from: https://towardsdatascience.com/deep-autoencoders-using-tensorflow-c68f075fd1a3

class Agent():
    def __init__(self, array_size = (200, 200, 2), lr = 0.01):
        self.array_size = array_size
        self.init_model(lr = lr)
    
    def step(self, new_screen, movement):
        out = self.train_model(new_screen, movement)
        
        return out
    
    def close_sess(self):
        self.sess.close()
    
    def predict(self, array):
        out = self.sess.run([self.output], feed_dict={self.x: array})
        return out
    
    def train_model(self, array, y):
        _, c, out = self.sess.run([self.optimizer, self.cost, self.output], feed_dict={self.x: array, self.y: y})
        
        #print("loss {}".format(c))
        return c
    
    def init_model(self, lr):
        input_size = self.array_size
        print("input_size: {}".format(input_size))
        self.x = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], input_size[2]], name='InputData')
        self.y = tf.placeholder(tf.float32, [None, 2], name='OutputData')
        
        self.MovementAgent(self.x, self.y, 'mov_agent', input_size[0], input_size[1])
        with tf.name_scope('opt'):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
            
############################################################## END OF CLASS #####################################################        
    def MovementAgent(self, x, y, name, shapex, shapey):
        training = True
        #tf.name_scope(name) as scope
        with tf.variable_scope(name):
            #input_ma = tf.reshape(x, shape=[-1, shapey, shapex, 2])
            x1, x2 = tf.split(x, [1, 1], 3)
            
            #Pad x1
            paddings = tf.constant([[0, 0], [4, 4,], [4, 4], [0, 0],])
            x1_padded = tf.pad(x1,paddings,mode='CONSTANT')
            x2_padded = []
            
            #Pad x2 in 25 ways
            for i in range(8):
                for j in range(8):
                    paddings = tf.constant([[0, 0], [i, 8-i,], [j, 8-j], [0, 0],])
                    x2_padded.append(tf.pad(x2,paddings,mode='CONSTANT'))
            
            x2_padded = tf.concat(x2_padded,axis=3,name='concat_x2')
            
            sub = tf.math.subtract(x2_padded, x1_padded, name='sub1')
            sub = tf.math.square(sub)
            
            #Part 1
            x = tf.layers.batch_normalization(x, training=training)
            x = conv2d(sub, name='c11', kshape=[7, 7, 64, 16], reuse = False)
            x = maxpool2d(x, name='p11')
            x = tf.layers.batch_normalization(x, training=training)
            x = conv2d(x, name='c21', kshape=[5, 5, 16, 32], reuse = False, dilations = [1, 2, 2, 1])
            x = maxpool2d(x, name='p21')
            x = tf.layers.batch_normalization(x, training=training)
            x = conv2d(x, name='c31', kshape=[5, 5, 32, 64], reuse = False, dilations = [1, 2, 2, 1])
            x = maxpool2d(x, name='p31')
            x = tf.layers.batch_normalization(x, training=training)
            x = conv2d(x, name='c41', kshape=[5, 5, 64, 128], reuse = False, dilations = [1, 2, 2, 1])
            x = maxpool2d(x, name='p41')
            
            x = tf.layers.flatten(x)#, shape=[-1, 52*52*40])
            # Last part
            #x = fullyConnected(x, name='fc11', output_size=40)
            #x = dropout(x, name='do11', keep_rate=0.99)
            x = tf.layers.batch_normalization(x, training=training)
            out = fullyConnected(x, name='fc21', output_size=100)

            self.output = fullyConnected(out, name='output', output_size=2, activation = 'softmax')

            with tf.name_scope('cost'):
                self.cost = tf.losses.softmax_cross_entropy(y, self.output)
                #self.cost = tf.reduce_mean(tf.square(tf.subtract(self.output, y)))



def conv2d(input, name, kshape, strides=[1, 1, 1, 1], reuse = False, dilations = None):
    with tf.variable_scope(name, reuse=reuse): #tf.name_scope(name) as scope
        W = tf.get_variable(name='w_'+name,
                            shape=kshape,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_' + name,
                            shape=[kshape[3]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        out = tf.nn.conv2d(input,W,strides=strides, padding='SAME', dilations = dilations)
        out = tf.nn.bias_add(out, b)
        out = tf.nn.relu(out)
        return out
# ---------------------------------
def deconv2d(input, name, kshape, n_outputs, strides=[1, 1]):
    with tf.name_scope(name):
        out = tf.contrib.layers.conv2d_transpose(input,
                                                 num_outputs= n_outputs,
                                                 kernel_size=kshape,
                                                 stride=strides,
                                                 padding='SAME',
                                                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                                 biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                 activation_fn=tf.nn.relu)
        return out
#   ---------------------------------
def maxpool2d(x,name,kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.name_scope(name):
        out = tf.nn.max_pool(x,
                             ksize=kshape, #size of window
                             strides=strides,
                             padding='SAME')
        return out
#   ---------------------------------
def upsample(input, name, factor=[2,2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
        return out
#   ---------------------------------
def fullyConnected(input, name, output_size, activation = 'relu', reuse = False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.get_variable(name='w_'+name,
                            shape=[input_size, output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_'+name,
                            shape=[output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        input = tf.reshape(input, [-1, input_size])
        
        out = None
        if activation == 'relu':
            out = tf.nn.relu(tf.add(tf.matmul(input, W), b))
        else:
            out = tf.nn.softmax(tf.add(tf.matmul(input, W), b))
        return out
#   ---------------------------------
def dropout(input, name, keep_rate):
    with tf.name_scope(name):
        out = tf.nn.dropout(input, rate = 1 - keep_rate)
        return out        
