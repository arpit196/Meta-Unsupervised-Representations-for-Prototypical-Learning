import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import keras
from tensorflow.keras.models import load_model
from keras_multi_head import MultiHead

def calc_euclidian_dists(x, y):
    """
    Calculate euclidian distance between two 3D tensors.
    Args:
        x (tf.Tensor):
        y (tf.Tensor):
    Returns (tf.Tensor): 2-dim tensor with distances.
    """
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_mean(tf.math.pow(x - y, 2), 2)

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(SelfAttention, self).__init__()
        self.gamma = tf.Variable(tf.random.truncated_normal([1]),
                             name="gamma")
    
    
    def call(self, inputs):
        f = Conv2D(filters=16, kernel_size=3, padding='same')(inputs) # [bs, h, w, c']
        g = Conv2D(filters=16, kernel_size=3, padding='same')(inputs) # [bs, h, w, c']
        h = Conv2D(filters=16, kernel_size=3, padding='same')(inputs) # [bs, h, w, c]

            # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
        
        o = tf.reshape(o, shape=inputs.shape) # [bs, h, w, C]
        o = Conv2D(filters=16, kernel_size=3, padding='same')(o)
        
        x = self.gamma * o + x
        return x

class Prototypical(Model):
    """
    Implemenation of Prototypical Network.
    """
    
    def __init__(self, n_support, n_query, w, h, c):
        """
        Args:
            n_support (int): number of support examples.
            n_query (int): number of query examples.
            w (int): image width .
            h (int): image height.
            c (int): number of channels.
        """
        super(Prototypical, self).__init__()
        self.w, self.h, self.c = w, h, c
        #self.W = tf.Variable(tf.random.truncated_normal([3136]),
        #                     name="W")
        #self.W = tf.Variable(tf.random.truncated_normal([19]),
        #                      name="W")

        # Encoder as ResNet like CNN with 4 blocks
        '''
        self.meta_enc1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)), Flatten()]
        )'''
        
        self.l1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')
        self.l2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')
        self.l3=  tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')
        self.l4=  tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')
        self.l5=  tf.keras.layers.BatchNormalization()
        self.l6=  tf.keras.layers.ReLU()
        self.l7=  tf.keras.layers.MaxPool2D((2, 2))

        self.l8=  tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')
        self.l9=  tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')
        self.l10 = SelfAttention()
        
        self.l11=    tf.keras.layers.BatchNormalization()
        self.l12=    tf.keras.layers.ReLU()
        self.l13=    tf.keras.layers.MaxPool2D((2, 2))
        self.l14 =  Flatten()
        
        self.encoder = tf.keras.Sequential()
        self.encoder.add(self.l1)
        self.encoder.add(self.l2)
        self.encoder.add(self.l3)
        self.encoder.add(self.l4)
        self.encoder.add(self.l5)
        self.encoder.add(self.l6)
        self.encoder.add(self.l7)
        self.encoder.add(self.l8)
        self.encoder.add(self.l9)
        self.encoder.add(self.l10)
        
        self.encoder.add(self.l11)
        self.encoder.add(self.l12)
        self.encoder.add(self.l13)
        self.encoder.add(self.l14)
        
        #self.pro_encoder = tf.keras.Sequential()
        
        '''
        self.meta_encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)), Flatten(), Dense(128)]
        )
        '''
    '''    
    def proto_enc(self, inputs):
        pro_encoder = self.W[0]*self.l1(inputs)
        #pro_encoder = tf.keras.layers.multiply([self.W[0],pro_encoder])
        pro_encoder = self.W[1]*self.l2(pro_encoder)
        #pro_encoder = tf.keras.layers.multiply([self.W[1],pro_encoder])
        pro_encoder = self.W[2]*self.l3(pro_encoder)
        #pro_encoder = tf.keras.layers.multiply([self.W[2],pro_encoder])
        pro_encoder = self.W[3]*self.l4(pro_encoder)
        #pro_encoder = tf.keras.layers.multiply([self.W[3],pro_encoder])
        pro_encoder = self.W[4]*self.l5(pro_encoder)
        #pro_encoder = tf.keras.layers.multiply([self.W[4],pro_encoder])
        pro_encoder = self.W[5]*self.l6(pro_encoder)
        #pro_encoder = tf.keras.layers.multiply([self.W[5],pro_encoder])
        pro_encoder = self.W[6]*self.l7(pro_encoder)
        #pro_encoder = tf.keras.layers.multiply([self.W[6],pro_encoder])
        pro_encoder = self.W[7]*self.l8(pro_encoder)
        #pro_encoder = tf.keras.layers.multiply([self.W[7],pro_encoder])
        pro_encoder = self.W[8]*self.l9(pro_encoder)
        #pro_encoder = tf.keras.layers.multiply([self.W[8],pro_encoder])
        pro_encoder = self.W[9]*self.l10(pro_encoder)
        #pro_encoder = tf.keras.layers.multiply([self.W[9],pro_encoder])
        pro_encoder = self.W[10]*self.l11(pro_encoder)
        #pro_encoder = tf.keras.layers.multiply([self.W[10],pro_encoder])
        pro_encoder = self.W[11]*self.l12(pro_encoder)
        #pro_encoder = tf.keras.layers.multiply([self.W[11],pro_encoder])
        pro_encoder = self.W[12]*self.l13(pro_encoder)
        #pro_encoder = tf.keras.layers.multiply([self.W[12],pro_encoder])
        pro_encoder = self.W[13]*self.l14(pro_encoder)
        #pro_encoder = tf.keras.layers.multiply([self.W[13],pro_encoder])
        output = self.W[14]*self.l15(pro_encoder)
        #output = tf.keras.layers.multiply([self.W[14],pro_encoder])
        return output
    '''
    
    def call(self, support, query):
        n_class = support.shape[0]
        n_support = support.shape[1]
        n_query = query.shape[1]
        y = np.tile(np.arange(n_class)[:, np.newaxis], (1, n_query))
        y_onehot = tf.cast(tf.one_hot(y, n_class), tf.float32)
        
        uns_loss = 0
        
        # correct indices of support samples (just natural order)
        target_inds = tf.reshape(tf.range(n_class), [n_class, 1])
        target_inds = tf.tile(target_inds, [1, n_query])

        # merge support and query to forward through encoder
        cat = tf.concat([
            tf.reshape(support, [n_class * n_support,
                                 self.w, self.h, self.c]),
            tf.reshape(query, [n_class * n_query,
                               self.w, self.h, self.c])], axis=0)
        
        
        '''
        z = self.encoder(cat)
        z1 = tf.reshape(z[:n_class*n_support],[n_class, n_support, z.shape[-1]])
        
        for clss in range(n_class):
          for img1 in range(n_support):
            cnt=0
            enc1 = tf.expand_dims(z1[clss, img1,:],axis=0)
            tot_loss = 0
            enc_aug = self.encoder(tf.expand_dims(tf.image.flip_left_right(support[clss][img1]),axis=0))
            cnt+=1
            tot_loss = tot_loss + (calc_euclidian_dists(enc1,enc_aug)**2)
            for img2 in range(n_support):
              enc2 = tf.expand_dims(z1[clss, img2,:],axis=0)
              tot_loss = tot_loss + (calc_euclidian_dists(enc1,enc2)**2)
              cnt+=1
              if(cnt>8):
                break
                    
            for img3 in range(n_support):
              adv_cls = (clss+1)%n_class
              enc3 = tf.expand_dims(z1[adv_cls, img3,:],axis=0)
              tot_loss = tot_loss - (calc_euclidian_dists(enc1,enc3)**2)
              cnt+=1
              if(cnt>16):
                break
                
            uns_loss = uns_loss + tot_loss/(cnt*1.0)
            uns_loss = uns_loss + 0.5
        
        '''
        #z_meta = self.encoder(cat)
        
        #z_feat1 = self.meta_enc1(cat)
        
        '''
        z_att1 = tf.keras.layers.Attention()(
            [z_feat1, z_meta]
        )
        '''
        
        #z_fin = tf.concat([z],axis=1)

        
        #W_prototypes = self.meta_encoder(z)
        
        '''
        for ind,layer in enumerate(self.encoder.layers):
            print(layer)
            cat = self.W[ind]*layer(cat)
            cnt+=1
        '''
        z = self.encoder(cat)
        # Divide embedding into support and query
        z_prototypes = tf.reshape(z[:n_class * n_support],
                                  [n_class, n_support, z.shape[-1]])
        
        
        # Prototypes are means of n_support examples
        #z_prototypes = tf.multiply(z_prototypes,self.W)
        z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)
        z_query = z[n_class * n_support:]

        # Calculate distances between query and prototypes
        dists = calc_euclidian_dists(z_query, z_prototypes)

        # log softmax of calculated distances
        log_p_y = tf.nn.log_softmax(-dists, axis=-1)
        log_p_y = tf.reshape(log_p_y, [n_class, n_query, -1])
        
        loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1])) + uns_loss 
        eq = tf.cast(tf.equal(
            tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32), 
            tf.cast(y, tf.int32)), tf.float32)
        acc = tf.reduce_mean(eq)
        return loss, acc

    def save(self, model_path):
        """
        Save encoder to the file.
        Args:
            model_path (str): path to the .h5 file.
        Returns: None
        """
        self.encoder.save(model_path)

    def load(self, model_path):
        """
        Load encoder from the file.
        Args:
            model_path (str): path to the .h5 file.
        Returns: None
        """
        self.encoder(tf.zeros([1, self.w, self.h, self.c]))
        self.encoder.load_weights(model_path)
