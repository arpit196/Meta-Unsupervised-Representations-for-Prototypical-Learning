import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
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
    h=tf.shape(x)[0]
    w=tf.shape(x)[-1]
    return tf.reshape(x, [h, -1, w])

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(SelfAttention, self).__init__()
        self.gamma = tf.Variable(tf.random.truncated_normal([1]),
                             name="gamma")
        self.f1 = Conv2D(filters=4, kernel_size=3, padding='same',name='fir')
        self.g1 = Conv2D(filters=4, kernel_size=3, padding='same',name='sec')
        self.h1 = Conv2D(filters=16, kernel_size=3, padding='same',name='thir')
        self.o1 = Conv2D(filters=16, kernel_size=3, padding='same',name='four')
    
    def call(self, inputs):
        f = self.f1(inputs) # [bs, h, w, c']
        g = self.g1(inputs) # [bs, h, w, c']
        h = self.h1(inputs) # [bs, h, w, c]

            # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
        
        o = tf.reshape(o, shape=tf.shape(inputs)) # [bs, h, w, C]
        #o = tf.reshape(o, shape=tf.shape(inputs.shape)) # [bs, h, w, C]

        o = self.o1(o)
        
        inputs = self.gamma * o + inputs
        return inputs
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'f1': self.f1,
            'g1': self.g1,
            'h1': self.h1,
            'o1': self.o1
        })
        return config


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
        self.W = tf.Variable(tf.random.truncated_normal([1,16]), name="W")
        # Encoder as ResNet like CNN with 4 blocks
        inputs1 = tf.keras.layers.Input(shape=(self.w, self.h, self.c))
        '''
        meta_enc1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')(inputs1)
        meta_enc1 = tf.keras.layers.BatchNormalization()(meta_enc1)
        meta_enc1 = tf.keras.layers.ReLU()(meta_enc1)
        meta_enc1 = tf.keras.layers.MaxPool2D((2, 2))(meta_enc1)
        meta_enc1 = tf.keras.layers.GlobalAveragePooling2D()(meta_enc1)
        print(meta_enc1)
        '''
        
        self.l1= tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')
        self.l2= tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')
        self.l3= tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')
        self.l4= tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')
        self.l5= tf.keras.layers.BatchNormalization()
        self.l6= tf.keras.layers.ReLU()
        self.l7= tf.keras.layers.MaxPool2D((2, 2))

        self.l8=  tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')
        self.l9=  tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')
        #self.l10= SelfAttention()
        
        self.l11= tf.keras.layers.BatchNormalization()
        self.l12= tf.keras.layers.ReLU()
        self.l13= tf.keras.layers.MaxPool2D((2, 2))
        self.l14= Flatten()
        
        encoder=self.l1(inputs1)
        encoder1=self.l2(encoder)
        encoder2=self.l3(encoder1)
        encoder3=self.l4(encoder2)
        encoder4=self.l5(encoder3)
        encoder6=self.l6(encoder4)
        encoder7=self.l7(encoder6)
        encoder8=self.l8(encoder7)
        encoder9=self.l9(encoder8)
        #encoder10=self.l10(encoder9)
        encoder11=self.l11(encoder9)
        encoder12=self.l12(encoder11)
        encoder13=self.l13(encoder12)
        print(encoder13)
        encoder14=tf.keras.layers.Attention()([self.W, encoder13, encoder13])
        print(encoder14)
        encoder15=self.l14(encoder14)
        
        #meta_att = tf.keras.layers.Attention()([meta_enc1, encoder]) 
        self.encoder = Model(inputs = inputs1, outputs = encoder15)
        #self.meta_encoder = Model(inputs = inputs1, outputs = meta_enc1)
        
        '''
        self.decoder = tf.keras.Sequential()
        #self.decoder.add(tf.keras.layers.Reshape(tf.shape(encoder14))
        self.decoder.add(tf.keras.layers.UpSampling2D((2, 2)))
        self.decoder.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same'))
        self.decoder.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same'))
        self.decoder.add(tf.keras.layers.UpSampling2D((2, 2)))
        self.decoder.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same'))
        self.decoder.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same'))
        self.decoder.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same'))
        self.decoder.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same'))
        self.decoder.add(tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same'))
        '''
        '''
        self.meta_encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)), Flatten(), Dense(128)]
        )
        '''
    
    '''
    def generator(self,input, prototype):
        combined = input + self.W*prototype
        return self.decoder(combined)
    '''
                         
    def call(self, support, query):
        n_class = support.shape[0]
        n_support = support.shape[1]
        n_query = query.shape[1]
        y = np.tile(np.arange(n_class)[:, np.newaxis], (1, n_query))
        y_onehot = tf.cast(tf.one_hot(y, n_class), tf.float32)
        uns_loss = 0.0
        # correct indices of support samples (just natural order)
        target_inds = tf.reshape(tf.range(n_class), [n_class, 1])
        target_inds = tf.tile(target_inds, [1, n_query])

        # merge support and query to forward through encoder
        cat = tf.concat([
            tf.reshape(support, [n_class * n_support,
                                 self.w, self.h, self.c]),
            tf.reshape(query, [n_class * n_query,
                               self.w, self.h, self.c])], axis=0)
        
        
        z = self.encoder(cat)
        
        '''
        z_uns = self.uns_enc(z)
        print("zshape")
        print(z.shape)
        
        z1 = tf.reshape(z_uns[:n_class*n_support],[n_class, n_support, z_uns.shape[-1]])
        
        for clss in range(n_class):
          for img1 in range(n_support):
            cnt=0
            enc1 = tf.expand_dims(z1[clss, img1,:],axis=0)
            tot_loss = 0
            enc_aug = self.uns_enc(self.encoder(tf.expand_dims(tf.image.flip_left_right(support[clss][img1]),axis=0)))
            cnt+=1
            tot_loss = tot_loss + (calc_euclidian_dists(enc1,enc_aug)**2)
            for img2 in range(n_support):
              enc2 = tf.expand_dims(z1[clss, img2,:],axis=0)
              tot_loss = tot_loss + (calc_euclidian_dists(enc1,enc2)**2)
              cnt+=1
              if(cnt>10):
                break
                    
            for img3 in range(n_support):
              adv_cls = (clss+1)%n_class
              enc3 = tf.expand_dims(z1[adv_cls, img3,:],axis=0)
              tot_loss = tot_loss - (calc_euclidian_dists(enc1,enc3)**2)
              cnt+=1
              if(cnt>20):
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
        #z = self.encoder(cat)
        # Divide embedding into support and query
        #z_prototypes = self.meta_encoder(cat)
        z_prototypes = tf.reshape(z[:n_class * n_support],
                                  [n_class, n_support, z.shape[-1]])

        z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)
        #z1 = self.encoder(support[0][0])
        #z2 = generative(z1, z_prototypes[0])
        #z2 = self.decoder(z1)
        #uns_loss = uns_loss + tf.reduce_sum(z2-z1)
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
