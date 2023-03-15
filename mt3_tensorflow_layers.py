import tensorflow as tf
from tensorflow.keras import layers

class Mt3_Encoder_Layer(tf.keras.Model):
    def __init__(self, embed_dim=512, num_heads=6,
                 feed_forward_dim=2048, key_dim=64,
                 rate=0.1):
        super().__init__()

        self.pre_attention_layer_norm = layers.LayerNormalization(epsilon=1e-6, name='pre_attention_layer_norm', scale=False)

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,  use_bias=False, name='attention'
        )
        self.attention_dropout = layers.Dropout(rate)

        self.pre_mlp_layer_norm = layers.LayerNormalization(epsilon=1e-6, name='pre_mlp_layer_norm', scale=False)

        self.mlp = keras.Sequential(
                    [
                        layers.Dense(feed_forward_dim, use_bias=False),#, activation='gelu'
                        layers.Dense(embed_dim, use_bias=False),
                    ]
                )

        self.mlp_dropout = layers.Dropout(rate)

    def set_norm_attention_norm_weights(self, weights):
        self.pre_attention_layer_norm.set_weights([weights[0]])
        self.attention.set_weights(weights[1:-1])
        self.pre_mlp_layer_norm.set_weights([weights[-1]])

    def freeze_norm_attention_norm(self):
        self.pre_attention_layer_norm.trainable=False
        self.attention.trainable=False
        self.pre_mlp_layer_norm.trainable=False

    def set_norm_attention_norm_trainable(self):
        self.pre_attention_layer_norm.trainable=True
        self.attention.trainable=True
        self.pre_mlp_layer_norm.trainable=True

    def call(self, input, training=True):

        x = self.pre_attention_layer_norm(input)
        x = self.attention_dropout(self.attention(x, x), training=training)
        x = x + input
        y = self.pre_mlp_layer_norm(x)
        y = self.mlp_dropout(self.mlp(y), training=training)
        y= y + x

        return y


'''
mt3_Encoder_Layer=Mt3_Encoder_Layer()
mt3_Encoder_Layer(np.ones((1, 30, 512))).shape
mt3_Encoder_Layer.set_norm_attention_norm_weights(encoder_layer_0_weights)
mt3_Encoder_Layer(np.ones((1, 30, 512))).shape
mt3_Encoder_Layer.freeze_norm_attention_norm()
mt3_Encoder_Layer.set_norm_attention_norm_trainable()
'''

#@markdown Mt3_Encoder

class Mt3_Encoder(tf.keras.Model):
    def __init__(self, n_frames=30, embed_dim=512, num_heads=6,
                 feed_forward_dim=2048, key_dim=64,
                 n_layers=8, 
                 rate=0.1):
        super().__init__()
        self.layers_keys = ['layers_0', 'layers_1', 'layers_2', 'layers_3', 'layers_4', 'layers_5', 'layers_6', 'layers_7']
        self.n_layers=n_layers
        self.n_frames=n_frames
        self.continuous_inputs_projection=layers.Dense(embed_dim, use_bias=False)
        self.pos_encoding = sinusoidal((feed_forward_dim, embed_dim))
        self.mt3_encoder_layers = keras.Sequential(
            [
                Mt3_Encoder_Layer(embed_dim=embed_dim, num_heads=num_heads,
                                   feed_forward_dim=feed_forward_dim, key_dim=key_dim,
                                   rate=rate)
                for _ in range(n_layers)
            ]
        )
        self.encoder_norm = layers.LayerNormalization(epsilon=1e-6, name='encoder_norm', scale=False)

    def set_Encoder_weights_from_mt3(self, params):

        continuous_inputs_projection_weights = params['continuous_inputs_projection']['kernel']
        self.continuous_inputs_projection.set_weights([continuous_inputs_projection_weights])

        for ind, key in enumerate(self.layers_keys):
            layer_weigts = get_mt3_encoder_weights_to_Encoder_Norm_Attention_Norm(params[key])
            self.mt3_encoder_layers.layers[ind].set_norm_attention_norm_weights(layer_weigts)
            
    def freeze_copied_weights(self):
        self.continuous_inputs_projection.trainable=False
        for ind in range(self.n_layers):
            self.mt3_encoder_layers.layers[ind].freeze_norm_attention_norm()

    def set_copied_weights_trainable(self):
        self.continuous_inputs_projection.trainable=True
        for ind in range(self.n_layers):
            self.mt3_encoder_layers.layers[ind].set_norm_attention_norm_trainable()


    def call(self, x, training=True):

        x = self.continuous_inputs_projection(x)
        x = x + self.pos_encoding[tf.newaxis, :self.n_frames, :]
        x = self.mt3_encoder_layers(x, training=training)
        x = self.encoder_norm(x)

        return x

'''mt3_Encoder=Mt3_Encoder()
mt3_Encoder(np.ones((1, 30, 512))).shape
mt3_Encoder.set_Encoder_weights_from_mt3(params['encoder'])
mt3_Encoder.freeze_copied_weights()
mt3_Encoder.set_copied_weights_trainable()
'''

#@markdown Mt3_Encoder_Layer_3fc


class Mt3_Encoder_Layer_3fc(tf.keras.Model):
    
    def __init__(self, embed_dim=512, num_heads=6,
                 feed_forward_dim=1024, key_dim=64,
                 rate=0.1):
        super().__init__()

        self.pre_attention_layer_norm = layers.LayerNormalization(epsilon=1e-6, name='pre_attention_layer_norm', scale=False)

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,  use_bias=False, name='attention'
        )

        self.pre_mlp_layer_norm = layers.LayerNormalization(epsilon=1e-6, name='pre_mlp_layer_norm', scale=False)

        self.mlp_wi_0=layers.Dense(feed_forward_dim, use_bias=False)#, activation='gelu'
        self.mlp_wi_1=layers.Dense(feed_forward_dim, use_bias=False)
        self.mlp_wo=layers.Dense(embed_dim, use_bias=False)

    def call(self, input):

        x = self.pre_attention_layer_norm(input)
        x = self.attention(x, x)
        x = x + input
        y = self.pre_mlp_layer_norm(x)

        y = self.mlp_wi_0(y)*self.mlp_wi_1(y)
        y = self.mlp_wo(y)

        y= y + x

        return y


'''
mt3_Encoder_Layer_3fc=Mt3_Encoder_Layer_3fc()
mt3_Encoder_Layer_3fc(np.ones((1, 30, 512))).shape
'''


#@markdown Mt3_Encoder_3fc

class Mt3_Encoder_3fc(tf.keras.Model):
    def __init__(self, n_frames=30, embed_dim=512, num_heads=6,
                 feed_forward_dim=1024, key_dim=64,
                 n_layers=8, 
                 rate=0.1):
        super().__init__()
        self.layers_keys = ['layers_0', 'layers_1', 'layers_2', 'layers_3', 'layers_4', 'layers_5', 'layers_6', 'layers_7']

        self.continuous_inputs_projection=layers.Dense(embed_dim, use_bias=False)
        self.pos_encoding = sinusoidal((n_frames, embed_dim))
        self.mt3_encoder_layers = keras.Sequential(
            [
                Mt3_Encoder_Layer_3fc(embed_dim=embed_dim, num_heads=num_heads,
                                   feed_forward_dim=feed_forward_dim, key_dim=key_dim,
                                   rate=rate)
                for _ in range(n_layers)
            ]
        )
        self.encoder_norm = layers.LayerNormalization(epsilon=1e-6, name='pre_attention_layer_norm', scale=False)

    def set_Encoder_weights_from_mt3(self, params):

        continuous_inputs_projection_weights = params['continuous_inputs_projection']['kernel']
        self.continuous_inputs_projection.set_weights([continuous_inputs_projection_weights])

        for ind, key in enumerate(self.layers_keys):
            layer_weigts = get_mt3_encoder_weights(params[key])
            self.mt3_encoder_layers.layers[ind].set_weights(layer_weigts)
            
    def freeze_copied_weights(self):
        self.continuous_inputs_projection.trainable=False
        for ind in range(len(self.layers_keys)):
            self.mt3_encoder_layers.layers[ind].trainable=False

    def set_copied_weights_trainable(self):
        self.continuous_inputs_projection.trainable=True
        for ind in range(len(self.layers_keys)):
            self.mt3_encoder_layers.layers[ind].trainable=True


    def call(self, x):

        x = self.continuous_inputs_projection(x)
        x = x + self.pos_encoding[tf.newaxis, :, :]
        x = self.mt3_encoder_layers(x)
        x = self.encoder_norm(x)

        return x


'''
mt3_Encoder_3fc=Mt3_Encoder_3fc()
mt3_Encoder_3fc(np.ones((1, 30, 512))).shape
mt3_Encoder_3fc.set_Encoder_weights_from_mt3(params['encoder'])
mt3_Encoder_3fc.freeze_copied_weights()
mt3_Encoder_3fc.set_copied_weights_trainable()
'''

#@markdown Mt3_Decoder_Layer_3fc


class Mt3_Decoder_Layer_3fc(tf.keras.Model):
    
    def __init__(self, embed_dim=512, num_heads=6,
                 feed_forward_dim=1024, key_dim=64,
                 rate=0.1):
        super().__init__()

        self.pre_self_attention_layer_norm = layers.LayerNormalization(epsilon=1e-6, name='pre_self_attention_layer_norm', scale=False)

        self.self_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,  use_bias=False, name='self_attention'
        )

        self.pre_cross_attention_layer_norm = layers.LayerNormalization(epsilon=1e-6, name='pre_cross_attention_layer_norm', scale=False)

        self.encoder_decoder_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,  use_bias=False, name='encoder_decoder_attention'
        )

        self.pre_mlp_layer_norm = layers.LayerNormalization(epsilon=1e-6, name='pre_mlp_layer_norm', scale=False)

        self.mlp_wi_0=layers.Dense(feed_forward_dim, use_bias=False)#, activation='gelu'
        self.mlp_wi_1=layers.Dense(feed_forward_dim, use_bias=False)
        self.mlp_wo=layers.Dense(embed_dim, use_bias=False)

    def call(self, enc_out, target, use_causal_mask=True):

        x = self.pre_self_attention_layer_norm(target)
        x = self.self_attention(x, x, use_causal_mask=use_causal_mask)
        x = x + target
        y = self.pre_cross_attention_layer_norm(x)
        y = self.encoder_decoder_attention(y, enc_out)
        x = y + x
        z = self.pre_mlp_layer_norm(y)

        z = self.mlp_wi_0(z)*self.mlp_wi_1(z)
        z = self.mlp_wo(z)
        z = z + y

        return z

