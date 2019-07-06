
# coding: utf-8

# In[1]:


from keras.layers import *
import keras.backend as K
from keras.engine.topology import Layer


# In[2]:


# soft-attention

def unchanged_shape(input_shape):
    # function for Lambda layer 
    return input_shape

def soft_attention_alignment(input_1, input_2):
    """
    两输入为三维张量(bs, sl1, size) (bs, sl2, size)
    
    return (bs, sl2, size), (bs, sl1, size)
    """
    attention = Dot(axes=-1)([input_1, input_2])  # (bs, sl1, size)·(bs, sl2, size) ==> (bs, sl1, sl2)
    
    w_att_1 = Lambda(lambda x: K.softmax(x, axis=1), output_shape=unchanged_shape)(attention)  # (bs, sl1, sl2)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: K.softmax(x, axis=2), 
                                     output_shape=unchanged_shape)(attention))  # (bs, sl2, sl1)
    
    in1_aligned = Dot(axes=1)([w_att_1, input_1])  # (bs, sl1, sl2)·(bs, sl1, size)  ==> (bs, sl2, size)
    in2_aligned = Dot(axes=1)([w_att_2, input_2])  # (bs, sl2, sl1)·(bs, sl2, size)  ==> (bs, sl1, size)

    return in1_aligned, in2_aligned   # (bs, sl2, size)  (bs, sl1, size)  与输入shape相反

# # 测试
# a = K.ones((3, 5, 7))
# b = K.ones((3, 20, 7))
# res1, res2 = soft_attention_alignment(a, b)
# print(K.int_shape(res1), K.int_shape(res2))
# # >>>(3, 20, 7) (3, 5, 7)


# In[3]:


# co-attention

def co_attention(input_1, input_2):
    """
    两输入为三维张量(bs, sl, size) (bs, sl, size)  (要求步长相同)
    
    return 
    """
    dense_w = TimeDistributed(Dense(1))
    atten = Lambda(lambda x: K.batch_dot(x[0], x[1]))([input_1, Permute((2, 1))(input_2)]) 
    # (bs, sl, size), (bs, size, sl)  ==>  (bs, sl, sl)

    atten_1 = dense_w(atten)   # (bs, sl, 1)
    atten_1 = Flatten()(atten_1)   # (bs, sl)
    atten_1 = Activation('softmax')(atten_1)
    atten_1 = Reshape((1, -1))(atten_1)   # (bs, 1, sl)
    
    atten_2 = dense_w(Permute((2, 1))(atten))   # (bs, sl, 1)
    atten_2 = Flatten()(atten_2)
    atten_2 = Activation('softmax')(atten_2)
    atten_2 = Reshape((1, -1))(atten_2)   # (bs, 1, sl)
    
    out1 = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten_1, input_1])  # (bs, 1, sl)·(bs, sl, size)  ==> (bs, 1, size)
    out1 = Flatten()(out1)   # (bs, size)
    out2 = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten_2, input_2])  # (bs, 1, sl)·(bs, sl, size)  ==> (bs, 1, size)
    out2 = Flatten()(out2)   # (bs, size)
    
    return out1, out2  # (bs, size), (bs, size)

# # 测试
# a = K.ones((3, 5, 7))
# b = K.ones((3, 5, 7))
# res1, res2 = co_attention(a, b)
# print(K.int_shape(res1), K.int_shape(res2))
# # >>>(3, 7) (3, 7)


# In[4]:


# 层级attention

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':   # 默认添加最后一个维度  return => (samples, steps, feaures)
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)   # (samples, steps, features), (feaures, feaures, 1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.  # 支持masking
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention  # 用上下文向量支持attention
    
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.  
    # Output shape
        2D tensor with shape: `(samples, features)`.  
    
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.  # dimension based on GRU shape
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())  # [None， features]
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3  # 输入长度必须为3 

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),  # 相同 features  (size, size)
                                 initializer=self.init,  # initializer.get('glorot_uniform')
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,  # 正则化
                                 constraint=self.W_constraint)   # 约束
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),  # (size,)
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),   # (size, )
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)  # .build(input_shape) 

    def call(self, x):
        uit = dot_product(x, self.W)  # (bs,sl,size)(size,size)  ==> (bs, sl, size)

        if self.bias:
            uit += self.b   # (bs, sl, size)

        uit = K.tanh(uit)  # (bs, sl, size)  得到uit
        ait = dot_product(uit, self.u)   # (bs, sl, size), (size, 1)  ==> (bs, sl, 1)  => # (bs, sl)

        a = K.softmax(ait)  # (bs, sl)
        a = K.expand_dims(a)     # (bs, sl)
        weighted_input = x * a   #  (bs, sl, size) * ((bs, sl, 1) => (bs, sl, size)

        return K.sum(weighted_input, axis=1)   # (bs, size)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]   # 不用括号


# In[15]:


# multi-head self-attention  来源：https://spaces.ac.cn/archives/4765

class Position_Embedding(Layer):  # 位置embedding
    
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size #必须为偶数   可以自定义位置embedding的维度
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)
        
    def call(self, x):   # (bs, sl, size)
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])   # size
        batch_size,seq_len = K.shape(x)[0],K.shape(x)[1]   # bs, sl
        position_j = 1. / K.pow(10000., 2*K.arange(self.size/2, dtype='float32') / self.size)   # (size/2,)
        position_j = K.expand_dims(position_j, 0)  # (1, size/2)
        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1) - 1   #K.arange不支持变长，只好用这种方法生成  (bs, sl)
        position_i = K.expand_dims(position_i, 2)   # (bs, sl, 1)
        position_ij = K.dot(position_i, position_j)  # (bs, sl, 1) · (1, size/2)  ==>  (bs, sl, size/2)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)   # (bs, sl, size)
        if self.mode == 'sum':
            return position_ij + x   # (bs, sl, size)
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)   # (bs, sl, size*2)
        
    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)


# 多头自注意力
class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head   # 注意力头数
        self.size_per_head = size_per_head   # 每个注意力头的大小
        self.out_dim = nb_head*size_per_head   # 输出维度
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        
        self.q_kernel = self.add_weight(name='q_kernel', 
                                  shape=(input_shape[0][-1], self.out_dim),
                                  initializer='glorot_uniform')   # (size, att_dim)
        self.k_kernel = self.add_weight(name='k_kernel', 
                                  shape=(input_shape[1][-1], self.out_dim),
                                  initializer='glorot_uniform')   # (size, att_dim)
        self.v_kernel = self.add_weight(name='v_kernel', 
                                  shape=(input_shape[2][-1], self.out_dim),
                                  initializer='glorot_uniform')   # (size, att_dim)
        super(Attention, self).build(input_shape)
        
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'sum':
                return x * mask
            else:
                return x - (1-mask)*1e10
                
    def call(self, inputs):
        #如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        #如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = input[4]

        #对q、k、v做线性变换
        qw = K.dot(q, self.q_kernel)  # (bs, sl, size) (size, att_dim)  ==>  (bs, sl, att_dim)
        kw = K.dot(k, self.k_kernel)  
        vw = K.dot(v, self.v_kernel)
        
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))   # (bs, sl, nb_head, size_ph)
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        
        # 维度置换
        qw = K.permute_dimensions(qw, (0,2,1,3))   # (bs, nb_head, sl, size_ph)
        kw = K.permute_dimensions(kw, (0,2,1,3))
        vw = K.permute_dimensions(vw, (0,2,1,3))
        
        #计算内积，然后mask，然后softmax
        a = K.batch_dot(qw, kw, axes=[3,3]) / self.size_per_head**0.5   # (bs, nb_head, sl, sl)
        a = K.permute_dimensions(a, (0,3,2,1))   # (bs, sl, sl, nb_head)
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0,3,2,1))   # (bs, nb_head, sl, sl)
        a = K.softmax(a)   # (bs, nb_head, sl, sl)
        
        #输出并mask
        o = K.batch_dot(a, vw, axes=[3,2])  # (bs,nb_head,sl,sl) (bs,nb_head,sl,size_ph) => (bs,nb_head,sl,size_ph)
        o = K.permute_dimensions(o, (0,2,1,3))   # (bs,sl,nb_head,size_ph)
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))  # (bs,sl,att_dim)
        o = self.mask(o, q_mask, 'mul')
        return o
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


# In[16]:


a = K.ones((3, 5, 7))
res = Attention(8, 16)([a, a, a])
K.int_shape(res)

