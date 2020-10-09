#%%
import tensorflow as tf
# %%
x_shape = (8, 107, 128)
x = tf.random.normal(x_shape)
A_shape = (8,107,107)
A = tf.random.normal(A_shape)
z = tf.matmul(A,x)
print(z.shape)

# %%
