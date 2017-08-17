
# coding: utf-8

# In[1]:

import tensorflow as tf
a=tf.constant(5)
b=tf.constant(3)

with tf.name_scope("MyOperation"):
    c=tf.multiply(a,b,name="Mult1")
    d=tf.multiply(2,c,name="Mult_2")
    e=tf.square(a,name="Sqr_A")
    f=tf.square(b,name="Sqr_B")

g=tf.add(e,f,name="Add")
h=tf.subtract(g,d,name="Subtract")

with tf.Session() as sess:
    writer = tf.summary.FileWriter("/tmp/tboard/output1_a2", sess.graph)
    print(sess.run(h))
    writer.close()


# In[ ]:



