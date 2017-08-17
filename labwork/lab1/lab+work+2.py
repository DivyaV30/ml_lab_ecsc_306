
# coding: utf-8

# In[5]:

import tensorflow as tf
x=tf.constant([2.3,5.4,0.9,0.8],shape=[1,4],name='x')
y=tf.constant([0.3,0.4,2.9,3.8],shape=[1,4],name='y')
with tf.name_scope("mean"):
    with tf.name_scope("mean_of_x_y"):
        x1=tf.reduce_mean(x)
        y1=tf.reduce_mean(y)
        sess=tf.Session()
    print("the mean of x1",sess.run(x1))
    print("the mean of x2",sess.run(y1))
with tf.name_scope("variance"):
    with tf.name_scope("scope_SUB1"):
        a=tf.subtract(x,x1)
        sess1=tf.Session()
    print("variance of a",sess1.run(a))
    with tf.name_scope("scope_SUB2"):
        b=tf.subtract(y,y1)
        sess2=tf.Session()
    print(" variance of b",sess2.run(b))
    with tf.name_scope("scope_sqr"):
        a1=tf.multiply(a,a)
        sess3=tf.Session()
    print(sess3.run(a1))
    with tf.name_scope("scope_of_Summation"):
        var=tf.reduce_sum(a1)
        sess4=tf.Session()
    print(sess4.run(var))
with tf.name_scope("covariance"):
    with tf.name_scope("Multiplication"):
        mul1=tf.multiply(a,b)
        sess5=tf.Session()
    print(sess5.run(mul1))
    with tf.name_scope("scope_of_Summation2"):
        mul2=tf.reduce_sum(mul1)
        sess6=tf.Session()
    print(sess6.run(mul2))
with tf.name_scope("value_of_c"):
    with tf.name_scope("Division"):
        c=tf.divide(mul2,var)
        sess7=tf.Session()
        print(sess7.run(c))
   
    with tf.name_scope("value_of_m"):
        with tf.name_scope("Multiplication2"):
            c1=tf.multiply(c,x1)
            sess8=tf.Session()
            print(sess8.run(c1))
    with tf.name_scope("scope_SUB3"):
            c2=tf.subtract(y1,c1)
            sess9=tf.Session()
            print(sess9.run(c2))    
            writer = tf.summary.FileWriter("/tmp/tboard/assignment2", sess.graph)
            writer.close()


# In[ ]:




# In[ ]:



