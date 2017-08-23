
# coding: utf-8

# In[2]:

import tensorflow as tf


x=tf.constant([12.25,22.33,33.12,44.67],shape=[1,4],name='f')
y=tf.constant([14.25,24.33,35.12,46.67],shape=[1,4],name='s')

e=tf.constant("y=mx+c")
ms=tf.constant("m:")
mc=tf.constant("c:")
ac=tf.constant("Actual values of Y")
pc=tf.constant("predicted values of Y")

# MEAN OF X, Y
mx = tf.reduce_mean(x)
my = tf.reduce_mean(y)

    
# X-MEAN , Y-MEAN
vx=tf.Variable(x-mx, name='varx')
vy=tf.Variable(y-my, name='vary')

   
#VS=X-MEAN SQUARE, SUM OF VS
vs=tf.square(vx)
sm=tf.reduce_sum(vs)

    
#COVARIENCE  OF X,Y
cV=tf.multiply(vx,vy)
cm=tf.reduce_sum(cV)

#calculation of m and c
m=tf.divide(cm,sm)
c=my-(mx*m)

#Predicted values of Y
p=tf.multiply(x,m)
pry=tf.add(p,c)     



rms=tf.sqrt(tf.reduce_mean(tf.squared_difference(y,pry)))
    
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
 
print(sess.run(e))
print(sess.run(ms),sess.run(m))
print(sess.run(mc),sess.run(c))

print(sess.run(ac),sess.run(y))
print(sess.run(pc),sess.run(pry))
print(sess.run(rms))




    


# In[ ]:




# In[ ]:



