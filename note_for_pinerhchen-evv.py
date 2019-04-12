# coding=utf-8
"""summary的一个测试程序"""
import tensorflow as tf

sess = tf.InteractiveSession()
scalar_var = tf.Variable(0.0, name='scalar_var')
# 标量类型的summary
scalar_ops = tf.summary.scalar(name='scalar_var_summary', tensor=scalar_var)
scalar_writer = tf.summary.FileWriter('Log', sess.graph)

init = tf.global_variables_initializer()
sess.run(init)
for idx in range(50):
    sess.run(tf.assign(scalar_var, idx))
    print (sess.run(scalar_var))
    # 运行summary操作
    scalar_res = sess.run(scalar_ops)
    #　将summary的结果写入磁盘
    scalar_writer.add_summary(scalar_res, idx)
