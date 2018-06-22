import tensorflow as tf
import numpy as np

# W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
# print(tf.random_uniform([16, 4], 0, 0.01))
import numpy as np
import numpy as np

# shape = [[1, 2],
#          [3, 4]]
# print(*zip(range(2, 4), shape))
#
# print(*zip(range(5, 10), range(10, 20)))
#
# print(*iter(range(4)))

# print(np.transpose(shape))
# print(shape)

# print(39/4)
# print(39//4)
# print(39%4)

oneHot = np.zeros(23)
shape = 'z'
rotation = '90'
rot = {'0': 0, '90': 1, '180': 2, '270': 3}
choices = {'s': [0, 1, 2, 3], 'z': [4, 5, 6, 7], 'r': [8, 9, 10, 11], 'L': [12, 13, 14, 15], 'o': [16],
           'I': [17, 18], 'T': [19, 20, 21, 22]}
# oneHot[choices.get(shape)[rot.get(rotation)]] = 1

# print(oneHot)

a = np.array([[4, 1, 8, 4, 4], [5, 4, 2, 1, 1], [5, 4, 2, 3, 3]])
print(a[0:3, 1:3])
