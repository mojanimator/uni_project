import random

arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
memory = (arr)
print(memory)
batch_size = 5

samples = list(zip(list(random.sample(memory, batch_size))))
print(samples)
stri = ""
map(lambda x: stri.join(x), samples)
print(list(stri))
