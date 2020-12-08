#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

apples = fruit[0]
bananas = fruit[1]
oranges = fruit[2]
peaches = fruit[3]

r = [0, 1, 2]
names = ["Farrah", "Fred", "Felicia"]

s12 = np.add(apples, bananas).tolist()
s123 = np.add(s12, oranges).tolist()


plt.bar(r, apples, color='red', label='apples', width=0.5)
plt.bar(r, bananas, bottom=apples, color='yellow', label='bananas', width=0.5)
plt.bar(r, oranges, bottom=s12, color='#ff8000', label='oranges', width=0.5)
plt.bar(r, peaches, bottom=s123, color='#ffe5b4', label='peaches', width=0.5)

plt.xticks(r, names)
plt.legend(loc="upper right")
plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80, 10)
plt.title('Number of Fruit per Person')

plt.show()
