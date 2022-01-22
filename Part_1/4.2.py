import numpy as np
import matplotlib.pyplot as plt


x = np.arange(100)
y = np.array(5)

z=x+y

xLastTen = x[90:]

xUpdate = np.arange(0, 1000, 10)

xDotProduct = x.dot(x)

xAsteriskProduct = x * x

xReshape = xUpdate.reshape((10, 10))

yNew = np.arange(1,11)
zNew = xReshape * yNew[:, np.newaxis]
print(zNew)

for i in range(10):
   plt.plot(zNew[i])
plt.show()


for i in range(10):
    ax = plt.subplot(5, 2, i + 1)
    plt.plot(zNew[i])
plt.show()
plt.savefig('figure1.png')
