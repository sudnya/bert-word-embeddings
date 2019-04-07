
import numpy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


filename = "output-features.npy"

array = numpy.load(filename)

array = numpy.reshape(array, (-1, array.shape[-1]))

pca = PCA(n_components=3)

components = pca.fit_transform(array)
#print(components.shape)
plt.figure(1)
plt.scatter(components[:,0], components[:,1])
plt.figure(2)
plt.scatter(components[:,0], components[:,2])
plt.figure(3)
plt.scatter(components[:,1], components[:,2])
plt.show()


