
import numpy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


filename = "output-features.npy"

array = numpy.load(filename)

array = numpy.reshape(array, (-1, array.shape[-1]))

pca = PCA(n_components=2)

components = pca.fit_transform(array)
#print(components.shape)

plt.scatter(components[:,0], components[:,1])
plt.show()


