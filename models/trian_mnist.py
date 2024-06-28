import mnist
from knn_pycuda import *

img = mnist.train_images()
lbl = mnist.train_labels()
test_img = mnist.train_images()

a = 1

moudle = KNN(3)
moudle.fit(img, lbl)
moudle.predict(test_img)