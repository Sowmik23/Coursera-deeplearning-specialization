import numpy as np 
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt 


def generate_data():
	np.random.seed(0)
	X, y = datasets.make_moons(200, noise=0.20)
	return X, y


def visualize(X, y, clf):
	# plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
	# plt.show()
	plt_decision_boundary(lambda x: clf.predict(x), X, y)
	plt.title("Logistic Regression")


def plt_decision_boundary(pred_func, X, y):
	x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
	y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5

	h = 0.01
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
	plt.show()


def classify(X, y):
	clf = linear_model.LogisticRegressionCV()
	clf.fit(X,y)
	return clf


def main():
	X, y = generate_data()
	print(X)
	print("y : \n")
	print(y)
	print("\n\n")

	clf = classify(X,y)
	visualize(X,y,clf)

	print("End of program")


if __name__=="__main__":
	main()