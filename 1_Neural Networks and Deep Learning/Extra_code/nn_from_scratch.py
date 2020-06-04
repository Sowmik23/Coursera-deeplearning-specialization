# 01
# Packages import
import matplotlib.pyplot as plt 
import numpy as np 
import sklearn 
import sklearn.datasets
import sklearn.linear_model
import matplotlib

# Display plots inline and change default figure size
# %matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

# 02
# create random datasets
np.random.seed(3)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:, 1], s= 40, c=y, cmap=plt.cm.Spectral)


# 03
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X,y)


# 04
# Helper function to plot a decision boundary
def plt_decision_boundary(pred_func):
	x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
	y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5

	h = 0.01
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
	# plt.show()


# 12
# plot the decision boundary
plt_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")


# 15
num_examples = len(X) #train data set size
nn_input_dim = 2 # dimension of input layer
nn_output_dim = 2#dimension of output layer

#Gradient descent parameters
epsilon = 0.01
reg_lambda = 0.01

# 07
# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

	# forward propagation to calculate predictions
	z1 = X.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

	# calculating the loss
	correct_logprobs = -np.log(probs[range(num_examples), y])
	data_loss = np.sum(correct_logprobs)

	data_loss+= reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

	return 1./num_examples*data_loss



# 08
# Helper function to predict an output(0 or 1)
def predict(model, x):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

	# Forward propagation
	z1 = x.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
	return np.argmax(probs, axis=1)


# 16
# - nn_hdim: Number of nodes in the hidden layer 
# - num_passes: Number of passes through the training data for gradient descent 
# - print_loss: If True, print the loss every 1000 iterations 

def build_model(nn_hdim, num_passes=20000, print_loss=False):

	# Initialize the parameters
	np.random.seed(0)
	W1 = np.random.randn(nn_input_dim, nn_hdim)/np.sqrt(nn_input_dim)
	b1 = np.zeros((1, nn_hdim))
	W2 = np.random.randn(nn_hdim, nn_output_dim)/np.sqrt(nn_hdim)
	b2 = np.zeros((1, nn_output_dim))

	# we return this at the end
	model = {}

	# gradient descent for each batch
	for i in range(0, num_passes):
		
		# Forward propagation
		z1 = X.dot(W1) + b1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2) + b2
		exp_scores = np.exp(z2)
		probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

		# Backpropagation
		delta3 = probs
		delta3[range(num_examples), y] -=1
		dW2 = (a1.T).dot(delta3)
		db2 = np.sum(delta3, axis=0, keepdims=True)
		delta2 = delta3.dot(W2.T)*(1-np.power(a1, 2))
		dW1 = np.dot(X.T, delta2)
		db1 = np.sum(delta2, axis=0)


		# Add regularization terms(b1 and b2 don't have it)
		dW2 += reg_lambda*W2
		dW1 += reg_lambda*W1


		# Gradient descent parameter update
		W1 += -epsilon*dW1
		b1 += -epsilon*db1
		W2 += -epsilon*dW2
		b2 += -epsilon*db2


		# Assign new parameters to the model
		model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


		# print the loss
		if print_loss and i%1000==0:
			print("Loss after iteration %i: %f" %(i, calculate_loss(model)))

	return model


# 17
# build a model with a 3-dimensional hidden layer
model = build_model(3, print_loss=True)

# Plot the decision boundary
plt_decision_boundary(lambda x: predict(model, x))
plt.title("Decision boundary for hidden layer size 3")


# 14
plt.figure(figsize=(16,32))
hidden_layer_dimensions = [1,2,3,4,5,20,50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
	plt.subplot(5,2, i+1)
	plt.title('Hidden Layer size %d' %nn_hdim)
	model = build_model(nn_hdim)
	plt_decision_boundary(lambda x: predict(model, x))
plt.show()
