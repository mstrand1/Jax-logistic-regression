from jax import jit,grad,vmap,device_put,random
import jax.numpy as jnp
from functools import partial


class JaxReg:
    """
    Logistic regression classifier with GPU acceleration support through Google's JAX. The point of this class is fitting speed: I want this
    to fit a model for very large datasets (k49 in particular) as quickly as possible!

    - jit compilation utilized in sigma and loss methods (strongest in sigma due to matrix mult.). We need to 'partial' the
      jit function because it is used within a class.

    - jax.numpy (jnp) operations are JAX implementations of numpy functions.

    - jax.grad used as the gradient function. Returns gradient with respect to first parameter.

    - jax.vmap is used to 'vectorize' the jax.grad function. Used to compute gradient of batch elements at once, in parallel.
    """

    def __init__(self, learning_rate=.001, num_epochs=50, size_batch=20):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.size_batch = size_batch

    def fit(self, data, y):
        self.K = max(y) + 1
        ones = jnp.ones((data.shape[0], 1))
        X = jnp.concatenate((ones, data), axis=1)
        W = jnp.zeros((jnp.shape(X)[1], max(y) + 1))

        self.coeff = self.mb_gd(W, X, y)

    # New mini-batch gradient descent function (because jitted functions require arrays which do not change shape)
    def mb_gd(self, W, X, y):
        num_epochs = self.num_epochs
        size_batch = self.size_batch
        eta = self.learning_rate
        N = X.shape[0]

        # Define the gradient function using jit, vmap, and the jax's own gradient function, grad.
        # vmap is especially useful for mini-batch GD since we compute all gradients of the batch at once, in parallel.
        # Special paramaters in_axes,out_axes define the axis of the input paramters (W, X, y) and output (gradients of batches)
        # upon which to vectorize. grads_b = loss_grad(W, X_batch, y_batch) has shape (batch_size, p+1, k) for p variables and k classes.

        loss_grad = jit(vmap(grad(self.loss), in_axes=(None, 0, 0), out_axes=0))

        for e in range(num_epochs):
            shuffle_index = random.permutation(random.PRNGKey(e), N)
            for m in range(0, N, size_batch):
                i = shuffle_index[m:m + size_batch]

                grads_b = loss_grad(W, X[i, :], y[i])  # 3D jax array of size (batch_size, p+1, k): gradients for each batch element

                W -= eta * jnp.mean(grads_b, axis=0)  # Update W with average over each batch
        return W

    def predict(self, data):
        ones = jnp.ones((data.shape[0], 1))
        X = jnp.concatenate((ones, data), axis=1)  # Augment to account for intercept
        W = self.coeff
        y_pred = jnp.argmax(self.sigma(X, W),
                            axis=1)  # Predicted class is largest probability returned by softmax array
        return y_pred

    def score(self, data, y_true):
        ones = jnp.ones((data.shape[0], 1))
        X = jnp.concatenate((ones, data), axis=1)
        y_pred = self.predict(data)
        acc = jnp.mean(y_pred == y_true)
        return acc

    # jitting 'sigma' is the biggest speed-up compared to the original implementation
    @partial(jit, static_argnums=0)
    def sigma(self, X, W):
        if X.ndim == 1:
            X = jnp.reshape(X, (-1, X.shape[0]))  # jax.grad seems to necessitate a reshape: X -> (1,p+1)
        s = jnp.exp(jnp.matmul(X, W))
        total = jnp.sum(s, axis=1).reshape(-1, 1)
        return s / total

    @partial(jit, static_argnums=0)
    def loss(self, W, X, y):
        f_value = self.sigma(X, W)
        loss_vector = jnp.zeros(X.shape[0])
        for k in range(self.K):
            loss_vector += jnp.log(f_value + 1e-10)[:, k] * (y == k)
        return -jnp.mean(loss_vector)