# Jax-logistic-regression
Logistic regression classifier using JAX to support GPU acceleration.

This class is an update of a logistic regression class used in my intro to machine learning course. The major difference is the handling of the gradient descent operations,
which were rewritten using jax's grad, jit, and vmap functions. The goal with this project is speed - I've found that using JaxReg with GPU acceleration gives a ~29x speed 
increase over the original class.
