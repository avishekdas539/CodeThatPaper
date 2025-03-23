# Fully Connected Linear/ Dense Layer From Scratch - Only ```NumPy```

## Maths Behind Dense Layer
### Forward Pass

A fully connected linear layer consists of weights and biases with activation function. The shape information of different matrices are given in the table below.


| Parameter | Notation | Shape | Description |
| ----------| -------- | ----- | ----------- |
| Input Dimension | $n_i$ | $int \space (1,) $ |Dimension of input vector. |
| Output Dimension | $n_o$ | $int \space (1,) $ |Dimension of output vector. |
| Weight | $W$ | $(n_o, n_i)$ | Weight matrix of the linear layer. |
| Bias | $b$ | $(n_o, 1)$ | Bias matrix of the linear layer. |
| Input Vector | $X$ | $(n_i, m)$ | Input Vector on which the linear operation will be performed. |
| Pre Activation Output | $Z$ | $(n_o, m)$ | Output vector before applying activation. |
| Output Vector| $Y$ | $(n_o, m)$ | Output vector after all linear transformation and activation. |


The equations for the forward pass of vector $X$ is shown below.

$Z_{(n_o, m)} = W_{(n_o, n_i)} \times X_{(n_i,m)} + b_{(n_o,1)}$

$Y_{(n_o,m)} = f(Z_{(n_o,m)})$

```python
class FullyConnectedLayer:

    def __init__(self, n_i : int, n_o : int, activation):
        self.w = np.random.randn(n_o, n_i) * (2/n_i)
        self.b = np.random.randn(n_o, 1) * (2/n_i)
        self.activation = activation

    def forward(self, x : np.ndarray):
        z = np.matmul(self.w, x) + self.b
        y = self.activation.forward(z)
        return y, z
```

### Back Propagration

The back propagation of loss and gradient update is considered as gradient descent methodology. A general notation for a ```FullyConnectedLayer``` of parameter updates using gradient descent is shown below.

| Parameter | Notation | Shape | Description |
| ----------| -------- | ----- | ----------- |
| ```y_prev``` | $Y_{(n-1)}$ | $(n_i,m)$ | Input vector to the layer. |
| ```z``` | $Z_n$ | $(n_o,m)$ | Pre Activation output of current layer. |
| ```prev_grad_z``` | $\frac {\partial L} {\partial Z_{[n+1]}}$ | $(n_o,m)$ | Gradient of loss function with respect to $Z_{(n+1)}$, next layer's pre activation output. |
| ```next_w``` | $W_{(n+1)}$ | $(n_o,n_{o_{next}})$ |Weight matrix of next layer. |
| ```curr_grad_z``` | $\frac {\partial L} {\partial Z_{[n]}}$ | $(n_o,m)$ | Gradient of loss function with respect to $Z_{(n)}$, current layer's pre activation output. |
| ```dw``` | $\frac {\partial L} {\partial W_{[n]}}$ | $(n_o, n_i)$ | Gradient of loss w.r.t current layer's weight. |
| ```db``` | $\frac {\partial L} {\partial b_{[n]}}$ | $(n_o, 1)$ | Gradient of loss w.r.t current layer's bias. |
| ```lr``` | $\eta$ | $float (1,)$ | Learning rate for parameter update. |


$\frac {\partial L} {\partial Z_{[n]}}$ =  $(W_{(n+1)}^T \times \frac {\partial L} {\partial Z_{[n+1]}}) \odot \frac {\partial f} {\partial Z_{[n]}}$ ...... If intermediate layer

$\frac {\partial L} {\partial Z_{[n]}}$ =  $\frac {\partial L} {\partial Z_{[n+1]}} \odot \frac {\partial f} {\partial Z_{[n]}}$ ...... If last layer

$\frac {\partial L} {\partial W_{[n]}} = \frac {1} {m} * (\frac {\partial L} {\partial Z_{[n]}} \times Y_{(n-1)})$

$\frac {\partial L} {\partial b_{[n]}} = \frac {1} {m} * \sum_{axis=1} \frac {\partial L} {\partial Z_{[n]}}$

$W_{new} = W_{old} - \eta \frac {\partial L} {\partial W_{[n]}}$

$b_{new} = b_{old} - \eta \frac {\partial L} {\partial b_{[n]}}$

```python
class FullyConnectedLayer:

    def backward(self, y_prev : np.ndarray, 
                z : np.ndarray, 
                prev_grad_z : np.ndarray, 
                next_w : np.ndarray = None, 
                lr=0.001
        ):
        if next_w is not None: # next layer has weight means, this is not the last layer
            curr_grad_z = np.matmul(next_w.T, prev_grad_z) * self.activation.grad(z)
        else: # this is the last layer
            curr_grad_z = prev_grad_z * self.activation.grad(z)
        
        m = y_prev.shape[1]

        dw = 1/m * np.matmul(curr_grad_z, y_prev.T)
        db = 1/m * np.sum(curr_grad_z, axis=1, keepdims=True)

        self.w = self.w - lr * dw
        self.b = self.b - lr * db
        
        return curr_grad_z
```