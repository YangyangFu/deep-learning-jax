# Neural Style Transfer

Given a content image and a style image, neural style transfer generates a new image that is the content of the content image and the style of the style image.
For example, make your own photo (content image) like a Van Gogh (style image) painting.

## What does ConvNet learn?
- visualize the layer in a ConvNet
    - the shallower layers learn simple features, such as edges and simple textures
    - the deeper layers learn complex features, such as complex textures and object parts

## Cost Function

- Cost function: content cost function + style cost function
  - For a content image C, and a style image S, the generated image is G
  - to quantify how good is the generated image G, we have to define a cost function
    - $J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S, G)$
- Find the generated image G that minimizes the cost function J(G)
  - initialize G randomly
  - use gradient descent to minimize J(G)
    - $G := G - \frac{\partial J(G)}{\partial G}$
- Content cost function
    - use a hidden layer $l$ to compute the content cost
      - $l$ is usually chosen to be a layer in the middle of the network, to capture both low-level and high-level features
    - use a pretrained ConvNet (e.g., VGG network)
    - let $a^{[l](C)}$ and $a^{[l](G)}$ be the activation of layer $l$ on the images C and G, respectively
    - if $a^{[l](C)}$ and $a^{[l](G)}$ are similar, then the generated image G has the same content as the content image C
    - the cost function is then defined as:
      - for chosen layer$l$, $J^{[l]}_{content}(C, G) = \frac{1}{2} ||a^{[l](C)} - a^{[l](G)}||^2$

- Style cost function
  - use layer l's activation to measure "style"
  - define the style as correlation between activations across channels
  - style matrix: $\mathcal{G}$ is also called a "Gram matrix" in kernel methods
    - $\mathcal{G}^{[l]}$ is the style matrix of layer $l$, $(n_c, n_c)$
    - $a^{[l]}_{i,j,k}$ is the activation at position $(i,j,k)$ in layer $l$, $(i,j,k)$ are the height, width, and channel indices
    - $\mathcal{G}^{[l]}_{kk'} = \sum_{i=1}^{n_H^{[l]}} \sum_{j=1}^{n_W^{[l]}} a^{[l]}_{i,j,k} a^{[l]}_{i,j,k'}$
    - style matrix for style image $\mathcal{G}^{[l](S)}$, style matrix for generated image $\mathcal{G}^{[l](G)}$
  - the cost function is then defined as
    - for each layer $l$, $J^{[l]}_{style}(S, G) = \frac{1}{(2n_H^{[l]}n_W^{[l]}n_C^{[l]})^2} \sum_{k} \sum_{k'} (\mathcal{G}^{[l](S)}_{kk'} - \mathcal{G}^{[l](G)}_{kk'})^2$ 
      - Note the constant $\frac{1}{(2n_H^{[l]}n_W^{[l]}n_C^{[l]})^2}$ is to make sure that the magnitude of the terms in the style cost function are not too large. It doenst matter as the cost is weighted at the top level using parameter $\beta$
    - total cost: summation over all layers
      - $J_{style}(S, G) = \sum_l \lambda^{[l]} J^{[l]}_{style}(S, G)$
      - $\lambda^{[l]}$ is hyperparameter to control the contribution of each layer to the total cost