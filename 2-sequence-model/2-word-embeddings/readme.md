# NLP and Word Embeddings

## Word Representation

- one-hot vector:
  - cons:
    - large vector size
    - no similarity between words
  - pros:
    - easy to implement
    - easy to compute 
- word embedding - featurized representation:
  - learn a vector representation for each word, such as categories of words, or similarity between words

## Using word embeddings
- learn word embeddings from large text corpus (1 - 100B words)
  - or download pre-trained embedding online
- transfer embedding to new task with smaller training set
  - e.g., say, 100k words
- optionally, continue to finetune the word embeddings with new data

relations to face recognition:
- face recognition:
  - using Seasame Net to learning image features, econded by a 128-d vector
- word embedding:
  - learn a 300-d feature vector for each word in the vocabulary


here an example table for word embedding for words `king`, `queen`, `man`, `woman`:

 | --      | Man (5391) | Woman (5375) | King (3064) | Queen (3673) | Apple (6473) | Orange (7650) |
 | ------- | ---------- | ------------ | ----------- | ------------ | ------------ | ------------- |
 | Gender  | -1         | 1            | -0.95       | 0.97         | -0.06        | 0.05          |
 | Royalty | 0.03       | 0.04         | 0.96        | 0.96         | -0.21        | -0.24         |
 | Age     | 0.07       | 0.06         | 0.8         | 0.76         | 0.03         | 0.05          |
 | Food    | 0.01       | 0.02         | 0.01        | 0.02         | 0.91         | 0.89          |

Analogies using word vectors:
- t-SNE visualization of word embeddings:
  - mapping 300-d word vectors to 2-d space
- find the similarity of words
  - what is a possible word for this equation: $e_{man} - e_{woman} \approx e_{king} - e_{?}$
  - find the word that can maximize the similarity: $\argmax_{w} similarity(e_w, e_{king} - (e_{man}- e_{woman}))$
- similarity 
  - cosine similarity: $sim(e_{1}, e_{2}) = cos(\theta) = \frac{e_{1} \cdot e_{2}}{||e_{1}|| ||e_{2}||}$
    - 1: similar
    - -1: opposite

## Embedding matrix

- embedding matrix: $E \in \mathbb{R}^{n_{e} \times n_{v}}$
  - $n_{v}$: vocabulary size, e.g., 10000
  - $n_{e}$: embedding size, e.g., 300
  - $e_{i}$: $i$-th column of $E$, and is the embedding vector for word $i$
  - $e_{i} = E^TO_i$, where $O_i$ is the one-hot vector for word $i$

## Learn word embeddings

General Procedure:
- define vocabulary size $n_{v}$ and embedding size $n_{e}$
- for each word $i$, set $O_i$ to be the one-hot vector corresponding to the word
- define context and target words -> embedding vectors
  - e.g., using previous 4 words as context and predict the next word
  - the features are the context words, and the label is the target word
  - the feature size can be $4 \times n_{e}$
- softmax output
  - neuron size is $n_{v}$


context/target pairs:
- context: 
  - last 4 words
  - 4 words on left and right
  - last 1 word
  - nearby 1 word: skip-gram model
    - skip 1 word, and predict the target


## Word2Vec

instead of choosing the last 4 words or nearby 1 word as the context, we can randomly choose a word from the context words, and predict the target word.

Model:
- vocabulary size: $n_{v}$, embedding size: $n_{e}$, contxt vector $O_c$
- $O_c \rightarrow C \rightarrow e_c \rightarrow softmax \rightarrow \hat{y}$
- softmax:
  - $p(t|c) = \frac{e^{\theta_t^T e_c}}{\sum_{j}^{n_v} e^{\theta_{j}^T e_c}}$
  - $\theta_t$ is the parameters associated with output $t$
- loss function 
  - $L(\hat{y}, y) = -\sum_{i=1}^{n_v} y_i \log \hat{y}_i$
  - ideally, $y_i = 1$ if $i$ is the target word, otherwise $y_i = 0$ 
  - Problems:
    - the softmax is expensive to compute as the vocabulary size is large
    - solutions
      - hierarchical softmax: use a binary tree to represent the vocabulary. common words are near the root, and rare words are near the leaves
      - negative sampling: 

## Negative sampling

define a new learning problem
- input: context word $c$, target word $t$ (positive example), $k$ random word $t'$ (negative example) from the vocabulary
  - k is 5-20 for small dataset, 2-5 for large dataset


Model:
- softmax: $p(t|c) = \frac{e^{\theta_t^T e_c}}{\sum_{j}^{n_v} e^{\theta_{j}^T e_c}}$
- logistic:
  - $p(y=1|c,t) = \sigma(\theta_t^T e_c)$
- 10000 binary classification problems
  - during training, we can reduce to k+1 logistic regression classifiers

Negative sampling:
- how to select negative samples
  - choose based on frequency
    - results in frequent words, like, `the`, `of`, `a`, not good enough
  - a simple modification,
    - $p(w_i) = \frac{f(w_i)^{0.75}}{\sum_{j=1}^{n_v} f(w_j)^{0.75}}$
    - $f(w_i)$ is the frequency of word $w_i$

## GloVe word vectors
An even simpler algorithm for learning word embeddings - Global vectors for word representation
- for $(c, t)$
- $X_{ij}$: number of times word $j$ appears in the context of word $i$

Model:
- minimize the difference bewteen $\theta_i^T e_j$ and $\log X_{ij}$
  - $\min \sum_{i}^{n_v} \sum_{j}^{n_v} f(X_{ij})(\theta_i^T e_j + b_i + b_j^{'} - \log X_{ij})^2$
  - $f(X_{ij})$ serves as weights. 0 if $X_{ij} = 0$, 1 if $X_{ij} > 0$

