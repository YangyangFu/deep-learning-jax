# Sequence to Sequence Architectures

## Basic Models

- encoder
  - pretrained features/embeddings
  - e.g. CNN
- decoder
  - trainable with given data
  - e.g., RNN

**Language model**

**Machine translation model**:

Instead of generating "random" sequences in typical language model, machine translation tries to find the output sequence given the input sequence to maximize the joint conditional probability of the output sequence given the input sequence:
- $\max p(y^{<1>, ..., y^{<T_y>}} | x)$
- beam search
  - step 1: find the first word $p(\hat y^{<1>} | x)$. Instead of choosing only the first highest probability word, we can choose the top $k$ words with the highest probability. $k$ is also called the beam width.
  - step 2: find the second word $p(\hat y^{<2>} | x, \hat y^{<1>})$ based on the top $k$ words from step 1.
    - we get for the first two words; $p(\hat y^{<1>}, \hat y^{<2>} | x) = p(\hat y^{<1>} | x) p(\hat y^{<2>} | x, \hat y^{<1>})$
  - repeat until end

**Why not Greedy Search**
- generate the first word with the highest probability, then the second word with the highest probability, and so on
  - pick one word at each step is not guaranteed to give the best sequence as it will mostly like choose the most common next word based on training data


## Beam Search Modifications

Length normalization
- for each step in beam search we maximize the probability of the sequence up to that step
  - $\max \Pi_{t=1}^{T_y} P(y^{<t>} | x, y^{<1>}, ..., y^{<t-1>})$
  - with a log function, this becomes 
    - $\max \sum_{t=1}^{T_y} \log P(y^{<t>} | x, y^{<1>}, ..., y^{<t-1>})$
- add a length normalization term to the objective function
  - $\max \frac{1}{T_y^{\alpha}} \sum_{t=1}^{T_y} \log P(y^{<t>} | x, y^{<1>}, ..., y^{<t-1>})$
  - $\alpha$ is a hyperparameter
    - $\alpha = 0$ is no length normalization
    - $\alpha = 1$ is full length normalization
    - need to be tuned

## Error Analysis in Beam Search

Since beam search is not guaranteed to give the best sequence, we can use error analysis to find out what went wrong and improve the model. 
- check the probability against the ground truth: $P(y^* | x)$ vs $P(\hat y | x)$ 
  - if $P(y^* | x) > P(\hat y | x)$, then the model is not confident enough to give the correct answer. Thus the beam search is at fault. we can increase the beam width $k$.
  - if $P(y^* | x) < P(\hat y | x)$, then the model is confident but wrong. Thus the RNN model is at fault. we can improve the model.

## BLEU Score
Sometimes there are multiple equal best translations, then which one to choose? BLEU score is a metric to evaluate the quality of the translation.
- for example
  - human reference 1: the cat on the mat
  - human reference 2: there is a cat on the mat
  - machine translation: the the the the the the the
- precision
  - $p_n = \frac{\sum_{i=1}^{m} \sum_{n-gram \in y_i} Count_{clip}(n-gram)}{\sum_{i=1}^{m} \sum_{n-gram \in y_i} Count(n-gram)}$
  - machine translation in the above example: $\frac{2}{7}$

## Attention Model 

Problem of long sequences:
- RNNs have a hard time to learn from long sequences, the BLEU score decreases as the length of the sequence increases

**Attention model**:
- input sequence $x = (x^{<1>}, ..., x^{<T_x>})$
- pre-attention bidirectional RNN as encoder
  - output hidden states from both directions: $a^{<t'>} = [\overrightarrow{a}^{<t'>}, \overleftarrow{a}^{<t'>}]$
  - $t' \in [1, T_x]$
- attention model
  - idea: for each output step $t$, we look at all the input steps $t'$ and decide how much attention to pay to each input step
  - inputs: 
    - $s^{<t-1>}$ - hidden state of the decoder at time step $t-1$, where $t \in [1, T_y]$
    - $a = [a^{<1>}, ..., a^{<T_x>}]$ 
  - attention weights
    - $\alpha^{<t, t'>} = \frac{\exp(e^{<t, t'>})}{\sum_{t'=1}^{T_x} \exp(e^{<t, t'>})}$ 
      - how much attention to pay when generating $y^{<t>}$ based on $x^{<t'>}$
      - sum of all attention weights for each time step is 1
    - $e^{<t, t'>} = NN(s^{<t-1>}, a^{<t'>})$, where $e$ is the energy
      - $s^{<t-1>}$ is the hidden state of the decoder at time step $t-1$
      - $a^{<t'>}$ is the hidden state of the encoder at time step $t'$
      - $NN$ is a neural network
  - output: context vector
    - $c^{<t>} = \sum_{t'=1}^{T_x} \alpha^{<t, t'>} a^{<t'>}$
- post-attention RNN as a decoder 
  - $s^{<t>}, \hat y^{<t>} = \text{RNN}(s^{<t-1>}, c^{<t>})$
  - $s$ is the hidden state, (although, we use $a$ for hidden state in the above, to distinguish friom attention weights, we use $s$ here)
  - note that, unlike typical decoder structure, the post-attention RNN does not take $\hat y^{<t-1>}$ as input


The `detailed architecture of attention model` is as follows:
- for each time step $t$ in the decoder
  - Inputs: 
    - hiddent state from decoder at previous time step: $s^{<t-1>}$, $\rightarrow$ shape $(b, n_s,)$, where 'b' is batch
    - hidden states from encoder for all encoder timesteps: $a = [a^{<1>}, ..., a^{<T_x>}]$, $\rightarrow$ shape $(b, T_x, n_a)$ 
  - Dense layer to get $e^{<t>} = [e^{<t, 1>}, ..., e^{<t, T_x>}]$
    - concatenate $s^{<t-1>}$ with each $a^{<t'>}$, $\rightarrow$ shape $(b, T_x, n_s + n_a)$
    - $e^{<t, t'>} = NN(s^{<t-1>}, a^{<t'>})$
    - the shared dense layer is used for all $t'$, which is in total $T_x$ times
    - output a scalar $e^{<t, t'>}$ for each $t'$, $\rightarrow$ shape $(b, T_x)$
  - Softmax layer
    - connect each $e^{<t, t'>}$ to a softmax layer, therefore the input shape for the softmax layer is $(b, T_x)$
    - output $\alpha^{<t, t'>}$ for each $t'$ $\rightarrow$ shape $(b, T_x)$
  - Sum up the weighted $a^{<t'>}$ to get the context vector $c^{<t>}$
    - $c^{<t>} = \sum_{t'=1}^{T_x} \alpha^{<t, t'>} a^{<t'>}$
  - Output $c^{<t>}$ $\rightarrow$ shape $(b, n_a)$
 
## Applications

**Speech Recognition**
- x: audio clip
- y: text transcript
- spectrogram
  - x-axis: time
  - y-axis: frequency
  - color: amplitude


**Trigger Word Detection**
wake up device when a trigger word is detected
- e.g., hello siri, etc
- approach
  - set the target label to 1 for the time steps of the trigger word, and 0 for the rest
    - this could work
    - but this would lead to imbalanced data as most of the time steps are 0
  - instead of set 1 for a single time step, we can set 1 to a fixed time steps before and after the trigger word
    - this would lead to a more balanced data
    - but this would lead to a lot of false positives 
- **how is this similar to object dection in computer vision?**