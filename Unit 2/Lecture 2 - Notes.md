
**Global Vectors for Word Representation**
- consists of a weighted least squares model that trains a global word-word co-occurrence counts -> and uses statistics to train the model 

Let $X$ denote our word-word co-occurrence matrix, where $X_{ij}$ represents the number of times a word $j$ occurs in the context of the word $i$. Let $X_i = \sum_k X_{ik}$ represents the number of times any word $k$ appears in the context of the word $i$. 

Thus, we let $P_{ik} = P(w_j | w_i) = \frac{X_{ij}}{X_i}$ be the probability of $j$ appear in the context of word $i$. 

Recall: For the skip-gram model, we use $softmax$ to compute the probability of word $j$ appearing in the context of word $i$,, thus we can calculate a similar result:
