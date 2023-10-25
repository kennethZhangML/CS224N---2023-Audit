
1.1 **Representing Words**
Definition: $words \in V$ where words are elements of a finite vocabulary
	- word tokens are **instances** of the word type
	- use the same representation for any occurrence of a word token in context

1.2 **Vector Representation of Independent Components**
- set of 1-hot or standard basis vectors
- $v_{tea} = [0, 0, 1, ..., 0]^T = e_3$
- $v_{coffee} = [0, 0, 0, 1, ..., 0]^T = e_4$ 

1.3 **Distributional Semantics and Word2Vec**
Distributional Hypothesis: The meaning of a word can be derived from the distribution of contexts in which it appears; or, words similar to a given word will have similar distributions of surrounding words, but how does one represent this encoding and learn it?

1.3.1 **Co-occurrence Matrices and Document Context**
For a given finite vocabulary $V$ we make a matrix of size $|V| \times |V|$ of zeros. 
- Then, we walk through a sequence of documents, where for each:
	- for each word, $w$, in a doc, add all counts of the other words $w'$ to the row corresponding to $w$ at the column corresponding to $w'$
- Finally, normalize the rows by the sum 

We will call this a **co-occurrence matrix** $X$.
The word embedding $X_{tea} \in R^{|v|}$, thus, in the co-occurrence matrix has more value than a 1-hot encoding using the standard basis vector. 

Additionally, we could instead say that a word $w'$ is only co-occurring with $w$ if that $w'$ appears much closer within a few words: shorter windows encode syntactic properties
e.g., nouns tend to appear right next to $the$ or $is$ 
e.g., document-level windows represent what kinds of documents they appear in 

**Drawbacks of Co-occurrence Matrix**: emphasizing importance of common words
	- taking log token frequency 

1.4 **Word2Vec**
Representing each word in a fixed vocabulary as low-dim vectors and learns value of each word's vector to be predictive via a simple function of the distribution of words in a short context

1.4.1 SkipGram Word2Vec
We have a finite vocabulary $V$ and let $C \in V$ represent a centre word and $O \in V$ represent a word appearing in the context of the centre word. 
Let $U \in R^{|v| \times d}$ and $V \in R^{|V| \times d}$ where each word is associated with a single row of $U$ and one of $V$.

Thus, the probabilistic model is a follows: $p_{U, V} (o | c) = \frac{exp u_0^T v_c}{\sum_{w \in V} exp u_w^T v_c}$ 
- Similar to $Softmax$, where we take arbitrary scores and produce probability distribution where larger-scored elements get higher probabilities 
- vector probabilities over all words given centre word $c$ is $p_{U, V} (. | c) \in R^{|V|}$ 
- Estimation: cross-entropy loss with true distribution $P*(O | C)$ 
	- $min_{U, V} E_{o, c} [- log p_{U, V} (o | c)]$, where we minimize with respect to parameters $U, V$ 
	- the expectation over values of $o, c$ drawn from the distributions of $O, V$ the negative log probability under $(U, V)$ model of that value of $o$ given that value of $c''$ 

1.5 **Estimating Word2Vec over a Corpus**
(1) How to calculate expectation (minimize cross-entropy loss) 
	Empirical Loss: Given $D$, a set of documents, where each document is a sequence of words $w_1^{(d)}, ..., w_m^{(d)}$ with all $w \in V$, we let $k \in N_{++}$ be a positive integer window size.
	$O$ takes on the value of each word $w_i$ in each document 
	- for each such $w_i$ the outside words are ${w_{i - k}, ..., w_{i - 1}, w_{i + 1}, ..., w_{i + k}}$ 
	- $L(U, V) = \sum_{d \in D} sum_{i = 1}^m \sum_{j = 1}^k - log p_{U, V} (w_{i - j}^{(d)} | w_i^{(d)})$ 
		- Taking sum over all documents, over all words in the document, over all words in the occurring window of the likelihood of the outside word, given centre word 
	- **Two Ways to Minimize:**
		- Find good $U, V$ for objective of maximally increasing the value of $f$ 
		- Initialize $U^{(0)}, V^{(0)}$ along $N(0, 0.0001)^{|V| \times d}$ 
		- Perform number of iterations for $U^{(i + 1)} = U^{(i)} - \alpha \triangledown_{u} L(U^{(i)}, V^{(i)})$ 
	- Stochastic Gradients
		- approximate $L(U, V)$ for a few samples for each step
			- Thus, we sample documents $d_1, ..., d_l$ ~ $D$ and computing
				- $L(U, V) = \sum_{d_1, ..., d_l} \sum_{i = 1}^m \sum_{j = 1}^k -log P_{U, V} (w_{i - j}^{(d)} | w_i^{(d)})$ 

1.5 (Extended) Working through a gradient
Compute partial gradient of loss with respect to $v_c$ for single instance of $c$, the centre word
- pass gradient operator through sums
- Goal: "observed" - "expected"
	- we have the vector for the word actually observed $u_c$ 
	- $v_c$ is a vector for the word that is observed than the word expected 

	$L(U, V) = \sum_{d_1, ..., d_l} \sum_{i = 1}^m \sum_{j = 1}^k -log P_{U, V} (w_{i - j}^{(d)} | w_i^{(d)})$ 
	$\implies \triangledown_{v_c} log \frac{exp u_o^T v_c}{\sum_{i = 1}^n u_w^T v_c}$ = $\triangledown_{v_c} log \times exp \times u_o^T v_c - \triangledown_{v_c} log \sum_{i = 1}^n exp \times u_w^T v_c$ 
	$\implies \triangledown log \times exp \times u_o^T v_c = \triangledown_{v_i} \times u_o^T v_c$
	Then, differentiate using log and chain rule to get the derivative of the dot product of the inverse relation between our sums of the exponential vector matrix multiplications over the sums of our inverse $softmax$, which results in "observed" - "expected" word vectors.
	 

