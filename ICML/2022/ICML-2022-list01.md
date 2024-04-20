## [0] PAC-Bayesian Bounds on Rate-Efficient Classifiers

        **Authors**: *Alhabib Abbas, Yiannis Andreopoulos*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/abbas22a.html](https://proceedings.mlr.press/v162/abbas22a.html)

        **Abstract**:

        We derive analytic bounds on the noise invariance of majority vote classifiers operating on compressed inputs. Specifically, starting from recent bounds on the true risk of majority vote classifiers, we extend the applicability of PAC-Bayesian theory to quantify the resilience of majority votes to input noise stemming from compression. The derived bounds are intuitive in binary classification settings, where they can be measured as expressions of voter differentials and voter pair agreement. By combining measures of input distortion with analytic guarantees on noise invariance, we prescribe rate-efficient machines to compress inputs without affecting subsequent classification. Our validation shows how bounding noise invariance can inform the compression stage for any majority vote classifier such that worst-case implications of bad input reconstructions are known, and inputs can be compressed to the minimum amount of information needed prior to inference.

        ----

        ## [1] Sharp-MAML: Sharpness-Aware Model-Agnostic Meta Learning

        **Authors**: *Momin Abbas, Quan Xiao, Lisha Chen, Pin-Yu Chen, Tianyi Chen*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/abbas22b.html](https://proceedings.mlr.press/v162/abbas22b.html)

        **Abstract**:

        Model-agnostic meta learning (MAML) is currently one of the dominating approaches for few-shot meta-learning. Albeit its effectiveness, the optimization of MAML can be challenging due to the innate bilevel problem structure. Specifically, the loss landscape of MAML is much more complex with possibly  more saddle points and local minimizers than its empirical risk minimization counterpart. To address this challenge, we leverage the recently invented sharpness-aware minimization and develop a sharpness-aware MAML approach that we term Sharp-MAML. We empirically demonstrate that Sharp-MAML and its computation-efficient variant can outperform the plain-vanilla MAML baseline (e.g., +3% accuracy on Mini-Imagenet). We complement the empirical study with the convergence rate analysis and the generalization bound of Sharp-MAML. To the best of our knowledge, this is the first empirical and theoretical study on sharpness-aware minimization in the context of bilevel learning.

        ----

        ## [2] An Initial Alignment between Neural Network and Target is Needed for Gradient Descent to Learn

        **Authors**: *Emmanuel Abbe, Elisabetta Cornacchia, Jan Hazla, Christopher Marquis*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/abbe22a.html](https://proceedings.mlr.press/v162/abbe22a.html)

        **Abstract**:

        This paper introduces the notion of “Initial Alignment” (INAL) between a neural network at initialization and a target function. It is proved that if a network and a Boolean target function do not have a noticeable INAL, then noisy gradient descent with normalized i.i.d. initialization will not learn in polynomial time. Thus a certain amount of knowledge about the target (measured by the INAL) is needed in the architecture design. This also provides an answer to an open problem posed in (AS-NeurIPS’20). The results are based on deriving lower-bounds for descent algorithms on symmetric neural networks without explicit knowledge of the target function beyond its INAL.

        ----

        ## [3] Active Sampling for Min-Max Fairness

        **Authors**: *Jacob D. Abernethy, Pranjal Awasthi, Matthäus Kleindessner, Jamie Morgenstern, Chris Russell, Jie Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/abernethy22a.html](https://proceedings.mlr.press/v162/abernethy22a.html)

        **Abstract**:

        We propose simple active sampling and reweighting strategies for optimizing min-max fairness that can be applied to any classification or regression model learned via loss minimization. The key intuition behind our approach is to use at each timestep a datapoint from the group that is worst off under the current model for updating the model. The ease of implementation and the generality of our robust formulation make it an attractive option for improving model performance on disadvantaged groups. For convex learning problems, such as linear or logistic regression, we provide a fine-grained analysis, proving the rate of convergence to a min-max fair solution.

        ----

        ## [4] Meaningfully debugging model mistakes using conceptual counterfactual explanations

        **Authors**: *Abubakar Abid, Mert Yüksekgönül, James Zou*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/abid22a.html](https://proceedings.mlr.press/v162/abid22a.html)

        **Abstract**:

        Understanding and explaining the mistakes made by trained models is critical to many machine learning objectives, such as improving robustness, addressing concept drift, and mitigating biases. However, this is often an ad hoc process that involves manually looking at the model’s mistakes on many test samples and guessing at the underlying reasons for those incorrect predictions. In this paper, we propose a systematic approach, conceptual counterfactual explanations (CCE), that explains why a classifier makes a mistake on a particular test sample(s) in terms of human-understandable concepts (e.g. this zebra is misclassified as a dog because of faint stripes). We base CCE on two prior ideas: counterfactual explanations and concept activation vectors, and validate our approach on well-known pretrained models, showing that it explains the models’ mistakes meaningfully. In addition, for new models trained on data with spurious correlations, CCE accurately identifies the spurious correlation as the cause of model mistakes from a single misclassified test sample. On two challenging medical applications, CCE generated useful insights, confirmed by clinicians, into biases and mistakes the model makes in real-world settings. The code for CCE is publicly available and can easily be applied to explain mistakes in new models.

        ----

        ## [5] Batched Dueling Bandits

        **Authors**: *Arpit Agarwal, Rohan Ghuge, Viswanath Nagarajan*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/agarwal22a.html](https://proceedings.mlr.press/v162/agarwal22a.html)

        **Abstract**:

        The K-armed dueling bandit problem, where the feedback is in the form of noisy pairwise comparisons, has been widely studied. Previous works have only focused on the sequential setting where the policy adapts after every comparison. However, in many applications such as search ranking and recommendation systems, it is preferable to perform comparisons in a limited number of parallel batches. We study the batched K-armed dueling bandit problem under two standard settings: (i) existence of a Condorcet winner, and (ii) strong stochastic transitivity and stochastic triangle inequality. For both settings, we obtain algorithms with a smooth trade-off between the number of batches and regret. Our regret bounds match the best known sequential regret bounds (up to poly-logarithmic factors), using only a logarithmic number of batches. We complement our regret analysis with a nearly-matching lower bound. Finally, we also validate our theoretical results via experiments on synthetic and real data.

        ----

        ## [6] Hierarchical Shrinkage: Improving the accuracy and interpretability of tree-based models

        **Authors**: *Abhineet Agarwal, Yan Shuo Tan, Omer Ronen, Chandan Singh, Bin Yu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/agarwal22b.html](https://proceedings.mlr.press/v162/agarwal22b.html)

        **Abstract**:

        Decision trees and random forests (RF) are a cornerstone of modern machine learning practice. Due to their tendency to overfit, trees are typically regularized by a variety of techniques that modify their structure (e.g. pruning). We introduce Hierarchical Shrinkage (HS), a post-hoc algorithm which regularizes the tree not by altering its structure, but by shrinking the prediction over each leaf toward the sample means over each of its ancestors, with weights depending on a single regularization parameter and the number of samples in each ancestor. Since HS is a post-hoc method, it is extremely fast, compatible with any tree-growing algorithm and can be used synergistically with other regularization techniques. Extensive experiments over a wide variety of real-world datasets show that HS substantially increases the predictive performance of decision trees even when used in conjunction with other regularization techniques. Moreover, we find that applying HS to individual trees in a RF often improves its accuracy and interpretability by simplifying and stabilizing decision boundaries and SHAP values. We further explain HS by showing that it to be equivalent to ridge regression on a basis that is constructed of decision stumps associated to the internal nodes of a tree. All code and models are released in a full-fledged package available on Github

        ----

        ## [7] Deep equilibrium networks are sensitive to initialization statistics

        **Authors**: *Atish Agarwala, Samuel S. Schoenholz*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/agarwala22a.html](https://proceedings.mlr.press/v162/agarwala22a.html)

        **Abstract**:

        Deep equilibrium networks (DEQs) are a promising way to construct models which trade off memory for compute. However, theoretical understanding of these models is still lacking compared to traditional networks, in part because of the repeated application of a single set of weights. We show that DEQs are sensitive to the higher order statistics of the matrix families from which they are initialized. In particular, initializing with orthogonal or symmetric matrices allows for greater stability in training. This gives us a practical prescription for initializations which allow for training with a broader range of initial weight scales.

        ----

        ## [8] Learning of Cluster-based Feature Importance for Electronic Health Record Time-series

        **Authors**: *Henrique Aguiar, Mauro D. Santos, Peter J. Watkinson, Tingting Zhu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/aguiar22a.html](https://proceedings.mlr.press/v162/aguiar22a.html)

        **Abstract**:

        The recent availability of Electronic Health Records (EHR) has allowed for the development of algorithms predicting inpatient risk of deterioration and trajectory evolution. However, prediction of disease progression with EHR is challenging since these data are sparse, heterogeneous, multi-dimensional, and multi-modal time-series. As such, clustering is regularly used to identify similar groups within the patient cohort to improve prediction. Current models have shown some success in obtaining cluster representations of patient trajectories. However, they i) fail to obtain clinical interpretability for each cluster, and ii) struggle to learn meaningful cluster numbers in the context of imbalanced distribution of disease outcomes. We propose a supervised deep learning model to cluster EHR data based on the identification of clinically understandable phenotypes with regard to both outcome prediction and patient trajectory. We introduce novel loss functions to address the problems of class imbalance and cluster collapse, and furthermore propose a feature-time attention mechanism to identify cluster-based phenotype importance across time and feature dimensions. We tested our model in two datasets corresponding to distinct medical settings. Our model yielded added interpretability to cluster formation and outperformed benchmarks by at least 4% in relevant metrics.

        ----

        ## [9] On the Convergence of the Shapley Value in Parametric Bayesian Learning Games

        **Authors**: *Lucas Agussurja, Xinyi Xu, Bryan Kian Hsiang Low*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/agussurja22a.html](https://proceedings.mlr.press/v162/agussurja22a.html)

        **Abstract**:

        Measuring contributions is a classical problem in cooperative game theory where the Shapley value is the most well-known solution concept. In this paper, we establish the convergence property of the Shapley value in parametric Bayesian learning games where players perform a Bayesian inference using their combined data, and the posterior-prior KL divergence is used as the characteristic function. We show that for any two players, under some regularity conditions, their difference in Shapley value converges in probability to the difference in Shapley value of a limiting game whose characteristic function is proportional to the log-determinant of the joint Fisher information. As an application, we present an online collaborative learning framework that is asymptotically Shapley-fair. Our result enables this to be achieved without any costly computations of posterior-prior KL divergences. Only a consistent estimator of the Fisher information is needed. The effectiveness of our framework is demonstrated with experiments using real-world data.

        ----

        ## [10] Individual Preference Stability for Clustering

        **Authors**: *Saba Ahmadi, Pranjal Awasthi, Samir Khuller, Matthäus Kleindessner, Jamie Morgenstern, Pattara Sukprasert, Ali Vakilian*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ahmadi22a.html](https://proceedings.mlr.press/v162/ahmadi22a.html)

        **Abstract**:

        In this paper, we propose a natural notion of individual preference (IP) stability for clustering, which asks that every data point, on average, is closer to the points in its own cluster than to the points in any other cluster. Our notion can be motivated from several perspectives, including game theory and algorithmic fairness. We study several questions related to our proposed notion. We first show that deciding whether a given data set allows for an IP-stable clustering in general is NP-hard. As a result, we explore the design of efficient algorithms for finding IP-stable clusterings in some restricted metric spaces. We present a polytime algorithm to find a clustering satisfying exact IP-stability on the real line, and an efficient algorithm to find an IP-stable 2-clustering for a tree metric. We also consider relaxing the stability constraint, i.e., every data point should not be too far from its own cluster compared to any other cluster. For this case, we provide polytime algorithms with different guarantees. We evaluate some of our algorithms and several standard clustering approaches on real data sets.

        ----

        ## [11] Understanding the unstable convergence of gradient descent

        **Authors**: *Kwangjun Ahn, Jingzhao Zhang, Suvrit Sra*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ahn22a.html](https://proceedings.mlr.press/v162/ahn22a.html)

        **Abstract**:

        Most existing analyses of (stochastic) gradient descent rely on the condition that for $L$-smooth costs, the step size is less than $2/L$. However, many works have observed that in machine learning applications step sizes often do not fulfill this condition, yet (stochastic) gradient descent still converges, albeit in an unstable manner. We investigate this unstable convergence phenomenon from first principles, and discuss key causes behind it. We also identify its main characteristics, and how they interrelate based on both theory and experiments, offering a principled view toward understanding the phenomenon.

        ----

        ## [12] Minimum Cost Intervention Design for Causal Effect Identification

        **Authors**: *Sina Akbari, Jalal Etesami, Negar Kiyavash*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/akbari22a.html](https://proceedings.mlr.press/v162/akbari22a.html)

        **Abstract**:

        Pearl’s do calculus is a complete axiomatic approach to learn the identifiable causal effects from observational data. When such an effect is not identifiable, it is necessary to perform a collection of often costly interventions in the system to learn the causal effect. In this work, we consider the problem of designing the collection of interventions with the minimum cost to identify the desired effect. First, we prove that this prob-em is NP-complete, and subsequently propose an algorithm that can either find the optimal solution or a logarithmic-factor approximation of it. This is done by establishing a connection between our problem and the minimum hitting set problem. Additionally, we propose several polynomial time heuristic algorithms to tackle the computational complexity of the problem. Although these algorithms could potentially stumble on sub-optimal solutions, our simulations show that they achieve small regrets on random graphs.

        ----

        ## [13] How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models

        **Authors**: *Ahmed M. Alaa, Boris van Breugel, Evgeny S. Saveliev, Mihaela van der Schaar*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/alaa22a.html](https://proceedings.mlr.press/v162/alaa22a.html)

        **Abstract**:

        Devising domain- and model-agnostic evaluation metrics for generative models is an important and as yet unresolved problem. Most existing metrics, which were tailored solely to the image synthesis setup, exhibit a limited capacity for diagnosing the different modes of failure of generative models across broader application domains. In this paper, we introduce a 3-dimensional evaluation metric, ($\alpha$-Precision, $\beta$-Recall, Authenticity), that characterizes the fidelity, diversity and generalization performance of any generative model in a domain-agnostic fashion. Our metric unifies statistical divergence measures with precision-recall analysis, enabling sample- and distribution-level diagnoses of model fidelity and diversity. We introduce generalization as an additional, independent dimension (to the fidelity-diversity trade-off) that quantifies the extent to which a model copies training data{—}a crucial performance indicator when modeling sensitive data with requirements on privacy. The three metric components correspond to (interpretable) probabilistic quantities, and are estimated via sample-level binary classification. The sample-level nature of our metric inspires a novel use case which we call model auditing, wherein we judge the quality of individual samples generated by a (black-box) model, discarding low-quality samples and hence improving the overall model performance in a post-hoc manner.

        ----

        ## [14] A Natural Actor-Critic Framework for Zero-Sum Markov Games

        **Authors**: *Ahmet Alacaoglu, Luca Viano, Niao He, Volkan Cevher*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/alacaoglu22a.html](https://proceedings.mlr.press/v162/alacaoglu22a.html)

        **Abstract**:

        We introduce algorithms based on natural actor-critic and analyze their sample complexity for solving two player zero-sum Markov games in the tabular case. Our results improve the best-known sample complexities of policy gradient/actor-critic methods for convergence to Nash equilibrium in the multi-agent setting. We use the error propagation scheme in approximate dynamic programming, recent advances for global convergence of policy gradient methods, temporal difference learning, and techniques from stochastic primal-dual optimization. Our algorithms feature two stages, requiring agents to agree on an etiquette before starting their interactions, which is feasible for instance in self-play. However, the agents only access to joint reward and joint next state and not to each other’s actions or policies. Our complexity results match the best-known results for global convergence of policy gradient algorithms for single agent RL. We provide numerical verification of our methods for a two player bandit environment and a two player game, Alesia. We observe improved empirical performance as compared to the recently proposed optimistic gradient descent-ascent variant for Markov games.

        ----

        ## [15] Deploying Convolutional Networks on Untrusted Platforms Using 2D Holographic Reduced Representations

        **Authors**: *Mohammad Mahmudul Alam, Edward Raff, Tim Oates, James Holt*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/alam22a.html](https://proceedings.mlr.press/v162/alam22a.html)

        **Abstract**:

        Due to the computational cost of running inference for a neural network, the need to deploy the inferential steps on a third party’s compute environment or hardware is common. If the third party is not fully trusted, it is desirable to obfuscate the nature of the inputs and outputs, so that the third party can not easily determine what specific task is being performed. Provably secure protocols for leveraging an untrusted party exist but are too computational demanding to run in practice. We instead explore a different strategy of fast, heuristic security that we call Connectionist Symbolic Pseudo Secrets. By leveraging Holographic Reduced Representations (HRRs), we create a neural network with a pseudo-encryption style defense that empirically shows robustness to attack, even under threat models that unrealistically favor the adversary.

        ----

        ## [16] Optimistic Linear Support and Successor Features as a Basis for Optimal Policy Transfer

        **Authors**: *Lucas Nunes Alegre, Ana L. C. Bazzan, Bruno C. da Silva*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/alegre22a.html](https://proceedings.mlr.press/v162/alegre22a.html)

        **Abstract**:

        In many real-world applications, reinforcement learning (RL) agents might have to solve multiple tasks, each one typically modeled via a reward function. If reward functions are expressed linearly, and the agent has previously learned a set of policies for different tasks, successor features (SFs) can be exploited to combine such policies and identify reasonable solutions for new problems. However, the identified solutions are not guaranteed to be optimal. We introduce a novel algorithm that addresses this limitation. It allows RL agents to combine existing policies and directly identify optimal policies for arbitrary new problems, without requiring any further interactions with the environment. We first show (under mild assumptions) that the transfer learning problem tackled by SFs is equivalent to the problem of learning to optimize multiple objectives in RL. We then introduce an SF-based extension of the Optimistic Linear Support algorithm to learn a set of policies whose SFs form a convex coverage set. We prove that policies in this set can be combined via generalized policy improvement to construct optimal behaviors for any new linearly-expressible tasks, without requiring any additional training samples. We empirically show that our method outperforms state-of-the-art competing algorithms both in discrete and continuous domains under value function approximation.

        ----

        ## [17] Structured Stochastic Gradient MCMC

        **Authors**: *Antonios Alexos, Alex J. Boyd, Stephan Mandt*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/alexos22a.html](https://proceedings.mlr.press/v162/alexos22a.html)

        **Abstract**:

        Stochastic gradient Markov Chain Monte Carlo (SGMCMC) is a scalable algorithm for asymptotically exact Bayesian inference in parameter-rich models, such as Bayesian neural networks. However, since mixing can be slow in high dimensions, practitioners often resort to variational inference (VI). Unfortunately, VI makes strong assumptions on both the factorization and functional form of the posterior. To relax these assumptions, this work proposes a new non-parametric variational inference scheme that combines ideas from both SGMCMC and coordinate-ascent VI. The approach relies on a new Langevin-type algorithm that operates on a "self-averaged" posterior energy function, where parts of the latent variables are averaged over samples from earlier iterations of the Markov chain. This way, statistical dependencies between coordinates can be broken in a controlled way, allowing the chain to mix faster. This scheme can be further modified in a "dropout" manner, leading to even more scalability. We test our scheme for ResNet-20 on CIFAR-10, SVHN, and FMNIST. In all cases, we find improvements in convergence speed and/or final accuracy compared to SGMCMC and parametric VI.

        ----

        ## [18] XAI for Transformers: Better Explanations through Conservative Propagation

        **Authors**: *Ameen Ali, Thomas Schnake, Oliver Eberle, Grégoire Montavon, Klaus-Robert Müller, Lior Wolf*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ali22a.html](https://proceedings.mlr.press/v162/ali22a.html)

        **Abstract**:

        Transformers have become an important workhorse of machine learning, with numerous applications. This necessitates the development of reliable methods for increasing their transparency. Multiple interpretability methods, often based on gradient information, have been proposed. We show that the gradient in a Transformer reflects the function only locally, and thus fails to reliably identify the contribution of input features to the prediction. We identify Attention Heads and LayerNorm as main reasons for such unreliable explanations and propose a more stable way for propagation through these layers. Our proposal, which can be seen as a proper extension of the well-established LRP method to Transformers, is shown both theoretically and empirically to overcome the deficiency of a simple gradient-based approach, and achieves state-of-the-art explanation performance on a broad range of Transformer models and datasets.

        ----

        ## [19] RUMs from Head-to-Head Contests

        **Authors**: *Matteo Almanza, Flavio Chierichetti, Ravi Kumar, Alessandro Panconesi, Andrew Tomkins*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/almanza22a.html](https://proceedings.mlr.press/v162/almanza22a.html)

        **Abstract**:

        Random utility models (RUMs) encode the likelihood that a particular item will be selected from a slate of competing items. RUMs are well-studied objects in both discrete choice theory and, more recently, in the machine learning community, as they encode a fairly broad notion of rational user behavior. In this paper, we focus on slates of size two representing head-to-head contests. Given a tournament matrix $M$ such that $M_{i,j}$ is the probability that item $j$ will be selected from $\{i, j\}$, we consider the problem of finding the RUM that most closely reproduces $M$. For this problem we obtain a polynomial-time algorithm returning a RUM that approximately minimizes the average error over the pairs. Our experiments show that RUMs can perfectly represent many of the tournament matrices that have been considered in the literature; in fact, the maximum average error induced by RUMs on the matrices we considered is negligible ($\approx 0.001$). We also show that RUMs are competitive, on prediction tasks, with previous approaches.

        ----

        ## [20] Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval

        **Authors**: *Uri Alon, Frank F. Xu, Junxian He, Sudipta Sengupta, Dan Roth, Graham Neubig*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/alon22a.html](https://proceedings.mlr.press/v162/alon22a.html)

        **Abstract**:

        Retrieval-based language models (R-LM) model the probability of natural language text by combining a standard language model (LM) with examples retrieved from an external datastore at test time. While effective, a major bottleneck of using these models in practice is the computationally costly datastore search, which can be performed as frequently as every time step. In this paper, we present RetoMaton - retrieval automaton - which approximates the datastore search, based on (1) saving pointers between consecutive datastore entries, and (2) clustering of entries into "states". This effectively results in a weighted finite automaton built on top of the datastore, instead of representing the datastore as a flat list. The creation of the automaton is unsupervised, and a RetoMaton can be constructed from any text collection: either the original training corpus or from another domain. Traversing this automaton at inference time, in parallel to the LM inference, reduces its perplexity by up to 1.85, or alternatively saves up to 83% of the nearest neighbor searches over $k$NN-LM (Khandelwal et al., 2020) without hurting perplexity. Our code and trained models are available at https://github.com/neulab/retomaton .

        ----

        ## [21] Minimax Classification under Concept Drift with Multidimensional Adaptation and Performance Guarantees

        **Authors**: *Verónica Álvarez, Santiago Mazuelas, José Antonio Lozano*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/alvarez22a.html](https://proceedings.mlr.press/v162/alvarez22a.html)

        **Abstract**:

        The statistical characteristics of instance-label pairs often change with time in practical scenarios of supervised classification. Conventional learning techniques adapt to such concept drift accounting for a scalar rate of change by means of a carefully chosen learning rate, forgetting factor, or window size. However, the time changes in common scenarios are multidimensional, i.e., different statistical characteristics often change in a different manner. This paper presents adaptive minimax risk classifiers (AMRCs) that account for multidimensional time changes by means of a multivariate and high-order tracking of the time-varying underlying distribution. In addition, differently from conventional techniques, AMRCs can provide computable tight performance guarantees. Experiments on multiple benchmark datasets show the classification improvement of AMRCs compared to the state-of-the-art and the reliability of the presented performance guarantees.

        ----

        ## [22] Scalable First-Order Bayesian Optimization via Structured Automatic Differentiation

        **Authors**: *Sebastian E. Ament, Carla P. Gomes*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ament22a.html](https://proceedings.mlr.press/v162/ament22a.html)

        **Abstract**:

        Bayesian Optimization (BO) has shown great promise for the global optimization of functions that are expensive to evaluate, but despite many successes, standard approaches can struggle in high dimensions. To improve the performance of BO, prior work suggested incorporating gradient information into a Gaussian process surrogate of the objective, giving rise to kernel matrices of size $nd$ {\texttimes} $nd$ for $n$ observations in $d$ dimensions. Naı̈vely multiplying with (resp. inverting) these matrices requires $O(n^2d^2)$ (resp. $O(n^3d^3)$) operations, which becomes infeasible for moderate dimensions and sample sizes. Here, we observe that a wide range of kernels gives rise to structured matrices, enabling an exact $O(n^2d)$ matrix-vector multiply for gradient observations and $O(n^2d^2)$ for Hessian observations. Beyond canonical kernel classes, we derive a programmatic approach to leveraging this type of structure for transformations and combinations of the discussed kernel classes, which constitutes a structure-aware automatic differentiation algorithm. Our methods apply to virtually all canonical kernels and automatically extend to complex kernels, like the neural network, radial basis function network, and spectral mixture kernels without any additional derivations, enabling flexible, problem-dependent modeling while scaling first-order BO to high $d$.

        ----

        ## [23] Public Data-Assisted Mirror Descent for Private Model Training

        **Authors**: *Ehsan Amid, Arun Ganesh, Rajiv Mathews, Swaroop Ramaswamy, Shuang Song, Thomas Steinke, Vinith M. Suriyakumar, Om Thakkar, Abhradeep Thakurta*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/amid22a.html](https://proceedings.mlr.press/v162/amid22a.html)

        **Abstract**:

        In this paper, we revisit the problem of using in-distribution public data to improve the privacy/utility trade-offs for differentially private (DP) model training. (Here, public data refers to auxiliary data sets that have no privacy concerns.) We design a natural variant of DP mirror descent, where the DP gradients of the private/sensitive data act as the linear term, and the loss generated by the public data as the mirror map. We show that, for linear regression with feature vectors drawn from a non-isotropic sub-Gaussian distribution, our algorithm, PDA-DPMD (a variant of mirror descent), provides population risk guarantees that are asymptotically better than the best known guarantees under DP (without having access to public data), when the number of public data samples is sufficiently large. We further show that our algorithm has natural “noise stability” properties that control the variance due to noise added to ensure DP. We demonstrate the efficacy of our algorithm by showing privacy/utility trade-offs on four benchmark datasets (StackOverflow, WikiText-2, CIFAR-10, and EMNIST). We show that our algorithm not only significantly improves over traditional DP-SGD, which does not have access to public data, but to our knowledge is the first to improve over DP-SGD on models that have been pre-trained with public data.

        ----

        ## [24] On Last-Iterate Convergence Beyond Zero-Sum Games

        **Authors**: *Ioannis Anagnostides, Ioannis Panageas, Gabriele Farina, Tuomas Sandholm*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/anagnostides22a.html](https://proceedings.mlr.press/v162/anagnostides22a.html)

        **Abstract**:

        Most existing results about last-iterate convergence of learning dynamics are limited to two-player zero-sum games, and only apply under rigid assumptions about what dynamics the players follow. In this paper we provide new results and techniques that apply to broader families of games and learning dynamics. First, we show that in a class of games that includes constant-sum polymatrix and strategically zero-sum games, the trajectories of dynamics such as optimistic mirror descent (OMD) exhibit a boundedness property, which holds even when players employ different algorithms and prediction mechanisms. This property enables us to obtain $O(1/\sqrt{T})$ rates and optimal $O(1)$ regret bounds. Our analysis also reveals a surprising property: OMD either reaches arbitrarily close to a Nash equilibrium or it outperforms the robust price of anarchy in efficiency. Moreover, for potential games we establish convergence to an $\epsilon$-equilibrium after $O(1/\epsilon^2)$ iterations for mirror descent under a broad class of regularizers, as well as optimal $O(1)$ regret bounds for OMD variants. Our framework also extends to near-potential games, and unifies known analyses for distributed learning in Fisher’s market model. Finally, we analyze the convergence, efficiency, and robustness of optimistic gradient descent (OGD) in general-sum continuous games.

        ----

        ## [25] Online Algorithms with Multiple Predictions

        **Authors**: *Keerti Anand, Rong Ge, Amit Kumar, Debmalya Panigrahi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/anand22a.html](https://proceedings.mlr.press/v162/anand22a.html)

        **Abstract**:

        This paper studies online algorithms augmented with multiple machine-learned predictions. We give a generic algorithmic framework for online covering problems with multiple predictions that obtains an online solution that is competitive against the performance of the best solution obtained from the predictions. Our algorithm incorporates the use of predictions in the classic potential-based analysis of online algorithms. We apply our algorithmic framework to solve classical problems such as online set cover, (weighted) caching, and online facility location in the multiple predictions setting.

        ----

        ## [26] Learning to Hash Robustly, Guaranteed

        **Authors**: *Alexandr Andoni, Daniel Beaglehole*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/andoni22a.html](https://proceedings.mlr.press/v162/andoni22a.html)

        **Abstract**:

        The indexing algorithms for the high-dimensional nearest neighbor search (NNS) with the best worst-case guarantees are based on the randomized Locality Sensitive Hashing (LSH), and its derivatives. In practice, many heuristic approaches exist to "learn" the best indexing method in order to speed-up NNS, crucially adapting to the structure of the given dataset. Oftentimes, these heuristics outperform the LSH-based algorithms on real datasets, but, almost always, come at the cost of losing the guarantees of either correctness or robust performance on adversarial queries, or apply to datasets with an assumed extra structure/model. In this paper, we design an NNS algorithm for the Hamming space that has worst-case guarantees essentially matching that of theoretical algorithms, while optimizing the hashing to the structure of the dataset (think instance-optimal algorithms) for performance on the minimum-performing query. We evaluate the algorithm’s ability to optimize for a given dataset both theoretically and practically. On the theoretical side, we exhibit a natural setting (dataset model) where our algorithm is much better than the standard theoretical one. On the practical side, we run experiments that show that our algorithm has a 1.8x and 2.1x better recall on the worst-performing queries to the MNIST and ImageNet datasets.

        ----

        ## [27] Set Based Stochastic Subsampling

        **Authors**: *Bruno Andreis, Seanie Lee, Tuan A. Nguyen, Juho Lee, Eunho Yang, Sung Ju Hwang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/andreis22a.html](https://proceedings.mlr.press/v162/andreis22a.html)

        **Abstract**:

        Deep models are designed to operate on huge volumes of high dimensional data such as images. In order to reduce the volume of data these models must process, we propose a set-based two-stage end-to-end neural subsampling model that is jointly optimized with an arbitrary downstream task network (e.g. classifier). In the first stage, we efficiently subsample candidate elements using conditionally independent Bernoulli random variables by capturing coarse grained global information using set encoding functions, followed by conditionally dependent autoregressive subsampling of the candidate elements using Categorical random variables by modeling pair-wise interactions using set attention networks in the second stage. We apply our method to feature and instance selection and show that it outperforms the relevant baselines under low subsampling rates on a variety of tasks including image classification, image reconstruction, function reconstruction and few-shot classification. Additionally, for nonparametric models such as Neural Processes that require to leverage the whole training data at inference time, we show that our method enhances the scalability of these models.

        ----

        ## [28] Towards Understanding Sharpness-Aware Minimization

        **Authors**: *Maksym Andriushchenko, Nicolas Flammarion*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/andriushchenko22a.html](https://proceedings.mlr.press/v162/andriushchenko22a.html)

        **Abstract**:

        Sharpness-Aware Minimization (SAM) is a recent training method that relies on worst-case weight perturbations which significantly improves generalization in various settings. We argue that the existing justifications for the success of SAM which are based on a PAC-Bayes generalization bound and the idea of convergence to flat minima are incomplete. Moreover, there are no explanations for the success of using m-sharpness in SAM which has been shown as essential for generalization. To better understand this aspect of SAM, we theoretically analyze its implicit bias for diagonal linear networks. We prove that SAM always chooses a solution that enjoys better generalization properties than standard gradient descent for a certain class of problems, and this effect is amplified by using m-sharpness. We further study the properties of the implicit bias on non-linear networks empirically, where we show that fine-tuning a standard model with SAM can lead to significant generalization improvements. Finally, we provide convergence results of SAM for non-convex objectives when used with stochastic gradients. We illustrate these results empirically for deep networks and discuss their relation to the generalization behavior of SAM. The code of our experiments is available at https://github.com/tml-epfl/understanding-sam.

        ----

        ## [29] Fair and Fast k-Center Clustering for Data Summarization

        **Authors**: *Haris Angelidakis, Adam Kurpisz, Leon Sering, Rico Zenklusen*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/angelidakis22a.html](https://proceedings.mlr.press/v162/angelidakis22a.html)

        **Abstract**:

        We consider two key issues faced by many clustering methods when used for data summarization, namely (a) an unfair representation of "demographic groups” and (b) distorted summarizations, where data points in the summary represent subsets of the original data of vastly different sizes. Previous work made important steps towards handling separately each of these two issues in the context of the fundamental k-Center clustering objective through the study of fast algorithms for natural models that address them. We show that it is possible to effectively address both (a) and (b) simultaneously by presenting a clustering procedure that works for a canonical combined model and (i) is fast, both in theory and practice, (ii) exhibits a worst-case constant-factor guarantee, and (iii) gives promising computational results showing that there can be significant benefits in addressing both issues together instead of sequentially.

        ----

        ## [30] Interactive Correlation Clustering with Existential Cluster Constraints

        **Authors**: *Rico Angell, Nicholas Monath, Nishant Yadav, Andrew McCallum*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/angell22a.html](https://proceedings.mlr.press/v162/angell22a.html)

        **Abstract**:

        We consider the problem of clustering with user feedback. Existing methods express constraints about the input data points, most commonly through must-link and cannot-link constraints on data point pairs. In this paper, we introduce existential cluster constraints: a new form of feedback where users indicate the features of desired clusters. Specifically, users make statements about the existence of a cluster having (and not having) particular features. Our approach has multiple advantages: (1) constraints on clusters can express user intent more efficiently than point pairs; (2) in cases where the users’ mental model is of the desired clusters, it is more natural for users to express cluster-wise preferences; (3) it functions even when privacy restrictions prohibit users from seeing raw data. In addition to introducing existential cluster constraints, we provide an inference algorithm for incorporating our constraints into the output clustering. Finally, we demonstrate empirically that our proposed framework facilitates more accurate clustering with dramatically fewer user feedback inputs.

        ----

        ## [31] Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging

        **Authors**: *Anastasios N. Angelopoulos, Amit Pal Singh Kohli, Stephen Bates, Michael I. Jordan, Jitendra Malik, Thayer Alshaabi, Srigokul Upadhyayula, Yaniv Romano*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/angelopoulos22a.html](https://proceedings.mlr.press/v162/angelopoulos22a.html)

        **Abstract**:

        Image-to-image regression is an important learning task, used frequently in biological imaging. Current algorithms, however, do not generally offer statistical guarantees that protect against a model’s mistakes and hallucinations. To address this, we develop uncertainty quantification techniques with rigorous statistical guarantees for image-to-image regression problems. In particular, we show how to derive uncertainty intervals around each pixel that are guaranteed to contain the true value with a user-specified confidence probability. Our methods work in conjunction with any base machine learning model, such as a neural network, and endow it with formal mathematical guarantees{—}regardless of the true unknown data distribution or choice of model. Furthermore, they are simple to implement and computationally inexpensive. We evaluate our procedure on three image-to-image regression tasks: quantitative phase microscopy, accelerated magnetic resonance imaging, and super-resolution transmission electron microscopy of a Drosophila melanogaster brain.

        ----

        ## [32] AdaGrad Avoids Saddle Points

        **Authors**: *Kimon Antonakopoulos, Panayotis Mertikopoulos, Georgios Piliouras, Xiao Wang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/antonakopoulos22a.html](https://proceedings.mlr.press/v162/antonakopoulos22a.html)

        **Abstract**:

        Adaptive first-order methods in optimization have widespread ML applications due to their ability to adapt to non-convex landscapes. However, their convergence guarantees are typically stated in terms of vanishing gradient norms, which leaves open the issue of converging to undesirable saddle points (or even local maxima). In this paper, we focus on the AdaGrad family of algorithms - from scalar to full-matrix preconditioning - and we examine the question of whether the method’s trajectories avoid saddle points. A major challenge that arises here is that AdaGrad’s step-size (or, more accurately, the method’s preconditioner) evolves over time in a filtration-dependent way, i.e., as a function of all gradients observed in earlier iterations; as a result, avoidance results for methods with a constant or vanishing step-size do not apply. We resolve this challenge by combining a series of step-size stabilization arguments with a recursive representation of the AdaGrad preconditioner that allows us to employ center-stable techniques and ultimately show that the induced trajectories avoid saddle points from almost any initial condition.

        ----

        ## [33] UnderGrad: A Universal Black-Box Optimization Method with Almost Dimension-Free Convergence Rate Guarantees

        **Authors**: *Kimon Antonakopoulos, Dong Quan Vu, Volkan Cevher, Kfir Y. Levy, Panayotis Mertikopoulos*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/antonakopoulos22b.html](https://proceedings.mlr.press/v162/antonakopoulos22b.html)

        **Abstract**:

        Universal methods achieve optimal convergence rate guarantees in convex optimization without any prior knowledge of the problem’s regularity parameters or the attributes of the gradient oracle employed by the method. In this regard, existing state-of-the-art algorithms achieve an $O(1/T^2)$ convergence rate in Lipschitz smooth problems with a perfect gradient oracle, and an $O(1/sqrt{T})$ convergence speed when the underlying problem is non-smooth and/or the gradient oracle is stochastic. On the downside, these methods do not take into account the dependence of these guarantees on the problem’s dimensionality, and this can have a catastrophic impact on a method’s convergence, in both theory and practice. Our paper aims to bridge this gap by providing a scalable universal method - dubbed UnDERGrad - which enjoys an almost dimension-free oracle complexity in problems with a favorable geometry (like the simplex, $\ell_1$-ball or trace-constraints), while retaining the order-optimal dependence on T described above. These "best of both worlds" guarantees are achieved via a primal-dual update scheme inspired by the dual exploration method for variational inequalities.

        ----

        ## [34] Adapting the Linearised Laplace Model Evidence for Modern Deep Learning

        **Authors**: *Javier Antorán, David Janz, James Urquhart Allingham, Erik A. Daxberger, Riccardo Barbano, Eric T. Nalisnick, José Miguel Hernández-Lobato*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/antoran22a.html](https://proceedings.mlr.press/v162/antoran22a.html)

        **Abstract**:

        The linearised Laplace method for estimating model uncertainty has received renewed attention in the Bayesian deep learning community. The method provides reliable error bars and admits a closed-form expression for the model evidence, allowing for scalable selection of model hyperparameters. In this work, we examine the assumptions behind this method, particularly in conjunction with model selection. We show that these interact poorly with some now-standard tools of deep learning–stochastic approximation methods and normalisation layers–and make recommendations for how to better adapt this classic method to the modern setting. We provide theoretical support for our recommendations and validate them empirically on MLPs, classic CNNs, residual networks with and without normalisation layers, generative autoencoders and transformers.

        ----

        ## [35] EAT-C: Environment-Adversarial sub-Task Curriculum for Efficient Reinforcement Learning

        **Authors**: *Shuang Ao, Tianyi Zhou, Jing Jiang, Guodong Long, Xuan Song, Chengqi Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ao22a.html](https://proceedings.mlr.press/v162/ao22a.html)

        **Abstract**:

        Reinforcement learning (RL) is inefficient on long-horizon tasks due to sparse rewards and its policy can be fragile to slightly perturbed environments. We address these challenges via a curriculum of tasks with coupled environments, generated by two policies trained jointly with RL: (1) a co-operative planning policy recursively decomposing a hard task into a coarse-to-fine sub-task tree; and (2) an adversarial policy modifying the environment in each sub-task. They are complementary to acquire more informative feedback for RL: (1) provides dense reward of easier sub-tasks while (2) modifies sub-tasks’ environments to be more challenging and diverse. Conversely, they are trained by RL’s dense feedback on sub-tasks so their generated curriculum keeps adaptive to RL’s progress. The sub-task tree enables an easy-to-hard curriculum for every policy: its top-down construction gradually increases sub-tasks the planner needs to generate, while the adversarial training between the environment and RL follows a bottom-up traversal that starts from a dense sequence of easier sub-tasks allowing more frequent environment changes. We compare EAT-C with RL/planning targeting similar problems and methods with environment generators or adversarial agents. Extensive experiments on diverse tasks demonstrate the advantages of our method on improving RL’s efficiency and generalization.

        ----

        ## [36] Online Balanced Experimental Design

        **Authors**: *David Arbour, Drew Dimmery, Tung Mai, Anup B. Rao*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/arbour22a.html](https://proceedings.mlr.press/v162/arbour22a.html)

        **Abstract**:

        We consider the experimental design problem in an online environment, an important practical task for reducing the variance of estimates in randomized experiments which allows for greater precision, and in turn, improved decision making. In this work, we present algorithms that build on recent advances in online discrepancy minimization which accommodate both arbitrary treatment probabilities and multiple treatments. The proposed algorithms are computational efficient, minimize covariate imbalance, and include randomization which enables robustness to misspecification. We provide worst case bounds on the expected mean squared error of the causal estimate and show that the proposed estimator is no worse than an implicit ridge regression, which are within a logarithmic factor of the best known results for offline experimental design. We conclude with a detailed simulation study showing favorable results relative to complete randomization as well as to offline methods for experimental design with time complexities exceeding our algorithm, which has a linear dependence on the number of observations, by polynomial factors.

        ----

        ## [37] VariGrow: Variational Architecture Growing for Task-Agnostic Continual Learning based on Bayesian Novelty

        **Authors**: *Randy Ardywibowo, Zepeng Huo, Zhangyang Wang, Bobak J. Mortazavi, Shuai Huang, Xiaoning Qian*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ardywibowo22a.html](https://proceedings.mlr.press/v162/ardywibowo22a.html)

        **Abstract**:

        Continual Learning (CL) is the problem of sequentially learning a set of tasks and preserving all the knowledge acquired. Many existing methods assume that the data stream is explicitly divided into a sequence of known contexts (tasks), and use this information to know when to transfer knowledge from one context to another. Unfortunately, many real-world CL scenarios have no clear task nor context boundaries, motivating the study of task-agnostic CL, where neither the specific tasks nor their switches are known both in training and testing. This paper proposes a variational architecture growing framework dubbed VariGrow. By interpreting dynamically growing neural networks as a Bayesian approximation, and defining flexible implicit variational distributions, VariGrow detects if a new task is arriving through an energy-based novelty score. If the novelty score is high and the sample is “detected" as a new task, VariGrow will grow a new expert module to be responsible for it. Otherwise, the sample will be assigned to one of the existing experts who is most “familiar" with it (i.e., one with the lowest novelty score). We have tested VariGrow on several CIFAR and ImageNet-based benchmarks for the strict task-agnostic CL setting and demonstrate its consistent superior performance. Perhaps surprisingly, its performance can even be competitive compared to task-aware methods.

        ----

        ## [38] Thresholded Lasso Bandit

        **Authors**: *Kaito Ariu, Kenshi Abe, Alexandre Proutière*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ariu22a.html](https://proceedings.mlr.press/v162/ariu22a.html)

        **Abstract**:

        In this paper, we revisit the regret minimization problem in sparse stochastic contextual linear bandits, where feature vectors may be of large dimension $d$, but where the reward function depends on a few, say $s_0\ll d$, of these features only. We present Thresholded Lasso bandit, an algorithm that (i) estimates the vector defining the reward function as well as its sparse support, i.e., significant feature elements, using the Lasso framework with thresholding, and (ii) selects an arm greedily according to this estimate projected on its support. The algorithm does not require prior knowledge of the sparsity index $s_0$ and can be parameter-free under some symmetric assumptions. For this simple algorithm, we establish non-asymptotic regret upper bounds scaling as $\mathcal{O}( \log d + \sqrt{T} )$ in general, and as $\mathcal{O}( \log d + \log T)$ under the so-called margin condition (a probabilistic condition on the separation of the arm rewards). The regret of previous algorithms scales as $\mathcal{O}( \log d + \sqrt{T \log (d T)})$ and $\mathcal{O}( \log T \log d)$ in the two settings, respectively. Through numerical experiments, we confirm that our algorithm outperforms existing methods.

        ----

        ## [39] Gradient Based Clustering

        **Authors**: *Aleksandar Armacki, Dragana Bajovic, Dusan Jakovetic, Soummya Kar*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/armacki22a.html](https://proceedings.mlr.press/v162/armacki22a.html)

        **Abstract**:

        We propose a general approach for distance based clustering, using the gradient of the cost function that measures clustering quality with respect to cluster assignments and cluster center positions. The approach is an iterative two step procedure (alternating between cluster assignment and cluster center updates) and is applicable to a wide range of functions, satisfying some mild assumptions. The main advantage of the proposed approach is a simple and computationally cheap update rule. Unlike previous methods that specialize to a specific formulation of the clustering problem, our approach is applicable to a wide range of costs, including non-Bregman clustering methods based on the Huber loss. We analyze the convergence of the proposed algorithm, and show that it converges to the set of appropriately defined fixed points, under arbitrary center initialization. In the special case of Bregman cost functions, the algorithm converges to the set of centroidal Voronoi partitions, which is consistent with prior works. Numerical experiments on real data demonstrate the effectiveness of the proposed method.

        ----

        ## [40] Understanding Gradient Descent on the Edge of Stability in Deep Learning

        **Authors**: *Sanjeev Arora, Zhiyuan Li, Abhishek Panigrahi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/arora22a.html](https://proceedings.mlr.press/v162/arora22a.html)

        **Abstract**:

        Deep learning experiments by \citet{cohen2021gradient} using deterministic Gradient Descent (GD) revealed an Edge of Stability (EoS) phase when learning rate (LR) and sharpness (i.e., the largest eigenvalue of Hessian) no longer behave as in traditional optimization. Sharpness stabilizes around $2/$LR and loss goes up and down across iterations, yet still with an overall downward trend. The current paper mathematically analyzes a new mechanism of implicit regularization in the EoS phase, whereby GD updates due to non-smooth loss landscape turn out to evolve along some deterministic flow on the manifold of minimum loss. This is in contrast to many previous results about implicit bias either relying on infinitesimal updates or noise in gradient. Formally, for any smooth function $L$ with certain regularity condition, this effect is demonstrated for (1) Normalized GD, i.e., GD with a varying LR $\eta_t =\frac{\eta}{\norm{\nabla L(x(t))}}$ and loss $L$; (2) GD with constant LR and loss $\sqrt{L- \min_x L(x)}$. Both provably enter the Edge of Stability, with the associated flow on the manifold minimizing $\lambda_{1}(\nabla^2 L)$. The above theoretical results have been corroborated by an experimental study.

        ----

        ## [41] Private optimization in the interpolation regime: faster rates and hardness results

        **Authors**: *Hilal Asi, Karan N. Chadha, Gary Cheng, John C. Duchi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/asi22a.html](https://proceedings.mlr.press/v162/asi22a.html)

        **Abstract**:

        In non-private stochastic convex optimization, stochastic gradient methods converge much faster on interpolation problems—namely, problems where there exists a solution that simultaneously minimizes all of the sample losses—than on non-interpolating ones; similar improvements are not known in the private setting. In this paper, we investigate differentially private stochastic optimization in the interpolation regime. First, we show that without additional assumptions, interpolation problems do not exhibit an improved convergence rates with differential privacy. However, when the functions exhibit quadratic growth around the optimum, we show (near) exponential improvements in the private sample complexity. In particular, we propose an adaptive algorithm that improves the sample complexity to achieve expected error $\alpha$ from $\frac{d}{\diffp \sqrt{\alpha}}$ to $\frac{1}{\alpha^\rho} + \frac{d}{\diffp} \log\paren{\frac{1}{\alpha}}$ for any fixed $\rho >0$, while retaining the standard minimax-optimal sample complexity for non-interpolation problems. We prove a lower bound that shows the dimension-dependent term in the expression above is tight. Furthermore, we provide a superefficiency result which demonstrates the necessity of the polynomial term for adaptive algorithms: any algorithm that has a polylogarithmic sample complexity for interpolation problems cannot achieve the minimax-optimal rates for the family of non-interpolation problems.

        ----

        ## [42] Optimal Algorithms for Mean Estimation under Local Differential Privacy

        **Authors**: *Hilal Asi, Vitaly Feldman, Kunal Talwar*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/asi22b.html](https://proceedings.mlr.press/v162/asi22b.html)

        **Abstract**:

        We study the problem of mean estimation of $\ell_2$-bounded vectors under the constraint of local differential privacy. While the literature has a variety of algorithms that achieve the (asymptotic) optimal rates for this problem, the performance of these algorithms in practice can vary significantly due to varying (and often large) hidden constants. In this work, we investigate the question of designing the randomizer with the smallest variance. We show that PrivUnit (Bhowmick et al. 2018) with optimized parameters achieves the optimal variance among a large family of natural randomizers. To prove this result, we establish some properties of local randomizers, and use symmetrization arguments that allow us to write the optimal randomizer as the optimizer of a certain linear program. These structural results, which should extend to other problems, then allow us to show that the optimal randomizer belongs to the PrivUnit family. We also develop a new variant of PrivUnit based on the Gaussian distribution which is more amenable to mathematical analysis and enjoys the same optimality guarantees. This allows us to establish several useful properties on the exact constants of the optimal error as well as to numerically estimate these constants.

        ----

        ## [43] Asymptotically-Optimal Gaussian Bandits with Side Observations

        **Authors**: *Alexia Atsidakou, Orestis Papadigenopoulos, Constantine Caramanis, Sujay Sanghavi, Sanjay Shakkottai*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/atsidakou22a.html](https://proceedings.mlr.press/v162/atsidakou22a.html)

        **Abstract**:

        We study the problem of Gaussian bandits with general side information, as first introduced by Wu, Szepesvári, and György. In this setting, the play of an arm reveals information about other arms, according to an arbitrary a priori known side information matrix: each element of this matrix encodes the fidelity of the information that the “row" arm reveals about the “column" arm. In the case of Gaussian noise, this model subsumes standard bandits, full-feedback, and graph-structured feedback as special cases. In this work, we first construct an LP-based asymptotic instance-dependent lower bound on the regret. The LP optimizes the cost (regret) required to reliably estimate the suboptimality gap of each arm. This LP lower bound motivates our main contribution: the first known asymptotically optimal algorithm for this general setting.

        ----

        ## [44] Congested Bandits: Optimal Routing via Short-term Resets

        **Authors**: *Pranjal Awasthi, Kush Bhatia, Sreenivas Gollapudi, Kostas Kollias*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/awasthi22a.html](https://proceedings.mlr.press/v162/awasthi22a.html)

        **Abstract**:

        For traffic routing platforms, the choice of which route to recommend to a user depends on the congestion on these routes – indeed, an individual’s utility depends on the number of people using the recommended route at that instance. Motivated by this, we introduce the problem of Congested Bandits where each arm’s reward is allowed to depend on the number of times it was played in the past $\Delta$ timesteps. This dependence on past history of actions leads to a dynamical system where an algorithm’s present choices also affect its future pay-offs, and requires an algorithm to plan for this. We study the congestion aware formulation in the multi-armed bandit (MAB) setup and in the contextual bandit setup with linear rewards. For the multi-armed setup, we propose a UCB style algorithm and show that its policy regret scales as $\tilde{O}(\sqrt{K \Delta T})$. For the linear contextual bandit setup, our algorithm, based on an iterative least squares planner, achieves policy regret $\tilde{O}(\sqrt{dT} + \Delta)$. From an experimental standpoint, we corroborate the no-regret properties of our algorithms via a simulation study.

        ----

        ## [45] Do More Negative Samples Necessarily Hurt In Contrastive Learning?

        **Authors**: *Pranjal Awasthi, Nishanth Dikkala, Pritish Kamath*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/awasthi22b.html](https://proceedings.mlr.press/v162/awasthi22b.html)

        **Abstract**:

        Recent investigations in noise contrastive estimation suggest, both empirically as well as theoretically, that while having more “negative samples” in the contrastive loss improves downstream classification performance initially, beyond a threshold, it hurts downstream performance due to a “collision-coverage” trade-off. But is such a phenomenon inherent in contrastive learning? We show in a simple theoretical setting, where positive pairs are generated by sampling from the underlying latent class (introduced by Saunshi et al. (ICML 2019)), that the downstream performance of the representation optimizing the (population) contrastive loss in fact does not degrade with the number of negative samples. Along the way, we give a structural characterization of the optimal representation in our framework, for noise contrastive estimation. We also provide empirical support for our theoretical results on CIFAR-10 and CIFAR-100 datasets.

        ----

        ## [46] H-Consistency Bounds for Surrogate Loss Minimizers

        **Authors**: *Pranjal Awasthi, Anqi Mao, Mehryar Mohri, Yutao Zhong*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/awasthi22c.html](https://proceedings.mlr.press/v162/awasthi22c.html)

        **Abstract**:

        We present a detailed study of estimation errors in terms of surrogate loss estimation errors. We refer to such guarantees as H-consistency bounds, since they account for the hypothesis set H adopted. These guarantees are significantly stronger than H-calibration or H-consistency. They are also more informative than similar excess error bounds derived in the literature, when H is the family of all measurable functions. We prove general theorems providing such guarantees, for both the distribution-dependent and distribution-independent settings. We show that our bounds are tight, modulo a convexity assumption. We also show that previous excess error bounds can be recovered as special cases of our general results. We then present a series of explicit bounds in the case of the zero-one loss, with multiple choices of the surrogate loss and for both the family of linear functions and neural networks with one hidden-layer. We further prove more favorable distribution-dependent guarantees in that case. We also present a series of explicit bounds in the case of the adversarial loss, with surrogate losses based on the supremum of the $\rho$-margin, hinge or sigmoid loss and for the same two general hypothesis sets. Here too, we prove several enhancements of these guarantees under natural distributional assumptions. Finally, we report the results of simulations illustrating our bounds and their tightness.

        ----

        ## [47] Iterative Hard Thresholding with Adaptive Regularization: Sparser Solutions Without Sacrificing Runtime

        **Authors**: *Kyriakos Axiotis, Maxim Sviridenko*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/axiotis22a.html](https://proceedings.mlr.press/v162/axiotis22a.html)

        **Abstract**:

        We propose a simple modification to the iterative hard thresholding (IHT) algorithm, which recovers asymptotically sparser solutions as a function of the condition number. When aiming to minimize a convex function f(x) with condition number $\kappa$ subject to x being an s-sparse vector, the standard IHT guarantee is a solution with relaxed sparsity $O(s\kappa^2)$, while our proposed algorithm, regularized IHT, returns a solution with sparsity $O(s\kappa)$. Our algorithm significantly improves over ARHT [Axiotis & Sviridenko, 2021] which also achieves $O(s\kappa)$, as it does not require re-optimization in each iteration (and so is much faster), is deterministic, and does not require knowledge of the optimal solution value f(x*) or the optimal sparsity level s. Our main technical tool is an adaptive regularization framework, in which the algorithm progressively learns the weights of an l_2 regularization term that will allow convergence to sparser solutions. We also apply this framework to low rank optimization, where we achieve a similar improvement of the best known condition number dependence from $\kappa^2$ to $\kappa$.

        ----

        ## [48] Proving Theorems using Incremental Learning and Hindsight Experience Replay

        **Authors**: *Eser Aygün, Ankit Anand, Laurent Orseau, Xavier Glorot, Stephen Marcus McAleer, Vlad Firoiu, Lei M. Zhang, Doina Precup, Shibl Mourad*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/aygun22a.html](https://proceedings.mlr.press/v162/aygun22a.html)

        **Abstract**:

        Traditional automated theorem proving systems for first-order logic depend on speed-optimized search and many handcrafted heuristics designed to work over a wide range of domains. Machine learning approaches in the literature either depend on these traditional provers to bootstrap themselves, by leveraging these heuristics, or can struggle due to limited existing proof data. The latter issue can be explained by the lack of a smooth difficulty gradient in theorem proving datasets; large gaps in difficulty between different theorems can make training harder or even impossible. In this paper, we adapt the idea of hindsight experience replay from reinforcement learning to the automated theorem proving domain, so as to use the intermediate data generated during unsuccessful proof attempts. We build a first-order logic prover by disabling all the smart clause-scoring heuristics of the state-of-the-art E prover and replacing them with a clause-scoring neural network learned by using hindsight experience replay in an incremental learning setting. Clauses are represented as graphs and presented to transformer networks with spectral features. We show that provers trained in this way can outperform previous machine learning approaches and compete with the state of the art heuristic-based theorem prover E in its best configuration, on the popular benchmarks MPTP2078, M2k and Mizar40. The proofs generated by our algorithm are also almost always significantly shorter than E’s proofs.

        ----

        ## [49] Near-optimal rate of consistency for linear models with missing values

        **Authors**: *Alexis Ayme, Claire Boyer, Aymeric Dieuleveut, Erwan Scornet*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ayme22a.html](https://proceedings.mlr.press/v162/ayme22a.html)

        **Abstract**:

        Missing values arise in most real-world data sets due to the aggregation of multiple sources and intrinsically missing information (sensor failure, unanswered questions in surveys...). In fact, the very nature of missing values usually prevents us from running standard learning algorithms. In this paper, we focus on the extensively-studied linear models, but in presence of missing values, which turns out to be quite a challenging task. Indeed, the Bayes predictor can be decomposed as a sum of predictors corresponding to each missing pattern. This eventually requires to solve a number of learning tasks, exponential in the number of input features, which makes predictions impossible for current real-world datasets. First, we propose a rigorous setting to analyze a least-square type estimator and establish a bound on the excess risk which increases exponentially in the dimension. Consequently, we leverage the missing data distribution to propose a new algorithm, and derive associated adaptive risk bounds that turn out to be minimax optimal. Numerical experiments highlight the benefits of our method compared to state-of-the-art algorithms used for predictions with missing values.

        ----

        ## [50] How Tempering Fixes Data Augmentation in Bayesian Neural Networks

        **Authors**: *Gregor Bachmann, Lorenzo Noci, Thomas Hofmann*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bachmann22a.html](https://proceedings.mlr.press/v162/bachmann22a.html)

        **Abstract**:

        While Bayesian neural networks (BNNs) provide a sound and principled alternative to standard neural networks, an artificial sharpening of the posterior usually needs to be applied to reach comparable performance. This is in stark contrast to theory, dictating that given an adequate prior and a well-specified model, the untempered Bayesian posterior should achieve optimal performance. Despite the community’s extensive efforts, the observed gains in performance still remain disputed with several plausible causes pointing at its origin. While data augmentation has been empirically recognized as one of the main drivers of this effect, a theoretical account of its role, on the other hand, is largely missing. In this work we identify two interlaced factors concurrently influencing the strength of the cold posterior effect, namely the correlated nature of augmentations and the degree of invariance of the employed model to such transformations. By theoretically analyzing simplified settings, we prove that tempering implicitly reduces the misspecification arising from modeling augmentations as i.i.d. data. The temperature mimics the role of the effective sample size, reflecting the gain in information provided by the augmentations. We corroborate our theoretical findings with extensive empirical evaluations, scaling to realistic BNNs. By relying on the framework of group convolutions, we experiment with models of varying inherent degree of invariance, confirming its hypothesized relationship with the optimal temperature.

        ----

        ## [51] ASAPSGD: Instance-based Adaptiveness to Staleness in Asynchronous SGD

        **Authors**: *Karl Bäckström, Marina Papatriantafilou, Philippas Tsigas*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/backstrom22a.html](https://proceedings.mlr.press/v162/backstrom22a.html)

        **Abstract**:

        Concurrent algorithmic implementations of Stochastic Gradient Descent (SGD) give rise to critical questions for compute-intensive Machine Learning (ML). Asynchrony implies speedup in some contexts, and challenges in others, as stale updates may lead to slower, or non-converging executions. While previous works showed asynchrony-adaptiveness can improve stability and speedup by reducing the step size for stale updates according to static rules, there is no one-size-fits-all adaptation rule, since the optimal strategy depends on several factors. We introduce (i) $\mathtt{ASAP.SGD}$, an analytical framework capturing necessary and desired properties of staleness-adaptive step size functions and (ii) \textsc{tail}-$\tau$, a method for utilizing key properties of the execution instance, generating a tailored strategy that not only dampens the impact of stale updates, but also leverages fresh ones. We recover convergence bounds for adaptiveness functions satisfying the $\mathtt{ASAP.SGD}$ conditions for general, convex and non-convex problems, and establish novel bounds for ones satisfying the Polyak-Lojasiewicz property. We evaluate \textsc{tail}-$\tau$ with representative AsyncSGD concurrent algorithms, for Deep Learning problems, showing \textsc{tail}-$\tau$ is a vital complement to AsyncSGD, with (i) persistent speedup in wall-clock convergence time in the parallelism spectrum, (ii) considerably lower risk of non-convergence, as well as (iii) precision levels for which original SGD implementations fail.

        ----

        ## [52] From Noisy Prediction to True Label: Noisy Prediction Calibration via Generative Model

        **Authors**: *HeeSun Bae, Seungjae Shin, Byeonghu Na, JoonHo Jang, Kyungwoo Song, Il-Chul Moon*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bae22a.html](https://proceedings.mlr.press/v162/bae22a.html)

        **Abstract**:

        Noisy labels are inevitable yet problematic in machine learning society. It ruins the generalization of a classifier by making the classifier over-fitted to noisy labels. Existing methods on noisy label have focused on modifying the classifier during the training procedure. It has two potential problems. First, these methods are not applicable to a pre-trained classifier without further access to training. Second, it is not easy to train a classifier and regularize all negative effects from noisy labels, simultaneously. We suggest a new branch of method, Noisy Prediction Calibration (NPC) in learning with noisy labels. Through the introduction and estimation of a new type of transition matrix via generative model, NPC corrects the noisy prediction from the pre-trained classifier to the true label as a post-processing scheme. We prove that NPC theoretically aligns with the transition matrix based methods. Yet, NPC empirically provides more accurate pathway to estimate true label, even without involvement in classifier learning. Also, NPC is applicable to any classifier trained with noisy label methods, if training instances and its predictions are available. Our method, NPC, boosts the classification performances of all baseline models on both synthetic and real-world datasets. The implemented code is available at https://github.com/BaeHeeSun/NPC.

        ----

        ## [53] data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language

        **Authors**: *Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/baevski22a.html](https://proceedings.mlr.press/v162/baevski22a.html)

        **Abstract**:

        While the general idea of self-supervised learning is identical across modalities, the actual algorithms and objectives differ widely because they were developed with a single modality in mind. To get us closer to general self-supervised learning, we present data2vec, a framework that uses the same learning method for either speech, NLP or computer vision. The core idea is to predict latent representations of the full input data based on a masked view of the input in a self-distillation setup using a standard Transformer architecture. Instead of predicting modality-specific targets such as words, visual tokens or units of human speech which are local in nature, data2vec predicts contextualized latent representations that contain information from the entire input. Experiments on the major benchmarks of speech recognition, image classification, and natural language understanding demonstrate a new state of the art or competitive performance to predominant approaches.

        ----

        ## [54] End-to-End Balancing for Causal Continuous Treatment-Effect Estimation

        **Authors**: *Mohammad Taha Bahadori, Eric Tchetgen Tchetgen, David Heckerman*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bahadori22a.html](https://proceedings.mlr.press/v162/bahadori22a.html)

        **Abstract**:

        We study the problem of observational causal inference with continuous treatment. We focus on the challenge of estimating the causal response curve for infrequently-observed treatment values. We design a new algorithm based on the framework of entropy balancing which learns weights that directly maximize causal inference accuracy using end-to-end optimization. Our weights can be customized for different datasets and causal inference algorithms. We propose a new theory for consistency of entropy balancing for continuous treatments. Using synthetic and real-world data, we show that our proposed algorithm outperforms the entropy balancing in terms of causal inference accuracy.

        ----

        ## [55] A Hierarchical Transitive-Aligned Graph Kernel for Un-attributed Graphs

        **Authors**: *Lu Bai, Lixin Cui, Edwin R. Hancock*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bai22a.html](https://proceedings.mlr.press/v162/bai22a.html)

        **Abstract**:

        In this paper, we develop a new graph kernel, namely the Hierarchical Transitive-Aligned Kernel, by transitively aligning the vertices between graphs through a family of hierarchical prototype graphs. Comparing to most existing state-of-the-art graph kernels, the proposed kernel has three theoretical advantages. First, it incorporates the locational correspondence information between graphs into the kernel computation, and thus overcomes the shortcoming of ignoring structural correspondences arising in most R-convolution kernels. Second, it guarantees the transitivity between the correspondence information that is not available for most existing matching kernels. Third, it incorporates the information of all graphs under comparisons into the kernel computation process, and thus encapsulates richer characteristics. Experimental evaluations demonstrate the effectiveness of the new transitive-aligned kernel.

        ----

        ## [56] Near-Optimal Learning of Extensive-Form Games with Imperfect Information

        **Authors**: *Yu Bai, Chi Jin, Song Mei, Tiancheng Yu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bai22b.html](https://proceedings.mlr.press/v162/bai22b.html)

        **Abstract**:

        This paper resolves the open question of designing near-optimal algorithms for learning imperfect-information extensive-form games from bandit feedback. We present the first line of algorithms that require only $\widetilde{\mathcal{O}}((XA+YB)/\varepsilon^2)$ episodes of play to find an $\varepsilon$-approximate Nash equilibrium in two-player zero-sum games, where $X,Y$ are the number of information sets and $A,B$ are the number of actions for the two players. This improves upon the best known sample complexity of $\widetilde{\mathcal{O}}((X^2A+Y^2B)/\varepsilon^2)$ by a factor of $\widetilde{\mathcal{O}}(\max\{X, Y\})$, and matches the information-theoretic lower bound up to logarithmic factors. We achieve this sample complexity by two new algorithms: Balanced Online Mirror Descent, and Balanced Counterfactual Regret Minimization. Both algorithms rely on novel approaches of integrating balanced exploration policies into their classical counterparts. We also extend our results to learning Coarse Correlated Equilibria in multi-player general-sum games.

        ----

        ## [57] Gaussian Mixture Variational Autoencoder with Contrastive Learning for Multi-Label Classification

        **Authors**: *Junwen Bai, Shufeng Kong, Carla P. Gomes*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bai22c.html](https://proceedings.mlr.press/v162/bai22c.html)

        **Abstract**:

        Multi-label classification (MLC) is a prediction task where each sample can have more than one label. We propose a novel contrastive learning boosted multi-label prediction model based on a Gaussian mixture variational autoencoder (C-GMVAE), which learns a multimodal prior space and employs a contrastive loss. Many existing methods introduce extra complex neural modules like graph neural networks to capture the label correlations, in addition to the prediction modules. We find that by using contrastive learning in the supervised setting, we can exploit label information effectively in a data-driven manner, and learn meaningful feature and label embeddings which capture the label correlations and enhance the predictive power. Our method also adopts the idea of learning and aligning latent spaces for both features and labels. In contrast to previous works based on a unimodal prior, C-GMVAE imposes a Gaussian mixture structure on the latent space, to alleviate the posterior collapse and over-regularization issues. C-GMVAE outperforms existing methods on multiple public datasets and can often match other models’ full performance with only 50% of the training data. Furthermore, we show that the learnt embeddings provide insights into the interpretation of label-label interactions.

        ----

        ## [58] A3T: Alignment-Aware Acoustic and Text Pretraining for Speech Synthesis and Editing

        **Authors**: *He Bai, Renjie Zheng, Junkun Chen, Mingbo Ma, Xintong Li, Liang Huang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bai22d.html](https://proceedings.mlr.press/v162/bai22d.html)

        **Abstract**:

        Recently, speech representation learning has improved many speech-related tasks such as speech recognition, speech classification, and speech-to-text translation. However, all the above tasks are in the direction of speech understanding, but for the inverse direction, speech synthesis, the potential of representation learning is yet to be realized, due to the challenging nature of generating high-quality speech. To address this problem, we propose our framework, Alignment-Aware Acoustic-Text Pretraining (A$^3$T), which reconstructs masked acoustic signals with text input and acoustic-text alignment during training. In this way, the pretrained model can generate high quality reconstructed spectrogram, which can be applied to the speech editing and unseen speaker TTS directly. Experiments show A$^3$T outperforms SOTA models on speech editing, and improves multi-speaker speech synthesis without the external speaker verification model.

        ----

        ## [59] Stability Based Generalization Bounds for Exponential Family Langevin Dynamics

        **Authors**: *Arindam Banerjee, Tiancong Chen, Xinyan Li, Yingxue Zhou*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/banerjee22a.html](https://proceedings.mlr.press/v162/banerjee22a.html)

        **Abstract**:

        Recent years have seen advances in generalization bounds for noisy stochastic algorithms, especially stochastic gradient Langevin dynamics (SGLD) based on stability (Mou et al., 2018; Li et al., 2020) and information theoretic approaches (Xu & Raginsky, 2017; Negrea et al., 2019; Steinke & Zakynthinou, 2020). In this paper, we unify and substantially generalize stability based generalization bounds and make three technical contributions. First, we bound the generalization error in terms of expected (not uniform) stability which arguably leads to quantitatively sharper bounds. Second, as our main contribution, we introduce Exponential Family Langevin Dynamics (EFLD), a substantial generalization of SGLD, which includes noisy versions of Sign-SGD and quantized SGD as special cases. We establish data dependent expected stability based generalization bounds for any EFLD algorithm with a O(1/n) sample dependence and dependence on gradient discrepancy rather than the norm of gradients, yielding significantly sharper bounds. Third, we establish optimization guarantees for special cases of EFLD. Further, empirical results on benchmarks illustrate that our bounds are non-vacuous, quantitatively sharper than existing bounds, and behave correctly under noisy labels.

        ----

        ## [60] Certified Neural Network Watermarks with Randomized Smoothing

        **Authors**: *Arpit Bansal, Ping-Yeh Chiang, Michael J. Curry, Rajiv Jain, Curtis Wigington, Varun Manjunatha, John P. Dickerson, Tom Goldstein*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bansal22a.html](https://proceedings.mlr.press/v162/bansal22a.html)

        **Abstract**:

        Watermarking is a commonly used strategy to protect creators’ rights to digital images, videos and audio. Recently, watermarking methods have been extended to deep learning models – in principle, the watermark should be preserved when an adversary tries to copy the model. However, in practice, watermarks can often be removed by an intelligent adversary. Several papers have proposed watermarking methods that claim to be empirically resistant to different types of removal attacks, but these new techniques often fail in the face of new or better-tuned adversaries. In this paper, we propose the first certifiable watermarking method. Using the randomized smoothing technique, we show that our watermark is guaranteed to be unremovable unless the model parameters are changed by more than a certain $\ell_2$ threshold. In addition to being certifiable, our watermark is also empirically more robust compared to previous watermarking methods.

        ----

        ## [61] Data Scaling Laws in NMT: The Effect of Noise and Architecture

        **Authors**: *Yamini Bansal, Behrooz Ghorbani, Ankush Garg, Biao Zhang, Colin Cherry, Behnam Neyshabur, Orhan Firat*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bansal22b.html](https://proceedings.mlr.press/v162/bansal22b.html)

        **Abstract**:

        In this work, we study the effect of varying the architecture and training data quality on the data scaling properties of Neural Machine Translation (NMT). First, we establish that the test loss of encoder-decoder transformer models scales as a power law in the number of training samples, with a dependence on the model size. Then, we systematically vary aspects of the training setup to understand how they impact the data scaling laws. In particular, we change the following (1) Architecture and task setup: We compare to a transformer-LSTM hybrid, and a decoder-only transformer with a language modeling loss (2) Noise level in the training distribution: We experiment with filtering, and adding iid synthetic noise. In all the above cases, we find that the data scaling exponents are minimally impacted, suggesting that marginally worse architectures or training data can be compensated for by adding more data. Lastly, we find that using back-translated data instead of parallel data, can significantly degrade the scaling exponent.

        ----

        ## [62] Learning Stable Classifiers by Transferring Unstable Features

        **Authors**: *Yujia Bao, Shiyu Chang, Regina Barzilay*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bao22a.html](https://proceedings.mlr.press/v162/bao22a.html)

        **Abstract**:

        While unbiased machine learning models are essential for many applications, bias is a human-defined concept that can vary across tasks. Given only input-label pairs, algorithms may lack sufficient information to distinguish stable (causal) features from unstable (spurious) features. However, related tasks often share similar biases – an observation we may leverage to develop stable classifiers in the transfer setting. In this work, we explicitly inform the target classifier about unstable features in the source tasks. Specifically, we derive a representation that encodes the unstable features by contrasting different data environments in the source task. We achieve robustness by clustering data of the target task according to this representation and minimizing the worst-case risk across these clusters. We evaluate our method on both text and image classifications. Empirical results demonstrate that our algorithm is able to maintain robustness on the target task for both synthetically generated environments and real-world environments. Our code is available at https://github.com/YujiaBao/Tofu.

        ----

        ## [63] Fast Composite Optimization and Statistical Recovery in Federated Learning

        **Authors**: *Yajie Bao, Michael Crawshaw, Shan Luo, Mingrui Liu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bao22b.html](https://proceedings.mlr.press/v162/bao22b.html)

        **Abstract**:

        As a prevalent distributed learning paradigm, Federated Learning (FL) trains a global model on a massive amount of devices with infrequent communication. This paper investigates a class of composite optimization and statistical recovery problems in the FL setting, whose loss function consists of a data-dependent smooth loss and a non-smooth regularizer. Examples include sparse linear regression using Lasso, low-rank matrix recovery using nuclear norm regularization, etc. In the existing literature, federated composite optimization algorithms are designed only from an optimization perspective without any statistical guarantees. In addition, they do not consider commonly used (restricted) strong convexity in statistical recovery problems. We advance the frontiers of this problem from both optimization and statistical perspectives. From optimization upfront, we propose a new algorithm named Fast Federated Dual Averaging for strongly convex and smooth loss and establish state-of-the-art iteration and communication complexity in the composite setting. In particular, we prove that it enjoys a fast rate, linear speedup, and reduced communication rounds. From statistical upfront, for restricted strongly convex and smooth loss, we design another algorithm, namely Multi-stage Federated Dual Averaging, and prove a high probability complexity bound with linear speedup up to optimal statistical precision. Numerical experiments in both synthetic and real data demonstrate that our methods perform better than other baselines. To the best of our knowledge, this is the first work providing fast optimization algorithms and statistical recovery guarantees for composite problems in FL.

        ----

        ## [64] Generative Modeling for Multi-task Visual Learning

        **Authors**: *Zhipeng Bao, Martial Hebert, Yu-Xiong Wang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bao22c.html](https://proceedings.mlr.press/v162/bao22c.html)

        **Abstract**:

        Generative modeling has recently shown great promise in computer vision, but it has mostly focused on synthesizing visually realistic images. In this paper, motivated by multi-task learning of shareable feature representations, we consider a novel problem of learning a shared generative model that is useful across various visual perception tasks. Correspondingly, we propose a general multi-task oriented generative modeling (MGM) framework, by coupling a discriminative multi-task network with a generative network. While it is challenging to synthesize both RGB images and pixel-level annotations in multi-task scenarios, our framework enables us to use synthesized images paired with only weak annotations (i.e., image-level scene labels) to facilitate multiple visual tasks. Experimental evaluation on challenging multi-task benchmarks, including NYUv2 and Taskonomy, demonstrates that our MGM framework improves the performance of all the tasks by large margins, consistently outperforming state-of-the-art multi-task approaches in different sample-size regimes.

        ----

        ## [65] Estimating the Optimal Covariance with Imperfect Mean in Diffusion Probabilistic Models

        **Authors**: *Fan Bao, Chongxuan Li, Jiacheng Sun, Jun Zhu, Bo Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bao22d.html](https://proceedings.mlr.press/v162/bao22d.html)

        **Abstract**:

        Diffusion probabilistic models (DPMs) are a class of powerful deep generative models (DGMs). Despite their success, the iterative generation process over the full timesteps is much less efficient than other DGMs such as GANs. Thus, the generation performance on a subset of timesteps is crucial, which is greatly influenced by the covariance design in DPMs. In this work, we consider diagonal and full covariances to improve the expressive power of DPMs. We derive the optimal result for such covariances, and then correct it when the mean of DPMs is imperfect. Both the optimal and the corrected ones can be decomposed into terms of conditional expectations over functions of noise. Building upon it, we propose to estimate the optimal covariance and its correction given imperfect mean by learning these conditional expectations. Our method can be applied to DPMs with both discrete and continuous timesteps. We consider the diagonal covariance in our implementation for computational efficiency. For an efficient practical implementation, we adopt a parameter sharing scheme and a two-stage training process. Empirically, our method outperforms a wide variety of covariance design on likelihood results, and improves the sample quality especially on a small number of timesteps.

        ----

        ## [66] On the Surrogate Gap between Contrastive and Supervised Losses

        **Authors**: *Han Bao, Yoshihiro Nagano, Kento Nozawa*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bao22e.html](https://proceedings.mlr.press/v162/bao22e.html)

        **Abstract**:

        Contrastive representation learning encourages data representation to make semantically similar pairs closer than randomly drawn negative samples, which has been successful in various domains such as vision, language, and graphs. Recent theoretical studies have attempted to explain the benefit of the large negative sample size by upper-bounding the downstream classification loss with the contrastive loss. However, the previous surrogate bounds have two drawbacks: they are only legitimate for a limited range of negative sample sizes and prohibitively large even within that range. Due to these drawbacks, there still does not exist a consensus on how negative sample size theoretically correlates with downstream classification performance. Following the simplified setting where positive pairs are drawn from the true distribution (not generated by data augmentation; as supposed in previous studies), this study establishes surrogate upper and lower bounds for the downstream classification loss for all negative sample sizes that best explain the empirical observations on the negative sample size in the earlier studies. Our bounds suggest that the contrastive loss can be viewed as a surrogate objective of the downstream loss and larger negative sample sizes improve downstream classification because the surrogate gap between contrastive and supervised losses decays. We verify that our theory is consistent with experiments on synthetic, vision, and language datasets.

        ----

        ## [67] Representation Topology Divergence: A Method for Comparing Neural Network Representations

        **Authors**: *Serguei Barannikov, Ilya Trofimov, Nikita Balabin, Evgeny Burnaev*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/barannikov22a.html](https://proceedings.mlr.press/v162/barannikov22a.html)

        **Abstract**:

        Comparison of data representations is a complex multi-aspect problem. We propose a method for comparing two data representations. We introduce the Representation Topology Divergence (RTD) score measuring the dissimilarity in multi-scale topology between two point clouds of equal size with a one-to-one correspondence between points. The two data point clouds can lie in different ambient spaces. The RTD score is one of the few topological data analysis based practical methods applicable to real machine learning datasets. Experiments show the agreement of RTD with the intuitive assessment of data representation similarity. The proposed RTD score is sensitive to the data representation’s fine topological structure. We use the RTD score to gain insights on neural networks representations in computer vision and NLP domains for various problems: training dynamics analysis, data distribution shift, transfer learning, ensemble learning, disentanglement assessment.

        ----

        ## [68] Sparse Mixed Linear Regression with Guarantees: Taming an Intractable Problem with Invex Relaxation

        **Authors**: *Adarsh Barik, Jean Honorio*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/barik22a.html](https://proceedings.mlr.press/v162/barik22a.html)

        **Abstract**:

        In this paper, we study the problem of sparse mixed linear regression on an unlabeled dataset that is generated from linear measurements from two different regression parameter vectors. Since the data is unlabeled, our task is to not only figure out a good approximation of regression parameter vectors but also label the dataset correctly. In its original form, this problem is NP-hard. The most popular algorithms to solve this problem (such as Expectation-Maximization) have a tendency to stuck at local minima. We provide a novel invex relaxation for this intractable problem which leads to a solution with provable theoretical guarantees. This relaxation enables exact recovery of data labels. Furthermore, we recover close approximation of regression parameter vectors which match the true parameter vectors in support and sign. Our formulation uses a carefully constructed primal dual witnesses framework for the invex problem. Furthermore, we show that the sample complexity of our method is only logarithmic in terms of the dimension of the regression parameter vectors.

        ----

        ## [69] Neural Fisher Discriminant Analysis: Optimal Neural Network Embeddings in Polynomial Time

        **Authors**: *Burak Bartan, Mert Pilanci*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bartan22a.html](https://proceedings.mlr.press/v162/bartan22a.html)

        **Abstract**:

        Fisher’s Linear Discriminant Analysis (FLDA) is a statistical analysis method that linearly embeds data points to a lower dimensional space to maximize a discrimination criterion such that the variance between classes is maximized while the variance within classes is minimized. We introduce a natural extension of FLDA that employs neural networks, called Neural Fisher Discriminant Analysis (NFDA). This method finds the optimal two-layer neural network that embeds data points to optimize the same discrimination criterion. We use tools from convex optimization to transform the optimal neural network embedding problem into a convex problem. The resulting problem is easy to interpret and solve to global optimality. We evaluate the method’s performance on synthetic and real datasets.

        ----

        ## [70] Fictitious Play and Best-Response Dynamics in Identical Interest and Zero-Sum Stochastic Games

        **Authors**: *Lucas Baudin, Rida Laraki*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/baudin22a.html](https://proceedings.mlr.press/v162/baudin22a.html)

        **Abstract**:

        This paper proposes an extension of a popular decentralized discrete-time learning procedure when repeating a static game called fictitious play (FP) (Brown, 1951; Robinson, 1951) to a dynamic model called discounted stochastic game (Shapley, 1953). Our family of discrete-time FP procedures is proven to converge to the set of stationary Nash equilibria in identical interest discounted stochastic games. This extends similar convergence results for static games (Monderer & Shapley, 1996a). We then analyze the continuous-time counterpart of our FP procedures, which include as a particular case the best-response dynamic introduced and studied by Leslie et al. (2020) in the context of zero-sum stochastic games. We prove the converge of this dynamics to stationary Nash equilibria in identical-interest and zero-sum discounted stochastic games. Thanks to stochastic approximations, we can infer from the continuous-time convergence some discrete time results such as the convergence to stationary equilibria in zero-sum and team stochastic games (Holler, 2020).

        ----

        ## [71] Information Discrepancy in Strategic Learning

        **Authors**: *Yahav Bechavod, Chara Podimata, Zhiwei Steven Wu, Juba Ziani*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bechavod22a.html](https://proceedings.mlr.press/v162/bechavod22a.html)

        **Abstract**:

        We initiate the study of the effects of non-transparency in decision rules on individuals’ ability to improve in strategic learning settings. Inspired by real-life settings, such as loan approvals and college admissions, we remove the assumption typically made in the strategic learning literature, that the decision rule is fully known to individuals, and focus instead on settings where it is inaccessible. In their lack of knowledge, individuals try to infer this rule by learning from their peers (e.g., friends and acquaintances who previously applied for a loan), naturally forming groups in the population, each with possibly different type and level of information regarding the decision rule. We show that, in equilibrium, the principal’s decision rule optimizing welfare across sub-populations may cause a strong negative externality: the true quality of some of the groups can actually deteriorate. On the positive side, we show that, in many natural cases, optimal improvement can be guaranteed simultaneously for all sub-populations. We further introduce a measure we term information overlap proxy, and demonstrate its usefulness in characterizing the disparity in improvements across sub-populations. Finally, we identify a natural condition under which improvement can be guaranteed for all sub-populations while maintaining high predictive accuracy. We complement our theoretical analysis with experiments on real-world datasets.

        ----

        ## [72] On the Hidden Biases of Policy Mirror Ascent in Continuous Action Spaces

        **Authors**: *Amrit Singh Bedi, Souradip Chakraborty, Anjaly Parayil, Brian M. Sadler, Pratap Tokekar, Alec Koppel*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bedi22a.html](https://proceedings.mlr.press/v162/bedi22a.html)

        **Abstract**:

        We focus on parameterized policy search for reinforcement learning over continuous action spaces. Typically, one assumes the score function associated with a policy is bounded, which {fails to hold even for Gaussian policies. } To properly address this issue, one must introduce an exploration tolerance parameter to quantify the region in which it is bounded. Doing so incurs a persistent bias that appears in the attenuation rate of the expected policy gradient norm, which is inversely proportional to the radius of the action space. To mitigate this hidden bias, heavy-tailed policy parameterizations may be used, which exhibit a bounded score function, but doing so can cause instability in algorithmic updates. To address these issues, in this work, we study the convergence of policy gradient algorithms under heavy-tailed parameterizations, which we propose to stabilize with a combination of mirror ascent-type updates and gradient tracking. Our main theoretical contribution is the establishment that this scheme converges with constant batch sizes, whereas prior works require these parameters to respectively shrink to null or grow to infinity. Experimentally, this scheme under a heavy-tailed policy parameterization yields improved reward accumulation across a variety of settings as compared with standard benchmarks.

        ----

        ## [73] Imitation Learning by Estimating Expertise of Demonstrators

        **Authors**: *Mark Beliaev, Andy Shih, Stefano Ermon, Dorsa Sadigh, Ramtin Pedarsani*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/beliaev22a.html](https://proceedings.mlr.press/v162/beliaev22a.html)

        **Abstract**:

        Many existing imitation learning datasets are collected from multiple demonstrators, each with different expertise at different parts of the environment. Yet, standard imitation learning algorithms typically treat all demonstrators as homogeneous, regardless of their expertise, absorbing the weaknesses of any suboptimal demonstrators. In this work, we show that unsupervised learning over demonstrator expertise can lead to a consistent boost in the performance of imitation learning algorithms. We develop and optimize a joint model over a learned policy and expertise levels of the demonstrators. This enables our model to learn from the optimal behavior and filter out the suboptimal behavior of each demonstrator. Our model learns a single policy that can outperform even the best demonstrator, and can be used to estimate the expertise of any demonstrator at any state. We illustrate our findings on real-robotic continuous control tasks from Robomimic and discrete environments such as MiniGrid and chess, out-performing competing methods in 21 out of 23 settings, with an average of 7% and up to 60% improvement in terms of the final reward.

        ----

        ## [74] Matching Normalizing Flows and Probability Paths on Manifolds

        **Authors**: *Heli Ben-Hamu, Samuel Cohen, Joey Bose, Brandon Amos, Maximilian Nickel, Aditya Grover, Ricky T. Q. Chen, Yaron Lipman*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ben-hamu22a.html](https://proceedings.mlr.press/v162/ben-hamu22a.html)

        **Abstract**:

        Continuous Normalizing Flows (CNFs) are a class of generative models that transform a prior distribution to a model distribution by solving an ordinary differential equation (ODE). We propose to train CNFs on manifolds by minimizing probability path divergence (PPD), a novel family of divergences between the probability density path generated by the CNF and a target probability density path. PPD is formulated using a logarithmic mass conservation formula which is a linear first order partial differential equation relating the log target probabilities and the CNF’s defining vector field. PPD has several key benefits over existing methods: it sidesteps the need to solve an ODE per iteration, readily applies to manifold data, scales to high dimensions, and is compatible with a large family of target paths interpolating pure noise and data in finite time. Theoretically, PPD is shown to bound classical probability divergences. Empirically, we show that CNFs learned by minimizing PPD achieve state-of-the-art results in likelihoods and sample quality on existing low-dimensional manifold benchmarks, and is the first example of a generative model to scale to moderately high dimensional manifolds.

        ----

        ## [75] Stochastic Contextual Dueling Bandits under Linear Stochastic Transitivity Models

        **Authors**: *Viktor Bengs, Aadirupa Saha, Eyke Hüllermeier*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bengs22a.html](https://proceedings.mlr.press/v162/bengs22a.html)

        **Abstract**:

        We consider the regret minimization task in a dueling bandits problem with context information. In every round of the sequential decision problem, the learner makes a context-dependent selection of two choice alternatives (arms) to be compared with each other and receives feedback in the form of noisy preference information. We assume that the feedback process is determined by a linear stochastic transitivity model with contextualized utilities (CoLST), and the learner’s task is to include the best arm (with highest latent context-dependent utility) in the duel. We propose a computationally efficient algorithm, \Algo{CoLSTIM}, which makes its choice based on imitating the feedback process using perturbed context-dependent utility estimates of the underlying CoLST model. If each arm is associated with a $d$-dimensional feature vector, we show that \Algo{CoLSTIM} achieves a regret of order $\tilde O( \sqrt{dT})$ after $T$ learning rounds. Additionally, we also establish the optimality of \Algo{CoLSTIM} by showing a lower bound for the weak regret that refines the existing average regret analysis. Our experiments demonstrate its superiority over state-of-art algorithms for special cases of CoLST models.

        ----

        ## [76] Neural Inverse Kinematic

        **Authors**: *Raphael Bensadoun, Shir Gur, Nitsan Blau, Lior Wolf*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bensadoun22a.html](https://proceedings.mlr.press/v162/bensadoun22a.html)

        **Abstract**:

        Inverse kinematic (IK) methods recover the parameters of the joints, given the desired position of selected elements in the kinematic chain. While the problem is well-defined and low-dimensional, it has to be solved rapidly, accounting for multiple possible solutions. In this work, we propose a neural IK method that employs the hierarchical structure of the problem to sequentially sample valid joint angles conditioned on the desired position and on the preceding joints along the chain. In our solution, a hypernetwork $f$ recovers the parameters of multiple primary networks {$g_1,g_2,…,g_N$, where $N$ is the number of joints}, such that each $g_i$ outputs a distribution of possible joint angles, and is conditioned on the sampled values obtained from the previous primary networks $g_j, j
Cite this Paper



    BibTeX
  



@InProceedings{pmlr-v162-bensadoun22a,
  title = 	 {Neural Inverse Kinematic},
  author =       {Bensadoun, Raphael and Gur, Shir and Blau, Nitsan and Wolf, Lior},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {1787--1797},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/bensadoun22a/bensadoun22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/bensadoun22a.html},
  abstract = 	 {Inverse kinematic (IK) methods recover the parameters of the joints, given the desired position of selected elements in the kinematic chain. While the problem is well-defined and low-dimensional, it has to be solved rapidly, accounting for multiple possible solutions. In this work, we propose a neural IK method that employs the hierarchical structure of the problem to sequentially sample valid joint angles conditioned on the desired position and on the preceding joints along the chain. In our solution, a hypernetwork $f$ recovers the parameters of multiple primary networks {$g_1,g_2,…,g_N$, where $N$ is the number of joints}, such that each $g_i$ outputs a distribution of possible joint angles, and is conditioned on the sampled values obtained from the previous primary networks $g_j, j

Copy to Clipboard
Download




    Endnote
  


%0 Conference Paper
%T Neural Inverse Kinematic
%A Raphael Bensadoun
%A Shir Gur
%A Nitsan Blau
%A Lior Wolf
%B Proceedings of the 39th International Conference on Machine Learning
%C Proceedings of Machine Learning Research
%D 2022
%E Kamalika Chaudhuri
%E Stefanie Jegelka
%E Le Song
%E Csaba Szepesvari
%E Gang Niu
%E Sivan Sabato	
%F pmlr-v162-bensadoun22a
%I PMLR
%P 1787--1797
%U https://proceedings.mlr.press/v162/bensadoun22a.html
%V 162
%X Inverse kinematic (IK) methods recover the parameters of the joints, given the desired position of selected elements in the kinematic chain. While the problem is well-defined and low-dimensional, it has to be solved rapidly, accounting for multiple possible solutions. In this work, we propose a neural IK method that employs the hierarchical structure of the problem to sequentially sample valid joint angles conditioned on the desired position and on the preceding joints along the chain. In our solution, a hypernetwork $f$ recovers the parameters of multiple primary networks {$g_1,g_2,…,g_N$, where $N$ is the number of joints}, such that each $g_i$ outputs a distribution of possible joint angles, and is conditioned on the sampled values obtained from the previous primary networks $g_j, j

Copy to Clipboard
Download




    APA
  



Bensadoun, R., Gur, S., Blau, N. & Wolf, L.. (2022). Neural Inverse Kinematic. Proceedings of the 39th International Conference on Machine Learning, in Proceedings of Machine Learning Research 162:1787-1797 Available from https://proceedings.mlr.press/v162/bensadoun22a.html.



Copy to Clipboard
Download



Related Material


Download PDF

        ----

        ## [77] Volatility Based Kernels and Moving Average Means for Accurate Forecasting with Gaussian Processes

        **Authors**: *Gregory W. Benton, Wesley J. Maddox, Andrew Gordon Wilson*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/benton22a.html](https://proceedings.mlr.press/v162/benton22a.html)

        **Abstract**:

        A broad class of stochastic volatility models are defined by systems of stochastic differential equations, and while these models have seen widespread success in domains such as finance and statistical climatology, they typically lack an ability to condition on historical data to produce a true posterior distribution. To address this fundamental limitation, we show how to re-cast a class of stochastic volatility models as a hierarchical Gaussian process (GP) model with specialized covariance functions. This GP model retains the inductive biases of the stochastic volatility model while providing the posterior predictive distribution given by GP inference. Within this framework, we take inspiration from well studied domains to introduce a new class of models, Volt and Magpie, that significantly outperform baselines in stock and wind speed forecasting, and naturally extend to the multitask setting.

        ----

        ## [78] Gradient Descent on Neurons and its Link to Approximate Second-order Optimization

        **Authors**: *Frederik Benzing*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/benzing22a.html](https://proceedings.mlr.press/v162/benzing22a.html)

        **Abstract**:

        Second-order optimizers are thought to hold the potential to speed up neural network training, but due to the enormous size of the curvature matrix, they typically require approximations to be computationally tractable. The most successful family of approximations are Kronecker-Factored, block-diagonal curvature estimates (KFAC). Here, we combine tools from prior work to evaluate exact second-order updates with careful ablations to establish a surprising result: Due to its approximations, KFAC is not closely related to second-order updates, and in particular, it significantly outperforms true second-order updates. This challenges widely held believes and immediately raises the question why KFAC performs so well. Towards answering this question we present evidence strongly suggesting that KFAC approximates a first-order algorithm, which performs gradient descent on neurons rather than weights. Finally, we show that this optimizer often improves over KFAC in terms of computational cost and data-efficiency.

        ----

        ## [79] Safe Learning in Tree-Form Sequential Decision Making: Handling Hard and Soft Constraints

        **Authors**: *Martino Bernasconi, Federico Cacciamani, Matteo Castiglioni, Alberto Marchesi, Nicola Gatti, Francesco Trovò*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bernasconi22a.html](https://proceedings.mlr.press/v162/bernasconi22a.html)

        **Abstract**:

        We study decision making problems in which an agent sequentially interacts with a stochastic environment defined by means of a tree structure. The agent repeatedly faces the environment over time, and, after each round, it perceives a utility and a cost, which are both stochastic. The goal of the agent is to learn an optimal strategy in an online fashion, while, at the same time, keeping costs below a given safety threshold. Our model naturally fits many real-world scenarios, such as, e.g., opponent exploitation in games and web link selection. We study the hard-threshold problem of achieving sublinear regret while guaranteeing that the threshold constraint is satisfied at every iteration with high probability. First, we show that, in general, any algorithm with such a guarantee incurs in a linear regret. This motivates the introduction of a relaxed problem, namely the soft-threshold problem, in which we only require that the cumulative violation of the threshold constraint grows sublinearly, and, thus, we can provide an algorithm with sublinear regret. Next, we show how, in the hard-threshold problem, a sublinear regret algorithm can be designed under the additional assumption that there exists a known strategy strictly satisfying the threshold constraint. We also show that our regret bounds are tight. Finally, we cast the opponent exploitation problem to our model, and we experimentally evaluate our algorithms on a standard testbed of games.

        ----

        ## [80] Skin Deep Unlearning: Artefact and Instrument Debiasing in the Context of Melanoma Classification

        **Authors**: *Peter J. Bevan, Amir Atapour-Abarghouei*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bevan22a.html](https://proceedings.mlr.press/v162/bevan22a.html)

        **Abstract**:

        Convolutional Neural Networks have demonstrated dermatologist-level performance in the classification of melanoma from skin lesion images, but prediction irregularities due to biases seen within the training data are an issue that should be addressed before widespread deployment is possible. In this work, we robustly remove bias and spurious variation from an automated melanoma classification pipeline using two leading bias unlearning techniques. We show that the biases introduced by surgical markings and rulers presented in previous studies can be reasonably mitigated using these bias removal methods. We also demonstrate the generalisation benefits of unlearning spurious variation relating to the imaging instrument used to capture lesion images. Our experimental results provide evidence that the effects of each of the aforementioned biases are notably reduced, with different debiasing techniques excelling at different tasks.

        ----

        ## [81] Approximate Bayesian Computation with Domain Expert in the Loop

        **Authors**: *Ayush Bharti, Louis Filstroff, Samuel Kaski*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bharti22a.html](https://proceedings.mlr.press/v162/bharti22a.html)

        **Abstract**:

        Approximate Bayesian computation (ABC) is a popular likelihood-free inference method for models with intractable likelihood functions. As ABC methods usually rely on comparing summary statistics of observed and simulated data, the choice of the statistics is crucial. This choice involves a trade-off between loss of information and dimensionality reduction, and is often determined based on domain knowledge. However, handcrafting and selecting suitable statistics is a laborious task involving multiple trial-and-error steps. In this work, we introduce an active learning method for ABC statistics selection which reduces the domain expert’s work considerably. By involving the experts, we are able to handle misspecified models, unlike the existing dimension reduction methods. Moreover, empirical results show better posterior estimates than with existing methods, when the simulation budget is limited.

        ----

        ## [82] Minimax M-estimation under Adversarial Contamination

        **Authors**: *Sujay Bhatt, Guanhua Fang, Ping Li, Gennady Samorodnitsky*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bhatt22a.html](https://proceedings.mlr.press/v162/bhatt22a.html)

        **Abstract**:

        We present a new finite-sample analysis of Catoni’s M-estimator under adversarial contamination, where an adversary is allowed to corrupt a fraction of the samples arbitrarily. We make minimal assumptions on the distribution of the uncontaminated random variables, namely, we only assume the existence of a known upper bound $\upsilon_{\varepsilon} > 0$ on the $(1+\varepsilon)^{th}$ central moment of the random variables, namely, for $\varepsilon \in (0,1]$ \[ \mathbb{E}_{X_1 \sim \mathcal{D}} \Big| X_1 - \mu \Big|^{1+\varepsilon} \leq \upsilon_{\varepsilon}. \]{We} provide a lower bound on the minimax error rate for the mean estimation problem under adversarial corruption under this weak assumption, and establish that the proposed M-estimator achieves this lower bound (up to multiplicative constants). When the variance is infinite, the tolerance to contamination of any estimator reduces as $\varepsilon \downarrow 0$. We establish a tight upper bound that characterizes this bargain. To illustrate the usefulness of the derived robust M-estimator in an online setting, we present a bandit algorithm for the partially identifiable best arm identification problem that improves upon the sample complexity of the state of the art algorithms.

        ----

        ## [83] Nearly Optimal Catoni's M-estimator for Infinite Variance

        **Authors**: *Sujay Bhatt, Guanhua Fang, Ping Li, Gennady Samorodnitsky*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bhatt22b.html](https://proceedings.mlr.press/v162/bhatt22b.html)

        **Abstract**:

        In this paper, we extend the remarkable M-estimator of Catoni \citep{Cat12} to situations where the variance is infinite. In particular, given a sequence of i.i.d random variables $\{X_i\}_{i=1}^n$ from distribution $\mathcal{D}$ over $\mathbb{R}$ with mean $\mu$, we only assume the existence of a known upper bound $\upsilon_{\varepsilon} > 0$ on the $(1+\varepsilon)^{th}$ central moment of the random variables, namely, for $\varepsilon \in (0,1]$ \[ \mathbb{E}_{X_1 \sim \mathcal{D}} \Big| X_1 - \mu \Big|^{1+\varepsilon} \leq \upsilon_{\varepsilon}. \]{The} extension is non-trivial owing to the difficulty in characterizing the roots of certain polynomials of degree smaller than $2$. The proposed estimator has the same order of magnitude and the same asymptotic constant as in \citet{Cat12}, but for the case of bounded moments. We further propose a version of the estimator that does not require even the knowledge of $\upsilon_{\varepsilon}$, but adapts the moment bound in a data-driven manner. Finally, to illustrate the usefulness of the derived non-asymptotic confidence bounds, we consider an application in multi-armed bandits and propose best arm identification algorithms, in the fixed confidence setting, that outperform the state of the art.

        ----

        ## [84] Personalization Improves Privacy-Accuracy Tradeoffs in Federated Learning

        **Authors**: *Alberto Bietti, Chen-Yu Wei, Miroslav Dudík, John Langford, Zhiwei Steven Wu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bietti22a.html](https://proceedings.mlr.press/v162/bietti22a.html)

        **Abstract**:

        Large-scale machine learning systems often involve data distributed across a collection of users. Federated learning algorithms leverage this structure by communicating model updates to a central server, rather than entire datasets. In this paper, we study stochastic optimization algorithms for a personalized federated learning setting involving local and global models subject to user-level (joint) differential privacy. While learning a private global model induces a cost of privacy, local learning is perfectly private. We provide generalization guarantees showing that coordinating local learning with private centralized learning yields a generically useful and improved tradeoff between accuracy and privacy. We illustrate our theoretical results with experiments on synthetic and real-world datasets.

        ----

        ## [85] Non-Vacuous Generalisation Bounds for Shallow Neural Networks

        **Authors**: *Felix Biggs, Benjamin Guedj*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/biggs22a.html](https://proceedings.mlr.press/v162/biggs22a.html)

        **Abstract**:

        We focus on a specific class of shallow neural networks with a single hidden layer, namely those with $L_2$-normalised data and either a sigmoid-shaped Gaussian error function (“erf”) activation or a Gaussian Error Linear Unit (GELU) activation. For these networks, we derive new generalisation bounds through the PAC-Bayesian theory; unlike most existing such bounds they apply to neural networks with deterministic rather than randomised parameters. Our bounds are empirically non-vacuous when the network is trained with vanilla stochastic gradient descent on MNIST and Fashion-MNIST.

        ----

        ## [86] Structure-preserving GANs

        **Authors**: *Jeremiah Birrell, Markos A. Katsoulakis, Luc Rey-Bellet, Wei Zhu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/birrell22a.html](https://proceedings.mlr.press/v162/birrell22a.html)

        **Abstract**:

        Generative adversarial networks (GANs), a class of distribution-learning methods based on a two-player game between a generator and a discriminator, can generally be formulated as a minmax problem based on the variational representation of a divergence between the unknown and the generated distributions. We introduce structure-preserving GANs as a data-efficient framework for learning distributions with additional structure such as group symmetry, by developing new variational representations for divergences. Our theory shows that we can reduce the discriminator space to its projection on the invariant discriminator space, using the conditional expectation with respect to the sigma-algebra associated to the underlying structure. In addition, we prove that the discriminator space reduction must be accompanied by a careful design of structured generators, as flawed designs may easily lead to a catastrophic “mode collapse” of the learned distribution. We contextualize our framework by building symmetry-preserving GANs for distributions with intrinsic group symmetry, and demonstrate that both players, namely the equivariant generator and invariant discriminator, play important but distinct roles in the learning process. Empirical experiments and ablation studies across a broad range of data sets, including real-world medical imaging, validate our theory, and show our proposed methods achieve significantly improved sample fidelity and diversity—almost an order of magnitude measured in Frechet Inception Distance—especially in the small data regime.

        ----

        ## [87] Scalable Spike-and-Slab

        **Authors**: *Niloy Biswas, Lester Mackey, Xiao-Li Meng*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/biswas22a.html](https://proceedings.mlr.press/v162/biswas22a.html)

        **Abstract**:

        Spike-and-slab priors are commonly used for Bayesian variable selection, due to their interpretability and favorable statistical properties. However, existing samplers for spike-and-slab posteriors incur prohibitive computational costs when the number of variables is large. In this article, we propose Scalable Spike-and-Slab (S^3), a scalable Gibbs sampling implementation for high-dimensional Bayesian regression with the continuous spike-and-slab prior of George & McCulloch (1993). For a dataset with n observations and p covariates, S^3 has order max{n^2 p_t, np} computational cost at iteration t where p_t never exceeds the number of covariates switching spike-and-slab states between iterations t and t-1 of the Markov chain. This improves upon the order n^2 p per-iteration cost of state-of-the-art implementations as, typically, p_t is substantially smaller than p. We apply S^3 on synthetic and real-world datasets, demonstrating orders of magnitude speed-ups over existing exact samplers and significant gains in inferential quality over approximate samplers with comparable cost.

        ----

        ## [88] Breaking Down Out-of-Distribution Detection: Many Methods Based on OOD Training Data Estimate a Combination of the Same Core Quantities

        **Authors**: *Julian Bitterwolf, Alexander Meinke, Maximilian Augustin, Matthias Hein*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bitterwolf22a.html](https://proceedings.mlr.press/v162/bitterwolf22a.html)

        **Abstract**:

        It is an important problem in trustworthy machine learning to recognize out-of-distribution (OOD) inputs which are inputs unrelated to the in-distribution task. Many out-of-distribution detection methods have been suggested in recent years. The goal of this paper is to recognize common objectives as well as to identify the implicit scoring functions of different OOD detection methods. We focus on the sub-class of methods that use surrogate OOD data during training in order to learn an OOD detection score that generalizes to new unseen out-distributions at test time. We show that binary discrimination between in- and (different) out-distributions is equivalent to several distinct formulations of the OOD detection problem. When trained in a shared fashion with a standard classifier, this binary discriminator reaches an OOD detection performance similar to that of Outlier Exposure. Moreover, we show that the confidence loss which is used by Outlier Exposure has an implicit scoring function which differs in a non-trivial fashion from the theoretically optimal scoring function in the case where training and test out-distribution are the same, which again is similar to the one used when training an Energy-Based OOD detector or when adding a background class. In practice, when trained in exactly the same way, all these methods perform similarly.

        ----

        ## [89] A query-optimal algorithm for finding counterfactuals

        **Authors**: *Guy Blanc, Caleb Koch, Jane Lange, Li-Yang Tan*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/blanc22a.html](https://proceedings.mlr.press/v162/blanc22a.html)

        **Abstract**:

        We design an algorithm for finding counterfactuals with strong theoretical guarantees on its performance. For any monotone model $f : X^d \to \{0,1\}$ and instance $x^\star$, our algorithm makes \[{S}(f)^{O(\Delta_f(x^\star))}\cdot \log d\]{queries} to $f$ and returns an {\sl optimal} counterfactual for $x^\star$: a nearest instance $x’$ to $x^\star$ for which $f(x’)\ne f(x^\star)$. Here $S(f)$ is the sensitivity of $f$, a discrete analogue of the Lipschitz constant, and $\Delta_f(x^\star)$ is the distance from $x^\star$ to its nearest counterfactuals. The previous best known query complexity was $d^{\,O(\Delta_f(x^\star))}$, achievable by brute-force local search. We further prove a lower bound of $S(f)^{\Omega(\Delta_f(x^\star))} + \Omega(\log d)$ on the query complexity of any algorithm, thereby showing that the guarantees of our algorithm are essentially optimal.

        ----

        ## [90] Popular decision tree algorithms are provably noise tolerant

        **Authors**: *Guy Blanc, Jane Lange, Ali Malik, Li-Yang Tan*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/blanc22b.html](https://proceedings.mlr.press/v162/blanc22b.html)

        **Abstract**:

        Using the framework of boosting, we prove that all impurity-based decision tree learning algorithms, including the classic ID3, C4.5, and CART, are highly noise tolerant. Our guarantees hold under the strongest noise model of nasty noise, and we provide near-matching upper and lower bounds on the allowable noise rate. We further show that these algorithms, which are simple and have long been central to everyday machine learning, enjoy provable guarantees in the noisy setting that are unmatched by existing algorithms in the theoretical literature on decision tree learning. Taken together, our results add to an ongoing line of research that seeks to place the empirical success of these practical decision tree algorithms on firm theoretical footing.

        ----

        ## [91] Optimizing Sequential Experimental Design with Deep Reinforcement Learning

        **Authors**: *Tom Blau, Edwin V. Bonilla, Iadine Chades, Amir Dezfouli*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/blau22a.html](https://proceedings.mlr.press/v162/blau22a.html)

        **Abstract**:

        Bayesian approaches developed to solve the optimal design of sequential experiments are mathematically elegant but computationally challenging. Recently, techniques using amortization have been proposed to make these Bayesian approaches practical, by training a parameterized policy that proposes designs efficiently at deployment time. However, these methods may not sufficiently explore the design space, require access to a differentiable probabilistic model and can only optimize over continuous design spaces. Here, we address these limitations by showing that the problem of optimizing policies can be reduced to solving a Markov decision process (MDP). We solve the equivalent MDP with modern deep reinforcement learning techniques. Our experiments show that our approach is also computationally efficient at deployment time and exhibits state-of-the-art performance on both continuous and discrete design spaces, even when the probabilistic model is a black box.

        ----

        ## [92] Lagrangian Method for Q-Function Learning (with Applications to Machine Translation)

        **Authors**: *Bojun Huang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bojun22a.html](https://proceedings.mlr.press/v162/bojun22a.html)

        **Abstract**:

        This paper discusses a new approach to the fundamental problem of learning optimal Q-functions. In this approach, optimal Q-functions are formulated as saddle points of a nonlinear Lagrangian function derived from the classic Bellman optimality equation. The paper shows that the Lagrangian enjoys strong duality, in spite of its nonlinearity, which paves the way to a general Lagrangian method to Q-function learning. As a demonstration, the paper develops an imitation learning algorithm based on the duality theory, and applies the algorithm to a state-of-the-art machine translation benchmark. The paper then turns to demonstrate a symmetry breaking phenomenon regarding the optimality of the Lagrangian saddle points, which justifies a largely overlooked direction in developing the Lagrangian method.

        ----

        ## [93] Generalized Results for the Existence and Consistency of the MLE in the Bradley-Terry-Luce Model

        **Authors**: *Heejong Bong, Alessandro Rinaldo*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bong22a.html](https://proceedings.mlr.press/v162/bong22a.html)

        **Abstract**:

        Ranking problems based on pairwise comparisons, such as those arising in online gaming, often involve a large pool of items to order. In these situations, the gap in performance between any two items can be significant, and the smallest and largest winning probabilities can be very close to zero or one. Furthermore, each item may be compared only to a subset of all the items, so that not all pairwise comparisons are observed. In this paper, we study the performance of the Bradley-Terry-Luce model for ranking from pairwise comparison data under more realistic settings than those considered in the literature so far. In particular, we allow for near-degenerate winning probabilities and arbitrary comparison designs. We obtain novel results about the existence of the maximum likelihood estimator (MLE) and the corresponding $\ell_2$ estimation error without the bounded winning probability assumption commonly used in the literature and for arbitrary comparison graph topologies. Central to our approach is the reliance on the Fisher information matrix to express the dependence on the graph topologies and the impact of the values of the winning probabilities on the estimation risk and on the conditions for the existence of the MLE. Our bounds recover existing results as special cases but are more broadly applicable.

        ----

        ## [94] How to Train Your Wide Neural Network Without Backprop: An Input-Weight Alignment Perspective

        **Authors**: *Akhilan Boopathy, Ila Fiete*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/boopathy22a.html](https://proceedings.mlr.press/v162/boopathy22a.html)

        **Abstract**:

        Recent works have examined theoretical and empirical properties of wide neural networks trained in the Neural Tangent Kernel (NTK) regime. Given that biological neural networks are much wider than their artificial counterparts, we consider NTK regime wide neural networks as a possible model of biological neural networks. Leveraging NTK theory, we show theoretically that gradient descent drives layerwise weight updates that are aligned with their input activity correlations weighted by error, and demonstrate empirically that the result also holds in finite-width wide networks. The alignment result allows us to formulate a family of biologically-motivated, backpropagation-free learning rules that are theoretically equivalent to backpropagation in infinite-width networks. We test these learning rules on benchmark problems in feedforward and recurrent neural networks and demonstrate, in wide networks, comparable performance to backpropagation. The proposed rules are particularly effective in low data regimes, which are common in biological learning settings.

        ----

        ## [95] Improving Language Models by Retrieving from Trillions of Tokens

        **Authors**: *Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, Laurent Sifre*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/borgeaud22a.html](https://proceedings.mlr.press/v162/borgeaud22a.html)

        **Abstract**:

        We enhance auto-regressive language models by conditioning on document chunks retrieved from a large corpus, based on local similarity with preceding tokens. With a 2 trillion token database, our Retrieval-Enhanced Transformer (RETRO) obtains comparable performance to GPT-3 and Jurassic-1 on the Pile, despite using 25{\texttimes} fewer parameters. After fine-tuning, RETRO performance translates to downstream knowledge-intensive tasks such as question answering. RETRO combines a frozen Bert retriever, a differentiable encoder and a chunked cross-attention mechanism to predict tokens based on an order of magnitude more data than what is typically consumed during training. We typically train RETRO from scratch, yet can also rapidly RETROfit pre-trained transformers with retrieval and still achieve good performance. Our work opens up new avenues for improving language models through explicit memory at unprecedented scale.

        ----

        ## [96] Lie Point Symmetry Data Augmentation for Neural PDE Solvers

        **Authors**: *Johannes Brandstetter, Max Welling, Daniel E. Worrall*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/brandstetter22a.html](https://proceedings.mlr.press/v162/brandstetter22a.html)

        **Abstract**:

        Neural networks are increasingly being used to solve partial differential equations (PDEs), replacing slower numerical solvers. However, a critical issue is that neural PDE solvers require high-quality ground truth data, which usually must come from the very solvers they are designed to replace. Thus, we are presented with a proverbial chicken-and-egg problem. In this paper, we present a method, which can partially alleviate this problem, by improving neural PDE solver sample complexity—Lie point symmetry data augmentation (LPSDA). In the context of PDEs, it turns out we are able to quantitatively derive an exhaustive list of data transformations, based on the Lie point symmetry group of the PDEs in question, something not possible in other application areas. We present this framework and demonstrate how it can easily be deployed to improve neural PDE solver sample complexity by an order of magnitude.

        ----

        ## [97] An iterative clustering algorithm for the Contextual Stochastic Block Model with optimality guarantees

        **Authors**: *Guillaume Braun, Hemant Tyagi, Christophe Biernacki*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/braun22a.html](https://proceedings.mlr.press/v162/braun22a.html)

        **Abstract**:

        Real-world networks often come with side information that can help to improve the performance of network analysis tasks such as clustering. Despite a large number of empirical and theoretical studies conducted on network clustering methods during the past decade, the added value of side information and the methods used to incorporate it optimally in clustering algorithms are relatively less understood. We propose a new iterative algorithm to cluster networks with side information for nodes (in the form of covariates) and show that our algorithm is optimal under the Contextual Symmetric Stochastic Block Model. Our algorithm can be applied to general Contextual Stochastic Block Models and avoids hyperparameter tuning in contrast to previously proposed methods. We confirm our theoretical results on synthetic data experiments where our algorithm significantly outperforms other methods, and show that it can also be applied to signed graphs. Finally we demonstrate the practical interest of our method on real data.

        ----

        ## [98] Tractable Dendritic RNNs for Reconstructing Nonlinear Dynamical Systems

        **Authors**: *Manuel Brenner, Florian Hess, Jonas M. Mikhaeil, Leonard F. Bereska, Zahra Monfared, Po-Chen Kuo, Daniel Durstewitz*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/brenner22a.html](https://proceedings.mlr.press/v162/brenner22a.html)

        **Abstract**:

        In many scientific disciplines, we are interested in inferring the nonlinear dynamical system underlying a set of observed time series, a challenging task in the face of chaotic behavior and noise. Previous deep learning approaches toward this goal often suffered from a lack of interpretability and tractability. In particular, the high-dimensional latent spaces often required for a faithful embedding, even when the underlying dynamics lives on a lower-dimensional manifold, can hamper theoretical analysis. Motivated by the emerging principles of dendritic computation, we augment a dynamically interpretable and mathematically tractable piecewise-linear (PL) recurrent neural network (RNN) by a linear spline basis expansion. We show that this approach retains all the theoretically appealing properties of the simple PLRNN, yet boosts its capacity for approximating arbitrary nonlinear dynamical systems in comparatively low dimensions. We employ two frameworks for training the system, one combining BPTT with teacher forcing, and another based on fast and scalable variational inference. We show that the dendritically expanded PLRNN achieves better reconstructions with fewer parameters and dimensions on various dynamical systems benchmarks and compares favorably to other methods, while retaining a tractable and interpretable structure.

        ----

        ## [99] Learning to Predict Graphs with Fused Gromov-Wasserstein Barycenters

        **Authors**: *Luc Brogat-Motte, Rémi Flamary, Céline Brouard, Juho Rousu, Florence d'Alché-Buc*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/brogat-motte22a.html](https://proceedings.mlr.press/v162/brogat-motte22a.html)

        **Abstract**:

        This paper introduces a novel and generic framework to solve the flagship task of supervised labeled graph prediction by leveraging Optimal Transport tools. We formulate the problem as regression with the Fused Gromov-Wasserstein (FGW) loss and propose a predictive model relying on a FGW barycenter whose weights depend on inputs. First we introduce a non-parametric estimator based on kernel ridge regression for which theoretical results such as consistency and excess risk bound are proved. Next we propose an interpretable parametric model where the barycenter weights are modeled with a neural network and the graphs on which the FGW barycenter is calculated are additionally learned. Numerical experiments show the strength of the method and its ability to interpolate in the labeled graph space on simulated data and on a difficult metabolic identification problem where it can reach very good performance with very little engineering.

        ----

        ## [100] Efficient Learning of CNNs using Patch Based Features

        **Authors**: *Alon Brutzkus, Amir Globerson, Eran Malach, Alon Regev Netser, Shai Shalev-Shwartz*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/brutzkus22a.html](https://proceedings.mlr.press/v162/brutzkus22a.html)

        **Abstract**:

        Recent work has demonstrated the effectiveness of using patch based representations when learning from image data. Here we provide theoretical support for this observation, by showing that a simple semi-supervised algorithm that uses patch statistics can efficiently learn labels produced by a one-hidden-layer Convolutional Neural Network (CNN). Since CNNs are known to be computationally hard to learn in the worst case, our analysis holds under some distributional assumptions. We show that these assumptions are necessary and sufficient for our results to hold. We verify that the distributional assumptions hold on real-world data by experimenting on the CIFAR-10 dataset, and find that the analyzed algorithm outperforms a vanilla one-hidden-layer CNN. Finally, we demonstrate that by running the algorithm in a layer-by-layer fashion we can build a deep model which gives further improvements, hinting that this method provides insights about the behavior of deep CNNs.

        ----

        ## [101] Causal structure-based root cause analysis of outliers

        **Authors**: *Kailash Budhathoki, Lenon Minorics, Patrick Blöbaum, Dominik Janzing*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/budhathoki22a.html](https://proceedings.mlr.press/v162/budhathoki22a.html)

        **Abstract**:

        Current techniques for explaining outliers cannot tell what caused the outliers. We present a formal method to identify "root causes" of outliers, amongst variables. The method requires a causal graph of the variables along with the functional causal model. It quantifies the contribution of each variable to the target outlier score, which explains to what extent each variable is a "root cause" of the target outlier. We study the empirical performance of the method through simulations and present a real-world case study identifying "root causes" of extreme river flows.

        ----

        ## [102] IGLUE: A Benchmark for Transfer Learning across Modalities, Tasks, and Languages

        **Authors**: *Emanuele Bugliarello, Fangyu Liu, Jonas Pfeiffer, Siva Reddy, Desmond Elliott, Edoardo Maria Ponti, Ivan Vulic*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/bugliarello22a.html](https://proceedings.mlr.press/v162/bugliarello22a.html)

        **Abstract**:

        Reliable evaluation benchmarks designed for replicability and comprehensiveness have driven progress in machine learning. Due to the lack of a multilingual benchmark, however, vision-and-language research has mostly focused on English language tasks. To fill this gap, we introduce the Image-Grounded Language Understanding Evaluation benchmark. IGLUE brings together{—}by both aggregating pre-existing datasets and creating new ones{—}visual question answering, cross-modal retrieval, grounded reasoning, and grounded entailment tasks across 20 diverse languages. Our benchmark enables the evaluation of multilingual multimodal models for transfer learning, not only in a zero-shot setting, but also in newly defined few-shot learning setups. Based on the evaluation of the available state-of-the-art models, we find that translate-test transfer is superior to zero-shot transfer and that few-shot learning is hard to harness for many tasks. Moreover, downstream performance is partially explained by the amount of available unlabelled textual data for pretraining, and only weakly by the typological distance of target{–}source languages. We hope to encourage future research efforts in this area by releasing the benchmark to the community.

        ----

        ## [103] Interactive Inverse Reinforcement Learning for Cooperative Games

        **Authors**: *Thomas Kleine Büning, Anne-Marie George, Christos Dimitrakakis*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/buning22a.html](https://proceedings.mlr.press/v162/buning22a.html)

        **Abstract**:

        We study the problem of designing autonomous agents that can learn to cooperate effectively with a potentially suboptimal partner while having no access to the joint reward function. This problem is modeled as a cooperative episodic two-agent Markov decision process. We assume control over only the first of the two agents in a Stackelberg formulation of the game, where the second agent is acting so as to maximise expected utility given the first agent’s policy. How should the first agent act in order to learn the joint reward function as quickly as possible and so that the joint policy is as close to optimal as possible? We analyse how knowledge about the reward function can be gained in this interactive two-agent scenario. We show that when the learning agent’s policies have a significant effect on the transition function, the reward function can be learned efficiently.

        ----

        ## [104] Convolutional and Residual Networks Provably Contain Lottery Tickets

        **Authors**: *Rebekka Burkholz*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/burkholz22a.html](https://proceedings.mlr.press/v162/burkholz22a.html)

        **Abstract**:

        The Lottery Ticket Hypothesis continues to have a profound practical impact on the quest for small scale deep neural networks that solve modern deep learning tasks at competitive performance. These lottery tickets are identified by pruning large randomly initialized neural networks with architectures that are as diverse as their applications. Yet, theoretical insights that attest their existence have been mostly focused on deed fully-connected feed forward networks with ReLU activation functions. We prove that also modern architectures consisting of convolutional and residual layers that can be equipped with almost arbitrary activation functions can contain lottery tickets with high probability.

        ----

        ## [105] Near-Optimal Algorithms for Autonomous Exploration and Multi-Goal Stochastic Shortest Path

        **Authors**: *Haoyuan Cai, Tengyu Ma, Simon S. Du*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cai22a.html](https://proceedings.mlr.press/v162/cai22a.html)

        **Abstract**:

        We revisit the incremental autonomous exploration problem proposed by Lim and Auer (2012). In this setting, the agent aims to learn a set of near-optimal goal-conditioned policies to reach the $L$-controllable states: states that are incrementally reachable from an initial state $s_0$ within $L$ steps in expectation. We introduce a new algorithm with stronger sample complexity bounds than existing ones. Furthermore, we also prove the first lower bound for the autonomous exploration problem. In particular, the lower bound implies that our proposed algorithm, Value-Aware Autonomous Exploration, is nearly minimax-optimal when the number of $L$-controllable states grows polynomially with respect to $L$. Key in our algorithm design is a connection between autonomous exploration and multi-goal stochastic shortest path, a new problem that naturally generalizes the classical stochastic shortest path problem. This new problem and its connection to autonomous exploration can be of independent interest.

        ----

        ## [106] Convergence of Invariant Graph Networks

        **Authors**: *Chen Cai, Yusu Wang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cai22b.html](https://proceedings.mlr.press/v162/cai22b.html)

        **Abstract**:

        Although theoretical properties such as expressive power and over-smoothing of graph neural networks (GNN) have been extensively studied recently, its convergence property is a relatively new direction. In this paper, we investigate the convergence of one powerful GNN, Invariant Graph Network (IGN) over graphs sampled from graphons. We first prove the stability of linear layers for general $k$-IGN (of order $k$) based on a novel interpretation of linear equivariant layers. Building upon this result, we prove the convergence of $k$-IGN under the model of \citet{ruiz2020graphon}, where we access the edge weight but the convergence error is measured for graphon inputs. Under the more natural (and more challenging) setting of \citet{keriven2020convergence} where one can only access 0-1 adjacency matrix sampled according to edge probability, we first show a negative result that the convergence of any IGN is not possible. We then obtain the convergence of a subset of IGNs, denoted as IGN-small, after the edge probability estimation. We show that IGN-small still contains function class rich enough that can approximate spectral GNNs arbitrarily well. Lastly, we perform experiments on various graphon models to verify our statements.

        ----

        ## [107] Reinforcement Learning from Partial Observation: Linear Function Approximation with Provable Sample Efficiency

        **Authors**: *Qi Cai, Zhuoran Yang, Zhaoran Wang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cai22c.html](https://proceedings.mlr.press/v162/cai22c.html)

        **Abstract**:

        We study reinforcement learning for partially observed Markov decision processes (POMDPs) with infinite observation and state spaces, which remains less investigated theoretically. To this end, we make the first attempt at bridging partial observability and function approximation for a class of POMDPs with a linear structure. In detail, we propose a reinforcement learning algorithm (Optimistic Exploration via Adversarial Integral Equation or OP-TENET) that attains an $\epsilon$-optimal policy within $O(1/\epsilon^2)$ episodes. In particular, the sample complexity scales polynomially in the intrinsic dimension of the linear structure and is independent of the size of the observation and state spaces. The sample efficiency of OP-TENET is enabled by a sequence of ingredients: (i) a Bellman operator with finite memory, which represents the value function in a recursive manner, (ii) the identification and estimation of such an operator via an adversarial integral equation, which features a smoothed discriminator tailored to the linear structure, and (iii) the exploration of the observation and state spaces via optimism, which is based on quantifying the uncertainty in the adversarial integral equation.

        ----

        ## [108] Scaling Gaussian Process Optimization by Evaluating a Few Unique Candidates Multiple Times

        **Authors**: *Daniele Calandriello, Luigi Carratino, Alessandro Lazaric, Michal Valko, Lorenzo Rosasco*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/calandriello22a.html](https://proceedings.mlr.press/v162/calandriello22a.html)

        **Abstract**:

        Computing a Gaussian process (GP) posterior has a computational cost cubical in the number of historical points. A reformulation of the same GP posterior highlights that this complexity mainly depends on how many unique historical points are considered. This can have important implication in active learning settings, where the set of historical points is constructed sequentially by the learner. We show that sequential black-box optimization based on GPs (GP-Opt) can be made efficient by sticking to a candidate solution for multiple evaluation steps and switch only when necessary. Limiting the number of switches also limits the number of unique points in the history of the GP. Thus, the efficient GP reformulation can be used to exactly and cheaply compute the posteriors required to run the GP-Opt algorithms. This approach is especially useful in real-world applications of GP-Opt with high switch costs (e.g. switching chemicals in wet labs, data/model loading in hyperparameter optimization). As examples of this meta-approach, we modify two well-established GP-Opt algorithms, GP-UCB and GP-EI, to switch candidates as infrequently as possible adapting rules from batched GP-Opt. These versions preserve all the theoretical no-regret guarantees while improving practical aspects of the algorithms such as runtime, memory complexity, and the ability of batching candidates and evaluating them in parallel.

        ----

        ## [109] Adaptive Gaussian Process Change Point Detection

        **Authors**: *Edoardo Caldarelli, Philippe Wenk, Stefan Bauer, Andreas Krause*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/caldarelli22a.html](https://proceedings.mlr.press/v162/caldarelli22a.html)

        **Abstract**:

        Detecting change points in time series, i.e., points in time at which some observed process suddenly changes, is a fundamental task that arises in many real-world applications, with consequences for safety and reliability. In this work, we propose ADAGA, a novel Gaussian process-based solution to this problem, that leverages a powerful heuristics we developed based on statistical hypothesis testing. In contrast to prior approaches, ADAGA adapts to changes both in mean and covariance structure of the temporal process. In extensive experiments, we show its versatility and applicability to different classes of change points, demonstrating that it is significantly more accurate than current state-of-the-art alternatives.

        ----

        ## [110] Measuring dissimilarity with diffeomorphism invariance

        **Authors**: *Théophile Cantelobre, Carlo Ciliberto, Benjamin Guedj, Alessandro Rudi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cantelobre22a.html](https://proceedings.mlr.press/v162/cantelobre22a.html)

        **Abstract**:

        Measures of similarity (or dissimilarity) are a key ingredient to many machine learning algorithms. We introduce DID, a pairwise dissimilarity measure applicable to a wide range of data spaces, which leverages the data’s internal structure to be invariant to diffeomorphisms. We prove that DID enjoys properties which make it relevant for theoretical study and practical use. By representing each datum as a function, DID is defined as the solution to an optimization problem in a Reproducing Kernel Hilbert Space and can be expressed in closed-form. In practice, it can be efficiently approximated via Nystr{ö}m sampling. Empirical experiments support the merits of DID.

        ----

        ## [111] A Model-Agnostic Randomized Learning Framework based on Random Hypothesis Subspace Sampling

        **Authors**: *Yiting Cao, Chao Lan*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cao22a.html](https://proceedings.mlr.press/v162/cao22a.html)

        **Abstract**:

        We propose a model-agnostic randomized learning framework based on Random Hypothesis Subspace Sampling (RHSS). Given any hypothesis class, it randomly samples $k$ hypotheses and learns a near-optimal model from their span by simply solving a linear least square problem in $O(n k^2)$ time, where $n$ is the number of training instances. On the theory side, we derive the performance guarantee of RHSS from a generic subspace approximation perspective, leveraging properties of metric entropy and random matrices. On the practical side, we apply the RHSS framework to learn kernel, network and tree based models. Experimental results show they converge efficiently as $k$ increases and outperform their model-specific counterparts including random Fourier feature, random vector functional link and extra tree on real-world data sets.

        ----

        ## [112] Gaussian Process Uniform Error Bounds with Unknown Hyperparameters for Safety-Critical Applications

        **Authors**: *Alexandre Capone, Armin Lederer, Sandra Hirche*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/capone22a.html](https://proceedings.mlr.press/v162/capone22a.html)

        **Abstract**:

        Gaussian processes have become a promising tool for various safety-critical settings, since the posterior variance can be used to directly estimate the model error and quantify risk. However, state-of-the-art techniques for safety-critical settings hinge on the assumption that the kernel hyperparameters are known, which does not apply in general. To mitigate this, we introduce robust Gaussian process uniform error bounds in settings with unknown hyperparameters. Our approach computes a confidence region in the space of hyperparameters, which enables us to obtain a probabilistic upper bound for the model error of a Gaussian process with arbitrary hyperparameters. We do not require to know any bounds for the hyperparameters a priori, which is an assumption commonly found in related work. Instead, we are able to derive bounds from data in an intuitive fashion. We additionally employ the proposed technique to derive performance guarantees for a class of learning-based control problems. Experiments show that the bound performs significantly better than vanilla and fully Bayesian Gaussian processes.

        ----

        ## [113] Burst-Dependent Plasticity and Dendritic Amplification Support Target-Based Learning and Hierarchical Imitation Learning

        **Authors**: *Cristiano Capone, Cosimo Lupo, Paolo Muratore, Pier Stanislao Paolucci*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/capone22b.html](https://proceedings.mlr.press/v162/capone22b.html)

        **Abstract**:

        The brain can learn to solve a wide range of tasks with high temporal and energetic efficiency. However, most biological models are composed of simple single-compartment neurons and cannot achieve the state-of-the-art performances of artificial intelligence. We propose a multi-compartment model of pyramidal neuron, in which bursts and dendritic input segregation give the possibility to plausibly support a biological target-based learning. In target-based learning, the internal solution of a problem (a spatio-temporal pattern of bursts in our case) is suggested to the network, bypassing the problems of error backpropagation and credit assignment. Finally, we show that this neuronal architecture naturally supports the orchestration of “hierarchical imitation learning”, enabling the decomposition of challenging long-horizon decision-making tasks into simpler subtasks.

        ----

        ## [114] A Marriage between Adversarial Team Games and 2-player Games: Enabling Abstractions, No-regret Learning, and Subgame Solving

        **Authors**: *Luca Carminati, Federico Cacciamani, Marco Ciccone, Nicola Gatti*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/carminati22a.html](https://proceedings.mlr.press/v162/carminati22a.html)

        **Abstract**:

        Ex ante correlation is becoming the mainstream approach for sequential adversarial team games, where a team of players faces another team in a zero-sum game. It is known that team members’ asymmetric information makes both equilibrium computation \textsf{APX}-hard and team’s strategies not directly representable on the game tree. This latter issue prevents the adoption of successful tools for huge 2-player zero-sum games such as, e.g., abstractions, no-regret learning, and subgame solving. This work shows that we can recover from this weakness by bridging the gap between sequential adversarial team games and 2-player games. In particular, we propose a new, suitable game representation that we call team-public-information, in which a team is represented as a single coordinator who only knows information common to the whole team and prescribes to each member an action for any possible private state. The resulting representation is highly explainable, being a 2-player tree in which the team’s strategies are behavioral with a direct interpretation and more expressive than the original extensive form when designing abstractions. Furthermore, we prove payoff equivalence of our representation, and we provide techniques that, starting directly from the extensive form, generate dramatically more compact representations without information loss. Finally, we experimentally evaluate our techniques when applied to a standard testbed, comparing their performance with the current state of the art.

        ----

        ## [115] RECAPP: Crafting a More Efficient Catalyst for Convex Optimization

        **Authors**: *Yair Carmon, Arun Jambulapati, Yujia Jin, Aaron Sidford*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/carmon22a.html](https://proceedings.mlr.press/v162/carmon22a.html)

        **Abstract**:

        The accelerated proximal point method (APPA), also known as "Catalyst", is a well-established reduction from convex optimization to approximate proximal point computation (i.e., regularized minimization). This reduction is conceptually elegant and yields strong convergence rate guarantees. However, these rates feature an extraneous logarithmic term arising from the need to compute each proximal point to high accuracy. In this work, we propose a novel Relaxed Error Criterion for Accelerated Proximal Point (RECAPP) that eliminates the need for high accuracy subproblem solutions. We apply RECAPP to two canonical problems: finite-sum and max-structured minimization. For finite-sum problems, we match the best known complexity, previously obtained by carefully-designed problem-specific algorithms. For minimizing max_y f(x,y) where f is convex in x and strongly-concave in y, we improve on the best known (Catalyst-based) bound by a logarithmic factor.

        ----

        ## [116] Estimating and Penalizing Induced Preference Shifts in Recommender Systems

        **Authors**: *Micah D. Carroll, Anca D. Dragan, Stuart Russell, Dylan Hadfield-Menell*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/carroll22a.html](https://proceedings.mlr.press/v162/carroll22a.html)

        **Abstract**:

        The content that a recommender system (RS) shows to users influences them. Therefore, when choosing a recommender to deploy, one is implicitly also choosing to induce specific internal states in users. Even more, systems trained via long-horizon optimization will have direct incentives to manipulate users, e.g. shift their preferences so they are easier to satisfy. We focus on induced preference shifts in users. We argue that {–} before deployment {–} system designers should: estimate the shifts a recommender would induce; evaluate whether such shifts would be undesirable; and perhaps even actively optimize to avoid problematic shifts. These steps involve two challenging ingredients: estimation requires anticipating how hypothetical policies would influence user preferences if deployed {–} we do this by using historical user interaction data to train a predictive user model which implicitly contains their preference dynamics; evaluation and optimization additionally require metrics to assess whether such influences are manipulative or otherwise unwanted {–} we use the notion of "safe shifts", that define a trust region within which behavior is safe: for instance, the natural way in which users would shift without interference from the system could be deemed "safe". In simulated experiments, we show that our learned preference dynamics model is effective in estimating user preferences and how they would respond to new recommenders. Additionally, we show that recommenders that optimize for staying in the trust region can avoid manipulative behaviors while still generating engagement.

        ----

        ## [117] YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for Everyone

        **Authors**: *Edresson Casanova, Julian Weber, Christopher Dane Shulby, Arnaldo Cândido Júnior, Eren Gölge, Moacir A. Ponti*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/casanova22a.html](https://proceedings.mlr.press/v162/casanova22a.html)

        **Abstract**:

        YourTTS brings the power of a multilingual approach to the task of zero-shot multi-speaker TTS. Our method builds upon the VITS model and adds several novel modifications for zero-shot multi-speaker and multilingual training. We achieved state-of-the-art (SOTA) results in zero-shot multi-speaker TTS and results comparable to SOTA in zero-shot voice conversion on the VCTK dataset. Additionally, our approach achieves promising results in a target language with a single-speaker dataset, opening possibilities for zero-shot multi-speaker TTS and zero-shot voice conversion systems in low-resource languages. Finally, it is possible to fine-tune the YourTTS model with less than 1 minute of speech and achieve state-of-the-art results in voice similarity and with reasonable quality. This is important to allow synthesis for speakers with a very different voice or recording characteristics from those seen during training.

        ----

        ## [118] The Infinite Contextual Graph Markov Model

        **Authors**: *Daniele Castellana, Federico Errica, Davide Bacciu, Alessio Micheli*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/castellana22a.html](https://proceedings.mlr.press/v162/castellana22a.html)

        **Abstract**:

        The Contextual Graph Markov Model (CGMM) is a deep, unsupervised, and probabilistic model for graphs that is trained incrementally on a layer-by-layer basis. As with most Deep Graph Networks, an inherent limitation is the need to perform an extensive model selection to choose the proper size of each layer’s latent representation. In this paper, we address this problem by introducing the Infinite Contextual Graph Markov Model (iCGMM), the first deep Bayesian nonparametric model for graph learning. During training, iCGMM can adapt the complexity of each layer to better fit the underlying data distribution. On 8 graph classification tasks, we show that iCGMM: i) successfully recovers or improves CGMM’s performances while reducing the hyper-parameters’ search space; ii) performs comparably to most end-to-end supervised methods. The results include studies on the importance of depth, hyper-parameters, and compression of the graph embeddings. We also introduce a novel approximated inference procedure that better deals with larger graph topologies.

        ----

        ## [119] Compressed-VFL: Communication-Efficient Learning with Vertically Partitioned Data

        **Authors**: *Timothy J. Castiglia, Anirban Das, Shiqiang Wang, Stacy Patterson*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/castiglia22a.html](https://proceedings.mlr.press/v162/castiglia22a.html)

        **Abstract**:

        We propose Compressed Vertical Federated Learning (C-VFL) for communication-efficient training on vertically partitioned data. In C-VFL, a server and multiple parties collaboratively train a model on their respective features utilizing several local iterations and sharing compressed intermediate results periodically. Our work provides the first theoretical analysis of the effect message compression has on distributed training over vertically partitioned data. We prove convergence of non-convex objectives at a rate of $O(\frac{1}{\sqrt{T}})$ when the compression error is bounded over the course of training. We provide specific requirements for convergence with common compression techniques, such as quantization and top-$k$ sparsification. Finally, we experimentally show compression can reduce communication by over $90%$ without a significant decrease in accuracy over VFL without compression.

        ----

        ## [120] Online Learning with Knapsacks: the Best of Both Worlds

        **Authors**: *Matteo Castiglioni, Andrea Celli, Christian Kroer*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/castiglioni22a.html](https://proceedings.mlr.press/v162/castiglioni22a.html)

        **Abstract**:

        We study online learning problems in which a decision maker wants to maximize their expected reward without violating a finite set of $m$ resource constraints. By casting the learning process over a suitably defined space of strategy mixtures, we recover strong duality on a Lagrangian relaxation of the underlying optimization problem, even for general settings with non-convex reward and resource-consumption functions. Then, we provide the first best-of-both-worlds type framework for this setting, with no-regret guarantees both under stochastic and adversarial inputs. Our framework yields the same regret guarantees of prior work in the stochastic case. On the other hand, when budgets grow at least linearly in the time horizon, it allows us to provide a constant competitive ratio in the adversarial case, which improves over the $O(m \log T)$ competitive ratio of Immorlica et al. [FOCS’19]. Moreover, our framework allows the decision maker to handle non-convex reward and cost functions. We provide two game-theoretic applications of our framework to give further evidence of its flexibility.

        ----

        ## [121] Stabilizing Off-Policy Deep Reinforcement Learning from Pixels

        **Authors**: *Edoardo Cetin, Philip J. Ball, Stephen J. Roberts, Oya Çeliktutan*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cetin22a.html](https://proceedings.mlr.press/v162/cetin22a.html)

        **Abstract**:

        Off-policy reinforcement learning (RL) from pixel observations is notoriously unstable. As a result, many successful algorithms must combine different domain-specific practices and auxiliary losses to learn meaningful behaviors in complex environments. In this work, we provide novel analysis demonstrating that these instabilities arise from performing temporal-difference learning with a convolutional encoder and low-magnitude rewards. We show that this new visual deadly triad causes unstable training and premature convergence to degenerate solutions, a phenomenon we name catastrophic self-overfitting. Based on our analysis, we propose A-LIX, a method providing adaptive regularization to the encoder’s gradients that explicitly prevents the occurrence of catastrophic self-overfitting using a dual objective. By applying A-LIX, we significantly outperform the prior state-of-the-art on the DeepMind Control and Atari benchmarks without any data augmentation or auxiliary losses.

        ----

        ## [122] Accelerated, Optimal and Parallel: Some results on model-based stochastic optimization

        **Authors**: *Karan N. Chadha, Gary Cheng, John C. Duchi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chadha22a.html](https://proceedings.mlr.press/v162/chadha22a.html)

        **Abstract**:

        The Approximate-Proximal Point (APROX) family of model-based stochastic optimization algorithms improve over standard stochastic gradient methods, as they are robust to step size choices, adaptive to problem difficulty, converge on a broader range of problems than stochastic gradient methods, and converge very fast on interpolation problems, all while retaining nice minibatching properties \cite{AsiDu19siopt,AsiChChDu20}. In this paper, we propose an acceleration scheme for the APROX family and provide non-asymptotic convergence guarantees, which are order-optimal in all problem-dependent constants and provide even larger minibatching speedups. For interpolation problems where the objective satisfies additional growth conditions, we show that our algorithm achieves linear convergence rates for a wide range of stepsizes. In this setting, we also prove matching lower bounds, identifying new fundamental constants and showing the optimality of the APROX family. We corroborate our theoretical results with empirical testing to demonstrate the gains accurate modeling, acceleration, and minibatching provide.

        ----

        ## [123] Robust Imitation Learning against Variations in Environment Dynamics

        **Authors**: *Jongseong Chae, Seungyul Han, Whiyoung Jung, Myungsik Cho, Sungho Choi, Youngchul Sung*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chae22a.html](https://proceedings.mlr.press/v162/chae22a.html)

        **Abstract**:

        In this paper, we propose a robust imitation learning (IL) framework that improves the robustness of IL when environment dynamics are perturbed. The existing IL framework trained in a single environment can catastrophically fail with perturbations in environment dynamics because it does not capture the situation that underlying environment dynamics can be changed. Our framework effectively deals with environments with varying dynamics by imitating multiple experts in sampled environment dynamics to enhance the robustness in general variations in environment dynamics. In order to robustly imitate the multiple sample experts, we minimize the risk with respect to the Jensen-Shannon divergence between the agent’s policy and each of the sample experts. Numerical results show that our algorithm significantly improves robustness against dynamics perturbations compared to conventional IL baselines.

        ----

        ## [124] Fairness with Adaptive Weights

        **Authors**: *Junyi Chai, Xiaoqian Wang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chai22a.html](https://proceedings.mlr.press/v162/chai22a.html)

        **Abstract**:

        Fairness is now an important issue in machine learning. There are arising concerns that automated decision-making systems reflect real-world biases. Although a wide range of fairness-related methods have been proposed in recent years, the under-representation problem has been less studied. Due to the uneven distribution of samples from different populations, machine learning models tend to be biased against minority groups when trained by minimizing the average empirical risk across all samples. In this paper, we propose a novel adaptive reweighing method to address representation bias. The goal of our method is to achieve group-level balance among different demographic groups by learning adaptive weights for each sample. Our approach emphasizes more on error-prone samples in prediction and enhances adequate representation of minority groups for fairness. We derive a closed-form solution for adaptive weight assignment and propose an efficient algorithm with theoretical convergence guarantees. We theoretically analyze the fairness of our model and empirically verify that our method strikes a balance between fairness and accuracy. In experiments, our method achieves comparable or better performance than state-of-the-art methods in both classification and regression tasks. Furthermore, our method exhibits robustness to label noise on various benchmark datasets.

        ----

        ## [125] UNIREX: A Unified Learning Framework for Language Model Rationale Extraction

        **Authors**: *Aaron Chan, Maziar Sanjabi, Lambert Mathias, Liang Tan, Shaoliang Nie, Xiaochang Peng, Xiang Ren, Hamed Firooz*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chan22a.html](https://proceedings.mlr.press/v162/chan22a.html)

        **Abstract**:

        An extractive rationale explains a language model’s (LM’s) prediction on a given task instance by highlighting the text inputs that most influenced the prediction. Ideally, rationale extraction should be faithful (reflective of LM’s actual behavior) and plausible (convincing to humans), without compromising the LM’s (i.e., task model’s) task performance. Although attribution algorithms and select-predict pipelines are commonly used in rationale extraction, they both rely on certain heuristics that hinder them from satisfying all three desiderata. In light of this, we propose UNIREX, a flexible learning framework which generalizes rationale extractor optimization as follows: (1) specify architecture for a learned rationale extractor; (2) select explainability objectives (\ie faithfulness and plausibility criteria); and (3) jointly train the task model and rationale extractor on the task using selected objectives. UNIREX enables replacing prior works’ heuristic design choices with a generic learned rationale extractor in (1) and optimizing it for all three desiderata in (2)-(3). To facilitate comparison between methods w.r.t. multiple desiderata, we introduce the Normalized Relative Gain (NRG) metric. On five English text classification datasets, our best UNIREX configuration outperforms baselines by an average of 32.9% NRG. Plus, UNIREX rationale extractors’ faithfulness can even generalize to unseen datasets and tasks.

        ----

        ## [126] Revisiting Label Smoothing and Knowledge Distillation Compatibility: What was Missing?

        **Authors**: *Keshigeyan Chandrasegaran, Ngoc-Trung Tran, Yunqing Zhao, Ngai-Man Cheung*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chandrasegaran22a.html](https://proceedings.mlr.press/v162/chandrasegaran22a.html)

        **Abstract**:

        This work investigates the compatibility between label smoothing (LS) and knowledge distillation (KD). Contemporary findings addressing this thesis statement take dichotomous standpoints: Muller et al. (2019) and Shen et al. (2021b). Critically, there is no effort to understand and resolve these contradictory findings, leaving the primal question \text{-} to smooth or not to smooth a teacher network? \text{-} unanswered. The main contributions of our work are the discovery, analysis and validation of systematic diffusion as the missing concept which is instrumental in understanding and resolving these contradictory findings. This systematic diffusion essentially curtails the benefits of distilling from an LS-trained teacher, thereby rendering KD at increased temperatures ineffective. Our discovery is comprehensively supported by large-scale experiments, analyses and case studies including image classification, neural machine translation and compact student distillation tasks spanning across multiple datasets and teacher-student architectures. Based on our analysis, we suggest practitioners to use an LS-trained teacher with a low-temperature transfer to achieve high performance students. Code and models are available at https://keshik6.github.io/revisiting-ls-kd-compatibility/

        ----

        ## [127] Style Equalization: Unsupervised Learning of Controllable Generative Sequence Models

        **Authors**: *Jen-Hao Rick Chang, Ashish Shrivastava, Hema Koppula, Xiaoshuai Zhang, Oncel Tuzel*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chang22a.html](https://proceedings.mlr.press/v162/chang22a.html)

        **Abstract**:

        Controllable generative sequence models with the capability to extract and replicate the style of specific examples enable many applications, including narrating audiobooks in different voices, auto-completing and auto-correcting written handwriting, and generating missing training samples for downstream recognition tasks. However, under an unsupervised-style setting, typical training algorithms for controllable sequence generative models suffer from the training-inference mismatch, where the same sample is used as content and style input during training but unpaired samples are given during inference. In this paper, we tackle the training-inference mismatch encountered during unsupervised learning of controllable generative sequence models. The proposed method is simple yet effective, where we use a style transformation module to transfer target style information into an unrelated style input. This method enables training using unpaired content and style samples and thereby mitigate the training-inference mismatch. We apply style equalization to text-to-speech and text-to-handwriting synthesis on three datasets. We conduct thorough evaluation, including both quantitative and qualitative user studies. Our results show that by mitigating the training-inference mismatch with the proposed style equalization, we achieve style replication scores comparable to real data in our user studies.

        ----

        ## [128] Learning Bellman Complete Representations for Offline Policy Evaluation

        **Authors**: *Jonathan D. Chang, Kaiwen Wang, Nathan Kallus, Wen Sun*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chang22b.html](https://proceedings.mlr.press/v162/chang22b.html)

        **Abstract**:

        We study representation learning for Offline Reinforcement Learning (RL), focusing on the important task of Offline Policy Evaluation (OPE). Recent work shows that, in contrast to supervised learning, realizability of the Q-function is not enough for learning it. Two sufficient conditions for sample-efficient OPE are Bellman completeness and coverage. Prior work often assumes that representations satisfying these conditions are given, with results being mostly theoretical in nature. In this work, we propose BCRL, which directly learns from data an approximately linear Bellman complete representation with good coverage. With this learned representation, we perform OPE using Least Square Policy Evaluation (LSPE) with linear functions in our learned representation. We present an end-to-end theoretical analysis, showing that our two-stage algorithm enjoys polynomial sample complexity provided some representation in the rich class considered is linear Bellman complete. Empirically, we extensively evaluate our algorithm on challenging, image-based continuous control tasks from the Deepmind Control Suite. We show our representation enables better OPE compared to previous representation learning methods developed for off-policy RL (e.g., CURL, SPR). BCRL achieve competitive OPE error with the state-of-the-art method Fitted Q-Evaluation (FQE), and beats FQE when evaluating beyond the initial state distribution. Our ablations show that both linear Bellman complete and coverage components of our method are crucial.

        ----

        ## [129] Sample Efficient Learning of Predictors that Complement Humans

        **Authors**: *Mohammad-Amin Charusaie, Hussein Mozannar, David A. Sontag, Samira Samadi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/charusaie22a.html](https://proceedings.mlr.press/v162/charusaie22a.html)

        **Abstract**:

        One of the goals of learning algorithms is to complement and reduce the burden on human decision makers. The expert deferral setting wherein an algorithm can either predict on its own or defer the decision to a downstream expert helps accomplish this goal. A fundamental aspect of this setting is the need to learn complementary predictors that improve on the human’s weaknesses rather than learning predictors optimized for average error. In this work, we provide the first theoretical analysis of the benefit of learning complementary predictors in expert deferral. To enable efficiently learning such predictors, we consider a family of consistent surrogate loss functions for expert deferral and analyze their theoretical properties. Finally, we design active learning schemes that require minimal amount of data of human expert predictions in order to learn accurate deferral systems.

        ----

        ## [130] Nyström Kernel Mean Embeddings

        **Authors**: *Antoine Chatalic, Nicolas Schreuder, Lorenzo Rosasco, Alessandro Rudi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chatalic22a.html](https://proceedings.mlr.press/v162/chatalic22a.html)

        **Abstract**:

        Kernel mean embeddings are a powerful tool to represent probability distributions over arbitrary spaces as single points in a Hilbert space. Yet, the cost of computing and storing such embeddings prohibits their direct use in large-scale settings. We propose an efficient approximation procedure based on the Nystr{ö}m method, which exploits a small random subset of the dataset. Our main result is an upper bound on the approximation error of this procedure. It yields sufficient conditions on the subsample size to obtain the standard (1/sqrt(n)) rate while reducing computational costs. We discuss applications of this result for the approximation of the maximum mean discrepancy and quadrature rules, and we illustrate our theoretical findings with numerical experiments.

        ----

        ## [131] Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

        **Authors**: *Tianlong Chen, Xuxi Chen, Xiaolong Ma, Yanzhi Wang, Zhangyang Wang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22a.html](https://proceedings.mlr.press/v162/chen22a.html)

        **Abstract**:

        The lottery ticket hypothesis (LTH) has shown that dense models contain highly sparse subnetworks (i.e., winning tickets) that can be trained in isolation to match full accuracy. Despite many exciting efforts being made, there is one "commonsense" rarely challenged: a winning ticket is found by iterative magnitude pruning (IMP) and hence the resultant pruned subnetworks have only unstructured sparsity. That gap limits the appeal of winning tickets in practice, since the highly irregular sparse patterns are challenging to accelerate on hardware. Meanwhile, directly substituting structured pruning for unstructured pruning in IMP damages performance more severely and is usually unable to locate winning tickets. In this paper, we demonstrate the first positive result that a structurally sparse winning ticket can be effectively found in general. The core idea is to append "post-processing techniques" after each round of (unstructured) IMP, to enforce the formation of structural sparsity. Specifically, we first "re-fill" pruned elements back in some channels deemed to be important, and then "re-group" non-zero elements to create flexible group-wise structural patterns. Both our identified channel- and group-wise structural subnetworks win the lottery, with substantial inference speedups readily supported by existing hardware. Extensive experiments, conducted on diverse datasets across multiple network backbones, consistently validate our proposal, showing that the hardware acceleration roadblock of LTH is now removed. Specifically, the structural winning tickets obtain up to {64.93%, 64.84%, 60.23%} running time savings at {36% 80%, 74%, 58%} sparsity on {CIFAR, Tiny-ImageNet, ImageNet}, while maintaining comparable accuracy. Code is at https://github.com/VITA-Group/Structure-LTH.

        ----

        ## [132] Learning Domain Adaptive Object Detection with Probabilistic Teacher

        **Authors**: *Meilin Chen, Weijie Chen, Shicai Yang, Jie Song, Xinchao Wang, Lei Zhang, Yunfeng Yan, Donglian Qi, Yueting Zhuang, Di Xie, Shiliang Pu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22b.html](https://proceedings.mlr.press/v162/chen22b.html)

        **Abstract**:

        Self-training for unsupervised domain adaptive object detection is a challenging task, of which the performance depends heavily on the quality of pseudo boxes. Despite the promising results, prior works have largely overlooked the uncertainty of pseudo boxes during self-training. In this paper, we present a simple yet effective framework, termed as Probabilistic Teacher (PT), which aims to capture the uncertainty of unlabeled target data from a gradually evolving teacher and guides the learning of a student in a mutually beneficial manner. Specifically, we propose to leverage the uncertainty-guided consistency training to promote classification adaptation and localization adaptation, rather than filtering pseudo boxes via an elaborate confidence threshold. In addition, we conduct anchor adaptation in parallel with localization adaptation, since anchor can be regarded as a learnable parameter. Together with this framework, we also present a novel Entropy Focal Loss (EFL) to further facilitate the uncertainty-guided self-training. Equipped with EFL, PT outperforms all previous baselines by a large margin and achieve new state-of-the-arts.

        ----

        ## [133] The Fundamental Price of Secure Aggregation in Differentially Private Federated Learning

        **Authors**: *Wei-Ning Chen, Christopher A. Choquette-Choo, Peter Kairouz, Ananda Theertha Suresh*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22c.html](https://proceedings.mlr.press/v162/chen22c.html)

        **Abstract**:

        We consider the problem of training a $d$ dimensional model with distributed differential privacy (DP) where secure aggregation (SecAgg) is used to ensure that the server only sees the noisy sum of $n$ model updates in every training round. Taking into account the constraints imposed by SecAgg, we characterize the fundamental communication cost required to obtain the best accuracy achievable under $\varepsilon$ central DP (i.e. under a fully trusted server and no communication constraints). Our results show that $\tilde{O}\lp \min(n^2\varepsilon^2, d) \rp$ bits per client are both sufficient and necessary, and this fundamental limit can be achieved by a linear scheme based on sparse random projections. This provides a significant improvement relative to state-of-the-art SecAgg distributed DP schemes which use $\tilde{O}(d\log(d/\varepsilon^2))$ bits per client. Empirically, we evaluate our proposed scheme on real-world federated learning tasks. We find that our theoretical analysis is well matched in practice. In particular, we show that we can reduce the communication cost to under $1.78$ bits per parameter in realistic privacy settings without decreasing test-time performance. Our work hence theoretically and empirically specifies the fundamental price of using SecAgg.

        ----

        ## [134] Perfectly Balanced: Improving Transfer and Robustness of Supervised Contrastive Learning

        **Authors**: *Mayee F. Chen, Daniel Y. Fu, Avanika Narayan, Michael Zhang, Zhao Song, Kayvon Fatahalian, Christopher Ré*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22d.html](https://proceedings.mlr.press/v162/chen22d.html)

        **Abstract**:

        An ideal learned representation should display transferability and robustness. Supervised contrastive learning (SupCon) is a promising method for training accurate models, but produces representations that do not capture these properties due to class collapse—when all points in a class map to the same representation. Recent work suggests that "spreading out" these representations improves them, but the precise mechanism is poorly understood. We argue that creating spread alone is insufficient for better representations, since spread is invariant to permutations within classes. Instead, both the correct degree of spread and a mechanism for breaking this invariance are necessary. We first prove that adding a weighted class-conditional InfoNCE loss to SupCon controls the degree of spread. Next, we study three mechanisms to break permutation invariance: using a constrained encoder, adding a class-conditional autoencoder, and using data augmentation. We show that the latter two encourage clustering of latent subclasses under more realistic conditions than the former. Using these insights, we show that adding a properly-weighted class-conditional InfoNCE loss and a class-conditional autoencoder to SupCon achieves 11.1 points of lift on coarse-to-fine transfer across 5 standard datasets and 4.7 points on worst-group robustness on 3 datasets, setting state-of-the-art on CelebA by 11.5 points.

        ----

        ## [135] Strategies for Safe Multi-Armed Bandits with Logarithmic Regret and Risk

        **Authors**: *Tianrui Chen, Aditya Gangrade, Venkatesh Saligrama*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22e.html](https://proceedings.mlr.press/v162/chen22e.html)

        **Abstract**:

        We investigate a natural but surprisingly unstudied approach to the multi-armed bandit problem under safety risk constraints. Each arm is associated with an unknown law on safety risks and rewards, and the learner’s goal is to maximise reward whilst not playing unsafe arms, as determined by a given threshold on the mean risk. We formulate a pseudo-regret for this setting that enforces this safety constraint in a per-round way by softly penalising any violation, regardless of the gain in reward due to the same. This has practical relevance to scenarios such as clinical trials, where one must maintain safety for each round rather than in an aggregated sense. We describe doubly optimistic strategies for this scenario, which maintain optimistic indices for both safety risk and reward. We show that schema based on both frequentist and Bayesian indices satisfy tight gap-dependent logarithmic regret bounds, and further that these play unsafe arms only logarithmically many times in total. This theoretical analysis is complemented by simulation studies demonstrating the effectiveness of the proposed schema, and probing the domains in which their use is appropriate.

        ----

        ## [136] On the Sample Complexity of Learning Infinite-horizon Discounted Linear Kernel MDPs

        **Authors**: *Yuanzhou Chen, Jiafan He, Quanquan Gu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22f.html](https://proceedings.mlr.press/v162/chen22f.html)

        **Abstract**:

        We study reinforcement learning for infinite-horizon discounted linear kernel MDPs, where the transition probability function is linear in a predefined feature mapping. Existing UCLK \citep{zhou2020provably} algorithm for this setting only has a regret guarantee, which cannot lead to a tight sample complexity bound. In this paper, we extend the uniform-PAC sample complexity from episodic setting to the infinite-horizon discounted setting, and propose a novel algorithm dubbed UPAC-UCLK that achieves an $\Tilde{O}\big(d^2/((1-\gamma)^4\epsilon^2)+1/((1-\gamma)^6\epsilon^2)\big)$ uniform-PAC sample complexity, where $d$ is the dimension of the feature mapping, $\gamma \in(0,1)$ is the discount factor of the MDP and $\epsilon$ is the accuracy parameter. To the best of our knowledge, this is the first $\tilde{O}(1/\epsilon^2)$ sample complexity bound for learning infinite-horizon discounted MDPs with linear function approximation (without access to the generative model).

        ----

        ## [137] Streaming Algorithms for Support-Aware Histograms

        **Authors**: *Justin Y. Chen, Piotr Indyk, Tal Wagner*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22g.html](https://proceedings.mlr.press/v162/chen22g.html)

        **Abstract**:

        Histograms, i.e., piece-wise constant approximations, are a popular tool used to represent data distributions. Traditionally, the difference between the histogram and the underlying distribution (i.e., the approximation error) is measured using the L_p norm, which sums the differences between the two functions over all items in the domain. Although useful in many applications, the drawback of this error measure is that it treats approximation errors of all items in the same way, irrespective of whether the mass of an item is important for the downstream application that uses the approximation. As a result, even relatively simple distributions cannot be approximated by succinct histograms without incurring large error. In this paper, we address this issue by adapting the definition of approximation so that only the errors of the items that belong to the support of the distribution are considered. Under this definition, we develop efficient 1-pass and 2-pass streaming algorithms that compute near-optimal histograms in sub-linear space. We also present lower bounds on the space complexity of this problem. Surprisingly, under this notion of error, there is an exponential gap in the space complexity of 1-pass and 2-pass streaming algorithms. Finally, we demonstrate the utility of our algorithms on a collection of real and synthetic data sets.

        ----

        ## [138] Improved No-Regret Algorithms for Stochastic Shortest Path with Linear MDP

        **Authors**: *Liyu Chen, Rahul Jain, Haipeng Luo*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22h.html](https://proceedings.mlr.press/v162/chen22h.html)

        **Abstract**:

        We introduce two new no-regret algorithms for the stochastic shortest path (SSP) problem with a linear MDP that significantly improve over the only existing results of (Vial et al., 2021). Our first algorithm is computationally efficient and achieves a regret bound $O(\sqrt{d^3B_{\star}^2T_{\star} K})$, where $d$ is the dimension of the feature space, $B_{\star}$ and $T_{\star}$ are upper bounds of the expected costs and hitting time of the optimal policy respectively, and $K$ is the number of episodes. The same algorithm with a slight modification also achieves logarithmic regret of order $O(\frac{d^3B_{\star}^4}{c_{\min}^2\text{\rm gap}_{\min} }\ln^5\frac{dB_{\star} K}{c_{\min}})$, where $\text{\rm gap}_{\min}$ is the minimum sub-optimality gap and $c_{\min}$ is the minimum cost over all state-action pairs. Our result is obtained by developing a simpler and improved analysis for the finite-horizon approximation of (Cohen et al., 2021) with a smaller approximation error, which might be of independent interest. On the other hand, using variance-aware confidence sets in a global optimization problem, our second algorithm is computationally inefficient but achieves the first “horizon-free” regret bound $O(d^{3.5}B_{\star}\sqrt{K})$ with no polynomial dependency on $T_{\star}$ or $1/c_{\min}$, almost matching the $\Omega(dB_{\star}\sqrt{K})$ lower bound from (Min et al., 2021).

        ----

        ## [139] Learning Infinite-horizon Average-reward Markov Decision Process with Constraints

        **Authors**: *Liyu Chen, Rahul Jain, Haipeng Luo*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22i.html](https://proceedings.mlr.press/v162/chen22i.html)

        **Abstract**:

        We study regret minimization for infinite-horizon average-reward Markov Decision Processes (MDPs) under cost constraints. We start by designing a policy optimization algorithm with carefully designed action-value estimator and bonus term, and show that for ergodic MDPs, our algorithm ensures $O(\sqrt{T})$ regret and constant constraint violation, where $T$ is the total number of time steps. This strictly improves over the algorithm of (Singh et al., 2020), whose regret and constraint violation are both $O(T^{2/3})$. Next, we consider the most general class of weakly communicating MDPs. Through a finite-horizon approximation, we develop another algorithm with $O(T^{2/3})$ regret and constraint violation, which can be further improved to $O(\sqrt{T})$ via a simple modification, albeit making the algorithm computationally inefficient. As far as we know, these are the first set of provable algorithms for weakly communicating MDPs with cost constraints.

        ----

        ## [140] Active Multi-Task Representation Learning

        **Authors**: *Yifang Chen, Kevin G. Jamieson, Simon S. Du*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22j.html](https://proceedings.mlr.press/v162/chen22j.html)

        **Abstract**:

        To leverage the power of big data from source domains and overcome the scarcity of target domain samples, representation learning based on multi-task pretraining has become a standard approach in many applications. However, large-scale pretraining is often computationally expensive and not affordable for small organizations. When there is only one target task, most source tasks can be irrelevant, and we can actively sample a subset of source data from the most To leverage the power of big data from source tasks and overcome the scarcity of the target task samples, representation learning based on multi-task pretraining has become a standard approach in many applications. However, up until now, choosing which source tasks to include in the multi-task learning has been more art than science. In this paper, we give the first formal study on resource task sampling by leveraging the techniques from active learning. We propose an algorithm that iteratively estimates the relevance of each source task to the target task and samples from each source task based on the estimated relevance. Theoretically, we show that for the linear representation class, to achieve the same error rate, our algorithm can save up to a textit{number of source tasks} factor in the source task sample complexity, compared with the naive uniform sampling from all source tasks. We also provide experiments on real-world computer vision datasets to illustrate the effectiveness of our proposed method on both linear and convolutional neural network representation classes. We believe our paper serves as an important initial step to bring techniques from active learning to representation learning.

        ----

        ## [141] On Collective Robustness of Bagging Against Data Poisoning

        **Authors**: *Ruoxin Chen, Zenan Li, Jie Li, Junchi Yan, Chentao Wu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22k.html](https://proceedings.mlr.press/v162/chen22k.html)

        **Abstract**:

        Bootstrap aggregating (bagging) is an effective ensemble protocol, which is believed can enhance robustness by its majority voting mechanism. Recent works further prove the sample-wise robustness certificates for certain forms of bagging (e.g. partition aggregation). Beyond these particular forms, in this paper, we propose the first collective certification for general bagging to compute the tight robustness against the global poisoning attack. Specifically, we compute the maximum number of simultaneously changed predictions via solving a binary integer linear programming (BILP) problem. Then we analyze the robustness of vanilla bagging and give the upper bound of the tolerable poison budget. Based on this analysis, we propose hash bagging to improve the robustness of vanilla bagging almost for free. This is achieved by modifying the random subsampling in vanilla bagging to a hash-based deterministic subsampling, as a way of controlling the influence scope for each poisoning sample universally. Our extensive experiments show the notable advantage in terms of applicability and robustness. Our code is available at https://github.com/Emiyalzn/ICML22-CRB.

        ----

        ## [142] Online Active Regression

        **Authors**: *Cheng Chen, Yi Li, Yiming Sun*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22l.html](https://proceedings.mlr.press/v162/chen22l.html)

        **Abstract**:

        Active regression considers a linear regression problem where the learner receives a large number of data points but can only observe a small number of labels. Since online algorithms can deal with incremental training data and take advantage of low computational cost, we consider an online extension of the active regression problem: the learner receives data points one by one and immediately decides whether it should collect the corresponding labels. The goal is to efficiently maintain the regression of received data points with a small budget of label queries. We propose novel algorithms for this problem under $\ell_p$ loss where $p\in[1,2]$. To achieve a $(1+\epsilon)$-approximate solution, our proposed algorithms only requires $\tilde{\mathcal{O}}(d/poly(\epsilon))$ queries of labels. The numerical results verify our theoretical results and show that our methods have comparable performance with offline active regression algorithms.

        ----

        ## [143] Selling Data To a Machine Learner: Pricing via Costly Signaling

        **Authors**: *Junjie Chen, Minming Li, Haifeng Xu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22m.html](https://proceedings.mlr.press/v162/chen22m.html)

        **Abstract**:

        We consider a new problem of selling data to a machine learner who looks to purchase data to train his machine learning model. A key challenge in this setup is that neither the seller nor the machine learner knows the true quality of data. When designing a revenue-maximizing mechanism, a data seller faces the tradeoff between the cost and precision of data quality estimation. To address this challenge, we study a natural class of mechanisms that price data via costly signaling. Motivated by the assumption of i.i.d. data points as in classic machine learning models, we first consider selling homogeneous data and derive an optimal selling mechanism. We then turn to the sale of heterogeneous data, motivated by the sale of multiple data sets, and show that 1) on the negative side, it is NP-hard to approximate the optimal mechanism within a constant ratio e/(e+1) + o(1); while 2) on the positive side, there is a 1/k-approximate algorithm, where k is the number of the machine learner’s private types.

        ----

        ## [144] ME-GAN: Learning Panoptic Electrocardio Representations for Multi-view ECG Synthesis Conditioned on Heart Diseases

        **Authors**: *Jintai Chen, Kuanlun Liao, Kun Wei, Haochao Ying, Danny Z. Chen, Jian Wu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22n.html](https://proceedings.mlr.press/v162/chen22n.html)

        **Abstract**:

        Electrocardiogram (ECG) is a widely used non-invasive diagnostic tool for heart diseases. Many studies have devised ECG analysis models (e.g., classifiers) to assist diagnosis. As an upstream task, researches have built generative models to synthesize ECG data, which are beneficial to providing training samples, privacy protection, and annotation reduction. However, previous generative methods for ECG often neither synthesized multi-view data, nor dealt with heart disease conditions. In this paper, we propose a novel disease-aware generative adversarial network for multi-view ECG synthesis called ME-GAN, which attains panoptic electrocardio representations conditioned on heart diseases and projects the representations onto multiple standard views to yield ECG signals. Since ECG manifestations of heart diseases are often localized in specific waveforms, we propose a new "mixup normalization" to inject disease information precisely into suitable locations. In addition, we propose a "view discriminator" to revert disordered ECG views into a pre-determined order, supervising the generator to obtain ECG representing correct view characteristics. Besides, a new metric, rFID, is presented to assess the quality of the synthesized ECG signals. Comprehensive experiments verify that our ME-GAN performs well on multi-view ECG signal synthesis with trusty morbid manifestations.

        ----

        ## [145] Weisfeiler-Lehman Meets Gromov-Wasserstein

        **Authors**: *Samantha Chen, Sunhyuk Lim, Facundo Mémoli, Zhengchao Wan, Yusu Wang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22o.html](https://proceedings.mlr.press/v162/chen22o.html)

        **Abstract**:

        The Weisfeiler-Lehman (WL) test is a classical procedure for graph isomorphism testing. The WL test has also been widely used both for designing graph kernels and for analyzing graph neural networks. In this paper, we propose the Weisfeiler-Lehman (WL) distance, a notion of distance between labeled measure Markov chains (LMMCs), of which labeled graphs are special cases. The WL distance is polynomial time computable and is also compatible with the WL test in the sense that the former is positive if and only if the WL test can distinguish the two involved graphs. The WL distance captures and compares subtle structures of the underlying LMMCs and, as a consequence of this, it is more discriminating than the distance between graphs used for defining the state-of-the-art Wasserstein Weisfeiler-Lehman graph kernel. Inspired by the structure of the WL distance we identify a neural network architecture on LMMCs which turns out to be universal w.r.t. continuous functions defined on the space of all LMMCs (which includes all graphs) endowed with the WL distance. Finally, the WL distance turns out to be stable w.r.t. a natural variant of the Gromov-Wasserstein (GW) distance for comparing metric Markov chains that we identify. Hence, the WL distance can also be construed as a polynomial time lower bound for the GW distance which is in general NP-hard to compute.

        ----

        ## [146] On Non-local Convergence Analysis of Deep Linear Networks

        **Authors**: *Kun Chen, Dachao Lin, Zhihua Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22p.html](https://proceedings.mlr.press/v162/chen22p.html)

        **Abstract**:

        In this paper, we study the non-local convergence properties of deep linear networks. Specifically, under the quadratic loss, we consider optimizing deep linear networks in which there is at least a layer with only one neuron. We describe the convergent point of trajectories with an arbitrary balanced starting point under gradient flow, including the paths which converge to one of the saddle points. We also show specific convergence rates of trajectories that converge to the global minimizers by stages. We conclude that the rates vary from polynomial to linear. As far as we know, our results are the first to give a non-local analysis of deep linear neural networks with arbitrary balanced initialization, rather than the lazy training regime which has dominated the literature on neural networks or the restricted benign initialization.

        ----

        ## [147] Flow-based Recurrent Belief State Learning for POMDPs

        **Authors**: *Xiaoyu Chen, Yao Mark Mu, Ping Luo, Shengbo Li, Jianyu Chen*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22q.html](https://proceedings.mlr.press/v162/chen22q.html)

        **Abstract**:

        Partially Observable Markov Decision Process (POMDP) provides a principled and generic framework to model real world sequential decision making processes but yet remains unsolved, especially for high dimensional continuous space and unknown models. The main challenge lies in how to accurately obtain the belief state, which is the probability distribution over the unobservable environment states given historical information. Accurately calculating this belief state is a precondition for obtaining an optimal policy of POMDPs. Recent advances in deep learning techniques show great potential to learn good belief states. However, existing methods can only learn approximated distribution with limited flexibility. In this paper, we introduce the \textbf{F}l\textbf{O}w-based \textbf{R}ecurrent \textbf{BE}lief \textbf{S}tate model (FORBES), which incorporates normalizing flows into the variational inference to learn general continuous belief states for POMDPs. Furthermore, we show that the learned belief states can be plugged into downstream RL algorithms to improve performance. In experiments, we show that our methods successfully capture the complex belief states that enable multi-modal predictions as well as high quality reconstructions, and results on challenging visual-motor control tasks show that our method achieves superior performance and sample efficiency.

        ----

        ## [148] Structure-Aware Transformer for Graph Representation Learning

        **Authors**: *Dexiong Chen, Leslie O'Bray, Karsten M. Borgwardt*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22r.html](https://proceedings.mlr.press/v162/chen22r.html)

        **Abstract**:

        The Transformer architecture has gained growing attention in graph representation learning recently, as it naturally overcomes several limitations of graph neural networks (GNNs) by avoiding their strict structural inductive biases and instead only encoding the graph structure via positional encoding. Here, we show that the node representations generated by the Transformer with positional encoding do not necessarily capture structural similarity between them. To address this issue, we propose the Structure-Aware Transformer, a class of simple and flexible graph Transformers built upon a new self-attention mechanism. This new self-attention incorporates structural information into the original self-attention by extracting a subgraph representation rooted at each node before computing the attention. We propose several methods for automatically generating the subgraph representation and show theoretically that the resulting representations are at least as expressive as the subgraph representations. Empirically, our method achieves state-of-the-art performance on five graph prediction benchmarks. Our structure-aware framework can leverage any existing GNN to extract the subgraph representation, and we show that it systematically improves performance relative to the base GNN model, successfully combining the advantages of GNNs and Transformers. Our code is available at https://github.com/BorgwardtLab/SAT.

        ----

        ## [149] The Poisson Binomial Mechanism for Unbiased Federated Learning with Secure Aggregation

        **Authors**: *Wei-Ning Chen, Ayfer Özgür, Peter Kairouz*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22s.html](https://proceedings.mlr.press/v162/chen22s.html)

        **Abstract**:

        We introduce the Poisson Binomial mechanism (PBM), a discrete differential privacy mechanism for distributed mean estimation (DME) with applications to federated learning and analytics. We provide a tight analysis of its privacy guarantees, showing that it achieves the same privacy-accuracy trade-offs as the continuous Gaussian mechanism. Our analysis is based on a novel bound on the Rényi divergence of two Poisson binomial distributions that may be of independent interest. Unlike previous discrete DP schemes based on additive noise, our mechanism encodes local information into a parameter of the binomial distribution, and hence the output distribution is discrete with bounded support. Moreover, the support does not increase as the privacy budget goes to zero as in the case of additive schemes which require the addition of more noise to achieve higher privacy; on the contrary, the support becomes smaller as eps goes to zero. The bounded support enables us to combine our mechanism with secure aggregation (SecAgg), a multi-party cryptographic protocol, without the need of performing modular clipping which results in an unbiased estimator of the sum of the local vectors. This in turn allows us to apply it in the private FL setting and provide an upper bound on the convergence rate of the SGD algorithm. Moreover, since the support of the output distribution becomes smaller as $\varepsilon \ra 0$, the communication cost of our scheme decreases with the privacy constraint $\varepsilon$, outperforming all previous distributed DP schemes based on additive noise in the high privacy or low communication regimes.

        ----

        ## [150] Learning Mixtures of Linear Dynamical Systems

        **Authors**: *Yanxi Chen, H. Vincent Poor*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22t.html](https://proceedings.mlr.press/v162/chen22t.html)

        **Abstract**:

        We study the problem of learning a mixture of multiple linear dynamical systems (LDSs) from unlabeled short sample trajectories, each generated by one of the LDS models. Despite the wide applicability of mixture models for time-series data, learning algorithms that come with end-to-end performance guarantees are largely absent from existing literature. There are multiple sources of technical challenges, including but not limited to (1) the presence of latent variables (i.e. the unknown labels of trajectories); (2) the possibility that the sample trajectories might have lengths much smaller than the dimension $d$ of the LDS models; and (3) the complicated temporal dependence inherent to time-series data. To tackle these challenges, we develop a two-stage meta-algorithm, which is guaranteed to efficiently recover each ground-truth LDS model up to error $\tilde{O}(\sqrt{d/T})$, where $T$ is the total sample size. We validate our theoretical studies with numerical experiments, confirming the efficacy of the proposed algorithm.

        ----

        ## [151] On Well-posedness and Minimax Optimal Rates of Nonparametric Q-function Estimation in Off-policy Evaluation

        **Authors**: *Xiaohong Chen, Zhengling Qi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22u.html](https://proceedings.mlr.press/v162/chen22u.html)

        **Abstract**:

        We study the off-policy evaluation (OPE) problem in an infinite-horizon Markov decision process with continuous states and actions. We recast the $Q$-function estimation into a special form of the nonparametric instrumental variables (NPIV) estimation problem. We first show that under one mild condition the NPIV formulation of $Q$-function estimation is well-posed in the sense of $L^2$-measure of ill-posedness with respect to the data generating distribution, bypassing a strong assumption on the discount factor $\gamma$ imposed in the recent literature for obtaining the $L^2$ convergence rates of various $Q$-function estimators. Thanks to this new well-posed property, we derive the first minimax lower bounds for the convergence rates of nonparametric estimation of $Q$-function and its derivatives in both sup-norm and $L^2$-norm, which are shown to be the same as those for the classical nonparametric regression (Stone, 1982). We then propose a sieve two-stage least squares estimator and establish its rate-optimality in both norms under some mild conditions. Our general results on the well-posedness and the minimax lower bounds are of independent interest to study not only other nonparametric estimators for $Q$-function but also efficient estimation on the value of any target policy in off-policy settings.

        ----

        ## [152] Faster Fundamental Graph Algorithms via Learned Predictions

        **Authors**: *Justin Y. Chen, Sandeep Silwal, Ali Vakilian, Fred Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22v.html](https://proceedings.mlr.press/v162/chen22v.html)

        **Abstract**:

        We consider the question of speeding up classic graph algorithms with machine-learned predictions. In this model, algorithms are furnished with extra advice learned from past or similar instances. Given the additional information, we aim to improve upon the traditional worst-case run-time guarantees. Our contributions are the following: (i) We give a faster algorithm for minimum-weight bipartite matching via learned duals, improving the recent result by Dinitz, Im, Lavastida, Moseley and Vassilvitskii (NeurIPS, 2021); (ii) We extend the learned dual approach to the single-source shortest path problem (with negative edge lengths), achieving an almost linear runtime given sufficiently accurate predictions which improves upon the classic fastest algorithm due to Goldberg (SIAM J. Comput., 1995); (iii) We provide a general reduction-based framework for learning-based graph algorithms, leading to new algorithms for degree-constrained subgraph and minimum-cost 0-1 flow, based on reductions to bipartite matching and the shortest path problem. Finally, we give a set of general learnability theorems, showing that the predictions required by our algorithms can be efficiently learned in a PAC fashion.

        ----

        ## [153] Improve Single-Point Zeroth-Order Optimization Using High-Pass and Low-Pass Filters

        **Authors**: *Xin Chen, Yujie Tang, Na Li*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22w.html](https://proceedings.mlr.press/v162/chen22w.html)

        **Abstract**:

        Single-point zeroth-order optimization (SZO) is useful in solving online black-box optimization and control problems in time-varying environments, as it queries the function value only once at each time step. However, the vanilla SZO method is known to suffer from a large estimation variance and slow convergence, which seriously limits its practical application. In this work, we borrow the idea of high-pass and low-pass filters from extremum seeking control (continuous-time version of SZO) and develop a novel SZO method called HLF-SZO by integrating these filters. It turns out that the high-pass filter coincides with the residual feedback method, and the low-pass filter can be interpreted as the momentum method. As a result, the proposed HLF-SZO achieves a much smaller variance and much faster convergence than the vanilla SZO method, and empirically outperforms the residual-feedback SZO method, which are verified via extensive numerical experiments.

        ----

        ## [154] Deep Variational Graph Convolutional Recurrent Network for Multivariate Time Series Anomaly Detection

        **Authors**: *Wenchao Chen, Long Tian, Bo Chen, Liang Dai, Zhibin Duan, Mingyuan Zhou*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22x.html](https://proceedings.mlr.press/v162/chen22x.html)

        **Abstract**:

        Anomaly detection within multivariate time series (MTS) is an essential task in both data mining and service quality management. Many recent works on anomaly detection focus on designing unsupervised probabilistic models to extract robust normal patterns of MTS. In this paper, we model sensor dependency and stochasticity within MTS by developing an embedding-guided probabilistic generative network. We combine it with adaptive variational graph convolutional recurrent network %and get variational GCRN (VGCRN) to model both spatial and temporal fine-grained correlations in MTS. To explore hierarchical latent representations, we further extend VGCRN into a deep variational network, which captures multilevel information at different layers and is robust to noisy time series. Moreover, we develop an upward-downward variational inference scheme that considers both forecasting-based and reconstruction-based losses, achieving an accurate posterior approximation of latent variables with better MTS representations. The experiments verify the superiority of the proposed method over current state-of-the-art methods.

        ----

        ## [155] Auxiliary Learning with Joint Task and Data Scheduling

        **Authors**: *Hong Chen, Xin Wang, Chaoyu Guan, Yue Liu, Wenwu Zhu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22y.html](https://proceedings.mlr.press/v162/chen22y.html)

        **Abstract**:

        Existing auxiliary learning approaches only consider the relationships between the target task and the auxiliary tasks, ignoring the fact that data samples within an auxiliary task could contribute differently to the target task, which results in inefficient auxiliary information usage and non-robustness to data noise. In this paper, we propose to learn a joint task and data schedule for auxiliary learning, which captures the importance of different data samples in each auxiliary task to the target task. However, learning such a joint schedule is challenging due to the large number of additional parameters required for the schedule. To tackle the challenge, we propose a joint task and data scheduling (JTDS) model for auxiliary learning. The JTDS model captures the joint task-data importance through a task-data scheduler, which creates a mapping from task, feature and label information to the schedule in a parameter-efficient way. Particularly, we formulate the scheduler and the task learning process as a bi-level optimization problem. In the lower optimization, the task learning model is updated with the scheduled gradient, while in the upper optimization, the task-data scheduler is updated with the implicit gradient. Experimental results show that our JTDS model significantly outperforms the state-of-the-art methods under supervised, semi-supervised and corrupted label settings.

        ----

        ## [156] Optimization-Induced Graph Implicit Nonlinear Diffusion

        **Authors**: *Qi Chen, Yifei Wang, Yisen Wang, Jiansheng Yang, Zhouchen Lin*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22z.html](https://proceedings.mlr.press/v162/chen22z.html)

        **Abstract**:

        Due to the over-smoothing issue, most existing graph neural networks can only capture limited dependencies with their inherently finite aggregation layers. To overcome this limitation, we propose a new kind of graph convolution, called Graph Implicit Nonlinear Diffusion (GIND), which implicitly has access to infinite hops of neighbors while adaptively aggregating features with nonlinear diffusion to prevent over-smoothing. Notably, we show that the learned representation can be formalized as the minimizer of an explicit convex optimization objective. With this property, we can theoretically characterize the equilibrium of our GIND from an optimization perspective. More interestingly, we can induce new structural variants by modifying the corresponding optimization objective. To be specific, we can embed prior properties to the equilibrium, as well as introducing skip connections to promote training stability. Extensive experiments show that GIND is good at capturing long-range dependencies, and performs well on both homophilic and heterophilic graphs with nonlinear diffusion. Moreover, we show that the optimization-induced variants of our models can boost the performance and improve training stability and efficiency as well. As a result, our GIND obtains significant improvements on both node-level and graph-level tasks.

        ----

        ## [157] Robust Meta-learning with Sampling Noise and Label Noise via Eigen-Reptile

        **Authors**: *Dong Chen, Lingfei Wu, Siliang Tang, Xiao Yun, Bo Long, Yueting Zhuang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22aa.html](https://proceedings.mlr.press/v162/chen22aa.html)

        **Abstract**:

        Recent years have seen a surge of interest in meta-learning techniques for tackling the few-shot learning (FSL) problem. However, the meta-learner is prone to overfitting since there are only a few available samples, which can be identified as sampling noise on a clean dataset. Besides, when handling the data with noisy labels, the meta-learner could be extremely sensitive to label noise on a corrupted dataset. To address these two challenges, we present Eigen-Reptile (ER) that updates the meta-parameters with the main direction of historical task-specific parameters. Specifically, the main direction is computed in a fast way, where the scale of the calculated matrix is related to the number of gradient steps for the specific task instead of the number of parameters. Furthermore, to obtain a more accurate main direction for Eigen-Reptile in the presence of many noisy labels, we further propose Introspective Self-paced Learning (ISPL). We have theoretically and experimentally demonstrated the soundness and effectiveness of the proposed Eigen-Reptile and ISPL. Particularly, our experiments on different tasks show that the proposed method is able to outperform or achieve highly competitive performance compared with other gradient-based methods with or without noisy labels. The code and data for the proposed method are provided for research purposes https://github.com/Anfeather/Eigen-Reptile.

        ----

        ## [158] Adaptive Model Design for Markov Decision Process

        **Authors**: *Siyu Chen, Donglin Yang, Jiayang Li, Senmiao Wang, Zhuoran Yang, Zhaoran Wang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22ab.html](https://proceedings.mlr.press/v162/chen22ab.html)

        **Abstract**:

        In a Markov decision process (MDP), an agent interacts with the environment via perceptions and actions. During this process, the agent aims to maximize its own gain. Hence, appropriate regulations are often required, if we hope to take the external costs/benefits of its actions into consideration. In this paper, we study how to regulate such an agent by redesigning model parameters that can affect the rewards and/or the transition kernels. We formulate this problem as a bilevel program, in which the lower-level MDP is regulated by the upper-level model designer. To solve the resulting problem, we develop a scheme that allows the designer to iteratively predict the agent’s reaction by solving the MDP and then adaptively update model parameters based on the predicted reaction. The algorithm is first theoretically analyzed and then empirically tested on several MDP models arising in economics and robotics.

        ----

        ## [159] State Transition of Dendritic Spines Improves Learning of Sparse Spiking Neural Networks

        **Authors**: *Yanqi Chen, Zhaofei Yu, Wei Fang, Zhengyu Ma, Tiejun Huang, Yonghong Tian*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22ac.html](https://proceedings.mlr.press/v162/chen22ac.html)

        **Abstract**:

        Spiking Neural Networks (SNNs) are considered a promising alternative to Artificial Neural Networks (ANNs) for their event-driven computing paradigm when deployed on energy-efficient neuromorphic hardware. Recently, deep SNNs have shown breathtaking performance improvement through cutting-edge training strategy and flexible structure, which also scales up the number of parameters and computational burdens in a single network. Inspired by the state transition of dendritic spines in the filopodial model of spinogenesis, we model different states of SNN weights, facilitating weight optimization for pruning. Furthermore, the pruning speed can be regulated by using different functions describing the growing threshold of state transition. We organize these techniques as a dynamic pruning algorithm based on nonlinear reparameterization mapping from spine size to SNN weights. Our approach yields sparse deep networks on the large-scale dataset (SEW ResNet18 on ImageNet) while maintaining state-of-the-art low performance loss ( 3% at 88.8% sparsity) compared to existing pruning methods on directly trained SNNs. Moreover, we find out pruning speed regulation while learning is crucial to avoiding disastrous performance degradation at the final stages of training, which may shed light on future work on SNN pruning.

        ----

        ## [160] Efficient Online ML API Selection for Multi-Label Classification Tasks

        **Authors**: *Lingjiao Chen, Matei Zaharia, James Zou*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22ad.html](https://proceedings.mlr.press/v162/chen22ad.html)

        **Abstract**:

        Multi-label classification tasks such as OCR and multi-object recognition are a major focus of the growing machine learning as a service industry. While many multi-label APIs are available, it is challenging for users to decide which API to use for their own data and budget, due to the heterogeneity in their prices and performance. Recent work has shown how to efficiently select and combine single label APIs to optimize performance and cost. However, its computation cost is exponential in the number of labels, and is not suitable for settings like OCR. In this work, we propose FrugalMCT, a principled framework that adaptively selects the APIs to use for different data in an online fashion while respecting the user’s budget. It allows combining ML APIs’ predictions for any single data point, and selects the best combination based on an accuracy estimator. We run systematic experiments using ML APIs from Google, Microsoft, Amazon, IBM, Tencent, and other providers for tasks including multi-label image classification, scene text recognition, and named entity recognition. Across these tasks, FrugalMCT can achieve over 90% cost reduction while matching the accuracy of the best single API, or up to 8% better accuracy while matching the best API’s cost.

        ----

        ## [161] Data-Efficient Double-Win Lottery Tickets from Robust Pre-training

        **Authors**: *Tianlong Chen, Zhenyu Zhang, Sijia Liu, Yang Zhang, Shiyu Chang, Zhangyang Wang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22ae.html](https://proceedings.mlr.press/v162/chen22ae.html)

        **Abstract**:

        Pre-training serves as a broadly adopted starting point for transfer learning on various downstream tasks. Recent investigations of lottery tickets hypothesis (LTH) demonstrate such enormous pre-trained models can be replaced by extremely sparse subnetworks (a.k.a. matching subnetworks) without sacrificing transferability. However, practical security-crucial applications usually pose more challenging requirements beyond standard transfer, which also demand these subnetworks to overcome adversarial vulnerability. In this paper, we formulate a more rigorous concept, Double-Win Lottery Tickets, in which a located subnetwork from a pre-trained model can be independently transferred on diverse downstream tasks, to reach BOTH the same standard and robust generalization, under BOTH standard and adversarial training regimes, as the full pre-trained model can do. We comprehensively examine various pre-training mechanisms and find that robust pre-training tends to craft sparser double-win lottery tickets with superior performance over the standard counterparts. For example, on downstream CIFAR-10/100 datasets, we identify double-win matching subnetworks with the standard, fast adversarial, and adversarial pre-training from ImageNet, at 89.26%/73.79%, 89.26%/79.03%, and 91.41%/83.22% sparsity, respectively. Furthermore, we observe the obtained double-win lottery tickets can be more data-efficient to transfer, under practical data-limited (e.g., 1% and 10%) downstream schemes. Our results show that the benefits from robust pre-training are amplified by the lottery ticket scheme, as well as the data-limited transfer setting. Codes are available at https://github.com/VITA-Group/Double-Win-LTH.

        ----

        ## [162] Linearity Grafting: Relaxed Neuron Pruning Helps Certifiable Robustness

        **Authors**: *Tianlong Chen, Huan Zhang, Zhenyu Zhang, Shiyu Chang, Sijia Liu, Pin-Yu Chen, Zhangyang Wang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22af.html](https://proceedings.mlr.press/v162/chen22af.html)

        **Abstract**:

        Certifiable robustness is a highly desirable property for adopting deep neural networks (DNNs) in safety-critical scenarios, but often demands tedious computations to establish. The main hurdle lies in the massive amount of non-linearity in large DNNs. To trade off the DNN expressiveness (which calls for more non-linearity) and robustness certification scalability (which prefers more linearity), we propose a novel solution to strategically manipulate neurons, by "grafting" appropriate levels of linearity. The core of our proposal is to first linearize insignificant ReLU neurons, to eliminate the non-linear components that are both redundant for DNN performance and harmful to its certification. We then optimize the associated slopes and intercepts of the replaced linear activations for restoring model performance while maintaining certifiability. Hence, typical neuron pruning could be viewed as a special case of grafting a linear function of the fixed zero slopes and intercept, that might overly restrict the network flexibility and sacrifice its performance. Extensive experiments on multiple datasets and network backbones show that our linearity grafting can (1) effectively tighten certified bounds; (2) achieve competitive certifiable robustness without certified robust training (i.e., over 30% improvements on CIFAR-10 models); and (3) scale up complete verification to large adversarially trained models with 17M parameters. Codes are available at https://github.com/VITA-Group/Linearity-Grafting.

        ----

        ## [163] Human-in-the-loop: Provably Efficient Preference-based Reinforcement Learning with General Function Approximation

        **Authors**: *Xiaoyu Chen, Han Zhong, Zhuoran Yang, Zhaoran Wang, Liwei Wang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22ag.html](https://proceedings.mlr.press/v162/chen22ag.html)

        **Abstract**:

        We study human-in-the-loop reinforcement learning (RL) with trajectory preferences, where instead of receiving a numeric reward at each step, the RL agent only receives preferences over trajectory pairs from a human overseer. The goal of the RL agent is to learn the optimal policy which is most preferred by the human overseer. Despite the empirical success in various real-world applications, the theoretical understanding of preference-based RL (PbRL) is only limited to the tabular case. In this paper, we propose the first optimistic model-based algorithm for PbRL with general function approximation, which estimates the model using value-targeted regression and calculates the exploratory policies by solving an optimistic planning problem. We prove that our algorithm achieves the regret bound of $\tilde{O} (\operatorname{poly}(d H) \sqrt{K} )$, where $d$ is the complexity measure of the transition and preference model depending on the Eluder dimension and log-covering numbers, $H$ is the planning horizon, $K$ is the number of episodes, and $\tilde O(\cdot)$ omits logarithmic terms. Our lower bound indicates that our algorithm is near-optimal when specialized to the linear setting. Furthermore, we extend the PbRL problem by formulating a novel problem called RL with $n$-wise comparisons, and provide the first sample-efficient algorithm for this new setting. To the best of our knowledge, this is the first theoretical result for PbRL with (general) function approximation.

        ----

        ## [164] Sample and Communication-Efficient Decentralized Actor-Critic Algorithms with Finite-Time Analysis

        **Authors**: *Ziyi Chen, Yi Zhou, Rong-Rong Chen, Shaofeng Zou*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chen22ah.html](https://proceedings.mlr.press/v162/chen22ah.html)

        **Abstract**:

        Actor-critic (AC) algorithms have been widely used in decentralized multi-agent systems to learn the optimal joint control policy. However, existing decentralized AC algorithms either need to share agents’ sensitive information or lack communication-efficiency. In this work, we develop decentralized AC and natural AC (NAC) algorithms that avoid sharing agents’ local information and are sample and communication-efficient. In both algorithms, agents share only noisy rewards and use mini-batch local policy gradient updates to ensure high sample and communication efficiency. Particularly for decentralized NAC, we develop a decentralized Markovian SGD algorithm with an adaptive mini-batch size to efficiently compute the natural policy gradient. Under Markovian sampling and linear function approximation, we prove that the proposed decentralized AC and NAC algorithms achieve the state-of-the-art sample complexities $\mathcal{O}(\epsilon^{-2}\ln\epsilon^{-1})$ and $\mathcal{O}(\epsilon^{-3}\ln\epsilon^{-1})$, respectively, and achieve an improved communication complexity $\mathcal{O}(\epsilon^{-1}\ln\epsilon^{-1})$. Numerical experiments demonstrate that the proposed algorithms achieve lower sample and communication complexities than the existing decentralized AC algorithms.

        ----

        ## [165] Task-aware Privacy Preservation for Multi-dimensional Data

        **Authors**: *Jiangnan Cheng, Ao Tang, Sandeep Chinchali*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cheng22a.html](https://proceedings.mlr.press/v162/cheng22a.html)

        **Abstract**:

        Local differential privacy (LDP) can be adopted to anonymize richer user data attributes that will be input to sophisticated machine learning (ML) tasks. However, today’s LDP approaches are largely task-agnostic and often lead to severe performance loss – they simply inject noise to all data attributes according to a given privacy budget, regardless of what features are most relevant for the ultimate task. In this paper, we address how to significantly improve the ultimate task performance with multi-dimensional user data by considering a task-aware privacy preservation problem. The key idea is to use an encoder-decoder framework to learn (and anonymize) a task-relevant latent representation of user data. We obtain an analytical near-optimal solution for the linear setting with mean-squared error (MSE) task loss. We also provide an approximate solution through a gradient-based learning algorithm for general nonlinear cases. Extensive experiments demonstrate that our task-aware approach significantly improves ultimate task accuracy compared to standard benchmark LDP approaches with the same level of privacy guarantee.

        ----

        ## [166] Adversarially Trained Actor Critic for Offline Reinforcement Learning

        **Authors**: *Ching-An Cheng, Tengyang Xie, Nan Jiang, Alekh Agarwal*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cheng22b.html](https://proceedings.mlr.press/v162/cheng22b.html)

        **Abstract**:

        We propose Adversarially Trained Actor Critic (ATAC), a new model-free algorithm for offline reinforcement learning (RL) under insufficient data coverage, based on the concept of relative pessimism. ATAC is designed as a two-player Stackelberg game framing of offline RL: A policy actor competes against an adversarially trained value critic, who finds data-consistent scenarios where the actor is inferior to the data-collection behavior policy. We prove that, when the actor attains no regret in the two-player game, running ATAC produces a policy that provably 1) outperforms the behavior policy over a wide range of hyperparameters that control the degree of pessimism, and 2) competes with the best policy covered by data with appropriately chosen hyperparameters. Compared with existing works, notably our framework offers both theoretical guarantees for general function approximation and a deep RL implementation scalable to complex environments and large datasets. In the D4RL benchmark, ATAC consistently outperforms state-of-the-art offline RL algorithms on a range of continuous control tasks.

        ----

        ## [167] Quantum-Inspired Algorithms from Randomized Numerical Linear Algebra

        **Authors**: *Nadiia Chepurko, Kenneth L. Clarkson, Lior Horesh, Honghao Lin, David P. Woodruff*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chepurko22a.html](https://proceedings.mlr.press/v162/chepurko22a.html)

        **Abstract**:

        We create classical (non-quantum) dynamic data structures supporting queries for recommender systems and least-squares regression that are comparable to their quantum analogues. De-quantizing such algorithms has received a flurry of attention in recent years; we obtain sharper bounds for these problems. More significantly, we achieve these improvements by arguing that the previous quantum-inspired algorithms for these problems are doing leverage or ridge-leverage score sampling in disguise; these are powerful and standard techniques in randomized numerical linear algebra. With this recognition, we are able to employ the large body of work in numerical linear algebra to obtain algorithms for these problems that are simpler or faster (or both) than existing approaches. Our experiments demonstrate that the proposed data structures also work well on real-world datasets.

        ----

        ## [168] RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets and Random Forests

        **Authors**: *Victor Chernozhukov, Whitney Newey, Victor Quintas-Martinez, Vasilis Syrgkanis*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chernozhukov22a.html](https://proceedings.mlr.press/v162/chernozhukov22a.html)

        **Abstract**:

        Many causal and policy effects of interest are defined by linear functionals of high-dimensional or non-parametric regression functions. $\sqrt{n}$-consistent and asymptotically normal estimation of the object of interest requires debiasing to reduce the effects of regularization and/or model selection on the object of interest. Debiasing is typically achieved by adding a correction term to the plug-in estimator of the functional, which leads to properties such as semi-parametric efficiency, double robustness, and Neyman orthogonality. We implement an automatic debiasing procedure based on automatically learning the Riesz representation of the linear functional using Neural Nets and Random Forests. Our method only relies on black-box evaluation oracle access to the linear functional and does not require knowledge of its analytic form. We propose a multitasking Neural Net debiasing method with stochastic gradient descent minimization of a combined Riesz representer and regression loss, while sharing representation layers for the two functions. We also propose a Random Forest method which learns a locally linear representation of the Riesz function. Even though our method applies to arbitrary functionals, we experimentally find that it performs well compared to the state of art neural net based algorithm of Shi et al. (2019) for the case of the average treatment effect functional. We also evaluate our method on the problem of estimating average marginal effects with continuous treatments, using semi-synthetic data of gasoline price changes on gasoline demand.

        ----

        ## [169] Self-supervised learning with random-projection quantizer for speech recognition

        **Authors**: *Chung-Cheng Chiu, James Qin, Yu Zhang, Jiahui Yu, Yonghui Wu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chiu22a.html](https://proceedings.mlr.press/v162/chiu22a.html)

        **Abstract**:

        We present a simple and effective self-supervised learning approach for speech recognition. The approach learns a model to predict the masked speech signals, in the form of discrete labels generated with a random-projection quantizer. In particular the quantizer projects speech inputs with a randomly initialized matrix, and does a nearest-neighbor lookup in a randomly-initialized codebook. Neither the matrix nor the codebook are updated during self-supervised learning. Since the random-projection quantizer is not trained and is separated from the speech recognition model, the design makes the approach flexible and is compatible with universal speech recognition architecture. On LibriSpeech our approach achieves similar word-error-rates as previous work using self-supervised learning with non-streaming models, and provides lower word-error-rates than previous work with streaming models. On multilingual tasks the approach also provides significant improvement over wav2vec 2.0 and w2v-BERT.

        ----

        ## [170] Discrete Probabilistic Inverse Optimal Transport

        **Authors**: *Wei-Ting Chiu, Pei Wang, Patrick Shafto*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chiu22b.html](https://proceedings.mlr.press/v162/chiu22b.html)

        **Abstract**:

        Inverse Optimal Transport (IOT) studies the problem of inferring the underlying cost that gives rise to an observation on coupling two probability measures. Couplings appear as the outcome of matching sets (e.g. dating) and moving distributions (e.g. transportation). Compared to Optimal transport (OT), the mathematical theory of IOT is undeveloped. We formalize and systematically analyze the properties of IOT using tools from the study of entropy-regularized OT. Theoretical contributions include characterization of the manifold of cross-ratio equivalent costs, the implications of model priors, and derivation of an MCMC sampler. Empirical contributions include visualizations of cross-ratio equivalent effect on basic examples, simulations validating theoretical results and experiments on real world data.

        ----

        ## [171] Selective Network Linearization for Efficient Private Inference

        **Authors**: *Minsu Cho, Ameya Joshi, Brandon Reagen, Siddharth Garg, Chinmay Hegde*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cho22a.html](https://proceedings.mlr.press/v162/cho22a.html)

        **Abstract**:

        Private inference (PI) enables inferences directly on cryptographically secure data. While promising to address many privacy issues, it has seen limited use due to extreme runtimes. Unlike plaintext inference, where latency is dominated by FLOPs, in PI non-linear functions (namely ReLU) are the bottleneck. Thus, practical PI demands novel ReLU-aware optimizations. To reduce PI latency we propose a gradient-based algorithm that selectively linearizes ReLUs while maintaining prediction accuracy. We evaluate our algorithm on several standard PI benchmarks. The results demonstrate up to $4.25%$ more accuracy (iso-ReLU count at 50K) or $2.2\times$ less latency (iso-accuracy at 70%) than the current state of the art and advance the Pareto frontier across the latency-accuracy space. To complement empirical results, we present a “no free lunch" theorem that sheds light on how and when network linearization is possible while maintaining prediction accuracy.

        ----

        ## [172] From block-Toeplitz matrices to differential equations on graphs: towards a general theory for scalable masked Transformers

        **Authors**: *Krzysztof Choromanski, Han Lin, Haoxian Chen, Tianyi Zhang, Arijit Sehanobish, Valerii Likhosherstov, Jack Parker-Holder, Tamás Sarlós, Adrian Weller, Thomas Weingarten*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/choromanski22a.html](https://proceedings.mlr.press/v162/choromanski22a.html)

        **Abstract**:

        In this paper we provide, to the best of our knowledge, the first comprehensive approach for incorporating various masking mechanisms into Transformers architectures in a scalable way. We show that recent results on linear causal attention (Choromanski et al., 2021) and log-linear RPE-attention (Luo et al., 2021) are special cases of this general mechanism. However by casting the problem as a topological (graph-based) modulation of unmasked attention, we obtain several results unknown before, including efficient d-dimensional RPE-masking and graph-kernel masking. We leverage many mathematical techniques ranging from spectral analysis through dynamic programming and random walks to new algorithms for solving Markov processes on graphs. We provide a corresponding empirical evaluation.

        ----

        ## [173] Shuffle Private Linear Contextual Bandits

        **Authors**: *Sayak Ray Chowdhury, Xingyu Zhou*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chowdhury22a.html](https://proceedings.mlr.press/v162/chowdhury22a.html)

        **Abstract**:

        Differential privacy (DP) has been recently introduced to linear contextual bandits to formally address the privacy concerns in its associated personalized services to participating users (e.g., recommendations). Prior work largely focus on two trust models of DP – the central model, where a central server is responsible for protecting users’ sensitive data, and the (stronger) local model, where information needs to be protected directly on users’ side. However, there remains a fundamental gap in the utility achieved by learning algorithms under these two privacy models, e.g., if all users are unique within a learning horizon $T$, $\widetilde{O}(\sqrt{T})$ regret in the central model as compared to $\widetilde{O}(T^{3/4})$ regret in the local model. In this work, we aim to achieve a stronger model of trust than the central model, while suffering a smaller regret than the local model by considering recently popular shuffle model of privacy. We propose a general algorithmic framework for linear contextual bandits under the shuffle trust model, where there exists a trusted shuffler – in between users and the central server– that randomly permutes a batch of users data before sending those to the server. We then instantiate this framework with two specific shuffle protocols – one relying on privacy amplification of local mechanisms, and another incorporating a protocol for summing vectors and matrices of bounded norms. We prove that both these instantiations lead to regret guarantees that significantly improve on that of the local model, and can potentially be of the order $\widetilde{O}(T^{3/5})$ if all users are unique. We also verify this regret behavior with simulations on synthetic data. Finally, under the practical scenario of non-unique users, we show that the regret of our shuffle private algorithm scale as $\widetilde{O}(T^{2/3})$, which matches what the central model could achieve in this case.

        ----

        ## [174] DNA: Domain Generalization with Diversified Neural Averaging

        **Authors**: *Xu Chu, Yujie Jin, Wenwu Zhu, Yasha Wang, Xin Wang, Shanghang Zhang, Hong Mei*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chu22a.html](https://proceedings.mlr.press/v162/chu22a.html)

        **Abstract**:

        The inaccessibility of the target domain data causes domain generalization (DG) methods prone to forget target discriminative features, and challenges the pervasive theme in existing literature in pursuing a single classifier with an ideal joint risk. In contrast, this paper investigates model misspecification and attempts to bridge DG with classifier ensemble theoretically and methodologically. By introducing a pruned Jensen-Shannon (PJS) loss, we show that the target square-root risk w.r.t. the PJS loss of the $\rho$-ensemble (the averaged classifier weighted by a quasi-posterior $\rho$) is bounded by the averaged source square-root risk of the Gibbs classifiers. We derive a tighter bound by enforcing a positive principled diversity measure of the classifiers. We give a PAC-Bayes upper bound on the target square-root risk of the $\rho$-ensemble. Methodologically, we propose a diversified neural averaging (DNA) method for DG, which optimizes the proposed PAC-Bayes bound approximately. The DNA method samples Gibbs classifiers transversely and longitudinally by simultaneously considering the dropout variational family and optimization trajectory. The $\rho$-ensemble is approximated by averaging the longitudinal weights in a single run with dropout shut down, ensuring a fast ensemble with low computational overhead. Empirically, the proposed DNA method achieves the state-of-the-art classification performance on standard DG benchmark datasets.

        ----

        ## [175] TPC: Transformation-Specific Smoothing for Point Cloud Models

        **Authors**: *Wenda Chu, Linyi Li, Bo Li*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/chu22b.html](https://proceedings.mlr.press/v162/chu22b.html)

        **Abstract**:

        Point cloud models with neural network architectures have achieved great success and been widely used in safety-critical applications, such as Lidar-based recognition systems in autonomous vehicles. However, such models are shown vulnerable against adversarial attacks which aim to apply stealthy semantic transformations such as rotation and tapering to mislead model predictions. In this paper, we propose a transformation-specific smoothing framework TPC, which provides tight and scalable robustness guarantees for point cloud models against semantic transformation attacks. We first categorize common 3D transformations into two categories: composable (e.g., rotation) and indirectly composable (e.g., tapering), and we present generic robustness certification strategies for both categories. We then specify unique certification protocols for a range of specific semantic transformations and derive strong robustness guarantees. Extensive experiments on several common 3D transformations show that TPC significantly outperforms the state of the art. For example, our framework boosts the certified accuracy against twisting transformation along z-axis (within $\pm$20{\textdegree}) from 20.3% to 83.8%. Codes and models are available at https://github.com/Qianhewu/Point-Cloud-Smoothing.

        ----

        ## [176] Unified Scaling Laws for Routed Language Models

        **Authors**: *Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake A. Hechtman, Trevor Cai, Sebastian Borgeaud, George van den Driessche, Eliza Rutherford, Tom Hennigan, Matthew J. Johnson, Albin Cassirer, Chris Jones, Elena Buchatskaya, David Budden, Laurent Sifre, Simon Osindero, Oriol Vinyals, Marc'Aurelio Ranzato, Jack W. Rae, Erich Elsen, Koray Kavukcuoglu, Karen Simonyan*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/clark22a.html](https://proceedings.mlr.press/v162/clark22a.html)

        **Abstract**:

        The performance of a language model has been shown to be effectively modeled as a power-law in its parameter count. Here we study the scaling behaviors of Routing Networks: architectures that conditionally use only a subset of their parameters while processing an input. For these models, parameter count and computational requirement form two independent axes along which an increase leads to better performance. In this work we derive and justify scaling laws defined on these two variables which generalize those known for standard language models and describe the performance of a wide range of routing architectures trained via three different techniques. Afterwards we provide two applications of these laws: first deriving an Effective Parameter Count along which all models scale at the same rate, and then using the scaling coefficients to give a quantitative comparison of the three routing techniques considered. Our analysis derives from an extensive evaluation of Routing Networks across five orders of magnitude of size, including models with hundreds of experts and hundreds of billions of parameters.

        ----

        ## [177] Context-Aware Drift Detection

        **Authors**: *Oliver Cobb, Arnaud Van Looveren*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cobb22a.html](https://proceedings.mlr.press/v162/cobb22a.html)

        **Abstract**:

        When monitoring machine learning systems, two-sample tests of homogeneity form the foundation upon which existing approaches to drift detection build. They are used to test for evidence that the distribution underlying recent deployment data differs from that underlying the historical reference data. Often, however, various factors such as time-induced correlation mean that batches of recent deployment data are not expected to form an i.i.d. sample from the historical data distribution. Instead we may wish to test for differences in the distributions conditional on context that is permitted to change. To facilitate this we borrow machinery from the causal inference domain to develop a more general drift detection framework built upon a foundation of two-sample tests for conditional distributional treatment effects. We recommend a particular instantiation of the framework based on maximum conditional mean discrepancies. We then provide an empirical study demonstrating its effectiveness for various drift detection problems of practical interest, such as detecting drift in the distributions underlying subpopulations of data in a manner that is insensitive to their respective prevalences. The study additionally demonstrates applicability to ImageNet-scale vision problems.

        ----

        ## [178] On the Robustness of CountSketch to Adaptive Inputs

        **Authors**: *Edith Cohen, Xin Lyu, Jelani Nelson, Tamás Sarlós, Moshe Shechner, Uri Stemmer*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cohen22a.html](https://proceedings.mlr.press/v162/cohen22a.html)

        **Abstract**:

        The last decade saw impressive progress towards understanding the performance of algorithms in adaptive settings, where subsequent inputs may depend on the output from prior inputs. Adaptive settings arise in processes with feedback or with adversarial attacks. Existing designs of robust algorithms are generic wrappers of non-robust counterparts and leave open the possibility of better tailored designs. The lowers bounds (attacks) are similarly worst-case and their significance to practical setting is unclear. Aiming to understand these questions, we study the robustness of \texttt{CountSketch}, a popular dimensionality reduction technique that maps vectors to a lower dimension using randomized linear measurements. The sketch supports recovering $\ell_2$-heavy hitters of a vector (entries with $v[i]^2 \geq \frac{1}{k}\|\boldsymbol{v}\|^2_2$). We show that the classic estimator is not robust, and can be attacked with a number of queries of the order of the sketch size. We propose a robust estimator (for a slightly modified sketch) that allows for quadratic number of queries in the sketch size, which is an improvement factor of $\sqrt{k}$ (for $k$ heavy hitters) over prior "blackbox" approaches.

        ----

        ## [179] Diffusion bridges vector quantized variational autoencoders

        **Authors**: *Max Cohen, Guillaume Quispe, Sylvain Le Corff, Charles Ollion, Eric Moulines*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cohen22b.html](https://proceedings.mlr.press/v162/cohen22b.html)

        **Abstract**:

        Vector Quantized-Variational AutoEncoders (VQ-VAE) are generative models based on discrete latent representations of the data, where inputs are mapped to a finite set of learned embeddings. To generate new samples, an autoregressive prior distribution over the discrete states must be trained separately. This prior is generally very complex and leads to slow generation. In this work, we propose a new model to train the prior and the encoder/decoder networks simultaneously. We build a diffusion bridge between a continuous coded vector and a non-informative prior distribution. The latent discrete states are then given as random functions of these continuous vectors. We show that our model is competitive with the autoregressive prior on the mini-Imagenet and CIFAR dataset and is efficient in both optimization and sampling. Our framework also extends the standard VQ-VAE and enables end-to-end training.

        ----

        ## [180] Online and Consistent Correlation Clustering

        **Authors**: *Vincent Cohen-Addad, Silvio Lattanzi, Andreas Maggiori, Nikos Parotsidis*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cohen-addad22a.html](https://proceedings.mlr.press/v162/cohen-addad22a.html)

        **Abstract**:

        In the correlation clustering problem the input is a signed graph where the sign indicates whether each pair of points should be placed in the same cluster or not. The goal of the problem is to compute a clustering which minimizes the number of disagreements with such recommendation. Thanks to its many practical applications, correlation clustering is a fundamental unsupervised learning problem and has been extensively studied in many different settings. In this paper we study the problem in the classic online setting with recourse; The vertices of the graphs arrive in an online manner and the goal is to maintain an approximate clustering while minimizing the number of times each vertex changes cluster. Our main contribution is an algorithm that achieves logarithmic recourse per vertex in the worst case. We also complement this result with a tight lower bound. Finally we show experimentally that our algorithm achieves better performances than state-of-the-art algorithms on real world data.

        ----

        ## [181] Massively Parallel k-Means Clustering for Perturbation Resilient Instances

        **Authors**: *Vincent Cohen-Addad, Vahab S. Mirrokni, Peilin Zhong*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cohen-addad22b.html](https://proceedings.mlr.press/v162/cohen-addad22b.html)

        **Abstract**:

        We consider $k$-means clustering of $n$ data points in Euclidean space in the Massively Parallel Computation (MPC) model, a computational model which is an abstraction of modern massively parallel computing system such as MapReduce. Recent work provides evidence that getting $O(1)$-approximate $k$-means solution for general input points using $o(\log n)$ rounds in the MPC model may be impossible under certain conditions [Ghaffari, Kuhn \& Uitto’2019]. However, the real-world data points usually have better structures. One instance of interest is the set of data points which is perturbation resilient [Bilu \& Linial’2010]. In particular, a point set is $\alpha$-perturbation resilient for $k$-means if perturbing pairwise distances by multiplicative factors in the range $[1,\alpha]$ does not change the optimum $k$-means clusters. We bypass the worst case lower bound by considering the perturbation resilient input points and showing $o(\log n)$ rounds $k$-means clustering algorithms for these instances in the MPC model. Specifically, we show a fully scalable $(1+\varepsilon)$-approximate $k$-means clustering algorithm for $O(\alpha)$-perturbation resilient instance in the MPC model using $O(1)$ rounds and ${O}_{\varepsilon,d}(n^{1+1/\alpha^2+o(1)})$ total space. If the space per machine is sufficiently larger than $k$, i.e., at least $k\cdot n^{\Omega(1)}$, we also develop an optimal $k$-means clustering algorithm for $O(\alpha)$-perturbation resilient instance in MPC using $O(1)$ rounds and ${O}_d(n^{1+o(1)}\cdot(n^{1/\alpha^2}+k))$ total space.

        ----

        ## [182] One-Pass Diversified Sampling with Application to Terabyte-Scale Genomic Sequence Streams

        **Authors**: *Benjamin Coleman, Benito Geordie, Li Chou, Ryan A. Leo Elworth, Todd J. Treangen, Anshumali Shrivastava*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/coleman22a.html](https://proceedings.mlr.press/v162/coleman22a.html)

        **Abstract**:

        A popular approach to reduce the size of a massive dataset is to apply efficient online sampling to the stream of data as it is read or generated. Online sampling routines are currently restricted to variations of reservoir sampling, where each sample is selected uniformly and independently of other samples. This renders them unsuitable for large-scale applications in computational biology, such as metagenomic community profiling and protein function annotation, which suffer from severe class imbalance. To maintain a representative and diverse sample, we must identify and preferentially select data that are likely to belong to rare classes. We argue that existing schemes for diversity sampling have prohibitive overhead for large-scale problems and high-throughput streams. We propose an efficient sampling routine that uses an online representation of the data distribution as a prefilter to retain elements from rare groups. We apply this method to several genomic data analysis tasks and demonstrate significant speedup in downstream analysis without sacrificing the quality of the results. Because our algorithm is 2x faster and uses 1000x less memory than coreset, reservoir and sketch-based alternatives, we anticipate that it will become a useful preprocessing step for applications with large-scale streaming data.

        ----

        ## [183] Transfer and Marginalize: Explaining Away Label Noise with Privileged Information

        **Authors**: *Mark Collier, Rodolphe Jenatton, Effrosyni Kokiopoulou, Jesse Berent*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/collier22a.html](https://proceedings.mlr.press/v162/collier22a.html)

        **Abstract**:

        Supervised learning datasets often have privileged information, in the form of features which are available at training time but are not available at test time e.g. the ID of the annotator that provided the label. We argue that privileged information is useful for explaining away label noise, thereby reducing the harmful impact of noisy labels. We develop a simple and efficient method for supervised learning with neural networks: it transfers via weight sharing the knowledge learned with privileged information and approximately marginalizes over privileged information at test time. Our method, TRAM (TRansfer and Marginalize), has minimal training time overhead and has the same test-time cost as not using privileged information. TRAM performs strongly on CIFAR-10H, ImageNet and Civil Comments benchmarks.

        ----

        ## [184] MAML and ANIL Provably Learn Representations

        **Authors**: *Liam Collins, Aryan Mokhtari, Sewoong Oh, Sanjay Shakkottai*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/collins22a.html](https://proceedings.mlr.press/v162/collins22a.html)

        **Abstract**:

        Recent empirical evidence has driven conventional wisdom to believe that gradient-based meta-learning (GBML) methods perform well at few-shot learning because they learn an expressive data representation that is shared across tasks. However, the mechanics of GBML have remained largely mysterious from a theoretical perspective. In this paper, we prove that two well-known GBML methods, MAML and ANIL, as well as their first-order approximations, are capable of learning common representation among a set of given tasks. Specifically, in the well-known multi-task linear representation learning setting, they are able to recover the ground-truth representation at an exponentially fast rate. Moreover, our analysis illuminates that the driving force causing MAML and ANIL to recover the underlying representation is that they adapt the final layer of their model, which harnesses the underlying task diversity to improve the representation in all directions of interest. To the best of our knowledge, these are the first results to show that MAML and/or ANIL learn expressive representations and to rigorously explain why they do so.

        ----

        ## [185] Entropic Causal Inference: Graph Identifiability

        **Authors**: *Spencer Compton, Kristjan H. Greenewald, Dmitriy A. Katz, Murat Kocaoglu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/compton22a.html](https://proceedings.mlr.press/v162/compton22a.html)

        **Abstract**:

        Entropic causal inference is a recent framework for learning the causal graph between two variables from observational data by finding the information-theoretically simplest structural explanation of the data, i.e., the model with smallest entropy. In our work, we first extend the causal graph identifiability result in the two-variable setting under relaxed assumptions. We then show the first identifiability result using the entropic approach for learning causal graphs with more than two nodes. Our approach utilizes the property that ancestrality between a source node and its descendants can be determined using the bivariate entropic tests. We provide a sound sequential peeling algorithm for general graphs that relies on this property. We also propose a heuristic algorithm for small graphs that shows strong empirical performance. We rigorously evaluate the performance of our algorithms on synthetic data generated from a variety of models, observing improvement over prior work. Finally we test our algorithms on real-world datasets.

        ----

        ## [186] Mitigating Gender Bias in Face Recognition using the von Mises-Fisher Mixture Model

        **Authors**: *Jean-Rémy Conti, Nathan Noiry, Stéphan Clémençon, Vincent Despiegel, Stéphane Gentric*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/conti22a.html](https://proceedings.mlr.press/v162/conti22a.html)

        **Abstract**:

        In spite of the high performance and reliability of deep learning algorithms in a wide range of everyday applications, many investigations tend to show that a lot of models exhibit biases, discriminating against specific subgroups of the population (e.g. gender, ethnicity). This urges the practitioner to develop fair systems with a uniform/comparable performance across sensitive groups. In this work, we investigate the gender bias of deep Face Recognition networks. In order to measure this bias, we introduce two new metrics, BFAR and BFRR, that better reflect the inherent deployment needs of Face Recognition systems. Motivated by geometric considerations, we mitigate gender bias through a new post-processing methodology which transforms the deep embeddings of a pre-trained model to give more representation power to discriminated subgroups. It consists in training a shallow neural network by minimizing a Fair von Mises-Fisher loss whose hyperparameters account for the intra-class variance of each gender. Interestingly, we empirically observe that these hyperparameters are correlated with our fairness metrics. In fact, extensive numerical experiments on a variety of datasets show that a careful selection significantly reduces gender bias.

        ----

        ## [187] Counterfactual Transportability: A Formal Approach

        **Authors**: *Juan D. Correa, Sanghack Lee, Elias Bareinboim*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/correa22a.html](https://proceedings.mlr.press/v162/correa22a.html)

        **Abstract**:

        Generalizing causal knowledge across environments is a common challenge shared across many of the data-driven disciplines, including AI and ML. Experiments are usually performed in one environment (e.g., in a lab, on Earth, in a training ground), almost invariably, with the intent of being used elsewhere (e.g., outside the lab, on Mars, in the real world), in an environment that is related but somewhat different than the original one, where certain conditions and mechanisms are likely to change. This generalization task has been studied in the causal inference literature under the rubric of transportability (Pearl and Bareinboim, 2011). While most transportability works focused on generalizing associational and interventional distributions, the generalization of counterfactual distributions has not been formally studied. In this paper, we investigate the transportability of counterfactuals from an arbitrary combination of observational and experimental distributions coming from disparate domains. Specifically, we introduce a sufficient and necessary graphical condition and develop an efficient, sound, and complete algorithm for transporting counterfactual quantities across domains in nonparametric settings. Failure of the algorithm implies the impossibility of generalizing the target counterfactual from the available data without further assumptions.

        ----

        ## [188] Label-Free Explainability for Unsupervised Models

        **Authors**: *Jonathan Crabbé, Mihaela van der Schaar*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/crabbe22a.html](https://proceedings.mlr.press/v162/crabbe22a.html)

        **Abstract**:

        Unsupervised black-box models are challenging to interpret. Indeed, most existing explainability methods require labels to select which component(s) of the black-box’s output to interpret. In the absence of labels, black-box outputs often are representation vectors whose components do not correspond to any meaningful quantity. Hence, choosing which component(s) to interpret in a label-free unsupervised/self-supervised setting is an important, yet unsolved problem. To bridge this gap in the literature, we introduce two crucial extensions of post-hoc explanation techniques: (1) label-free feature importance and (2) label-free example importance that respectively highlight influential features and training examples for a black-box to construct representations at inference time. We demonstrate that our extensions can be successfully implemented as simple wrappers around many existing feature and example importance methods. We illustrate the utility of our label-free explainability paradigm through a qualitative and quantitative comparison of representation spaces learned by various autoencoders trained on distinct unsupervised tasks.

        ----

        ## [189] Evaluating the Adversarial Robustness of Adaptive Test-time Defenses

        **Authors**: *Francesco Croce, Sven Gowal, Thomas Brunner, Evan Shelhamer, Matthias Hein, A. Taylan Cemgil*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/croce22a.html](https://proceedings.mlr.press/v162/croce22a.html)

        **Abstract**:

        Adaptive defenses, which optimize at test time, promise to improve adversarial robustness. We categorize such adaptive test-time defenses, explain their potential benefits and drawbacks, and evaluate a representative variety of the latest adaptive defenses for image classification. Unfortunately, none significantly improve upon static defenses when subjected to our careful case study evaluation. Some even weaken the underlying static model while simultaneously increasing inference computation. While these results are disappointing, we still believe that adaptive test-time defenses are a promising avenue of research and, as such, we provide recommendations for their thorough evaluation. We extend the checklist of Carlini et al. (2019) by providing concrete steps specific to adaptive defenses.

        ----

        ## [190] Adversarial Robustness against Multiple and Single lp-Threat Models via Quick Fine-Tuning of Robust Classifiers

        **Authors**: *Francesco Croce, Matthias Hein*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/croce22b.html](https://proceedings.mlr.press/v162/croce22b.html)

        **Abstract**:

        A major drawback of adversarially robust models, in particular for large scale datasets like ImageNet, is the extremely long training time compared to standard models. Moreover, models should be robust not only to one $l_p$-threat model but ideally to all of them. In this paper we propose Extreme norm Adversarial Training (E-AT) for multiple-norm robustness which is based on geometric properties of $l_p$-balls. E-AT costs up to three times less than other adversarial training methods for multiple-norm robustness. Using E-AT we show that for ImageNet a single epoch and for CIFAR-10 three epochs are sufficient to turn any $l_p$-robust model into a multiple-norm robust model. In this way we get the first multiple-norm robust model for ImageNet and boost the state-of-the-art for multiple-norm robustness to more than $51%$ on CIFAR-10. Finally, we study the general transfer via fine-tuning of adversarial robustness between different individual $l_p$-threat models and improve the previous SOTA $l_1$-robustness on both CIFAR-10 and ImageNet. Extensive experiments show that our scheme works across datasets and architectures including vision transformers.

        ----

        ## [191] Self-conditioning Pre-Trained Language Models

        **Authors**: *Xavier Suau Cuadros, Luca Zappella, Nicholas Apostoloff*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cuadros22a.html](https://proceedings.mlr.press/v162/cuadros22a.html)

        **Abstract**:

        In this paper we aim to investigate the mechanisms that guide text generation with pre-trained Transformer-based Language Models (TLMs). Grounded on the Product of Experts formulation by Hinton (1999), we describe a generative mechanism that exploits expert units which naturally exist in TLMs. Such units are responsible for detecting concepts in the input and conditioning text generation on such concepts. We describe how to identify expert units and how to activate them during inference in order to induce any desired concept in the generated output. We find that the activation of a surprisingly small amount of units is sufficient to steer text generation (as little as 3 units in a model with 345M parameters). While the objective of this work is to learn more about how TLMs work, we show that our method is effective for conditioning without fine-tuning or using extra parameters, even on fine-grained homograph concepts. Additionally, we show that our method can be used to correct gender bias present in the output of TLMs and achieves gender parity for all evaluated contexts. We compare our method with FUDGE and PPLM-BoW, and show that our approach is able to achieve gender parity at a lower perplexity and better Self-BLEU score. The proposed method is accessible to a wide audience thanks to its simplicity and minimal compute needs. The findings in this paper are a step forward in understanding the generative mechanisms of TLMs.

        ----

        ## [192] Only tails matter: Average-Case Universality and Robustness in the Convex Regime

        **Authors**: *Leonardo Cunha, Gauthier Gidel, Fabian Pedregosa, Damien Scieur, Courtney Paquette*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cunha22a.html](https://proceedings.mlr.press/v162/cunha22a.html)

        **Abstract**:

        The recently developed average-case analysis of optimization methods allows a more fine-grained and representative convergence analysis than usual worst-case results. In exchange, this analysis requires a more precise hypothesis over the data generating process, namely assuming knowledge of the expected spectral distribution (ESD) of the random matrix associated with the problem. This work shows that the concentration of eigenvalues near the edges of the ESD determines a problem’s asymptotic average complexity. This a priori information on this concentration is a more grounded assumption than complete knowledge of the ESD. This approximate concentration is effectively a middle ground between the coarseness of the worst-case scenario convergence and the restrictive previous average-case analysis. We also introduce the Generalized Chebyshev method, asymptotically optimal under a hypothesis on this concentration and globally optimal when the ESD follows a Beta distribution. We compare its performance to classical optimization algorithms, such as gradient descent or Nesterov’s scheme, and we show that, in the average-case context, Nesterov’s method is universally nearly optimal asymptotically.

        ----

        ## [193] Principal Component Flows

        **Authors**: *Edmond Cunningham, Adam D. Cobb, Susmit Jha*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/cunningham22a.html](https://proceedings.mlr.press/v162/cunningham22a.html)

        **Abstract**:

        Normalizing flows map an independent set of latent variables to their samples using a bijective transformation. Despite the exact correspondence between samples and latent variables, their high level relationship is not well understood. In this paper we characterize the geometric structure of flows using principal manifolds and understand the relationship between latent variables and samples using contours. We introduce a novel class of normalizing flows, called principal component flows (PCF), whose contours are its principal manifolds, and a variant for injective flows (iPCF) that is more efficient to train than regular injective flows. PCFs can be constructed using any flow architecture, are trained with a regularized maximum likelihood objective and can perform density estimation on all of their principal manifolds. In our experiments we show that PCFs and iPCFs are able to learn the principal manifolds over a variety of datasets. Additionally, we show that PCFs can perform density estimation on data that lie on a manifold with variable dimensionality, which is not possible with existing normalizing flows.

        ----

        ## [194] Deep symbolic regression for recurrence prediction

        **Authors**: *Stéphane d'Ascoli, Pierre-Alexandre Kamienny, Guillaume Lample, François Charton*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/d-ascoli22a.html](https://proceedings.mlr.press/v162/d-ascoli22a.html)

        **Abstract**:

        Symbolic regression, i.e. predicting a function from the observation of its values, is well-known to be a challenging task. In this paper, we train Transformers to infer the function or recurrence relation underlying sequences of integers or floats, a typical task in human IQ tests which has hardly been tackled in the machine learning literature. We evaluate our integer model on a subset of OEIS sequences, and show that it outperforms built-in Mathematica functions for recurrence prediction. We also demonstrate that our float model is able to yield informative approximations of out-of-vocabulary functions and constants, e.g. $\operatorname{bessel0}(x)\approx \frac{\sin(x)+\cos(x)}{\sqrt{\pi x}}$ and $1.644934\approx \pi^2/6$.

        ----

        ## [195] Continuous Control with Action Quantization from Demonstrations

        **Authors**: *Robert Dadashi, Léonard Hussenot, Damien Vincent, Sertan Girgin, Anton Raichuk, Matthieu Geist, Olivier Pietquin*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dadashi22a.html](https://proceedings.mlr.press/v162/dadashi22a.html)

        **Abstract**:

        In this paper, we propose a novel Reinforcement Learning (RL) framework for problems with continuous action spaces: Action Quantization from Demonstrations (AQuaDem). The proposed approach consists in learning a discretization of continuous action spaces from human demonstrations. This discretization returns a set of plausible actions (in light of the demonstrations) for each input state, thus capturing the priors of the demonstrator and their multimodal behavior. By discretizing the action space, any discrete action deep RL technique can be readily applied to the continuous control problem. Experiments show that the proposed approach outperforms state-of-the-art methods such as SAC in the RL setup, and GAIL in the Imitation Learning setup. We provide a website with interactive videos: https://google-research.github.io/aquadem/ and make the code available: https://github.com/google-research/google-research/tree/master/aquadem.

        ----

        ## [196] Dialog Inpainting: Turning Documents into Dialogs

        **Authors**: *Zhuyun Dai, Arun Tejasvi Chaganty, Vincent Y. Zhao, Aida Amini, Qazi Mamunur Rashid, Mike Green, Kelvin Guu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dai22a.html](https://proceedings.mlr.press/v162/dai22a.html)

        **Abstract**:

        Many important questions (e.g. "How to eat healthier?") require conversation to establish context and explore in depth. However, conversational question answering (ConvQA) systems have long been stymied by scarce training data that is expensive to collect. To address this problem, we propose a new technique for synthetically generating diverse and high-quality dialog data: dialog inpainting. Our approach takes the text of any document and transforms it into a two-person dialog between the writer and an imagined reader: we treat sentences from the article as utterances spoken by the writer, and then use a dialog inpainter to predict what the imagined reader asked or said in between each of the writer’s utterances. By applying this approach to passages from Wikipedia and the web, we produce WikiDialog and WebDialog, two datasets totalling 19 million diverse information-seeking dialogs – 1,000x larger than the largest existing ConvQA dataset. Furthermore, human raters judge the answer adequacy and conversationality of WikiDialog to be as good or better than existing manually-collected datasets. Remarkably, our approach shows strong zero-shot capability, generating high quality synthetic data without using any in-domain ConvQA data. Using our inpainted data to pre-train ConvQA retrieval systems, we significantly advance state-of-the-art across three benchmarks (QReCC, OR-QuAC, TREC CAsT) yielding up to 40% relative gains on standard evaluation metrics.

        ----

        ## [197] DisPFL: Towards Communication-Efficient Personalized Federated Learning via Decentralized Sparse Training

        **Authors**: *Rong Dai, Li Shen, Fengxiang He, Xinmei Tian, Dacheng Tao*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dai22b.html](https://proceedings.mlr.press/v162/dai22b.html)

        **Abstract**:

        Personalized federated learning is proposed to handle the data heterogeneity problem amongst clients by learning dedicated tailored local models for each user. However, existing works are often built in a centralized way, leading to high communication pressure and high vulnerability when a failure or an attack on the central server occurs. In this work, we propose a novel personalized federated learning framework in a decentralized (peer-to-peer) communication protocol named DisPFL, which employs personalized sparse masks to customize sparse local models on the edge. To further save the communication and computation cost, we propose a decentralized sparse training technique, which means that each local model in DisPFL only maintains a fixed number of active parameters throughout the whole local training and peer-to-peer communication process. Comprehensive experiments demonstrate that DisPFL significantly saves the communication bottleneck for the busiest node among all clients and, at the same time, achieves higher model accuracy with less computation cost and communication rounds. Furthermore, we demonstrate that our method can easily adapt to heterogeneous local clients with varying computation complexities and achieves better personalized performances.

        ----

        ## [198] Marginal Distribution Adaptation for Discrete Sets via Module-Oriented Divergence Minimization

        **Authors**: *Hanjun Dai, Mengjiao Yang, Yuan Xue, Dale Schuurmans, Bo Dai*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dai22c.html](https://proceedings.mlr.press/v162/dai22c.html)

        **Abstract**:

        Distributions over discrete sets capture the essential statistics including the high-order correlation among elements. Such information provides powerful insight for decision making across various application domains, e.g., product assortment based on product distribution in shopping carts. While deep generative models trained on pre-collected data can capture existing distributions, such pre-trained models are usually not capable of aligning with a target domain in the presence of distribution shift due to reasons such as temporal shift or the change in the population mix. We develop a general framework to adapt a generative model subject to a (possibly counterfactual) target data distribution with both sampling and computation efficiency. Concretely, instead of re-training a full model from scratch, we reuse the learned modules to preserve the correlations between set elements, while only adjusting corresponding components to align with target marginal constraints. We instantiate the approach for three commonly used forms of discrete set distribution—latent variable, autoregressive, and energy based models—and provide efficient solutions for marginal-constrained optimization in either primal or dual forms. Experiments on both synthetic and real-world e-commerce and EHR datasets show that the proposed framework is able to practically align a generative model to match marginal constraints under distribution shift.

        ----

        ## [199] Balancing Sample Efficiency and Suboptimality in Inverse Reinforcement Learning

        **Authors**: *Angelo Damiani, Giorgio Manganini, Alberto Maria Metelli, Marcello Restelli*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/damiani22a.html](https://proceedings.mlr.press/v162/damiani22a.html)

        **Abstract**:

        We propose a novel formulation for the Inverse Reinforcement Learning (IRL) problem, which jointly accounts for the compatibility with the expert behavior of the identified reward and its effectiveness for the subsequent forward learning phase. Albeit quite natural, especially when the final goal is apprenticeship learning (learning policies from an expert), this aspect has been completely overlooked by IRL approaches so far. We propose a new model-free IRL method that is remarkably able to autonomously find a trade-off between the error induced on the learned policy when potentially choosing a sub-optimal reward, and the estimation error caused by using finite samples in the forward learning phase, which can be controlled by explicitly optimizing also the discount factor of the related learning problem. The approach is based on a min-max formulation for the robust selection of the reward parameters and the discount factor so that the distance between the expert’s policy and the learned policy is minimized in the successive forward learning task when a finite and possibly small number of samples is available. Differently from the majority of other IRL techniques, our approach does not involve any planning or forward Reinforcement Learning problems to be solved. After presenting the formulation, we provide a numerical scheme for the optimization, and we show its effectiveness on an illustrative numerical case.

        ----

        

[Go to the next page](ICML-2022-list02.md)

[Go to the catalog section](README.md)