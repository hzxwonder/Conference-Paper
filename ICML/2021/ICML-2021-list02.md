## [200] Environment Inference for Invariant Learning

**Authors**: *Elliot Creager, Jörn-Henrik Jacobsen, Richard S. Zemel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/creager21a.html](http://proceedings.mlr.press/v139/creager21a.html)

**Abstract**:

Learning models that gracefully handle distribution shifts is central to research on domain generalization, robust optimization, and fairness. A promising formulation is domain-invariant learning, which identifies the key issue of learning which features are domain-specific versus domain-invariant. An important assumption in this area is that the training examples are partitioned into “domains” or “environments”. Our focus is on the more common setting where such partitions are not provided. We propose EIIL, a general framework for domain-invariant learning that incorporates Environment Inference to directly infer partitions that are maximally informative for downstream Invariant Learning. We show that EIIL outperforms invariant learning methods on the CMNIST benchmark without using environment labels, and significantly outperforms ERM on worst-group performance in the Waterbirds dataset. Finally, we establish connections between EIIL and algorithmic fairness, which enables EIIL to improve accuracy and calibration in a fair prediction problem.

----

## [201] Mind the Box: l1-APGD for Sparse Adversarial Attacks on Image Classifiers

**Authors**: *Francesco Croce, Matthias Hein*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/croce21a.html](http://proceedings.mlr.press/v139/croce21a.html)

**Abstract**:

We show that when taking into account also the image domain $[0,1]^d$, established $l_1$-projected gradient descent (PGD) attacks are suboptimal as they do not consider that the effective threat model is the intersection of the $l_1$-ball and $[0,1]^d$. We study the expected sparsity of the steepest descent step for this effective threat model and show that the exact projection onto this set is computationally feasible and yields better performance. Moreover, we propose an adaptive form of PGD which is highly effective even with a small budget of iterations. Our resulting $l_1$-APGD is a strong white-box attack showing that prior works overestimated their $l_1$-robustness. Using $l_1$-APGD for adversarial training we get a robust classifier with SOTA $l_1$-robustness. Finally, we combine $l_1$-APGD and an adaptation of the Square Attack to $l_1$ into $l_1$-AutoAttack, an ensemble of attacks which reliably assesses adversarial robustness for the threat model of $l_1$-ball intersected with $[0,1]^d$.

----

## [202] Parameterless Transductive Feature Re-representation for Few-Shot Learning

**Authors**: *Wentao Cui, Yuhong Guo*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cui21a.html](http://proceedings.mlr.press/v139/cui21a.html)

**Abstract**:

Recent literature in few-shot learning (FSL) has shown that transductive methods often outperform their inductive counterparts. However, most transductive solutions, particularly the meta-learning based ones, require inserting trainable parameters on top of some inductive baselines to facilitate transduction. In this paper, we propose a parameterless transductive feature re-representation framework that differs from all existing solutions from the following perspectives. (1) It is widely compatible with existing FSL methods, including meta-learning and fine tuning based models. (2) The framework is simple and introduces no extra training parameters when applied to any architecture. We conduct experiments on three benchmark datasets by applying the framework to both representative meta-learning baselines and state-of-the-art FSL methods. Our framework consistently improves performances in all experiments and refreshes the state-of-the-art FSL results.

----

## [203] Randomized Algorithms for Submodular Function Maximization with a k-System Constraint

**Authors**: *Shuang Cui, Kai Han, Tianshuai Zhu, Jing Tang, Benwei Wu, He Huang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cui21b.html](http://proceedings.mlr.press/v139/cui21b.html)

**Abstract**:

Submodular optimization has numerous applications such as crowdsourcing and viral marketing. In this paper, we study the problem of non-negative submodular function maximization subject to a $k$-system constraint, which generalizes many other important constraints in submodular optimization such as cardinality constraint, matroid constraint, and $k$-extendible system constraint. The existing approaches for this problem are all based on deterministic algorithmic frameworks, and the best approximation ratio achieved by these algorithms (for a general submodular function) is $k+2\sqrt{k+2}+3$. We propose a randomized algorithm with an improved approximation ratio of $(1+\sqrt{k})^2$, while achieving nearly-linear time complexity significantly lower than that of the state-of-the-art algorithm. We also show that our algorithm can be further generalized to address a stochastic case where the elements can be adaptively selected, and propose an approximation ratio of $(1+\sqrt{k+1})^2$ for the adaptive optimization case. The empirical performance of our algorithms is extensively evaluated in several applications related to data mining and social computing, and the experimental results demonstrate the superiorities of our algorithms in terms of both utility and efficiency.

----

## [204] GBHT: Gradient Boosting Histogram Transform for Density Estimation

**Authors**: *Jingyi Cui, Hanyuan Hang, Yisen Wang, Zhouchen Lin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cui21c.html](http://proceedings.mlr.press/v139/cui21c.html)

**Abstract**:

In this paper, we propose a density estimation algorithm called \textit{Gradient Boosting Histogram Transform} (GBHT), where we adopt the \textit{Negative Log Likelihood} as the loss function to make the boosting procedure available for the unsupervised tasks. From a learning theory viewpoint, we first prove fast convergence rates for GBHT with the smoothness assumption that the underlying density function lies in the space $C^{0,\alpha}$. Then when the target density function lies in spaces $C^{1,\alpha}$, we present an upper bound for GBHT which is smaller than the lower bound of its corresponding base learner, in the sense of convergence rates. To the best of our knowledge, we make the first attempt to theoretically explain why boosting can enhance the performance of its base learners for density estimation problems. In experiments, we not only conduct performance comparisons with the widely used KDE, but also apply GBHT to anomaly detection to showcase a further application of GBHT.

----

## [205] ProGraML: A Graph-based Program Representation for Data Flow Analysis and Compiler Optimizations

**Authors**: *Chris Cummins, Zacharias V. Fisches, Tal Ben-Nun, Torsten Hoefler, Michael F. P. O'Boyle, Hugh Leather*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cummins21a.html](http://proceedings.mlr.press/v139/cummins21a.html)

**Abstract**:

Machine learning (ML) is increasingly seen as a viable approach for building compiler optimization heuristics, but many ML methods cannot replicate even the simplest of the data flow analyses that are critical to making good optimization decisions. We posit that if ML cannot do that, then it is insufficiently able to reason about programs. We formulate data flow analyses as supervised learning tasks and introduce a large open dataset of programs and their corresponding labels from several analyses. We use this dataset to benchmark ML methods and show that they struggle on these fundamental program reasoning tasks. We propose ProGraML - Program Graphs for Machine Learning - a language-independent, portable representation of program semantics. ProGraML overcomes the limitations of prior works and yields improved performance on downstream optimization tasks.

----

## [206] Combining Pessimism with Optimism for Robust and Efficient Model-Based Deep Reinforcement Learning

**Authors**: *Sebastian Curi, Ilija Bogunovic, Andreas Krause*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/curi21a.html](http://proceedings.mlr.press/v139/curi21a.html)

**Abstract**:

In real-world tasks, reinforcement learning (RL) agents frequently encounter situations that are not present during training time. To ensure reliable performance, the RL agents need to exhibit robustness to such worst-case situations. The robust-RL framework addresses this challenge via a minimax optimization between an agent and an adversary. Previous robust RL algorithms are either sample inefficient, lack robustness guarantees, or do not scale to large problems. We propose the Robust Hallucinated Upper-Confidence RL (RH-UCRL) algorithm to provably solve this problem while attaining near-optimal sample complexity guarantees. RH-UCRL is a model-based reinforcement learning (MBRL) algorithm that effectively distinguishes between epistemic and aleatoric uncertainty and efficiently explores both the agent and the adversary decision spaces during policy learning. We scale RH-UCRL to complex tasks via neural networks ensemble models as well as neural network policies. Experimentally we demonstrate that RH-UCRL outperforms other robust deep RL algorithms in a variety of adversarial environments.

----

## [207] Quantifying Availability and Discovery in Recommender Systems via Stochastic Reachability

**Authors**: *Mihaela Curmei, Sarah Dean, Benjamin Recht*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/curmei21a.html](http://proceedings.mlr.press/v139/curmei21a.html)

**Abstract**:

In this work, we consider how preference models in interactive recommendation systems determine the availability of content and users’ opportunities for discovery. We propose an evaluation procedure based on stochastic reachability to quantify the maximum probability of recommending a target piece of content to an user for a set of allowable strategic modifications. This framework allows us to compute an upper bound on the likelihood of recommendation with minimal assumptions about user behavior. Stochastic reachability can be used to detect biases in the availability of content and diagnose limitations in the opportunities for discovery granted to users. We show that this metric can be computed efficiently as a convex program for a variety of practical settings, and further argue that reachability is not inherently at odds with accuracy. We demonstrate evaluations of recommendation algorithms trained on large datasets of explicit and implicit ratings. Our results illustrate how preference models, selection rules, and user interventions impact reachability and how these effects can be distributed unevenly.

----

## [208] Dynamic Balancing for Model Selection in Bandits and RL

**Authors**: *Ashok Cutkosky, Christoph Dann, Abhimanyu Das, Claudio Gentile, Aldo Pacchiano, Manish Purohit*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cutkosky21a.html](http://proceedings.mlr.press/v139/cutkosky21a.html)

**Abstract**:

We propose a framework for model selection by combining base algorithms in stochastic bandits and reinforcement learning. We require a candidate regret bound for each base algorithm that may or may not hold. We select base algorithms to play in each round using a “balancing condition” on the candidate regret bounds. Our approach simultaneously recovers previous worst-case regret bounds, while also obtaining much smaller regret in natural scenarios when some base learners significantly exceed their candidate bounds. Our framework is relevant in many settings, including linear bandits and MDPs with nested function classes, linear bandits with unknown misspecification, and tuning confidence parameters of algorithms such as LinUCB. Moreover, unlike recent efforts in model selection for linear stochastic bandits, our approach can be extended to consider adversarial rather than stochastic contexts.

----

## [209] ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases

**Authors**: *Stéphane d'Ascoli, Hugo Touvron, Matthew L. Leavitt, Ari S. Morcos, Giulio Biroli, Levent Sagun*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/d-ascoli21a.html](http://proceedings.mlr.press/v139/d-ascoli21a.html)

**Abstract**:

Convolutional architectures have proven extremely successful for vision tasks. Their hard inductive biases enable sample-efficient learning, but come at the cost of a potentially lower performance ceiling. Vision Transformers (ViTs) rely on more flexible self-attention layers, and have recently outperformed CNNs for image classification. However, they require costly pre-training on large external datasets or distillation from pre-trained convolutional networks. In this paper, we ask the following question: is it possible to combine the strengths of these two architectures while avoiding their respective limitations? To this end, we introduce gated positional self-attention (GPSA), a form of positional self-attention which can be equipped with a “soft" convolutional inductive bias. We initialise the GPSA layers to mimic the locality of convolutional layers, then give each attention head the freedom to escape locality by adjusting a gating parameter regulating the attention paid to position versus content information. The resulting convolutional-like ViT architecture, ConViT, outperforms the DeiT on ImageNet, while offering a much improved sample efficiency. We further investigate the role of locality in learning by first quantifying how it is encouraged in vanilla self-attention layers, then analysing how it is escaped in GPSA layers. We conclude by presenting various ablations to better understand the success of the ConViT. Our code and models are released publicly at https://github.com/facebookresearch/convit.

----

## [210] Consistent regression when oblivious outliers overwhelm

**Authors**: *Tommaso d'Orsi, Gleb Novikov, David Steurer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/d-orsi21a.html](http://proceedings.mlr.press/v139/d-orsi21a.html)

**Abstract**:

We consider a robust linear regression model $y=X\beta^* + \eta$, where an adversary oblivious to the design $X\in \mathbb{R}^{n\times d}$ may choose $\eta$ to corrupt all but an $\alpha$ fraction of the observations $y$ in an arbitrary way. Prior to our work, even for Gaussian $X$, no estimator for $\beta^*$ was known to be consistent in this model except for quadratic sample size $n \gtrsim (d/\alpha)^2$ or for logarithmic inlier fraction $\alpha\ge 1/\log n$. We show that consistent estimation is possible with nearly linear sample size and inverse-polynomial inlier fraction. Concretely, we show that the Huber loss estimator is consistent for every sample size $n= \omega(d/\alpha^2)$ and achieves an error rate of $O(d/\alpha^2n)^{1/2}$ (both bounds are optimal up to constant factors). Our results extend to designs far beyond the Gaussian case and only require the column span of $X$ to not contain approximately sparse vectors (similar to the kind of assumption commonly made about the kernel space for compressed sensing). We provide two technically similar proofs. One proof is phrased in terms of strong convexity, extending work of [Tsakonas et al. ’14], and particularly short. The other proof highlights a connection between the Huber loss estimator and high-dimensional median computations. In the special case of Gaussian designs, this connection leads us to a strikingly simple algorithm based on computing coordinate-wise medians that achieves nearly optimal guarantees in linear time, and that can exploit sparsity of $\beta^*$. The model studied here also captures heavy-tailed noise distributions that may not even have a first moment.

----

## [211] Offline Reinforcement Learning with Pseudometric Learning

**Authors**: *Robert Dadashi, Shideh Rezaeifar, Nino Vieillard, Léonard Hussenot, Olivier Pietquin, Matthieu Geist*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dadashi21a.html](http://proceedings.mlr.press/v139/dadashi21a.html)

**Abstract**:

Offline Reinforcement Learning methods seek to learn a policy from logged transitions of an environment, without any interaction. In the presence of function approximation, and under the assumption of limited coverage of the state-action space of the environment, it is necessary to enforce the policy to visit state-action pairs close to the support of logged transitions. In this work, we propose an iterative procedure to learn a pseudometric (closely related to bisimulation metrics) from logged transitions, and use it to define this notion of closeness. We show its convergence and extend it to the function approximation setting. We then use this pseudometric to define a new lookup based bonus in an actor-critic algorithm: PLOFF. This bonus encourages the actor to stay close, in terms of the defined pseudometric, to the support of logged transitions. Finally, we evaluate the method on hand manipulation and locomotion tasks.

----

## [212] A Tale of Two Efficient and Informative Negative Sampling Distributions

**Authors**: *Shabnam Daghaghi, Tharun Medini, Nicholas Meisburger, Beidi Chen, Mengnan Zhao, Anshumali Shrivastava*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/daghaghi21a.html](http://proceedings.mlr.press/v139/daghaghi21a.html)

**Abstract**:

Softmax classifiers with a very large number of classes naturally occur in many applications such as natural language processing and information retrieval. The calculation of full softmax is costly from the computational and energy perspective. There have been various sampling approaches to overcome this challenge, popularly known as negative sampling (NS). Ideally, NS should sample negative classes from a distribution that is dependent on the input data, the current parameters, and the correct positive class. Unfortunately, due to the dynamically updated parameters and data samples, there is no sampling scheme that is provably adaptive and samples the negative classes efficiently. Therefore, alternative heuristics like random sampling, static frequency-based sampling, or learning-based biased sampling, which primarily trade either the sampling cost or the adaptivity of samples per iteration are adopted. In this paper, we show two classes of distributions where the sampling scheme is truly adaptive and provably generates negative samples in near-constant time. Our implementation in C++ on CPU is significantly superior, both in terms of wall-clock time and accuracy, compared to the most optimized TensorFlow implementations of other popular negative sampling approaches on powerful NVIDIA V100 GPU.

----

## [213] SiameseXML: Siamese Networks meet Extreme Classifiers with 100M Labels

**Authors**: *Kunal Dahiya, Ananye Agarwal, Deepak Saini, Gururaj K, Jian Jiao, Amit Singh, Sumeet Agarwal, Purushottam Kar, Manik Varma*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dahiya21a.html](http://proceedings.mlr.press/v139/dahiya21a.html)

**Abstract**:

Deep extreme multi-label learning (XML) requires training deep architectures that can tag a data point with its most relevant subset of labels from an extremely large label set. XML applications such as ad and product recommendation involve labels rarely seen during training but which nevertheless hold the key to recommendations that delight users. Effective utilization of label metadata and high quality predictions for rare labels at the scale of millions of labels are thus key challenges in contemporary XML research. To address these, this paper develops the SiameseXML framework based on a novel probabilistic model that naturally motivates a modular approach melding Siamese architectures with high-capacity extreme classifiers, and a training pipeline that effortlessly scales to tasks with 100 million labels. SiameseXML offers predictions 2–13% more accurate than leading XML methods on public benchmark datasets, as well as in live A/B tests on the Bing search engine, it offers significant gains in click-through-rates, coverage, revenue and other online metrics over state-of-the-art techniques currently in production. Code for SiameseXML is available at https://github.com/Extreme-classification/siamesexml

----

## [214] Fixed-Parameter and Approximation Algorithms for PCA with Outliers

**Authors**: *Yogesh Dahiya, Fedor V. Fomin, Fahad Panolan, Kirill Simonov*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dahiya21b.html](http://proceedings.mlr.press/v139/dahiya21b.html)

**Abstract**:

PCA with Outliers is the fundamental problem of identifying an underlying low-dimensional subspace in a data set corrupted with outliers. A large body of work is devoted to the information-theoretic aspects of this problem. However, from the computational perspective, its complexity is still not well-understood. We study this problem from the perspective of parameterized complexity by investigating how parameters like the dimension of the data, the subspace dimension, the number of outliers and their structure, and approximation error, influence the computational complexity of the problem. Our algorithmic methods are based on techniques of randomized linear algebra and algebraic geometry.

----

## [215] Sliced Iterative Normalizing Flows

**Authors**: *Biwei Dai, Uros Seljak*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dai21a.html](http://proceedings.mlr.press/v139/dai21a.html)

**Abstract**:

We develop an iterative (greedy) deep learning (DL) algorithm which is able to transform an arbitrary probability distribution function (PDF) into the target PDF. The model is based on iterative Optimal Transport of a series of 1D slices, matching on each slice the marginal PDF to the target. The axes of the orthogonal slices are chosen to maximize the PDF difference using Wasserstein distance at each iteration, which enables the algorithm to scale well to high dimensions. As special cases of this algorithm, we introduce two sliced iterative Normalizing Flow (SINF) models, which map from the data to the latent space (GIS) and vice versa (SIG). We show that SIG is able to generate high quality samples of image datasets, which match the GAN benchmarks, while GIS obtains competitive results on density estimation tasks compared to the density trained NFs, and is more stable, faster, and achieves higher p(x) when trained on small training sets. SINF approach deviates significantly from the current DL paradigm, as it is greedy and does not use concepts such as mini-batching, stochastic gradient descent and gradient back-propagation through deep layers.

----

## [216] Convex Regularization in Monte-Carlo Tree Search

**Authors**: *Tuan Dam, Carlo D'Eramo, Jan Peters, Joni Pajarinen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dam21a.html](http://proceedings.mlr.press/v139/dam21a.html)

**Abstract**:

Monte-Carlo planning and Reinforcement Learning (RL) are essential to sequential decision making. The recent AlphaGo and AlphaZero algorithms have shown how to successfully combine these two paradigms to solve large-scale sequential decision problems. These methodologies exploit a variant of the well-known UCT algorithm to trade off the exploitation of good actions and the exploration of unvisited states, but their empirical success comes at the cost of poor sample-efficiency and high computation time. In this paper, we overcome these limitations by introducing the use of convex regularization in Monte-Carlo Tree Search (MCTS) to drive exploration efficiently and to improve policy updates. First, we introduce a unifying theory on the use of generic convex regularizers in MCTS, deriving the first regret analysis of regularized MCTS and showing that it guarantees an exponential convergence rate. Second, we exploit our theoretical framework to introduce novel regularized backup operators for MCTS, based on the relative entropy of the policy update and, more importantly, on the Tsallis entropy of the policy, for which we prove superior theoretical guarantees. We empirically verify the consequence of our theoretical results on a toy problem. Finally, we show how our framework can easily be incorporated in AlphaGo and we empirically show the superiority of convex regularization, w.r.t. representative baselines, on well-known RL problems across several Atari games.

----

## [217] Demonstration-Conditioned Reinforcement Learning for Few-Shot Imitation

**Authors**: *Christopher R. Dance, Julien Perez, Théo Cachet*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dance21a.html](http://proceedings.mlr.press/v139/dance21a.html)

**Abstract**:

In few-shot imitation, an agent is given a few demonstrations of a previously unseen task, and must then successfully perform that task. We propose a novel approach to learning few-shot-imitation agents that we call demonstration-conditioned reinforcement learning (DCRL). Given a training set consisting of demonstrations, reward functions and transition distributions for multiple tasks, the idea is to work with a policy that takes demonstrations as input, and to train this policy to maximize the average of the cumulative reward over the set of training tasks. Relative to previously proposed few-shot imitation methods that use behaviour cloning or infer reward functions from demonstrations, our method has the disadvantage that it requires reward functions at training time. However, DCRL also has several advantages, such as the ability to improve upon suboptimal demonstrations, to operate given state-only demonstrations, and to cope with a domain shift between the demonstrator and the agent. Moreover, we show that DCRL outperforms methods based on behaviour cloning by a large margin, on navigation tasks and on robotic manipulation tasks from the Meta-World benchmark.

----

## [218] Re-understanding Finite-State Representations of Recurrent Policy Networks

**Authors**: *Mohamad H. Danesh, Anurag Koul, Alan Fern, Saeed Khorram*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/danesh21a.html](http://proceedings.mlr.press/v139/danesh21a.html)

**Abstract**:

We introduce an approach for understanding control policies represented as recurrent neural networks. Recent work has approached this problem by transforming such recurrent policy networks into finite-state machines (FSM) and then analyzing the equivalent minimized FSM. While this led to interesting insights, the minimization process can obscure a deeper understanding of a machine’s operation by merging states that are semantically distinct. To address this issue, we introduce an analysis approach that starts with an unminimized FSM and applies more-interpretable reductions that preserve the key decision points of the policy. We also contribute an attention tool to attain a deeper understanding of the role of observations in the decisions. Our case studies on 7 Atari games and 3 control benchmarks demonstrate that the approach can reveal insights that have not been previously noticed.

----

## [219] Newton Method over Networks is Fast up to the Statistical Precision

**Authors**: *Amir Daneshmand, Gesualdo Scutari, Pavel E. Dvurechensky, Alexander V. Gasnikov*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/daneshmand21a.html](http://proceedings.mlr.press/v139/daneshmand21a.html)

**Abstract**:

We propose a distributed cubic regularization of the Newton method for solving (constrained) empirical risk minimization problems over a network of agents, modeled as undirected graph. The algorithm employs an inexact, preconditioned Newton step at each agent’s side: the gradient of the centralized loss is iteratively estimated via a gradient-tracking consensus mechanism and the Hessian is subsampled over the local data sets. No Hessian matrices are exchanged over the network. We derive global complexity bounds for convex and strongly convex losses. Our analysis reveals an interesting interplay between sample and iteration/communication complexity: statistically accurate solutions are achievable in roughly the same number of iterations of the centralized cubic Newton, with a communication cost per iteration of the order of $\widetilde{\mathcal{O}}\big(1/\sqrt{1-\rho}\big)$, where $\rho$ characterizes the connectivity of the network. This represents a significant improvement with respect to existing, statistically oblivious, distributed Newton-based methods over networks.

----

## [220] BasisDeVAE: Interpretable Simultaneous Dimensionality Reduction and Feature-Level Clustering with Derivative-Based Variational Autoencoders

**Authors**: *Dominic Danks, Christopher Yau*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/danks21a.html](http://proceedings.mlr.press/v139/danks21a.html)

**Abstract**:

The Variational Autoencoder (VAE) performs effective nonlinear dimensionality reduction in a variety of problem settings. However, the black-box neural network decoder function typically employed limits the ability of the decoder function to be constrained and interpreted, making the use of VAEs problematic in settings where prior knowledge should be embedded within the decoder. We present DeVAE, a novel VAE-based model with a derivative-based forward mapping, allowing for greater control over decoder behaviour via specification of the decoder function in derivative space. Additionally, we show how DeVAE can be paired with a sparse clustering prior to create BasisDeVAE and perform interpretable simultaneous dimensionality reduction and feature-level clustering. We demonstrate the performance and scalability of the DeVAE and BasisDeVAE models on synthetic and real-world data and present how the derivative-based approach allows for expressive yet interpretable forward models which respect prior knowledge.

----

## [221] Intermediate Layer Optimization for Inverse Problems using Deep Generative Models

**Authors**: *Giannis Daras, Joseph Dean, Ajil Jalal, Alex Dimakis*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/daras21a.html](http://proceedings.mlr.press/v139/daras21a.html)

**Abstract**:

We propose Intermediate Layer Optimization (ILO), a novel optimization algorithm for solving inverse problems with deep generative models. Instead of optimizing only over the initial latent code, we progressively change the input layer obtaining successively more expressive generators. To explore the higher dimensional spaces, our method searches for latent codes that lie within a small l1 ball around the manifold induced by the previous layer. Our theoretical analysis shows that by keeping the radius of the ball relatively small, we can improve the established error bound for compressed sensing with deep generative models. We empirically show that our approach outperforms state-of-the-art methods introduced in StyleGAN2 and PULSE for a wide range of inverse problems including inpainting, denoising, super-resolution and compressed sensing.

----

## [222] Measuring Robustness in Deep Learning Based Compressive Sensing

**Authors**: *Mohammad Zalbagi Darestani, Akshay S. Chaudhari, Reinhard Heckel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/darestani21a.html](http://proceedings.mlr.press/v139/darestani21a.html)

**Abstract**:

Deep neural networks give state-of-the-art accuracy for reconstructing images from few and noisy measurements, a problem arising for example in accelerated magnetic resonance imaging (MRI). However, recent works have raised concerns that deep-learning-based image reconstruction methods are sensitive to perturbations and are less robust than traditional methods: Neural networks (i) may be sensitive to small, yet adversarially-selected perturbations, (ii) may perform poorly under distribution shifts, and (iii) may fail to recover small but important features in an image. In order to understand the sensitivity to such perturbations, in this work, we measure the robustness of different approaches for image reconstruction including trained and un-trained neural networks as well as traditional sparsity-based methods. We find, contrary to prior works, that both trained and un-trained methods are vulnerable to adversarial perturbations. Moreover, both trained and un-trained methods tuned for a particular dataset suffer very similarly from distribution shifts. Finally, we demonstrate that an image reconstruction method that achieves higher reconstruction quality, also performs better in terms of accurately recovering fine details. Our results indicate that the state-of-the-art deep-learning-based image reconstruction methods provide improved performance than traditional methods without compromising robustness.

----

## [223] SAINT-ACC: Safety-Aware Intelligent Adaptive Cruise Control for Autonomous Vehicles Using Deep Reinforcement Learning

**Authors**: *Lokesh Chandra Das, Myounggyu Won*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/das21a.html](http://proceedings.mlr.press/v139/das21a.html)

**Abstract**:

We present a novel adaptive cruise control (ACC) system namely SAINT-ACC: {S}afety-{A}ware {Int}elligent {ACC} system (SAINT-ACC) that is designed to achieve simultaneous optimization of traffic efficiency, driving safety, and driving comfort through dynamic adaptation of the inter-vehicle gap based on deep reinforcement learning (RL). A novel dual RL agent-based approach is developed to seek and adapt the optimal balance between traffic efficiency and driving safety/comfort by effectively controlling the driving safety model parameters and inter-vehicle gap based on macroscopic and microscopic traffic information collected from dynamically changing and complex traffic environments. Results obtained through over 12,000 simulation runs with varying traffic scenarios and penetration rates demonstrate that SAINT-ACC significantly enhances traffic flow, driving safety and comfort compared with a state-of-the-art approach.

----

## [224] Lipschitz normalization for self-attention layers with application to graph neural networks

**Authors**: *George Dasoulas, Kevin Scaman, Aladin Virmaux*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dasoulas21a.html](http://proceedings.mlr.press/v139/dasoulas21a.html)

**Abstract**:

Attention based neural networks are state of the art in a large range of applications. However, their performance tends to degrade when the number of layers increases. In this work, we show that enforcing Lipschitz continuity by normalizing the attention scores can significantly improve the performance of deep attention models. First, we show that, for deep graph attention networks (GAT), gradient explosion appears during training, leading to poor performance of gradient-based training algorithms. To address this issue, we derive a theoretical analysis of the Lipschitz continuity of attention modules and introduce LipschitzNorm, a simple and parameter-free normalization for self-attention mechanisms that enforces the model to be Lipschitz continuous. We then apply LipschitzNorm to GAT and Graph Transformers and show that their performance is substantially improved in the deep setting (10 to 30 layers). More specifically, we show that a deep GAT model with LipschitzNorm achieves state of the art results for node label prediction tasks that exhibit long-range dependencies, while showing consistent improvements over their unnormalized counterparts in benchmark node classification tasks.

----

## [225] Householder Sketch for Accurate and Accelerated Least-Mean-Squares Solvers

**Authors**: *Jyotikrishna Dass, Rabi N. Mahapatra*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dass21a.html](http://proceedings.mlr.press/v139/dass21a.html)

**Abstract**:

Least-Mean-Squares (\textsc{LMS}) solvers comprise a class of fundamental optimization problems such as linear regression, and regularized regressions such as Ridge, LASSO, and Elastic-Net. Data summarization techniques for big data generate summaries called coresets and sketches to speed up model learning under streaming and distributed settings. For example, \citep{nips2019} design a fast and accurate Caratheodory set on input data to boost the performance of existing \textsc{LMS} solvers. In retrospect, we explore classical Householder transformation as a candidate for sketching and accurately solving LMS problems. We find it to be a simpler, memory-efficient, and faster alternative that always existed to the above strong baseline. We also present a scalable algorithm based on the construction of distributed Householder sketches to solve \textsc{LMS} problem across multiple worker nodes. We perform thorough empirical analysis with large synthetic and real datasets to evaluate the performance of Householder sketch and compare with \citep{nips2019}. Our results show Householder sketch speeds up existing \textsc{LMS} solvers in the scikit-learn library up to $100$x-$400$x. Also, it is $10$x-$100$x faster than the above baseline with similar numerical stability. The distributed algorithm demonstrates linear scalability with a near-negligible communication overhead.

----

## [226] Byzantine-Resilient High-Dimensional SGD with Local Iterations on Heterogeneous Data

**Authors**: *Deepesh Data, Suhas N. Diggavi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/data21a.html](http://proceedings.mlr.press/v139/data21a.html)

**Abstract**:

We study stochastic gradient descent (SGD) with local iterations in the presence of Byzantine clients, motivated by the federated learning. The clients, instead of communicating with the server in every iteration, maintain their local models, which they update by taking several SGD iterations based on their own datasets and then communicate the net update with the server, thereby achieving communication-efficiency. Furthermore, only a subset of clients communicates with the server at synchronization times. The Byzantine clients may collude and send arbitrary vectors to the server to disrupt the learning process. To combat the adversary, we employ an efficient high-dimensional robust mean estimation algorithm at the server to filter-out corrupt vectors; and to analyze the outlier-filtering procedure, we develop a novel matrix concentration result that may be of independent interest. We provide convergence analyses for both strongly-convex and non-convex smooth objectives in the heterogeneous data setting. We believe that ours is the first Byzantine-resilient local SGD algorithm and analysis with non-trivial guarantees. We corroborate our theoretical results with preliminary experiments for neural network training.

----

## [227] Catformer: Designing Stable Transformers via Sensitivity Analysis

**Authors**: *Jared Quincy Davis, Albert Gu, Krzysztof Choromanski, Tri Dao, Christopher Ré, Chelsea Finn, Percy Liang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/davis21a.html](http://proceedings.mlr.press/v139/davis21a.html)

**Abstract**:

Transformer architectures are widely used, but training them is non-trivial, requiring custom learning rate schedules, scaling terms, residual connections, careful placement of submodules such as normalization, and so on. In this paper, we improve upon recent analysis of Transformers and formalize a notion of sensitivity to capture the difficulty of training. Sensitivity characterizes how the variance of activation and gradient norms change in expectation when parameters are randomly perturbed. We analyze the sensitivity of previous Transformer architectures and design a new architecture, the Catformer, which replaces residual connections or RNN-based gating mechanisms with concatenation. We prove that Catformers are less sensitive than other Transformer variants and demonstrate that this leads to more stable training. On DMLab30, a suite of high-dimension reinforcement tasks, Catformer outperforms other transformers, including Gated Transformer-XL—the state-of-the-art architecture designed to address stability—by 13%.

----

## [228] Diffusion Source Identification on Networks with Statistical Confidence

**Authors**: *Quinlan Dawkins, Tianxi Li, Haifeng Xu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dawkins21a.html](http://proceedings.mlr.press/v139/dawkins21a.html)

**Abstract**:

Diffusion source identification on networks is a problem of fundamental importance in a broad class of applications, including controlling the spreading of rumors on social media, identifying a computer virus over cyber networks, or identifying the disease center during epidemiology. Though this problem has received significant recent attention, most known approaches are well-studied in only very restrictive settings and lack theoretical guarantees for more realistic networks. We introduce a statistical framework for the study of this problem and develop a confidence set inference approach inspired by hypothesis testing. Our method efficiently produces a small subset of nodes, which provably covers the source node with any pre-specified confidence level without restrictive assumptions on network structures. To our knowledge, this is the first diffusion source identification method with a practically useful theoretical guarantee on general networks. We demonstrate our approach via extensive synthetic experiments on well-known random network models, a large data set of real-world networks as well as a mobility network between cities concerning the COVID-19 spreading in January 2020.

----

## [229] Bayesian Deep Learning via Subnetwork Inference

**Authors**: *Erik A. Daxberger, Eric T. Nalisnick, James Urquhart Allingham, Javier Antorán, José Miguel Hernández-Lobato*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/daxberger21a.html](http://proceedings.mlr.press/v139/daxberger21a.html)

**Abstract**:

The Bayesian paradigm has the potential to solve core issues of deep neural networks such as poor calibration and data inefficiency. Alas, scaling Bayesian inference to large weight spaces often requires restrictive approximations. In this work, we show that it suffices to perform inference over a small subset of model weights in order to obtain accurate predictive posteriors. The other weights are kept as point estimates. This subnetwork inference framework enables us to use expressive, otherwise intractable, posterior approximations over such subsets. In particular, we implement subnetwork linearized Laplace as a simple, scalable Bayesian deep learning method: We first obtain a MAP estimate of all weights and then infer a full-covariance Gaussian posterior over a subnetwork using the linearized Laplace approximation. We propose a subnetwork selection strategy that aims to maximally preserve the model’s predictive uncertainty. Empirically, our approach compares favorably to ensembles and less expressive posterior approximations over full networks.

----

## [230] Adversarial Robustness Guarantees for Random Deep Neural Networks

**Authors**: *Giacomo De Palma, Bobak Toussi Kiani, Seth Lloyd*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/de-palma21a.html](http://proceedings.mlr.press/v139/de-palma21a.html)

**Abstract**:

The reliability of deep learning algorithms is fundamentally challenged by the existence of adversarial examples, which are incorrectly classified inputs that are extremely close to a correctly classified input. We explore the properties of adversarial examples for deep neural networks with random weights and biases, and prove that for any p$\geq$1, the \ell^p distance of any given input from the classification boundary scales as one over the square root of the dimension of the input times the \ell^p norm of the input. The results are based on the recently proved equivalence between Gaussian processes and deep neural networks in the limit of infinite width of the hidden layers, and are validated with experiments on both random deep neural networks and deep neural networks trained on the MNIST and CIFAR10 datasets. The results constitute a fundamental advance in the theoretical understanding of adversarial examples, and open the way to a thorough theoretical characterization of the relation between network architecture and robustness to adversarial perturbations.

----

## [231] High-Dimensional Gaussian Process Inference with Derivatives

**Authors**: *Filip de Roos, Alexandra Gessner, Philipp Hennig*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/de-roos21a.html](http://proceedings.mlr.press/v139/de-roos21a.html)

**Abstract**:

Although it is widely known that Gaussian processes can be conditioned on observations of the gradient, this functionality is of limited use due to the prohibitive computational cost of $\mathcal{O}(N^3 D^3)$ in data points $N$ and dimension $D$. The dilemma of gradient observations is that a single one of them comes at the same cost as $D$ independent function evaluations, so the latter are often preferred. Careful scrutiny reveals, however, that derivative observations give rise to highly structured kernel Gram matrices for very general classes of kernels (inter alia, stationary kernels). We show that in the \emph{low-data} regime $N < D$, the Gram matrix can be decomposed in a manner that reduces the cost of inference to $\mathcal{O}(N^2D + (N^2)^3)$ (i.e., linear in the number of dimensions) and, in special cases, to $\mathcal{O}(N^2D + N^3)$. This reduction in complexity opens up new use-cases for inference with gradients especially in the high-dimensional regime, where the information-to-cost ratio of gradient observations significantly increases. We demonstrate this potential in a variety of tasks relevant for machine learning, such as optimization and Hamiltonian Monte Carlo with predictive gradients.

----

## [232] Transfer-Based Semantic Anomaly Detection

**Authors**: *Lucas Deecke, Lukas Ruff, Robert A. Vandermeulen, Hakan Bilen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/deecke21a.html](http://proceedings.mlr.press/v139/deecke21a.html)

**Abstract**:

Detecting semantic anomalies is challenging due to the countless ways in which they may appear in real-world data. While enhancing the robustness of networks may be sufficient for modeling simplistic anomalies, there is no good known way of preparing models for all potential and unseen anomalies that can potentially occur, such as the appearance of new object classes. In this paper, we show that a previously overlooked strategy for anomaly detection (AD) is to introduce an explicit inductive bias toward representations transferred over from some large and varied semantic task. We rigorously verify our hypothesis in controlled trials that utilize intervention, and show that it gives rise to surprisingly effective auxiliary objectives that outperform previous AD paradigms.

----

## [233] Grid-Functioned Neural Networks

**Authors**: *Javier Dehesa, Andrew Vidler, Julian A. Padget, Christof Lutteroth*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dehesa21a.html](http://proceedings.mlr.press/v139/dehesa21a.html)

**Abstract**:

We introduce a new neural network architecture that we call "grid-functioned" neural networks. It utilises a grid structure of network parameterisations that can be specialised for different subdomains of the problem, while maintaining smooth, continuous behaviour. The grid gives the user flexibility to prevent gross features from overshadowing important minor ones. We present a full characterisation of its computational and spatial complexity, and demonstrate its potential, compared to a traditional architecture, over a set of synthetic regression problems. We further illustrate the benefits through a real-world 3D skeletal animation case study, where it offers the same visual quality as a state-of-the-art model, but with lower computational complexity and better control accuracy.

----

## [234] Multidimensional Scaling: Approximation and Complexity

**Authors**: *Erik D. Demaine, Adam Hesterberg, Frederic Koehler, Jayson Lynch, John C. Urschel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/demaine21a.html](http://proceedings.mlr.press/v139/demaine21a.html)

**Abstract**:

Metric Multidimensional scaling (MDS) is a classical method for generating meaningful (non-linear) low-dimensional embeddings of high-dimensional data. MDS has a long history in the statistics, machine learning, and graph drawing communities. In particular, the Kamada-Kawai force-directed graph drawing method is equivalent to MDS and is one of the most popular ways in practice to embed graphs into low dimensions. Despite its ubiquity, our theoretical understanding of MDS remains limited as its objective function is highly non-convex. In this paper, we prove that minimizing the Kamada-Kawai objective is NP-hard and give a provable approximation algorithm for optimizing it, which in particular is a PTAS on low-diameter graphs. We supplement this result with experiments suggesting possible connections between our greedy approximation algorithm and gradient-based methods.

----

## [235] What Does Rotation Prediction Tell Us about Classifier Accuracy under Varying Testing Environments?

**Authors**: *Weijian Deng, Stephen Gould, Liang Zheng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/deng21a.html](http://proceedings.mlr.press/v139/deng21a.html)

**Abstract**:

Understanding classifier decision under novel environments is central to the community, and a common practice is evaluating it on labeled test sets. However, in real-world testing, image annotations are difficult and expensive to obtain, especially when the test environment is changing. A natural question then arises: given a trained classifier, can we evaluate its accuracy on varying unlabeled test sets? In this work, we train semantic classification and rotation prediction in a multi-task way. On a series of datasets, we report an interesting finding, i.e., the semantic classification accuracy exhibits a strong linear relationship with the accuracy of the rotation prediction task (Pearson’s Correlation r > 0.88). This finding allows us to utilize linear regression to estimate classifier performance from the accuracy of rotation prediction which can be obtained on the test set through the freely generated rotation labels.

----

## [236] Toward Better Generalization Bounds with Locally Elastic Stability

**Authors**: *Zhun Deng, Hangfeng He, Weijie J. Su*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/deng21b.html](http://proceedings.mlr.press/v139/deng21b.html)

**Abstract**:

Algorithmic stability is a key characteristic to ensure the generalization ability of a learning algorithm. Among different notions of stability, \emph{uniform stability} is arguably the most popular one, which yields exponential generalization bounds. However, uniform stability only considers the worst-case loss change (or so-called sensitivity) by removing a single data point, which is distribution-independent and therefore undesirable. There are many cases that the worst-case sensitivity of the loss is much larger than the average sensitivity taken over the single data point that is removed, especially in some advanced models such as random feature models or neural networks. Many previous works try to mitigate the distribution independent issue by proposing weaker notions of stability, however, they either only yield polynomial bounds or the bounds derived do not vanish as sample size goes to infinity. Given that, we propose \emph{locally elastic stability} as a weaker and distribution-dependent stability notion, which still yields exponential generalization bounds. We further demonstrate that locally elastic stability implies tighter generalization bounds than those derived based on uniform stability in many situations by revisiting the examples of bounded support vector machines, regularized least square regressions, and stochastic gradient descent.

----

## [237] Revenue-Incentive Tradeoffs in Dynamic Reserve Pricing

**Authors**: *Yuan Deng, Sébastien Lahaie, Vahab S. Mirrokni, Song Zuo*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/deng21c.html](http://proceedings.mlr.press/v139/deng21c.html)

**Abstract**:

Online advertisements are primarily sold via repeated auctions with reserve prices. In this paper, we study how to set reserves to boost revenue based on the historical bids of strategic buyers, while controlling the impact of such a policy on the incentive compatibility of the repeated auctions. Adopting an incentive compatibility metric which quantifies the incentives to shade bids, we propose a novel class of reserve pricing policies and provide analytical tradeoffs between their revenue performance and bid-shading incentives. The policies are inspired by the exponential mechanism from the literature on differential privacy, but our study uncovers mechanisms with significantly better revenue-incentive tradeoffs than the exponential mechanism in practice. We further empirically evaluate the tradeoffs on synthetic data as well as real ad auction data from a major ad exchange to verify and support our theoretical findings.

----

## [238] Heterogeneity for the Win: One-Shot Federated Clustering

**Authors**: *Don Kurian Dennis, Tian Li, Virginia Smith*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dennis21a.html](http://proceedings.mlr.press/v139/dennis21a.html)

**Abstract**:

In this work, we explore the unique challenges—and opportunities—of unsupervised federated learning (FL). We develop and analyze a one-shot federated clustering scheme, kfed, based on the widely-used Lloyd’s method for $k$-means clustering. In contrast to many supervised problems, we show that the issue of statistical heterogeneity in federated networks can in fact benefit our analysis. We analyse kfed under a center separation assumption and compare it to the best known requirements of its centralized counterpart. Our analysis shows that in heterogeneous regimes where the number of clusters per device $(k’)$ is smaller than the total number of clusters over the network $k$, $(k’\le \sqrt{k})$, we can use heterogeneity to our advantage—significantly weakening the cluster separation requirements for kfed. From a practical viewpoint, kfed also has many desirable properties: it requires only round of communication, can run asynchronously, and can handle partial participation or node/network failures. We motivate our analysis with experiments on common FL benchmarks, and highlight the practical utility of one-shot clustering through use-cases in personalized FL and device sampling.

----

## [239] Kernel Continual Learning

**Authors**: *Mohammad Mahdi Derakhshani, Xiantong Zhen, Ling Shao, Cees Snoek*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/derakhshani21a.html](http://proceedings.mlr.press/v139/derakhshani21a.html)

**Abstract**:

This paper introduces kernel continual learning, a simple but effective variant of continual learning that leverages the non-parametric nature of kernel methods to tackle catastrophic forgetting. We deploy an episodic memory unit that stores a subset of samples for each task to learn task-specific classifiers based on kernel ridge regression. This does not require memory replay and systematically avoids task interference in the classifiers. We further introduce variational random features to learn a data-driven kernel for each task. To do so, we formulate kernel continual learning as a variational inference problem, where a random Fourier basis is incorporated as the latent variable. The variational posterior distribution over the random Fourier basis is inferred from the coreset of each task. In this way, we are able to generate more informative kernels specific to each task, and, more importantly, the coreset size can be reduced to achieve more compact memory, resulting in more efficient continual learning based on episodic memory. Extensive evaluation on four benchmarks demonstrates the effectiveness and promise of kernels for continual learning.

----

## [240] Bayesian Optimization over Hybrid Spaces

**Authors**: *Aryan Deshwal, Syrine Belakaria, Janardhan Rao Doppa*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/deshwal21a.html](http://proceedings.mlr.press/v139/deshwal21a.html)

**Abstract**:

We consider the problem of optimizing hybrid structures (mixture of discrete and continuous input variables) via expensive black-box function evaluations. This problem arises in many real-world applications. For example, in materials design optimization via lab experiments, discrete and continuous variables correspond to the presence/absence of primitive elements and their relative concentrations respectively. The key challenge is to accurately model the complex interactions between discrete and continuous variables. In this paper, we propose a novel approach referred as Hybrid Bayesian Optimization (HyBO) by utilizing diffusion kernels, which are naturally defined over continuous and discrete variables. We develop a principled approach for constructing diffusion kernels over hybrid spaces by utilizing the additive kernel formulation, which allows additive interactions of all orders in a tractable manner. We theoretically analyze the modeling strength of additive hybrid kernels and prove that it has the universal approximation property. Our experiments on synthetic and six diverse real-world benchmarks show that HyBO significantly outperforms the state-of-the-art methods.

----

## [241] Navigation Turing Test (NTT): Learning to Evaluate Human-Like Navigation

**Authors**: *Sam Devlin, Raluca Georgescu, Ida Momennejad, Jaroslaw Rzepecki, Evelyn Zuniga, Gavin Costello, Guy Leroy, Ali Shaw, Katja Hofmann*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/devlin21a.html](http://proceedings.mlr.press/v139/devlin21a.html)

**Abstract**:

A key challenge on the path to developing agents that learn complex human-like behavior is the need to quickly and accurately quantify human-likeness. While human assessments of such behavior can be highly accurate, speed and scalability are limited. We address these limitations through a novel automated Navigation Turing Test (ANTT) that learns to predict human judgments of human-likeness. We demonstrate the effectiveness of our automated NTT on a navigation task in a complex 3D environment. We investigate six classification models to shed light on the types of architectures best suited to this task, and validate them against data collected through a human NTT. Our best models achieve high accuracy when distinguishing true human and agent behavior. At the same time, we show that predicting finer-grained human assessment of agents’ progress towards human-like behavior remains unsolved. Our work takes an important step towards agents that more effectively learn complex human-like behavior.

----

## [242] Versatile Verification of Tree Ensembles

**Authors**: *Laurens Devos, Wannes Meert, Jesse Davis*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/devos21a.html](http://proceedings.mlr.press/v139/devos21a.html)

**Abstract**:

Machine learned models often must abide by certain requirements (e.g., fairness or legal). This has spurred interested in developing approaches that can provably verify whether a model satisfies certain properties. This paper introduces a generic algorithm called Veritas that enables tackling multiple different verification tasks for tree ensemble models like random forests (RFs) and gradient boosted decision trees (GBDTs). This generality contrasts with previous work, which has focused exclusively on either adversarial example generation or robustness checking. Veritas formulates the verification task as a generic optimization problem and introduces a novel search space representation. Veritas offers two key advantages. First, it provides anytime lower and upper bounds when the optimization problem cannot be solved exactly. In contrast, many existing methods have focused on exact solutions and are thus limited by the verification problem being NP-complete. Second, Veritas produces full (bounded suboptimal) solutions that can be used to generate concrete examples. We experimentally show that our method produces state-of-the-art robustness estimates, especially when executed with strict time constraints. This is exceedingly important when checking the robustness of large datasets. Additionally, we show that Veritas enables tackling more real-world verification scenarios.

----

## [243] On the Inherent Regularization Effects of Noise Injection During Training

**Authors**: *Oussama Dhifallah, Yue M. Lu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dhifallah21a.html](http://proceedings.mlr.press/v139/dhifallah21a.html)

**Abstract**:

Randomly perturbing networks during the training process is a commonly used approach to improving generalization performance. In this paper, we present a theoretical study of one particular way of random perturbation, which corresponds to injecting artificial noise to the training data. We provide a precise asymptotic characterization of the training and generalization errors of such randomly perturbed learning problems on a random feature model. Our analysis shows that Gaussian noise injection in the training process is equivalent to introducing a weighted ridge regularization, when the number of noise injections tends to infinity. The explicit form of the regularization is also given. Numerical results corroborate our asymptotic predictions, showing that they are accurate even in moderate problem dimensions. Our theoretical predictions are based on a new correlated Gaussian equivalence conjecture that generalizes recent results in the study of random feature models.

----

## [244] Hierarchical Agglomerative Graph Clustering in Nearly-Linear Time

**Authors**: *Laxman Dhulipala, David Eisenstat, Jakub Lacki, Vahab S. Mirrokni, Jessica Shi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dhulipala21a.html](http://proceedings.mlr.press/v139/dhulipala21a.html)

**Abstract**:

We study the widely-used hierarchical agglomerative clustering (HAC) algorithm on edge-weighted graphs. We define an algorithmic framework for hierarchical agglomerative graph clustering that provides the first efficient $\tilde{O}(m)$ time exact algorithms for classic linkage measures, such as complete- and WPGMA-linkage, as well as other measures. Furthermore, for average-linkage, arguably the most popular variant of HAC, we provide an algorithm that runs in $\tilde{O}(n\sqrt{m})$ time. For this variant, this is the first exact algorithm that runs in subquadratic time, as long as $m=n^{2-\epsilon}$ for some constant $\epsilon > 0$. We complement this result with a simple $\epsilon$-close approximation algorithm for average-linkage in our framework that runs in $\tilde{O}(m)$ time. As an application of our algorithms, we consider clustering points in a metric space by first using $k$-NN to generate a graph from the point set, and then running our algorithms on the resulting weighted graph. We validate the performance of our algorithms on publicly available datasets, and show that our approach can speed up clustering of point datasets by a factor of 20.7–76.5x.

----

## [245] Learning Online Algorithms with Distributional Advice

**Authors**: *Ilias Diakonikolas, Vasilis Kontonis, Christos Tzamos, Ali Vakilian, Nikos Zarifis*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/diakonikolas21a.html](http://proceedings.mlr.press/v139/diakonikolas21a.html)

**Abstract**:

We study the problem of designing online algorithms given advice about the input. While prior work had focused on deterministic advice, we only assume distributional access to the instances of interest, and the goal is to learn a competitive algorithm given access to i.i.d. samples. We aim to be competitive against an adversary with prior knowledge of the distribution, while also performing well against worst-case inputs. We focus on the classical online problems of ski-rental and prophet-inequalities, and provide sample complexity bounds for the underlying learning tasks. First, we point out that for general distributions it is information-theoretically impossible to beat the worst-case competitive-ratio with any finite sample size. As our main contribution, we establish strong positive results for well-behaved distributions. Specifically, for the broad class of log-concave distributions, we show that $\mathrm{poly}(1/\epsilon)$ samples suffice to obtain $(1+\epsilon)$-competitive ratio. Finally, we show that this sample upper bound is close to best possible, even for very simple classes of distributions.

----

## [246] A Wasserstein Minimax Framework for Mixed Linear Regression

**Authors**: *Theo Diamandis, Yonina C. Eldar, Alireza Fallah, Farzan Farnia, Asuman E. Ozdaglar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/diamandis21a.html](http://proceedings.mlr.press/v139/diamandis21a.html)

**Abstract**:

Multi-modal distributions are commonly used to model clustered data in statistical learning tasks. In this paper, we consider the Mixed Linear Regression (MLR) problem. We propose an optimal transport-based framework for MLR problems, Wasserstein Mixed Linear Regression (WMLR), which minimizes the Wasserstein distance between the learned and target mixture regression models. Through a model-based duality analysis, WMLR reduces the underlying MLR task to a nonconvex-concave minimax optimization problem, which can be provably solved to find a minimax stationary point by the Gradient Descent Ascent (GDA) algorithm. In the special case of mixtures of two linear regression models, we show that WMLR enjoys global convergence and generalization guarantees. We prove that WMLR’s sample complexity grows linearly with the dimension of data. Finally, we discuss the application of WMLR to the federated learning task where the training samples are collected by multiple agents in a network. Unlike the Expectation-Maximization algorithm, WMLR directly extends to the distributed, federated learning setting. We support our theoretical results through several numerical experiments, which highlight our framework’s ability to handle the federated learning setting with mixture models.

----

## [247] Context-Aware Online Collective Inference for Templated Graphical Models

**Authors**: *Charles Dickens, Connor Pryor, Eriq Augustine, Alexander Miller, Lise Getoor*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dickens21a.html](http://proceedings.mlr.press/v139/dickens21a.html)

**Abstract**:

In this work, we examine online collective inference, the problem of maintaining and performing inference over a sequence of evolving graphical models. We utilize templated graphical models (TGM), a general class of graphical models expressed via templates and instantiated with data. A key challenge is minimizing the cost of instantiating the updated model. To address this, we define a class of exact and approximate context-aware methods for updating an existing TGM. These methods avoid a full re-instantiation by using the context of the updates to only add relevant components to the graphical model. Further, we provide stability bounds for the general online inference problem and regret bounds for a proposed approximation. Finally, we implement our approach in probabilistic soft logic, and test it on several online collective inference tasks. Through these experiments we verify the bounds on regret and stability, and show that our approximate online approach consistently runs two to five times faster than the offline alternative while, surprisingly, maintaining the quality of the predictions.

----

## [248] ARMS: Antithetic-REINFORCE-Multi-Sample Gradient for Binary Variables

**Authors**: *Aleksandar Dimitriev, Mingyuan Zhou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dimitriev21a.html](http://proceedings.mlr.press/v139/dimitriev21a.html)

**Abstract**:

Estimating the gradients for binary variables is a task that arises frequently in various domains, such as training discrete latent variable models. What has been commonly used is a REINFORCE based Monte Carlo estimation method that uses either independent samples or pairs of negatively correlated samples. To better utilize more than two samples, we propose ARMS, an Antithetic REINFORCE-based Multi-Sample gradient estimator. ARMS uses a copula to generate any number of mutually antithetic samples. It is unbiased, has low variance, and generalizes both DisARM, which we show to be ARMS with two samples, and the leave-one-out REINFORCE (LOORF) estimator, which is ARMS with uncorrelated samples. We evaluate ARMS on several datasets for training generative models, and our experimental results show that it outperforms competing methods. We also develop a version of ARMS for optimizing the multi-sample variational bound, and show that it outperforms both VIMCO and DisARM. The code is publicly available.

----

## [249] XOR-CD: Linearly Convergent Constrained Structure Generation

**Authors**: *Fan Ding, Jianzhu Ma, Jinbo Xu, Yexiang Xue*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ding21a.html](http://proceedings.mlr.press/v139/ding21a.html)

**Abstract**:

We propose XOR-Contrastive Divergence learning (XOR-CD), a provable approach for constrained structure generation, which remains difficult for state-of-the-art neural network and constraint reasoning approaches. XOR-CD harnesses XOR-Sampling to generate samples from the model distribution in CD learning and is guaranteed to generate valid structures. In addition, XOR-CD has a linear convergence rate towards the global maximum of the likelihood function within a vanishing constant in learning exponential family models. Constraint satisfaction enabled by XOR-CD also boosts its learning performance. Our real-world experiments on data-driven experimental design, dispatching route generation, and sequence-based protein homology detection demonstrate the superior performance of XOR-CD compared to baseline approaches in generating valid structures as well as capturing the inductive bias in the training set.

----

## [250] Dual Principal Component Pursuit for Robust Subspace Learning: Theory and Algorithms for a Holistic Approach

**Authors**: *Tianyu Ding, Zhihui Zhu, René Vidal, Daniel P. Robinson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ding21b.html](http://proceedings.mlr.press/v139/ding21b.html)

**Abstract**:

The Dual Principal Component Pursuit (DPCP) method has been proposed to robustly recover a subspace of high-relative dimension from corrupted data. Existing analyses and algorithms of DPCP, however, mainly focus on finding a normal to a single hyperplane that contains the inliers. Although these algorithms can be extended to a subspace of higher co-dimension through a recursive approach that sequentially finds a new basis element of the space orthogonal to the subspace, this procedure is computationally expensive and lacks convergence guarantees. In this paper, we consider a DPCP approach for simultaneously computing the entire basis of the orthogonal complement subspace (we call this a holistic approach) by solving a non-convex non-smooth optimization problem over the Grassmannian. We provide geometric and statistical analyses for the global optimality and prove that it can tolerate as many outliers as the square of the number of inliers, under both noiseless and noisy settings. We then present a Riemannian regularity condition for the problem, which is then used to prove that a Riemannian subgradient method converges linearly to a neighborhood of the orthogonal subspace with error proportional to the noise level.

----

## [251] Coded-InvNet for Resilient Prediction Serving Systems

**Authors**: *Tuan Dinh, Kangwook Lee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dinh21a.html](http://proceedings.mlr.press/v139/dinh21a.html)

**Abstract**:

Inspired by a new coded computation algorithm for invertible functions, we propose Coded-InvNet a new approach to design resilient prediction serving systems that can gracefully handle stragglers or node failures. Coded-InvNet leverages recent findings in the deep learning literature such as invertible neural networks, Manifold Mixup, and domain translation algorithms, identifying interesting research directions that span across machine learning and systems. Our experimental results show that Coded-InvNet can outperform existing approaches, especially when the compute resource overhead is as low as 10%. For instance, without knowing which of the ten workers is going to fail, our algorithm can design a backup task so that it can correctly recover the missing prediction result with an accuracy of 85.9%, significantly outperforming the previous SOTA by 32.5%.

----

## [252] Estimation and Quantization of Expected Persistence Diagrams

**Authors**: *Vincent Divol, Théo Lacombe*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/divol21a.html](http://proceedings.mlr.press/v139/divol21a.html)

**Abstract**:

Persistence diagrams (PDs) are the most common descriptors used to encode the topology of structured data appearing in challenging learning tasks; think e.g. of graphs, time series or point clouds sampled close to a manifold. Given random objects and the corresponding distribution of PDs, one may want to build a statistical summary—such as a mean—of these random PDs, which is however not a trivial task as the natural geometry of the space of PDs is not linear. In this article, we study two such summaries, the Expected Persistence Diagram (EPD), and its quantization. The EPD is a measure supported on $\mathbb{R}^2$, which may be approximated by its empirical counterpart. We prove that this estimator is optimal from a minimax standpoint on a large class of models with a parametric rate of convergence. The empirical EPD is simple and efficient to compute, but possibly has a very large support, hindering its use in practice. To overcome this issue, we propose an algorithm to compute a quantization of the empirical EPD, a measure with small support which is shown to approximate with near-optimal rates a quantization of the theoretical EPD.

----

## [253] On Energy-Based Models with Overparametrized Shallow Neural Networks

**Authors**: *Carles Domingo-Enrich, Alberto Bietti, Eric Vanden-Eijnden, Joan Bruna*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/domingo-enrich21a.html](http://proceedings.mlr.press/v139/domingo-enrich21a.html)

**Abstract**:

Energy-based models (EBMs) are a simple yet powerful framework for generative modeling. They are based on a trainable energy function which defines an associated Gibbs measure, and they can be trained and sampled from via well-established statistical tools, such as MCMC. Neural networks may be used as energy function approximators, providing both a rich class of expressive models as well as a flexible device to incorporate data structure. In this work we focus on shallow neural networks. Building from the incipient theory of overparametrized neural networks, we show that models trained in the so-called ’active’ regime provide a statistical advantage over their associated ’lazy’ or kernel regime, leading to improved adaptivity to hidden low-dimensional structure in the data distribution, as already observed in supervised learning. Our study covers both the maximum likelihood and Stein Discrepancy estimators, and we validate our theoretical results with numerical experiments on synthetic data.

----

## [254] Kernel-Based Reinforcement Learning: A Finite-Time Analysis

**Authors**: *Omar Darwiche Domingues, Pierre Ménard, Matteo Pirotta, Emilie Kaufmann, Michal Valko*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/domingues21a.html](http://proceedings.mlr.press/v139/domingues21a.html)

**Abstract**:

We consider the exploration-exploitation dilemma in finite-horizon reinforcement learning problems whose state-action space is endowed with a metric. We introduce Kernel-UCBVI, a model-based optimistic algorithm that leverages the smoothness of the MDP and a non-parametric kernel estimator of the rewards and transitions to efficiently balance exploration and exploitation. For problems with $K$ episodes and horizon $H$, we provide a regret bound of $\widetilde{O}\left( H^3 K^{\frac{2d}{2d+1}}\right)$, where $d$ is the covering dimension of the joint state-action space. This is the first regret bound for kernel-based RL using smoothing kernels, which requires very weak assumptions on the MDP and applies to a wide range of tasks. We empirically validate our approach in continuous MDPs with sparse rewards.

----

## [255] Attention is not all you need: pure attention loses rank doubly exponentially with depth

**Authors**: *Yihe Dong, Jean-Baptiste Cordonnier, Andreas Loukas*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dong21a.html](http://proceedings.mlr.press/v139/dong21a.html)

**Abstract**:

Attention-based architectures have become ubiquitous in machine learning. Yet, our understanding of the reasons for their effectiveness remains limited. This work proposes a new way to understand self-attention networks: we show that their output can be decomposed into a sum of smaller terms—or paths—each involving the operation of a sequence of attention heads across layers. Using this path decomposition, we prove that self-attention possesses a strong inductive bias towards "token uniformity". Specifically, without skip connections or multi-layer perceptrons (MLPs), the output converges doubly exponentially to a rank-1 matrix. On the other hand, skip connections and MLPs stop the output from degeneration. Our experiments verify the convergence results on standard transformer architectures.

----

## [256] How rotational invariance of common kernels prevents generalization in high dimensions

**Authors**: *Konstantin Donhauser, Mingqi Wu, Fanny Yang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/donhauser21a.html](http://proceedings.mlr.press/v139/donhauser21a.html)

**Abstract**:

Kernel ridge regression is well-known to achieve minimax optimal rates in low-dimensional settings. However, its behavior in high dimensions is much less understood. Recent work establishes consistency for high-dimensional kernel regression for a number of specific assumptions on the data distribution. In this paper, we show that in high dimensions, the rotational invariance property of commonly studied kernels (such as RBF, inner product kernels and fully-connected NTK of any depth) leads to inconsistent estimation unless the ground truth is a low-degree polynomial. Our lower bound on the generalization error holds for a wide range of distributions and kernels with different eigenvalue decays. This lower bound suggests that consistency results for kernel ridge regression in high dimensions generally require a more refined analysis that depends on the structure of the kernel beyond its eigenvalue decay.

----

## [257] Fast Stochastic Bregman Gradient Methods: Sharp Analysis and Variance Reduction

**Authors**: *Radu-Alexandru Dragomir, Mathieu Even, Hadrien Hendrikx*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dragomir21a.html](http://proceedings.mlr.press/v139/dragomir21a.html)

**Abstract**:

We study the problem of minimizing a relatively-smooth convex function using stochastic Bregman gradient methods. We first prove the convergence of Bregman Stochastic Gradient Descent (BSGD) to a region that depends on the noise (magnitude of the gradients) at the optimum. In particular, BSGD quickly converges to the exact minimizer when this noise is zero (interpolation setting, in which the data is fit perfectly). Otherwise, when the objective has a finite sum structure, we show that variance reduction can be used to counter the effect of noise. In particular, fast convergence to the exact minimizer can be obtained under additional regularity assumptions on the Bregman reference function. We illustrate the effectiveness of our approach on two key applications of relative smoothness: tomographic reconstruction with Poisson noise and statistical preconditioning for distributed optimization.

----

## [258] Bilinear Classes: A Structural Framework for Provable Generalization in RL

**Authors**: *Simon S. Du, Sham M. Kakade, Jason D. Lee, Shachar Lovett, Gaurav Mahajan, Wen Sun, Ruosong Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/du21a.html](http://proceedings.mlr.press/v139/du21a.html)

**Abstract**:

This work introduces Bilinear Classes, a new structural framework, which permit generalization in reinforcement learning in a wide variety of settings through the use of function approximation. The framework incorporates nearly all existing models in which a polynomial sample complexity is achievable, and, notably, also includes new models, such as the Linear Q*/V* model in which both the optimal Q-function and the optimal V-function are linear in some known feature space. Our main result provides an RL algorithm which has polynomial sample complexity for Bilinear Classes; notably, this sample complexity is stated in terms of a reduction to the generalization error of an underlying supervised learning sub-problem. These bounds nearly match the best known sample complexity bounds for existing models. Furthermore, this framework also extends to the infinite dimensional (RKHS) setting: for the the Linear Q*/V* model, linear MDPs, and linear mixture MDPs, we provide sample complexities that have no explicit dependence on the explicit feature dimension (which could be infinite), but instead depends only on information theoretic quantities.

----

## [259] Improved Contrastive Divergence Training of Energy-Based Models

**Authors**: *Yilun Du, Shuang Li, Joshua B. Tenenbaum, Igor Mordatch*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/du21b.html](http://proceedings.mlr.press/v139/du21b.html)

**Abstract**:

Contrastive divergence is a popular method of training energy-based models, but is known to have difficulties with training stability. We propose an adaptation to improve contrastive divergence training by scrutinizing a gradient term that is difficult to calculate and is often left out for convenience. We show that this gradient term is numerically significant and in practice is important to avoid training instabilities, while being tractable to estimate. We further highlight how data augmentation and multi-scale processing can be used to improve model robustness and generation quality. Finally, we empirically evaluate stability of model architectures and show improved performance on a host of benchmarks and use cases, such as image generation, OOD detection, and compositional generation.

----

## [260] Order-Agnostic Cross Entropy for Non-Autoregressive Machine Translation

**Authors**: *Cunxiao Du, Zhaopeng Tu, Jing Jiang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/du21c.html](http://proceedings.mlr.press/v139/du21c.html)

**Abstract**:

We propose a new training objective named order-agnostic cross entropy (OaXE) for fully non-autoregressive translation (NAT) models. OaXE improves the standard cross-entropy loss to ameliorate the effect of word reordering, which is a common source of the critical multimodality problem in NAT. Concretely, OaXE removes the penalty for word order errors, and computes the cross entropy loss based on the best possible alignment between model predictions and target tokens. Since the log loss is very sensitive to invalid references, we leverage cross entropy initialization and loss truncation to ensure the model focuses on a good part of the search space. Extensive experiments on major WMT benchmarks demonstrate that OaXE substantially improves translation performance, setting new state of the art for fully NAT models. Further analyses show that OaXE indeed alleviates the multimodality problem by reducing token repetitions and increasing prediction confidence. Our code, data, and trained models are available at https://github.com/tencent-ailab/ICML21_OAXE.

----

## [261] Putting the "Learning" into Learning-Augmented Algorithms for Frequency Estimation

**Authors**: *Elbert Du, Franklyn Wang, Michael Mitzenmacher*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/du21d.html](http://proceedings.mlr.press/v139/du21d.html)

**Abstract**:

In learning-augmented algorithms, algorithms are enhanced using information from a machine learning algorithm. In turn, this suggests that we should tailor our machine-learning approach for the target algorithm. We here consider this synergy in the context of the learned count-min sketch from (Hsu et al., 2019). Learning here is used to predict heavy hitters from a data stream, which are counted explicitly outside the sketch. We show that an approximately sufficient statistic for the performance of the underlying count-min sketch is given by the coverage of the predictor, or the normalized $L^1$ norm of keys that are filtered by the predictor to be explicitly counted. We show that machine learning models which are trained to optimize for coverage lead to large improvements in performance over prior approaches according to the average absolute frequency error. Our source code can be found at https://github.com/franklynwang/putting-the-learning-in-LAA.

----

## [262] Estimating α-Rank from A Few Entries with Low Rank Matrix Completion

**Authors**: *Yali Du, Xue Yan, Xu Chen, Jun Wang, Haifeng Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/du21e.html](http://proceedings.mlr.press/v139/du21e.html)

**Abstract**:

Multi-agent evaluation aims at the assessment of an agent’s strategy on the basis of interaction with others. Typically, existing methods such as $\alpha$-rank and its approximation still require to exhaustively compare all pairs of joint strategies for an accurate ranking, which in practice is computationally expensive. In this paper, we aim to reduce the number of pairwise comparisons in recovering a satisfying ranking for $n$ strategies in two-player meta-games, by exploring the fact that agents with similar skills may achieve similar payoffs against others. Two situations are considered: the first one is when we can obtain the true payoffs; the other one is when we can only access noisy payoff. Based on these formulations, we leverage low-rank matrix completion and design two novel algorithms for noise-free and noisy evaluations respectively. For both of these settings, we theorize that $O(nr \log n)$ ($n$ is the number of agents and $r$ is the rank of the payoff matrix) payoff entries are required to achieve sufficiently well strategy evaluation performance. Empirical results on evaluating the strategies in three synthetic games and twelve real world games demonstrate that strategy evaluation from a few entries can lead to comparable performance to algorithms with full knowledge of the payoff matrix.

----

## [263] Learning Diverse-Structured Networks for Adversarial Robustness

**Authors**: *Xuefeng Du, Jingfeng Zhang, Bo Han, Tongliang Liu, Yu Rong, Gang Niu, Junzhou Huang, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/du21f.html](http://proceedings.mlr.press/v139/du21f.html)

**Abstract**:

In adversarial training (AT), the main focus has been the objective and optimizer while the model has been less studied, so that the models being used are still those classic ones in standard training (ST). Classic network architectures (NAs) are generally worse than searched NA in ST, which should be the same in AT. In this paper, we argue that NA and AT cannot be handled independently, since given a dataset, the optimal NA in ST would be no longer optimal in AT. That being said, AT is time-consuming itself; if we directly search NAs in AT over large search spaces, the computation will be practically infeasible. Thus, we propose diverse-structured network (DS-Net), to significantly reduce the size of the search space: instead of low-level operations, we only consider predefined atomic blocks, where an atomic block is a time-tested building block like the residual block. There are only a few atomic blocks and thus we can weight all atomic blocks rather than find the best one in a searched block of DS-Net, which is an essential tradeoff between exploring diverse structures and exploiting the best structures. Empirical results demonstrate the advantages of DS-Net, i.e., weighting the atomic blocks.

----

## [264] Risk Bounds and Rademacher Complexity in Batch Reinforcement Learning

**Authors**: *Yaqi Duan, Chi Jin, Zhiyuan Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/duan21a.html](http://proceedings.mlr.press/v139/duan21a.html)

**Abstract**:

This paper considers batch Reinforcement Learning (RL) with general value function approximation. Our study investigates the minimal assumptions to reliably estimate/minimize Bellman error, and characterizes the generalization performance by (local) Rademacher complexities of general function classes, which makes initial steps in bridging the gap between statistical learning theory and batch RL. Concretely, we view the Bellman error as a surrogate loss for the optimality gap, and prove the followings: (1) In double sampling regime, the excess risk of Empirical Risk Minimizer (ERM) is bounded by the Rademacher complexity of the function class. (2) In the single sampling regime, sample-efficient risk minimization is not possible without further assumptions, regardless of algorithms. However, with completeness assumptions, the excess risk of FQI and a minimax style algorithm can be again bounded by the Rademacher complexity of the corresponding function classes. (3) Fast statistical rates can be achieved by using tools of local Rademacher complexity. Our analysis covers a wide range of function classes, including finite classes, linear spaces, kernel spaces, sparse linear features, etc.

----

## [265] Sawtooth Factorial Topic Embeddings Guided Gamma Belief Network

**Authors**: *Zhibin Duan, Dongsheng Wang, Bo Chen, Chaojie Wang, Wenchao Chen, Yewen Li, Jie Ren, Mingyuan Zhou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/duan21b.html](http://proceedings.mlr.press/v139/duan21b.html)

**Abstract**:

Hierarchical topic models such as the gamma belief network (GBN) have delivered promising results in mining multi-layer document representations and discovering interpretable topic taxonomies. However, they often assume in the prior that the topics at each layer are independently drawn from the Dirichlet distribution, ignoring the dependencies between the topics both at the same layer and across different layers. To relax this assumption, we propose sawtooth factorial topic embedding guided GBN, a deep generative model of documents that captures the dependencies and semantic similarities between the topics in the embedding space. Specifically, both the words and topics are represented as embedding vectors of the same dimension. The topic matrix at a layer is factorized into the product of a factor loading matrix and a topic embedding matrix, the transpose of which is set as the factor loading matrix of the layer above. Repeating this particular type of factorization, which shares components between adjacent layers, leads to a structure referred to as sawtooth factorization. An auto-encoding variational inference network is constructed to optimize the model parameter via stochastic gradient descent. Experiments on big corpora show that our models outperform other neural topic models on extracting deeper interpretable topics and deriving better document representations.

----

## [266] Exponential Reduction in Sample Complexity with Learning of Ising Model Dynamics

**Authors**: *Arkopal Dutt, Andrey Y. Lokhov, Marc Vuffray, Sidhant Misra*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/dutt21a.html](http://proceedings.mlr.press/v139/dutt21a.html)

**Abstract**:

The usual setting for learning the structure and parameters of a graphical model assumes the availability of independent samples produced from the corresponding multivariate probability distribution. However, for many models the mixing time of the respective Markov chain can be very large and i.i.d. samples may not be obtained. We study the problem of reconstructing binary graphical models from correlated samples produced by a dynamical process, which is natural in many applications. We analyze the sample complexity of two estimators that are based on the interaction screening objective and the conditional likelihood loss. We observe that for samples coming from a dynamical process far from equilibrium, the sample complexity reduces exponentially compared to a dynamical process that mixes quickly.

----

## [267] Reinforcement Learning Under Moral Uncertainty

**Authors**: *Adrien Ecoffet, Joel Lehman*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ecoffet21a.html](http://proceedings.mlr.press/v139/ecoffet21a.html)

**Abstract**:

An ambitious goal for machine learning is to create agents that behave ethically: The capacity to abide by human moral norms would greatly expand the context in which autonomous agents could be practically and safely deployed, e.g. fully autonomous vehicles will encounter charged moral decisions that complicate their deployment. While ethical agents could be trained by rewarding correct behavior under a specific moral theory (e.g. utilitarianism), there remains widespread disagreement about the nature of morality. Acknowledging such disagreement, recent work in moral philosophy proposes that ethical behavior requires acting under moral uncertainty, i.e. to take into account when acting that one’s credence is split across several plausible ethical theories. This paper translates such insights to the field of reinforcement learning, proposes two training methods that realize different points among competing desiderata, and trains agents in simple environments to act under moral uncertainty. The results illustrate (1) how such uncertainty can help curb extreme behavior from commitment to single theories and (2) several technical complications arising from attempting to ground moral philosophy in RL (e.g. how can a principled trade-off between two competing but incomparable reward functions be reached). The aim is to catalyze progress towards morally-competent agents and highlight the potential of RL to contribute towards the computational grounding of moral philosophy.

----

## [268] Confidence-Budget Matching for Sequential Budgeted Learning

**Authors**: *Yonathan Efroni, Nadav Merlis, Aadirupa Saha, Shie Mannor*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/efroni21a.html](http://proceedings.mlr.press/v139/efroni21a.html)

**Abstract**:

A core element in decision-making under uncertainty is the feedback on the quality of the performed actions. However, in many applications, such feedback is restricted. For example, in recommendation systems, repeatedly asking the user to provide feedback on the quality of recommendations will annoy them. In this work, we formalize decision-making problems with querying budget, where there is a (possibly time-dependent) hard limit on the number of reward queries allowed. Specifically, we focus on multi-armed bandits, linear contextual bandits, and reinforcement learning problems. We start by analyzing the performance of ‘greedy’ algorithms that query a reward whenever they can. We show that in fully stochastic settings, doing so performs surprisingly well, but in the presence of any adversity, this might lead to linear regret. To overcome this issue, we propose the Confidence-Budget Matching (CBM) principle that queries rewards when the confidence intervals are wider than the inverse square root of the available budget. We analyze the performance of CBM based algorithms in different settings and show that it performs well in the presence of adversity in the contexts, initial states, and budgets.

----

## [269] Self-Paced Context Evaluation for Contextual Reinforcement Learning

**Authors**: *Theresa Eimer, André Biedenkapp, Frank Hutter, Marius Lindauer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/eimer21a.html](http://proceedings.mlr.press/v139/eimer21a.html)

**Abstract**:

Reinforcement learning (RL) has made a lot of advances for solving a single problem in a given environment; but learning policies that generalize to unseen variations of a problem remains challenging. To improve sample efficiency for learning on such instances of a problem domain, we present Self-Paced Context Evaluation (SPaCE). Based on self-paced learning, SPaCE automatically generates instance curricula online with little computational overhead. To this end, SPaCE leverages information contained in state values during training to accelerate and improve training performance as well as generalization capabilities to new \tasks from the same problem domain. Nevertheless, SPaCE is independent of the problem domain at hand and can be applied on top of any RL agent with state-value function approximation. We demonstrate SPaCE’s ability to speed up learning of different value-based RL agents on two environments, showing better generalization capabilities and up to 10x faster learning compared to naive approaches such as round robin or SPDRL, as the closest state-of-the-art approach.

----

## [270] Provably Strict Generalisation Benefit for Equivariant Models

**Authors**: *Bryn Elesedy, Sheheryar Zaidi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/elesedy21a.html](http://proceedings.mlr.press/v139/elesedy21a.html)

**Abstract**:

It is widely believed that engineering a model to be invariant/equivariant improves generalisation. Despite the growing popularity of this approach, a precise characterisation of the generalisation benefit is lacking. By considering the simplest case of linear models, this paper provides the first provably non-zero improvement in generalisation for invariant/equivariant models when the target distribution is invariant/equivariant with respect to a compact group. Moreover, our work reveals an interesting relationship between generalisation, the number of training examples and properties of the group action. Our results rest on an observation of the structure of function spaces under averaging operators which, along with its consequences for feature averaging, may be of independent interest.

----

## [271] Efficient Iterative Amortized Inference for Learning Symmetric and Disentangled Multi-Object Representations

**Authors**: *Patrick Emami, Pan He, Sanjay Ranka, Anand Rangarajan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/emami21a.html](http://proceedings.mlr.press/v139/emami21a.html)

**Abstract**:

Unsupervised multi-object representation learning depends on inductive biases to guide the discovery of object-centric representations that generalize. However, we observe that methods for learning these representations are either impractical due to long training times and large memory consumption or forego key inductive biases. In this work, we introduce EfficientMORL, an efficient framework for the unsupervised learning of object-centric representations. We show that optimization challenges caused by requiring both symmetry and disentanglement can in fact be addressed by high-cost iterative amortized inference by designing the framework to minimize its dependence on it. We take a two-stage approach to inference: first, a hierarchical variational autoencoder extracts symmetric and disentangled representations through bottom-up inference, and second, a lightweight network refines the representations with top-down feedback. The number of refinement steps taken during training is reduced following a curriculum, so that at test time with zero steps the model achieves 99.1% of the refined decomposition performance. We demonstrate strong object decomposition and disentanglement on the standard multi-object benchmark while achieving nearly an order of magnitude faster training and test time inference over the previous state-of-the-art model.

----

## [272] Implicit Bias of Linear RNNs

**Authors**: *Melikasadat Emami, Mojtaba Sahraee-Ardakan, Parthe Pandit, Sundeep Rangan, Alyson K. Fletcher*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/emami21b.html](http://proceedings.mlr.press/v139/emami21b.html)

**Abstract**:

Contemporary wisdom based on empirical studies suggests that standard recurrent neural networks (RNNs) do not perform well on tasks requiring long-term memory. However, RNNs’ poor ability to capture long-term dependencies has not been fully understood. This paper provides a rigorous explanation of this property in the special case of linear RNNs. Although this work is limited to linear RNNs, even these systems have traditionally been difficult to analyze due to their non-linear parameterization. Using recently-developed kernel regime analysis, our main result shows that as the number of hidden units goes to infinity, linear RNNs learned from random initializations are functionally equivalent to a certain weighted 1D-convolutional network. Importantly, the weightings in the equivalent model cause an implicit bias to elements with smaller time lags in the convolution, and hence shorter memory. The degree of this bias depends on the variance of the transition matrix at initialization and is related to the classic exploding and vanishing gradients problem. The theory is validated with both synthetic and real data experiments.

----

## [273] Global Optimality Beyond Two Layers: Training Deep ReLU Networks via Convex Programs

**Authors**: *Tolga Ergen, Mert Pilanci*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ergen21a.html](http://proceedings.mlr.press/v139/ergen21a.html)

**Abstract**:

Understanding the fundamental mechanism behind the success of deep neural networks is one of the key challenges in the modern machine learning literature. Despite numerous attempts, a solid theoretical analysis is yet to be developed. In this paper, we develop a novel unified framework to reveal a hidden regularization mechanism through the lens of convex optimization. We first show that the training of multiple three-layer ReLU sub-networks with weight decay regularization can be equivalently cast as a convex optimization problem in a higher dimensional space, where sparsity is enforced via a group $\ell_1$-norm regularization. Consequently, ReLU networks can be interpreted as high dimensional feature selection methods. More importantly, we then prove that the equivalent convex problem can be globally optimized by a standard convex optimization solver with a polynomial-time complexity with respect to the number of samples and data dimension when the width of the network is fixed. Finally, we numerically validate our theoretical results via experiments involving both synthetic and real datasets.

----

## [274] Revealing the Structure of Deep Neural Networks via Convex Duality

**Authors**: *Tolga Ergen, Mert Pilanci*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ergen21b.html](http://proceedings.mlr.press/v139/ergen21b.html)

**Abstract**:

We study regularized deep neural networks (DNNs) and introduce a convex analytic framework to characterize the structure of the hidden layers. We show that a set of optimal hidden layer weights for a norm regularized DNN training problem can be explicitly found as the extreme points of a convex set. For the special case of deep linear networks, we prove that each optimal weight matrix aligns with the previous layers via duality. More importantly, we apply the same characterization to deep ReLU networks with whitened data and prove the same weight alignment holds. As a corollary, we also prove that norm regularized deep ReLU networks yield spline interpolation for one-dimensional datasets which was previously known only for two-layer networks. Furthermore, we provide closed-form solutions for the optimal layer weights when data is rank-one or whitened. The same analysis also applies to architectures with batch normalization even for arbitrary data. Therefore, we obtain a complete explanation for a recent empirical observation termed Neural Collapse where class means collapse to the vertices of a simplex equiangular tight frame.

----

## [275] Whitening for Self-Supervised Representation Learning

**Authors**: *Aleksandr Ermolov, Aliaksandr Siarohin, Enver Sangineto, Nicu Sebe*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ermolov21a.html](http://proceedings.mlr.press/v139/ermolov21a.html)

**Abstract**:

Most of the current self-supervised representation learning (SSL) methods are based on the contrastive loss and the instance-discrimination task, where augmented versions of the same image instance ("positives") are contrasted with instances extracted from other images ("negatives"). For the learning to be effective, many negatives should be compared with a positive pair, which is computationally demanding. In this paper, we propose a different direction and a new loss function for SSL, which is based on the whitening of the latent-space features. The whitening operation has a "scattering" effect on the batch samples, avoiding degenerate solutions where all the sample representations collapse to a single point. Our solution does not require asymmetric networks and it is conceptually simple. Moreover, since negatives are not needed, we can extract multiple positive pairs from the same image instance. The source code of the method and of all the experiments is available at: https://github.com/htdt/self-supervised.

----

## [276] Graph Mixture Density Networks

**Authors**: *Federico Errica, Davide Bacciu, Alessio Micheli*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/errica21a.html](http://proceedings.mlr.press/v139/errica21a.html)

**Abstract**:

We introduce the Graph Mixture Density Networks, a new family of machine learning models that can fit multimodal output distributions conditioned on graphs of arbitrary topology. By combining ideas from mixture models and graph representation learning, we address a broader class of challenging conditional density estimation problems that rely on structured data. In this respect, we evaluate our method on a new benchmark application that leverages random graphs for stochastic epidemic simulations. We show a significant improvement in the likelihood of epidemic outcomes when taking into account both multimodality and structure. The empirical analysis is complemented by two real-world regression tasks showing the effectiveness of our approach in modeling the output prediction uncertainty. Graph Mixture Density Networks open appealing research opportunities in the study of structure-dependent phenomena that exhibit non-trivial conditional output distributions.

----

## [277] Cross-Gradient Aggregation for Decentralized Learning from Non-IID Data

**Authors**: *Yasaman Esfandiari, Sin Yong Tan, Zhanhong Jiang, Aditya Balu, Ethan Herron, Chinmay Hegde, Soumik Sarkar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/esfandiari21a.html](http://proceedings.mlr.press/v139/esfandiari21a.html)

**Abstract**:

Decentralized learning enables a group of collaborative agents to learn models using a distributed dataset without the need for a central parameter server. Recently, decentralized learning algorithms have demonstrated state-of-the-art results on benchmark data sets, comparable with centralized algorithms. However, the key assumption to achieve competitive performance is that the data is independently and identically distributed (IID) among the agents which, in real-life applications, is often not applicable. Inspired by ideas from continual learning, we propose Cross-Gradient Aggregation (CGA), a novel decentralized learning algorithm where (i) each agent aggregates cross-gradient information, i.e., derivatives of its model with respect to its neighbors’ datasets, and (ii) updates its model using a projected gradient based on quadratic programming (QP). We theoretically analyze the convergence characteristics of CGA and demonstrate its efficiency on non-IID data distributions sampled from the MNIST and CIFAR-10 datasets. Our empirical comparisons show superior learning performance of CGA over existing state-of-the-art decentralized learning algorithms, as well as maintaining the improved performance under information compression to reduce peer-to-peer communication overhead. The code is available here on GitHub.

----

## [278] Weight-covariance alignment for adversarially robust neural networks

**Authors**: *Panagiotis Eustratiadis, Henry Gouk, Da Li, Timothy M. Hospedales*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/eustratiadis21a.html](http://proceedings.mlr.press/v139/eustratiadis21a.html)

**Abstract**:

Stochastic Neural Networks (SNNs) that inject noise into their hidden layers have recently been shown to achieve strong robustness against adversarial attacks. However, existing SNNs are usually heuristically motivated, and often rely on adversarial training, which is computationally costly. We propose a new SNN that achieves state-of-the-art performance without relying on adversarial training, and enjoys solid theoretical justification. Specifically, while existing SNNs inject learned or hand-tuned isotropic noise, our SNN learns an anisotropic noise distribution to optimize a learning-theoretic bound on adversarial robustness. We evaluate our method on a number of popular benchmarks, show that it can be applied to different architectures, and that it provides robustness to a variety of white-box and black-box attacks, while being simple and fast to train compared to existing alternatives.

----

## [279] Data augmentation for deep learning based accelerated MRI reconstruction with limited data

**Authors**: *Zalan Fabian, Reinhard Heckel, Mahdi Soltanolkotabi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fabian21a.html](http://proceedings.mlr.press/v139/fabian21a.html)

**Abstract**:

Deep neural networks have emerged as very successful tools for image restoration and reconstruction tasks. These networks are often trained end-to-end to directly reconstruct an image from a noisy or corrupted measurement of that image. To achieve state-of-the-art performance, training on large and diverse sets of images is considered critical. However, it is often difficult and/or expensive to collect large amounts of training images. Inspired by the success of Data Augmentation (DA) for classification problems, in this paper, we propose a pipeline for data augmentation for accelerated MRI reconstruction and study its effectiveness at reducing the required training data in a variety of settings. Our DA pipeline, MRAugment, is specifically designed to utilize the invariances present in medical imaging measurements as naive DA strategies that neglect the physics of the problem fail. Through extensive studies on multiple datasets we demonstrate that in the low-data regime DA prevents overfitting and can match or even surpass the state of the art while using significantly fewer training data, whereas in the high-data regime it has diminishing returns. Furthermore, our findings show that DA improves the robustness of the model against various shifts in the test distribution.

----

## [280] Poisson-Randomised DirBN: Large Mutation is Needed in Dirichlet Belief Networks

**Authors**: *Xuhui Fan, Bin Li, Yaqiong Li, Scott A. Sisson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fan21a.html](http://proceedings.mlr.press/v139/fan21a.html)

**Abstract**:

The Dirichlet Belief Network (DirBN) was recently proposed as a promising deep generative model to learn interpretable deep latent distributions for objects. However, its current representation capability is limited since its latent distributions across different layers is prone to form similar patterns and can thus hardly use multi-layer structure to form flexible distributions. In this work, we propose Poisson-randomised Dirichlet Belief Networks (Pois-DirBN), which allows large mutations for the latent distributions across layers to enlarge the representation capability. Based on our key idea of inserting Poisson random variables in the layer-wise connection, Pois-DirBN first introduces a component-wise propagation mechanism to enable latent distributions to have large variations across different layers. Then, we develop a layer-wise Gibbs sampling algorithm to infer the latent distributions, leading to a larger number of effective layers compared to DirBN. In addition, we integrate out latent distributions and form a multi-stochastic deep integer network, which provides an alternative view on Pois-DirBN. We apply Pois-DirBN to relational modelling and validate its effectiveness through improved link prediction performance and more interpretable latent distribution visualisations. The code can be downloaded at https://github.com/xuhuifan/Pois_DirBN.

----

## [281] Model-based Reinforcement Learning for Continuous Control with Posterior Sampling

**Authors**: *Ying Fan, Yifei Ming*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fan21b.html](http://proceedings.mlr.press/v139/fan21b.html)

**Abstract**:

Balancing exploration and exploitation is crucial in reinforcement learning (RL). In this paper, we study model-based posterior sampling for reinforcement learning (PSRL) in continuous state-action spaces theoretically and empirically. First, we show the first regret bound of PSRL in continuous spaces which is polynomial in the episode length to the best of our knowledge. With the assumption that reward and transition functions can be modeled by Bayesian linear regression, we develop a regret bound of $\tilde{O}(H^{3/2}d\sqrt{T})$, where $H$ is the episode length, $d$ is the dimension of the state-action space, and $T$ indicates the total time steps. This result matches the best-known regret bound of non-PSRL methods in linear MDPs. Our bound can be extended to nonlinear cases as well with feature embedding: using linear kernels on the feature representation $\phi$, the regret bound becomes $\tilde{O}(H^{3/2}d_{\phi}\sqrt{T})$, where $d_\phi$ is the dimension of the representation space. Moreover, we present MPC-PSRL, a model-based posterior sampling algorithm with model predictive control for action selection. To capture the uncertainty in models, we use Bayesian linear regression on the penultimate layer (the feature representation layer $\phi$) of neural networks. Empirical results show that our algorithm achieves the state-of-the-art sample efficiency in benchmark continuous control tasks compared to prior model-based algorithms, and matches the asymptotic performance of model-free algorithms.

----

## [282] SECANT: Self-Expert Cloning for Zero-Shot Generalization of Visual Policies

**Authors**: *Linxi Fan, Guanzhi Wang, De-An Huang, Zhiding Yu, Li Fei-Fei, Yuke Zhu, Animashree Anandkumar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fan21c.html](http://proceedings.mlr.press/v139/fan21c.html)

**Abstract**:

Generalization has been a long-standing challenge for reinforcement learning (RL). Visual RL, in particular, can be easily distracted by irrelevant factors in high-dimensional observation space. In this work, we consider robust policy learning which targets zero-shot generalization to unseen visual environments with large distributional shift. We propose SECANT, a novel self-expert cloning technique that leverages image augmentation in two stages to *decouple* robust representation learning from policy optimization. Specifically, an expert policy is first trained by RL from scratch with weak augmentations. A student network then learns to mimic the expert policy by supervised learning with strong augmentations, making its representation more robust against visual variations compared to the expert. Extensive experiments demonstrate that SECANT significantly advances the state of the art in zero-shot generalization across 4 challenging domains. Our average reward improvements over prior SOTAs are: DeepMind Control (+26.5%), robotic manipulation (+337.8%), vision-based autonomous driving (+47.7%), and indoor object navigation (+15.8%). Code release and video are available at https://linxifan.github.io/secant-site/.

----

## [283] On Estimation in Latent Variable Models

**Authors**: *Guanhua Fang, Ping Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fang21a.html](http://proceedings.mlr.press/v139/fang21a.html)

**Abstract**:

Latent variable models have been playing a central role in statistics, econometrics, machine learning with applications to repeated observation study, panel data inference, user behavior analysis, etc. In many modern applications, the inference based on latent variable models involves one or several of the following features: the presence of complex latent structure, the observed and latent variables being continuous or discrete, constraints on parameters, and data size being large. Therefore, solving an estimation problem for general latent variable models is highly non-trivial. In this paper, we consider a gradient based method via using variance reduction technique to accelerate estimation procedure. Theoretically, we show the convergence results for the proposed method under general and mild model assumptions. The algorithm has better computational complexity compared with the classical gradient methods and maintains nice statistical properties. Various numerical results corroborate our theory.

----

## [284] On Variational Inference in Biclustering Models

**Authors**: *Guanhua Fang, Ping Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fang21b.html](http://proceedings.mlr.press/v139/fang21b.html)

**Abstract**:

Biclustering structures exist ubiquitously in data matrices and the biclustering problem was first formalized by John Hartigan (1972) to cluster rows and columns simultaneously. In this paper, we develop a theory for the estimation of general biclustering models, where the data is assumed to follow certain statistical distribution with underlying biclustering structure. Due to the existence of latent variables, directly computing the maximal likelihood estimator is prohibitively difficult in practice and we instead consider the variational inference (VI) approach to solve the parameter estimation problem. Although variational inference method generally has good empirical performance, there are very few theoretical results around VI. In this paper, we obtain the precise estimation bound of variational estimator and show that it matches the minimax rate in terms of estimation error under mild assumptions in biclustering setting. Furthermore, we study the convergence property of the coordinate ascent variational inference algorithm, where both local and global convergence results have been provided. Numerical results validate our new theories.

----

## [285] Learning Bounds for Open-Set Learning

**Authors**: *Zhen Fang, Jie Lu, Anjin Liu, Feng Liu, Guangquan Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fang21c.html](http://proceedings.mlr.press/v139/fang21c.html)

**Abstract**:

Traditional supervised learning aims to train a classifier in the closed-set world, where training and test samples share the same label space. In this paper, we target a more challenging and re_x0002_alistic setting: open-set learning (OSL), where there exist test samples from the classes that are unseen during training. Although researchers have designed many methods from the algorith_x0002_mic perspectives, there are few methods that pro_x0002_vide generalization guarantees on their ability to achieve consistent performance on different train_x0002_ing samples drawn from the same distribution. Motivated by the transfer learning and probably approximate correct (PAC) theory, we make a bold attempt to study OSL by proving its general_x0002_ization error-given training samples with size n, the estimation error will get close to order Op(1/$\sqrt{}$n). This is the first study to provide a generalization bound for OSL, which we do by theoretically investigating the risk of the tar_x0002_get classifier on unknown classes. According to our theory, a novel algorithm, called auxiliary open-set risk (AOSR) is proposed to address the OSL problem. Experiments verify the efficacy of AOSR. The code is available at github.com/AnjinLiu/Openset_Learning_AOSR.

----

## [286] Streaming Bayesian Deep Tensor Factorization

**Authors**: *Shikai Fang, Zheng Wang, Zhimeng Pan, Ji Liu, Shandian Zhe*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fang21d.html](http://proceedings.mlr.press/v139/fang21d.html)

**Abstract**:

Despite the success of existing tensor factorization methods, most of them conduct a multilinear decomposition, and rarely exploit powerful modeling frameworks, like deep neural networks, to capture a variety of complicated interactions in data. More important, for highly expressive, deep factorization, we lack an effective approach to handle streaming data, which are ubiquitous in real-world applications. To address these issues, we propose SBTD, a Streaming Bayesian Deep Tensor factorization method. We first use Bayesian neural networks (NNs) to build a deep tensor factorization model. We assign a spike-and-slab prior over each NN weight to encourage sparsity and to prevent overfitting. We then use multivariate Delta’s method and moment matching to approximate the posterior of the NN output and calculate the running model evidence, based on which we develop an efficient streaming posterior inference algorithm in the assumed-density-filtering and expectation propagation framework. Our algorithm provides responsive incremental updates for the posterior of the latent factors and NN weights upon receiving newly observed tensor entries, and meanwhile identify and inhibit redundant/useless weights. We show the advantages of our approach in four real-world applications.

----

## [287] PID Accelerated Value Iteration Algorithm

**Authors**: *Amir Massoud Farahmand, Mohammad Ghavamzadeh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/farahmand21a.html](http://proceedings.mlr.press/v139/farahmand21a.html)

**Abstract**:

The convergence rate of Value Iteration (VI), a fundamental procedure in dynamic programming and reinforcement learning, for solving MDPs can be slow when the discount factor is close to one. We propose modifications to VI in order to potentially accelerate its convergence behaviour. The key insight is the realization that the evolution of the value function approximations $(V_k)_{k \geq 0}$ in the VI procedure can be seen as a dynamical system. This opens up the possibility of using techniques from \emph{control theory} to modify, and potentially accelerate, this dynamics. We present such modifications based on simple controllers, such as PD (Proportional-Derivative), PI (Proportional-Integral), and PID. We present the error dynamics of these variants of VI, and provably (for certain classes of MDPs) and empirically (for more general classes) show that the convergence rate can be significantly improved. We also propose a gain adaptation mechanism in order to automatically select the controller gains, and empirically show the effectiveness of this procedure.

----

## [288] Near-Optimal Entrywise Anomaly Detection for Low-Rank Matrices with Sub-Exponential Noise

**Authors**: *Vivek F. Farias, Andrew A. Li, Tianyi Peng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/farias21a.html](http://proceedings.mlr.press/v139/farias21a.html)

**Abstract**:

We study the problem of identifying anomalies in a low-rank matrix observed with sub-exponential noise, motivated by applications in retail and inventory management. State of the art approaches to anomaly detection in low-rank matrices apparently fall short, since they require that non-anomalous entries be observed with vanishingly small noise (which is not the case in our problem, and indeed in many applications). So motivated, we propose a conceptually simple entrywise approach to anomaly detection in low-rank matrices. Our approach accommodates a general class of probabilistic anomaly models. We extend recent work on entrywise error guarantees for matrix completion, establishing such guarantees for sub-exponential matrices, where in addition to missing entries, a fraction of entries are corrupted by (an also unknown) anomaly model. Viewing the anomaly detection as a classification task, to the best of our knowledge, we are the first to achieve the min-max optimal detection rate (up to log factors). Using data from a massive consumer goods retailer, we show that our approach provides significant improvements over incumbent approaches to anomaly detection.

----

## [289] Connecting Optimal Ex-Ante Collusion in Teams to Extensive-Form Correlation: Faster Algorithms and Positive Complexity Results

**Authors**: *Gabriele Farina, Andrea Celli, Nicola Gatti, Tuomas Sandholm*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/farina21a.html](http://proceedings.mlr.press/v139/farina21a.html)

**Abstract**:

We focus on the problem of finding an optimal strategy for a team of players that faces an opponent in an imperfect-information zero-sum extensive-form game. Team members are not allowed to communicate during play but can coordinate before the game. In this setting, it is known that the best the team can do is sample a profile of potentially randomized strategies (one per player) from a joint (a.k.a. correlated) probability distribution at the beginning of the game. In this paper, we first provide new modeling results about computing such an optimal distribution by drawing a connection to a different literature on extensive-form correlation. Second, we provide an algorithm that allows one for capping the number of profiles employed in the solution. This begets an anytime algorithm by increasing the cap. We find that often a handful of well-chosen such profiles suffices to reach optimal utility for the team. This enables team members to reach coordination through a simple and understandable plan. Finally, inspired by this observation and leveraging theoretical concepts that we introduce, we develop an efficient column-generation algorithm for finding an optimal distribution for the team. We evaluate it on a suite of common benchmark games. It is three orders of magnitude faster than the prior state of the art on games that the latter can solve and it can also solve several games that were previously unsolvable.

----

## [290] Train simultaneously, generalize better: Stability of gradient-based minimax learners

**Authors**: *Farzan Farnia, Asuman E. Ozdaglar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/farnia21a.html](http://proceedings.mlr.press/v139/farnia21a.html)

**Abstract**:

The success of minimax learning problems of generative adversarial networks (GANs) has been observed to depend on the minimax optimization algorithm used for their training. This dependence is commonly attributed to the convergence speed and robustness properties of the underlying optimization algorithm. In this paper, we show that the optimization algorithm also plays a key role in the generalization performance of the trained minimax model. To this end, we analyze the generalization properties of standard gradient descent ascent (GDA) and proximal point method (PPM) algorithms through the lens of algorithmic stability as defined by Bousquet & Elisseeff, 2002 under both convex-concave and nonconvex-nonconcave minimax settings. While the GDA algorithm is not guaranteed to have a vanishing excess risk in convex-concave problems, we show the PPM algorithm enjoys a bounded excess risk in the same setup. For nonconvex-nonconcave problems, we compare the generalization performance of stochastic GDA and GDmax algorithms where the latter fully solves the maximization subproblem at every iteration. Our generalization analysis suggests the superiority of GDA provided that the minimization and maximization subproblems are solved simultaneously with similar learning rates. We discuss several numerical results indicating the role of optimization algorithms in the generalization of learned minimax models.

----

## [291] Unbalanced minibatch Optimal Transport; applications to Domain Adaptation

**Authors**: *Kilian Fatras, Thibault Séjourné, Rémi Flamary, Nicolas Courty*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fatras21a.html](http://proceedings.mlr.press/v139/fatras21a.html)

**Abstract**:

Optimal transport distances have found many applications in machine learning for their capacity to compare non-parametric probability distributions. Yet their algorithmic complexity generally prevents their direct use on large scale datasets. Among the possible strategies to alleviate this issue, practitioners can rely on computing estimates of these distances over subsets of data, i.e. minibatches. While computationally appealing, we highlight in this paper some limits of this strategy, arguing it can lead to undesirable smoothing effects. As an alternative, we suggest that the same minibatch strategy coupled with unbalanced optimal transport can yield more robust behaviors. We discuss the associated theoretical properties, such as unbiased estimators, existence of gradients and concentration bounds. Our experimental study shows that in challenging problems associated to domain adaptation, the use of unbalanced optimal transport leads to significantly better results, competing with or surpassing recent baselines.

----

## [292] Risk-Sensitive Reinforcement Learning with Function Approximation: A Debiasing Approach

**Authors**: *Yingjie Fei, Zhuoran Yang, Zhaoran Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fei21a.html](http://proceedings.mlr.press/v139/fei21a.html)

**Abstract**:

We study function approximation for episodic reinforcement learning with entropic risk measure. We first propose an algorithm with linear function approximation. Compared to existing algorithms, which suffer from improper regularization and regression biases, this algorithm features debiasing transformations in backward induction and regression procedures. We further propose an algorithm with general function approximation, which features implicit debiasing transformations. We prove that both algorithms achieve a sublinear regret and demonstrate a trade-off between generality and efficiency. Our analysis provides a unified framework for function approximation in risk-sensitive reinforcement learning, which leads to the first sublinear regret bounds in the setting.

----

## [293] Lossless Compression of Efficient Private Local Randomizers

**Authors**: *Vitaly Feldman, Kunal Talwar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/feldman21a.html](http://proceedings.mlr.press/v139/feldman21a.html)

**Abstract**:

Locally Differentially Private (LDP) Reports are commonly used for collection of statistics and machine learning in the federated setting. In many cases the best known LDP algorithms require sending prohibitively large messages from the client device to the server (such as when constructing histograms over a large domain or learning a high-dimensional model). Here we demonstrate a general approach that, under standard cryptographic assumptions, compresses every efficient LDP algorithm with negligible loss in privacy and utility guarantees. The practical implication of our result is that in typical applications every message can be compressed to the size of the server’s pseudo-random generator seed. From this general approach we derive low-communication algorithms for the problems of frequency estimation and high-dimensional mean estimation. Our algorithms are simpler and more accurate than existing low-communication LDP algorithms for these well-studied problems.

----

## [294] Dimensionality Reduction for the Sum-of-Distances Metric

**Authors**: *Zhili Feng, Praneeth Kacham, David P. Woodruff*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/feng21a.html](http://proceedings.mlr.press/v139/feng21a.html)

**Abstract**:

We give a dimensionality reduction procedure to approximate the sum of distances of a given set of $n$ points in $R^d$ to any “shape” that lies in a $k$-dimensional subspace. Here, by “shape” we mean any set of points in $R^d$. Our algorithm takes an input in the form of an $n \times d$ matrix $A$, where each row of $A$ denotes a data point, and outputs a subspace $P$ of dimension $O(k^{3}/\epsilon^6)$ such that the projections of each of the $n$ points onto the subspace $P$ and the distances of each of the points to the subspace $P$ are sufficient to obtain an $\epsilon$-approximation to the sum of distances to any arbitrary shape that lies in a $k$-dimensional subspace of $R^d$. These include important problems such as $k$-median, $k$-subspace approximation, and $(j,l)$ subspace clustering with $j \cdot l \leq k$. Dimensionality reduction reduces the data storage requirement to $(n+d)k^{3}/\epsilon^6$ from nnz$(A)$. Here nnz$(A)$ could potentially be as large as $nd$. Our algorithm runs in time nnz$(A)/\epsilon^2 + (n+d)$poly$(k/\epsilon)$, up to logarithmic factors. For dense matrices, where nnz$(A) \approx nd$, we give a faster algorithm, that runs in time $nd + (n+d)$poly$(k/\epsilon)$ up to logarithmic factors. Our dimensionality reduction algorithm can also be used to obtain poly$(k/\epsilon)$ size coresets for $k$-median and $(k,1)$-subspace approximation problems in polynomial time.

----

## [295] Reserve Price Optimization for First Price Auctions in Display Advertising

**Authors**: *Zhe Feng, Sébastien Lahaie, Jon Schneider, Jinchao Ye*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/feng21b.html](http://proceedings.mlr.press/v139/feng21b.html)

**Abstract**:

The display advertising industry has recently transitioned from second- to first-price auctions as its primary mechanism for ad allocation and pricing. In light of this, publishers need to re-evaluate and optimize their auction parameters, notably reserve prices. In this paper, we propose a gradient-based algorithm to adaptively update and optimize reserve prices based on estimates of bidders’ responsiveness to experimental shocks in reserves. Our key innovation is to draw on the inherent structure of the revenue objective in order to reduce the variance of gradient estimates and improve convergence rates in both theory and practice. We show that revenue in a first-price auction can be usefully decomposed into a \emph{demand} component and a \emph{bidding} component, and introduce techniques to reduce the variance of each component. We characterize the bias-variance trade-offs of these techniques and validate the performance of our proposed algorithm through experiments on synthetic data and real display ad auctions data from a major ad exchange.

----

## [296] Uncertainty Principles of Encoding GANs

**Authors**: *Ruili Feng, Zhouchen Lin, Jiapeng Zhu, Deli Zhao, Jingren Zhou, Zheng-Jun Zha*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/feng21c.html](http://proceedings.mlr.press/v139/feng21c.html)

**Abstract**:

The compelling synthesis results of Generative Adversarial Networks (GANs) demonstrate rich semantic knowledge in their latent codes. To obtain this knowledge for downstream applications, encoding GANs has been proposed to learn encoders, such that real world data can be encoded to latent codes, which can be fed to generators to reconstruct those data. However, despite the theoretical guarantees of precise reconstruction in previous works, current algorithms generally reconstruct inputs with non-negligible deviations from inputs. In this paper we study this predicament of encoding GANs, which is indispensable research for the GAN community. We prove three uncertainty principles of encoding GANs in practice: a) the ‘perfect’ encoder and generator cannot be continuous at the same time, which implies that current framework of encoding GANs is ill-posed and needs rethinking; b) neural networks cannot approximate the underlying encoder and generator precisely at the same time, which explains why we cannot get ‘perfect’ encoders and generators as promised in previous theories; c) neural networks cannot be stable and accurate at the same time, which demonstrates the difficulty of training and trade-off between fidelity and disentanglement encountered in previous works. Our work may eliminate gaps between previous theories and empirical results, promote the understanding of GANs, and guide network designs for follow-up works.

----

## [297] Pointwise Binary Classification with Pairwise Confidence Comparisons

**Authors**: *Lei Feng, Senlin Shu, Nan Lu, Bo Han, Miao Xu, Gang Niu, Bo An, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/feng21d.html](http://proceedings.mlr.press/v139/feng21d.html)

**Abstract**:

To alleviate the data requirement for training effective binary classifiers in binary classification, many weakly supervised learning settings have been proposed. Among them, some consider using pairwise but not pointwise labels, when pointwise labels are not accessible due to privacy, confidentiality, or security reasons. However, as a pairwise label denotes whether or not two data points share a pointwise label, it cannot be easily collected if either point is equally likely to be positive or negative. Thus, in this paper, we propose a novel setting called pairwise comparison (Pcomp) classification, where we have only pairs of unlabeled data that we know one is more likely to be positive than the other. Firstly, we give a Pcomp data generation process, derive an unbiased risk estimator (URE) with theoretical guarantee, and further improve URE using correction functions. Secondly, we link Pcomp classification to noisy-label learning to develop a progressive URE and improve it by imposing consistency regularization. Finally, we demonstrate by experiments the effectiveness of our methods, which suggests Pcomp is a valuable and practically useful type of pairwise supervision besides the pairwise label.

----

## [298] Provably Correct Optimization and Exploration with Non-linear Policies

**Authors**: *Fei Feng, Wotao Yin, Alekh Agarwal, Lin Yang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/feng21e.html](http://proceedings.mlr.press/v139/feng21e.html)

**Abstract**:

Policy optimization methods remain a powerful workhorse in empirical Reinforcement Learning (RL), with a focus on neural policies that can easily reason over complex and continuous state and/or action spaces. Theoretical understanding of strategic exploration in policy-based methods with non-linear function approximation, however, is largely missing. In this paper, we address this question by designing ENIAC, an actor-critic method that allows non-linear function approximation in the critic. We show that under certain assumptions, e.g., a bounded eluder dimension $d$ for the critic class, the learner finds to a near-optimal policy in $\widetilde{O}(\mathrm{poly}(d))$ exploration rounds. The method is robust to model misspecification and strictly extends existing works on linear function approximation. We also develop some computational optimizations of our approach with slightly worse statistical guarantees, and an empirical adaptation building on existing deep RL tools. We empirically evaluate this adaptation, and show that it outperforms prior heuristics inspired by linear methods, establishing the value in correctly reasoning about the agent’s uncertainty under non-linear function approximation.

----

## [299] KD3A: Unsupervised Multi-Source Decentralized Domain Adaptation via Knowledge Distillation

**Authors**: *Haozhe Feng, Zhaoyang You, Minghao Chen, Tianye Zhang, Minfeng Zhu, Fei Wu, Chao Wu, Wei Chen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/feng21f.html](http://proceedings.mlr.press/v139/feng21f.html)

**Abstract**:

Conventional unsupervised multi-source domain adaptation (UMDA) methods assume all source domains can be accessed directly. However, this assumption neglects the privacy-preserving policy, where all the data and computations must be kept decentralized. There exist three challenges in this scenario: (1) Minimizing the domain distance requires the pairwise calculation of the data from the source and target domains, while the data on the source domain is not available. (2) The communication cost and privacy security limit the application of existing UMDA methods, such as the domain adversarial training. (3) Since users cannot govern the data quality, the irrelevant or malicious source domains are more likely to appear, which causes negative transfer. To address the above problems, we propose a privacy-preserving UMDA paradigm named Knowledge Distillation based Decentralized Domain Adaptation (KD3A), which performs domain adaptation through the knowledge distillation on models from different source domains. The extensive experiments show that KD3A significantly outperforms state-of-the-art UMDA approaches. Moreover, the KD3A is robust to the negative transfer and brings a 100x reduction of communication cost compared with other decentralized UMDA methods.

----

## [300] Understanding Noise Injection in GANs

**Authors**: *Ruili Feng, Deli Zhao, Zheng-Jun Zha*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/feng21g.html](http://proceedings.mlr.press/v139/feng21g.html)

**Abstract**:

Noise injection is an effective way of circumventing overfitting and enhancing generalization in machine learning, the rationale of which has been validated in deep learning as well. Recently, noise injection exhibits surprising effectiveness when generating high-fidelity images in Generative Adversarial Networks (GANs) (e.g. StyleGAN). Despite its successful applications in GANs, the mechanism of its validity is still unclear. In this paper, we propose a geometric framework to theoretically analyze the role of noise injection in GANs. First, we point out the existence of the adversarial dimension trap inherent in GANs, which leads to the difficulty of learning a proper generator. Second, we successfully model the noise injection framework with exponential maps based on Riemannian geometry. Guided by our theories, we propose a general geometric realization for noise injection. Under our novel framework, the simple noise injection used in StyleGAN reduces to the Euclidean case. The goal of our work is to make theoretical steps towards understanding the underlying mechanism of state-of-the-art GAN algorithms. Experiments on image generation and GAN inversion validate our theory in practice.

----

## [301] GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings

**Authors**: *Matthias Fey, Jan Eric Lenssen, Frank Weichert, Jure Leskovec*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fey21a.html](http://proceedings.mlr.press/v139/fey21a.html)

**Abstract**:

We present GNNAutoScale (GAS), a framework for scaling arbitrary message-passing GNNs to large graphs. GAS prunes entire sub-trees of the computation graph by utilizing historical embeddings from prior training iterations, leading to constant GPU memory consumption in respect to input node size without dropping any data. While existing solutions weaken the expressive power of message passing due to sub-sampling of edges or non-trainable propagations, our approach is provably able to maintain the expressive power of the original GNN. We achieve this by providing approximation error bounds of historical embeddings and show how to tighten them in practice. Empirically, we show that the practical realization of our framework, PyGAS, an easy-to-use extension for PyTorch Geometric, is both fast and memory-efficient, learns expressive node representations, closely resembles the performance of their non-scaling counterparts, and reaches state-of-the-art performance on large-scale graphs.

----

## [302] PsiPhi-Learning: Reinforcement Learning with Demonstrations using Successor Features and Inverse Temporal Difference Learning

**Authors**: *Angelos Filos, Clare Lyle, Yarin Gal, Sergey Levine, Natasha Jaques, Gregory Farquhar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/filos21a.html](http://proceedings.mlr.press/v139/filos21a.html)

**Abstract**:

We study reinforcement learning (RL) with no-reward demonstrations, a setting in which an RL agent has access to additional data from the interaction of other agents with the same environment. However, it has no access to the rewards or goals of these agents, and their objectives and levels of expertise may vary widely. These assumptions are common in multi-agent settings, such as autonomous driving. To effectively use this data, we turn to the framework of successor features. This allows us to disentangle shared features and dynamics of the environment from agent-specific rewards and policies. We propose a multi-task inverse reinforcement learning (IRL) algorithm, called \emph{inverse temporal difference learning} (ITD), that learns shared state features, alongside per-agent successor features and preference vectors, purely from demonstrations without reward labels. We further show how to seamlessly integrate ITD with learning from online environment interactions, arriving at a novel algorithm for reinforcement learning with demonstrations, called $\Psi \Phi$-learning (pronounced ‘Sci-Fi’). We provide empirical evidence for the effectiveness of $\Psi \Phi$-learning as a method for improving RL, IRL, imitation, and few-shot transfer, and derive worst-case bounds for its performance in zero-shot transfer to new tasks.

----

## [303] A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups

**Authors**: *Marc Finzi, Max Welling, Andrew Gordon Wilson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/finzi21a.html](http://proceedings.mlr.press/v139/finzi21a.html)

**Abstract**:

Symmetries and equivariance are fundamental to the generalization of neural networks on domains such as images, graphs, and point clouds. Existing work has primarily focused on a small number of groups, such as the translation, rotation, and permutation groups. In this work we provide a completely general algorithm for solving for the equivariant layers of matrix groups. In addition to recovering solutions from other works as special cases, we construct multilayer perceptrons equivariant to multiple groups that have never been tackled before, including $\mathrm{O}(1,3)$, $\mathrm{O}(5)$, $\mathrm{Sp}(n)$, and the Rubik’s cube group. Our approach outperforms non-equivariant baselines, with applications to particle physics and modeling dynamical systems. We release our software library to enable researchers to construct equivariant layers for arbitrary

----

## [304] Few-Shot Conformal Prediction with Auxiliary Tasks

**Authors**: *Adam Fisch, Tal Schuster, Tommi S. Jaakkola, Regina Barzilay*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fisch21a.html](http://proceedings.mlr.press/v139/fisch21a.html)

**Abstract**:

We develop a novel approach to conformal prediction when the target task has limited data available for training. Conformal prediction identifies a small set of promising output candidates in place of a single prediction, with guarantees that the set contains the correct answer with high probability. When training data is limited, however, the predicted set can easily become unusably large. In this work, we obtain substantially tighter prediction sets while maintaining desirable marginal guarantees by casting conformal prediction as a meta-learning paradigm over exchangeable collections of auxiliary tasks. Our conformalization algorithm is simple, fast, and agnostic to the choice of underlying model, learning algorithm, or dataset. We demonstrate the effectiveness of this approach across a number of few-shot classification and regression tasks in natural language processing, computer vision, and computational chemistry for drug discovery.

----

## [305] Scalable Certified Segmentation via Randomized Smoothing

**Authors**: *Marc Fischer, Maximilian Baader, Martin T. Vechev*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fischer21a.html](http://proceedings.mlr.press/v139/fischer21a.html)

**Abstract**:

We present a new certification method for image and point cloud segmentation based on randomized smoothing. The method leverages a novel scalable algorithm for prediction and certification that correctly accounts for multiple testing, necessary for ensuring statistical guarantees. The key to our approach is reliance on established multiple-testing correction mechanisms as well as the ability to abstain from classifying single pixels or points while still robustly segmenting the overall input. Our experimental evaluation on synthetic data and challenging datasets, such as Pascal Context, Cityscapes, and ShapeNet, shows that our algorithm can achieve, for the first time, competitive accuracy and certification guarantees on real-world segmentation tasks. We provide an implementation at https://github.com/eth-sri/segmentation-smoothing.

----

## [306] What's in the Box? Exploring the Inner Life of Neural Networks with Robust Rules

**Authors**: *Jonas Fischer, Anna Oláh, Jilles Vreeken*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fischer21b.html](http://proceedings.mlr.press/v139/fischer21b.html)

**Abstract**:

We propose a novel method for exploring how neurons within neural networks interact. In particular, we consider activation values of a network for given data, and propose to mine noise-robust rules of the form X {\rightarrow} Y , where X and Y are sets of neurons in different layers. We identify the best set of rules by the Minimum Description Length Principle as the rules that together are most descriptive of the activation data. To learn good rule sets in practice, we propose the unsupervised ExplaiNN algorithm. Extensive evaluation shows that the patterns it discovers give clear insight in how networks perceive the world: they identify shared, respectively class-specific traits, compositionality within the network, as well as locality in convolutional layers. Moreover, these patterns are not only easily interpretable, but also supercharge prototyping as they identify which groups of neurons to consider in unison.

----

## [307] Online Learning with Optimism and Delay

**Authors**: *Genevieve Flaspohler, Francesco Orabona, Judah Cohen, Soukayna Mouatadid, Miruna Oprescu, Paulo Orenstein, Lester Mackey*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/flaspohler21a.html](http://proceedings.mlr.press/v139/flaspohler21a.html)

**Abstract**:

Inspired by the demands of real-time climate and weather forecasting, we develop optimistic online learning algorithms that require no parameter tuning and have optimal regret guarantees under delayed feedback. Our algorithms—DORM, DORM+, and AdaHedgeD—arise from a novel reduction of delayed online learning to optimistic online learning that reveals how optimistic hints can mitigate the regret penalty caused by delay. We pair this delay-as-optimism perspective with a new analysis of optimistic learning that exposes its robustness to hinting errors and a new meta-algorithm for learning effective hinting strategies in the presence of delay. We conclude by benchmarking our algorithms on four subseasonal climate forecasting tasks, demonstrating low regret relative to state-of-the-art forecasting models.

----

## [308] Online A-Optimal Design and Active Linear Regression

**Authors**: *Xavier Fontaine, Pierre Perrault, Michal Valko, Vianney Perchet*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fontaine21a.html](http://proceedings.mlr.press/v139/fontaine21a.html)

**Abstract**:

We consider in this paper the problem of optimal experiment design where a decision maker can choose which points to sample to obtain an estimate $\hat{\beta}$ of the hidden parameter $\beta^{\star}$ of an underlying linear model. The key challenge of this work lies in the heteroscedasticity assumption that we make, meaning that each covariate has a different and unknown variance. The goal of the decision maker is then to figure out on the fly the optimal way to allocate the total budget of $T$ samples between covariates, as sampling several times a specific one will reduce the variance of the estimated model around it (but at the cost of a possible higher variance elsewhere). By trying to minimize the $\ell^2$-loss $\mathbb{E} [\lVert\hat{\beta}-\beta^{\star}\rVert^2]$ the decision maker is actually minimizing the trace of the covariance matrix of the problem, which corresponds then to online A-optimal design. Combining techniques from bandit and convex optimization we propose a new active sampling algorithm and we compare it with existing ones. We provide theoretical guarantees of this algorithm in different settings, including a $\mathcal{O}(T^{-2})$ regret bound in the case where the covariates form a basis of the feature space, generalizing and improving existing results. Numerical experiments validate our theoretical findings.

----

## [309] Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design

**Authors**: *Adam Foster, Desi R. Ivanova, Ilyas Malik, Tom Rainforth*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/foster21a.html](http://proceedings.mlr.press/v139/foster21a.html)

**Abstract**:

We introduce Deep Adaptive Design (DAD), a method for amortizing the cost of adaptive Bayesian experimental design that allows experiments to be run in real-time. Traditional sequential Bayesian optimal experimental design approaches require substantial computation at each stage of the experiment. This makes them unsuitable for most real-world applications, where decisions must typically be made quickly. DAD addresses this restriction by learning an amortized design network upfront and then using this to rapidly run (multiple) adaptive experiments at deployment time. This network represents a design policy which takes as input the data from previous steps, and outputs the next design using a single forward pass; these design decisions can be made in milliseconds during the live experiment. To train the network, we introduce contrastive information bounds that are suitable objectives for the sequential setting, and propose a customized network architecture that exploits key symmetries. We demonstrate that DAD successfully amortizes the process of experimental design, outperforming alternative strategies on a number of problems.

----

## [310] Efficient Online Learning for Dynamic k-Clustering

**Authors**: *Dimitris Fotakis, Georgios Piliouras, Stratis Skoulakis*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fotakis21a.html](http://proceedings.mlr.press/v139/fotakis21a.html)

**Abstract**:

In this work, we study dynamic clustering problems from the perspective of online learning. We consider an online learning problem, called \textit{Dynamic $k$-Clustering}, in which $k$ centers are maintained in a metric space over time (centers may change positions) such as a dynamically changing set of $r$ clients is served in the best possible way. The connection cost at round $t$ is given by the \textit{$p$-norm} of the vector formed by the distance of each client to its closest center at round $t$, for some $p\geq 1$. We design a \textit{$\Theta\left( \min(k,r) \right)$-regret} polynomial-time online learning algorithm, while we show that, under some well-established computational complexity conjectures, \textit{constant-regret} cannot be achieved in polynomial-time. In addition to the efficient solution of Dynamic $k$-Clustering, our work contributes to the long line of research of combinatorial online learning.

----

## [311] Clustered Sampling: Low-Variance and Improved Representativity for Clients Selection in Federated Learning

**Authors**: *Yann Fraboni, Richard Vidal, Laetitia Kameni, Marco Lorenzi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fraboni21a.html](http://proceedings.mlr.press/v139/fraboni21a.html)

**Abstract**:

This work addresses the problem of optimizing communications between server and clients in federated learning (FL). Current sampling approaches in FL are either biased, or non optimal in terms of server-clients communications and training stability. To overcome this issue, we introduce clustered sampling for clients selection. We prove that clustered sampling leads to better clients representatitivity and to reduced variance of the clients stochastic aggregation weights in FL. Compatibly with our theory, we provide two different clustering approaches enabling clients aggregation based on 1) sample size, and 2) models similarity. Through a series of experiments in non-iid and unbalanced scenarios, we demonstrate that model aggregation through clustered sampling consistently leads to better training convergence and variability when compared to standard sampling approaches. Our approach does not require any additional operation on the clients side, and can be seamlessly integrated in standard FL implementations. Finally, clustered sampling is compatible with existing methods and technologies for privacy enhancement, and for communication reduction through model compression.

----

## [312] Agnostic Learning of Halfspaces with Gradient Descent via Soft Margins

**Authors**: *Spencer Frei, Yuan Cao, Quanquan Gu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/frei21a.html](http://proceedings.mlr.press/v139/frei21a.html)

**Abstract**:

We analyze the properties of gradient descent on convex surrogates for the zero-one loss for the agnostic learning of halfspaces. We show that when a quantity we refer to as the \textit{soft margin} is well-behaved—a condition satisfied by log-concave isotropic distributions among others—minimizers of convex surrogates for the zero-one loss are approximate minimizers for the zero-one loss itself. As standard convex optimization arguments lead to efficient guarantees for minimizing convex surrogates of the zero-one loss, our methods allow for the first positive guarantees for the classification error of halfspaces learned by gradient descent using the binary cross-entropy or hinge loss in the presence of agnostic label noise.

----

## [313] Provable Generalization of SGD-trained Neural Networks of Any Width in the Presence of Adversarial Label Noise

**Authors**: *Spencer Frei, Yuan Cao, Quanquan Gu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/frei21b.html](http://proceedings.mlr.press/v139/frei21b.html)

**Abstract**:

We consider a one-hidden-layer leaky ReLU network of arbitrary width trained by stochastic gradient descent (SGD) following an arbitrary initialization. We prove that SGD produces neural networks that have classification accuracy competitive with that of the best halfspace over the distribution for a broad class of distributions that includes log-concave isotropic and hard margin distributions. Equivalently, such networks can generalize when the data distribution is linearly separable but corrupted with adversarial label noise, despite the capacity to overfit. To the best of our knowledge, this is the first work to show that overparameterized neural networks trained by SGD can generalize when the data is corrupted with adversarial label noise.

----

## [314] Post-selection inference with HSIC-Lasso

**Authors**: *Tobias Freidling, Benjamin Poignard, Héctor Climente-González, Makoto Yamada*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/freidling21a.html](http://proceedings.mlr.press/v139/freidling21a.html)

**Abstract**:

Detecting influential features in non-linear and/or high-dimensional data is a challenging and increasingly important task in machine learning. Variable selection methods have thus been gaining much attention as well as post-selection inference. Indeed, the selected features can be significantly flawed when the selection procedure is not accounted for. We propose a selective inference procedure using the so-called model-free "HSIC-Lasso" based on the framework of truncated Gaussians combined with the polyhedral lemma. We then develop an algorithm, which allows for low computational costs and provides a selection of the regularisation parameter. The performance of our method is illustrated by both artificial and real-world data based experiments, which emphasise a tight control of the type-I error, even for small sample sizes.

----

## [315] Variational Data Assimilation with a Learned Inverse Observation Operator

**Authors**: *Thomas Frerix, Dmitrii Kochkov, Jamie A. Smith, Daniel Cremers, Michael P. Brenner, Stephan Hoyer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/frerix21a.html](http://proceedings.mlr.press/v139/frerix21a.html)

**Abstract**:

Variational data assimilation optimizes for an initial state of a dynamical system such that its evolution fits observational data. The physical model can subsequently be evolved into the future to make predictions. This principle is a cornerstone of large scale forecasting applications such as numerical weather prediction. As such, it is implemented in current operational systems of weather forecasting agencies across the globe. However, finding a good initial state poses a difficult optimization problem in part due to the non-invertible relationship between physical states and their corresponding observations. We learn a mapping from observational data to physical states and show how it can be used to improve optimizability. We employ this mapping in two ways: to better initialize the non-convex optimization problem, and to reformulate the objective function in better behaved physics space instead of observation space. Our experimental results for the Lorenz96 model and a two-dimensional turbulent fluid flow demonstrate that this procedure significantly improves forecast quality for chaotic systems.

----

## [316] Bayesian Quadrature on Riemannian Data Manifolds

**Authors**: *Christian Fröhlich, Alexandra Gessner, Philipp Hennig, Bernhard Schölkopf, Georgios Arvanitidis*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/frohlich21a.html](http://proceedings.mlr.press/v139/frohlich21a.html)

**Abstract**:

Riemannian manifolds provide a principled way to model nonlinear geometric structure inherent in data. A Riemannian metric on said manifolds determines geometry-aware shortest paths and provides the means to define statistical models accordingly. However, these operations are typically computationally demanding. To ease this computational burden, we advocate probabilistic numerical methods for Riemannian statistics. In particular, we focus on Bayesian quadrature (BQ) to numerically compute integrals over normal laws on Riemannian manifolds learned from data. In this task, each function evaluation relies on the solution of an expensive initial value problem. We show that by leveraging both prior knowledge and an active exploration scheme, BQ significantly reduces the number of required evaluations and thus outperforms Monte Carlo methods on a wide range of integration problems. As a concrete application, we highlight the merits of adopting Riemannian geometry with our proposed framework on a nonlinear dataset from molecular dynamics.

----

## [317] Learn-to-Share: A Hardware-friendly Transfer Learning Framework Exploiting Computation and Parameter Sharing

**Authors**: *Cheng Fu, Hanxian Huang, Xinyun Chen, Yuandong Tian, Jishen Zhao*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fu21a.html](http://proceedings.mlr.press/v139/fu21a.html)

**Abstract**:

Task-specific fine-tuning on pre-trained transformers has achieved performance breakthroughs in multiple NLP tasks. Yet, as both computation and parameter size grows linearly with the number of sub-tasks, it is increasingly difficult to adopt such methods to the real world due to unrealistic memory and computation overhead on computing devices. Previous works on fine-tuning focus on reducing the growing parameter size to save storage cost by parameter sharing. However, compared to storage, the constraint of computation is a more critical issue with the fine-tuning models in modern computing environments. In this work, we propose LeTS, a framework that leverages both computation and parameter sharing across multiple tasks. Compared to traditional fine-tuning, LeTS proposes a novel neural architecture that contains a fixed pre-trained transformer model, plus learnable additive components for sub-tasks. The learnable components reuse the intermediate activations in the fixed pre-trained model, decoupling computation dependency. Differentiable neural architecture search is used to determine a task-specific computation sharing scheme, and a novel early stage pruning is applied to additive components for sparsity to achieve parameter sharing. Extensive experiments show that with 1.4% of extra parameters per task, LeTS reduces the computation by 49.5% on GLUE benchmarks with only 0.2% accuracy loss compared to full fine-tuning.

----

## [318] Learning Task Informed Abstractions

**Authors**: *Xiang Fu, Ge Yang, Pulkit Agrawal, Tommi S. Jaakkola*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fu21b.html](http://proceedings.mlr.press/v139/fu21b.html)

**Abstract**:

Current model-based reinforcement learning methods struggle when operating from complex visual scenes due to their inability to prioritize task-relevant features. To mitigate this problem, we propose learning Task Informed Abstractions (TIA) that explicitly separates reward-correlated visual features from distractors. For learning TIA, we introduce the formalism of Task Informed MDP (TiMDP) that is realized by training two models that learn visual features via cooperative reconstruction, but one model is adversarially dissociated from the reward signal. Empirical evaluation shows that TIA leads to significant performance gains over state-of-the-art methods on many visual control tasks where natural and unconstrained visual distractions pose a formidable challenge. Project page: https://xiangfu.co/tia

----

## [319] Double-Win Quant: Aggressively Winning Robustness of Quantized Deep Neural Networks via Random Precision Training and Inference

**Authors**: *Yonggan Fu, Qixuan Yu, Meng Li, Vikas Chandra, Yingyan Lin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fu21c.html](http://proceedings.mlr.press/v139/fu21c.html)

**Abstract**:

Quantization is promising in enabling powerful yet complex deep neural networks (DNNs) to be deployed into resource constrained platforms. However, quantized DNNs are vulnerable to adversarial attacks unless being equipped with sophisticated techniques, leading to a dilemma of struggling between DNNs’ efficiency and robustness. In this work, we demonstrate a new perspective regarding quantization’s role in DNNs’ robustness, advocating that quantization can be leveraged to largely boost DNNs’ robustness, and propose a framework dubbed Double-Win Quant that can boost the robustness of quantized DNNs over their full precision counterparts by a large margin. Specifically, we for the first time identify that when an adversarially trained model is quantized to different precisions in a post-training manner, the associated adversarial attacks transfer poorly between different precisions. Leveraging this intriguing observation, we further develop Double-Win Quant integrating random precision inference and training to further reduce and utilize the poor adversarial transferability, enabling an aggressive “win-win" in terms of DNNs’ robustness and efficiency. Extensive experiments and ablation studies consistently validate Double-Win Quant’s effectiveness and advantages over state-of-the-art (SOTA) adversarial training methods across various attacks/models/datasets. Our codes are available at: https://github.com/RICE-EIC/Double-Win-Quant.

----

## [320] Auto-NBA: Efficient and Effective Search Over the Joint Space of Networks, Bitwidths, and Accelerators

**Authors**: *Yonggan Fu, Yongan Zhang, Yang Zhang, David D. Cox, Yingyan Lin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fu21d.html](http://proceedings.mlr.press/v139/fu21d.html)

**Abstract**:

While maximizing deep neural networks’ (DNNs’) acceleration efficiency requires a joint search/design of three different yet highly coupled aspects, including the networks, bitwidths, and accelerators, the challenges associated with such a joint search have not yet been fully understood and addressed. The key challenges include (1) the dilemma of whether to explode the memory consumption due to the huge joint space or achieve sub-optimal designs, (2) the discrete nature of the accelerator design space that is coupled yet different from that of the networks and bitwidths, and (3) the chicken and egg problem associated with network-accelerator co-search, i.e., co-search requires operation-wise hardware cost, which is lacking during search as the optimal accelerator depending on the whole network is still unknown during search. To tackle these daunting challenges towards optimal and fast development of DNN accelerators, we propose a framework dubbed Auto-NBA to enable jointly searching for the Networks, Bitwidths, and Accelerators, by efficiently localizing the optimal design within the huge joint design space for each target dataset and acceleration specification. Our Auto-NBA integrates a heterogeneous sampling strategy to achieve unbiased search with constant memory consumption, and a novel joint-search pipeline equipped with a generic differentiable accelerator search engine. Extensive experiments and ablation studies validate that both Auto-NBA generated networks and accelerators consistently outperform state-of-the-art designs (including co-search/exploration techniques, hardware-aware NAS methods, and DNN accelerators), in terms of search time, task accuracy, and accelerator efficiency. Our codes are available at: https://github.com/RICE-EIC/Auto-NBA.

----

## [321] A Deep Reinforcement Learning Approach to Marginalized Importance Sampling with the Successor Representation

**Authors**: *Scott Fujimoto, David Meger, Doina Precup*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fujimoto21a.html](http://proceedings.mlr.press/v139/fujimoto21a.html)

**Abstract**:

Marginalized importance sampling (MIS), which measures the density ratio between the state-action occupancy of a target policy and that of a sampling distribution, is a promising approach for off-policy evaluation. However, current state-of-the-art MIS methods rely on complex optimization tricks and succeed mostly on simple toy problems. We bridge the gap between MIS and deep reinforcement learning by observing that the density ratio can be computed from the successor representation of the target policy. The successor representation can be trained through deep reinforcement learning methodology and decouples the reward optimization from the dynamics of the environment, making the resulting algorithm stable and applicable to high-dimensional domains. We evaluate the empirical performance of our approach on a variety of challenging Atari and MuJoCo environments.

----

## [322] Learning disentangled representations via product manifold projection

**Authors**: *Marco Fumero, Luca Cosmo, Simone Melzi, Emanuele Rodolà*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/fumero21a.html](http://proceedings.mlr.press/v139/fumero21a.html)

**Abstract**:

We propose a novel approach to disentangle the generative factors of variation underlying a given set of observations. Our method builds upon the idea that the (unknown) low-dimensional manifold underlying the data space can be explicitly modeled as a product of submanifolds. This definition of disentanglement gives rise to a novel weakly-supervised algorithm for recovering the unknown explanatory factors behind the data. At training time, our algorithm only requires pairs of non i.i.d. data samples whose elements share at least one, possibly multidimensional, generative factor of variation. We require no knowledge on the nature of these transformations, and do not make any limiting assumption on the properties of each subspace. Our approach is easy to implement, and can be successfully applied to different kinds of data (from images to 3D surfaces) undergoing arbitrary transformations. In addition to standard synthetic benchmarks, we showcase our method in challenging real-world applications, where we compare favorably with the state of the art.

----

## [323] Policy Information Capacity: Information-Theoretic Measure for Task Complexity in Deep Reinforcement Learning

**Authors**: *Hiroki Furuta, Tatsuya Matsushima, Tadashi Kozuno, Yutaka Matsuo, Sergey Levine, Ofir Nachum, Shixiang Shane Gu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/furuta21a.html](http://proceedings.mlr.press/v139/furuta21a.html)

**Abstract**:

Progress in deep reinforcement learning (RL) research is largely enabled by benchmark task environments. However, analyzing the nature of those environments is often overlooked. In particular, we still do not have agreeable ways to measure the difficulty or solvability of a task, given that each has fundamentally different actions, observations, dynamics, rewards, and can be tackled with diverse RL algorithms. In this work, we propose policy information capacity (PIC) – the mutual information between policy parameters and episodic return – and policy-optimal information capacity (POIC) – between policy parameters and episodic optimality – as two environment-agnostic, algorithm-agnostic quantitative metrics for task difficulty. Evaluating our metrics across toy environments as well as continuous control benchmark tasks from OpenAI Gym and DeepMind Control Suite, we empirically demonstrate that these information-theoretic metrics have higher correlations with normalized task solvability scores than a variety of alternatives. Lastly, we show that these metrics can also be used for fast and compute-efficient optimizations of key design parameters such as reward shaping, policy architectures, and MDP properties for better solvability by RL algorithms without ever running full RL experiments.

----

## [324] An Information-Geometric Distance on the Space of Tasks

**Authors**: *Yansong Gao, Pratik Chaudhari*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gao21a.html](http://proceedings.mlr.press/v139/gao21a.html)

**Abstract**:

This paper prescribes a distance between learning tasks modeled as joint distributions on data and labels. Using tools in information geometry, the distance is defined to be the length of the shortest weight trajectory on a Riemannian manifold as a classifier is fitted on an interpolated task. The interpolated task evolves from the source to the target task using an optimal transport formulation. This distance, which we call the "coupled transfer distance" can be compared across different classifier architectures. We develop an algorithm to compute the distance which iteratively transports the marginal on the data of the source task to that of the target task while updating the weights of the classifier to track this evolving data distribution. We develop theory to show that our distance captures the intuitive idea that a good transfer trajectory is the one that keeps the generalization gap small during transfer, in particular at the end on the target task. We perform thorough empirical validation and analysis across diverse image classification datasets to show that the coupled transfer distance correlates strongly with the difficulty of fine-tuning.

----

## [325] Maximum Mean Discrepancy Test is Aware of Adversarial Attacks

**Authors**: *Ruize Gao, Feng Liu, Jingfeng Zhang, Bo Han, Tongliang Liu, Gang Niu, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gao21b.html](http://proceedings.mlr.press/v139/gao21b.html)

**Abstract**:

The maximum mean discrepancy (MMD) test could in principle detect any distributional discrepancy between two datasets. However, it has been shown that the MMD test is unaware of adversarial attacks–the MMD test failed to detect the discrepancy between natural data and adversarial data. Given this phenomenon, we raise a question: are natural and adversarial data really from different distributions? The answer is affirmative–the previous use of the MMD test on the purpose missed three key factors, and accordingly, we propose three components. Firstly, the Gaussian kernel has limited representation power, and we replace it with an effective deep kernel. Secondly, the test power of the MMD test was neglected, and we maximize it following asymptotic statistics. Finally, adversarial data may be non-independent, and we overcome this issue with the help of wild bootstrap. By taking care of the three factors, we verify that the MMD test is aware of adversarial attacks, which lights up a novel road for adversarial data detection based on two-sample tests.

----

## [326] Unsupervised Co-part Segmentation through Assembly

**Authors**: *Qingzhe Gao, Bin Wang, Libin Liu, Baoquan Chen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gao21c.html](http://proceedings.mlr.press/v139/gao21c.html)

**Abstract**:

Co-part segmentation is an important problem in computer vision for its rich applications. We propose an unsupervised learning approach for co-part segmentation from images. For the training stage, we leverage motion information embedded in videos and explicitly extract latent representations to segment meaningful object parts. More importantly, we introduce a dual procedure of part-assembly to form a closed loop with part-segmentation, enabling an effective self-supervision. We demonstrate the effectiveness of our approach with a host of extensive experiments, ranging from human bodies, hands, quadruped, and robot arms. We show that our approach can achieve meaningful and compact part segmentation, outperforming state-of-the-art approaches on diverse benchmarks.

----

## [327] Discriminative Complementary-Label Learning with Weighted Loss

**Authors**: *Yi Gao, Min-Ling Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gao21d.html](http://proceedings.mlr.press/v139/gao21d.html)

**Abstract**:

Complementary-label learning (CLL) deals with the weak supervision scenario where each training instance is associated with one \emph{complementary} label, which specifies the class label that the instance does \emph{not} belong to. Given the training instance ${\bm x}$, existing CLL approaches aim at modeling the \emph{generative} relationship between the complementary label $\bar y$, i.e. $P(\bar y\mid {\bm x})$, and the ground-truth label $y$, i.e. $P(y\mid {\bm x})$. Nonetheless, as the ground-truth label is not directly accessible for complementarily labeled training instance, strong generative assumptions may not hold for real-world CLL tasks. In this paper, we derive a simple and theoretically-sound \emph{discriminative} model towards $P(\bar y\mid {\bm x})$, which naturally leads to a risk estimator with estimation error bound at $\mathcal{O}(1/\sqrt{n})$ convergence rate. Accordingly, a practical CLL approach is proposed by further introducing weighted loss to the empirical risk to maximize the predictive gap between potential ground-truth label and complementary label. Extensive experiments clearly validate the effectiveness of the proposed discriminative complementary-label learning approach.

----

## [328] RATT: Leveraging Unlabeled Data to Guarantee Generalization

**Authors**: *Saurabh Garg, Sivaraman Balakrishnan, J. Zico Kolter, Zachary C. Lipton*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/garg21a.html](http://proceedings.mlr.press/v139/garg21a.html)

**Abstract**:

To assess generalization, machine learning scientists typically either (i) bound the generalization gap and then (after training) plug in the empirical risk to obtain a bound on the true risk; or (ii) validate empirically on holdout data. However, (i) typically yields vacuous guarantees for overparameterized models; and (ii) shrinks the training set and its guarantee erodes with each re-use of the holdout set. In this paper, we leverage unlabeled data to produce generalization bounds. After augmenting our (labeled) training set with randomly labeled data, we train in the standard fashion. Whenever classifiers achieve low error on the clean data but high error on the random data, our bound ensures that the true risk is low. We prove that our bound is valid for 0-1 empirical risk minimization and with linear classifiers trained by gradient descent. Our approach is especially useful in conjunction with deep learning due to the early learning phenomenon whereby networks fit true labels before noisy labels but requires one intuitive assumption. Empirically, on canonical computer vision and NLP tasks, our bound provides non-vacuous generalization guarantees that track actual performance closely. This work enables practitioners to certify generalization even when (labeled) holdout data is unavailable and provides insights into the relationship between random label noise and generalization.

----

## [329] On Proximal Policy Optimization's Heavy-tailed Gradients

**Authors**: *Saurabh Garg, Joshua Zhanson, Emilio Parisotto, Adarsh Prasad, J. Zico Kolter, Zachary C. Lipton, Sivaraman Balakrishnan, Ruslan Salakhutdinov, Pradeep Ravikumar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/garg21b.html](http://proceedings.mlr.press/v139/garg21b.html)

**Abstract**:

Modern policy gradient algorithms such as Proximal Policy Optimization (PPO) rely on an arsenal of heuristics, including loss clipping and gradient clipping, to ensure successful learning. These heuristics are reminiscent of techniques from robust statistics, commonly used for estimation in outlier-rich ("heavy-tailed") regimes. In this paper, we present a detailed empirical study to characterize the heavy-tailed nature of the gradients of the PPO surrogate reward function. We demonstrate that the gradients, especially for the actor network, exhibit pronounced heavy-tailedness and that it increases as the agent’s policy diverges from the behavioral policy (i.e., as the agent goes further off policy). Further examination implicates the likelihood ratios and advantages in the surrogate reward as the main sources of the observed heavy-tailedness. We then highlight issues arising due to the heavy-tailed nature of the gradients. In this light, we study the effects of the standard PPO clipping heuristics, demonstrating that these tricks primarily serve to offset heavy-tailedness in gradients. Thus motivated, we propose incorporating GMOM, a high-dimensional robust estimator, into PPO as a substitute for three clipping tricks. Despite requiring less hyperparameter tuning, our method matches the performance of PPO (with all heuristics enabled) on a battery of MuJoCo continuous control tasks.

----

## [330] What does LIME really see in images?

**Authors**: *Damien Garreau, Dina Mardaoui*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/garreau21a.html](http://proceedings.mlr.press/v139/garreau21a.html)

**Abstract**:

The performance of modern algorithms on certain computer vision tasks such as object recognition is now close to that of humans. This success was achieved at the price of complicated architectures depending on millions of parameters and it has become quite challenging to understand how particular predictions are made. Interpretability methods propose to give us this understanding. In this paper, we study LIME, perhaps one of the most popular. On the theoretical side, we show that when the number of generated examples is large, LIME explanations are concentrated around a limit explanation for which we give an explicit expression. We further this study for elementary shape detectors and linear models. As a consequence of this analysis, we uncover a connection between LIME and integrated gradients, another explanation method. More precisely, the LIME explanations are similar to the sum of integrated gradients over the superpixels used in the preprocessing step of LIME.

----

## [331] Parametric Graph for Unimodal Ranking Bandit

**Authors**: *Camille-Sovanneary Gauthier, Romaric Gaudel, Élisa Fromont, Boammani Aser Lompo*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gauthier21a.html](http://proceedings.mlr.press/v139/gauthier21a.html)

**Abstract**:

We tackle the online ranking problem of assigning $L$ items to $K$ positions on a web page in order to maximize the number of user clicks. We propose an original algorithm, easy to implement and with strong theoretical guarantees to tackle this problem in the Position-Based Model (PBM) setting, well suited for applications where items are displayed on a grid. Besides learning to rank, our algorithm, GRAB (for parametric Graph for unimodal RAnking Bandit), also learns the parameter of a compact graph over permutations of $K$ items among $L$. The logarithmic regret bound of this algorithm is a direct consequence of the unimodality property of the bandit setting with respect to the learned graph. Experiments against state-of-the-art learning algorithms which also tackle the PBM setting, show that our method is more efficient while giving regret performance on par with the best known algorithms on simulated and real life datasets.

----

## [332] Let's Agree to Degree: Comparing Graph Convolutional Networks in the Message-Passing Framework

**Authors**: *Floris Geerts, Filip Mazowiecki, Guillermo A. Pérez*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/geerts21a.html](http://proceedings.mlr.press/v139/geerts21a.html)

**Abstract**:

In this paper we cast neural networks defined on graphs as message-passing neural networks (MPNNs) to study the distinguishing power of different classes of such models. We are interested in when certain architectures are able to tell vertices apart based on the feature labels given as input with the graph. We consider two variants of MPNNS: anonymous MPNNs whose message functions depend only on the labels of vertices involved; and degree-aware MPNNs whose message functions can additionally use information regarding the degree of vertices. The former class covers popular graph neural network (GNN) formalisms for which the distinguished power is known. The latter covers graph convolutional networks (GCNs), introduced by Kipf and Welling, for which the distinguishing power was unknown. We obtain lower and upper bounds on the distinguishing power of (anonymous and degree-aware) MPNNs in terms of the distinguishing power of the Weisfeiler-Lehman (WL) algorithm. Our main results imply that (i) the distinguishing power of GCNs is bounded by the WL algorithm, but they may be one step ahead; (ii) the WL algorithm cannot be simulated by “plain vanilla” GCNs but the addition of a trade-off parameter between features of the vertex and those of its neighbours (as proposed by Kipf and Welling) resolves this problem.

----

## [333] On the difficulty of unbiased alpha divergence minimization

**Authors**: *Tomas Geffner, Justin Domke*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/geffner21a.html](http://proceedings.mlr.press/v139/geffner21a.html)

**Abstract**:

Several approximate inference algorithms have been proposed to minimize an alpha-divergence between an approximating distribution and a target distribution. Many of these algorithms introduce bias, the magnitude of which becomes problematic in high dimensions. Other algorithms are unbiased. These often seem to suffer from high variance, but little is rigorously known. In this work we study unbiased methods for alpha-divergence minimization through the Signal-to-Noise Ratio (SNR) of the gradient estimator. We study several representative scenarios where strong analytical results are possible, such as fully-factorized or Gaussian distributions. We find that when alpha is not zero, the SNR worsens exponentially in the dimensionality of the problem. This casts doubt on the practicality of these methods. We empirically confirm these theoretical results.

----

## [334] How and Why to Use Experimental Data to Evaluate Methods for Observational Causal Inference

**Authors**: *Amanda Gentzel, Purva Pruthi, David D. Jensen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gentzel21a.html](http://proceedings.mlr.press/v139/gentzel21a.html)

**Abstract**:

Methods that infer causal dependence from observational data are central to many areas of science, including medicine, economics, and the social sciences. A variety of theoretical properties of these methods have been proven, but empirical evaluation remains a challenge, largely due to the lack of observational data sets for which treatment effect is known. We describe and analyze observational sampling from randomized controlled trials (OSRCT), a method for evaluating causal inference methods using data from randomized controlled trials (RCTs). This method can be used to create constructed observational data sets with corresponding unbiased estimates of treatment effect, substantially increasing the number of data sets available for evaluating causal inference methods. We show that, in expectation, OSRCT creates data sets that are equivalent to those produced by randomly sampling from empirical data sets in which all potential outcomes are available. We then perform a large-scale evaluation of seven causal inference methods over 37 data sets, drawn from RCTs, as well as simulators, real-world computational systems, and observational data sets augmented with a synthetic response variable. We find notable performance differences when comparing across data from different sources, demonstrating the importance of using data from a variety of sources when evaluating any causal inference method.

----

## [335] Strategic Classification in the Dark

**Authors**: *Ganesh Ghalme, Vineet Nair, Itay Eilat, Inbal Talgam-Cohen, Nir Rosenfeld*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ghalme21a.html](http://proceedings.mlr.press/v139/ghalme21a.html)

**Abstract**:

Strategic classification studies the interaction between a classification rule and the strategic agents it governs. Agents respond by manipulating their features, under the assumption that the classifier is known. However, in many real-life scenarios of high-stake classification (e.g., credit scoring), the classifier is not revealed to the agents, which leads agents to attempt to learn the classifier and game it too. In this paper we generalize the strategic classification model to such scenarios and analyze the effect of an unknown classifier. We define the ”price of opacity” as the difference between the prediction error under the opaque and transparent policies, characterize it, and give a sufficient condition for it to be strictly positive, in which case transparency is the recommended policy. Our experiments show how Hardt et al.’s robust classifier is affected by keeping agents in the dark.

----

## [336] EMaQ: Expected-Max Q-Learning Operator for Simple Yet Effective Offline and Online RL

**Authors**: *Seyed Kamyar Seyed Ghasemipour, Dale Schuurmans, Shixiang Shane Gu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ghasemipour21a.html](http://proceedings.mlr.press/v139/ghasemipour21a.html)

**Abstract**:

Off-policy reinforcement learning (RL) holds the promise of sample-efficient learning of decision-making policies by leveraging past experience. However, in the offline RL setting – where a fixed collection of interactions are provided and no further interactions are allowed – it has been shown that standard off-policy RL methods can significantly underperform. In this work, we closely investigate an important simplification of BCQ (Fujimoto et al., 2018) – a prior approach for offline RL – removing a heuristic design choice. Importantly, in contrast to their original theoretical considerations, we derive this simplified algorithm through the introduction of a novel backup operator, Expected-Max Q-Learning (EMaQ), which is more closely related to the resulting practical algorithm. Specifically, in addition to the distribution support, EMaQ explicitly considers the number of samples and the proposal distribution, allowing us to derive new sub-optimality bounds. In the offline RL setting – the main focus of this work – EMaQ matches and outperforms prior state-of-the-art in the D4RL benchmarks (Fu et al., 2020). In the online RL setting, we demonstrate that EMaQ is competitive with Soft Actor Critic (SAC). The key contributions of our empirical findings are demonstrating the importance of careful generative model design for estimating behavior policies, and an intuitive notion of complexity for offline RL problems. With its simple interpretation and fewer moving parts, such as no explicit function approximator representing the policy, EMaQ serves as a strong yet easy to implement baseline for future work.

----

## [337] Differentially Private Aggregation in the Shuffle Model: Almost Central Accuracy in Almost a Single Message

**Authors**: *Badih Ghazi, Ravi Kumar, Pasin Manurangsi, Rasmus Pagh, Amer Sinha*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ghazi21a.html](http://proceedings.mlr.press/v139/ghazi21a.html)

**Abstract**:

The shuffle model of differential privacy has attracted attention in the literature due to it being a middle ground between the well-studied central and local models. In this work, we study the problem of summing (aggregating) real numbers or integers, a basic primitive in numerous machine learning tasks, in the shuffle model. We give a protocol achieving error arbitrarily close to that of the (Discrete) Laplace mechanism in central differential privacy, while each user only sends 1 + o(1) short messages in expectation.

----

## [338] The Power of Adaptivity for Stochastic Submodular Cover

**Authors**: *Rohan Ghuge, Anupam Gupta, Viswanath Nagarajan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ghuge21a.html](http://proceedings.mlr.press/v139/ghuge21a.html)

**Abstract**:

In the stochastic submodular cover problem, the goal is to select a subset of stochastic items of minimum expected cost to cover a submodular function. Solutions in this setting correspond to a sequential decision process that selects items one by one “adaptively” (depending on prior observations). While such adaptive solutions achieve the best objective, the inherently sequential nature makes them undesirable in many applications. We ask: \emph{how well can solutions with only a few adaptive rounds approximate fully-adaptive solutions?} We consider both cases where the stochastic items are independent, and where they are correlated. For both situations, we obtain nearly tight answers, establishing smooth tradeoffs between the number of adaptive rounds and the solution quality, relative to fully adaptive solutions. Experiments on synthetic and real datasets validate the practical performance of our algorithms, showing qualitative improvements in the solutions as we allow more rounds of adaptivity; in practice, solutions using just a few rounds of adaptivity are nearly as good as fully adaptive solutions.

----

## [339] Differentially Private Quantiles

**Authors**: *Jennifer Gillenwater, Matthew Joseph, Alex Kulesza*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gillenwater21a.html](http://proceedings.mlr.press/v139/gillenwater21a.html)

**Abstract**:

Quantiles are often used for summarizing and understanding data. If that data is sensitive, it may be necessary to compute quantiles in a way that is differentially private, providing theoretical guarantees that the result does not reveal private information. However, when multiple quantiles are needed, existing differentially private algorithms fare poorly: they either compute quantiles individually, splitting the privacy budget, or summarize the entire distribution, wasting effort. In either case the result is reduced accuracy. In this work we propose an instance of the exponential mechanism that simultaneously estimates exactly $m$ quantiles from $n$ data points while guaranteeing differential privacy. The utility function is carefully structured to allow for an efficient implementation that returns estimates of all $m$ quantiles in time $O(mn\log(n) + m^2n)$. Experiments show that our method significantly outperforms the current state of the art on both real and synthetic data while remaining efficient enough to be practical.

----

## [340] Query Complexity of Adversarial Attacks

**Authors**: *Grzegorz Gluch, Rüdiger L. Urbanke*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gluch21a.html](http://proceedings.mlr.press/v139/gluch21a.html)

**Abstract**:

There are two main attack models considered in the adversarial robustness literature: black-box and white-box. We consider these threat models as two ends of a fine-grained spectrum, indexed by the number of queries the adversary can ask. Using this point of view we investigate how many queries the adversary needs to make to design an attack that is comparable to the best possible attack in the white-box model. We give a lower bound on that number of queries in terms of entropy of decision boundaries of the classifier. Using this result we analyze two classical learning algorithms on two synthetic tasks for which we prove meaningful security guarantees. The obtained bounds suggest that some learning algorithms are inherently more robust against query-bounded adversaries than others.

----

## [341] Spectral Normalisation for Deep Reinforcement Learning: An Optimisation Perspective

**Authors**: *Florin Gogianu, Tudor Berariu, Mihaela Rosca, Claudia Clopath, Lucian Busoniu, Razvan Pascanu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gogianu21a.html](http://proceedings.mlr.press/v139/gogianu21a.html)

**Abstract**:

Most of the recent deep reinforcement learning advances take an RL-centric perspective and focus on refinements of the training objective. We diverge from this view and show we can recover the performance of these developments not by changing the objective, but by regularising the value-function estimator. Constraining the Lipschitz constant of a single layer using spectral normalisation is sufficient to elevate the performance of a Categorical-DQN agent to that of a more elaborated agent on the challenging Atari domain. We conduct ablation studies to disentangle the various effects normalisation has on the learning dynamics and show that is sufficient to modulate the parameter updates to recover most of the performance of spectral normalisation. These findings hint towards the need to also focus on the neural component and its learning dynamics to tackle the peculiarities of Deep Reinforcement Learning.

----

## [342] 12-Lead ECG Reconstruction via Koopman Operators

**Authors**: *Tomer Golany, Kira Radinsky, Daniel Freedman, Saar Minha*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/golany21a.html](http://proceedings.mlr.press/v139/golany21a.html)

**Abstract**:

32% of all global deaths in the world are caused by cardiovascular diseases. Early detection, especially for patients with ischemia or cardiac arrhythmia, is crucial. To reduce the time between symptoms onset and treatment, wearable ECG sensors were developed to allow for the recording of the full 12-lead ECG signal at home. However, if even a single lead is not correctly positioned on the body that lead becomes corrupted, making automatic diagnosis on the basis of the full signal impossible. In this work, we present a methodology to reconstruct missing or noisy leads using the theory of Koopman Operators. Given a dataset consisting of full 12-lead ECGs, we learn a dynamical system describing the evolution of the 12 individual signals together in time. The Koopman theory indicates that there exists a high-dimensional embedding space in which the operator which propagates from one time instant to the next is linear. We therefore learn both the mapping to this embedding space, as well as the corresponding linear operator. Armed with this representation, we are able to impute missing leads by solving a least squares system in the embedding space, which can be achieved efficiently due to the sparse structure of the system. We perform an empirical evaluation using 12-lead ECG signals from thousands of patients, and show that we are able to reconstruct the signals in such way that enables accurate clinical diagnosis.

----

## [343] Function Contrastive Learning of Transferable Meta-Representations

**Authors**: *Muhammad Waleed Gondal, Shruti Joshi, Nasim Rahaman, Stefan Bauer, Manuel Wuthrich, Bernhard Schölkopf*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gondal21a.html](http://proceedings.mlr.press/v139/gondal21a.html)

**Abstract**:

Meta-learning algorithms adapt quickly to new tasks that are drawn from the same task distribution as the training tasks. The mechanism leading to fast adaptation is the conditioning of a downstream predictive model on the inferred representation of the task’s underlying data generative process, or \emph{function}. This \emph{meta-representation}, which is computed from a few observed examples of the underlying function, is learned jointly with the predictive model. In this work, we study the implications of this joint training on the transferability of the meta-representations. Our goal is to learn meta-representations that are robust to noise in the data and facilitate solving a wide range of downstream tasks that share the same underlying functions. To this end, we propose a decoupled encoder-decoder approach to supervised meta-learning, where the encoder is trained with a contrastive objective to find a good representation of the underlying function. In particular, our training scheme is driven by the self-supervision signal indicating whether two sets of examples stem from the same function. Our experiments on a number of synthetic and real-world datasets show that the representations we obtain outperform strong baselines in terms of downstream performance and noise robustness, even when these baselines are trained in an end-to-end manner.

----

## [344] Active Slices for Sliced Stein Discrepancy

**Authors**: *Wenbo Gong, Kaibo Zhang, Yingzhen Li, José Miguel Hernández-Lobato*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gong21a.html](http://proceedings.mlr.press/v139/gong21a.html)

**Abstract**:

Sliced Stein discrepancy (SSD) and its kernelized variants have demonstrated promising successes in goodness-of-fit tests and model learning in high dimensions. Despite the theoretical elegance, their empirical performance depends crucially on the search of the optimal slicing directions to discriminate between two distributions. Unfortunately, previous gradient-based optimisation approach returns sub-optimal results for the slicing directions: it is computationally expensive, sensitive to initialization, and it lacks theoretical guarantee for convergence. We address these issues in two steps. First, we show in theory that the requirement of using optimal slicing directions in the kernelized version of SSD can be relaxed, validating the resulting discrepancy with finite random slicing directions. Second, given that good slicing directions are crucial for practical performance, we propose a fast algorithm for finding good slicing directions based on ideas of active sub-space construction and spectral decomposition. Experiments in goodness-of-fit tests and model learning show that our approach achieves both the best performance and the fastest convergence. Especially, we demonstrate 14-80x speed-up in goodness-of-fit tests when compared with the gradient-based approach.

----

## [345] On the Problem of Underranking in Group-Fair Ranking

**Authors**: *Sruthi Gorantla, Amit Deshpande, Anand Louis*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gorantla21a.html](http://proceedings.mlr.press/v139/gorantla21a.html)

**Abstract**:

Bias in ranking systems, especially among the top ranks, can worsen social and economic inequalities, polarize opinions, and reinforce stereotypes. On the other hand, a bias correction for minority groups can cause more harm if perceived as favoring group-fair outcomes over meritocracy. Most group-fair ranking algorithms post-process a given ranking and output a group-fair ranking. In this paper, we formulate the problem of underranking in group-fair rankings based on how close the group-fair rank of each item is to its original rank, and prove a lower bound on the trade-off achievable for simultaneous underranking and group fairness in ranking. We give a fair ranking algorithm that takes any given ranking and outputs another ranking with simultaneous underranking and group fairness guarantees comparable to the lower bound we prove. Our experimental results confirm the theoretical trade-off between underranking and group fairness, and also show that our algorithm achieves the best of both when compared to the state-of-the-art baselines.

----

## [346] MARINA: Faster Non-Convex Distributed Learning with Compression

**Authors**: *Eduard Gorbunov, Konstantin Burlachenko, Zhize Li, Peter Richtárik*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gorbunov21a.html](http://proceedings.mlr.press/v139/gorbunov21a.html)

**Abstract**:

We develop and analyze MARINA: a new communication efficient method for non-convex distributed learning over heterogeneous datasets. MARINA employs a novel communication compression strategy based on the compression of gradient differences that is reminiscent of but different from the strategy employed in the DIANA method of Mishchenko et al. (2019). Unlike virtually all competing distributed first-order methods, including DIANA, ours is based on a carefully designed biased gradient estimator, which is the key to its superior theoretical and practical performance. The communication complexity bounds we prove for MARINA are evidently better than those of all previous first-order methods. Further, we develop and analyze two variants of MARINA: VR-MARINA and PP-MARINA. The first method is designed for the case when the local loss functions owned by clients are either of a finite sum or of an expectation form, and the second method allows for a partial participation of clients {–} a feature important in federated learning. All our methods are superior to previous state-of-the-art methods in terms of oracle/communication complexity. Finally, we provide a convergence analysis of all methods for problems satisfying the Polyak-{Ł}ojasiewicz condition.

----

## [347] Systematic Analysis of Cluster Similarity Indices: How to Validate Validation Measures

**Authors**: *Martijn Gösgens, Alexey Tikhonov, Liudmila Prokhorenkova*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gosgens21a.html](http://proceedings.mlr.press/v139/gosgens21a.html)

**Abstract**:

Many cluster similarity indices are used to evaluate clustering algorithms, and choosing the best one for a particular task remains an open problem. We demonstrate that this problem is crucial: there are many disagreements among the indices, these disagreements do affect which algorithms are preferred in applications, and this can lead to degraded performance in real-world systems. We propose a theoretical framework to tackle this problem: we develop a list of desirable properties and conduct an extensive theoretical analysis to verify which indices satisfy them. This allows for making an informed choice: given a particular application, one can first select properties that are desirable for the task and then identify indices satisfying these. Our work unifies and considerably extends existing attempts at analyzing cluster similarity indices: we introduce new properties, formalize existing ones, and mathematically prove or disprove each property for an extensive list of validation indices. This broader and more rigorous approach leads to recommendations that considerably differ from how validation indices are currently being chosen by practitioners. Some of the most popular indices are even shown to be dominated by previously overlooked ones.

----

## [348] Revisiting Point Cloud Shape Classification with a Simple and Effective Baseline

**Authors**: *Ankit Goyal, Hei Law, Bowei Liu, Alejandro Newell, Jia Deng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/goyal21a.html](http://proceedings.mlr.press/v139/goyal21a.html)

**Abstract**:

Processing point cloud data is an important component of many real-world systems. As such, a wide variety of point-based approaches have been proposed, reporting steady benchmark improvements over time. We study the key ingredients of this progress and uncover two critical results. First, we find that auxiliary factors like different evaluation schemes, data augmentation strategies, and loss functions, which are independent of the model architecture, make a large difference in performance. The differences are large enough that they obscure the effect of architecture. When these factors are controlled for, PointNet++, a relatively older network, performs competitively with recent methods. Second, a very simple projection-based method, which we refer to as SimpleView, performs surprisingly well. It achieves on par or better results than sophisticated state-of-the-art methods on ModelNet40 while being half the size of PointNet++. It also outperforms state-of-the-art methods on ScanObjectNN, a real-world point cloud benchmark, and demonstrates better cross-dataset generalization. Code is available at https://github.com/princeton-vl/SimpleView.

----

## [349] Dissecting Supervised Constrastive Learning

**Authors**: *Florian Graf, Christoph D. Hofer, Marc Niethammer, Roland Kwitt*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/graf21a.html](http://proceedings.mlr.press/v139/graf21a.html)

**Abstract**:

Minimizing cross-entropy over the softmax scores of a linear map composed with a high-capacity encoder is arguably the most popular choice for training neural networks on supervised learning tasks. However, recent works show that one can directly optimize the encoder instead, to obtain equally (or even more) discriminative representations via a supervised variant of a contrastive objective. In this work, we address the question whether there are fundamental differences in the sought-for representation geometry in the output space of the encoder at minimal loss. Specifically, we prove, under mild assumptions, that both losses attain their minimum once the representations of each class collapse to the vertices of a regular simplex, inscribed in a hypersphere. We provide empirical evidence that this configuration is attained in practice and that reaching a close-to-optimal state typically indicates good generalization performance. Yet, the two losses show remarkably different optimization behavior. The number of iterations required to perfectly fit to data scales superlinearly with the amount of randomly flipped labels for the supervised contrastive loss. This is in contrast to the approximately linear scaling previously reported for networks trained with cross-entropy.

----

## [350] Oops I Took A Gradient: Scalable Sampling for Discrete Distributions

**Authors**: *Will Grathwohl, Kevin Swersky, Milad Hashemi, David Duvenaud, Chris J. Maddison*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/grathwohl21a.html](http://proceedings.mlr.press/v139/grathwohl21a.html)

**Abstract**:

We propose a general and scalable approximate sampling strategy for probabilistic models with discrete variables. Our approach uses gradients of the likelihood function with respect to its discrete inputs to propose updates in a Metropolis-Hastings sampler. We show empirically that this approach outperforms generic samplers in a number of difficult settings including Ising models, Potts models, restricted Boltzmann machines, and factorial hidden Markov models. We also demonstrate our improved sampler for training deep energy-based models on high dimensional discrete image data. This approach outperforms variational auto-encoders and existing energy-based models. Finally, we give bounds showing that our approach is near-optimal in the class of samplers which propose local updates.

----

## [351] Detecting Rewards Deterioration in Episodic Reinforcement Learning

**Authors**: *Ido Greenberg, Shie Mannor*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/greenberg21a.html](http://proceedings.mlr.press/v139/greenberg21a.html)

**Abstract**:

In many RL applications, once training ends, it is vital to detect any deterioration in the agent performance as soon as possible. Furthermore, it often has to be done without modifying the policy and under minimal assumptions regarding the environment. In this paper, we address this problem by focusing directly on the rewards and testing for degradation. We consider an episodic framework, where the rewards within each episode are not independent, nor identically-distributed, nor Markov. We present this problem as a multivariate mean-shift detection problem with possibly partial observations. We define the mean-shift in a way corresponding to deterioration of a temporal signal (such as the rewards), and derive a test for this problem with optimal statistical power. Empirically, on deteriorated rewards in control problems (generated using various environment modifications), the test is demonstrated to be more powerful than standard tests - often by orders of magnitude. We also suggest a novel Bootstrap mechanism for False Alarm Rate control (BFAR), applicable to episodic (non-i.i.d) signal and allowing our test to run sequentially in an online manner. Our method does not rely on a learned model of the environment, is entirely external to the agent, and in fact can be applied to detect changes or drifts in any episodic signal.

----

## [352] Crystallization Learning with the Delaunay Triangulation

**Authors**: *Jiaqi Gu, Guosheng Yin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gu21a.html](http://proceedings.mlr.press/v139/gu21a.html)

**Abstract**:

Based on the Delaunay triangulation, we propose the crystallization learning to estimate the conditional expectation function in the framework of nonparametric regression. By conducting the crystallization search for the Delaunay simplices closest to the target point in a hierarchical way, the crystallization learning estimates the conditional expectation of the response by fitting a local linear model to the data points of the constructed Delaunay simplices. Instead of conducting the Delaunay triangulation for the entire feature space which would encounter enormous computational difficulty, our approach focuses only on the neighborhood of the target point and thus greatly expedites the estimation for high-dimensional cases. Because the volumes of Delaunay simplices are adaptive to the density of feature data points, our method selects neighbor data points uniformly in all directions and thus is more robust to the local geometric structure of the data than existing nonparametric regression methods. We develop the asymptotic properties of the crystallization learning and conduct numerical experiments on both synthetic and real data to demonstrate the advantages of our method in estimation of the conditional expectation function and prediction of the response.

----

## [353] AutoAttend: Automated Attention Representation Search

**Authors**: *Chaoyu Guan, Xin Wang, Wenwu Zhu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/guan21a.html](http://proceedings.mlr.press/v139/guan21a.html)

**Abstract**:

Self-attention mechanisms have been widely adopted in many machine learning areas, including Natural Language Processing (NLP) and Graph Representation Learning (GRL), etc. However, existing works heavily rely on hand-crafted design to obtain customized attention mechanisms. In this paper, we automate Key, Query and Value representation design, which is one of the most important steps to obtain effective self-attentions. We propose an automated self-attention representation model, AutoAttend, which can automatically search powerful attention representations for downstream tasks leveraging Neural Architecture Search (NAS). In particular, we design a tailored search space for attention representation automation, which is flexible to produce effective attention representation designs. Based on the design prior obtained from attention representations in previous works, we further regularize our search space to reduce the space complexity without the loss of expressivity. Moreover, we propose a novel context-aware parameter sharing mechanism considering special characteristics of each sub-architecture to provide more accurate architecture estimations when conducting parameter sharing in our tailored search space. Experiments show the superiority of our proposed AutoAttend model over previous state-of-the-arts on eight text classification tasks in NLP and four node classification tasks in GRL.

----

## [354] Operationalizing Complex Causes: A Pragmatic View of Mediation

**Authors**: *Limor Gultchin, David S. Watson, Matt J. Kusner, Ricardo Silva*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gultchin21a.html](http://proceedings.mlr.press/v139/gultchin21a.html)

**Abstract**:

We examine the problem of causal response estimation for complex objects (e.g., text, images, genomics). In this setting, classical \emph{atomic} interventions are often not available (e.g., changes to characters, pixels, DNA base-pairs). Instead, we only have access to indirect or \emph{crude} interventions (e.g., enrolling in a writing program, modifying a scene, applying a gene therapy). In this work, we formalize this problem and provide an initial solution. Given a collection of candidate mediators, we propose (a) a two-step method for predicting the causal responses of crude interventions; and (b) a testing procedure to identify mediators of crude interventions. We demonstrate, on a range of simulated and real-world-inspired examples, that our approach allows us to efficiently estimate the effect of crude interventions with limited data from new treatment regimes.

----

## [355] On a Combination of Alternating Minimization and Nesterov's Momentum

**Authors**: *Sergey Guminov, Pavel E. Dvurechensky, Nazarii Tupitsa, Alexander V. Gasnikov*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/guminov21a.html](http://proceedings.mlr.press/v139/guminov21a.html)

**Abstract**:

Alternating minimization (AM) procedures are practically efficient in many applications for solving convex and non-convex optimization problems. On the other hand, Nesterov’s accelerated gradient is theoretically optimal first-order method for convex optimization. In this paper we combine AM and Nesterov’s acceleration to propose an accelerated alternating minimization algorithm. We prove $1/k^2$ convergence rate in terms of the objective for convex problems and $1/k$ in terms of the squared gradient norm for non-convex problems, where $k$ is the iteration counter. Our method does not require any knowledge of neither convexity of the problem nor function parameters such as Lipschitz constant of the gradient, i.e. it is adaptive to convexity and smoothness and is uniformly optimal for smooth convex and non-convex problems. Further, we develop its primal-dual modification for strongly convex problems with linear constraints and prove the same $1/k^2$ for the primal objective residual and constraints feasibility.

----

## [356] Decentralized Single-Timescale Actor-Critic on Zero-Sum Two-Player Stochastic Games

**Authors**: *Hongyi Guo, Zuyue Fu, Zhuoran Yang, Zhaoran Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/guo21a.html](http://proceedings.mlr.press/v139/guo21a.html)

**Abstract**:

We study the global convergence and global optimality of the actor-critic algorithm applied for the zero-sum two-player stochastic games in a decentralized manner. We focus on the single-timescale setting where the critic is updated by applying the Bellman operator only once and the actor is updated by policy gradient with the information from the critic. Our algorithm is in a decentralized manner, as we assume that each player has no access to the actions of the other one, which, in a way, protects the privacy of both players. Moreover, we consider linear function approximations for both actor and critic, and we prove that the sequence of joint policy generated by our decentralized linear algorithm converges to the minimax equilibrium at a sublinear rate \(\cO(\sqrt{K})\), where \(K\){is} the number of iterations. To the best of our knowledge, we establish the global optimality and convergence of decentralized actor-critic algorithm on zero-sum two-player stochastic games with linear function approximations for the first time.

----

## [357] Adversarial Policy Learning in Two-player Competitive Games

**Authors**: *Wenbo Guo, Xian Wu, Sui Huang, Xinyu Xing*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/guo21b.html](http://proceedings.mlr.press/v139/guo21b.html)

**Abstract**:

In a two-player deep reinforcement learning task, recent work shows an attacker could learn an adversarial policy that triggers a target agent to perform poorly and even react in an undesired way. However, its efficacy heavily relies upon the zero-sum assumption made in the two-player game. In this work, we propose a new adversarial learning algorithm. It addresses the problem by resetting the optimization goal in the learning process and designing a new surrogate optimization function. Our experiments show that our method significantly improves adversarial agents’ exploitability compared with the state-of-art attack. Besides, we also discover that our method could augment an agent with the ability to abuse the target game’s unfairness. Finally, we show that agents adversarially re-trained against our adversarial agents could obtain stronger adversary-resistance.

----

## [358] Soft then Hard: Rethinking the Quantization in Neural Image Compression

**Authors**: *Zongyu Guo, Zhizheng Zhang, Runsen Feng, Zhibo Chen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/guo21c.html](http://proceedings.mlr.press/v139/guo21c.html)

**Abstract**:

Quantization is one of the core components in lossy image compression. For neural image compression, end-to-end optimization requires differentiable approximations of quantization, which can generally be grouped into three categories: additive uniform noise, straight-through estimator and soft-to-hard annealing. Training with additive uniform noise approximates the quantization error variationally but suffers from the train-test mismatch. The other two methods do not encounter this mismatch but, as shown in this paper, hurt the rate-distortion performance since the latent representation ability is weakened. We thus propose a novel soft-then-hard quantization strategy for neural image compression that first learns an expressive latent space softly, then closes the train-test mismatch with hard quantization. In addition, beyond the fixed integer-quantization, we apply scaled additive uniform noise to adaptively control the quantization granularity by deriving a new variational upper bound on actual rate. Experiments demonstrate that our proposed methods are easy to adopt, stable to train, and highly effective especially on complex compression models.

----

## [359] UneVEn: Universal Value Exploration for Multi-Agent Reinforcement Learning

**Authors**: *Tarun Gupta, Anuj Mahajan, Bei Peng, Wendelin Boehmer, Shimon Whiteson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gupta21a.html](http://proceedings.mlr.press/v139/gupta21a.html)

**Abstract**:

VDN and QMIX are two popular value-based algorithms for cooperative MARL that learn a centralized action value function as a monotonic mixing of per-agent utilities. While this enables easy decentralization of the learned policy, the restricted joint action value function can prevent them from solving tasks that require significant coordination between agents at a given timestep. We show that this problem can be overcome by improving the joint exploration of all agents during training. Specifically, we propose a novel MARL approach called Universal Value Exploration (UneVEn) that learns a set of related tasks simultaneously with a linear decomposition of universal successor features. With the policies of already solved related tasks, the joint exploration process of all agents can be improved to help them achieve better coordination. Empirical results on a set of exploration games, challenging cooperative predator-prey tasks requiring significant coordination among agents, and StarCraft II micromanagement benchmarks show that UneVEn can solve tasks where other state-of-the-art MARL methods fail.

----

## [360] Distribution-Free Calibration Guarantees for Histogram Binning without Sample Splitting

**Authors**: *Chirag Gupta, Aaditya Ramdas*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gupta21b.html](http://proceedings.mlr.press/v139/gupta21b.html)

**Abstract**:

We prove calibration guarantees for the popular histogram binning (also called uniform-mass binning) method of Zadrozny and Elkan (2001). Histogram binning has displayed strong practical performance, but theoretical guarantees have only been shown for sample split versions that avoid ’double dipping’ the data. We demonstrate that the statistical cost of sample splitting is practically significant on a credit default dataset. We then prove calibration guarantees for the original method that double dips the data, using a certain Markov property of order statistics. Based on our results, we make practical recommendations for choosing the number of bins in histogram binning. In our illustrative simulations, we propose a new tool for assessing calibration—validity plots—which provide more information than an ECE estimate.

----

## [361] Correcting Exposure Bias for Link Recommendation

**Authors**: *Shantanu Gupta, Hao Wang, Zachary C. Lipton, Yuyang Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gupta21c.html](http://proceedings.mlr.press/v139/gupta21c.html)

**Abstract**:

Link prediction methods are frequently applied in recommender systems, e.g., to suggest citations for academic papers or friends in social networks. However, exposure bias can arise when users are systematically underexposed to certain relevant items. For example, in citation networks, authors might be more likely to encounter papers from their own field and thus cite them preferentially. This bias can propagate through naively trained link predictors, leading to both biased evaluation and high generalization error (as assessed by true relevance). Moreover, this bias can be exacerbated by feedback loops. We propose estimators that leverage known exposure probabilities to mitigate this bias and consequent feedback loops. Next, we provide a loss function for learning the exposure probabilities from data. Finally, experiments on semi-synthetic data based on real-world citation networks, show that our methods reliably identify (truly) relevant citations. Additionally, our methods lead to greater diversity in the recommended papers’ fields of study. The code is available at github.com/shantanu95/exposure-bias-link-rec.

----

## [362] The Heavy-Tail Phenomenon in SGD

**Authors**: *Mert Gürbüzbalaban, Umut Simsekli, Lingjiong Zhu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gurbuzbalaban21a.html](http://proceedings.mlr.press/v139/gurbuzbalaban21a.html)

**Abstract**:

In recent years, various notions of capacity and complexity have been proposed for characterizing the generalization properties of stochastic gradient descent (SGD) in deep learning. Some of the popular notions that correlate well with the performance on unseen data are (i) the ‘flatness’ of the local minimum found by SGD, which is related to the eigenvalues of the Hessian, (ii) the ratio of the stepsize $\eta$ to the batch-size $b$, which essentially controls the magnitude of the stochastic gradient noise, and (iii) the ‘tail-index’, which measures the heaviness of the tails of the network weights at convergence. In this paper, we argue that these three seemingly unrelated perspectives for generalization are deeply linked to each other. We claim that depending on the structure of the Hessian of the loss at the minimum, and the choices of the algorithm parameters $\eta$ and $b$, the SGD iterates will converge to a \emph{heavy-tailed} stationary distribution. We rigorously prove this claim in the setting of quadratic optimization: we show that even in a simple linear regression problem with independent and identically distributed data whose distribution has finite moments of all order, the iterates can be heavy-tailed with infinite variance. We further characterize the behavior of the tails with respect to algorithm parameters, the dimension, and the curvature. We then translate our results into insights about the behavior of SGD in deep learning. We support our theory with experiments conducted on synthetic data, fully connected, and convolutional neural networks.

----

## [363] Knowledge Enhanced Machine Learning Pipeline against Diverse Adversarial Attacks

**Authors**: *Nezihe Merve Gürel, Xiangyu Qi, Luka Rimanic, Ce Zhang, Bo Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gurel21a.html](http://proceedings.mlr.press/v139/gurel21a.html)

**Abstract**:

Despite the great successes achieved by deep neural networks (DNNs), recent studies show that they are vulnerable against adversarial examples, which aim to mislead DNNs by adding small adversarial perturbations. Several defenses have been proposed against such attacks, while many of them have been adaptively attacked. In this work, we aim to enhance the ML robustness from a different perspective by leveraging domain knowledge: We propose a Knowledge Enhanced Machine Learning Pipeline (KEMLP) to integrate domain knowledge (i.e., logic relationships among different predictions) into a probabilistic graphical model via first-order logic rules. In particular, we develop KEMLP by integrating a diverse set of weak auxiliary models based on their logical relationships to the main DNN model that performs the target task. Theoretically, we provide convergence results and prove that, under mild conditions, the prediction of KEMLP is more robust than that of the main DNN model. Empirically, we take road sign recognition as an example and leverage the relationships between road signs and their shapes and contents as domain knowledge. We show that compared with adversarial training and other baselines, KEMLP achieves higher robustness against physical attacks, $\mathcal{L}_p$ bounded attacks, unforeseen attacks, and natural corruptions under both whitebox and blackbox settings, while still maintaining high clean accuracy.

----

## [364] Adapting to Delays and Data in Adversarial Multi-Armed Bandits

**Authors**: *András György, Pooria Joulani*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/gyorgy21a.html](http://proceedings.mlr.press/v139/gyorgy21a.html)

**Abstract**:

We consider the adversarial multi-armed bandit problem under delayed feedback. We analyze variants of the Exp3 algorithm that tune their step size using only information (about the losses and delays) available at the time of the decisions, and obtain regret guarantees that adapt to the observed (rather than the worst-case) sequences of delays and/or losses. First, through a remarkably simple proof technique, we show that with proper tuning of the step size, the algorithm achieves an optimal (up to logarithmic factors) regret of order $\sqrt{\log(K)(TK + D)}$ both in expectation and in high probability, where $K$ is the number of arms, $T$ is the time horizon, and $D$ is the cumulative delay. The high-probability version of the bound, which is the first high-probability delay-adaptive bound in the literature, crucially depends on the use of implicit exploration in estimating the losses. Then, following Zimmert and Seldin (2019), we extend these results so that the algorithm can “skip” rounds with large delays, resulting in regret bounds of order $\sqrt{TK\log(K)} + |R| + \sqrt{D_{\bar{R}}\log(K)}$, where $R$ is an arbitrary set of rounds (which are skipped) and $D_{\bar{R}}$ is the cumulative delay of the feedback for other rounds. Finally, we present another, data-adaptive (AdaGrad-style) version of the algorithm for which the regret adapts to the observed (delayed) losses instead of only adapting to the cumulative delay (this algorithm requires an a priori upper bound on the maximum delay, or the advance knowledge of the delay for each decision when it is made). The resulting bound can be orders of magnitude smaller on benign problems, and it can be shown that the delay only affects the regret through the loss of the best arm.

----

## [365] Rate-Distortion Analysis of Minimum Excess Risk in Bayesian Learning

**Authors**: *Hassan Hafez-Kolahi, Behrad Moniri, Shohreh Kasaei, Mahdieh Soleymani Baghshah*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hafez-kolahi21a.html](http://proceedings.mlr.press/v139/hafez-kolahi21a.html)

**Abstract**:

In parametric Bayesian learning, a prior is assumed on the parameter $W$ which determines the distribution of samples. In this setting, Minimum Excess Risk (MER) is defined as the difference between the minimum expected loss achievable when learning from data and the minimum expected loss that could be achieved if $W$ was observed. In this paper, we build upon and extend the recent results of (Xu & Raginsky, 2020) to analyze the MER in Bayesian learning and derive information-theoretic bounds on it. We formulate the problem as a (constrained) rate-distortion optimization and show how the solution can be bounded above and below by two other rate-distortion functions that are easier to study. The lower bound represents the minimum possible excess risk achievable by \emph{any} process using $R$ bits of information from the parameter $W$. For the upper bound, the optimization is further constrained to use $R$ bits from the training set, a setting which relates MER to information-theoretic bounds on the generalization gap in frequentist learning. We derive information-theoretic bounds on the difference between these upper and lower bounds and show that they can provide order-wise tight rates for MER under certain conditions. This analysis gives more insight into the information-theoretic nature of Bayesian learning as well as providing novel bounds.

----

## [366] Regret Minimization in Stochastic Non-Convex Learning via a Proximal-Gradient Approach

**Authors**: *Nadav Hallak, Panayotis Mertikopoulos, Volkan Cevher*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hallak21a.html](http://proceedings.mlr.press/v139/hallak21a.html)

**Abstract**:

This paper develops a methodology for regret minimization with stochastic first-order oracle feedback in online, constrained, non-smooth, non-convex problems. In this setting, the minimization of external regret is beyond reach for first-order methods, and there are no gradient-based algorithmic frameworks capable of providing a solution. On that account, we propose a conceptual approach that leverages non-convex optimality measures, leading to a suitable generalization of the learner’s local regret. We focus on a local regret measure defined via a proximal-gradient mapping, that also encompasses the original notion proposed by Hazan et al. (2017). To achieve no local regret in this setting, we develop a proximal-gradient method based on stochastic first-order feedback, and a simpler method for when access to a perfect first-order oracle is possible. Both methods are order-optimal (in the min-max sense), and we also establish a bound on the number of proximal-gradient queries these methods require. As an important application of our results, we also obtain a link between online and offline non-convex stochastic optimization manifested as a new proximal-gradient scheme with complexity guarantees matching those obtained via variance reduction techniques.

----

## [367] Diversity Actor-Critic: Sample-Aware Entropy Regularization for Sample-Efficient Exploration

**Authors**: *Seungyul Han, Youngchul Sung*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/han21a.html](http://proceedings.mlr.press/v139/han21a.html)

**Abstract**:

In this paper, sample-aware policy entropy regularization is proposed to enhance the conventional policy entropy regularization for better exploration. Exploiting the sample distribution obtainable from the replay buffer, the proposed sample-aware entropy regularization maximizes the entropy of the weighted sum of the policy action distribution and the sample action distribution from the replay buffer for sample-efficient exploration. A practical algorithm named diversity actor-critic (DAC) is developed by applying policy iteration to the objective function with the proposed sample-aware entropy regularization. Numerical results show that DAC significantly outperforms existing recent algorithms for reinforcement learning.

----

## [368] Adversarial Combinatorial Bandits with General Non-linear Reward Functions

**Authors**: *Yanjun Han, Yining Wang, Xi Chen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/han21b.html](http://proceedings.mlr.press/v139/han21b.html)

**Abstract**:

In this paper we study the adversarial combinatorial bandit with a known non-linear reward function, extending existing work on adversarial linear combinatorial bandit. {The adversarial combinatorial bandit with general non-linear reward is an important open problem in bandit literature, and it is still unclear whether there is a significant gap from the case of linear reward, stochastic bandit, or semi-bandit feedback.} We show that, with $N$ arms and subsets of $K$ arms being chosen at each of $T$ time periods, the minimax optimal regret is $\widetilde\Theta_{d}(\sqrt{N^d T})$ if the reward function is a $d$-degree polynomial with $d< K$, and $\Theta_K(\sqrt{N^K T})$ if the reward function is not a low-degree polynomial. {Both bounds are significantly different from the bound $O(\sqrt{\mathrm{poly}(N,K)T})$ for the linear case, which suggests that there is a fundamental gap between the linear and non-linear reward structures.} Our result also finds applications to adversarial assortment optimization problem in online recommendation. We show that in the worst-case of adversarial assortment problem, the optimal algorithm must treat each individual $\binom{N}{K}$ assortment as independent.

----

## [369] A Collective Learning Framework to Boost GNN Expressiveness for Node Classification

**Authors**: *Mengyue Hang, Jennifer Neville, Bruno Ribeiro*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hang21a.html](http://proceedings.mlr.press/v139/hang21a.html)

**Abstract**:

Collective Inference (CI) is a procedure designed to boost weak relational classifiers, specially for node classification tasks. Graph Neural Networks (GNNs) are strong classifiers that have been used with great success. Unfortunately, most existing practical GNNs are not most-expressive (universal). Thus, it is an open question whether one can improve strong relational node classifiers, such as GNNs, with CI. In this work, we investigate this question and propose {\em collective learning} for GNNs —a general collective classification approach for node representation learning that increases their representation power. We show that previous attempts to incorporate CI into GNNs fail to boost their expressiveness because they do not adapt CI’s Monte Carlo sampling to representation learning. We evaluate our proposed framework with a variety of state-of-the-art GNNs. Our experiments show a consistent, significant boost in node classification accuracy —regardless of the choice of underlying GNN— for inductive node classification in partially-labeled graphs, across five real-world network datasets.

----

## [370] Grounding Language to Entities and Dynamics for Generalization in Reinforcement Learning

**Authors**: *Austin W. Hanjie, Victor Zhong, Karthik Narasimhan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hanjie21a.html](http://proceedings.mlr.press/v139/hanjie21a.html)

**Abstract**:

We investigate the use of natural language to drive the generalization of control policies and introduce the new multi-task environment Messenger with free-form text manuals describing the environment dynamics. Unlike previous work, Messenger does not assume prior knowledge connecting text and state observations {—} the control policy must simultaneously ground the game manual to entity symbols and dynamics in the environment. We develop a new model, EMMA (Entity Mapper with Multi-modal Attention) which uses an entity-conditioned attention module that allows for selective focus over relevant descriptions in the manual for each entity in the environment. EMMA is end-to-end differentiable and learns a latent grounding of entities and dynamics from text to observations using only environment rewards. EMMA achieves successful zero-shot generalization to unseen games with new dynamics, obtaining a 40% higher win rate compared to multiple baselines. However, win rate on the hardest stage of Messenger remains low (10%), demonstrating the need for additional work in this direction.

----

## [371] Sparse Feature Selection Makes Batch Reinforcement Learning More Sample Efficient

**Authors**: *Botao Hao, Yaqi Duan, Tor Lattimore, Csaba Szepesvári, Mengdi Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hao21a.html](http://proceedings.mlr.press/v139/hao21a.html)

**Abstract**:

This paper provides a statistical analysis of high-dimensional batch reinforcement learning (RL) using sparse linear function approximation. When there is a large number of candidate features, our result sheds light on the fact that sparsity-aware methods can make batch RL more sample efficient. We first consider the off-policy policy evaluation problem. To evaluate a new target policy, we analyze a Lasso fitted Q-evaluation method and establish a finite-sample error bound that has no polynomial dependence on the ambient dimension. To reduce the Lasso bias, we further propose a post model-selection estimator that applies fitted Q-evaluation to the features selected via group Lasso. Under an additional signal strength assumption, we derive a sharper instance-dependent error bound that depends on a divergence function measuring the distribution mismatch between the data distribution and occupancy measure of the target policy. Further, we study the Lasso fitted Q-iteration for batch policy optimization and establish a finite-sample error bound depending on the ratio between the number of relevant features and restricted minimal eigenvalue of the data’s covariance. In the end, we complement the results with minimax lower bounds for batch-data policy evaluation/optimization that nearly match our upper bounds. The results suggest that having well-conditioned data is crucial for sparse batch policy learning.

----

## [372] Bootstrapping Fitted Q-Evaluation for Off-Policy Inference

**Authors**: *Botao Hao, Xiang Ji, Yaqi Duan, Hao Lu, Csaba Szepesvári, Mengdi Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hao21b.html](http://proceedings.mlr.press/v139/hao21b.html)

**Abstract**:

Bootstrapping provides a flexible and effective approach for assessing the quality of batch reinforcement learning, yet its theoretical properties are poorly understood. In this paper, we study the use of bootstrapping in off-policy evaluation (OPE), and in particular, we focus on the fitted Q-evaluation (FQE) that is known to be minimax-optimal in the tabular and linear-model cases. We propose a bootstrapping FQE method for inferring the distribution of the policy evaluation error and show that this method is asymptotically efficient and distributionally consistent for off-policy statistical inference. To overcome the computation limit of bootstrapping, we further adapt a subsampling procedure that improves the runtime by an order of magnitude. We numerically evaluate the bootrapping method in classical RL environments for confidence interval estimation, estimating the variance of off-policy evaluator, and estimating the correlation between multiple off-policy evaluators.

----

## [373] Compressed Maximum Likelihood

**Authors**: *Yi Hao, Alon Orlitsky*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hao21c.html](http://proceedings.mlr.press/v139/hao21c.html)

**Abstract**:

Maximum likelihood (ML) is one of the most fundamental and general statistical estimation techniques. Inspired by recent advances in estimating distribution functionals, we propose $\textit{compressed maximum likelihood}$ (CML) that applies ML to the compressed samples. We then show that CML is sample-efficient for several essential learning tasks over both discrete and continuous domains, including learning densities with structures, estimating probability multisets, and inferring symmetric distribution functionals.

----

## [374] Valid Causal Inference with (Some) Invalid Instruments

**Authors**: *Jason S. Hartford, Victor Veitch, Dhanya Sridhar, Kevin Leyton-Brown*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hartford21a.html](http://proceedings.mlr.press/v139/hartford21a.html)

**Abstract**:

Instrumental variable methods provide a powerful approach to estimating causal effects in the presence of unobserved confounding. But a key challenge when applying them is the reliance on untestable "exclusion" assumptions that rule out any relationship between the instrument variable and the response that is not mediated by the treatment. In this paper, we show how to perform consistent IV estimation despite violations of the exclusion assumption. In particular, we show that when one has multiple candidate instruments, only a majority of these candidates—or, more generally, the modal candidate-response relationship—needs to be valid to estimate the causal effect. Our approach uses an estimate of the modal prediction from an ensemble of instrumental variable estimators. The technique is simple to apply and is "black-box" in the sense that it may be used with any instrumental variable estimator as long as the treatment effect is identified for each valid instrument independently. As such, it is compatible with recent machine-learning based estimators that allow for the estimation of conditional average treatment effects (CATE) on complex, high dimensional data. Experimentally, we achieve accurate estimates of conditional average treatment effects using an ensemble of deep network-based estimators, including on a challenging simulated Mendelian Randomization problem.

----

## [375] Model Performance Scaling with Multiple Data Sources

**Authors**: *Tatsunori Hashimoto*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hashimoto21a.html](http://proceedings.mlr.press/v139/hashimoto21a.html)

**Abstract**:

Real-world machine learning systems are often trained using a mix of data sources with varying cost and quality. Understanding how the size and composition of a training dataset affect model performance is critical for advancing our understanding of generalization, as well as designing more effective data collection policies. We show that there is a simple scaling law that predicts the loss incurred by a model even under varying dataset composition. Our work expands recent observations of scaling laws for log-linear generalization error in the i.i.d setting and uses this to cast model performance prediction as a learning problem. Using the theory of optimal experimental design, we derive a simple rational function approximation to generalization error that can be fitted using a few model training runs. Our approach can achieve highly accurate ($r^2\approx .9$) predictions of model performance under substantial extrapolation in two different standard supervised learning tasks and is accurate ($r^2 \approx .83$) on more challenging machine translation and question answering tasks where many baselines achieve worse-than-random performance.

----

## [376] Hierarchical VAEs Know What They Don't Know

**Authors**: *Jakob Drachmann Havtorn, Jes Frellsen, Søren Hauberg, Lars Maaløe*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/havtorn21a.html](http://proceedings.mlr.press/v139/havtorn21a.html)

**Abstract**:

Deep generative models have been demonstrated as state-of-the-art density estimators. Yet, recent work has found that they often assign a higher likelihood to data from outside the training distribution. This seemingly paradoxical behavior has caused concerns over the quality of the attained density estimates. In the context of hierarchical variational autoencoders, we provide evidence to explain this behavior by out-of-distribution data having in-distribution low-level features. We argue that this is both expected and desirable behavior. With this insight in hand, we develop a fast, scalable and fully unsupervised likelihood-ratio score for OOD detection that requires data to be in-distribution across all feature-levels. We benchmark the method on a vast set of data and model combinations and achieve state-of-the-art results on out-of-distribution detection.

----

## [377] Defense against backdoor attacks via robust covariance estimation

**Authors**: *Jonathan Hayase, Weihao Kong, Raghav Somani, Sewoong Oh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hayase21a.html](http://proceedings.mlr.press/v139/hayase21a.html)

**Abstract**:

Modern machine learning increasingly requires training on a large collection of data from multiple sources, not all of which can be trusted. A particularly frightening scenario is when a small fraction of corrupted data changes the behavior of the trained model when triggered by an attacker-specified watermark. Such a compromised model will be deployed unnoticed as the model is accurate otherwise. There has been promising attempts to use the intermediate representations of such a model to separate corrupted examples from clean ones. However, these methods require a significant fraction of the data to be corrupted, in order to have strong enough signal for detection. We propose a novel defense algorithm using robust covariance estimation to amplify the spectral signature of corrupted data. This defense is able to completely remove backdoors whenever the benchmark backdoor attacks are successful, even in regimes where previous methods have no hope for detecting poisoned examples.

----

## [378] Boosting for Online Convex Optimization

**Authors**: *Elad Hazan, Karan Singh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hazan21a.html](http://proceedings.mlr.press/v139/hazan21a.html)

**Abstract**:

We consider the decision-making framework of online convex optimization with a very large number of experts. This setting is ubiquitous in contextual and reinforcement learning problems, where the size of the policy class renders enumeration and search within the policy class infeasible. Instead, we consider generalizing the methodology of online boosting. We define a weak learning algorithm as a mechanism that guarantees multiplicatively approximate regret against a base class of experts. In this access model, we give an efficient boosting algorithm that guarantees near-optimal regret against the convex hull of the base class. We consider both full and partial (a.k.a. bandit) information feedback models. We also give an analogous efficient boosting algorithm for the i.i.d. statistical setting. Our results simultaneously generalize online boosting and gradient boosting guarantees to contextual learning model, online convex optimization and bandit linear optimization settings.

----

## [379] PipeTransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models

**Authors**: *Chaoyang He, Shen Li, Mahdi Soltanolkotabi, Salman Avestimehr*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/he21a.html](http://proceedings.mlr.press/v139/he21a.html)

**Abstract**:

The size of Transformer models is growing at an unprecedented rate. It has taken less than one year to reach trillion-level parameters since the release of GPT-3 (175B). Training such models requires both substantial engineering efforts and enormous computing resources, which are luxuries most research teams cannot afford. In this paper, we propose PipeTransformer, which leverages automated elastic pipelining for efficient distributed training of Transformer models. In PipeTransformer, we design an adaptive on the fly freeze algorithm that can identify and freeze some layers gradually during training, and an elastic pipelining system that can dynamically allocate resources to train the remaining active layers. More specifically, PipeTransformer automatically excludes frozen layers from the pipeline, packs active layers into fewer GPUs, and forks more replicas to increase data-parallel width. We evaluate PipeTransformer using Vision Transformer (ViT) on ImageNet and BERT on SQuAD and GLUE datasets. Our results show that compared to the state-of-the-art baseline, PipeTransformer attains up to 2.83-fold speedup without losing accuracy. We also provide various performance analyses for a more comprehensive understanding of our algorithmic and system-wise design. Finally, we have modularized our training system with flexible APIs and made the source code publicly available at https://DistML.ai.

----

## [380] SoundDet: Polyphonic Moving Sound Event Detection and Localization from Raw Waveform

**Authors**: *Yuhang He, Niki Trigoni, Andrew Markham*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/he21b.html](http://proceedings.mlr.press/v139/he21b.html)

**Abstract**:

We present a new framework SoundDet, which is an end-to-end trainable and light-weight framework, for polyphonic moving sound event detection and localization. Prior methods typically approach this problem by preprocessing raw waveform into time-frequency representations, which is more amenable to process with well-established image processing pipelines. Prior methods also detect in segment-wise manner, leading to incomplete and partial detections. SoundDet takes a novel approach and directly consumes the raw, multichannel waveform and treats the spatio-temporal sound event as a complete “sound-object" to be detected. Specifically, SoundDet consists of a backbone neural network and two parallel heads for temporal detection and spatial localization, respectively. Given the large sampling rate of raw waveform, the backbone network first learns a set of phase-sensitive and frequency-selective bank of filters to explicitly retain direction-of-arrival information, whilst being highly computationally and parametrically efficient than standard 1D/2D convolution. A dense sound event proposal map is then constructed to handle the challenges of predicting events with large varying temporal duration. Accompanying the dense proposal map are a temporal overlapness map and a motion smoothness map that measure a proposal’s confidence to be an event from temporal detection accuracy and movement consistency perspective. Involving the two maps guarantees SoundDet to be trained in a spatio-temporally unified manner. Experimental results on the public DCASE dataset show the advantage of SoundDet on both segment-based evaluation and our newly proposed event-based evaluation system.

----

## [381] Logarithmic Regret for Reinforcement Learning with Linear Function Approximation

**Authors**: *Jiafan He, Dongruo Zhou, Quanquan Gu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/he21c.html](http://proceedings.mlr.press/v139/he21c.html)

**Abstract**:

Reinforcement learning (RL) with linear function approximation has received increasing attention recently. However, existing work has focused on obtaining $\sqrt{T}$-type regret bound, where $T$ is the number of interactions with the MDP. In this paper, we show that logarithmic regret is attainable under two recently proposed linear MDP assumptions provided that there exists a positive sub-optimality gap for the optimal action-value function. More specifically, under the linear MDP assumption (Jin et al., 2020), the LSVI-UCB algorithm can achieve $\tilde{O}(d^{3}H^5/\text{gap}_{\text{min}}\cdot \log(T))$regret; and under the linear mixture MDP assumption (Ayoub et al., 2020), the UCRL-VTR algorithm can achieve $\tilde{O}(d^{2}H^5/\text{gap}_{\text{min}}\cdot \log^3(T))$ regret, where $d$ is the dimension of feature mapping, $H$ is the length of episode, $\text{gap}_{\text{min}}$ is the minimal sub-optimality gap, and $\tilde O$ hides all logarithmic terms except $\log(T)$. To the best of our knowledge, these are the first logarithmic regret bounds for RL with linear function approximation. We also establish gap-dependent lower bounds for the two linear MDP models.

----

## [382] Finding Relevant Information via a Discrete Fourier Expansion

**Authors**: *Mohsen Heidari, Jithin K. Sreedharan, Gil I. Shamir, Wojciech Szpankowski*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/heidari21a.html](http://proceedings.mlr.press/v139/heidari21a.html)

**Abstract**:

A fundamental obstacle in learning information from data is the presence of nonlinear redundancies and dependencies in it. To address this, we propose a Fourier-based approach to extract relevant information in the supervised setting. We first develop a novel Fourier expansion for functions of correlated binary random variables. This expansion is a generalization of the standard Fourier analysis on the Boolean cube beyond product probability spaces. We further extend our Fourier analysis to stochastic mappings. As an important application of this analysis, we investigate learning with feature subset selection. We reformulate this problem in the Fourier domain and introduce a computationally efficient measure for selecting features. Bridging the Bayesian error rate with the Fourier coefficients, we demonstrate that the Fourier expansion provides a powerful tool to characterize nonlinear dependencies in the features-label relation. Via theoretical analysis, we show that our proposed measure finds provably asymptotically optimal feature subsets. Lastly, we present an algorithm based on our measure and verify our findings via numerical experiments on various datasets.

----

## [383] Zeroth-Order Non-Convex Learning via Hierarchical Dual Averaging

**Authors**: *Amélie Héliou, Matthieu Martin, Panayotis Mertikopoulos, Thibaud Rahier*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/heliou21a.html](http://proceedings.mlr.press/v139/heliou21a.html)

**Abstract**:

We propose a hierarchical version of dual averaging for zeroth-order online non-convex optimization {–} i.e., learning processes where, at each stage, the optimizer is facing an unknown non-convex loss function and only receives the incurred loss as feedback. The proposed class of policies relies on the construction of an online model that aggregates loss information as it arrives, and it consists of two principal components: (a) a regularizer adapted to the Fisher information metric (as opposed to the metric norm of the ambient space); and (b) a principled exploration of the problem’s state space based on an adapted hierarchical schedule. This construction enables sharper control of the model’s bias and variance, and allows us to derive tight bounds for both the learner’s static and dynamic regret {–} i.e., the regret incurred against the best dynamic policy in hindsight over the horizon of play.

----

## [384] Improving Molecular Graph Neural Network Explainability with Orthonormalization and Induced Sparsity

**Authors**: *Ryan Henderson, Djork-Arné Clevert, Floriane Montanari*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/henderson21a.html](http://proceedings.mlr.press/v139/henderson21a.html)

**Abstract**:

Rationalizing which parts of a molecule drive the predictions of a molecular graph convolutional neural network (GCNN) can be difficult. To help, we propose two simple regularization techniques to apply during the training of GCNNs: Batch Representation Orthonormalization (BRO) and Gini regularization. BRO, inspired by molecular orbital theory, encourages graph convolution operations to generate orthonormal node embeddings. Gini regularization is applied to the weights of the output layer and constrains the number of dimensions the model can use to make predictions. We show that Gini and BRO regularization can improve the accuracy of state-of-the-art GCNN attribution methods on artificial benchmark datasets. In a real-world setting, we demonstrate that medicinal chemists significantly prefer explanations extracted from regularized models. While we only study these regularizers in the context of GCNNs, both can be applied to other types of neural networks.

----

## [385] Muesli: Combining Improvements in Policy Optimization

**Authors**: *Matteo Hessel, Ivo Danihelka, Fabio Viola, Arthur Guez, Simon Schmitt, Laurent Sifre, Theophane Weber, David Silver, Hado van Hasselt*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hessel21a.html](http://proceedings.mlr.press/v139/hessel21a.html)

**Abstract**:

We propose a novel policy update that combines regularized policy optimization with model learning as an auxiliary loss. The update (henceforth Muesli) matches MuZero’s state-of-the-art performance on Atari. Notably, Muesli does so without using deep search: it acts directly with a policy network and has computation speed comparable to model-free baselines. The Atari results are complemented by extensive ablations, and by additional results on continuous control and 9x9 Go.

----

## [386] Learning Representations by Humans, for Humans

**Authors**: *Sophie Hilgard, Nir Rosenfeld, Mahzarin R. Banaji, Jack Cao, David C. Parkes*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hilgard21a.html](http://proceedings.mlr.press/v139/hilgard21a.html)

**Abstract**:

When machine predictors can achieve higher performance than the human decision-makers they support, improving the performance of human decision-makers is often conflated with improving machine accuracy. Here we propose a framework to directly support human decision-making, in which the role of machines is to reframe problems rather than to prescribe actions through prediction. Inspired by the success of representation learning in improving performance of machine predictors, our framework learns human-facing representations optimized for human performance. This “Mind Composed with Machine” framework incorporates a human decision-making model directly into the representation learning paradigm and is trained with a novel human-in-the-loop training procedure. We empirically demonstrate the successful application of the framework to various tasks and representational forms.

----

## [387] Optimizing Black-box Metrics with Iterative Example Weighting

**Authors**: *Gaurush Hiranandani, Jatin Mathur, Harikrishna Narasimhan, Mahdi Milani Fard, Sanmi Koyejo*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hiranandani21a.html](http://proceedings.mlr.press/v139/hiranandani21a.html)

**Abstract**:

We consider learning to optimize a classification metric defined by a black-box function of the confusion matrix. Such black-box learning settings are ubiquitous, for example, when the learner only has query access to the metric of interest, or in noisy-label and domain adaptation applications where the learner must evaluate the metric via performance evaluation using a small validation sample. Our approach is to adaptively learn example weights on the training dataset such that the resulting weighted objective best approximates the metric on the validation sample. We show how to model and estimate the example weights and use them to iteratively post-shift a pre-trained class probability estimator to construct a classifier. We also analyze the resulting procedure’s statistical properties. Experiments on various label noise, domain shift, and fair classification setups confirm that our proposal compares favorably to the state-of-the-art baselines for each application.

----

## [388] Trees with Attention for Set Prediction Tasks

**Authors**: *Roy Hirsch, Ran Gilad-Bachrach*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hirsch21a.html](http://proceedings.mlr.press/v139/hirsch21a.html)

**Abstract**:

In many machine learning applications, each record represents a set of items. For example, when making predictions from medical records, the medications prescribed to a patient are a set whose size is not fixed and whose order is arbitrary. However, most machine learning algorithms are not designed to handle set structures and are limited to processing records of fixed size. Set-Tree, presented in this work, extends the support for sets to tree-based models, such as Random-Forest and Gradient-Boosting, by introducing an attention mechanism and set-compatible split criteria. We evaluate the new method empirically on a wide range of problems ranging from making predictions on sub-atomic particle jets to estimating the redshift of galaxies. The new method outperforms existing tree-based methods consistently and significantly. Moreover, it is competitive and often outperforms Deep Learning. We also discuss the theoretical properties of Set-Trees and explain how they enable item-level explainability.

----

## [389] Multiplicative Noise and Heavy Tails in Stochastic Optimization

**Authors**: *Liam Hodgkinson, Michael W. Mahoney*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hodgkinson21a.html](http://proceedings.mlr.press/v139/hodgkinson21a.html)

**Abstract**:

Although stochastic optimization is central to modern machine learning, the precise mechanisms underlying its success, and in particular, the precise role of the stochasticity, still remain unclear. Modeling stochastic optimization algorithms as discrete random recurrence relations, we show that multiplicative noise, as it commonly arises due to variance in local rates of convergence, results in heavy-tailed stationary behaviour in the parameters. Theoretical results are obtained characterizing this for a large class of (non-linear and even non-convex) models and optimizers (including momentum, Adam, and stochastic Newton), demonstrating that this phenomenon holds generally. We describe dependence on key factors, including step size, batch size, and data variability, all of which exhibit similar qualitative behavior to recent empirical results on state-of-the-art neural network models. Furthermore, we empirically illustrate how multiplicative noise and heavy-tailed structure improve capacity for basin hopping and exploration of non-convex loss surfaces, over commonly-considered stochastic dynamics with only additive noise and light-tailed structure.

----

## [390] MC-LSTM: Mass-Conserving LSTM

**Authors**: *Pieter-Jan Hoedt, Frederik Kratzert, Daniel Klotz, Christina Halmich, Markus Holzleitner, Grey Nearing, Sepp Hochreiter, Günter Klambauer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hoedt21a.html](http://proceedings.mlr.press/v139/hoedt21a.html)

**Abstract**:

The success of Convolutional Neural Networks (CNNs) in computer vision is mainly driven by their strong inductive bias, which is strong enough to allow CNNs to solve vision-related tasks with random weights, meaning without learning. Similarly, Long Short-Term Memory (LSTM) has a strong inductive bias towards storing information over time. However, many real-world systems are governed by conservation laws, which lead to the redistribution of particular quantities {—} e.g.in physical and economical systems. Our novel Mass-Conserving LSTM (MC-LSTM) adheres to these conservation laws by extending the inductive bias of LSTM to model the redistribution of those stored quantities. MC-LSTMs set a new state-of-the-art for neural arithmetic units at learning arithmetic operations, such as addition tasks,which have a strong conservation law, as the sum is constant over time. Further, MC-LSTM is applied to traffic forecasting, modeling a pendulum, and a large benchmark dataset in hydrology, where it sets a new state-of-the-art for predicting peak flows. In the hydrology example, we show that MC-LSTM states correlate with real world processes and are therefore interpretable.

----

## [391] Learning Curves for Analysis of Deep Networks

**Authors**: *Derek Hoiem, Tanmay Gupta, Zhizhong Li, Michal Shlapentokh-Rothman*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hoiem21a.html](http://proceedings.mlr.press/v139/hoiem21a.html)

**Abstract**:

Learning curves model a classifier’s test error as a function of the number of training samples. Prior works show that learning curves can be used to select model parameters and extrapolate performance. We investigate how to use learning curves to evaluate design choices, such as pretraining, architecture, and data augmentation. We propose a method to robustly estimate learning curves, abstract their parameters into error and data-reliance, and evaluate the effectiveness of different parameterizations. Our experiments exemplify use of learning curves for analysis and yield several interesting observations.

----

## [392] Equivariant Learning of Stochastic Fields: Gaussian Processes and Steerable Conditional Neural Processes

**Authors**: *Peter Holderrieth, Michael J. Hutchinson, Yee Whye Teh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/holderrieth21a.html](http://proceedings.mlr.press/v139/holderrieth21a.html)

**Abstract**:

Motivated by objects such as electric fields or fluid streams, we study the problem of learning stochastic fields, i.e. stochastic processes whose samples are fields like those occurring in physics and engineering. Considering general transformations such as rotations and reflections, we show that spatial invariance of stochastic fields requires an inference model to be equivariant. Leveraging recent advances from the equivariance literature, we study equivariance in two classes of models. Firstly, we fully characterise equivariant Gaussian processes. Secondly, we introduce Steerable Conditional Neural Processes (SteerCNPs), a new, fully equivariant member of the Neural Process family. In experiments with Gaussian process vector fields, images, and real-world weather data, we observe that SteerCNPs significantly improve the performance of previous models and equivariance leads to improvements in transfer learning tasks.

----

## [393] Latent Programmer: Discrete Latent Codes for Program Synthesis

**Authors**: *Joey Hong, David Dohan, Rishabh Singh, Charles Sutton, Manzil Zaheer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hong21a.html](http://proceedings.mlr.press/v139/hong21a.html)

**Abstract**:

A key problem in program synthesis is searching over the large space of possible programs. Human programmers might decide the high-level structure of the desired program before thinking about the details; motivated by this intuition, we consider two-level search for program synthesis, in which the synthesizer first generates a plan, a sequence of symbols that describes the desired program at a high level, before generating the program. We propose to learn representations of programs that can act as plans to organize such a two-level search. Discrete latent codes are appealing for this purpose, and can be learned by applying recent work on discrete autoencoders. Based on these insights, we introduce the Latent Programmer (LP), a program synthesis method that first predicts a discrete latent code from input/output examples, and then generates the program in the target language. We evaluate the LP on two domains, demonstrating that it yields an improvement in accuracy, especially on longer programs for which search is most difficult.

----

## [394] Chebyshev Polynomial Codes: Task Entanglement-based Coding for Distributed Matrix Multiplication

**Authors**: *Sangwoo Hong, Heecheol Yang, Youngseok Yoon, Taehyun Cho, Jungwoo Lee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hong21b.html](http://proceedings.mlr.press/v139/hong21b.html)

**Abstract**:

Distributed computing has been a prominent solution to efficiently process massive datasets in parallel. However, the existence of stragglers is one of the major concerns that slows down the overall speed of distributed computing. To deal with this problem, we consider a distributed matrix multiplication scenario where a master assigns multiple tasks to each worker to exploit stragglers’ computing ability (which is typically wasted in conventional distributed computing). We propose Chebyshev polynomial codes, which can achieve order-wise improvement in encoding complexity at the master and communication load in distributed matrix multiplication using task entanglement. The key idea of task entanglement is to reduce the number of encoded matrices for multiple tasks assigned to each worker by intertwining encoded matrices. We experimentally demonstrate that, in cloud environments, Chebyshev polynomial codes can provide significant reduction in overall processing time in distributed computing for matrix multiplication, which is a key computational component in modern deep learning.

----

## [395] Federated Learning of User Verification Models Without Sharing Embeddings

**Authors**: *Hossein Hosseini, Hyunsin Park, Sungrack Yun, Christos Louizos, Joseph Soriaga, Max Welling*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hosseini21a.html](http://proceedings.mlr.press/v139/hosseini21a.html)

**Abstract**:

We consider the problem of training User Verification (UV) models in federated setup, where each user has access to the data of only one class and user embeddings cannot be shared with the server or other users. To address this problem, we propose Federated User Verification (FedUV), a framework in which users jointly learn a set of vectors and maximize the correlation of their instance embeddings with a secret linear combination of those vectors. We show that choosing the linear combinations from the codewords of an error-correcting code allows users to collaboratively train the model without revealing their embedding vectors. We present the experimental results for user verification with voice, face, and handwriting data and show that FedUV is on par with existing approaches, while not sharing the embeddings with other users or the server.

----

## [396] The Limits of Min-Max Optimization Algorithms: Convergence to Spurious Non-Critical Sets

**Authors**: *Ya-Ping Hsieh, Panayotis Mertikopoulos, Volkan Cevher*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hsieh21a.html](http://proceedings.mlr.press/v139/hsieh21a.html)

**Abstract**:

Compared to minimization, the min-max optimization in machine learning applications is considerably more convoluted because of the existence of cycles and similar phenomena. Such oscillatory behaviors are well-understood in the convex-concave regime, and many algorithms are known to overcome them. In this paper, we go beyond this basic setting and characterize the convergence properties of many popular methods in solving non-convex/non-concave problems. In particular, we show that a wide class of state-of-the-art schemes and heuristics may converge with arbitrarily high probability to attractors that are in no way min-max optimal or even stationary. Our work thus points out a potential pitfall among many existing theoretical frameworks, and we corroborate our theoretical claims by explicitly showcasing spurious attractors in simple two-dimensional problems.

----

## [397] Near-Optimal Representation Learning for Linear Bandits and Linear RL

**Authors**: *Jiachen Hu, Xiaoyu Chen, Chi Jin, Lihong Li, Liwei Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hu21a.html](http://proceedings.mlr.press/v139/hu21a.html)

**Abstract**:

This paper studies representation learning for multi-task linear bandits and multi-task episodic RL with linear value function approximation. We first consider the setting where we play $M$ linear bandits with dimension $d$ concurrently, and these bandits share a common $k$-dimensional linear representation so that $k\ll d$ and $k \ll M$. We propose a sample-efficient algorithm, MTLR-OFUL, which leverages the shared representation to achieve $\tilde{O}(M\sqrt{dkT} + d\sqrt{kMT} )$ regret, with $T$ being the number of total steps. Our regret significantly improves upon the baseline $\tilde{O}(Md\sqrt{T})$ achieved by solving each task independently. We further develop a lower bound that shows our regret is near-optimal when $d > M$. Furthermore, we extend the algorithm and analysis to multi-task episodic RL with linear value function approximation under low inherent Bellman error (Zanette et al., 2020a). To the best of our knowledge, this is the first theoretical result that characterize the benefits of multi-task representation learning for exploration in RL with function approximation.

----

## [398] On the Random Conjugate Kernel and Neural Tangent Kernel

**Authors**: *Zhengmian Hu, Heng Huang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hu21b.html](http://proceedings.mlr.press/v139/hu21b.html)

**Abstract**:

We investigate the distributions of Conjugate Kernel (CK) and Neural Tangent Kernel (NTK) for ReLU networks with random initialization. We derive the precise distributions and moments of the diagonal elements of these kernels. For a feedforward network, these values converge in law to a log-normal distribution when the network depth $d$ and width $n$ simultaneously tend to infinity and the variance of log diagonal elements is proportional to ${d}/{n}$. For the residual network, in the limit that number of branches $m$ increases to infinity and the width $n$ remains fixed, the diagonal elements of Conjugate Kernel converge in law to a log-normal distribution where the variance of log value is proportional to ${1}/{n}$, and the diagonal elements of NTK converge in law to a log-normal distributed variable times the conjugate kernel of one feedforward network. Our new theoretical analysis results suggest that residual network remains trainable in the limit of infinite branches and fixed network width. The numerical experiments are conducted and all results validate the soundness of our theoretical analysis.

----

## [399] Off-Belief Learning

**Authors**: *Hengyuan Hu, Adam Lerer, Brandon Cui, Luis Pineda, Noam Brown, Jakob N. Foerster*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hu21c.html](http://proceedings.mlr.press/v139/hu21c.html)

**Abstract**:

The standard problem setting in Dec-POMDPs is self-play, where the goal is to find a set of policies that play optimally together. Policies learned through self-play may adopt arbitrary conventions and implicitly rely on multi-step reasoning based on fragile assumptions about other agents’ actions and thus fail when paired with humans or independently trained agents at test time. To address this, we present off-belief learning (OBL). At each timestep OBL agents follow a policy $\pi_1$ that is optimized assuming past actions were taken by a given, fixed policy ($\pi_0$), but assuming that future actions will be taken by $\pi_1$. When $\pi_0$ is uniform random, OBL converges to an optimal policy that does not rely on inferences based on other agents’ behavior (an optimal grounded policy). OBL can be iterated in a hierarchy, where the optimal policy from one level becomes the input to the next, thereby introducing multi-level cognitive reasoning in a controlled manner. Unlike existing approaches, which may converge to any equilibrium policy, OBL converges to a unique policy, making it suitable for zero-shot coordination (ZSC). OBL can be scaled to high-dimensional settings with a fictitious transition mechanism and shows strong performance in both a toy-setting and the benchmark human-AI & ZSC problem Hanabi.

----



[Go to the previous page](ICML-2021-list01.md)

[Go to the next page](ICML-2021-list03.md)

[Go to the catalog section](README.md)