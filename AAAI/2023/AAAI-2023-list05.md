## [800] Wiener Graph Deconvolutional Network Improves Graph Self-Supervised Learning

        **Authors**: *Jiashun Cheng, Man Li, Jia Li, Fugee Tsung*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25870](https://doi.org/10.1609/aaai.v37i6.25870)

        **Abstract**:

        Graph self-supervised learning (SSL) has been vastly employed to learn representations from unlabeled graphs. Existing methods can be roughly divided into predictive learning and contrastive learning, where the latter one attracts more research attention with better empirical performance. We argue that, however, predictive models weaponed with powerful decoder could achieve comparable or even better representation power than contrastive models. In this work, we propose a Wiener Graph Deconvolutional Network (WGDN), an augmentation-adaptive decoder empowered by graph wiener filter to perform information reconstruction. Theoretical analysis proves the superior reconstruction ability of graph wiener filter. Extensive experimental results on various datasets demonstrate the effectiveness of our approach.

        ----

        ## [801] Partial-Label Regression

        **Authors**: *Xin Cheng, Deng-Bao Wang, Lei Feng, Min-Ling Zhang, Bo An*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25871](https://doi.org/10.1609/aaai.v37i6.25871)

        **Abstract**:

        Partial-label learning is a popular weakly supervised learning setting that allows each training example to be annotated with a set of candidate labels. Previous studies on partial-label learning only focused on the classification setting where candidate labels are all discrete, which cannot handle continuous labels with real values. In this paper, we provide the first attempt to investigate partial-label regression, where each training example is annotated with a set of real-valued candidate labels. To solve this problem, we first propose a simple baseline method that takes the average loss incurred by candidate labels as the predictive loss. The drawback of this method lies in that the loss incurred by the true label may be overwhelmed by other false labels. To overcome this drawback, we propose an identification method that takes the least loss incurred by candidate labels as the predictive loss. We further improve it by proposing a progressive identification method to differentiate candidate labels using progressively updated weights for incurred losses. We prove that the latter two methods are model-consistent and provide convergence analysis showing the optimal parametric convergence rate. Our proposed methods are theoretically grounded and can be compatible with any models, optimizers, and losses. Experiments validate the effectiveness of our proposed methods.

        ----

        ## [802] Offline Quantum Reinforcement Learning in a Conservative Manner

        **Authors**: *Zhihao Cheng, Kaining Zhang, Li Shen, Dacheng Tao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25872](https://doi.org/10.1609/aaai.v37i6.25872)

        **Abstract**:

        Recently, to reap the quantum advantage, empowering reinforcement learning (RL) with quantum computing has attracted much attention, which is dubbed as quantum RL (QRL). However, current QRL algorithms employ an online learning scheme, i.e., the policy that is run on a quantum computer needs to interact with the environment to collect experiences, which could be expensive and dangerous for practical applications. In this paper, we aim to solve this problem in an offline learning manner. To be more specific, we develop the first offline quantum RL (offline QRL) algorithm named CQ2L (Conservative Quantum Q-learning), which learns from offline samples and does not require any interaction with the environment. CQ2L utilizes variational quantum circuits (VQCs), which are improved with data re-uploading and scaling parameters, to represent Q-value functions of agents. To suppress the overestimation of Q-values resulting from offline data, we first employ a double Q-learning framework to reduce the overestimation bias; then a penalty term that encourages generating conservative Q-values is designed. We conduct abundant experiments to demonstrate that the proposed method CQ2L can successfully solve offline QRL tasks that the online counterpart could not.

        ----

        ## [803] Variational Wasserstein Barycenters with C-cyclical Monotonicity Regularization

        **Authors**: *Jinjin Chi, Zhiyao Yang, Ximing Li, Jihong Ouyang, Renchu Guan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25873](https://doi.org/10.1609/aaai.v37i6.25873)

        **Abstract**:

        Wasserstein barycenter, built on the theory of Optimal Transport (OT), provides a powerful framework to aggregate probability distributions, and it has increasingly attracted great attention within the machine learning community. However, it is often intractable to precisely compute, especially for high dimensional and continuous settings. To alleviate this problem, we develop a novel regularization by using the fact that c-cyclical monotonicity is often necessary and sufficient conditions for optimality in OT problems, and incorporate it into the dual formulation of Wasserstein barycenters. For efficient computations, we adopt a variational distribution as the approximation of the true continuous barycenter, so as to frame the Wasserstein barycenters problem as an optimization problem with respect to variational parameters. Upon those ideas, we propose a novel end-to-end continuous approximation method, namely Variational Wasserstein Barycenters with c-Cyclical Monotonicity Regularization (VWB-CMR), given sample access to the input distributions. We show theoretical convergence analysis and demonstrate the superior performance of VWB-CMR on synthetic data and real applications of subset posterior aggregation.

        ----

        ## [804] MobileTL: On-Device Transfer Learning with Inverted Residual Blocks

        **Authors**: *Hung-Yueh Chiang, Natalia Frumkin, Feng Liang, Diana Marculescu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25874](https://doi.org/10.1609/aaai.v37i6.25874)

        **Abstract**:

        Transfer learning on edge is challenging due to on-device limited resources. Existing work addresses this issue by training a subset of parameters or adding model patches. Developed with inference in mind, Inverted Residual Blocks (IRBs) split a convolutional layer into depthwise and pointwise convolutions, leading to more stacking layers, e.g., convolution, normalization, and activation layers. Though they are efficient for inference, IRBs require that additional activation maps are stored in memory for training weights for convolution layers and scales for normalization layers. As a result, their high memory cost prohibits training IRBs on resource-limited edge devices, and making them unsuitable in the context of transfer learning. To address this issue, we present MobileTL, a memory and computationally efficient on-device transfer learning method for models built with IRBs. MobileTL trains the shifts for internal normalization layers to avoid storing activation maps for the backward pass. Also, MobileTL approximates the backward computation of the activation layer (e.g., Hard-Swish and ReLU6) as a signed function which enables storing a binary mask instead of activation maps for the backward pass. MobileTL fine-tunes a few top blocks (close to output) rather than propagating the gradient through the whole network to reduce the computation cost. Our method reduces memory usage by 46% and 53% for MobileNetV2 and V3 IRBs, respectively. For MobileNetV3, we observe a 36% reduction in floating-point operations (FLOPs) when fine-tuning 5 blocks, while only incurring a 0.6% accuracy reduction on CIFAR10. Extensive experiments on multiple datasets demonstrate that our method is Pareto-optimal (best accuracy under given hardware constraints) compared to prior work in transfer learning for edge devices.

        ----

        ## [805] Learning Optimal Features via Partial Invariance

        **Authors**: *Moulik Choraria, Ibtihal Ferwana, Ankur Mani, Lav R. Varshney*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25875](https://doi.org/10.1609/aaai.v37i6.25875)

        **Abstract**:

        Learning models that are robust to distribution shifts is a key concern in the context of their real-life applicability. Invariant Risk Minimization (IRM) is a popular framework that aims to learn robust models from multiple environments. The success of IRM requires an important assumption: the underlying causal mechanisms/features remain invariant across environments. When not satisfied, we show that IRM can over-constrain the predictor and to remedy this, we propose a relaxation via partial invariance. In this work, we theoretically highlight the sub-optimality of IRM and then demonstrate how learning from a partition of training domains can help improve invariant models. Several experiments, conducted both in linear settings as well as with deep neural networks on tasks over both language and image data, allow us to verify our conclusions.

        ----

        ## [806] PrimeNet: Pre-training for Irregular Multivariate Time Series

        **Authors**: *Ranak Roy Chowdhury, Jiacheng Li, Xiyuan Zhang, Dezhi Hong, Rajesh K. Gupta, Jingbo Shang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25876](https://doi.org/10.1609/aaai.v37i6.25876)

        **Abstract**:

        Real-world applications often involve irregular time series, for which the time intervals between successive observations are non-uniform. Irregularity across multiple features in a multi-variate time series further results in a different subset of features at any given time (i.e., asynchronicity). Existing pre-training schemes for time-series, however, often assume regularity of time series and make no special treatment of irregularity. We argue that such irregularity offers insight about domain property of the data—for example, frequency of hospital visits may signal patient health condition—that can guide representation learning. In this work, we propose PrimeNet to learn a self-supervised representation for irregular multivariate time-series. Specifically, we design a time sensitive contrastive learning and data reconstruction task to pre-train a model. Irregular time-series exhibits considerable variations in sampling density over time. Hence, our triplet generation strategy follows the density of the original data points, preserving its native irregularity. Moreover, the sampling density variation over time makes data reconstruction difficult for different regions. Therefore, we design a data masking technique that always masks a constant time duration to accommodate reconstruction for regions of different sampling density. We learn with these tasks using unlabeled data to build a pre-trained model and fine-tune on a downstream task with limited labeled data, in contrast with existing fully supervised approach for irregular time-series, requiring large amounts of labeled data. Experiment results show that PrimeNet significantly outperforms state-of-the-art methods on naturally irregular and asynchronous data from Healthcare and IoT applications for several downstream tasks, including classification, interpolation, and regression.

        ----

        ## [807] Structured BFGS Method for Optimal Doubly Stochastic Matrix Approximation

        **Authors**: *Dejun Chu, Changshui Zhang, Shiliang Sun, Qing Tao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25877](https://doi.org/10.1609/aaai.v37i6.25877)

        **Abstract**:

        Doubly stochastic matrix plays an essential role in several areas such as statistics and machine learning. In this paper we consider the optimal approximation of a square matrix in the set of doubly stochastic matrices. A structured BFGS method is proposed to solve the dual of the primal problem. The resulting algorithm builds curvature information into the diagonal components of the true Hessian, so that it takes only additional linear cost to obtain the descent direction based on the gradient information without having to explicitly store the inverse Hessian approximation. The cost is substantially fewer than quadratic complexity of the classical BFGS algorithm. Meanwhile, a Newton-based line search method is presented for finding a suitable step size, which in practice uses the existing knowledge and takes only one iteration. The global convergence of our algorithm is established. We verify the advantages of our approach on both synthetic data and real data sets. The experimental results demonstrate that our algorithm outperforms the state-of-the-art solvers and enjoys outstanding scalability.

        ----

        ## [808] On the Complexity of PAC Learning in Hilbert Spaces

        **Authors**: *Sergei Chubanov*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25878](https://doi.org/10.1609/aaai.v37i6.25878)

        **Abstract**:

        We study the problem of binary classification from the point of
view of learning convex polyhedra in Hilbert spaces, to which
one can reduce any binary classification problem. The problem
of learning convex polyhedra in finite-dimensional spaces is
sufficiently well studied in the literature. We generalize this
problem to that in a Hilbert space and propose an algorithm
for learning a polyhedron which correctly classifies at least
1 − ε of the distribution, with a probability of at least 1 − δ,
where ε and δ are given parameters. Also, as a corollary, we
improve some previous bounds for polyhedral classification
in finite-dimensional spaces.

        ----

        ## [809] Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks Using an Incompetent Teacher

        **Authors**: *Vikram S. Chundawat, Ayush K. Tarun, Murari Mandal, Mohan S. Kankanhalli*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25879](https://doi.org/10.1609/aaai.v37i6.25879)

        **Abstract**:

        Machine unlearning has become an important area of research due to an increasing need for machine learning (ML) applications to comply with the emerging data privacy regulations. It facilitates the provision for removal of certain set or class of data from an already trained ML model without requiring retraining from scratch. Recently, several efforts have been put in to make unlearning to be effective and efficient. We propose a novel machine unlearning method by exploring the utility of competent and incompetent teachers in a student-teacher framework to induce forgetfulness. The knowledge from the competent and incompetent teachers is selectively transferred to the student to obtain a model that doesn't contain any information about the forget data. We experimentally show that this method generalizes well, is fast and effective. Furthermore, we introduce the zero retrain forgetting (ZRF) metric to evaluate any unlearning method. Unlike the existing unlearning metrics, the ZRF score does not depend on the availability of the expensive retrained model. This makes it useful for analysis of the unlearned model after deployment as well. We present results of experiments conducted for random subset forgetting and class forgetting on various deep networks and across different application domains. Code is at: https://github.com/vikram2000b/bad-teaching- unlearning

        ----

        ## [810] Scalable Spatiotemporal Graph Neural Networks

        **Authors**: *Andrea Cini, Ivan Marisca, Filippo Maria Bianchi, Cesare Alippi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25880](https://doi.org/10.1609/aaai.v37i6.25880)

        **Abstract**:

        Neural forecasting of spatiotemporal time series drives both research and industrial innovation in several relevant application domains. Graph neural networks (GNNs) are often the core component of the forecasting architecture. However, in most spatiotemporal GNNs, the computational complexity scales up to a quadratic factor with the length of the sequence times the number of links in the graph, hence hindering the application of these models to large graphs and long temporal sequences. While methods to improve scalability have been proposed in the context of static graphs, few research efforts have been devoted to the spatiotemporal case. To fill this gap, we propose a scalable architecture that exploits an efficient encoding of both temporal and spatial dynamics. In particular, we use a randomized recurrent neural network to embed the history of the input time series into high-dimensional state representations encompassing multi-scale temporal dynamics. Such representations are then propagated along the spatial dimension using different powers of the graph adjacency matrix to generate node embeddings characterized by a rich pool of spatiotemporal features. The resulting node embeddings can be efficiently pre-computed in an unsupervised manner, before being fed to a feed-forward decoder that learns to map the multi-scale spatiotemporal representations to predictions. The training procedure can then be parallelized node-wise by sampling the node embeddings without breaking any dependency, thus enabling scalability to large networks. Empirical results on relevant datasets show that our approach achieves results competitive with the state of the art, while dramatically reducing the computational burden.

        ----

        ## [811] Exploiting Multiple Abstractions in Episodic RL via Reward Shaping

        **Authors**: *Roberto Cipollone, Giuseppe De Giacomo, Marco Favorito, Luca Iocchi, Fabio Patrizi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25881](https://doi.org/10.1609/aaai.v37i6.25881)

        **Abstract**:

        One major limitation to the applicability of Reinforcement Learning (RL) to many practical domains is the large number of samples required to learn an optimal policy. To address this problem and improve learning efficiency, we consider a linear hierarchy of abstraction layers of the Markov Decision Process (MDP) underlying the target domain. Each layer is an MDP representing a coarser model of the one immediately below in the hierarchy. In this work, we propose a novel form of Reward Shaping where the solution obtained at the abstract level is used to offer rewards to the more concrete MDP, in such a way that the abstract solution guides the learning in the more complex domain. In contrast with other works in Hierarchical RL, our technique has few requirements in the design of the abstract models and it is also tolerant to modeling errors, thus making the proposed approach practical. We formally analyze the relationship between the abstract models and the exploration heuristic induced in the lower-level domain. Moreover, we prove that the method guarantees optimal convergence and we demonstrate its effectiveness experimentally.

        ----

        ## [812] Tricking the Hashing Trick: A Tight Lower Bound on the Robustness of CountSketch to Adaptive Inputs

        **Authors**: *Edith Cohen, Jelani Nelson, Tamás Sarlós, Uri Stemmer*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25882](https://doi.org/10.1609/aaai.v37i6.25882)

        **Abstract**:

        CountSketch and Feature Hashing  (the ``hashing trick'')  are popular randomized dimensionality reduction methods that support recovery of l2 -heavy hitters and approximate inner products. When the inputs are not adaptive (do not depend on prior outputs), classic estimators applied to a sketch of size O(l / epsilon) are accurate for a number of queries that is exponential in l.  When inputs are adaptive, however, an adversarial input can be constructed after O(l) queries with the classic estimator and the best known robust estimator only supports ~O(l^2) queries.  In this work we show that this quadratic dependence is in a sense inherent: We design an attack that after O(l^2) queries produces an adversarial input vector whose sketch is highly biased. Our attack uses ``natural'' non-adaptive inputs (only the final adversarial input is chosen adaptively) and universally applies  with any correct estimator, including one that is unknown to the attacker. In that, we expose inherent vulnerability of this fundamental method.

        ----

        ## [813] Continuous Mixtures of Tractable Probabilistic Models

        **Authors**: *Alvaro H. C. Correia, Gennaro Gala, Erik Quaeghebeur, Cassio P. de Campos, Robert Peharz*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25883](https://doi.org/10.1609/aaai.v37i6.25883)

        **Abstract**:

        Probabilistic models based on continuous latent spaces, such as variational autoencoders, can be understood as uncountable mixture models where components depend continuously on the latent code. They have proven to be expressive tools for generative and probabilistic modelling, but are at odds with tractable probabilistic inference, that is, computing marginals and conditionals of the represented probability distribution. Meanwhile, tractable probabilistic models such as probabilistic circuits (PCs) can be understood as hierarchical discrete mixture models, and thus are capable of performing exact inference efficiently but often show subpar performance in comparison to continuous latent-space models. In this paper, we investigate a hybrid approach, namely continuous mixtures of tractable models with a small latent dimension. While these models are analytically intractable, they are well amenable to numerical integration schemes based on a finite set of integration points. With a large enough number of integration points the approximation becomes de-facto exact. Moreover, for a finite set of integration points, the integration method effectively compiles the continuous mixture into a standard PC. In experiments, we show that this simple scheme proves remarkably effective, as PCs learnt this way set new state of the art for tractable models on many standard density estimation benchmarks.

        ----

        ## [814] End-to-End Learning for Optimization via Constraint-Enforcing Approximators

        **Authors**: *Rares Cristian, Pavithra Harsha, Georgia Perakis, Brian Leo Quanz, Ioannis Spantidakis*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25884](https://doi.org/10.1609/aaai.v37i6.25884)

        **Abstract**:

        In many real-world applications, predictive methods are used to provide inputs for downstream optimization problems. It has been shown that using the downstream task-based objective to learn the intermediate predictive model is often better than using only intermediate task objectives, such as prediction error. The learning task in the former approach is referred to as end-to-end learning. The difficulty in end-to-end learning lies in differentiating through the optimization problem. Therefore, we propose a neural network architecture that can learn to approximately solve these optimization problems, particularly ensuring its output satisfies the feasibility constraints via alternate projections. We show these projections converge at a geometric rate to the exact projection. Our approach is more computationally efficient than existing methods as we do not need to solve the original optimization problem at each iteration. Furthermore, our approach can be applied to a wider range of optimization problems. We apply this to a shortest path problem for which the first stage forecasting problem is a computer vision task of predicting edge costs from terrain maps, a capacitated multi-product newsvendor problem, and a maximum matching problem. We show that this method out-performs existing approaches in terms of final task-based loss and training time.

        ----

        ## [815] Practical Parallel Algorithms for Submodular Maximization Subject to a Knapsack Constraint with Nearly Optimal Adaptivity

        **Authors**: *Shuang Cui, Kai Han, Jing Tang, He Huang, Xueying Li, Aakas Zhiyuli*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25885](https://doi.org/10.1609/aaai.v37i6.25885)

        **Abstract**:

        Submodular maximization has wide applications in machine learning and data mining, where massive datasets have brought the great need for designing efficient and parallelizable algorithms. One measure of the parallelizability of a submodular maximization algorithm is its adaptivity complexity, which indicates the number of sequential rounds where a polynomial number of queries to the objective function can be executed in parallel. In this paper, we study the problem of non-monotone submodular maximization subject to a knapsack constraint, and propose the first combinatorial algorithm achieving an (8+epsilon)-approximation under O(log n) adaptive complexity, which is optimal up to a factor of O(loglog n). Moreover, under slightly larger adaptivity, we also propose approximation algorithms with nearly optimal query complexity of O(n), while achieving better approximation ratios. We show that our algorithms can also be applied to the special case of submodular maximization subject to a cardinality constraint, and achieve performance bounds comparable with those of state-of-the-art algorithms. Finally, the effectiveness of our approach is demonstrated by extensive experiments on real-world applications.

        ----

        ## [816] Opposite Online Learning via Sequentially Integrated Stochastic Gradient Descent Estimators

        **Authors**: *Wenhai Cui, Xiaoting Ji, Linglong Kong, Xiaodong Yan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25886](https://doi.org/10.1609/aaai.v37i6.25886)

        **Abstract**:

        Stochastic gradient descent  algorithm  (SGD)  has been popular in various fields of artificial intelligence as well as a prototype of online learning algorithms. This article proposes a novel and general framework of one-sided testing for streaming data based on SGD, which  determines whether the unknown parameter is greater than a certain positive constant. We construct the online-updated test statistic sequentially by integrating the selected batch-specific estimator or its opposite, which is referred to opposite online learning. The batch-specific online estimators are chosen strategically according to the proposed sequential tactics designed by two-armed bandit process. Theoretical results prove the advantage of the strategy ensuring the distribution of test statistic to be optimal under the null hypothesis and also supply the theoretical evidence of power enhancement compared with classical test statistic. In application, the proposed method is appealing for statistical inference of one-sided testing because it is scalable for any model. Finally, the superior finite-sample performance is evaluated by simulation studies.

        ----

        ## [817] Contrastive Learning with the Feature Reconstruction Amplifier

        **Authors**: *Wentao Cui, Liang Bai*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25887](https://doi.org/10.1609/aaai.v37i6.25887)

        **Abstract**:

        Contrastive learning has emerged as one of the most promising self-supervised methods. It can efficiently learn the transferable representations of samples through the instance-level discrimination task. In general, the performance of the contrastive learning method can be further improved by projecting the transferable high-dimensional representations into the low-dimensional feature space. This is because the model can learn more abstract discriminative information. However, when low-dimensional features cannot provide sufficient discriminative information to the model (e.g., the samples are very similar to each other), the existing contrastive learning method will be limited to a great extent. Therefore, in this paper, we propose a general module called the Feature Reconstruction Amplifier (FRA) for adding additional high-dimensional feature information to the model. Specifically, FRA reconstructs the low-dimensional feature embeddings with Gaussian noise vectors and projects them to a high-dimensional reconstruction space. In this reconstruction space, we can add additional feature information through the designed loss. We have verified the effectiveness of the module itself through exhaustive ablation experiments. In addition, we perform linear evaluation and transfer learning on five common visual datasets, the experimental results demonstrate that our method is superior to recent advanced contrastive learning methods.

        ----

        ## [818] Augmented Proximal Policy Optimization for Safe Reinforcement Learning

        **Authors**: *Juntao Dai, Jiaming Ji, Long Yang, Qian Zheng, Gang Pan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25888](https://doi.org/10.1609/aaai.v37i6.25888)

        **Abstract**:

        Safe reinforcement learning considers practical scenarios that maximize the return while satisfying safety constraints. Current algorithms, which suffer from training oscillations or approximation errors, still struggle to update the policy efficiently with precise constraint satisfaction. In this article, we propose Augmented Proximal Policy Optimization (APPO), which augments the Lagrangian function of the primal constrained problem via attaching a quadratic deviation term. The constructed multiplier-penalty function dampens cost oscillation for stable convergence while being equivalent to the primal constrained problem to precisely control safety costs. APPO alternately updates the policy and the Lagrangian multiplier via solving the constructed augmented primal-dual problem, which can be easily implemented by any first-order optimizer. We apply our APPO methods in diverse safety-constrained tasks, setting a new state of the art compared with a comprehensive list of safe RL baselines. Extensive experiments verify the merits of our method in easy implementation, stable convergence, and precise cost control.

        ----

        ## [819] GradPU: Positive-Unlabeled Learning via Gradient Penalty and Positive Upweighting

        **Authors**: *Songmin Dai, Xiaoqiang Li, Yue Zhou, Xichen Ye, Tong Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25889](https://doi.org/10.1609/aaai.v37i6.25889)

        **Abstract**:

        Positive-unlabeled learning is an essential problem in many real-world applications with only labeled positive and unlabeled data, especially when the negative samples are difficult to identify. Most existing positive-unlabeled learning methods will inevitably overfit the positive class to some extent due to the existence of unidentified positive samples. This paper first analyzes the overfitting problem and proposes to bound the generalization errors via Wasserstein distances. Based on that, we develop a simple yet effective positive-unlabeled learning method, GradPU, which consists of two key ingredients: A gradient-based regularizer that penalizes the gradient norms in the interpolated data region, which improves the generalization of positive class; An unnormalized upweighting mechanism that assigns larger weights to those positive samples that are hard, not-well-fitted and less frequently labeled. It enforces the training error of each positive sample to be small and increases the robustness to the labeling bias. We evaluate our proposed GradPU on three datasets: MNIST, FashionMNIST, and CIFAR10. The results demonstrate that GradPU achieves state-of-the-art performance on both unbiased and biased positive labeling scenarios.

        ----

        ## [820] Semi-Supervised Deep Regression with Uncertainty Consistency and Variational Model Ensembling via Bayesian Neural Networks

        **Authors**: *Weihang Dai, Xiaomeng Li, Kwang-Ting Cheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25890](https://doi.org/10.1609/aaai.v37i6.25890)

        **Abstract**:

        Deep regression is an important problem with numerous applications. These range from computer vision tasks such as age estimation from photographs, to medical tasks such as ejection fraction estimation from echocardiograms for disease tracking.
Semi-supervised approaches for deep regression are notably under-explored compared to classification and segmentation tasks, however. 
Unlike classification tasks, which rely on thresholding functions for generating class pseudo-labels, regression tasks use real number target predictions directly as pseudo-labels, making them more sensitive to prediction quality.
In this work, we propose a novel approach to semi-supervised regression, namely Uncertainty-Consistent Variational Model Ensembling (UCVME), which improves training by generating high-quality pseudo-labels and uncertainty estimates for heteroscedastic regression.
Given that aleatoric uncertainty is only dependent on input data by definition and should be equal for the same inputs, we present a novel uncertainty consistency loss for co-trained models.
Our consistency loss significantly improves uncertainty estimates and allows higher quality pseudo-labels to be assigned greater importance under heteroscedastic regression. 
Furthermore, we introduce a novel variational model ensembling approach to reduce prediction noise and generate more robust pseudo-labels. We analytically show our method generates higher quality targets for unlabeled data and further improves training. 
Experiments show that our method outperforms state-of-the-art alternatives on different tasks and can be competitive with supervised methods that use full labels. Code is available at https://github.com/xmed-lab/UCVME.

        ----

        ## [821] Tackling Data Heterogeneity in Federated Learning with Class Prototypes

        **Authors**: *Yutong Dai, Zeyuan Chen, Junnan Li, Shelby Heinecke, Lichao Sun, Ran Xu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25891](https://doi.org/10.1609/aaai.v37i6.25891)

        **Abstract**:

        Data heterogeneity across clients in federated learning (FL) settings is a widely acknowledged challenge. In response, personalized federated learning (PFL) emerged as a framework to curate local models for clients' tasks. In PFL, a common strategy is to develop local and global models jointly - the global model (for generalization) informs the local models, and the local models (for personalization) are aggregated to update the global model. A key observation is that if we can improve the generalization ability of local models, then we can improve the generalization of global models, which in turn builds better personalized models. In this work, we consider class imbalance, an overlooked type of data heterogeneity, in the classification setting. We propose FedNH, a novel method that improves the local models' performance for both personalization and generalization by combining the uniformity and semantics of class prototypes. FedNH initially distributes class prototypes uniformly in the latent space and smoothly infuses the class semantics into class prototypes. We show that imposing uniformity helps to combat prototype collapse while infusing class semantics improves local models. Extensive experiments were conducted on popular classification datasets under the cross-device setting. Our results demonstrate the effectiveness and stability of our method over recent works.

        ----

        ## [822] CrysGNN: Distilling Pre-trained Knowledge to Enhance Property Prediction for Crystalline Materials

        **Authors**: *Kishalay Das, Bidisha Samanta, Pawan Goyal, Seung-Cheol Lee, Satadeep Bhattacharjee, Niloy Ganguly*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25892](https://doi.org/10.1609/aaai.v37i6.25892)

        **Abstract**:

        In recent years, graph neural network (GNN) based approaches
have emerged as a powerful technique to encode complex
topological structure of crystal materials in an enriched repre-
sentation space. These models are often supervised in nature
and using the property-specific training data, learn relation-
ship between crystal structure and different properties like
formation energy, bandgap, bulk modulus, etc. Most of these
methods require a huge amount of property-tagged data to
train the system which may not be available for different prop-
erties. However, there is an availability of a huge amount
of crystal data with its chemical composition and structural
bonds. To leverage these untapped data, this paper presents
CrysGNN, a new pre-trained GNN framework for crystalline
materials, which captures both node and graph level structural
information of crystal graphs using a huge amount of unla-
belled material data. Further, we extract distilled knowledge
from CrysGNN and inject into different state of the art prop-
erty predictors to enhance their property prediction accuracy.
We conduct extensive experiments to show that with distilled
knowledge from the pre-trained model, all the SOTA algo-
rithms are able to outperform their own vanilla version with
good margins. We also observe that the distillation process
provides significant improvement over the conventional ap-
proach of finetuning the pre-trained model. We will release the
pre-trained model along with the large dataset of 800K crys-
tal graph which we carefully curated; so that the pre-trained
model can be plugged into any existing and upcoming models
to enhance their prediction accuracy.

        ----

        ## [823] Non-reversible Parallel Tempering for Deep Posterior Approximation

        **Authors**: *Wei Deng, Qian Zhang, Qi Feng, Faming Liang, Guang Lin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25893](https://doi.org/10.1609/aaai.v37i6.25893)

        **Abstract**:

        Parallel tempering (PT), also known as replica exchange, is the go-to workhorse for simulations of multi-modal distributions. The key to the success of PT is to adopt efficient swap schemes. The popular deterministic even-odd (DEO) scheme exploits the non-reversibility property and has successfully reduced the communication cost from quadratic to linear given the sufficiently many chains. However, such an innovation largely disappears in big data due to the limited chains and few bias-corrected swaps. To handle this issue, we generalize the DEO scheme to promote non-reversibility and propose a few solutions to tackle the underlying bias caused by the geometric stopping time. Notably, in big data scenarios, we obtain a nearly linear communication cost based on the optimal window size. In addition, we also adopt stochastic gradient descent (SGD) with large and constant learning rates as exploration kernels. Such a user-friendly nature enables us to conduct approximation tasks for complex posteriors without much tuning costs.

        ----

        ## [824] Stability-Based Generalization Analysis of the Asynchronous Decentralized SGD

        **Authors**: *Xiaoge Deng, Tao Sun, Shengwei Li, Dongsheng Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25894](https://doi.org/10.1609/aaai.v37i6.25894)

        **Abstract**:

        The generalization ability often determines the success of machine learning algorithms in practice. Therefore, it is of great theoretical and practical importance to understand and bound the generalization error of machine learning algorithms. In this paper, we provide the first generalization results of the popular stochastic gradient descent (SGD) algorithm in the distributed asynchronous decentralized setting. Our analysis is based on the uniform stability tool, where stable means that the learned model does not change much in small variations of the training set. Under some mild assumptions, we perform a comprehensive generalizability analysis of the asynchronous decentralized SGD, including generalization error and excess generalization error bounds for the strongly convex, convex, and non-convex cases. Our theoretical results reveal the effects of the learning rate, training data size, training iterations, decentralized communication topology, and asynchronous delay on the generalization performance of the asynchronous decentralized SGD. We also study the optimization error regarding the objective function values and investigate how the initial point affects the excess generalization error. Finally, we conduct extensive experiments on MNIST, CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets to validate the theoretical findings.

        ----

        ## [825] Integer Subspace Differential Privacy

        **Authors**: *Prathamesh Dharangutte, Jie Gao, Ruobin Gong, Fang-Yi Yu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25895](https://doi.org/10.1609/aaai.v37i6.25895)

        **Abstract**:

        We propose new differential privacy solutions for when external invariants and integer constraints are simultaneously enforced on the data product. These requirements arise in real world applications of private data curation, including the  public release of the 2020 U.S. Decennial Census. They pose a great challenge to the production of provably private data products with adequate statistical usability. We propose integer subspace differential privacy to rigorously articulate the privacy guarantee when data products maintain both the invariants and integer characteristics, and demonstrate the composition and post-processing properties of our proposal. To address the challenge of sampling from a potentially highly restricted discrete space, we devise a pair of unbiased additive mechanisms, the generalized Laplace and the generalized Gaussian mechanisms, by solving the Diophantine equations as defined by the constraints. The proposed mechanisms have good accuracy, with errors exhibiting sub-exponential and sub-Gaussian tail probabilities respectively. To implement our proposal, we design an MCMC algorithm and supply empirical convergence assessment using estimated upper bounds on the total variation distance via L-lag coupling. We demonstrate the efficacy of our proposal with applications to a synthetic problem with intersecting invariants, a sensitive contingency table with known margins, and the 2010 Census county-level demonstration data with mandated fixed state population totals.

        ----

        ## [826] Black-Box Adversarial Attack on Time Series Classification

        **Authors**: *Daizong Ding, Mi Zhang, Fuli Feng, Yuanmin Huang, Erling Jiang, Min Yang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25896](https://doi.org/10.1609/aaai.v37i6.25896)

        **Abstract**:

        With the increasing use of deep neural network (DNN) in time series classification (TSC), recent work reveals the threat of adversarial attack, where the adversary can construct adversarial examples to cause model mistakes. However, existing researches on the adversarial attack of TSC typically adopt an unrealistic white-box setting with model details transparent to the adversary. In this work, we study a more rigorous black-box setting with attack detection applied, which restricts gradient access and requires the adversarial example to be also stealthy. Theoretical analyses reveal that the key lies in: estimating black-box gradient with diversity and non-convexity of TSC models resolved, and restricting the l0 norm of the perturbation to construct adversarial samples. Towards this end, we propose a new framework named BlackTreeS, which solves the hard optimization issue for adversarial example construction with two simple yet effective modules. In particular, we propose a tree search strategy to find influential positions in a sequence, and independently estimate the black-box gradients for these positions. Extensive experiments on three real-world TSC datasets and five DNN based models validate the effectiveness of BlackTreeS, e.g., it improves the attack success rate from 19.3% to 27.3%, and decreases the detection success rate from 90.9% to 6.8% for LSTM on the UWave dataset.

        ----

        ## [827] C-NTPP: Learning Cluster-Aware Neural Temporal Point Process

        **Authors**: *Fangyu Ding, Junchi Yan, Haiyang Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25897](https://doi.org/10.1609/aaai.v37i6.25897)

        **Abstract**:

        Event sequences in continuous time space are ubiquitous across applications and have been intensively studied with both classic temporal point process (TPP) and its recent deep network variants. This work is motivated by an observation that many of event data exhibit inherent clustering patterns in terms of the sparse correlation among events, while such characteristics are seldom explicitly considered in existing neural TPP models whereby the history encoders are often embodied by RNNs or Transformers. In this work, we propose a c-NTPP (Cluster-Aware Neural Temporal Point Process) model, which leverages a sequential variational autoencoder framework to infer the latent cluster each event belongs to in the sequence. Specially, a novel event-clustered attention mechanism is devised to learn each cluster and then aggregate them together to obtain the final representation for each event. Extensive experiments show that c-NTPP achieves superior performance on both real-world and synthetic datasets, and it can also uncover the underlying clustering correlations.

        ----

        ## [828] Eliciting Structural and Semantic Global Knowledge in Unsupervised Graph Contrastive Learning

        **Authors**: *Kaize Ding, Yancheng Wang, Yingzhen Yang, Huan Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25898](https://doi.org/10.1609/aaai.v37i6.25898)

        **Abstract**:

        Graph Contrastive Learning (GCL) has recently drawn much research interest for learning generalizable node representations in a self-supervised manner. In general, the contrastive learning process in GCL is performed on top of the representations learned by a graph neural network (GNN) backbone, which transforms and propagates the node contextual information based on its local neighborhoods. However, nodes sharing similar characteristics may not always be geographically close, which poses a great challenge for unsupervised GCL efforts due to their inherent limitations in capturing such global graph knowledge. In this work, we address their inherent limitations by proposing a simple yet effective framework -- Simple Neural Networks with Structural and Semantic Contrastive Learning} (S^3-CL). Notably, by virtue of the proposed structural and semantic contrastive learning algorithms, even a simple neural network can learn expressive node representations that preserve valuable global structural and semantic patterns. Our experiments demonstrate that the node representations learned by S^3-CL) achieve superior performance on different downstream tasks compared with the state-of-the-art unsupervised GCL methods. Implementation and more experimental details are publicly available at https://github.com/kaize0409/S-3-CL.

        ----

        ## [829] Incremental Reinforcement Learning with Dual-Adaptive ε-Greedy Exploration

        **Authors**: *Wei Ding, Siyang Jiang, Hsi-Wen Chen, Ming-Syan Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25899](https://doi.org/10.1609/aaai.v37i6.25899)

        **Abstract**:

        Reinforcement learning (RL) has achieved impressive performance in various domains. However, most RL frameworks oversimplify the problem by assuming a fixed-yet-known environment and often have difficulty being generalized to real-world scenarios. In this paper, we address a new challenge with a more realistic setting, Incremental Reinforcement Learning, where the search space of the Markov Decision Process continually expands. While previous methods usually suffer from the lack of efficiency in exploring the unseen transitions, especially with increasing search space, we present a new exploration framework named Dual-Adaptive ϵ-greedy Exploration (DAE) to address the challenge of Incremental RL. Specifically, DAE employs a Meta Policy and an Explorer to avoid redundant computation on those sufficiently
learned samples. Furthermore, we release a testbed based on a synthetic environment and the Atari benchmark to validate the effectiveness of any exploration algorithms under Incremental RL. Experimental results demonstrate that the proposed framework can efficiently learn the unseen transitions in new environments, leading to notable performance improvement, i.e., an average of more than 80%, over eight baselines examined.

        ----

        ## [830] Provably Efficient Primal-Dual Reinforcement Learning for CMDPs with Non-stationary Objectives and Constraints

        **Authors**: *Yuhao Ding, Javad Lavaei*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25900](https://doi.org/10.1609/aaai.v37i6.25900)

        **Abstract**:

        We consider primal-dual-based reinforcement learning (RL) in episodic constrained Markov decision processes (CMDPs) with non-stationary objectives and constraints, which plays a central role in ensuring the safety of RL in time-varying environments. In this problem, the reward/utility functions and the state transition functions are both allowed to vary arbitrarily over time as long as their cumulative variations do not exceed certain known variation budgets. Designing safe RL algorithms in time-varying environments is particularly challenging because of the need to integrate the constraint violation reduction, safe exploration, and adaptation to the non-stationarity. To this end, we identify two alternative conditions on the time-varying constraints under which we can guarantee the safety in the long run. We also propose the Periodically Restarted Optimistic Primal-Dual Proximal Policy Optimization (PROPD-PPO) algorithm that can coordinate with both two conditions. Furthermore, a dynamic regret bound and a constraint violation bound are established for the proposed algorithm in both the linear kernel CMDP function approximation setting and the tabular CMDP setting under two alternative conditions. This paper provides the first provably efficient algorithm for non-stationary CMDPs with safe exploration.

        ----

        ## [831] Non-stationary Risk-Sensitive Reinforcement Learning: Near-Optimal Dynamic Regret, Adaptive Detection, and Separation Design

        **Authors**: *Yuhao Ding, Ming Jin, Javad Lavaei*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25901](https://doi.org/10.1609/aaai.v37i6.25901)

        **Abstract**:

        We study risk-sensitive reinforcement learning (RL) based on an entropic risk measure in episodic non-stationary Markov decision processes (MDPs). Both the reward functions and the state transition kernels are unknown and allowed to vary arbitrarily over time with a budget on their cumulative variations. When this variation budget is known a prior, we propose two restart-based algorithms, namely Restart-RSMB and Restart-RSQ, and establish their dynamic regrets. Based on these results, we further present a meta-algorithm that does not require any prior knowledge of the variation budget and can adaptively detect the non-stationarity on the exponential value functions. A dynamic regret lower bound is then established for non-stationary risk-sensitive RL to certify the near-optimality of the proposed algorithms. Our results also show that the risk control and the handling of the non-stationarity can be separately designed in the algorithm if the variation budget is known a prior, while the non-stationary detection mechanism in the adaptive algorithm depends on the risk parameter. This work offers the first non-asymptotic theoretical analyses for the non-stationary risk-sensitive RL in the literature.

        ----

        ## [832] SKDBERT: Compressing BERT via Stochastic Knowledge Distillation

        **Authors**: *Zixiang Ding, Guoqing Jiang, Shuai Zhang, Lin Guo, Wei Lin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25902](https://doi.org/10.1609/aaai.v37i6.25902)

        **Abstract**:

        In this paper, we propose Stochastic Knowledge Distillation (SKD) to obtain compact BERT-style language model dubbed SKDBERT. In each distillation iteration, SKD samples a teacher model from a pre-defined teacher team, which consists of multiple teacher models with multi-level capacities, to transfer knowledge into student model in an one-to-one manner. Sampling distribution plays an important role in SKD. We heuristically present three types of sampling distributions to assign appropriate probabilities for multi-level teacher models. SKD has two advantages: 1) it can preserve the diversities of multi-level teacher models via stochastically sampling single teacher model in each distillation iteration, and 2) it can also improve the efficacy of knowledge distillation via multi-level teacher models when large capacity gap exists between the teacher model and the student model. Experimental results on GLUE benchmark show that SKDBERT reduces the size of a BERT model by 40% while retaining 99.5% performances of language understanding and being 100% faster.

        ----

        ## [833] Model-Based Offline Reinforcement Learning with Local Misspecification

        **Authors**: *Kefan Dong, Yannis Flet-Berliac, Allen Nie, Emma Brunskill*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25903](https://doi.org/10.1609/aaai.v37i6.25903)

        **Abstract**:

        We present a model-based offline reinforcement learning policy performance lower bound that explicitly captures dynamics model misspecification and distribution mismatch and we propose an empirical algorithm for optimal offline policy selection. Theoretically, we prove a novel safe policy improvement theorem by establishing pessimism approximations to the value function. Our key insight is to jointly consider selecting over dynamics models and policies: as long as a dynamics model can accurately represent the dynamics of the state-action pairs visited by a given policy, it is possible to approximate the value of that particular policy. We analyze our lower bound in the LQR setting and also show competitive performance to previous lower bounds on policy selection across a set of D4RL tasks.

        ----

        ## [834] Can Label-Specific Features Help Partial-Label Learning?

        **Authors**: *Ruo-Jing Dong, Jun-Yi Hang, Tong Wei, Min-Ling Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25904](https://doi.org/10.1609/aaai.v37i6.25904)

        **Abstract**:

        Partial label learning (PLL) aims to learn from inexact data annotations where each training example is associated with a coarse candidate label set. Due to its practicability, many PLL algorithms have been proposed in recent literature. Most prior PLL works attempt to identify the ground-truth labels from candidate sets and the classifier is trained afterward by fitting the features of examples and their exact ground-truth labels. From a different perspective, we propose to enrich the feature space and raise the question ``Can label-specific features help PLL?'' rather than learning from examples with identical features for all classes. Despite its benefits, previous label-specific feature approaches rely on ground-truth labels to split positive and negative examples of each class and then conduct clustering analysis, which is not directly applicable in PLL. To remedy this problem, we propose an uncertainty-aware confidence region to accommodate false positive labels. We first employ graph-based label enhancement to yield smooth pseudo-labels and facilitate the confidence region split. After acquiring label-specific features, a family of binary classifiers is induced. Extensive experiments on both synthesized and real-world datasets are conducted and the results show that our method consistently outperforms eight baselines. Our code is released at
https://github.com/meteoseeker/UCL

        ----

        ## [835] Interpreting Unfairness in Graph Neural Networks via Training Node Attribution

        **Authors**: *Yushun Dong, Song Wang, Jing Ma, Ninghao Liu, Jundong Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25905](https://doi.org/10.1609/aaai.v37i6.25905)

        **Abstract**:

        Graph Neural Networks (GNNs) have emerged as the leading paradigm for solving graph analytical problems in various real-world applications. Nevertheless, GNNs could potentially render biased predictions towards certain demographic subgroups.  Understanding how the bias in predictions arises is critical, as it guides the design of GNN debiasing mechanisms. However, most existing works overwhelmingly focus on GNN debiasing, but fall short on explaining how such bias is induced. In this paper, we study a novel problem of interpreting GNN unfairness through attributing it to the influence of training nodes. Specifically, we propose a novel strategy named Probabilistic Distribution Disparity (PDD) to measure the bias exhibited in GNNs, and develop an algorithm to efficiently estimate the influence of each training node on such bias. We verify the validity of PDD and the effectiveness of influence estimation through experiments on real-world datasets. Finally, we also demonstrate how the proposed framework could be used for debiasing GNNs. Open-source code can be found at https://github.com/yushundong/BIND.

        ----

        ## [836] Robust and Fast Measure of Information via Low-Rank Representation

        **Authors**: *Yuxin Dong, Tieliang Gong, Shujian Yu, Hong Chen, Chen Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25906](https://doi.org/10.1609/aaai.v37i6.25906)

        **Abstract**:

        The matrix-based Rényi's entropy allows us to directly quantify information measures from given data, without explicit estimation of the underlying probability distribution. This intriguing property makes it widely applied in statistical inference and machine learning tasks. However, this information theoretical quantity is not robust against noise in the data, and is computationally prohibitive in large-scale applications. To address these issues, we propose a novel measure of information, termed low-rank matrix-based Rényi's entropy, based on low-rank representations of infinitely divisible kernel matrices. The proposed entropy functional inherits the specialty of of the original definition to directly quantify information from data, but enjoys additional advantages including robustness and effective calculation. Specifically, our low-rank variant is more sensitive to informative perturbations induced by changes in underlying distributions, while being insensitive to uninformative ones caused by noises. Moreover, low-rank Rényi's entropy can be efficiently approximated by random projection and Lanczos iteration techniques, reducing the overall complexity from O(n³) to O(n²s) or even O(ns²), where n is the number of data samples and s ≪ n. We conduct large-scale experiments to evaluate the effectiveness of this new information measure, demonstrating superior results compared to matrix-based Rényi's entropy in terms of both performance and computational efficiency.

        ----

        ## [837] Graph Anomaly Detection via Multi-Scale Contrastive Learning Networks with Augmented View

        **Authors**: *Jingcan Duan, Siwei Wang, Pei Zhang, En Zhu, Jingtao Hu, Hu Jin, Yue Liu, Zhibin Dong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25907](https://doi.org/10.1609/aaai.v37i6.25907)

        **Abstract**:

        Graph anomaly detection (GAD) is a vital task in graph-based machine learning and has been widely applied in many real-world applications. The primary goal of GAD is to capture anomalous nodes from graph datasets, which evidently deviate from the majority of nodes. Recent methods have paid attention to various scales of contrastive strategies for GAD, i.e., node-subgraph and node-node contrasts. However, they neglect the subgraph-subgraph comparison information which the normal and abnormal subgraph pairs behave differently in terms of embeddings and structures in GAD, resulting in sub-optimal task performance. In this paper, we fulfill the above idea in the proposed multi-view multi-scale contrastive learning framework with subgraph-subgraph contrast for the first practice. To be specific, we regard the original input graph as the first view and generate the second view by graph augmentation with edge modifications. With the guidance of maximizing the similarity of the subgraph pairs, the proposed subgraph-subgraph contrast contributes to more robust subgraph embeddings despite of the structure variation. Moreover, the introduced subgraph-subgraph contrast cooperates well with the widely-adopted node-subgraph and node-node contrastive counterparts for mutual GAD performance promotions. Besides, we also conduct sufficient experiments to investigate the impact of different graph augmentation approaches on detection performance. The comprehensive experimental results well demonstrate the superiority of our method compared with the state-of-the-art approaches and the effectiveness of the multi-view subgraph pair contrastive strategy for the GAD task. The source code is released at https://github.com/FelixDJC/GRADATE.

        ----

        ## [838] Diffeomorphic Information Neural Estimation

        **Authors**: *Bao Duong, Thin Nguyen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25908](https://doi.org/10.1609/aaai.v37i6.25908)

        **Abstract**:

        Mutual Information (MI) and Conditional Mutual Information (CMI) are multi-purpose tools from information theory that are able to naturally measure the statistical dependencies between random variables, thus they are usually of central interest in several statistical and machine learning tasks, such as conditional independence testing and representation learning. However, estimating CMI, or even MI, is infamously challenging due the intractable formulation. In this study, we introduce DINE (Diffeomorphic Information Neural Estimator)–a novel approach for estimating CMI of continuous random variables, inspired by the invariance of CMI over diffeomorphic maps. We show that the variables of interest can be replaced with appropriate surrogates that follow simpler distributions, allowing the CMI to be efficiently evaluated via analytical solutions. Additionally, we demonstrate the quality of the proposed estimator in comparison with state-of-the-arts in three important tasks, including estimating MI, CMI, as well as its application in conditional independence testing. The empirical evaluations show that DINE consistently outperforms competitors in all tasks and is able to adapt very well to complex and high-dimensional relationships.

        ----

        ## [839] Combining Slow and Fast: Complementary Filtering for Dynamics Learning

        **Authors**: *Katharina Ensinger, Sebastian Ziesche, Barbara Rakitsch, Michael Tiemann, Sebastian Trimpe*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25909](https://doi.org/10.1609/aaai.v37i6.25909)

        **Abstract**:

        Modeling an unknown dynamical system is crucial in order to predict the future behavior of the system. A standard approach is training recurrent models on measurement data. While these models typically provide exact short-term predictions, accumulating errors yield deteriorated long-term behavior. In contrast, models with reliable long-term predictions can often be obtained, either by training a robust but less detailed model, or by leveraging physics-based simulations. In both cases, inaccuracies in the models yield a lack of short-time details. Thus, different models with contrastive properties on different time horizons are available. This observation immediately raises the question: Can we obtain predictions that combine the best of both worlds? Inspired by sensor fusion tasks, we interpret the problem in the frequency domain and leverage classical methods from signal processing, in particular complementary filters. This filtering technique combines two signals by applying a high-pass filter to one signal, and low-pass filtering the other. Essentially, the high-pass filter extracts high-frequencies, whereas the low-pass filter extracts low frequencies. Applying this concept to dynamics model learning enables the construction of models that yield accurate long- and short-term predictions. Here, we propose two methods, one being purely learning-based and the other one being a hybrid model that requires an additional physics-based simulator.

        ----

        ## [840] Popularizing Fairness: Group Fairness and Individual Welfare

        **Authors**: *Andrew Estornell, Sanmay Das, Brendan Juba, Yevgeniy Vorobeychik*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25910](https://doi.org/10.1609/aaai.v37i6.25910)

        **Abstract**:

        Group-fair learning methods typically seek to ensure that some measure of prediction efficacy for (often historically) disadvantaged minority groups is comparable to that for the majority of the population. When a principal seeks to adopt a group-fair approach to replace another, the principal may face opposition from those who feel they may be harmed by the switch, and this, in turn, may deter adoption. We propose that a potential mitigation to this concern is to ensure that a group-fair model is also popular, in the sense that,  for a majority of the target population, it yields a preferred distribution over outcomes compared with the conventional model. In this paper, we show that state of the art fair learning approaches are often  unpopular in this sense. We propose several efficient algorithms for postprocessing an existing group-fair learning scheme to improve its popularity while retaining fairness. Through extensive experiments, we demonstrate that the proposed postprocessing approaches are highly effective in practice.

        ----

        ## [841] FairFed: Enabling Group Fairness in Federated Learning

        **Authors**: *Yahya H. Ezzeldin, Shen Yan, Chaoyang He, Emilio Ferrara, Amir Salman Avestimehr*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25911](https://doi.org/10.1609/aaai.v37i6.25911)

        **Abstract**:

        Training ML models which are fair across different demographic groups is of critical importance due to the increased integration of ML in crucial decision-making scenarios such as healthcare and recruitment. Federated learning has been viewed as a promising solution for collaboratively training machine learning models among multiple parties while maintaining their local data privacy. However, federated learning also poses new challenges in mitigating the potential bias against certain populations (e.g., demographic groups), as this typically requires centralized access to the sensitive information (e.g., race, gender) of each datapoint. Motivated by the importance and challenges of group fairness in federated learning, in this work, we propose FairFed, a novel algorithm for fairness-aware aggregation to enhance group fairness in federated learning. Our proposed approach is server-side and agnostic to the applied local debiasing thus allowing for flexible use of different local debiasing methods across clients. We evaluate FairFed empirically versus common baselines for fair ML and federated learning and demonstrate that it provides fairer models, particularly under highly heterogeneous data distributions across clients. We also demonstrate the benefits of FairFed in scenarios involving naturally distributed real-life data collected from different geographical locations or departments within an organization.

        ----

        ## [842] Goal-Conditioned Generators of Deep Policies

        **Authors**: *Francesco Faccio, Vincent Herrmann, Aditya Ramesh, Louis Kirsch, Jürgen Schmidhuber*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25912](https://doi.org/10.1609/aaai.v37i6.25912)

        **Abstract**:

        Goal-conditioned Reinforcement Learning (RL) aims at learning optimal policies, given goals encoded in special command inputs. Here we study goal-conditioned neural nets (NNs) that learn to generate deep NN policies in form of context-specific weight matrices, similar to Fast Weight Programmers and other methods from the 1990s. Using context commands of the form ``generate a policy that achieves a desired expected return,'' our NN generators combine powerful exploration of parameter space with generalization across commands to iteratively find better and better policies. A form of weight-sharing HyperNetworks and policy embeddings scales our method to generate deep NNs. Experiments show how a single learned policy generator can produce policies that achieve any return seen during training. Finally, we evaluate our algorithm on a set of continuous control tasks where it exhibits competitive performance.
Our code is public.

        ----

        ## [843] Directed Acyclic Graph Structure Learning from Dynamic Graphs

        **Authors**: *Shaohua Fan, Shuyang Zhang, Xiao Wang, Chuan Shi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25913](https://doi.org/10.1609/aaai.v37i6.25913)

        **Abstract**:

        Estimating the structure of directed acyclic graphs (DAGs) of features (variables) plays a vital role in revealing the latent data generation process and providing causal insights in various applications. Although there have been many studies on structure learning with various types of data, the structure learning on the dynamic graph has not been explored yet, and thus we study the learning problem of node feature generation mechanism on such ubiquitous dynamic graph data. In a dynamic graph, we propose to simultaneously estimate contemporaneous relationships and time-lagged interaction relationships between the node features. These two kinds of relationships form a DAG, which could effectively characterize the feature generation process in a concise way. To learn such a DAG, we cast the learning problem as a continuous score-based optimization problem, which consists of a differentiable score function to measure the validity of the learned DAGs and a smooth acyclicity constraint to ensure the acyclicity of the learned DAGs. These two components are translated into an unconstraint augmented Lagrangian objective which could be minimized by mature continuous optimization techniques. The resulting algorithm, named GraphNOTEARS, outperforms baselines on simulated data across a wide range of settings that may encounter in real-world applications. We also apply the proposed approach on two dynamic graphs constructed from the real-world Yelp dataset, demonstrating our method could learn the connections between node features, which conforms with the domain knowledge.

        ----

        ## [844] Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting

        **Authors**: *Wei Fan, Pengyang Wang, Dongkun Wang, Dongjie Wang, Yuanchun Zhou, Yanjie Fu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25914](https://doi.org/10.1609/aaai.v37i6.25914)

        **Abstract**:

        The distribution shift in Time Series Forecasting (TSF), indicating series distribution changes over time, largely hinders the performance of TSF models. Existing works towards distribution shift in time series are mostly limited in the quantification of distribution and, more importantly, overlook the potential shift between lookback and horizon windows. To address above challenges, we systematically summarize the distribution shift in TSF into two categories. Regarding lookback windows as input-space and horizon windows as output-space, there exist (i) intra-space shift, that the distribution within the input-space keeps shifted over time, and (ii) inter-space shift, that the distribution is shifted between input-space and output-space. Then we introduce, Dish-TS, a general neural paradigm for alleviating distribution shift in TSF. Specifically, for better distribution estimation, we propose the coefficient net (Conet), which can be any neural architectures, to map input sequences into learnable distribution coefficients. To relieve intra-space and inter-space shift, we organize Dish-TS as a Dual-Conet framework to separately learn the distribution of input- and output-space, which naturally captures the distribution difference of two spaces. In addition, we introduce a more effective training strategy for intractable Conet learning. Finally, we conduct extensive experiments on several datasets coupled with different state-of-the-art forecasting models. Experimental results show Dish-TS consistently boosts them with a more than 20% average improvement. Code is available at https://github.com/weifantt/Dish-TS.

        ----

        ## [845] Learning Decomposed Spatial Relations for Multi-Variate Time-Series Modeling

        **Authors**: *Yuchen Fang, Kan Ren, Caihua Shan, Yifei Shen, You Li, Weinan Zhang, Yong Yu, Dongsheng Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25915](https://doi.org/10.1609/aaai.v37i6.25915)

        **Abstract**:

        Modeling multi-variate time-series (MVTS) data is a long-standing research subject and has found wide applications. Recently, there is a surge of interest in modeling spatial relations between variables as graphs, i.e., first learning one static graph for each dataset and then exploiting the graph structure via graph neural networks. However, as spatial relations may differ substantially across samples, building one static graph for all the samples inherently limits flexibility and severely degrades the performance in practice. To address this issue, we propose a framework for fine-grained modeling and utilization of spatial correlation between variables. By analyzing the statistical properties of real-world datasets, a universal decomposition of spatial correlation graphs is first identified. Specifically, the hidden spatial relations can be decomposed into a prior part, which applies across all the samples, and a dynamic part, which varies between samples, and building different graphs is necessary to model these relations. To better coordinate the learning of the two relational graphs, we propose a min-max learning paradigm that not only regulates the common part of different dynamic graphs but also guarantees spatial distinguishability among samples. The experimental results show that our proposed model outperforms the state-of-the-art baseline methods on both time-series forecasting and time-series point prediction tasks.

        ----

        ## [846] Wasserstein Graph Distance Based on L1-Approximated Tree Edit Distance between Weisfeiler-Lehman Subtrees

        **Authors**: *Zhongxi Fang, Jianming Huang, Xun Su, Hiroyuki Kasai*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25916](https://doi.org/10.1609/aaai.v37i6.25916)

        **Abstract**:

        The Weisfeiler-Lehman (WL) test is a widely used algorithm in graph machine learning, including graph kernels, graph metrics, and graph neural networks. However, it focuses only on the consistency of the graph, which means that it is unable to detect slight structural differences. Consequently, this limits its ability to capture structural information, which also limits the performance of existing models that rely on the WL test. This limitation is particularly severe for traditional metrics defined by the WL test, which cannot precisely capture slight structural differences. In this paper, we propose a novel graph metric called the Wasserstein WL Subtree (WWLS) distance to address this problem. Our approach leverages the WL subtree as structural information for node neighborhoods and defines node metrics using the L1-approximated tree edit distance (L1-TED) between WL subtrees of nodes. Subsequently, we combine the Wasserstein distance and the L1-TED to define the WWLS distance, which can capture slight structural differences that may be difficult to detect using conventional metrics. We demonstrate that the proposed WWLS distance outperforms baselines in both metric validation and graph classification experiments.

        ----

        ## [847] Combinatorial Causal Bandits

        **Authors**: *Shi Feng, Wei Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25917](https://doi.org/10.1609/aaai.v37i6.25917)

        **Abstract**:

        In combinatorial causal bandits (CCB), the learning agent chooses at most K variables in each round to intervene, collects feedback from the observed variables, with the goal of minimizing expected regret on the target variable Y. We study under the context of binary generalized linear models (BGLMs) with a succinct parametric representation of the causal models. We present the algorithm BGLM-OFU for Markovian BGLMs (i.e., no hidden variables) based on the maximum likelihood estimation method and give regret analysis for it. For the special case of linear models with hidden variables, we apply causal inference techniques such as the do calculus to convert the original model into a Markovian model, and then show that our BGLM-OFU algorithm and another algorithm based on the linear regression both solve such linear models with hidden variables. Our novelty includes (a) considering the combinatorial intervention action space and the general causal graph structures including ones with hidden variables, (b) integrating and adapting techniques from diverse studies such as generalized linear bandits and online influence maximization, and (c) avoiding unrealistic assumptions (such as knowing the joint distribution of the parents of Y under all interventions) and regret factors exponential to causal graph size in prior studies.

        ----

        ## [848] Scalable Attributed-Graph Subspace Clustering

        **Authors**: *Chakib Fettal, Lazhar Labiod, Mohamed Nadif*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25918](https://doi.org/10.1609/aaai.v37i6.25918)

        **Abstract**:

        Over recent years, graph convolutional networks emerged as powerful node clustering methods and have set state of the art results for this task. In this paper, we argue that some of these methods are unnecessarily complex and propose a node clustering model that is more scalable while being more effective. The proposed model uses Laplacian smoothing to learn an initial representation of the graph before applying an efficient self-expressive subspace clustering procedure.
This is performed via learning a factored coefficient matrix. These factors are then embedded into a new feature space in such a way as to generate a valid affinity matrix (symmetric and non-negative) on which an implicit spectral clustering algorithm is performed. 
Experiments on several real-world attributed datasets demonstrate the cost-effective nature of our method with respect to the state of the art.

        ----

        ## [849] SigMaNet: One Laplacian to Rule Them All

        **Authors**: *Stefano Fiorini, Stefano Coniglio, Michele Ciavotta, Enza Messina*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25919](https://doi.org/10.1609/aaai.v37i6.25919)

        **Abstract**:

        This paper introduces SigMaNet, a generalized Graph Convolutional Network (GCN) capable of handling both undirected and directed graphs with weights not restricted in sign nor magnitude. The cornerstone of SigMaNet is the Sign-Magnetic Laplacian (LSM), a new Laplacian matrix that we introduce ex novo in this work. LSM allows us to bridge a gap in the current literature by extending the theory of spectral GCNs to (directed) graphs with both positive and negative weights. LSM exhibits several desirable properties not enjoyed by other Laplacian matrices on which several state-of-the-art architectures are based, among which encoding the edge direction and weight in a clear and natural way that is not negatively affected by the weight magnitude. LSM is also completely parameter-free, which is not the case of other Laplacian operators such as, e.g., the Magnetic Laplacian. The versatility and the performance of our proposed approach is amply demonstrated via computational experiments. Indeed, our results show that, for at least a metric, SigMaNet achieves the best performance in 15 out of 21 cases and either the first- or second-best performance in 21 cases out of 21, even when compared to architectures that are either more complex or that, due to being designed for a narrower class of graphs, should---but do not---achieve a better performance.

        ----

        ## [850] Optimal Decision Diagrams for Classification

        **Authors**: *Alexandre M. Florio, Pedro Martins, Maximilian Schiffer, Thiago Serra, Thibaut Vidal*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25920](https://doi.org/10.1609/aaai.v37i6.25920)

        **Abstract**:

        Decision diagrams for classification have some notable advantages over decision trees, as their internal connections can be determined at training time and their width is not bound to grow exponentially with their depth. Accordingly, decision diagrams are usually less prone to data fragmentation in internal nodes. However, the inherent complexity of training these classifiers acted as a long-standing barrier to their widespread adoption. In this context, we study the training of optimal decision diagrams (ODDs) from a mathematical programming perspective. We introduce a novel mixed-integer linear programming model for training and demonstrate its applicability for many datasets of practical importance. Further, we show how this model can be easily extended for fairness, parsimony, and stability notions. We present numerical analyses showing that our model allows training ODDs in short computational times, and that ODDs achieve better accuracy than optimal decision trees, while allowing for improved stability without significant accuracy losses.

        ----

        ## [851] Estimating Average Causal Effects from Patient Trajectories

        **Authors**: *Dennis Frauen, Tobias Hatt, Valentyn Melnychuk, Stefan Feuerriegel*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25921](https://doi.org/10.1609/aaai.v37i6.25921)

        **Abstract**:

        In medical practice, treatments are selected based on the expected causal effects on patient outcomes. Here, the gold standard for estimating causal effects are randomized controlled trials; however, such trials are costly and sometimes even unethical. Instead, medical practice is increasingly interested in estimating causal effects among patient (sub)groups from electronic health records, that is, observational data. In this paper, we aim at estimating the average causal effect (ACE) from observational data (patient trajectories) that are collected over time. For this, we propose DeepACE: an end-to-end deep learning model. DeepACE leverages the iterative G-computation formula to adjust for the bias induced by time-varying confounders. Moreover, we develop a novel sequential targeting procedure which ensures that DeepACE has favorable theoretical properties, i.e., is doubly robust and asymptotically efficient. To the best of our knowledge, this is the first work that proposes an end-to-end deep learning model tailored for estimating time-varying ACEs. We compare DeepACE in an extensive number of experiments, confirming that it achieves state-of-the-art performance. We further provide a case study for patients suffering from low back pain to demonstrate that DeepACE generates important and meaningful findings for clinical practice. Our work enables practitioners to develop effective treatment recommendations based on population effects.

        ----

        ## [852] Decorate the Newcomers: Visual Domain Prompt for Continual Test Time Adaptation

        **Authors**: *Yulu Gan, Yan Bai, Yihang Lou, Xianzheng Ma, Renrui Zhang, Nian Shi, Lin Luo*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25922](https://doi.org/10.1609/aaai.v37i6.25922)

        **Abstract**:

        Continual Test-Time Adaptation (CTTA) aims to adapt the source model to continually changing unlabeled target domains without access to the source data. Existing methods mainly focus on model-based adaptation in a self-training manner, such as predicting pseudo labels for new domain datasets. Since pseudo labels are noisy and unreliable, these methods suffer from catastrophic forgetting and error accumulation when dealing with dynamic data distributions. Motivated by the prompt learning in NLP, in this paper, we propose to learn an image-layer visual domain prompt for target domains while having the source model parameters frozen. During testing, the changing target datasets can be adapted to the source model by reformulating the input data with the learned visual prompts. Specifically, we devise two types of prompts, i.e., domains-specific prompts and domains-agnostic prompts, to extract current domain knowledge and maintain the domain-shared knowledge in the continual adaptation. Furthermore, we design a homeostasis-based adaptation strategy to suppress domain-sensitive parameters in domain-invariant prompts to learn domain-shared knowledge more effectively. This transition from the model-dependent paradigm to the model-free one enables us to bypass the catastrophic forgetting and error accumulation problems. Experiments show that our proposed method achieves significant performance gains over state-of-the-art methods on four widely-used benchmarks, including CIFAR-10C, CIFAR-100C, ImageNet-C, and VLCS datasets.

        ----

        ## [853] EffConv: Efficient Learning of Kernel Sizes for Convolution Layers of CNNs

        **Authors**: *Alireza Ganjdanesh, Shangqian Gao, Heng Huang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25923](https://doi.org/10.1609/aaai.v37i6.25923)

        **Abstract**:

        Determining kernel sizes of a CNN model is a crucial and non-trivial design choice and significantly impacts its performance. The majority of kernel size design methods rely on complex heuristic tricks or leverage neural architecture search that requires extreme computational resources. Thus, learning kernel sizes, using methods such as modeling kernels as a combination of basis functions, jointly with the model weights has been proposed as a workaround. However, previous methods cannot achieve satisfactory results or are inefficient for large-scale datasets. To fill this gap, we design a novel efficient kernel size learning method in which a size predictor model learns to predict optimal kernel sizes for a classifier given a desired number of parameters. It does so in collaboration with a kernel predictor model that predicts the weights of the kernels - given kernel sizes predicted by the size predictor - to minimize the training objective, and both models are trained end-to-end. Our method only needs a small fraction of the training epochs of the original CNN to train these two models and find proper kernel sizes for it. Thus, it offers an efficient and effective solution for the kernel size learning problem. Our extensive experiments on MNIST, CIFAR-10, STL-10, and ImageNet-32 demonstrate that our method can achieve the best training time vs. accuracy trade-off compared to previous kernel size learning methods and significantly outperform them on challenging datasets such as STL-10 and ImageNet-32. Our implementations are available at https://github.com/Alii-Ganjj/EffConv.

        ----

        ## [854] Fast Counterfactual Inference for History-Based Reinforcement Learning

        **Authors**: *Haichuan Gao, Tianren Zhang, Zhile Yang, Yuqing Guo, Jinsheng Ren, Shangqi Guo, Feng Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25924](https://doi.org/10.1609/aaai.v37i6.25924)

        **Abstract**:

        Incorporating sequence-to-sequence models into history-based Reinforcement Learning (RL) provides a general way to extend RL to partially-observable tasks. This method compresses history spaces according to the correlations between historical observations and the rewards. However, they do not adjust for the confounding correlations caused by data sampling and assign high beliefs to uninformative historical observations, leading to limited compression of history spaces. Counterfactual Inference (CI), which estimates causal effects by single-variable intervention, is a promising way to adjust for confounding. However, it is computationally infeasible to directly apply the single-variable intervention to a huge number of historical observations. This paper proposes to perform CI on observation sub-spaces instead of single observations and develop a coarse-to-fine CI algorithm, called Tree-based History Counterfactual Inference (T-HCI), to reduce the number of interventions exponentially. We show that T-HCI is computationally feasible in practice and brings significant sample efficiency gains in various challenging partially-observable tasks, including Maze, BabyAI, and robot manipulation tasks.

        ----

        ## [855] Robust Causal Graph Representation Learning against Confounding Effects

        **Authors**: *Hang Gao, Jiangmeng Li, Wenwen Qiang, Lingyu Si, Bing Xu, Changwen Zheng, Fuchun Sun*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25925](https://doi.org/10.1609/aaai.v37i6.25925)

        **Abstract**:

        The prevailing graph neural network models have achieved significant progress in graph representation learning. However, in this paper, we uncover an ever-overlooked phenomenon: the pre-trained graph representation learning model tested with full graphs underperforms the model tested with well-pruned graphs. This observation reveals that there exist confounders in graphs, which may interfere with the model learning semantic information, and current graph representation learning methods have not eliminated their influence. To tackle this issue, we propose Robust Causal Graph Representation Learning (RCGRL) to learn robust graph representations against confounding effects. RCGRL introduces an active approach to generate instrumental variables under unconditional moment restrictions, which empowers the graph representation learning model to eliminate confounders, thereby capturing discriminative information that is causally related to downstream predictions. We offer theorems and proofs to guarantee the theoretical effectiveness of the proposed approach. Empirically, we conduct extensive experiments on a synthetic dataset and multiple benchmark datasets. Experimental results demonstrate the effectiveness and generalization ability of RCGRL. Our codes are available at https://github.com/hang53/RCGRL.

        ----

        ## [856] Towards Decision-Friendly AUC: Learning Multi-Classifier with AUCµ

        **Authors**: *Peifeng Gao, Qianqian Xu, Peisong Wen, Huiyang Shao, Yuan He, Qingming Huang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25926](https://doi.org/10.1609/aaai.v37i6.25926)

        **Abstract**:

        Area Under the ROC Curve (AUC) is a widely used ranking metric in imbalanced learning due to its insensitivity to label distributions. As a well-known multiclass extension of AUC, Multiclass AUC (MAUC, a.k.a. M-metric) measures the average AUC of multiple binary classifiers. In this paper, we argue that simply optimizing MAUC is far from enough for imbalanced multi-classification. More precisely, MAUC only focuses on learning scoring functions via ranking optimization, while leaving the decision process unconsidered. Therefore, scoring functions being able to make good decisions might suffer from low performance in terms of MAUC. To overcome this issue, we turn to explore AUCµ, another multiclass variant of AUC, which further takes the decision process into consideration. Motivated by this fact, we propose a surrogate risk optimization framework to improve model performance from the perspective of AUCµ. Practically, we propose a two-stage training framework for multi-classification, where at the first stage a scoring function is learned maximizing AUCµ, and at the second stage we seek for a decision function to improve the F1-metric via our proposed soft F1. Theoretically, we first provide sufficient conditions that optimizing the surrogate losses could lead to the Bayes optimal scoring function. Afterward, we show that the proposed surrogate risk enjoys a generalization bound in order of O(1/√N). Experimental results on four benchmark datasets demonstrate the effectiveness of our proposed method in both AUCµ and F1-metric.

        ----

        ## [857] Long-Tail Cross Modal Hashing

        **Authors**: *Zijun Gao, Jun Wang, Guoxian Yu, Zhongmin Yan, Carlotta Domeniconi, Jinglin Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25927](https://doi.org/10.1609/aaai.v37i6.25927)

        **Abstract**:

        Existing Cross Modal Hashing (CMH) methods  are mainly designed for balanced data, while imbalanced data with long-tail distribution is more general in real-world. Several long-tail hashing methods have been proposed but they can not adapt for multi-modal data, due to the complex interplay between labels and individuality and commonality information of multi-modal data. Furthermore, CMH methods mostly mine the commonality of multi-modal data to learn hash codes, which may override tail labels encoded by the individuality of respective modalities. In this paper, we propose LtCMH (Long-tail CMH) to handle imbalanced multi-modal data. LtCMH firstly adopts auto-encoders to mine the individuality and commonality of different modalities by minimizing the dependency between the individuality of respective modalities and by enhancing the commonality of these modalities. Then it dynamically combines the individuality and commonality with direct features extracted from respective modalities to create meta features that enrich the representation of tail labels, and  binaries meta features to generate hash codes. LtCMH significantly outperforms state-of-the-art baselines on long-tail datasets and holds a better (or comparable) performance on datasets with balanced labels.

        ----

        ## [858] Handling Missing Data via Max-Entropy Regularized Graph Autoencoder

        **Authors**: *Ziqi Gao, Yifan Niu, Jiashun Cheng, Jianheng Tang, Lanqing Li, Tingyang Xu, Peilin Zhao, Fugee Tsung, Jia Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25928](https://doi.org/10.1609/aaai.v37i6.25928)

        **Abstract**:

        Graph neural networks (GNNs) are popular weapons for modeling relational data. Existing GNNs are not specified for attribute-incomplete graphs, making missing attribute imputation a burning issue. Until recently, many works notice that GNNs are coupled with spectral concentration, which means the spectrum obtained by GNNs concentrates on a local part in spectral domain, e.g., low-frequency due to oversmoothing issue. As a consequence, GNNs may be seriously flawed for reconstructing graph attributes as graph spectral concentration tends to cause a low imputation precision. In this work, we present a regularized graph autoencoder for graph attribute imputation, named MEGAE, which aims at mitigating spectral concentration problem by maximizing the graph spectral entropy. Notably, we first present the method for estimating graph spectral entropy without the eigen-decomposition of Laplacian matrix and provide the theoretical upper error bound. A maximum entropy regularization then acts in the latent space, which directly increases the graph spectral entropy. Extensive experiments show that MEGAE outperforms all the other state-of-the-art imputation methods on a variety of benchmark datasets.

        ----

        ## [859] Reinforced Approximate Exploratory Data Analysis

        **Authors**: *Shaddy Garg, Subrata Mitra, Tong Yu, Yash Gadhia, Arjun Kashettiwar*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25929](https://doi.org/10.1609/aaai.v37i6.25929)

        **Abstract**:

        Exploratory data analytics (EDA) is a sequential decision making process where analysts choose subsequent queries that might lead to some interesting insights based on the previous queries and corresponding results. Data processing systems often execute the queries on samples to produce results with low latency. Different downsampling strategy preserves different statistics of the data and have different magnitude of latency reductions. The optimum choice of sampling strategy often depends on the particular context of the analysis flow and the hidden intent of the analyst. In this paper, we are the first to consider the impact of sampling in interactive data exploration settings as they introduce approximation errors.
We propose a Deep Reinforcement Learning (DRL) based framework which can optimize the sample selection in order to keep the analysis and insight generation flow intact. Evaluations with real datasets show that our technique can preserve the original insight generation flow while improving the interaction latency, compared to baseline methods.

        ----

        ## [860] Learning Program Synthesis for Integer Sequences from Scratch

        **Authors**: *Thibault Gauthier, Josef Urban*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25930](https://doi.org/10.1609/aaai.v37i6.25930)

        **Abstract**:

        We present a self-learning approach for synthesizing programs from integer
sequences. Our method relies on a tree search guided by a learned policy. 
Our system is tested on the On-Line Encyclopedia of Integer Sequences.
There, it discovers, on its own, solutions for 27987 sequences starting from 
basic operators and without human-written training examples.

        ----

        ## [861] Semi-transductive Learning for Generalized Zero-Shot Sketch-Based Image Retrieval

        **Authors**: *Ce Ge, Jingyu Wang, Qi Qi, Haifeng Sun, Tong Xu, Jianxin Liao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25931](https://doi.org/10.1609/aaai.v37i6.25931)

        **Abstract**:

        Sketch-based image retrieval (SBIR) is an attractive research area where freehand sketches are used as queries to retrieve relevant images. Existing solutions have advanced the task to the challenging zero-shot setting (ZS-SBIR), where the trained models are tested on new classes without seen data. However, they are prone to overfitting under a realistic scenario when the test data includes both seen and unseen classes. In this paper, we study generalized ZS-SBIR (GZS-SBIR) and propose a novel semi-transductive learning paradigm. Transductive learning is performed on the image modality to explore the potential data distribution within unseen classes, and zero-shot learning is performed on the sketch modality sharing the learned knowledge through a semi-heterogeneous architecture. A hybrid metric learning strategy is proposed to establish semantics-aware ranking property and calibrate the joint embedding space. Extensive experiments are conducted on two large-scale benchmarks and four evaluation metrics. The results show that our method is superior over the state-of-the-art competitors in the challenging GZS-SBIR task.

        ----

        ## [862] Multi-Classifier Adversarial Optimization for Active Learning

        **Authors**: *Lin Geng, Ningzhong Liu, Jie Qin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25932](https://doi.org/10.1609/aaai.v37i6.25932)

        **Abstract**:

        Active learning (AL) aims to find a better trade-off between labeling costs and model performance by consciously selecting more informative samples to label. Recently, adversarial approaches have emerged as effective solutions. Most of them leverage generative adversarial networks to align feature distributions of labeled and unlabeled data, upon which discriminators are trained to better distinguish between them. However, these methods fail to consider the relationship between unlabeled samples and decision boundaries, and their training processes are often complex and unstable. To this end, this paper proposes a novel adversarial AL method, namely multi-classifier adversarial optimization for active learning (MAOAL). MAOAL employs task-specific decision boundaries for data alignment while selecting the most informative samples to label. To fulfill this, we introduce a novel classifier class confusion (C3) metric, which represents the classifier discrepancy as the inter-class correlation of classifier outputs. Without any additional hyper-parameters, the C3 metric further reduces the negative impacts of ambiguous samples in the process of distribution alignment and sample selection. More concretely, the network is trained adversarially by adding two auxiliary classifiers, reducing the distribution bias of labeled and unlabeled samples by minimizing the C3 loss between classifiers, while learning tighter decision boundaries and highlighting hard samples by maximizing the C3 loss. Finally, the unlabeled samples with the highest C3 loss are selected to label. Extensive experiments demonstrate the superiority of our approach over state-of-the-art AL methods in terms of image classification and object detection.

        ----

        ## [863] Differentially Private Heatmaps

        **Authors**: *Badih Ghazi, Junfeng He, Kai Kohlhoff, Ravi Kumar, Pasin Manurangsi, Vidhya Navalpakkam, Nachiappan Valliappan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25933](https://doi.org/10.1609/aaai.v37i6.25933)

        **Abstract**:

        We consider the task of producing heatmaps from users' aggregated data while protecting their privacy. We give a differentially private (DP) algorithm for this task and demonstrate its advantages over previous algorithms on real-world datasets.

Our core algorithmic primitive is a DP procedure that takes in a set of distributions and produces an output that is close in Earth Mover's Distance (EMD) to the average of the inputs. We prove theoretical bounds on the error of our algorithm under a certain sparsity assumption and that these are essentially optimal.

        ----

        ## [864] DiFA: Differentiable Feature Acquisition

        **Authors**: *Aritra Ghosh, Andrew S. Lan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25934](https://doi.org/10.1609/aaai.v37i6.25934)

        **Abstract**:

        Feature acquisition in predictive modeling is an important task in many practical applications. For example, in patient health prediction, we do not fully observe their personal features and need to dynamically select features to acquire. Our goal is to acquire a small subset of features that maximize prediction performance. Recently, some works reformulated feature acquisition as a Markov decision process and applied reinforcement learning (RL) algorithms, where the reward reflects both prediction performance and feature acquisition cost. However, RL algorithms only use zeroth-order information on the reward, which leads to slow empirical convergence, especially when there are many actions (number of features) to consider. For predictive modeling, it is possible to use first-order information on the reward, i.e., gradients, since we are often given an already collected dataset. Therefore, we propose differentiable feature acquisition (DiFA), which uses a differentiable representation of the feature selection policy to enable gradients to flow from the prediction loss to the policy parameters. We conduct extensive experiments on various real-world datasets and show that DiFA significantly outperforms existing feature acquisition methods when the number of features is large.

        ----

        ## [865] Local Intrinsic Dimensional Entropy

        **Authors**: *Rohan Ghosh, Mehul Motani*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25935](https://doi.org/10.1609/aaai.v37i6.25935)

        **Abstract**:

        Most entropy measures depend on the spread of the probability distribution over the sample space |X|, and the maximum entropy achievable scales proportionately with the sample space cardinality |X|. For a finite |X|, this yields robust entropy measures which satisfy many important properties, such as invariance to bijections, while the same is not true for continuous spaces (where |X|=infinity). Furthermore, since R and R^d (d in Z+) have the same cardinality (from Cantor's correspondence argument), cardinality-dependent entropy measures cannot encode the data dimensionality. In this work, we question the role of cardinality and distribution spread in defining entropy measures for continuous spaces, which can undergo multiple rounds of transformations and distortions, e.g., in neural networks. We find that the average value of the local intrinsic dimension of a distribution, denoted as ID-Entropy, can serve as a robust entropy measure for continuous spaces, while capturing the data dimensionality. We find that ID-Entropy satisfies many desirable properties and can be extended to conditional entropy, joint entropy and mutual-information variants. ID-Entropy also yields new information bottleneck principles and also links to causality. In the context of deep learning, for feedforward architectures, we show, theoretically and empirically, that the ID-Entropy of a hidden layer directly controls the generalization gap for both classifiers and auto-encoders, when the target function is Lipschitz continuous. Our work primarily shows that, for continuous spaces, taking a structural rather than a statistical approach yields entropy measures which preserve intrinsic data dimensionality, while being relevant for studying various architectures.

        ----

        ## [866] Improving Uncertainty Quantification of Deep Classifiers via Neighborhood Conformal Prediction: Novel Algorithm and Theoretical Analysis

        **Authors**: *Subhankar Ghosh, Taha Belkhouja, Yan Yan, Janardhan Rao Doppa*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25936](https://doi.org/10.1609/aaai.v37i6.25936)

        **Abstract**:

        Safe deployment of deep neural networks in high-stake real-world applications require theoretically sound uncertainty quantification. Conformal prediction (CP) is a principled framework for uncertainty quantification of deep models in the form of prediction set for classification tasks with a user-specified  coverage (i.e., true class label is contained with high probability). This paper proposes a novel algorithm referred to as Neighborhood Conformal Prediction (NCP) to improve the efficiency of uncertainty quantification from CP for deep classifiers (i.e., reduce prediction set size). The key idea behind NCP is to use the learned representation of the neural network to identify k nearest-neighbor calibration examples for a given testing input and assign them importance weights proportional to their distance to create adaptive prediction sets. We theoretically show that if the learned data representation of the neural network satisfies some mild conditions, NCP will produce smaller prediction sets than traditional CP algorithms. Our comprehensive experiments on CIFAR-10, CIFAR-100, and ImageNet datasets using diverse deep neural networks strongly demonstrate that NCP leads to significant reduction in prediction set size over prior CP methods.

        ----

        ## [867] Adaptive Hierarchy-Branch Fusion for Online Knowledge Distillation

        **Authors**: *Linrui Gong, Shaohui Lin, Baochang Zhang, Yunhang Shen, Ke Li, Ruizhi Qiao, Bo Ren, Muqing Li, Zhou Yu, Lizhuang Ma*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25937](https://doi.org/10.1609/aaai.v37i6.25937)

        **Abstract**:

        Online Knowledge Distillation (OKD) is designed to alleviate the dilemma that the high-capacity pre-trained teacher model is not available. However, the existing methods mostly focus on improving the ensemble prediction accuracy from multiple students (a.k.a. branches), which often overlook the homogenization problem that makes student model saturate quickly and hurts the performance. We assume that the intrinsic bottleneck of the homogenization problem comes from the identical branch architecture and coarse ensemble strategy. We propose a novel Adaptive Hierarchy-Branch Fusion framework for Online Knowledge Distillation, termed AHBF-OKD, which designs hierarchical branches and adaptive hierarchy-branch fusion module to boost the model diversity and aggregate complementary knowledge. Specifically, we first introduce hierarchical branch architectures to construct diverse peers by increasing the depth of branches monotonously on the basis of target branch. To effectively transfer knowledge from the most complex branch to the simplest target branch, we propose an adaptive hierarchy-branch fusion module to create hierarchical teacher assistants recursively, which regards the target branch as the smallest teacher assistant. During the training, the teacher assistant from the previous hierarchy is explicitly distilled by the teacher assistant and the branch from the current hierarchy. Thus, the important scores to different branches are effectively and adaptively allocated to reduce the branch homogenization. Extensive experiments demonstrate the effectiveness of AHBF-OKD on different datasets, including CIFAR-10/100 and ImageNet 2012. For example, on ImageNet 2012, the distilled ResNet-18 achieves Top-1 error of 29.28\%, which significantly outperforms the state-of-the-art  methods. The source code is available at https://github.com/linruigong965/AHBF.

        ----

        ## [868] Deep Latent Regularity Network for Modeling Stochastic Partial Differential Equations

        **Authors**: *Shiqi Gong, Peiyan Hu, Qi Meng, Yue Wang, Rongchan Zhu, Bingguang Chen, Zhiming Ma, Hao Ni, Tie-Yan Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25938](https://doi.org/10.1609/aaai.v37i6.25938)

        **Abstract**:

        Stochastic partial differential equations (SPDEs) are crucial for modelling dynamics with randomness in many areas including economics, physics, and atmospheric sciences. Recently, using deep learning approaches to learn the PDE solution for accelerating PDE simulation becomes increasingly popular. However, SPDEs have two unique properties that require new design on the models. First, the model to approximate the solution of SPDE should be generalizable over both initial conditions and the random sampled forcing term. Second, the random forcing terms usually have poor regularity whose statistics may diverge (e.g., the space-time white noise). To deal with the problems, in this work, we design a deep neural network called \emph{Deep Latent Regularity Net} (DLR-Net).  DLR-Net includes a regularity feature block as the main component, which maps the initial condition and the random forcing term to a set of regularity features. The processing of regularity features is inspired by regularity structure theory and the features provably compose a set of basis to represent the SPDE solution. The regularity features are then fed into a small backbone neural operator to get the output. We conduct experiments on various SPDEs including the dynamic $\Phi^4_1$ model and the stochastic 2D Navier-Stokes equation to predict their solutions, and the results demonstrate that the proposed DLR-Net can achieve SOTA accuracy compared with the baselines. Moreover, the inference time is over 20 times faster than the traditional numerical solver and is comparable with the baseline deep learning models.

        ----

        ## [869] Dynamic Representation Learning with Temporal Point Processes for Higher-Order Interaction Forecasting

        **Authors**: *Tony Gracious, Ambedkar Dukkipati*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25939](https://doi.org/10.1609/aaai.v37i6.25939)

        **Abstract**:

        The explosion of digital information and the growing involvement of people in social networks led to enormous research activity to develop methods that can extract meaningful information from interaction data. Commonly, interactions are represented by edges in a network or a graph, which implicitly assumes that the interactions are pairwise and static. However, real-world interactions deviate from these assumptions: (i) interactions can be multi-way, involving more than two nodes or individuals (e.g., family relationships, protein interactions), and (ii) interactions can change over a period of time (e.g., change of opinions and friendship status). While pairwise interactions have been studied in a dynamic network setting and multi-way interactions have been studied using hypergraphs in static networks, there exists no method, at present, that can predict multi-way interactions or hyperedges in dynamic settings. Existing related methods cannot answer temporal queries like what type of interaction will occur next and when it will occur. This paper proposes a temporal point process model for hyperedge prediction to address these problems. Our proposed model uses dynamic representation learning techniques for nodes in a neural point process framework to forecast hyperedges. We present several experimental results and set benchmark results. As far as our knowledge, this is the first work that uses the temporal point process to forecast hyperedges in dynamic networks.

        ----

        ## [870] An Adaptive Layer to Leverage Both Domain and Task Specific Information from Scarce Data

        **Authors**: *Gaël Guibon, Matthieu Labeau, Luce Lefeuvre, Chloé Clavel*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25940](https://doi.org/10.1609/aaai.v37i6.25940)

        **Abstract**:

        Many companies make use of customer service chats to help the customer and try to solve their problem. However, customer service data is confidential and as such, cannot easily be shared in the research community. This also implies that these
data are rarely labeled, making it difficult to take advantage of it with machine learning methods. In this paper we present the first work on a customer’s problem status prediction and identification of problematic conversations. Given very small
subsets of labeled textual conversations and unlabeled ones, we propose a semi-supervised framework dedicated to customer service data leveraging speaker role information to adapt the model to the domain and the task using a two-step process. Our framework, Task-Adaptive Fine-tuning, goes from predicting customer satisfaction to identifying the status of the customer’s problem, with the latter being the main objective of the multi-task setting. It outperforms recent inductive semi-supervised approaches on this novel task while only considering a relatively low number of parameters to train on during the final target task. We believe it can not only serve models dedicated to customer service but also to any other application making use of confidential conversational data where labeled sets are rare. Source code is available at https://github.com/gguibon/taft

        ----

        ## [871] Interpolating Graph Pair to Regularize Graph Classification

        **Authors**: *Hongyu Guo, Yongyi Mao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25941](https://doi.org/10.1609/aaai.v37i6.25941)

        **Abstract**:

        We present a simple and yet effective interpolation-based regularization technique, aiming to improve the generalization of  Graph Neural Networks (GNNs) on supervised graph classification. We leverage  Mixup, an effective regularizer for vision, where random sample pairs and their labels are interpolated to create synthetic images for training. Unlike images with grid-like coordinates, graphs have arbitrary structure and topology, which can be very sensitive to any modification that alters the graph's semantic meanings. This posts two unanswered questions for Mixup-like regularization schemes: Can we directly mix up a pair of graph inputs?  If so, how well does such mixing strategy  regularize the learning of GNNs? To answer these two questions, we propose ifMixup, which first adds dummy nodes  to make two graphs have the same input size and then simultaneously performs linear interpolation between  the aligned node feature vectors and the aligned edge representations of the two graphs. We empirically show that such simple mixing schema can effectively regularize the  classification learning, resulting in superior predictive accuracy to popular graph augmentation and GNN methods.

        ----

        ## [872] Graph Knows Unknowns: Reformulate Zero-Shot Learning as Sample-Level Graph Recognition

        **Authors**: *Jingcai Guo, Song Guo, Qihua Zhou, Ziming Liu, Xiaocheng Lu, Fushuo Huo*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25942](https://doi.org/10.1609/aaai.v37i6.25942)

        **Abstract**:

        Zero-shot learning (ZSL) is an extreme case of transfer learning that aims to recognize samples (e.g., images) of unseen classes relying on a train-set covering only seen classes and a set of auxiliary knowledge (e.g., semantic descriptors). Existing methods usually resort to constructing a visual-to-semantics mapping based on features extracted from each whole sample. However, since the visual and semantic spaces are inherently independent and may exist in different manifolds, these methods may easily suffer from the domain bias problem due to the knowledge transfer from seen to unseen classes. Unlike existing works, this paper investigates the fine-grained ZSL from a novel perspective of sample-level graph. Specifically, we decompose an input into several fine-grained elements and construct a graph structure per sample to measure and utilize element-granularity relations within each sample. Taking advantage of recently developed graph neural networks (GNNs), we formulate the ZSL problem to a graph-to-semantics mapping task, which can better exploit element-semantics correlation and local sub-structural information in samples. Experimental results on the widely used benchmark datasets demonstrate that the proposed method can mitigate the domain bias problem and achieve competitive performance against other representative methods.

        ----

        ## [873] Self-Supervised Bidirectional Learning for Graph Matching

        **Authors**: *Wenqi Guo, Lin Zhang, Shikui Tu, Lei Xu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25943](https://doi.org/10.1609/aaai.v37i6.25943)

        **Abstract**:

        Deep learning methods have demonstrated promising performance on the NP-hard Graph Matching (GM) problems. However, the state-of-the-art methods usually require the ground-truth labels, which may take extensive human efforts or be impractical to collect. In this paper, we present a robust self-supervised bidirectional learning method (IA-SSGM) to tackle GM in an unsupervised manner. It involves an affinity learning component and a classic GM solver. Specifically, we adopt the Hungarian solver to generate pseudo correspondence labels for the simple probabilistic relaxation of the affinity matrix. In addition, a bidirectional recycling consistency module is proposed to generate pseudo samples by recycling the pseudo correspondence back to permute the input. It imposes a consistency constraint between the pseudo affinity and the original one, which is theoretically supported to help reduce the matching error. Our method further develops a graph contrastive learning jointly with the affinity learning to enhance its robustness against the noise and outliers in real applications. Experiments deliver superior performance over the previous state-of-the-arts on five real-world benchmarks, especially under the more difficult outlier scenarios, demon- strating the effectiveness of our method.

        ----

        ## [874] Boosting Graph Neural Networks via Adaptive Knowledge Distillation

        **Authors**: *Zhichun Guo, Chunhui Zhang, Yujie Fan, Yijun Tian, Chuxu Zhang, Nitesh V. Chawla*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25944](https://doi.org/10.1609/aaai.v37i6.25944)

        **Abstract**:

        Graph neural networks (GNNs) have shown remarkable performance on diverse graph mining tasks. While sharing the same message passing framework, our study shows that different GNNs learn distinct knowledge from the same graph. This implies potential performance improvement by distilling the complementary knowledge from multiple models. However, knowledge distillation (KD) transfers knowledge from high-capacity teachers to a lightweight student, which deviates from our scenario: GNNs are often shallow. To transfer knowledge effectively, we need to tackle two challenges: how to transfer knowledge from compact teachers to a student with the same capacity; and, how to exploit student GNN's own learning ability. In this paper, we propose a novel adaptive KD framework, called BGNN, which sequentially transfers knowledge from multiple GNNs into a student GNN. We also introduce an adaptive temperature module and a weight boosting module. These modules guide the student to the appropriate knowledge for effective learning. Extensive experiments have demonstrated the effectiveness of BGNN. In particular, we achieve up to 3.05% improvement for node classification and 6.35% improvement for graph classification over vanilla GNNs.

        ----

        ## [875] Dream to Generalize: Zero-Shot Model-Based Reinforcement Learning for Unseen Visual Distractions

        **Authors**: *Jeongsoo Ha, Kyungsoo Kim, Yusung Kim*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25945](https://doi.org/10.1609/aaai.v37i6.25945)

        **Abstract**:

        Model-based reinforcement learning (MBRL) has been used to efficiently solve vision-based control tasks in high-dimensional image observations. Although recent MBRL algorithms perform well in trained observations, they fail when faced with visual distractions in observations. These task-irrelevant distractions (e.g., clouds, shadows, and light) may be constantly present in real-world scenarios. In this study, we propose a novel self-supervised method, Dream to Generalize (Dr. G), for zero-shot MBRL. Dr. G trains its encoder and world model with dual contrastive learning which efficiently captures task-relevant features among multi-view data augmentations. We also introduce a recurrent state inverse dynamics model that helps the world model to better understand the temporal structure. The proposed methods can enhance the robustness of the world model against visual distractions. To evaluate the generalization performance, we first train Dr. G on simple backgrounds and then test it on complex natural video backgrounds in the DeepMind Control suite, and the randomizing environments in Robosuite. Dr. G yields a performance improvement of 117% and 14% over prior works, respectively. Our code is open-sourced and available at https://github.com/JeongsooHa/DrG.git

        ----

        ## [876] Discriminability and Transferability Estimation: A Bayesian Source Importance Estimation Approach for Multi-Source-Free Domain Adaptation

        **Authors**: *Zhongyi Han, Zhiyan Zhang, Fan Wang, Rundong He, Wan Su, Xiaoming Xi, Yilong Yin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25946](https://doi.org/10.1609/aaai.v37i6.25946)

        **Abstract**:

        Source free domain adaptation (SFDA) transfers a single-source model to the unlabeled target domain without accessing the source data. With the intelligence development of various fields, a zoo of source models is more commonly available,  arising in a new setting called multi-source-free domain adaptation (MSFDA). We find that the critical inborn challenge of MSFDA is how to estimate the importance (contribution) of each source model. In this paper, we shed new Bayesian light on the fact that the posterior probability of source importance connects to discriminability and transferability. We propose Discriminability And Transferability Estimation (DATE), a universal solution for source importance estimation. Specifically, a proxy discriminability perception module equips with habitat uncertainty and density to evaluate each sample's surrounding environment. A source-similarity transferability perception module quantifies the data distribution similarity and encourages the transferability to be reasonably distributed with a domain diversity loss. Extensive experiments show that DATE can precisely and objectively estimate the source importance and outperform prior arts by non-trivial margins. Moreover, experiments demonstrate that DATE can take the most popular SFDA networks as backbones and make them become advanced MSFDA solutions.

        ----

        ## [877] Astromorphic Self-Repair of Neuromorphic Hardware Systems

        **Authors**: *Zhuangyu Han, A. N. M. Nafiul Islam, Abhronil Sengupta*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25947](https://doi.org/10.1609/aaai.v37i6.25947)

        **Abstract**:

        While neuromorphic computing architectures based on Spiking Neural Networks (SNNs) are increasingly gaining interest as a pathway toward bio-plausible machine learning, attention is still focused on computational units like the neuron and synapse. Shifting from this neuro-synaptic perspective, this paper attempts to explore the self-repair role of glial cells, in particular, astrocytes. The work investigates stronger correlations with astrocyte computational neuroscience models to develop macro-models with a higher degree of bio-fidelity that accurately captures the dynamic behavior of the self-repair process. Hardware-software co-design analysis reveals that bio-morphic astrocytic regulation has the potential to self-repair hardware realistic faults in neuromorphic hardware systems with significantly better accuracy and repair convergence for unsupervised learning tasks on the MNIST and F-MNIST datasets. Our implementation source code and trained models are available at https://github.com/NeuroCompLab-psu/Astromorphic_Self_Repair.

        ----

        ## [878] Estimating Regression Predictive Distributions with Sample Networks

        **Authors**: *Ali Harakeh, Jordan Sir Kwang Hu, Naiqing Guan, Steven L. Waslander, Liam Paull*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25948](https://doi.org/10.1609/aaai.v37i6.25948)

        **Abstract**:

        Estimating the uncertainty in deep neural network predictions is crucial for many real-world applications. A common approach to model uncertainty is to choose a parametric distribution and fit the data to it using maximum likelihood estimation. The chosen parametric form can be a poor fit to the data-generating distribution, resulting in unreliable uncertainty estimates. In this work, we propose SampleNet, a flexible and scalable architecture for modeling uncertainty that avoids specifying a parametric form on the output distribution. SampleNets do so by defining an empirical distribution using samples that are learned with the Energy Score and regularized with the Sinkhorn Divergence. SampleNets are shown to be able to well-fit a wide range of distributions and to outperform baselines on large-scale real-world regression tasks.

        ----

        ## [879] NAS-LID: Efficient Neural Architecture Search with Local Intrinsic Dimension

        **Authors**: *Xin He, Jiangchao Yao, Yuxin Wang, Zhenheng Tang, Ka Chun Cheung, Simon See, Bo Han, Xiaowen Chu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25949](https://doi.org/10.1609/aaai.v37i6.25949)

        **Abstract**:

        One-shot neural architecture search (NAS) substantially improves the search efficiency by training one supernet to estimate the performance of every possible child architecture (i.e., subnet). However, the inconsistency of characteristics among subnets incurs serious interference in the optimization, resulting in poor performance ranking correlation of subnets. Subsequent explorations decompose supernet weights via a particular criterion, e.g., gradient matching, to reduce the interference; yet they suffer from huge computational cost and low space separability. In this work, we propose a lightweight and effective local intrinsic dimension (LID)-based method NAS-LID. NAS-LID evaluates the geometrical properties of architectures by calculating the low-cost LID features layer-by-layer, and the similarity characterized by LID enjoys better separability compared with gradients, which thus effectively reduces the interference among subnets. Extensive experiments on NASBench-201 indicate that NAS-LID achieves superior performance with better efficiency. Specifically, compared to the gradient-driven method, NAS-LID can save up to 86% of GPU memory overhead when searching on NASBench-201. We also demonstrate the effectiveness of NAS-LID on ProxylessNAS and OFA spaces. Source code:https://github.com/marsggbo/NAS-LID.

        ----

        ## [880] Safeguarded Learned Convex Optimization

        **Authors**: *Howard Heaton, Xiaohan Chen, Zhangyang Wang, Wotao Yin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25950](https://doi.org/10.1609/aaai.v37i6.25950)

        **Abstract**:

        Applications abound in which optimization problems must be repeatedly solved, each time with new (but similar) data. Analytic optimization algorithms can be hand-designed to provably solve these problems in an iterative fashion. On one hand, data-driven algorithms can "learn to optimize" (L2O) with much fewer iterations and similar cost per iteration as general-purpose optimization algorithms. On the other hand, unfortunately, many L2O algorithms lack converge guarantees. To fuse the advantages of these approaches, we present a Safe-L2O framework. Safe-L2O updates incorporate a safeguard to guarantee convergence for convex problems with proximal and/or gradient oracles. The safeguard is simple and computationally cheap to implement, and it is activated only when the data-driven L2O updates would perform poorly or appear to diverge. This yields the numerical benefits of employing machine learning to create rapid L2O algorithms while still guaranteeing convergence. Our numerical examples show convergence of Safe-L2O algorithms, even when the provided data is not from the distribution of training data.

        ----

        ## [881] Improving Long-Horizon Imitation through Instruction Prediction

        **Authors**: *Joey Hejna, Pieter Abbeel, Lerrel Pinto*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25951](https://doi.org/10.1609/aaai.v37i7.25951)

        **Abstract**:

        Complex, long-horizon planning and its combinatorial nature pose steep challenges for learning-based agents. Difficulties in such settings are exacerbated in low data regimes where over-fitting stifles generalization and compounding errors hurt accuracy. In this work, we explore the use of an often unused source of auxiliary supervision: language. Inspired by recent advances in transformer-based models, we train agents with an instruction prediction loss that encourages learning temporally extended representations that operate at a high level of abstraction. Concretely, we demonstrate that instruction modeling significantly improves performance in planning environments when training with a limited number of demonstrations on the BabyAI and Crafter benchmarks. In further analysis we find that instruction modeling is most important for tasks that require complex reasoning, while understandably offering smaller gains in environments that require simple plans. More details and code can be found at \url{https://github.com/jhejna/instruction-prediction}.

        ----

        ## [882] Self-Supervised Learning for Anomalous Channel Detection in EEG Graphs: Application to Seizure Analysis

        **Authors**: *Thi Kieu Khanh Ho, Narges Armanfard*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25952](https://doi.org/10.1609/aaai.v37i7.25952)

        **Abstract**:

        Electroencephalogram (EEG) signals are effective tools towards seizure analysis where one of the most important challenges is accurate detection of seizure events and brain regions in which seizure happens or initiates. However, all existing machine learning-based algorithms for seizure analysis require access to the labeled seizure data while acquiring labeled data is very labor intensive, expensive, as well as clinicians dependent given the subjective nature of the visual qualitative interpretation of EEG signals. In this paper, we propose to detect seizure channels and clips in a self-supervised manner where no access to the seizure data is needed. The proposed method considers local structural and contextual information embedded in EEG graphs by employing positive and negative sub-graphs. We train our method through minimizing contrastive and generative losses. The employ of local EEG sub-graphs makes the algorithm an appropriate choice when accessing to the all EEG channels is impossible due to complications such as skull fractures. We conduct an extensive set of experiments on the largest seizure dataset and demonstrate that our proposed framework outperforms the state-of-the-art methods in the EEG-based seizure study. The proposed method is the only study that requires no access to the seizure data in its training phase, yet establishes a new state-of-the-art to the field, and outperforms all related supervised methods.

        ----

        ## [883] Improving Pareto Front Learning via Multi-Sample Hypernetworks

        **Authors**: *Long P. Hoang, Dung D. Le, Tran Anh Tuan, Tran Ngoc Thang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25953](https://doi.org/10.1609/aaai.v37i7.25953)

        **Abstract**:

        Pareto Front Learning (PFL) was recently introduced as an effective approach to obtain a mapping function from a given trade-off vector to a solution on the Pareto front, which solves the multi-objective optimization (MOO) problem. Due to the inherent trade-off between conflicting objectives, PFL offers a flexible approach in many scenarios in which the decision makers can not specify the preference of one Pareto solution over another, and must switch between them depending on the situation. However, existing PFL methods ignore the relationship between the solutions during the optimization process, which hinders the quality of the obtained front. To overcome this issue, we propose a novel PFL framework namely PHN-HVI, which employs a hypernetwork to generate multiple solutions from a set of diverse trade-off preferences and enhance the quality of the Pareto front by maximizing the Hypervolume indicator defined by these solutions. The experimental results on several MOO machine learning tasks show that the proposed framework significantly outperforms the baselines in producing the trade-off Pareto front.

        ----

        ## [884] Towards Better Visualizing the Decision Basis of Networks via Unfold and Conquer Attribution Guidance

        **Authors**: *Jung-Ho Hong, Woo-Jeoung Nam, Kyu-Sung Jeon, Seong-Whan Lee*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25954](https://doi.org/10.1609/aaai.v37i7.25954)

        **Abstract**:

        Revealing the transparency of Deep Neural Networks (DNNs) has been widely studied to describe the decision mechanisms of network inner structures. In this paper, we propose a novel post-hoc framework, Unfold and Conquer Attribution Guidance (UCAG), which enhances the explainability of the network decision by spatially scrutinizing the input features with respect to the model confidence. Addressing the phenomenon of missing detailed descriptions, UCAG sequentially complies with the confidence of slices of the image, leading to providing an abundant and clear interpretation. Therefore, it is possible to enhance the representation ability of explanation by preserving the detailed descriptions of assistant input features, which are commonly overwhelmed by the main meaningful regions. We conduct numerous evaluations to validate the performance in several metrics: i) deletion and insertion, ii) (energy-based) pointing games, and iii) positive and negative density maps. Experimental results, including qualitative comparisons, demonstrate that our method outperforms the existing methods with the nature of clear and detailed explanations and applicability.

        ----

        ## [885] Federated Robustness Propagation: Sharing Adversarial Robustness in Heterogeneous Federated Learning

        **Authors**: *Junyuan Hong, Haotao Wang, Zhangyang Wang, Jiayu Zhou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25955](https://doi.org/10.1609/aaai.v37i7.25955)

        **Abstract**:

        Federated learning (FL) emerges as a popular distributed learning schema that learns a model from a set of participating users without sharing raw data. One major challenge of FL comes with heterogeneous users, who may have distributionally different (or non-iid) data and varying computation resources. As federated users would use the model for prediction, they often demand the trained model to be robust against malicious attackers at test time. Whereas adversarial training (AT) provides a sound solution for centralized learning, extending its usage for federated users has imposed significant challenges, as many users may have very limited training data and tight computational budgets, to afford the data-hungry and costly AT. In this paper, we study a novel FL strategy: propagating adversarial robustness from rich-resource users that can afford AT, to those with poor resources that cannot afford it, during federated learning. We show that existing FL techniques cannot be effectively integrated with the strategy to propagate robustness among non-iid users and propose an efficient propagation approach by the proper use of batch-normalization. We demonstrate the rationality and effectiveness of our method through extensive experiments. Especially, the proposed method is shown to grant federated models remarkable robustness even when only a small portion of users afford AT during learning. Source code can be accessed at https://github.com/illidanlab/FedRBN.

        ----

        ## [886] Compressed Decentralized Learning of Conditional Mean Embedding Operators in Reproducing Kernel Hilbert Spaces

        **Authors**: *Boya Hou, Sina Sanjari, Nathan Dahlin, Subhonmesh Bose*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25956](https://doi.org/10.1609/aaai.v37i7.25956)

        **Abstract**:

        Conditional mean embedding (CME) operators encode conditional probability densities within Reproducing Kernel Hilbert Space (RKHS). In this paper, we present a decentralized algorithm for a collection of agents to cooperatively approximate CME over a network. Communication constraints limit the agents from sending all data to their neighbors; we only allow sparse representations of covariance operators to be exchanged among agents, compositions of which defines CME. Using a coherence-based compression scheme, we present a consensus-type algorithm that preserves the average of the approximations of the covariance operators across the network. We theoretically prove that the iterative dynamics in RKHS is stable. We then empirically study our algorithm to estimate CMEs to learn spectra of Koopman operators for Markovian dynamical systems and to execute approximate value iteration for Markov decision processes (MDPs).

        ----

        ## [887] RLEKF: An Optimizer for Deep Potential with Ab Initio Accuracy

        **Authors**: *Siyu Hu, Wentao Zhang, Qiuchen Sha, Feng Pan, Lin-Wang Wang, Weile Jia, Guangming Tan, Tong Zhao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25957](https://doi.org/10.1609/aaai.v37i7.25957)

        **Abstract**:

        It is imperative to accelerate the training of neural network force field such as Deep Potential, which usually requires thousands of images based on first-principles calculation and a couple of days to generate an accurate potential energy surface. To this end, we propose a novel optimizer named reorganized layer extended Kalman filtering (RLEKF), an optimized version of global extended Kalman filtering (GEKF) with a strategy of splitting big and gathering small layers to overcome the O(N^2) computational cost of GEKF. This strategy provides an approximation of the dense weights error covariance matrix with a sparse diagonal block matrix for GEKF. We implement both RLEKF and the baseline Adam in our alphaDynamics package and numerical experiments are performed on 13 unbiased datasets. Overall, RLEKF converges faster with slightly better accuracy. For example, a test on a typical system, bulk copper, shows that RLEKF converges faster by both the number of training epochs (x11.67) and wall-clock time (x1.19). Besides, we theoretically prove that the updates of weights converge and thus are against the gradient exploding problem. Experimental results verify that RLEKF is not sensitive to the initialization of weights. The RLEKF sheds light on other AI-for-science applications where training a large neural network (with tons of thousands parameters) is a bottleneck.

        ----

        ## [888] Background-Mixed Augmentation for Weakly Supervised Change Detection

        **Authors**: *Rui Huang, Ruofei Wang, Qing Guo, Jieda Wei, Yuxiang Zhang, Wei Fan, Yang Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25958](https://doi.org/10.1609/aaai.v37i7.25958)

        **Abstract**:

        Change detection (CD) is to decouple object changes (i.e., object missing or appearing) from background changes (i.e., environment variations) like light and season variations in two images captured in the same scene over a long time span, presenting critical applications in disaster management, urban development, etc. In particular, the endless patterns of background changes require detectors to have a high generalization against unseen environment variations, making this task significantly challenging. Recent deep learning-based methods develop novel network architectures or optimization strategies with paired-training examples, which do not handle the generalization issue explicitly and require huge manual pixel-level annotation efforts. In this work, for the first attempt in the CD community, we study the generalization issue of CD from the perspective of data augmentation and develop a novel weakly supervised training algorithm that only needs image-level labels. Different from general augmentation techniques for classification, we propose the background-mixed augmentation that is specifically designed for change detection by augmenting examples under the guidance of a set of background changing images and letting deep CD models see diverse environment variations. Moreover, we propose the augmented & real data consistency loss that encourages the generalization increase significantly. Our method as a general framework can enhance a wide range of existing deep learning-based detectors. We conduct extensive experiments in two public datasets and enhance four state-of-the-art methods, demonstrating the advantages of our method. We release the code at https://github.com/tsingqguo/bgmix.

        ----

        ## [889] Enabling Knowledge Refinement upon New Concepts in Abductive Learning

        **Authors**: *Yu-Xuan Huang, Wang-Zhou Dai, Yuan Jiang, Zhi-Hua Zhou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25959](https://doi.org/10.1609/aaai.v37i7.25959)

        **Abstract**:

        Recently there are great efforts on leveraging machine learning and logical reasoning. Many approaches start from a given knowledge base, and then try to utilize the knowledge to help machine learning. In real practice, however, the given knowledge base can often be incomplete or even noisy, and thus, it is crucial to develop the ability of knowledge refinement or enhancement. This paper proposes to enable the Abductive learning (ABL) paradigm to have the ability of knowledge refinement/enhancement. In particular, we focus on the problem that, in contrast to closed-environment tasks where a fixed set of symbols are enough to represent the concepts in the domain, in open-environment tasks new concepts may emerge. Ignoring those new concepts can lead to significant performance decay, whereas it is challenging to identify new concepts and add them to the existing knowledge base with potential conflicts resolved. We propose the ABL_nc approach which exploits machine learning in ABL to identify new concepts from data, exploits knowledge graph to match them with entities, and refines existing knowledge base to resolve conflicts. The refined/enhanced knowledge base can then be used in the next loop of ABL and help improve the performance of machine learning. Experiments on three neuro-symbolic learning tasks verified the effectiveness of the proposed approach.

        ----

        ## [890] Self-Supervised Graph Attention Networks for Deep Weighted Multi-View Clustering

        **Authors**: *Zongmo Huang, Yazhou Ren, Xiaorong Pu, Shudong Huang, Zenglin Xu, Lifang He*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25960](https://doi.org/10.1609/aaai.v37i7.25960)

        **Abstract**:

        As one of the most important research topics in the unsupervised learning field, Multi-View Clustering (MVC) has been widely studied in the past decade and numerous MVC methods have been developed. Among these methods, the recently emerged Graph Neural Networks (GNN) shine a light on modeling both topological structure and node attributes in the form of graphs, to guide unified embedding learning and clustering. However, the effectiveness of existing GNN-based MVC methods is still limited due to the insufficient consideration in utilizing the self-supervised information and graph information, which can be reflected from the following two aspects: 1) most of these models merely use the self-supervised information to guide the feature learning and fail to realize that such information can be also applied in graph learning and sample weighting; 2) the usage of graph information is generally limited to the feature aggregation in these models, yet it also provides valuable evidence in detecting noisy samples. To this end, in this paper we propose Self-Supervised Graph Attention Networks for Deep Weighted Multi-View Clustering (SGDMC), which promotes the performance of GNN-based deep MVC models by making full use of the self-supervised information and graph information. Specifically, a novel attention-allocating approach that considers both the similarity of node attributes and the self-supervised information is developed to comprehensively evaluate the relevance among different nodes. Meanwhile, to alleviate the negative impact caused by noisy samples and the discrepancy of cluster structures, we further design a sample-weighting strategy based on the attention graph as well as the discrepancy between the global pseudo-labels and the local cluster assignment. Experimental results on multiple real-world datasets demonstrate the effectiveness of our method over existing approaches.

        ----

        ## [891] Reward-Biased Maximum Likelihood Estimation for Neural Contextual Bandits: A Distributional Learning Perspective

        **Authors**: *Yu-Heng Hung, Ping-Chun Hsieh*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25961](https://doi.org/10.1609/aaai.v37i7.25961)

        **Abstract**:

        Reward-biased maximum likelihood estimation (RBMLE) is a classic principle in the adaptive control literature for tackling explore-exploit trade-offs. This paper studies the neural contextual bandit problem from a distributional perspective and proposes NeuralRBMLE, which leverages the likelihood of surrogate parametric distributions to learn the unknown reward distributions and thereafter adapts the RBMLE principle to achieve efficient exploration by properly adding a reward-bias term. NeuralRBMLE leverages the representation power of neural networks and directly encodes exploratory behavior in the parameter space, without constructing confidence intervals of the estimated rewards. We propose two variants of NeuralRBMLE algorithms: The first variant directly obtains the RBMLE estimator by gradient ascent, and the second variant simplifies RBMLE to a simple index policy through an approximation. We show that both algorithms achieve order-optimality. Through extensive experiments, we demonstrate that the NeuralRBMLE algorithms achieve comparable or better empirical regrets than the state-of-the-art methods on real-world datasets with non-linear reward functions.

        ----

        ## [892] Learning Noise-Induced Reward Functions for Surpassing Demonstrations in Imitation Learning

        **Authors**: *Liangyu Huo, Zulin Wang, Mai Xu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25962](https://doi.org/10.1609/aaai.v37i7.25962)

        **Abstract**:

        Imitation learning (IL) has recently shown impressive performance in training a reinforcement learning agent with human demonstrations, eliminating the difficulty of designing elaborate reward functions in complex environments. However, most IL methods work under the assumption of the optimality of the demonstrations and thus cannot learn policies to surpass the demonstrators. Some methods have been investigated to obtain better-than-demonstration (BD) performance with inner human feedback or preference labels. In this paper, we propose a method to learn rewards from suboptimal demonstrations via a weighted preference learning technique (LERP). Specifically, we first formulate the suboptimality of demonstrations as the inaccurate estimation of rewards. The inaccuracy is modeled with a reward noise random variable following the Gumbel distribution. Moreover, we derive an upper bound of the expected return with different noise coefficients and propose a theorem to surpass the demonstrations. Unlike existing literature, our analysis does not depend on the linear reward constraint. Consequently, we develop a BD model with a weighted preference learning technique. Experimental results on continuous control and high-dimensional discrete control tasks show the superiority of our LERP method over other state-of-the-art BD  methods.

        ----

        ## [893] XClusters: Explainability-First Clustering

        **Authors**: *Hyunseung Hwang, Steven Euijong Whang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25963](https://doi.org/10.1609/aaai.v37i7.25963)

        **Abstract**:

        We study the problem of explainability-first clustering where explainability becomes a first-class citizen for clustering. Previous clustering approaches use decision trees for explanation, but only after the clustering is completed. In contrast, our approach is to perform clustering and decision tree training holistically where the decision tree's performance and size also influence the clustering results. We assume the attributes for clustering and explaining are distinct, although this is not necessary. We observe that our problem is a monotonic optimization where the objective function is a difference of monotonic functions. We then propose an efficient branch-and-bound algorithm for finding the best parameters that lead to a balance of clustering accuracy and decision tree explainability. Our experiments show that our method can improve the explainability of any clustering that fits in our framework.

        ----

        ## [894] Model-Based Reinforcement Learning with Multinomial Logistic Function Approximation

        **Authors**: *Taehyun Hwang, Min-hwan Oh*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25964](https://doi.org/10.1609/aaai.v37i7.25964)

        **Abstract**:

        We study model-based reinforcement learning (RL) for episodic Markov decision processes (MDP) whose transition probability is parametrized by an unknown transition core with features of state and action. Despite much recent progress in analyzing algorithms in the linear MDP setting, the understanding of more general transition models is very restrictive. In this paper, we propose a provably efficient RL algorithm for the MDP whose state transition is given by a multinomial logistic model. We show that our proposed algorithm based on the upper confidence bounds achieves O(d√(H^3 T)) regret bound where d is the dimension of the transition core, H is the horizon, and T is the total number of steps. To the best of our knowledge, this is the first model-based RL algorithm with multinomial logistic function approximation with provable guarantees. We also comprehensively evaluate our proposed algorithm numerically and show that it consistently outperforms the existing methods, hence achieving both provable efficiency and practical superior performance.

        ----

        ## [895] Fast Regularized Discrete Optimal Transport with Group-Sparse Regularizers

        **Authors**: *Yasutoshi Ida, Sekitoshi Kanai, Kazuki Adachi, Atsutoshi Kumagai, Yasuhiro Fujiwara*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25965](https://doi.org/10.1609/aaai.v37i7.25965)

        **Abstract**:

        Regularized discrete optimal transport (OT) is a powerful tool to measure the distance between two discrete distributions that have been constructed from data samples on two different domains. While it has a wide range of applications in machine learning, in some cases the sampled data from only one of the domains will have class labels such as unsupervised domain adaptation. In this kind of problem setting, a group-sparse regularizer is frequently leveraged as a regularization term to handle class labels. In particular, it can preserve the label structure on the data samples by corresponding the data samples with the same class label to one group-sparse regularization term. As a result, we can measure the distance while utilizing label information by solving the regularized optimization problem with gradient-based algorithms. However, the gradient computation is expensive when the number of classes or data samples is large because the number of regularization terms and their respective sizes also turn out to be large. This paper proposes fast discrete OT with group-sparse regularizers. Our method is based on two ideas. The first is to safely skip the computations of the gradients that must be zero. The second is to efficiently extract the gradients that are expected to be nonzero. Our method is guaranteed to return the same value of the objective function as that of the original approach. Experiments demonstrate that our method is up to 8.6 times faster than the original method without degrading accuracy.

        ----

        ## [896] Neural Representations Reveal Distinct Modes of Class Fitting in Residual Convolutional Networks

        **Authors**: *Michal Jamroz, Marcin Kurdziel*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25966](https://doi.org/10.1609/aaai.v37i7.25966)

        **Abstract**:

        We leverage probabilistic models of neural representations to investigate how residual networks fit classes. To this end, we estimate class-conditional density models for representations learned by deep ResNets. We then use these models to characterize distributions of representations across learned classes. Surprisingly, we find that classes in the investigated models are not fitted in a uniform way. On the contrary: we uncover two groups of classes that are fitted with markedly different distributions of representations. These distinct modes of class-fitting are evident only in the deeper layers of the investigated models, indicating that they are not related to low-level image features. We show that the uncovered structure in neural representations correlate with memorization of training examples and adversarial robustness. Finally, we compare class-conditional distributions of neural representations between memorized and typical examples. This allows us to uncover where in the network structure class labels arise for memorized and standard inputs.

        ----

        ## [897] Audio-Visual Contrastive Learning with Temporal Self-Supervision

        **Authors**: *Simon Jenni, Alexander Black, John P. Collomosse*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25967](https://doi.org/10.1609/aaai.v37i7.25967)

        **Abstract**:

        We propose a self-supervised learning approach for videos that learns representations of both the RGB frames and the accompanying audio without human supervision.   
In contrast to images that capture the static scene appearance, videos also contain sound and temporal scene dynamics.  
To leverage the temporal and aural dimension inherent to videos, our method extends temporal self-supervision to the audio-visual setting and integrates it with multi-modal contrastive objectives.
As temporal self-supervision, we pose playback speed and direction recognition in both modalities and propose intra- and inter-modal temporal ordering tasks. 
Furthermore, we design a novel contrastive objective in which the usual pairs are supplemented with additional sample-dependent positives and negatives sampled from the evolving feature space. 
In our model, we apply such losses among video clips and between videos and their temporally corresponding audio clips. 
We verify our model design in extensive ablation experiments and evaluate the video and audio representations in transfer experiments to action recognition and retrieval on UCF101 and HMBD51, audio classification on ESC50, and robust video fingerprinting on VGG-Sound, with state-of-the-art results.

        ----

        ## [898] Confidence-Aware Training of Smoothed Classifiers for Certified Robustness

        **Authors**: *Jongheon Jeong, Seojin Kim, Jinwoo Shin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25968](https://doi.org/10.1609/aaai.v37i7.25968)

        **Abstract**:

        Any classifier can be "smoothed out" under Gaussian noise to build a new classifier that is provably robust to l2-adversarial perturbations, viz., by averaging its predictions over the noise via randomized smoothing. Under the smoothed classifiers, the fundamental trade-off between accuracy and (adversarial) robustness has been well evidenced in the literature: i.e., increasing the robustness of a classifier for an input can be at the expense of decreased accuracy for some other inputs. In this paper, we propose a simple training method leveraging this trade-off to obtain robust smoothed classifiers, in particular, through a sample-wise control of robustness over the training samples. We make this control feasible by using "accuracy under Gaussian noise" as an easy-to-compute proxy of adversarial robustness for an input. Specifically, we differentiate the training objective depending on this proxy to filter out samples that are unlikely to benefit from the worst-case (adversarial) objective. Our experiments show that the proposed method, despite its simplicity, consistently exhibits improved certified robustness upon state-of-the-art training methods. Somewhat surprisingly, we find these improvements persist even for other notions of robustness, e.g., to various types of common corruptions. Code is available at https://github.com/alinlab/smoothing-catrs.

        ----

        ## [899] Learnable Path in Neural Controlled Differential Equations

        **Authors**: *Sheo Yon Jhin, Minju Jo, Seungji Kook, Noseong Park*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25969](https://doi.org/10.1609/aaai.v37i7.25969)

        **Abstract**:

        Neural controlled differential equations (NCDEs), which are continuous analogues to recurrent neural networks (RNNs), are a specialized model in (irregular) time-series processing. In comparison with similar models, e.g., neural ordinary differential equations (NODEs), the key distinctive characteristics of NCDEs are i) the adoption of the continuous path created by an interpolation algorithm from each raw discrete time-series sample and ii) the adoption of the Riemann--Stieltjes integral. It is the continuous path which makes NCDEs be analogues to continuous RNNs. However, NCDEs use existing interpolation algorithms to create the path, which is unclear whether they can create an optimal path. To this end, we present a method to generate another latent path (rather than relying on existing interpolation algorithms), which is identical to learning an appropriate interpolation method. We design an encoder-decoder module based on NCDEs and NODEs, and a special training method for it. Our method shows the best performance in both time-series classification and forecasting.

        ----

        ## [900] DrugOOD: Out-of-Distribution Dataset Curator and Benchmark for AI-Aided Drug Discovery - a Focus on Affinity Prediction Problems with Noise Annotations

        **Authors**: *Yuanfeng Ji, Lu Zhang, Jiaxiang Wu, Bingzhe Wu, Lanqing Li, Long-Kai Huang, Tingyang Xu, Yu Rong, Jie Ren, Ding Xue, Houtim Lai, Wei Liu, Junzhou Huang, Shuigeng Zhou, Ping Luo, Peilin Zhao, Yatao Bian*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25970](https://doi.org/10.1609/aaai.v37i7.25970)

        **Abstract**:

        AI-aided drug discovery (AIDD) is gaining popularity due to its potential to make the search for new pharmaceuticals faster, less expensive, and more effective. Despite its extensive use in numerous fields (e.g., ADMET prediction, virtual screening), little research has been conducted on the out-of-distribution (OOD) learning problem with noise. We present DrugOOD, a systematic OOD dataset curator and benchmark for AIDD. Particularly, we focus on the drug-target binding affinity prediction problem, which involves both macromolecule (protein target) and small-molecule (drug compound). DrugOOD offers an automated dataset curator with user-friendly customization scripts, rich domain annotations aligned with biochemistry knowledge, realistic noise level annotations, and rigorous benchmarking of SOTA OOD algorithms, as opposed to only providing fixed datasets. Since the molecular data is often modeled as irregular graphs using graph neural network (GNN) backbones, DrugOOD also serves as a valuable testbed for graph OOD learning problems. Extensive empirical studies have revealed a significant performance gap between in-distribution and out-of-distribution experiments, emphasizing the need for the development of more effective schemes that permit OOD generalization under noise for AIDD.

        ----

        ## [901] MNER-QG: An End-to-End MRC Framework for Multimodal Named Entity Recognition with Query Grounding

        **Authors**: *Meihuizi Jia, Lei Shen, Xin Shen, Lejian Liao, Meng Chen, Xiaodong He, Zhendong Chen, Jiaqi Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25971](https://doi.org/10.1609/aaai.v37i7.25971)

        **Abstract**:

        Multimodal named entity recognition (MNER) is a critical step in information extraction, which aims to detect entity spans and classify them to corresponding entity types given a sentence-image pair. Existing methods either (1) obtain named entities with coarse-grained visual clues from attention mechanisms, or (2) first detect fine-grained visual regions with toolkits and then recognize named entities. However, they suffer from improper alignment between entity types and visual regions or error propagation in the two-stage manner, which finally imports irrelevant visual information into texts. In this paper, we propose a novel end-to-end framework named MNER-QG that can simultaneously perform MRC-based multimodal named entity recognition and query grounding. Specifically, with the assistance of queries, MNER-QG can provide prior knowledge of entity types and visual regions, and further enhance representations of both text and image. To conduct the query grounding task, we provide manual annotations and weak supervisions that are obtained via training a highly flexible visual grounding model with transfer learning. We conduct extensive experiments on two public MNER datasets, Twitter2015 and Twitter2017. Experimental results show that MNER-QG outperforms the current state-of-the-art models on the MNER task, and also improves the query grounding performance.

        ----

        ## [902] Learning from Training Dynamics: Identifying Mislabeled Data beyond Manually Designed Features

        **Authors**: *Qingrui Jia, Xuhong Li, Lei Yu, Jiang Bian, Penghao Zhao, Shupeng Li, Haoyi Xiong, Dejing Dou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25972](https://doi.org/10.1609/aaai.v37i7.25972)

        **Abstract**:

        While mislabeled or ambiguously-labeled samples in the training set could negatively affect the performance of deep models, diagnosing the dataset and identifying mislabeled samples helps to improve the generalization power. Training dynamics, i.e., the traces left by iterations of optimization algorithms, have recently been proved to be effective to localize mislabeled samples with hand-crafted features.
In this paper, beyond manually designed features, we introduce a novel learning-based solution, leveraging a noise detector, instanced by an LSTM network, which learns to predict whether a sample was mislabeled using the raw training dynamics as input. 
Specifically, the proposed method trains the noise detector in a supervised manner using the dataset with synthesized label noises and can adapt to various datasets (either naturally or synthesized label-noised) without retraining. 
We conduct extensive experiments to evaluate the proposed method.
We train the noise detector based on the synthesized label-noised CIFAR dataset and test such noise detector on Tiny ImageNet, CUB-200, Caltech-256, WebVision and Clothing1M. 
Results show that the proposed method precisely detects mislabeled samples on various datasets without further adaptation, and outperforms state-of-the-art methods.
Besides, more experiments demonstrate that the mislabel identification can guide a label correction, namely data debugging, providing orthogonal improvements of algorithm-centric state-of-the-art techniques from the data aspect.

        ----

        ## [903] Online Tuning for Offline Decentralized Multi-Agent Reinforcement Learning

        **Authors**: *Jiechuan Jiang, Zongqing Lu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25973](https://doi.org/10.1609/aaai.v37i7.25973)

        **Abstract**:

        Offline reinforcement learning could learn effective policies from a fixed dataset, which is promising for real-world applications. However, in offline decentralized multi-agent reinforcement learning, due to the discrepancy between the behavior policy and learned policy, the transition dynamics in offline experiences do not accord with the transition dynamics in online execution, which creates severe errors in value estimates, leading to uncoordinated low-performing policies. One way to overcome this problem is to bridge offline training and online tuning. However, considering both deployment efficiency and sample efficiency, we could only collect very limited online experiences, making it insufficient to use merely online data for updating the agent policy. To utilize both offline and online experiences to tune the policies of agents, we introduce online transition correction (OTC) to implicitly correct the offline transition dynamics by modifying sampling probabilities. We design two types of distances, i.e., embedding-based and value-based distance, to measure the similarity between transitions, and further propose an adaptive rank-based prioritization to sample transitions according to the transition similarity. OTC is simple yet effective to increase data efficiency and improve agent policies in online tuning. Empirically, OTC outperforms baselines in a variety of tasks.

        ----

        ## [904] Robust Domain Adaptation for Machine Reading Comprehension

        **Authors**: *Liang Jiang, Zhenyu Huang, Jia Liu, Zujie Wen, Xi Peng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25974](https://doi.org/10.1609/aaai.v37i7.25974)

        **Abstract**:

        Most domain adaptation methods for machine reading comprehension (MRC) use a pre-trained question-answer (QA) construction model to generate pseudo QA pairs for MRC transfer. Such a process will inevitably introduce mismatched pairs (i.e., Noisy Correspondence) due to i) the unavailable QA pairs in target documents, and ii) the domain shift during applying the QA construction model to the target domain. Undoubtedly, the noisy correspondence will degenerate the performance of MRC, which however is neglected by existing works. To solve such an untouched problem, we propose to construct QA pairs by additionally using the dialogue related to the documents, as well as a new domain adaptation method for MRC. Specifically, we propose Robust Domain Adaptation for Machine Reading Comprehension (RMRC) method which consists of an answer extractor (AE), a question selector (QS), and an MRC model. Specifically, RMRC filters out the irrelevant answers by estimating the correlation to the document via the AE, and extracts the questions by fusing the candidate questions in multiple rounds of dialogue chats via the QS. With the extracted QA pairs, MRC is fine-tuned and provides the feedback to optimize the QS through a novel reinforced self-training method. Thanks to the optimization of the QS, our method will greatly alleviate the noisy correspondence problem caused by the domain shift. To the best of our knowledge, this could be the first study to reveal the influence of noisy correspondence in domain adaptation MRC models and show a feasible solution to achieve the robustness against the mismatched pairs. Extensive experiments on three datasets demonstrate the effectiveness of our method.

        ----

        ## [905] Multi-View MOOC Quality Evaluation via Information-Aware Graph Representation Learning

        **Authors**: *Lu Jiang, Yibin Wang, Jianan Wang, Pengyang Wang, Minghao Yin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25975](https://doi.org/10.1609/aaai.v37i7.25975)

        **Abstract**:

        In this paper, we study the problem of MOOC quality evaluation that is essential for improving the course materials, promoting students' learning efficiency, and benefiting user services. 
While achieving promising performances, current works still suffer from the complicated interactions and relationships of entities in MOOC platforms. 
To tackle the challenges, we formulate the problem as a course representation learning task based, and develop an Information-aware Graph Representation Learning(IaGRL) for multi-view MOOC quality evaluation. 
Specifically, We first build a MOOC Heterogeneous Network (HIN) to represent the interactions and relationships among entities in MOOC platforms. 
And then we decompose the MOOC HIN into multiple single-relation graphs based on meta-paths to depict multi-view semantics of courses. 
The course representation learning can be further converted to a multi-view graph representation task. 
Different from traditional graph representation learning, the learned course representations are expected to match the following three types of validity: 
(1) the agreement on expressiveness between the raw course portfolio and the learned course representations; 
(2) the consistency between the representations in each view and the unified representations; 
(3) the alignment between the course and MOOC platform representations. 
Therefore, we propose to exploit mutual information for preserving the validity of course representations. 
We conduct extensive experiments over real-world MOOC datasets to demonstrate the effectiveness of our proposed method.

        ----

        ## [906] Spatio-Temporal Meta-Graph Learning for Traffic Forecasting

        **Authors**: *Renhe Jiang, Zhaonan Wang, Jiawei Yong, Puneet Jeph, Quanjun Chen, Yasumasa Kobayashi, Xuan Song, Shintaro Fukushima, Toyotaro Suzumura*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25976](https://doi.org/10.1609/aaai.v37i7.25976)

        **Abstract**:

        Traffic forecasting as a canonical task of multivariate time series forecasting has been a significant research topic in AI community. To address the spatio-temporal heterogeneity and non-stationarity implied in the traffic stream, in this study, we propose Spatio-Temporal Meta-Graph Learning as a novel Graph Structure Learning mechanism on spatio-temporal data. Specifically, we implement this idea into Meta-Graph Convolutional Recurrent Network (MegaCRN) by plugging the Meta-Graph Learner powered by a Meta-Node Bank into GCRN encoder-decoder. We conduct a comprehensive evaluation on two benchmark datasets (i.e., METR-LA and PEMS-BAY) and a new large-scale traffic speed dataset called EXPY-TKY that covers 1843 expressway road links in Tokyo. Our model outperformed the state-of-the-arts on all three datasets. Besides, through a series of qualitative evaluations, we demonstrate that our model can explicitly disentangle the road links and time slots with different patterns and be robustly adaptive to any anomalous traffic situations. Codes and datasets are available at https://github.com/deepkashiwa20/MegaCRN.

        ----

        ## [907] Complement Sparsification: Low-Overhead Model Pruning for Federated Learning

        **Authors**: *Xiaopeng Jiang, Cristian Borcea*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25977](https://doi.org/10.1609/aaai.v37i7.25977)

        **Abstract**:

        Federated Learning (FL) is a privacy-preserving distributed deep learning paradigm that involves substantial communication and computation effort, which is a problem for resource-constrained mobile and IoT devices. Model pruning/sparsification develops sparse models that could solve this problem, but existing sparsification solutions cannot satisfy at the same time the requirements for low bidirectional communication overhead between the server and the clients, low computation overhead at the clients, and good model accuracy, under the FL assumption that the server does not have access to raw data to fine-tune the pruned models. We propose Complement Sparsification (CS), a pruning mechanism that satisfies all these requirements through a complementary and collaborative pruning done at the server and the clients.  At each round, CS creates a global sparse model that contains the weights that capture the general data distribution of all clients, while the clients create local sparse models with the weights pruned from the global model to capture the local trends. For improved model performance, these two types of complementary sparse models are aggregated into a dense model in each round, which is subsequently pruned in an iterative process. CS requires little computation overhead on the top of vanilla FL for both the server and the clients. We demonstrate that CS is an approximation of vanilla FL and, thus, its models perform well. We evaluate CS experimentally with two popular FL benchmark datasets. CS achieves substantial reduction in bidirectional communication, while achieving performance comparable with vanilla FL. In addition, CS outperforms baseline pruning mechanisms for FL.

        ----

        ## [908] Energy-Motivated Equivariant Pretraining for 3D Molecular Graphs

        **Authors**: *Rui Jiao, Jiaqi Han, Wenbing Huang, Yu Rong, Yang Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25978](https://doi.org/10.1609/aaai.v37i7.25978)

        **Abstract**:

        Pretraining molecular representation models without labels is fundamental to various applications. Conventional methods mainly process 2D molecular graphs and focus solely on 2D tasks, making their pretrained models incapable of characterizing 3D geometry and thus defective for downstream 3D tasks. In this work, we tackle 3D molecular pretraining in a complete and novel sense. In particular, we first propose to adopt an equivariant energy-based model as the backbone for pretraining, which enjoys the merits of fulfilling the symmetry of 3D space. Then we develop a node-level pretraining loss for force prediction, where we further exploit the Riemann-Gaussian distribution to ensure the loss to be E(3)-invariant, enabling more robustness. Moreover, a graph-level noise scale prediction task is also leveraged to further promote the eventual performance. We evaluate our model pretrained from a large-scale 3D dataset GEOM-QM9 on two challenging 3D benchmarks: MD17 and QM9. Experimental results demonstrate the efficacy of our method against current state-of-the-art pretraining approaches, and verify the validity of our design for each proposed component. Code is available at https://github.com/jiaor17/3D-EMGP.

        ----

        ## [909] Local-Global Defense against Unsupervised Adversarial Attacks on Graphs

        **Authors**: *Di Jin, Bingdao Feng, Siqi Guo, Xiaobao Wang, Jianguo Wei, Zhen Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25979](https://doi.org/10.1609/aaai.v37i7.25979)

        **Abstract**:

        Unsupervised pre-training algorithms for graph representation learning are vulnerable to adversarial attacks, such as first-order perturbations on graphs, which will have an impact on particular downstream applications. Designing an effective representation learning strategy against white-box attacks remains a crucial open topic. Prior research attempts to improve representation robustness by maximizing mutual information between the representation and the perturbed graph, which is sub-optimal because it does not adapt its defense techniques to the severity of the attack. To address this issue, we propose an unsupervised defense method that combines local and global defense to improve the robustness of representation. Note that we put forward the Perturbed Edges Harmfulness (PEH) metric to determine the riskiness of the attack. Thus, when the edges are attacked, the model can automatically identify the risk of attack. We present a method of attention-based protection against high-risk attacks that penalizes attention coefficients of perturbed edges to encoders. Extensive experiments demonstrate that our strategies can enhance the robustness of representation against various adversarial attacks on three benchmark graphs.

        ----

        ## [910] Trafformer: Unify Time and Space in Traffic Prediction

        **Authors**: *Di Jin, Jiayi Shi, Rui Wang, Yawen Li, Yuxiao Huang, Yu-Bin Yang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25980](https://doi.org/10.1609/aaai.v37i7.25980)

        **Abstract**:

        Traffic prediction is an important component of the intelligent transportation system. Existing deep learning methods encode temporal information and spatial information separately or iteratively. However, the spatial and temporal information is highly correlated in a traffic network, so existing methods may not learn the complex spatial-temporal dependencies hidden in the traffic network due to the decomposed model design. To overcome this limitation, we propose a new model named Trafformer, which unifies spatial and temporal information in one transformer-style model. Trafformer enables every node at every timestamp interact with every other node in every other timestamp in just one step in the spatial-temporal correlation matrix. This design enables Trafformer to catch complex spatial-temporal dependencies. Following the same design principle, we use the generative style decoder to predict multiple timestamps in only one forward operation instead of the iterative style decoder in Transformer. Furthermore, to reduce the complexity brought about by the huge spatial-temporal self-attention matrix, we also propose two variants of Trafformer to further improve the training and inference speed without losing much effectivity. Extensive experiments on two traffic datasets demonstrate that Trafformer outperforms existing methods and provides a promising future direction for the spatial-temporal traffic prediction problem.

        ----

        ## [911] On Solution Functions of Optimization: Universal Approximation and Covering Number Bounds

        **Authors**: *Ming Jin, Vanshaj Khattar, Harshal Kaushik, Bilgehan Sel, Ruoxi Jia*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25981](https://doi.org/10.1609/aaai.v37i7.25981)

        **Abstract**:

        We study the expressibility and learnability of solution functions of convex optimization and their multi-layer architectural extension. The main results are: (1) the class of solution functions of linear programming (LP) and quadratic programming (QP) is a universal approximant for the smooth model class or some restricted Sobolev space, and we characterize the rate-distortion, (2) the approximation power is investigated through a viewpoint of regression error, where information about the target function is provided in terms of data observations, (3) compositionality in the form of deep architecture with optimization as a layer is shown to reconstruct some basic functions used in numerical analysis without error, which implies that (4) a substantial reduction in rate-distortion can be achieved with a universal network architecture, and (5) we discuss the statistical bounds of empirical covering numbers for LP/QP, as well as a generic optimization problem (possibly nonconvex) by exploiting tame geometry. Our results provide the **first rigorous analysis of the approximation and learning-theoretic properties of solution functions** with implications for algorithmic design and performance guarantees.

        ----

        ## [912] Pointerformer: Deep Reinforced Multi-Pointer Transformer for the Traveling Salesman Problem

        **Authors**: *Yan Jin, Yuandong Ding, Xuanhao Pan, Kun He, Li Zhao, Tao Qin, Lei Song, Jiang Bian*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25982](https://doi.org/10.1609/aaai.v37i7.25982)

        **Abstract**:

        Traveling Salesman Problem (TSP), as a classic routing optimization problem originally arising in the domain of transportation and logistics, has become a critical task in broader domains, such as manufacturing and biology. Recently, Deep Reinforcement Learning (DRL) has been increasingly employed to solve TSP due to its high inference efficiency. Nevertheless, most of existing end-to-end DRL algorithms only perform well on small TSP instances and can hardly generalize to large scale because of the drastically soaring memory consumption and computation time along with the enlarging problem scale. In this paper, we propose a novel end-to-end DRL approach, referred to as Pointerformer, based on multi-pointer Transformer. Particularly, Pointerformer adopts both reversible residual network in the encoder and multi-pointer network in the decoder to effectively contain memory consumption of the encoder-decoder architecture. To further improve the performance of TSP solutions, Pointerformer employs a feature augmentation method to explore the symmetries of TSP at both training and inference stages as well as an enhanced context embedding approach to include more comprehensive context information in the query. Extensive experiments on a randomly generated benchmark and a public benchmark have shown that, while achieving comparative results on most small-scale TSP instances as state-of-the-art DRL approaches do, Pointerformer can also well generalize to large-scale TSPs.

        ----

        ## [913] Knowledge-Constrained Answer Generation for Open-Ended Video Question Answering

        **Authors**: *Yao Jin, Guocheng Niu, Xinyan Xiao, Jian Zhang, Xi Peng, Jun Yu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25983](https://doi.org/10.1609/aaai.v37i7.25983)

        **Abstract**:

        Open-ended Video question answering (open-ended VideoQA) aims to understand video content and question semantics to generate the correct answers. Most of the best performing models define the problem as a discriminative task of multi-label classification. In real-world scenarios, however, it is difficult to define a candidate set that includes all possible answers. In this paper, we propose a Knowledge-constrained Generative VideoQA Algorithm (KcGA) with an encoder-decoder pipeline, which enables out-of-domain answer generation through an adaptive external knowledge module and a multi-stream information control mechanism. We use ClipBERT to extract the video-question features, extract framewise object-level external knowledge from a commonsense knowledge base and compute the contextual-aware episode memory units via an attention based GRU to form the external knowledge features, and exploit multi-stream information control mechanism to fuse video-question and external knowledge features such that the semantic complementation and alignment are well achieved. We evaluate our model on two open-ended benchmark datasets to demonstrate that we can effectively and robustly generate high-quality answers without restrictions of training data.

        ----

        ## [914] POEM: Polarization of Embeddings for Domain-Invariant Representations

        **Authors**: *Sang-Yeong Jo, Sung Whan Yoon*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25984](https://doi.org/10.1609/aaai.v37i7.25984)

        **Abstract**:

        Handling out-of-distribution samples is a long-lasting challenge for deep visual models. In particular, domain generalization (DG) is one of the most relevant tasks that aims to train a model with a generalization capability on novel domains. Most existing DG approaches share the same philosophy to minimize the discrepancy between domains by finding the domain-invariant representations. On the contrary, our proposed method called POEM acquires a strong DG capability by learning domain-invariant and domain-specific representations and polarizing them. Specifically, POEM co-trains category-classifying and domain-classifying embeddings while regularizing them to be orthogonal via minimizing the cosine-similarity between their features, i.e., the polarization of embeddings. The clear separation of embeddings suppresses domain-specific features in the domain-invariant embeddings. The concept of POEM shows a unique direction to enhance the domain robustness of representations that brings considerable and consistent performance gains when combined with existing DG methods. Extensive simulation results in popular DG benchmarks with the PACS, VLCS, OfficeHome, TerraInc, and DomainNet datasets show that POEM indeed facilitates the category-classifying embedding to be more domain-invariant.

        ----

        ## [915] An Efficient Algorithm for Fair Multi-Agent Multi-Armed Bandit with Low Regret

        **Authors**: *Matthew Jones, Huy L. Nguyen, Thy Dinh Nguyen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25985](https://doi.org/10.1609/aaai.v37i7.25985)

        **Abstract**:

        Recently a multi-agent variant of the classical multi-armed bandit was proposed to tackle fairness issues in online learning. Inspired by a long line of work in social choice and economics, the goal is to optimize the Nash social welfare instead of the total utility. Unfortunately previous algorithms either are not efficient or achieve sub-optimal regret in terms of the number of rounds. We propose a new efficient algorithm with lower regret than even previous inefficient ones. We also complement our efficient algorithm with an inefficient approach with regret that matches the lower bound for one agent. The experimental findings confirm the effectiveness of our efficient algorithm compared to the previous approaches.

        ----

        ## [916] Towards More Robust Interpretation via Local Gradient Alignment

        **Authors**: *Sunghwan Joo, Seokhyeon Jeong, Juyeon Heo, Adrian Weller, Taesup Moon*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25986](https://doi.org/10.1609/aaai.v37i7.25986)

        **Abstract**:

        Neural network interpretation methods, particularly feature attribution methods, are known to be fragile with respect to adversarial input perturbations. 
To address this, several methods for enhancing the local smoothness of the gradient while training have been proposed for attaining robust feature attributions.
However, the lack of considering the normalization of the attributions, which is essential in their visualizations, has been an obstacle to understanding and improving the robustness of feature attribution methods. 
In this paper, we provide new insights by taking such normalization into account. First, we show that for every non-negative homogeneous neural network, a naive l2-robust criterion for gradients is not normalization invariant, which means that two functions with the same normalized gradient can have different values. 
Second, we formulate a normalization invariant cosine distance-based criterion and derive its upper bound, which gives insight for why simply minimizing the Hessian norm at the input, as has been done in previous work, is not sufficient for attaining robust feature attribution. Finally, we propose to combine both l2 and cosine distance-based criteria as regularization terms to leverage the advantages of both in aligning the local gradient. As a result, we experimentally show that models trained with our method produce much more robust interpretations on CIFAR-10 and ImageNet-100 without significantly hurting the accuracy, compared to the recent baselines. To the best of our knowledge, this is the first work to verify the robustness of interpretation on a larger-scale dataset beyond CIFAR-10, thanks to the computational efficiency of our method.

        ----

        ## [917] Identifying Selection Bias from Observational Data

        **Authors**: *David Kaltenpoth, Jilles Vreeken*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25987](https://doi.org/10.1609/aaai.v37i7.25987)

        **Abstract**:

        Access to a representative sample from the population is an assumption that underpins all of machine learning. Selection effects can cause observations to instead come from a subpopulation, by which our inferences may be subject to bias. 
It is therefore important to know whether or not a sample is affected by selection effects. We study under which conditions we can identify selection bias and give results for both parametric and non-parametric families of distributions. Based on these results we develop two practical methods to determine whether or not an observed sample comes from a distribution subject to selection bias. Through extensive evaluation on synthetic and real world data we verify that our methods beat the state of the art both in detecting as well as characterizing selection bias.

        ----

        ## [918] PIXEL: Physics-Informed Cell Representations for Fast and Accurate PDE Solvers

        **Authors**: *Namgyu Kang, Byeonghyeon Lee, Youngjoon Hong, Seok-Bae Yun, Eunbyung Park*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25988](https://doi.org/10.1609/aaai.v37i7.25988)

        **Abstract**:

        With the increases in computational power and advances in machine learning, data-driven learning-based methods have gained significant attention in solving PDEs. Physics-informed neural networks (PINNs) have recently emerged and succeeded in various forward and inverse PDE problems thanks to their excellent properties, such as flexibility, mesh-free solutions, and unsupervised training. However, their slower convergence speed and relatively inaccurate solutions often limit their broader applicability in many science and engineering domains. This paper proposes a new kind of data-driven PDEs solver, physics-informed cell representations (PIXEL), elegantly combining classical numerical methods and learning-based approaches. We adopt a grid structure from the numerical methods to improve accuracy and convergence speed and overcome the spectral bias presented in PINNs. Moreover, the proposed method enjoys the same benefits in PINNs, e.g., using the same optimization frameworks to solve both forward and inverse PDE problems and readily enforcing PDE constraints with modern automatic differentiation techniques. We provide experimental results on various challenging PDEs that the original PINNs have struggled with and show that PIXEL achieves fast convergence speed and high accuracy. Project page: https://namgyukang.github.io/PIXEL/

        ----

        ## [919] On the Sample Complexity of Vanilla Model-Based Offline Reinforcement Learning with Dependent Samples

        **Authors**: *Mustafa O. Karabag, Ufuk Topcu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25989](https://doi.org/10.1609/aaai.v37i7.25989)

        **Abstract**:

        Offline reinforcement learning (offline RL) considers problems where learning is performed using only previously collected samples and is helpful for the settings in which collecting new data is costly or risky. In model-based offline RL, the learner performs estimation (or optimization) using a model constructed according to the empirical transition frequencies. We analyze the sample complexity of vanilla model-based offline RL with dependent samples in the infinite-horizon discounted-reward setting. In our setting, the samples obey the dynamics of the Markov decision process and, consequently, may have interdependencies. Under no assumption of independent samples, we provide a high-probability, polynomial sample complexity bound for vanilla model-based off-policy evaluation that requires partial or uniform coverage. We extend this result to the off-policy optimization under uniform coverage. As a comparison to the model-based approach, we analyze the sample complexity of off-policy evaluation with vanilla importance sampling in the infinite-horizon setting. Finally, we provide an estimator that outperforms the sample-mean estimator for almost deterministic dynamics that are prevalent in reinforcement learning.

        ----

        ## [920] Communication-Efficient Collaborative Best Arm Identification

        **Authors**: *Nikolai Karpov, Qin Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25990](https://doi.org/10.1609/aaai.v37i7.25990)

        **Abstract**:

        We investigate top-m arm identification, a basic problem in bandit theory, in a multi-agent learning model in which agents collaborate to learn an objective function.  We are interested in designing collaborative learning algorithms that achieve maximum speedup (compared to single-agent learning algorithms) using minimum communication cost, as communication is frequently the bottleneck in multi-agent learning.  We give both algorithmic and impossibility results, and conduct a set of experiments to demonstrate the effectiveness of our algorithms.

        ----

        ## [921] Variable-Based Calibration for Machine Learning Classifiers

        **Authors**: *Markelle Kelly, Padhraic Smyth*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25991](https://doi.org/10.1609/aaai.v37i7.25991)

        **Abstract**:

        The deployment of machine learning classifiers in high-stakes domains requires well-calibrated confidence scores for model predictions. In this paper we introduce the notion of variable-based calibration to characterize calibration properties of a model with respect to a variable of interest, generalizing traditional score-based metrics such as expected calibration error (ECE). In particular, we find that models with near-perfect ECE can exhibit significant miscalibration as a function of features of the data. We demonstrate this phenomenon both theoretically and in practice on multiple well-known datasets, and show that it can persist after the application of existing calibration methods. To mitigate this issue, we propose strategies for detection, visualization, and quantification of variable-based calibration error. We then examine the limitations of current score-based calibration methods and explore potential modifications. Finally, we discuss the implications of these findings, emphasizing that an understanding of calibration beyond simple aggregate measures is crucial for endeavors such as fairness and model interpretability.

        ----

        ## [922] Design Amortization for Bayesian Optimal Experimental Design

        **Authors**: *Noble Kennamer, Steven Walton, Alexander Ihler*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25992](https://doi.org/10.1609/aaai.v37i7.25992)

        **Abstract**:

        Bayesian optimal experimental design is a sub-field of statistics focused on developing methods to make efficient use of experimental resources. Any potential design is evaluated in terms of a utility function, such as the (theoretically well-justified) expected information gain (EIG); unfortunately however, under most circumstances the EIG is intractable to evaluate. In this work we build off of successful variational approaches, which optimize a parameterized variational model with respect to bounds on the EIG. Past work focused on learning a new variational model from scratch for each new design considered. Here we present a novel neural architecture that allows experimenters to optimize a single variational model that can estimate the EIG for potentially infinitely many designs. To further improve computational efficiency, we also propose to train the variational model on a significantly cheaper-to-evaluate lower bound, and show empirically that the resulting model provides an excellent guide for more accurate, but expensive to evaluate  bounds on the EIG. We demonstrate the effectiveness of our technique on generalized linear models, a class of statistical models that is widely used in the analysis of controlled experiments. Experiments show that our method is able to greatly improve accuracy over existing approximation strategies, and achieve these results with far better sample efficiency.

        ----

        ## [923] On Error and Compression Rates for Prototype Rules

        **Authors**: *Omer Kerem, Roi Weiss*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25993](https://doi.org/10.1609/aaai.v37i7.25993)

        **Abstract**:

        We study the close interplay between error and compression in the non-parametric multiclass classification setting in terms of prototype learning rules. We focus in particular on a recently proposed compression-based learning rule termed OptiNet. Beyond its computational merits, this rule has been recently shown to be universally consistent in any metric instance space that admits a universally consistent rule---the first learning algorithm known to enjoy this property. However, its error and compression rates have been left open. Here we derive such rates in the case where instances reside in Euclidean space under commonly posed smoothness and tail conditions on the data distribution. We first show that OptiNet achieves non-trivial compression rates while enjoying near minimax-optimal error rates. We then proceed to study a novel general compression scheme for further compressing prototype rules that locally adapts to the noise level without sacrificing accuracy. Applying it to OptiNet, we show that under a geometric margin condition further gain in the compression rate is achieved. Experimental results comparing the performance of the various methods are presented.

        ----

        ## [924] CertiFair: A Framework for Certified Global Fairness of Neural Networks

        **Authors**: *Haitham Khedr, Yasser Shoukry*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25994](https://doi.org/10.1609/aaai.v37i7.25994)

        **Abstract**:

        We consider the problem of whether a Neural Network (NN) model satisfies global individual fairness. Individual Fairness (defined in (Dwork et al. 2012)) suggests that similar individuals with respect to a certain task are to be treated similarly by the decision model. In this work, we have two main objectives. The first is to construct a verifier which checks whether the fairness property holds for a given NN in a classification task or provides a counterexample if it is violated, i.e., the model is fair if all similar individuals are classified the same, and unfair if a pair of similar individuals are classified differently. To that end, we construct a sound and complete verifier that verifies global individual fairness properties of ReLU NN classifiers using distance-based similarity metrics. The second objective of this paper is to provide a method for training provably fair NN classifiers from unfair (biased) data. We propose a fairness loss that can be used during training to enforce fair outcomes for similar individuals. We then provide provable bounds on the fairness of the resulting NN. We run experiments on commonly used fairness datasets that are publicly available and we show that global individual fairness can be improved by 96 % without a significant drop in test accuracy.

        ----

        ## [925] Key Feature Replacement of In-Distribution Samples for Out-of-Distribution Detection

        **Authors**: *Jaeyoung Kim, Seo Taek Kong, Dongbin Na, Kyu-Hwan Jung*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25995](https://doi.org/10.1609/aaai.v37i7.25995)

        **Abstract**:

        Out-of-distribution (OOD) detection can be used in deep learning-based applications to reject outlier samples from being unreliably classified by deep neural networks. Learning to classify between OOD and in-distribution samples is difficult because data comprising the former is extremely diverse. It has been observed that an auxiliary OOD dataset is most effective in training a ``rejection'' network when its samples are semantically similar to in-distribution images. We first deduce that OOD images are perceived by a deep neural network to be semantically similar to in-distribution samples when they share a common background, as deep networks are observed to incorrectly classify such images with high confidence. We then propose a simple yet effective Key In-distribution feature Replacement BY inpainting (KIRBY) procedure that constructs a surrogate OOD dataset by replacing class-discriminative features of in-distribution samples with marginal background features. The procedure can be implemented using off-the-shelf vision algorithms, where each step within the algorithm is shown to make the surrogate data increasingly similar to in-distribution data. Design choices in each step are studied extensively, and an exhaustive comparison with state-of-the-art algorithms demonstrates KIRBY's competitiveness on various benchmarks.

        ----

        ## [926] FLAME: Free-Form Language-Based Motion Synthesis & Editing

        **Authors**: *Jihoon Kim, Jiseob Kim, Sungjoon Choi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25996](https://doi.org/10.1609/aaai.v37i7.25996)

        **Abstract**:

        Text-based motion generation models are drawing a surge of interest for their potential for automating the motion-making process in the game, animation, or robot industries. In this paper, we propose a diffusion-based motion synthesis and editing model named FLAME. Inspired by the recent successes in diffusion models, we integrate diffusion-based generative models into the motion domain. FLAME can generate high-fidelity motions well aligned with the given text. Also, it can edit the parts of the motion, both frame-wise and joint-wise, without any fine-tuning. FLAME involves a new transformer-based architecture we devise to better handle motion data, which is found to be crucial to manage variable-length motions and well attend to free-form text. In experiments, we show that FLAME achieves state-of-the-art generation performances on three text-motion datasets: HumanML3D, BABEL, and KIT. We also demonstrate that FLAME’s editing capability can be extended to other tasks such as motion prediction or motion in-betweening, which have been previously covered by dedicated models.

        ----

        ## [927] Inverse-Reference Priors for Fisher Regularization of Bayesian Neural Networks

        **Authors**: *Keunseo Kim, Eun-Yeol Ma, Jeongman Choi, Heeyoung Kim*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25997](https://doi.org/10.1609/aaai.v37i7.25997)

        **Abstract**:

        Recent studies have shown that the generalization ability of deep neural networks (DNNs) is closely related to the Fisher information matrix (FIM) calculated during the early training phase. Several methods have been proposed to regularize the FIM for increased generalization of DNNs. However, they cannot be used directly for Bayesian neural networks (BNNs) because the variable parameters of BNNs make it difficult to calculate the FIM. To address this problem, we achieve regularization of the FIM of BNNs by specifying a new suitable prior distribution called the inverse-reference (IR) prior. To regularize the FIM, the IR prior is derived as the inverse of the reference prior that imposes minimal prior knowledge on the parameters and maximizes the trace of the FIM. We demonstrate that the IR prior can enhance the generalization ability of BNNs for large-scale data over previously used priors while providing adequate uncertainty quantifications using various benchmark image datasets and BNN structures.

        ----

        ## [928] Deep Visual Forced Alignment: Learning to Align Transcription with Talking Face Video

        **Authors**: *Minsu Kim, Chae Won Kim, Yong Man Ro*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25998](https://doi.org/10.1609/aaai.v37i7.25998)

        **Abstract**:

        Forced alignment refers to a technology that time-aligns a given transcription with a corresponding speech. However, as the forced alignment technologies have developed using speech audio, they might fail in alignment when the input speech audio is noise-corrupted or is not accessible. We focus on that there is another component that the speech can be inferred from, the speech video (i.e., talking face video). Since the drawbacks of audio-based forced alignment can be complemented using the visual information when the audio signal is under poor condition, we try to develop a novel video-based forced alignment method. However, different from audio forced alignment, it is challenging to develop a reliable visual forced alignment technology for the following two reasons: 1) Visual Speech Recognition (VSR) has a much lower performance compared to audio-based Automatic Speech Recognition (ASR), and 2) the translation from text to video is not reliable, so the method typically used for building audio forced alignment cannot be utilized in developing visual forced alignment. In order to alleviate these challenges, in this paper, we propose a new method that is appropriate for visual forced alignment, namely Deep Visual Forced Alignment (DVFA). The proposed DVFA can align the input transcription (i.e., sentence) with the talking face video without accessing the speech audio. Moreover, by augmenting the alignment task with anomaly case detection, DVFA can detect mismatches between the input transcription and the input video while performing the alignment. Therefore, we can robustly align the text with the talking face video even if there exist error words in the text. Through extensive experiments, we show the effectiveness of the proposed DVFA not only in the alignment task but also in interpreting the outputs of VSR models.

        ----

        ## [929] Better Generalized Few-Shot Learning Even without Base Data

        **Authors**: *Seong-Woong Kim, Dong-Wan Choi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.25999](https://doi.org/10.1609/aaai.v37i7.25999)

        **Abstract**:

        This paper introduces and studies zero-base generalized few-shot learning (zero-base GFSL), which is an extreme yet practical version of few-shot learning problem. Motivated by the cases where base data is not available due to privacy or ethical issues, the goal of zero-base GFSL is to newly incorporate the knowledge of few samples of novel classes into a pretrained model without any samples of base classes. According to our analysis, we discover the fact that both mean and variance of the weight distribution of novel classes are not properly established, compared to those of base classes. The existing GFSL methods attempt to make the weight norms balanced, which we find help only the variance part, but discard the importance of mean of weights particularly for novel classes, leading to the limited performance in the GFSL problem even with base data. In this paper, we overcome this limitation by proposing a simple yet effective normalization method that can effectively control both mean and variance of the weight distribution of novel classes without using any base samples and thereby achieve a satisfactory performance on both novel and base classes. Our experimental results somewhat surprisingly show that the proposed zero-base GFSL method that does not utilize any base samples even outperforms the existing GFSL methods that make the best use of base data.  Our implementation is available at: https://github.com/bigdata-inha/Zero-Base-GFSL.

        ----

        ## [930] Learning Topology-Specific Experts for Molecular Property Prediction

        **Authors**: *Suyeon Kim, Dongha Lee, SeongKu Kang, Seonghyeon Lee, Hwanjo Yu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26000](https://doi.org/10.1609/aaai.v37i7.26000)

        **Abstract**:

        Recently, graph neural networks (GNNs) have been successfully applied to predicting molecular properties, which is one of the most classical cheminformatics tasks with various applications. Despite their effectiveness, we empirically observe that training a single GNN model for diverse molecules with distinct structural patterns limits its prediction performance. In this paper, motivated by this observation, we propose TopExpert to leverage topology-specific prediction models (referred to as experts), each of which is responsible for each molecular group sharing similar topological semantics. That is, each expert learns topology-specific discriminative features while being trained with its corresponding topological group. To tackle the key challenge of grouping molecules by their topological patterns, we introduce a clustering-based gating module that assigns an input molecule into one of the clusters and further optimizes the gating module with two different types of self-supervision: topological semantics induced by GNNs and molecular scaffolds, respectively. Extensive experiments demonstrate that TopExpert has boosted the performance for molecular property prediction and also achieved better generalization for new molecules with unseen scaffolds than baselines. The code is available at https://github.com/kimsu55/ToxExpert.

        ----

        ## [931] Double Doubly Robust Thompson Sampling for Generalized Linear Contextual Bandits

        **Authors**: *Wonyoung Kim, Kyungbok Lee, Myunghee Cho Paik*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26001](https://doi.org/10.1609/aaai.v37i7.26001)

        **Abstract**:

        We propose a novel algorithm for generalized linear contextual bandits (GLBs) with a regret bound sublinear to the time horizon, the minimum eigenvalue of the covariance of contexts and a lower bound of the variance of rewards.
In several identified cases, our result is the first regret bound for generalized linear bandits (GLBs) achieving the regret bound sublinear to the dimension of contexts without discarding the observed rewards.
Previous approaches achieve the regret bound sublinear to the dimension of contexts by discarding the observed rewards, whereas our algorithm achieves the bound incorporating contexts from all arms in our double doubly robust (DDR) estimator.
The DDR estimator is a subclass of doubly robust estimator but with a tighter error bound.
We also provide a logarithmic cumulative regret bound under a probabilistic margin condition.
This is the first regret bound under the margin condition for linear models or GLMs when contexts are different for all arms but coefficients are common.
We conduct empirical studies using synthetic data and real examples, demonstrating the effectiveness of our algorithm.

        ----

        ## [932] Exploring Temporal Information Dynamics in Spiking Neural Networks

        **Authors**: *Youngeun Kim, Yuhang Li, Hyoungseob Park, Yeshwanth Venkatesha, Anna Hambitzer, Priyadarshini Panda*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26002](https://doi.org/10.1609/aaai.v37i7.26002)

        **Abstract**:

        Most existing Spiking Neural Network (SNN) works state that SNNs may utilize temporal information dynamics of spikes. However, an explicit analysis of temporal information dynamics is still missing. In this paper, we ask several important questions for providing a fundamental understanding of SNNs: What are temporal information dynamics inside SNNs? How can we measure the temporal information dynamics? How do the temporal information dynamics affect the overall learning performance? To answer these questions, we estimate the Fisher Information of the weights to measure the distribution of temporal information during training in an empirical manner. Surprisingly, as training goes on, Fisher information starts to concentrate in the early timesteps. After training, we observe that information becomes highly concentrated in earlier few timesteps, a phenomenon we refer to as temporal information concentration. We observe that the temporal information concentration phenomenon is a common learning feature of SNNs by conducting extensive experiments on various configurations such as architecture, dataset, optimization strategy, time constant, and timesteps. Furthermore, to reveal how temporal information concentration affects the performance of SNNs, we design a loss function to change the trend of temporal information. We find that temporal information concentration is crucial to building a robust SNN but has little effect on classification accuracy. Finally, we propose an efficient iterative pruning method based on our observation on temporal information concentration. 
Code is available at https://github.com/Intelligent-Computing-Lab-Yale/Exploring-Temporal-Information-Dynamics-in-Spiking-Neural-Networks.

        ----

        ## [933] FastAMI - a Monte Carlo Approach to the Adjustment for Chance in Clustering Comparison Metrics

        **Authors**: *Kai Klede, Leo Schwinn, Dario Zanca, Björn M. Eskofier*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26003](https://doi.org/10.1609/aaai.v37i7.26003)

        **Abstract**:

        Clustering is at the very core of machine learning, and its applications proliferate with the increasing availability of data. However, as datasets grow, comparing clusterings with an adjustment for chance becomes computationally difficult, preventing unbiased ground-truth comparisons and solution selection. We propose FastAMI, a Monte Carlo-based method to efficiently approximate the Adjusted Mutual Information (AMI) and extend it to the Standardized Mutual Information (SMI). The approach is compared with the exact calculation and a recently developed variant of the AMI based on pairwise permutations, using both synthetic and real data. In contrast to the exact calculation our method is fast enough to enable these adjusted information-theoretic comparisons for large datasets while maintaining considerably more accurate results than the pairwise approach.

        ----

        ## [934] A Gift from Label Smoothing: Robust Training with Adaptive Label Smoothing via Auxiliary Classifier under Label Noise

        **Authors**: *Jongwoo Ko, Bongsoo Yi, Se-Young Yun*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26004](https://doi.org/10.1609/aaai.v37i7.26004)

        **Abstract**:

        As deep neural networks can easily overfit noisy labels, robust training in the presence of noisy labels is becoming an important challenge in modern deep learning. While existing methods address this problem in various directions, they still produce unpredictable sub-optimal results since they rely on the posterior information estimated by the feature extractor corrupted by noisy labels. Lipschitz regularization successfully alleviates this problem by training a robust feature extractor, but it requires longer training time and expensive computations. Motivated by this, we propose a simple yet effective method, called ALASCA, which efficiently provides a robust feature extractor under label noise. ALASCA integrates two key ingredients: (1) adaptive label smoothing based on our theoretical analysis that label smoothing implicitly induces Lipschitz regularization, and (2) auxiliary classifiers that enable practical application of intermediate Lipschitz regularization with negligible computations. We conduct wide-ranging experiments for ALASCA and combine our proposed method with previous noise-robust methods on several synthetic and real-world datasets. Experimental results show that our framework consistently improves the robustness of feature extractors and the performance of existing baselines with efficiency.

        ----

        ## [935] Grouping Matrix Based Graph Pooling with Adaptive Number of Clusters

        **Authors**: *Sung Moon Ko, Sungjun Cho, Dae-Woong Jeong, Sehui Han, Moontae Lee, Honglak Lee*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26005](https://doi.org/10.1609/aaai.v37i7.26005)

        **Abstract**:

        Graph pooling is a crucial operation for encoding hierarchical structures within graphs. Most existing graph pooling approaches formulate the problem as a node clustering task which effectively captures the graph topology. Conventional methods ask users to specify an appropriate number of clusters as a hyperparameter, then assuming that all input graphs share the same number of clusters. In inductive settings where the number of clusters could vary, however, the model should be able to represent this variation in its pooling layers in order to learn suitable clusters. Thus we propose GMPool, a novel differentiable graph pooling architecture that automatically determines the appropriate number of clusters based on the input data. The main intuition involves a grouping matrix defined as a quadratic form of the pooling operator, which induces use of binary classification probabilities of pairwise combinations of nodes. GMPool obtains the pooling operator by first computing the grouping matrix, then decomposing it. Extensive evaluations on molecular property prediction tasks demonstrate that our method outperforms conventional methods.

        ----

        ## [936] The Influence of Dimensions on the Complexity of Computing Decision Trees

        **Authors**: *Stephen G. Kobourov, Maarten Löffler, Fabrizio Montecchiani, Marcin Pilipczuk, Ignaz Rutter, Raimund Seidel, Manuel Sorge, Jules Wulms*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26006](https://doi.org/10.1609/aaai.v37i7.26006)

        **Abstract**:

        A decision tree recursively splits a feature space \mathbb{R}^d and then assigns class labels based on the resulting partition. Decision trees have been part of the basic machine-learning toolkit for decades. A large body of work considers heuristic algorithms that compute a decision tree from training data, usually aiming to minimize in particular the size of the resulting tree. In contrast, little is known about the complexity of the underlying computational problem of computing a minimum-size tree for the given training data. We study this problem with respect to the number d of dimensions of the feature space \mathbb{R}^d, which contains n training examples. We show that it can be solved in O(n^(2d + 1)) time, but under reasonable complexity-theoretic assumptions it is not possible to achieve f(d) * n^o(d / log d) running time. The problem is solvable in (dR)^O(dR) * n^(1+o(1)) time, if there are exactly two classes and R is an upper bound on the number of tree leaves labeled with the first class.

        ----

        ## [937] Learning Similarity Metrics for Volumetric Simulations with Multiscale CNNs

        **Authors**: *Georg Kohl, Li-Wei Chen, Nils Thuerey*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26007](https://doi.org/10.1609/aaai.v37i7.26007)

        **Abstract**:

        Simulations that produce three-dimensional data are ubiquitous in science, ranging from fluid flows to plasma physics. We propose a similarity model based on entropy, which allows for the creation of physically meaningful ground truth distances for the similarity assessment of scalar and vectorial data, produced from transport and motion-based simulations. Utilizing two data acquisition methods derived from this model, we create collections of fields from numerical PDE solvers and existing simulation data repositories. Furthermore, a multiscale CNN architecture that computes a volumetric similarity metric (VolSiM) is proposed. To the best of our knowledge this is the first learning method inherently designed to address the challenges arising for the similarity assessment of high-dimensional simulation data. Additionally, the tradeoff between a large batch size and an accurate correlation computation for correlation-based loss functions is investigated, and the metric's invariance with respect to rotation and scale operations is analyzed. Finally, the robustness and generalization of VolSiM is evaluated on a large range of test data, as well as a particularly challenging turbulence case study, that is close to potential real-world applications.

        ----

        ## [938] Peeling the Onion: Hierarchical Reduction of Data Redundancy for Efficient Vision Transformer Training

        **Authors**: *Zhenglun Kong, Haoyu Ma, Geng Yuan, Mengshu Sun, Yanyue Xie, Peiyan Dong, Xin Meng, Xuan Shen, Hao Tang, Minghai Qin, Tianlong Chen, Xiaolong Ma, Xiaohui Xie, Zhangyang Wang, Yanzhi Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26008](https://doi.org/10.1609/aaai.v37i7.26008)

        **Abstract**:

        Vision transformers (ViTs) have recently obtained success in many applications, but their intensive computation and heavy memory usage at both training and inference time limit their generalization. Previous compression algorithms usually start from the pre-trained dense models and only focus on efficient inference, while time-consuming training is still unavoidable.  In contrast, this paper points out that the million-scale training data is redundant, which is the fundamental reason for the tedious training. To address the issue, this paper aims to introduce sparsity into data and proposes an end-to-end efficient training framework from three sparse perspectives, dubbed Tri-Level E-ViT. Specifically,  we leverage a hierarchical data redundancy reduction scheme, by exploring the sparsity under three levels: number of training examples in the dataset, number of patches (tokens) in each example, and number of connections between tokens that lie in attention weights. With extensive experiments, we demonstrate that our proposed technique can noticeably accelerate training for various ViT architectures while maintaining accuracy.   Remarkably, under certain ratios, we are able to improve the ViT accuracy rather than compromising it. For example, we can achieve 15.2% speedup with 72.6% (+0.4) Top-1 accuracy on Deit-T, and 15.7% speedup with 79.9% (+0.1) Top-1 accuracy on Deit-S. This proves the existence of data redundancy in ViT. Our code is released at https://github.com/ZLKong/Tri-Level-ViT

        ----

        ## [939] Adversarial Robust Deep Reinforcement Learning Requires Redefining Robustness

        **Authors**: *Ezgi Korkmaz*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26009](https://doi.org/10.1609/aaai.v37i7.26009)

        **Abstract**:

        Learning from raw high dimensional data via interaction with a given environment has been effectively achieved through the utilization of deep neural networks. Yet the observed degradation in policy performance caused by imperceptible worst-case policy dependent translations along high sensitivity directions (i.e. adversarial perturbations) raises concerns on the robustness of deep reinforcement learning policies. In our paper, we show that these high sensitivity directions do not lie only along particular worst-case directions, but rather are more abundant in the deep neural policy landscape and can be found via more natural means in a black-box setting. Furthermore, we show that vanilla training techniques intriguingly result in learning more robust policies compared to the policies learnt via the state-of-the-art adversarial training techniques. We believe our work lays out intriguing properties of the deep reinforcement learning policy manifold and our results can help to build robust and generalizable deep reinforcement learning policies.

        ----

        ## [940] Almost Cost-Free Communication in Federated Best Arm Identification

        **Authors**: *Srinivas Reddy Kota, P. N. Karthik, Vincent Y. F. Tan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26010](https://doi.org/10.1609/aaai.v37i7.26010)

        **Abstract**:

        We study the problem of best arm identification in a federated learning multi-armed bandit setup with a central server and multiple clients. Each client is associated with a multi-armed bandit in which each arm yields i.i.d. rewards following a Gaussian distribution with an unknown mean and known variance. The set of arms is assumed to be the same at all the clients. We define two notions of best arm local and global. The local best arm at a client is the arm with the largest mean among the arms local to the client, whereas the global best arm is the arm with the largest average mean across all the clients. We assume that each client can only observe the rewards from its local arms and thereby estimate its local best arm. The clients communicate with a central server on uplinks that entail a cost of C>=0 units per usage per uplink. The global best arm is estimated at the server. The goal is to identify the local best arms and the global best arm with minimal total cost, defined as the sum of the total number of arm selections at all the clients and the total communication cost, subject to an upper bound on the error probability. We propose a novel algorithm FedElim that is based on successive elimination and communicates only in exponential time steps and obtain a high probability instance-dependent upper bound on its total cost. The key takeaway from our paper is that for any C>=0 and error probabilities sufficiently small, the total number of arm selections (resp. the total cost) under FedElim is at most 2 (resp. 3) times the maximum total number of arm selections under its variant that communicates in every time step. Additionally, we show that the latter is optimal in expectation up to a constant factor, thereby demonstrating that communication is almost cost-free in FedElim. We numerically validate the efficacy of FedElim on two synthetic datasets and the MovieLens dataset.

        ----

        ## [941] UEQMS: UMAP Embedded Quick Mean Shift Algorithm for High Dimensional Clustering

        **Authors**: *Abhishek Kumar, Swagatam Das, Rammohan Mallipeddi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26011](https://doi.org/10.1609/aaai.v37i7.26011)

        **Abstract**:

        The mean shift algorithm is a simple yet very effective clustering method widely used for image and video segmentation as well as other exploratory data analysis applications. Recently, a new algorithm called MeanShift++ (MS++) for low-dimensional clustering was proposed with a speedup of 4000 times over the vanilla mean shift. In this work, starting with a first-of-its-kind theoretical analysis of MS++, we extend its reach to high-dimensional data clustering by integrating the Uniform Manifold Approximation and Projection (UMAP) based dimensionality reduction in the same framework. Analytically, we show that MS++ can indeed converge to a non-critical point. Subsequently, we suggest modifications to MS++ to improve its convergence characteristics. In addition, we propose a way to further speed up MS++ by avoiding the execution of the MS++ iterations for every data point. By incorporating UMAP with modified MS++, we design a faster algorithm, named UMAP embedded quick mean shift (UEQMS), for partitioning data with a relatively large number of recorded features. Through extensive experiments,  we showcase the efficacy of UEQMS over other state-of-the-art algorithms in terms of accuracy and runtime.

        ----

        ## [942] The Effect of Diversity in Meta-Learning

        **Authors**: *Ramnath Kumar, Tristan Deleu, Yoshua Bengio*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26012](https://doi.org/10.1609/aaai.v37i7.26012)

        **Abstract**:

        Recent studies show that task distribution plays a vital role in the meta-learner's performance. Conventional wisdom is that task diversity should improve the performance of meta-learning. In this work, we find evidence to the contrary; (i) our experiments draw into question the efficacy of our learned models: similar manifolds can be learned with a subset of the data (lower task diversity). This finding questions the advantage of providing more data to the model, and (ii) adding diversity to the task distribution (higher task diversity) sometimes hinders the model and does not lead to a significant improvement in performance as previously believed. To strengthen our findings, we provide both empirical and theoretical evidence.

        ----

        ## [943] Gradient Estimation for Binary Latent Variables via Gradient Variance Clipping

        **Authors**: *Russell Z. Kunes, Mingzhang Yin, Max Land, Doron Haviv, Dana Pe'er, Simon Tavaré*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26013](https://doi.org/10.1609/aaai.v37i7.26013)

        **Abstract**:

        Gradient estimation is often necessary for fitting generative models with discrete latent variables, in contexts such as reinforcement learning and variational autoencoder (VAE) training. The DisARM estimator achieves state of the art gradient variance for Bernoulli latent variable models in many contexts. However, DisARM and other estimators have potentially exploding variance near the boundary of the parameter space, where solutions tend to lie. To ameliorate this issue, we propose a new gradient estimator bitflip-1  that is lower variance at the boundaries of the parameter space. As bitflip-1 has complementary properties to existing estimators, we introduce an aggregated estimator, unbiased gradient variance clipping (UGC) that uses either a bitflip-1 or a DisARM gradient update for each coordinate.  We theoretically prove that UGC has uniformly lower variance than DisARM.
Empirically, we observe that UGC achieves the optimal value of the optimization objectives  in toy experiments, discrete VAE training, and in a best subset selection problem.

        ----

        ## [944] LoNe Sampler: Graph Node Embeddings by Coordinated Local Neighborhood Sampling

        **Authors**: *Konstantin Kutzkov*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26014](https://doi.org/10.1609/aaai.v37i7.26014)

        **Abstract**:

        Local graph neighborhood sampling is a fundamental computational problem that is at the heart of algorithms for node representation learning. 
Several works have presented algorithms for learning discrete node embeddings where graph nodes are represented by discrete features such as attributes of neighborhood nodes. Discrete embeddings offer several advantages compared to continuous word2vec-like node embeddings: ease of computation, scalability, and interpretability. We present LoNe Sampler, a suite of algorithms for generating discrete node embeddings by Local Neighborhood Sampling, and address two shortcomings of previous work. First, our algorithms have rigorously understood theoretical properties. Second, we show how to generate approximate explicit vector maps that avoid the expensive computation of a Gram matrix for the training of a kernel model. Experiments on benchmark datasets confirm the theoretical findings and demonstrate the advantages of the proposed methods.

        ----

        ## [945] WLD-Reg: A Data-Dependent Within-Layer Diversity Regularizer

        **Authors**: *Firas Laakom, Jenni Raitoharju, Alexandros Iosifidis, Moncef Gabbouj*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26015](https://doi.org/10.1609/aaai.v37i7.26015)

        **Abstract**:

        Neural networks are composed of multiple layers arranged in a hierarchical structure jointly trained with a gradient-based optimization, where the errors are back-propagated from the last layer back to the first one. At each optimization step, neurons at a given layer receive feedback from neurons belonging to higher layers of the hierarchy. In this paper, we propose to complement this traditional 'between-layer' feedback with additional 'within-layer' feedback to encourage the diversity of the activations within the same layer. To this end, we measure the pairwise similarity between the outputs of the neurons and use it to model the layer's overall diversity. We present an extensive empirical study confirming that the proposed approach enhances the performance of several state-of-the-art neural network models in multiple tasks. The code is publically available at https://github.com/firasl/AAAI-23-WLD-Reg.

        ----

        ## [946] SpatialFormer: Semantic and Target Aware Attentions for Few-Shot Learning

        **Authors**: *Jinxiang Lai, Siqian Yang, Wenlong Wu, Tao Wu, Guannan Jiang, Xi Wang, Jun Liu, Bin-Bin Gao, Wei Zhang, Yuan Xie, Chengjie Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26016](https://doi.org/10.1609/aaai.v37i7.26016)

        **Abstract**:

        Recent Few-Shot Learning (FSL) methods put emphasis on generating a discriminative embedding features to precisely measure the similarity between support and query sets. Current CNN-based cross-attention approaches generate discriminative representations via enhancing the mutually semantic similar regions of support and query pairs. However, it suffers from two problems: CNN structure produces inaccurate attention map based on local features, and mutually similar backgrounds cause distraction. To alleviate these problems, we design a novel SpatialFormer structure to generate more accurate attention regions based on global features. Different from the traditional Transformer modeling intrinsic instance-level similarity which causes accuracy degradation in FSL, our SpatialFormer explores the semantic-level similarity between pair inputs to boost the performance. Then we derive two specific attention modules, named SpatialFormer Semantic Attention (SFSA) and SpatialFormer Target Attention (SFTA), to enhance the target object regions while reduce the background distraction. Particularly, SFSA highlights the regions with same semantic information between pair features, and SFTA finds potential foreground object regions of novel feature that are similar to base categories. Extensive experiments show that our methods are effective and achieve new state-of-the-art results on few-shot classification benchmarks.

        ----

        ## [947] A Data Source for Reasoning Embodied Agents

        **Authors**: *Jack Lanchantin, Sainbayar Sukhbaatar, Gabriel Synnaeve, Yuxuan Sun, Kavya Srinet, Arthur Szlam*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26017](https://doi.org/10.1609/aaai.v37i7.26017)

        **Abstract**:

        Recent progress in using machine learning models for reasoning tasks has been driven by novel model architectures, large-scale pre-training protocols, and dedicated reasoning datasets for fine-tuning.  In this work, to further pursue these advances, we introduce a new data generator for machine reasoning that integrates with an embodied agent.  The generated data consists of templated text queries and answers, matched with world-states encoded into a database.  The world-states are a result of both world dynamics and the actions of the agent.  We show the results of several baseline models on instantiations of train sets.  These include pre-trained language models fine-tuned on a text-formatted representation of the database, and graph-structured Transformers operating on a knowledge-graph representation of the database.  We find that these models can answer some questions about the world-state, but struggle with others. These results hint at new research directions in designing neural reasoning models and database representations. Code to generate the data and train the models will be released at github.com/facebookresearch/neuralmemory

        ----

        ## [948] Generalization Bounds for Inductive Matrix Completion in Low-Noise Settings

        **Authors**: *Antoine Ledent, Rodrigo Alves, Yunwen Lei, Yann Guermeur, Marius Kloft*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26018](https://doi.org/10.1609/aaai.v37i7.26018)

        **Abstract**:

        (1) they scale like the standard deviation of the noise and in particular approach zero in the exact recovery case; (2) even in the presence of noise, they converge to zero when the sample size approaches infinity; and (3) for a fixed dimension of the side information, they only have a logarithmic dependence on the size of the matrix. Differently from many works in approximate recovery, we present results both for bounded Lipschitz losses and for the absolute loss, with the latter relying on Talagrand-type inequalities.  The proofs create a bridge between two approaches to the theoretical analysis of matrix completion, since they consist in a combination of techniques from both the exact recovery literature and the approximate recovery literature.

        ----

        ## [949] I'm Me, We're Us, and I'm Us: Tri-directional Contrastive Learning on Hypergraphs

        **Authors**: *Dongjin Lee, Kijung Shin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26019](https://doi.org/10.1609/aaai.v37i7.26019)

        **Abstract**:

        Although machine learning on hypergraphs has attracted considerable attention, most of the works have focused on (semi-)supervised learning, which may cause heavy labeling costs and poor generalization. Recently, contrastive learning has emerged as a successful unsupervised representation learning method. Despite the prosperous development of contrastive learning in other domains, contrastive learning on hypergraphs remains little explored. In this paper, we propose TriCL (Tri-directional Contrastive Learning), a general framework for contrastive learning on hypergraphs. Its main idea is tri-directional contrast, and specifically, it aims to maximize in two augmented views the agreement (a) between the same node, (b) between the same group of nodes, and (c) between each group and its members. Together with simple but surprisingly effective data augmentation and negative sampling schemes, these three forms of contrast enable TriCL to capture both node- and group-level structural information in node embeddings. Our extensive experiments using 14 baseline approaches, 10 datasets, and two tasks demonstrate the effectiveness of TriCL, and most noticeably, TriCL almost consistently outperforms not just unsupervised competitors but also (semi-)supervised competitors mostly by significant margins for node classification. The code and datasets are available at https://github.com/wooner49/TriCL.

        ----

        ## [950] Bespoke: A Block-Level Neural Network Optimization Framework for Low-Cost Deployment

        **Authors**: *Jong-Ryul Lee, Yong-Hyuk Moon*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26020](https://doi.org/10.1609/aaai.v37i7.26020)

        **Abstract**:

        As deep learning models become popular, there is a lot of need for deploying them to diverse device environments. Because it is costly to develop and optimize a neural network for every single environment, there is a line of research to search neural networks for multiple target environments efficiently. However, existing works for such a situation still suffer from requiring many GPUs and expensive costs. Motivated by this, we propose a novel neural network optimization framework named Bespoke for low-cost deployment. Our framework searches for a lightweight model by replacing parts of an original model with randomly selected alternatives, each of which comes from a pretrained neural network or the original model. In the practical sense, Bespoke has two significant merits. One is that it requires near zero cost for designing the search space of neural networks. The other merit is that it exploits the sub-networks of public pretrained neural networks, so the total cost is minimal compared to the existing works. We conduct experiments exploring Bespoke's the merits, and the results show that it finds efficient models for multiple targets with meager cost.

        ----

        ## [951] Time-Aware Random Walk Diffusion to Improve Dynamic Graph Learning

        **Authors**: *Jong-whi Lee, Jinhong Jung*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26021](https://doi.org/10.1609/aaai.v37i7.26021)

        **Abstract**:

        How can we augment a dynamic graph for improving the performance of dynamic graph neural networks? Graph augmentation has been widely utilized to boost the learning performance of GNN-based models. However, most existing approaches only enhance spatial structure within an input static graph by transforming the graph, and do not consider dynamics caused by time such as temporal locality, i.e., recent edges are more influential than earlier ones, which remains challenging for dynamic graph augmentation.
In this work, we propose TiaRa (Time-aware Random Walk Diffusion), a novel diffusion-based method for augmenting a dynamic graph represented as a discrete-time sequence of graph snapshots. For this purpose, we first design a time-aware random walk proximity so that a surfer can walk along the time dimension as well as edges, resulting in spatially and temporally localized scores. We then derive our diffusion matrices based on the time-aware random walk, and show they become enhanced adjacency matrices that both spatial and temporal localities are augmented. Throughout extensive experiments, we demonstrate that TiaRa effectively augments a given dynamic graph, and leads to significant improvements in dynamic GNN models for various graph datasets and tasks.

        ----

        ## [952] Demystifying Randomly Initialized Networks for Evaluating Generative Models

        **Authors**: *Junghyuk Lee, Jun-Hyuk Kim, Jong-Seok Lee*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26022](https://doi.org/10.1609/aaai.v37i7.26022)

        **Abstract**:

        Evaluation of generative models is mostly based on the comparison between the estimated distribution and the ground truth distribution in a certain feature space. To embed samples into informative features, previous works often use convolutional neural networks optimized for classification, which is criticized by recent studies. Therefore, various feature spaces have been explored to discover alternatives. Among them, a surprising approach is to use a randomly initialized neural network for feature embedding. However, the fundamental basis to employ the random features has not been sufficiently justified. In this paper, we rigorously investigate the feature space of models with random weights in comparison to that of trained models. Furthermore, we provide an empirical evidence to choose networks for random features to obtain consistent and reliable results. Our results indicate that the features from random networks can evaluate generative models well similarly to those from trained networks, and furthermore, the two types of features can be used together in a complementary way.

        ----

        ## [953] Layer-Wise Adaptive Model Aggregation for Scalable Federated Learning

        **Authors**: *Sunwoo Lee, Tuo Zhang, Amir Salman Avestimehr*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26023](https://doi.org/10.1609/aaai.v37i7.26023)

        **Abstract**:

        In Federated Learning (FL), a common approach for aggregating local solutions across clients is periodic full model averaging. It is, however, known that different layers of neural networks can have a different degree of model discrepancy across the clients. The conventional full aggregation scheme does not consider such a difference and synchronizes the whole model parameters at once, resulting in inefficient network bandwidth consumption. Aggregating the parameters that are similar across the clients does not make meaningful training progress while increasing the communication cost. We propose FedLAMA, a layer-wise adaptive model aggregation scheme for scalable FL. FedLAMA adjusts the aggregation interval in a layer-wise manner, jointly considering the model discrepancy and the communication cost. This fine-grained aggregation strategy enables to reduce the communication cost without significantly harming the model accuracy. Our extensive empirical study shows that, as the aggregation interval increases, FedLAMA shows a remarkably smaller accuracy drop than the periodic full aggregation, while achieving comparable communication efficiency.

        ----

        ## [954] Goal-Conditioned Q-learning as Knowledge Distillation

        **Authors**: *Alexander Levine, Soheil Feizi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26024](https://doi.org/10.1609/aaai.v37i7.26024)

        **Abstract**:

        Many applications of reinforcement learning can be formalized as goal-conditioned environments, where, in each episode, there is a "goal" that affects the rewards obtained during that episode but does not affect the dynamics. Various techniques have been proposed to improve performance in goal-conditioned environments, such as automatic curriculum generation and goal relabeling. In this work, we explore a connection between off-policy reinforcement learning in goal-conditioned settings and knowledge distillation. In particular: the current Q-value function and the target Q-value estimate are both functions of the goal, and we would like to train the Q-value function to match its target for all goals. We therefore apply Gradient-Based Attention Transfer (Zagoruyko and Komodakis 2017), a knowledge distillation technique, to the Q-function update. We empirically show that this can improve the performance of goal-conditioned off-policy reinforcement learning when the space of goals is high-dimensional. We also show that this technique can be adapted to allow for efficient learning in the case of multiple simultaneous sparse goals, where the agent can attain a reward by achieving any one of a large set of objectives, all specified at test time. Finally, to provide theoretical support, we give examples of classes of environments where (under some assumptions) standard off-policy algorithms such as DDPG require at least O(d^2) replay buffer transitions to learn an optimal policy, while our proposed technique requires only O(d) transitions, where d is the dimensionality of the goal and state space. Code and appendix are available at https://github.com/alevine0/ReenGAGE.

        ----

        ## [955] Optimism in Face of a Context: Regret Guarantees for Stochastic Contextual MDP

        **Authors**: *Orin Levy, Yishay Mansour*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26025](https://doi.org/10.1609/aaai.v37i7.26025)

        **Abstract**:

        We present regret minimization algorithms for stochastic contextual MDPs under minimum reachability assumption, using an access to an offline least square regression oracle.
We analyze three different settings: where the dynamics is known, where the dynamics is unknown but independent of the context and the most challenging setting where the dynamics is unknown and context-dependent. For the latter, our algorithm obtains regret bound (up to poly-logarithmic factors) of order  (H+1/pₘᵢₙ)H|S|³ᐟ²(|A|Tlog(max{|?|,|?|} /?))¹ᐟ²  with probability 1−?, where ? and ? are finite and realizable function classes used to approximate the dynamics and rewards respectively, pₘᵢₙ is the minimum reachability parameter, S is the set of states, A the set of actions, H the horizon, and T the number of episodes.
To our knowledge, our approach is the first optimistic approach applied to contextual MDPs with general function approximation (i.e., without additional knowledge regarding the function class, such as it being linear and etc.).
We present a lower bound of ?((TH|S||A|ln|?| /ln|A| )¹ᐟ² ), on the expected regret which holds even in the case of known dynamics.
Lastly, we discuss an extension of our results to CMDPs without minimum reachability, that obtains order of T³ᐟ⁴ regret.

        ----

        ## [956] Differentiable Meta Multigraph Search with Partial Message Propagation on Heterogeneous Information Networks

        **Authors**: *Chao Li, Hao Xu, Kun He*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26026](https://doi.org/10.1609/aaai.v37i7.26026)

        **Abstract**:

        Heterogeneous information networks (HINs) are widely employed for describing real-world data with intricate entities and relationships. To automatically utilize their semantic information, graph neural architecture search has recently been developed for various tasks of HINs. Existing works, on the other hand, show weaknesses in instability and inflexibility. To address these issues, we propose a novel method called Partial Message Meta Multigraph search (PMMM) to automatically optimize the neural architecture design on HINs. Specifically, to learn how graph neural networks (GNNs) propagate messages along various types of edges, PMMM adopts an efficient differentiable framework to search for a meaningful meta multigraph, which can capture more flexible and complex semantic relations than a meta graph. The differentiable search typically suffers from performance instability, so we further propose a stable algorithm called partial message search to ensure that the searched meta multigraph consistently surpasses the manually designed meta-structures, i.e., meta-paths. Extensive experiments on six benchmark datasets over two representative tasks, including node classification and recommendation, demonstrate the effectiveness of the proposed method. Our approach outperforms the state-of-the-art heterogeneous GNNs, finds out meaningful meta multigraphs, and is significantly more stable. Our code is available at https://github.com/JHL-HUST/PMMM.

        ----

        ## [957] Learning Adversarially Robust Sparse Networks via Weight Reparameterization

        **Authors**: *Chenhao Li, Qiang Qiu, Zhibin Zhang, Jiafeng Guo, Xueqi Cheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26027](https://doi.org/10.1609/aaai.v37i7.26027)

        **Abstract**:

        Although increasing model size can enhance the adversarial robustness of deep neural networks, in resource-constrained environments, there exist critical sparsity constraints. While the recent robust pruning technologies show promising direction to obtain adversarially robust sparse networks, they perform poorly with high sparsity. In this work, we bridge this performance gap by reparameterizing network parameters to simultaneously learn the sparse structure and the robustness. Specifically, we introduce Twin-Rep, which reparameterizes original weights into the product of two factors during training and performs pruning on the reparameterized weights to satisfy the target sparsity constraint. Twin-Rep implicitly adds the sparsity constraint without changing the robust training objective, thus can enhance robustness under high sparsity. We also introduce another variant of weight reparameterization for better channel pruning. When inferring, we restore the original weight structure to obtain compact and robust networks. Extensive experiments on diverse datasets demonstrate that our method achieves state-of-the-art results, outperforming the current sparse robust training method and robustness-aware pruning method. Our code is available at
https://github.com/UCAS-LCH/Twin-Rep.

        ----

        ## [958] ACE: Cooperative Multi-Agent Q-learning with Bidirectional Action-Dependency

        **Authors**: *Chuming Li, Jie Liu, Yinmin Zhang, Yuhong Wei, Yazhe Niu, Yaodong Yang, Yu Liu, Wanli Ouyang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26028](https://doi.org/10.1609/aaai.v37i7.26028)

        **Abstract**:

        Multi-agent reinforcement learning (MARL) suffers from the non-stationarity problem, which is the ever-changing targets at every iteration when multiple agents update their policies at the same time. Starting from first principle, in this paper, we manage to solve the non-stationarity problem by proposing bidirectional action-dependent Q-learning (ACE). Central to the development of ACE is the sequential decision making process wherein only one agent is allowed to take action at one time. Within this process, each agent maximizes its value function given the actions taken by the preceding agents at the inference stage. In the learning phase, each agent minimizes the TD error that is dependent on how the subsequent agents have reacted to their chosen action. Given the design of bidirectional dependency, ACE effectively turns a multi-agent MDP into a single-agent MDP. We implement the ACE framework by identifying the proper network representation to formulate the action dependency, so that the sequential decision process is computed implicitly in one forward pass. To validate ACE, we compare it with strong baselines on two MARL benchmarks. Empirical experiments demonstrate that ACE outperforms the state-of-the-art algorithms on Google Research Football and StarCraft Multi-Agent Challenge by a large margin. In particular, on SMAC tasks, ACE achieves 100% success rate on almost all the hard and super hard maps. We further study extensive research problems regarding ACE, including extension, generalization and practicability.

        ----

        ## [959] When Online Learning Meets ODE: Learning without Forgetting on Variable Feature Space

        **Authors**: *Diyang Li, Bin Gu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26029](https://doi.org/10.1609/aaai.v37i7.26029)

        **Abstract**:

        Machine learning systems that built upon varying feature space are ubiquitous across the world. When the set of practical or virtual features changes, the online learning approach can adjust the learned model accordingly rather than re-training from scratch and has been an attractive area of research. Despite its importance, most studies for algorithms that are capable of handling online features have no ensurance of stationarity point convergence, while the accuracy guaranteed methods are still limited to some simple cases such as L_1 or L_2 norms with square loss. To address this challenging problem, we develop an efficient Dynamic Feature Learning System (DFLS) to perform online learning on the unfixed feature set for more general statistical models and demonstrate how DFLS opens up many new applications. We are the first to achieve accurate & reliable feature-wise online learning for a broad class of models like logistic regression, spline interpolation, group Lasso and Poisson regression. By utilizing DFLS, the updated model is theoretically the same as the model trained from scratch using the entire new feature space. Specifically, we reparameterize the feature-varying procedure and devise the corresponding ordinary differential equation (ODE) system to compute the optimal solutions of the new model status. Simulation studies reveal that the proposed DFLS can substantially ease the computational cost without forgetting.

        ----

        ## [960] FanoutNet: A Neuralized PCB Fanout Automation Method Using Deep Reinforcement Learning

        **Authors**: *Haiyun Li, Jixin Zhang, Ning Xu, Mingyu Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26030](https://doi.org/10.1609/aaai.v37i7.26030)

        **Abstract**:

        In modern electronic manufacturing processes, multi-layer Printed Circuit Board (PCB) routing requires connecting more than hundreds of nets with perplexing topology under complex routing constraints and highly limited resources, so that takes intense effort and time of human engineers. PCB fanout as a pre-design of PCB routing has been proved to be an ideal technique to reduce the complexity of PCB routing by pre-allocating resources and pre-routing. However, current PCB fanout design heavily relies on the experience of human engineers, and there is no existing solution for PCB fanout automation in industry, which limits the quality of PCB routing automation. To address the problem, we propose a neuralized PCB fanout method by deep reinforcement learning. To the best of our knowledge, we are the first in the literature to propose the automation method for PCB fanout. We combine with Convolution Neural Network (CNN) and attention-based network to train our fanout policy model and value model. The models learn representations of PCB layout and netlist to make decisions and evaluations in place of human engineers. We employ Proximal Policy Optimization (PPO) to update the parameters of the models. In addition, we apply our PCB fanout method to a PCB router to improve the quality of PCB routing. Extensive experimental results on real-world industrial PCB benchmarks demonstrate that our approach achieves 100% routability in all industrial cases and improves wire length by an average of 6.8%, which makes a significant improvement compared with the state-of-the-art methods.

        ----

        ## [961] Causal Recurrent Variational Autoencoder for Medical Time Series Generation

        **Authors**: *Hongming Li, Shujian Yu, José C. Príncipe*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26031](https://doi.org/10.1609/aaai.v37i7.26031)

        **Abstract**:

        We propose causal recurrent variational autoencoder (CR-VAE), a novel generative model that is able to learn a Granger causal graph from a multivariate time series x and incorporates the underlying causal mechanism into its data generation process. Distinct to the classical recurrent VAEs, our CR-VAE uses a multi-head decoder, in which the p-th head is responsible for generating the p-th dimension of x (i.e., x^p). By imposing a sparsity-inducing penalty on the weights (of the decoder) and encouraging specific sets of weights to be zero, our CR-VAE learns a sparse adjacency matrix that encodes causal relations between all pairs of variables. Thanks to this causal matrix, our decoder strictly obeys the underlying principles of Granger causality, thereby making the data generating process transparent. We develop a two-stage approach to train the overall objective. Empirically, we evaluate the behavior of our model in synthetic data and two real-world human brain datasets involving, respectively, the electroencephalography (EEG) signals and the functional magnetic resonance imaging (fMRI) data. Our model consistently outperforms state-of-the-art time series generative models both qualitatively and quantitatively. Moreover, it also discovers a faithful causal graph with similar or improved accuracy over existing Granger causality-based causal inference methods.  Code of CR-VAE is publicly available at https://github.com/hongmingli1995/CR-VAE.

        ----

        ## [962] Dual Mutual Information Constraints for Discriminative Clustering

        **Authors**: *Hongyu Li, Lefei Zhang, Kehua Su*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26032](https://doi.org/10.1609/aaai.v37i7.26032)

        **Abstract**:

        Deep clustering is a fundamental task in machine learning and data mining that aims at learning clustering-oriented feature representations. In previous studies, most of deep clustering methods follow the idea of self-supervised representation learning by maximizing the consistency of all similar instance pairs while ignoring the effect of feature redundancy on clustering performance. In this paper, to address the above issue, we design a dual mutual information constrained clustering method named DMICC which is based on deep contrastive clustering architecture, in which the dual mutual information constraints are particularly employed with solid theoretical guarantees and experimental validations. Specifically, at the feature level, we reduce the redundancy among features by minimizing the mutual information across all the dimensionalities to encourage the neural network to extract more discriminative features. At the instance level, we maximize the mutual information of the similar instance pairs to obtain more unbiased and robust representations. The dual mutual information constraints happen simultaneously and thus complement each other to jointly optimize better features that are suitable for the clustering task. We also prove that our adopted mutual information constraints are superior in feature extraction, and the proposed dual mutual information constraints are clearly bounded and thus solvable. Extensive experiments on five benchmark datasets show that our proposed approach outperforms most other clustering algorithms. The code is available at https://github.com/Li-Hyn/DMICC.

        ----

        ## [963] AdaBoostC2: Boosting Classifiers Chains for Multi-Label Classification

        **Authors**: *Jiaxuan Li, Xiaoyan Zhu, Jiayin Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26033](https://doi.org/10.1609/aaai.v37i7.26033)

        **Abstract**:

        During the last decades, multi-label classification (MLC) has attracted the attention of more and more researchers due to its wide real-world applications. Many boosting methods for MLC have been proposed and achieved great successes. However, these methods only extend existing boosting frameworks to MLC and take loss functions in multi-label version to guide the iteration. These loss functions generally give a comprehensive evaluation on the label set entirety, and thus the characteristics of different labels are ignored. In this paper, we propose a multi-path AdaBoost framework specific to MLC, where each boosting path is established for distinct label and the combination of them is able to provide a maximum optimization to Hamming Loss. In each iteration, classifiers chain is taken as the base classifier to strengthen the connection between multiple AdaBoost paths and exploit the label correlation. Extensive experiments demonstrate the effectiveness of the proposed method.

        ----

        ## [964] Scaling Up Dynamic Graph Representation Learning via Spiking Neural Networks

        **Authors**: *Jintang Li, Zhouxin Yu, Zulun Zhu, Liang Chen, Qi Yu, Zibin Zheng, Sheng Tian, Ruofan Wu, Changhua Meng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26034](https://doi.org/10.1609/aaai.v37i7.26034)

        **Abstract**:

        Recent years have seen a surge in research on dynamic graph representation learning, which aims to model temporal graphs that are dynamic and evolving constantly over time. However, current work typically models graph dynamics with recurrent neural networks (RNNs), making them suffer seriously from computation and memory overheads on large temporal graphs. So far, scalability of dynamic graph representation learning on large temporal graphs remains one of the major challenges. In this paper, we present a scalable framework, namely SpikeNet, to efficiently capture the temporal and structural patterns of temporal graphs. We explore a new direction in that we can capture the evolving dynamics of temporal graphs with spiking neural networks (SNNs) instead of RNNs. As a low-power alternative to RNNs, SNNs explicitly model graph dynamics as spike trains of neuron populations and enable spike-based propagation in an efficient way. Experiments on three large real-world temporal graph datasets demonstrate that SpikeNet outperforms strong baselines on the temporal node classification task with lower computational costs. Particularly, SpikeNet generalizes to a large temporal graph (2.7M nodes and 13.9M edges) with significantly fewer parameters and computation overheads.

        ----

        ## [965] Improved Kernel Alignment Regret Bound for Online Kernel Learning

        **Authors**: *Junfan Li, Shizhong Liao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26035](https://doi.org/10.1609/aaai.v37i7.26035)

        **Abstract**:

        In this paper, we improve the kernel alignment regret bound for online kernel learning in the regime of the Hinge loss function. Previous algorithm achieves a regret of O((A_TT ln T)^{1/4}) at a computational complexity (space and per-round time) of O((A_TT ln T)^{1/2}), where A_T is called kernel alignment. We propose an algorithm whose regret bound and computational complexity are better than previous results. Our results depend on the decay rate of eigenvalues of the kernel matrix. If the eigenvalues of the kernel matrix decay exponentially, then our algorithm enjoys a regret of O((A_T)^{1/2}) at a computational complexity of O((ln T)^2). Otherwise, our algorithm enjoys a regret of O((A_TT)^{1/4}) at a computational complexity of O((A_TT)^{1/2}). We extend our algorithm to batch learning and obtain a O(T^{-1}(E[A_T])^{1/2}) excess risk bound which improves the previous O(T^{-1/2}) bound.

        ----

        ## [966] VBLC: Visibility Boosting and Logit-Constraint Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions

        **Authors**: *Mingjia Li, Binhui Xie, Shuang Li, Chi Harold Liu, Xinjing Cheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26036](https://doi.org/10.1609/aaai.v37i7.26036)

        **Abstract**:

        Generalizing models trained on normal visual conditions to target domains under adverse conditions is demanding in the practical systems. One prevalent solution is to bridge the domain gap between clear- and adverse-condition images to make satisfactory prediction on the target. However, previous methods often reckon on additional reference images of the same scenes taken from normal conditions, which are quite tough to collect in reality. Furthermore, most of them mainly focus on individual adverse condition such as nighttime or foggy, weakening the model versatility when encountering other adverse weathers. To overcome the above limitations, we propose a novel framework, Visibility Boosting and Logit-Constraint learning (VBLC), tailored for superior normal-toadverse adaptation. VBLC explores the potential of getting rid of reference images and resolving the mixture of adverse conditions simultaneously. In detail, we first propose the visibility boost module to dynamically improve target images via certain priors in the image level. Then, we figure out the overconfident drawback in the conventional cross-entropy loss for self-training method and devise the logit-constraint learning, which enforces a constraint on logit outputs during training to mitigate this pain point. To the best of our knowledge, this is a new perspective for tackling such a challenging task. Extensive experiments on two normal-to-adverse domain adaptation benchmarks, i.e., Cityscapes to ACDC and Cityscapes to FoggyCityscapes + RainCityscapes, verify the effectiveness of VBLC, where it establishes the new state of the art. Code is available at https://github.com/BIT-DA/VBLC.

        ----

        ## [967] Understanding the Generalization Performance of Spectral Clustering Algorithms

        **Authors**: *Shaojie Li, Sheng Ouyang, Yong Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26037](https://doi.org/10.1609/aaai.v37i7.26037)

        **Abstract**:

        The theoretical analysis of spectral clustering is mainly devoted to consistency, while there is little research on its generalization performance. In this paper, we study the excess risk bounds of the popular spectral clustering algorithms: relaxed RatioCut and relaxed NCut. Our analysis follows the two practical steps of spectral clustering algorithms: continuous solution and discrete solution. Firstly, we provide the convergence rate of the excess risk bounds between the empirical continuous optimal solution and the population-level continuous optimal solution. Secondly, we show the fundamental quantity influencing the excess risk between the empirical discrete optimal solution and the population-level discrete optimal solution. At the empirical level, algorithms can be designed to reduce this quantity. Based on our theoretical analysis, we propose two novel algorithms that can penalize this quantity and, additionally, can cluster the out-of-sample data without re-eigendecomposition on the overall samples. Numerical experiments on toy and real datasets confirm the effectiveness of our proposed algorithms.

        ----

        ## [968] Restructuring Graph for Higher Homophily via Adaptive Spectral Clustering

        **Authors**: *Shouheng Li, Dongwoo Kim, Qing Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26038](https://doi.org/10.1609/aaai.v37i7.26038)

        **Abstract**:

        While a growing body of literature has been studying new Graph Neural Networks (GNNs) that work on both homophilic and heterophilic graphs, little has been done on adapting classical GNNs to less-homophilic graphs. Although the ability to handle less-homophilic graphs is restricted, classical GNNs still stand out in several nice properties such as efficiency, simplicity, and explainability. In this work, we propose a novel graph restructuring method that can be integrated into any type of GNNs, including classical GNNs, to leverage the benefits of existing GNNs while alleviating their limitations. Our contribution is threefold: a) learning the weight of pseudo-eigenvectors for an adaptive spectral clustering that aligns well with known node labels, b) proposing a new density-aware homophilic metric that is robust to label imbalance, and c) reconstructing the adjacency matrix based on the result of adaptive spectral clustering to maximize the homophilic scores. The experimental results show that our graph restructuring method can significantly boost the performance of six classical GNNs by an average of 25% on less-homophilic graphs. The boosted performance is comparable to state-of-the-art methods.

        ----

        ## [969] Nearest-Neighbor Sampling Based Conditional Independence Testing

        **Authors**: *Shuai Li, Ziqi Chen, Hongtu Zhu, Christina Dan Wang, Wang Wen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26039](https://doi.org/10.1609/aaai.v37i7.26039)

        **Abstract**:

        The conditional randomization test (CRT) was recently proposed to test whether two random variables X and Y are conditionally independent given random variables Z. The CRT assumes that the conditional distribution of X given Z is known under the null hypothesis and then it is compared to the distribution of the observed samples of the original data. The aim of this paper is to develop a novel alternative of CRT by using nearest-neighbor sampling without assuming the exact form of the distribution of X given Z. Specifically, we utilize the computationally efficient 1-nearest-neighbor to approximate the conditional distribution that encodes the null hypothesis. Then, theoretically, we show that the distribution of the generated samples is very close to the true conditional distribution in terms of total variation distance.  Furthermore, we take the classifier-based conditional mutual information estimator as our test statistic. The test statistic as an empirical fundamental information theoretic quantity is able to well capture the conditional-dependence feature. We show that our proposed test is computationally very fast, while controlling type I and II errors quite well. Finally, we demonstrate the efficiency of our proposed test in both synthetic and real data analyses.

        ----

        ## [970] Towards Fine-Grained Explainability for Heterogeneous Graph Neural Network

        **Authors**: *Tong Li, Jiale Deng, Yanyan Shen, Luyu Qiu, Yongxiang Huang, Caleb Chen Cao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26040](https://doi.org/10.1609/aaai.v37i7.26040)

        **Abstract**:

        Heterogeneous graph neural networks (HGNs) are prominent approaches to node classification tasks on heterogeneous graphs. Despite the superior performance, insights about the predictions made from HGNs are obscure to humans. Existing explainability techniques are mainly proposed for GNNs on homogeneous graphs. They focus on highlighting salient graph objects to the predictions whereas the problem of how these objects affect the predictions remains unsolved. Given heterogeneous graphs with complex structures and rich semantics, it is imperative that salient objects can be accompanied with their influence paths to the predictions, unveiling the reasoning process of HGNs. In this paper, we develop xPath, a new framework that provides fine-grained explanations for black-box HGNs specifying a cause node with its influence path to the target node. In xPath, we differentiate the influence of a node on the prediction w.r.t. every individual influence path, and measure the influence by perturbing graph structure via a novel graph rewiring algorithm. Furthermore, we introduce a greedy search algorithm to find the most influential fine-grained explanations efficiently. Empirical results on various HGNs and heterogeneous graphs show that xPath yields faithful explanations efficiently, outperforming the adaptations of advanced GNN explanation approaches.

        ----

        ## [971] Metric Nearness Made Practical

        **Authors**: *Wenye Li, Fangchen Yu, Zichen Ma*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26041](https://doi.org/10.1609/aaai.v37i7.26041)

        **Abstract**:

        Given a square matrix with noisy dissimilarity measures between pairs of data samples, the metric nearness model computes the best approximation of the matrix from a set of valid distance metrics. Despite its wide applications in machine learning and data processing tasks, the model faces non-trivial computational requirements in seeking the solution due to the large number of metric constraints associated with the feasible region. Our work designed a practical approach in two stages to tackle the challenge and improve the model's scalability and applicability. The first stage computes a fast yet high-quality approximate solution from a set of isometrically embeddable metrics, further improved by an effective heuristic. The second stage refines the approximate solution with the Halpern-Lions-Wittmann-Bauschke projection algorithm, which converges quickly to the optimal solution. In empirical evaluations, the proposed approach runs at least an order of magnitude faster than the state-of-the-art solutions, with significantly improved scalability, complete conformity to constraints, less memory consumption, and other desirable features in real applications.

        ----

        ## [972] Predictive Exit: Prediction of Fine-Grained Early Exits for Computation- and Energy-Efficient Inference

        **Authors**: *Xiangjie Li, Chenfei Lou, Yuchi Chen, Zhengping Zhu, Yingtao Shen, Yehan Ma, An Zou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26042](https://doi.org/10.1609/aaai.v37i7.26042)

        **Abstract**:

        By adding exiting layers to the deep learning networks, early exit can terminate the inference earlier with accurate results. However, the passive decision-making of whether to exit or continue the next layer has to go through every pre-placed exiting layer until it exits. In addition, it is hard to adjust the configurations of the computing platforms alongside the inference proceeds. By incorporating a low-cost prediction engine, we propose a Predictive Exit framework for computation- and energy-efficient deep learning applications. Predictive Exit can forecast where the network will exit (i.e., establish the number of remaining layers to finish the inference), which effectively reduces the network computation cost by exiting on time without running every pre-placed exiting layer. Moreover, according to the number of remaining layers, proper computing configurations (i.e., frequency and voltage) are selected to execute the network to further save energy. Extensive experimental results demonstrate that Predictive Exit achieves up to 96.2% computation reduction and 72.9% energy-saving compared with classic deep learning networks; and 12.8% computation reduction and 37.6% energy-saving compared with the early exit under state-of-the-art exiting strategies, given the same inference accuracy and latency.

        ----

        ## [973] Learning with Partial Labels from Semi-supervised Perspective

        **Authors**: *Ximing Li, Yuanzhi Jiang, Changchun Li, Yiyuan Wang, Jihong Ouyang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26043](https://doi.org/10.1609/aaai.v37i7.26043)

        **Abstract**:

        Partial Label (PL) learning refers to the task of learning from the partially labeled data, where each training instance is ambiguously equipped with a set of candidate labels but only one is valid. Advances in the recent deep PL learning literature have shown that the deep learning paradigms, e.g., self-training, contrastive learning, or class activate values, can achieve promising performance. Inspired by the impressive success of deep Semi-Supervised (SS) learning, we transform the PL learning problem into the SS learning problem, and propose a novel PL learning method, namely Partial Label learning with Semi-supervised Perspective (PLSP). Specifically, we first form the pseudo-labeled dataset by selecting a small number of reliable pseudo-labeled instances with high-confidence prediction scores and treating the remaining instances as pseudo-unlabeled ones. Then we design a SS learning objective, consisting of a supervised loss for pseudo-labeled instances and a semantic consistency regularization for pseudo-unlabeled instances. We further introduce a complementary regularization for those non-candidate labels to constrain the model predictions on them to be as small as possible. Empirical results demonstrate that PLSP significantly outperforms the existing PL baseline methods, especially on high ambiguity levels. Code available: https://github.com/changchunli/PLSP.

        ----

        ## [974] Learning Compact Features via In-Training Representation Alignment

        **Authors**: *Xin Li, Xiangrui Li, Deng Pan, Yao Qiang, Dongxiao Zhu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26044](https://doi.org/10.1609/aaai.v37i7.26044)

        **Abstract**:

        Deep neural networks (DNNs) for supervised learning can be viewed as a pipeline of the feature extractor (i.e., last hidden layer) and a linear classifier (i.e., output layer) that are trained jointly with stochastic gradient descent (SGD) on the loss function (e.g., cross-entropy). In each epoch, the true gradient of the loss function is estimated using a mini-batch sampled from the training set and model parameters are then updated with the mini-batch gradients. Although the latter provides an unbiased estimation of the former, they are subject to substantial variances derived from the size and number of sampled mini-batches, leading to noisy and jumpy updates. To stabilize such undesirable variance in estimating the true gradients, we propose In-Training Representation Alignment (ITRA) that explicitly aligns feature distributions of two different mini-batches with a matching loss in the SGD training process. We also provide a rigorous analysis of the desirable effects of the matching loss on feature representation learning: (1) extracting compact feature representation; (2) reducing over-adaption on mini-batches via an adaptively weighting mechanism; and (3) accommodating to multi-modalities. Finally, we conduct large-scale experiments on both image and text classifications to demonstrate its superior performance to the strong baselines.

        ----

        ## [975] An Extreme-Adaptive Time Series Prediction Model Based on Probability-Enhanced LSTM Neural Networks

        **Authors**: *Yanhong Li, Jack Xu, David C. Anastasiu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26045](https://doi.org/10.1609/aaai.v37i7.26045)

        **Abstract**:

        Forecasting time series with extreme events has been a challenging and prevalent research topic, especially when the time series data are affected by complicated uncertain factors, such as is the case in hydrologic prediction. Diverse traditional and deep learning models have been applied to discover the nonlinear relationships and recognize the complex patterns in these types of data. However, existing methods usually ignore the negative influence of imbalanced data, or severe events, on model training. Moreover, methods are usually evaluated on a small number of generally well-behaved time series, which does not show their ability to generalize. To tackle these issues, we propose a novel probability-enhanced neural network model, called NEC+, which concurrently learns extreme and normal prediction functions and a way to choose among them via selective back propagation. We evaluate the proposed model on the difficult 3-day ahead hourly water level prediction task applied to 9 reservoirs in California. Experimental results demonstrate that the proposed model significantly outperforms state-of-the-art baselines and exhibits superior generalization ability on data with diverse distributions.

        ----

        ## [976] Implicit Stochastic Gradient Descent for Training Physics-Informed Neural Networks

        **Authors**: *Ye Li, Songcan Chen, Sheng-Jun Huang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26046](https://doi.org/10.1609/aaai.v37i7.26046)

        **Abstract**:

        Physics-informed neural networks (PINNs) have effectively been demonstrated in solving forward and inverse differential equation problems,
but they are still trapped in training failures when the target functions to be approximated exhibit high-frequency or multi-scale features.
In this paper, we propose to employ implicit stochastic gradient descent (ISGD) method to train PINNs for improving the stability of training process.
We heuristically analyze how ISGD overcome stiffness in the gradient flow dynamics of PINNs, especially for problems with multi-scale solutions.
We theoretically prove that for two-layer fully connected neural networks with large hidden nodes, randomly initialized ISGD converges to a globally optimal solution for the quadratic loss function.
Empirical results demonstrate that ISGD works well in practice and compares favorably to other gradient-based optimization methods such as SGD and Adam, while can also effectively address the numerical stiffness in training dynamics via gradient descent.

        ----

        ## [977] Provable Pathways: Learning Multiple Tasks over Multiple Paths

        **Authors**: *Yingcong Li, Samet Oymak*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26047](https://doi.org/10.1609/aaai.v37i7.26047)

        **Abstract**:

        Constructing useful representations across a large number of tasks is a key requirement for sample-efficient intelligent systems. A traditional idea in multitask learning (MTL) is building a shared representation across tasks which can then be adapted to new tasks by tuning last layers. A desirable refinement of using a shared one-fits-all representation is to construct task-specific representations. To this end, recent PathNet/muNet architectures represent individual tasks as pathways within a larger supernet. The subnetworks induced by pathways can be viewed as task-specific representations that are composition of modules within supernet's computation graph. This work explores the pathways proposal from the lens of statistical learning: We first develop novel generalization bounds for empirical risk minimization problems learning multiple tasks over multiple paths (Multipath MTL). In conjunction, we formalize the benefits of resulting multipath representation when adapting to new downstream tasks. Our bounds are expressed in terms of Gaussian complexity, lead to tangible guarantees for the class of linear representations, and provide novel insights into the quality and benefits of a multipath representation. When computation graph is a tree, Multipath MTL hierarchically clusters the tasks and builds cluster-specific representations. We provide further discussion and experiments for hierarchical MTL and rigorously identify the conditions under which Multipath MTL is provably superior to traditional MTL approaches with shallow supernets.

        ----

        ## [978] Towards Inference Efficient Deep Ensemble Learning

        **Authors**: *Ziyue Li, Kan Ren, Yifan Yang, Xinyang Jiang, Yuqing Yang, Dongsheng Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26048](https://doi.org/10.1609/aaai.v37i7.26048)

        **Abstract**:

        Ensemble methods can deliver surprising performance gains but also bring significantly higher computational costs, e.g., can be up to 2048X in large-scale ensemble tasks. However, we found that the majority of computations in ensemble methods are redundant. For instance, over 77% of samples in CIFAR-100 dataset can be correctly classified with only a single ResNet-18 model, which indicates that only around 23% of the samples need an ensemble of extra models. To this end, we propose an inference efficient ensemble learning method, to simultaneously optimize for effectiveness and efficiency in ensemble learning. More specifically, we regard ensemble of models as a sequential inference process and learn the optimal halting event for inference on a specific sample. At each timestep of the inference process, a common selector judges if the current ensemble has reached ensemble effectiveness and halt further inference, otherwise filters this challenging sample for the subsequent models to conduct more powerful ensemble. Both the base models and common selector are jointly optimized to dynamically adjust ensemble inference for different samples with various hardness, through the novel optimization goals including sequential ensemble boosting and computation saving. The experiments with different backbones on real-world datasets illustrate our method can bring up to 56% inference cost reduction while maintaining comparable performance to full ensemble, achieving significantly better ensemble utility than other baselines. Code and supplemental materials are available at https://seqml.github.io/irene.

        ----

        ## [979] SplitNet: A Reinforcement Learning Based Sequence Splitting Method for the MinMax Multiple Travelling Salesman Problem

        **Authors**: *Hebin Liang, Yi Ma, Zilin Cao, Tianyang Liu, Fei Ni, Zhigang Li, Jianye Hao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26049](https://doi.org/10.1609/aaai.v37i7.26049)

        **Abstract**:

        MinMax Multiple Travelling Salesman Problem (mTSP) is an important class of combinatorial optimization problems with many practical applications, of which the goal is to minimize the longest tour of all vehicles. Due to its high computational complexity, existing methods for solving this problem cannot obtain a solution of satisfactory quality with fast speed, especially when the scale of the problem is large. In this paper, we propose a learning-based method named SplitNet to transform the single TSP solutions into the MinMax mTSP solutions of the same instances. Specifically, we generate single TSP solution sequences and split them into mTSP subsequences using an attention-based model trained by reinforcement learning. We also design a decision region for the splitting policy, which significantly reduces the policy action space on instances of various scales and thus improves the generalization ability of SplitNet. The experimental results show that SplitNet generalizes well and outperforms existing learning-based baselines and Google OR-Tools on widely-used random datasets of different scales and public datasets with fast solving speed.

        ----

        ## [980] Stepdown SLOPE for Controlled Feature Selection

        **Authors**: *Jingxuan Liang, Xuelin Zhang, Hong Chen, Weifu Li, Xin Tang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26050](https://doi.org/10.1609/aaai.v37i7.26050)

        **Abstract**:

        Sorted L-One Penalized Estimation (SLOPE) has shown the nice theoretical property as well as empirical behavior recently on the false discovery rate (FDR) control of high-dimensional feature selection by adaptively imposing the non-increasing sequence of tuning parameters on the sorted L1 penalties. This paper goes beyond the previous concern limited to the FDR control by considering the stepdown-based SLOPE in order to control the probability of k or more false rejections (k-FWER) and the false discovery proportion (FDP). Two new SLOPEs, called k-SLOPE and F-SLOPE, are proposed to realize k-FWER and FDP control respectively, where the stepdown procedure is injected into the SLOPE scheme. For the proposed stepdown SLOPEs, we establish their theoretical guarantees on controlling k-FWER and FDP under the orthogonal design setting, and also provide an intuitive guideline for the choice of regularization parameter sequence in much general setting. Empirical evaluations on simulated data validate the effectiveness of our approaches on controlled feature selection and support our theoretical findings.

        ----

        ## [981] Positive Distribution Pollution: Rethinking Positive Unlabeled Learning from a Unified Perspective

        **Authors**: *Qianqiao Liang, Mengying Zhu, Yan Wang, Xiuyuan Wang, Wanjia Zhao, Mengyuan Yang, Hua Wei, Bing Han, Xiaolin Zheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26051](https://doi.org/10.1609/aaai.v37i7.26051)

        **Abstract**:

        Positive Unlabeled (PU) learning, which has a wide range of applications, is becoming increasingly prevalent. However, it suffers from problems such as data imbalance, selection bias, and prior agnostic in real scenarios. Existing studies focus on addressing part of these problems, which fail to provide a unified perspective to understand these problems. In this paper, we first rethink these problems by analyzing a typical PU scenario and come up with an insightful point of view that all these problems are inherently connected to one problem, i.e., positive distribution pollution, which refers to the inaccuracy in estimating positive data distribution under very little labeled data. Then, inspired by this insight, we devise a variational model named CoVPU, which addresses all three problems in a unified perspective by targeting the positive distribution pollution problem. CoVPU not only accurately separates the positive data from the unlabeled data based on discrete normalizing flows, but also effectively approximates the positive distribution based on our derived unbiased rebalanced risk estimator and supervises the approximation based on a novel prior-free variational loss. Rigorous theoretical analysis proves the convergence of CoVPU to an optimal Bayesian classifier. Extensive experiments demonstrate the superiority of CoVPU over the state-of-the-art PU learning methods under these problems.

        ----

        ## [982] Policy-Independent Behavioral Metric-Based Representation for Deep Reinforcement Learning

        **Authors**: *Weijian Liao, Zongzhang Zhang, Yang Yu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26052](https://doi.org/10.1609/aaai.v37i7.26052)

        **Abstract**:

        Behavioral metrics can calculate the distance between states or state-action pairs from the rewards and transitions difference. By virtue of their capability to filter out task-irrelevant information in theory, using them to shape a state embedding space becomes a new trend of representation learning for deep reinforcement learning (RL), especially when there are explicit distracting factors in observation backgrounds. However, due to the tight coupling between the metric and the RL policy, such metric-based methods may result in less informative embedding spaces which can weaken their aid to the baseline RL algorithm and even consume more samples to learn. We resolve this by proposing a new behavioral metric. It decouples the learning of RL policy and metric owing to its independence on RL policy. We theoretically justify its scalability to continuous state and action spaces and design a practical way to incorporate it into an RL procedure as a representation learning target. We evaluate our approach on DeepMind control tasks with default and distracting backgrounds. By statistically reliable evaluation protocols, our experiments demonstrate our approach is superior to previous metric-based methods in terms of sample efficiency and asymptotic performance in both backgrounds.

        ----

        ## [983] Geometry-Aware Network for Domain Adaptive Semantic Segmentation

        **Authors**: *Yinghong Liao, Wending Zhou, Xu Yan, Zhen Li, Yizhou Yu, Shuguang Cui*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26053](https://doi.org/10.1609/aaai.v37i7.26053)

        **Abstract**:

        Measuring and alleviating the discrepancies between the synthetic (source) and real scene (target) data is the core issue for domain adaptive semantic segmentation. Though recent works have introduced depth information in the source domain to reinforce the geometric and semantic knowledge transfer, they cannot extract the intrinsic 3D information of objects, including positions and shapes, merely based on 2D estimated depth. In this work, we propose a novel Geometry-Aware Network for Domain Adaptation (GANDA), leveraging more compact 3D geometric point cloud representations to shrink the domain gaps. In particular, we first utilize the auxiliary depth supervision from the source domain to obtain the depth prediction in the target domain to accomplish structure-texture disentanglement. Beyond depth estimation, we explicitly exploit 3D topology on the point clouds generated from RGB-D images for further coordinate-color disentanglement and pseudo-labels refinement in the target domain. Moreover, to improve the 2D classifier in the target domain, we perform domain-invariant geometric adaptation from source to target and unify the 2D semantic and 3D geometric segmentation results in two domains. Note that our GANDA is plug-and-play in any existing UDA framework. Qualitative and quantitative results demonstrate that our model outperforms state-of-the-arts on GTA5->Cityscapes and SYNTHIA->Cityscapes.

        ----

        ## [984] Social Bias Meets Data Bias: The Impacts of Labeling and Measurement Errors on Fairness Criteria

        **Authors**: *Yiqiao Liao, Parinaz Naghizadeh*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26054](https://doi.org/10.1609/aaai.v37i7.26054)

        **Abstract**:

        Although many fairness criteria have been proposed to ensure that machine learning algorithms do not exhibit or amplify our existing social biases, these algorithms are trained on datasets that can themselves be statistically biased. In this paper, we investigate the robustness of existing (demographic) fairness criteria when the algorithm is trained on biased data. We consider two forms of dataset bias: errors by prior decision makers in the labeling process, and errors in the measurement of the features of disadvantaged individuals. We analytically show that some constraints (such as Demographic Parity) can remain robust when facing certain statistical biases, while others (such as Equalized Odds) are significantly violated if trained on biased data. We provide numerical experiments based on three real-world datasets (the FICO, Adult, and German credit score datasets) supporting our analytical findings. While fairness criteria are primarily chosen under normative considerations in practice, our results show that naively applying a fairness constraint can lead to not only a loss in utility for the decision maker, but more severe unfairness when data bias exists. Thus, understanding how fairness criteria react to different forms of data bias presents a critical guideline for choosing among existing fairness criteria, or for proposing new criteria, when available datasets may be biased.

        ----

        ## [985] On the Expressive Flexibility of Self-Attention Matrices

        **Authors**: *Valerii Likhosherstov, Krzysztof Choromanski, Adrian Weller*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26055](https://doi.org/10.1609/aaai.v37i7.26055)

        **Abstract**:

        Transformer networks are able to capture patterns in data coming from many domains (text, images, videos, proteins, etc.) with little or no change to architecture components. We perform a theoretical analysis of the core component responsible for signal propagation between elements, i.e. the self-attention matrix. We ask the following question: Can self-attention matrix approximate arbitrary patterns? How small is the query dimension d required for such approximation? Our first result shows that the task of deciding whether approximation of a given pattern is possible or not is NP-hard for a fixed d greater than one. In practice, self-attention matrix typically exhibits two properties: it is sparse, and it changes dynamically depending on the input to the module. Motivated by this observation, we show that the self-attention matrix can provably approximate sparse matrices. While the parameters of self-attention are fixed, various sparse matrices can be approximated by only modifying the inputs. Our proof is based on the random projection technique and uses the seminal Johnson-Lindenstrauss lemma. In particular, we show that, in order to approximate any sparse matrix up to a given precision defined in terms of preserving matrix element ratios, d grows only logarithmically with the sequence length n.

        ----

        ## [986] Wasserstein Actor-Critic: Directed Exploration via Optimism for Continuous-Actions Control

        **Authors**: *Amarildo Likmeta, Matteo Sacco, Alberto Maria Metelli, Marcello Restelli*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26056](https://doi.org/10.1609/aaai.v37i7.26056)

        **Abstract**:

        Uncertainty quantification has been extensively used as a means to achieve efficient directed exploration in Reinforcement Learning (RL). However, state-of-the-art methods for continuous actions still suffer from high sample complexity requirements. Indeed, they either completely lack strategies for propagating the epistemic uncertainty throughout the updates, or they mix it with aleatoric uncertainty while learning the full return distribution (e.g., distributional RL). In this paper, we propose Wasserstein Actor-Critic (WAC), an actor-critic architecture inspired by the recent Wasserstein Q-Learning (WQL), that employs approximate Q-posteriors to represent the epistemic uncertainty and Wasserstein barycenters for uncertainty propagation across the state-action space. WAC enforces exploration in a principled way by guiding the policy learning process with the optimization of an upper bound of the Q-value estimates. Furthermore, we study some peculiar issues that arise when using function approximation, coupled with the uncertainty estimation, and propose a regularized loss for the uncertainty estimation. Finally, we evaluate our algorithm on standard MujoCo tasks as well as suite of continuous-actions domains, where exploration is crucial, in comparison with state-of-the-art baselines. Additional details and results can be found in the supplementary material with our Arxiv preprint.

        ----

        ## [987] Dual Label-Guided Graph Refinement for Multi-View Graph Clustering

        **Authors**: *Yawen Ling, Jianpeng Chen, Yazhou Ren, Xiaorong Pu, Jie Xu, Xiaofeng Zhu, Lifang He*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26057](https://doi.org/10.1609/aaai.v37i7.26057)

        **Abstract**:

        With the increase of multi-view graph data, multi-view graph clustering (MVGC) that can discover the hidden clusters without label supervision has attracted growing attention from researchers. Existing MVGC methods are often sensitive to the given graphs, especially influenced by the low quality graphs, i.e., they tend to be limited by the homophily assumption. However, the widespread real-world data hardly satisfy the homophily assumption. This gap limits the performance of existing MVGC methods on low homophilous graphs. To mitigate this limitation, our motivation is to extract high-level view-common information which is used to refine each view's graph, and reduce the influence of non-homophilous edges. To this end, we propose dual label-guided graph refinement for multi-view graph clustering (DuaLGR), to alleviate the vulnerability in facing low homophilous graphs. Specifically, DuaLGR consists of two modules named dual label-guided graph refinement module and graph encoder module. The first module is designed to extract the soft label from node features and graphs, and then learn a refinement matrix. In cooperation with the pseudo label from the second module, these graphs are refined and aggregated adaptively with different orders. Subsequently, a consensus graph can be generated in the guidance of the pseudo label. Finally, the graph encoder module encodes the consensus graph along with node features to produce the high-level pseudo label for iteratively clustering. The experimental results show the superior performance on coping with low homophilous graph data. The source code for DuaLGR is available at https://github.com/YwL-zhufeng/DuaLGR.

        ----

        ## [988] Metric Residual Network for Sample Efficient Goal-Conditioned Reinforcement Learning

        **Authors**: *Bo Liu, Yihao Feng, Qiang Liu, Peter Stone*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26058](https://doi.org/10.1609/aaai.v37i7.26058)

        **Abstract**:

        Goal-conditioned reinforcement learning (GCRL) has a wide range of potential real-world applications, including manipulation and navigation problems in robotics. Especially in such robotics tasks, sample efficiency is of the utmost importance for GCRL since, by default, the agent is only rewarded when it reaches its goal. While several methods have been proposed to improve the sample efficiency of GCRL, one relatively under-studied approach is the design of neural architectures to support sample efficiency.
In this work, we introduce a novel neural architecture for GCRL that achieves significantly better sample efficiency than the commonly-used monolithic network architecture. 
The key insight is that the 
optimal action-value function must satisfy the triangle inequality in a specific sense.
Furthermore, we introduce the metric residual network (MRN)  that deliberately decomposes the action-value function into the negated summation of a metric plus a residual asymmetric component. MRN provably approximates any optimal action-value function, thus making it a fitting neural architecture for GCRL.
We conduct comprehensive experiments across 12 standard benchmark environments in GCRL. The empirical results demonstrate that MRN uniformly outperforms other state-of-the-art GCRL neural architectures in terms of sample efficiency. The code is available at https://github.com/Cranial-XIX/metric-residual-network.

        ----

        ## [989] DICNet: Deep Instance-Level Contrastive Network for Double Incomplete Multi-View Multi-Label Classification

        **Authors**: *Chengliang Liu, Jie Wen, Xiaoling Luo, Chao Huang, Zhihao Wu, Yong Xu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26059](https://doi.org/10.1609/aaai.v37i7.26059)

        **Abstract**:

        In recent years, multi-view multi-label learning has aroused extensive research enthusiasm. However, multi-view multi-label data in the real world is commonly incomplete due to the uncertain factors of data collection and manual annotation, which means that not only multi-view features are often missing, and label completeness is also difficult to be satisfied. To deal with the double incomplete multi-view multi-label classification problem, we propose a deep instance-level contrastive network, namely DICNet. Different from conventional methods, our DICNet focuses on leveraging deep neural network to exploit the high-level semantic representations of samples rather than shallow-level features. First, we utilize the stacked autoencoders to build an end-to-end multi-view feature extraction framework to learn the view-specific representations of samples. Furthermore, in order to improve the consensus representation ability, we introduce an incomplete instance-level contrastive learning scheme to guide the encoders to better extract the consensus information of multiple views and use a multi-view weighted fusion module to enhance the discrimination of semantic features. Overall, our DICNet is adept in capturing consistent discriminative representations of multi-view multi-label data and avoiding the negative effects of missing views and missing labels. Extensive experiments performed on five datasets validate that our method outperforms other state-of-the-art methods.

        ----

        ## [990] Incomplete Multi-View Multi-Label Learning via Label-Guided Masked View- and Category-Aware Transformers

        **Authors**: *Chengliang Liu, Jie Wen, Xiaoling Luo, Yong Xu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26060](https://doi.org/10.1609/aaai.v37i7.26060)

        **Abstract**:

        As we all know, multi-view data is more expressive than single-view data and multi-label annotation enjoys richer supervision information than single-label, which makes multi-view multi-label learning widely applicable for various pattern recognition tasks. In this complex representation learning problem, three main challenges can be characterized as follows: i) How to learn consistent representations of samples across all views? ii) How to exploit and utilize category correlations of multi-label to guide inference? iii) How to avoid the negative impact resulting from the incompleteness of views or labels? To cope with these problems, we propose a general multi-view multi-label learning framework named label-guided masked view- and category-aware transformers in this paper. First, we design two transformer-style based modules for cross-view features aggregation and multi-label classification, respectively. The former aggregates information from different views in the process of extracting view-specific features, and the latter learns subcategory embedding to improve classification performance. Second, considering the imbalance of expressive power among views, an adaptively weighted view fusion module is proposed to obtain view-consistent embedding features. Third, we impose a label manifold constraint in sample-level representation learning to maximize the utilization of supervised information. Last but not least, all the modules are designed under the premise of incomplete views and labels, which makes our method adaptable to arbitrary multi-view and multi-label data. Extensive experiments on five datasets confirm that our method has clear advantages over other state-of-the-art methods.

        ----

        ## [991] Adaptive Discrete Communication Bottlenecks with Dynamic Vector Quantization for Heterogeneous Representational Coarseness

        **Authors**: *Dianbo Liu, Alex Lamb, Xu Ji, Pascal Tikeng Notsawo Jr., Michael Mozer, Yoshua Bengio, Kenji Kawaguchi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26061](https://doi.org/10.1609/aaai.v37i7.26061)

        **Abstract**:

        Vector Quantization (VQ) is a method for discretizing latent representations and has become a major part of the deep learning toolkit. It has been theoretically and empirically shown that discretization of representations leads to improved generalization, including in reinforcement learning where discretization can be used to bottleneck multi-agent communication to promote agent specialization and robustness. The discretization tightness of most VQ-based methods is defined by the number of discrete codes in the representation vector and the codebook size, which are fixed as hyperparameters. In this work, we propose learning to dynamically select discretization tightness conditioned on inputs, based on the hypothesis that data naturally contains variations in complexity that call for different levels of representational coarseness which is observed in many heterogeneous data sets. We show that dynamically varying tightness in communication bottlenecks can improve model performance on visual reasoning and reinforcement learning tasks with heterogeneity in representations.

        ----

        ## [992] Combating Mode Collapse via Offline Manifold Entropy Estimation

        **Authors**: *Haozhe Liu, Bing Li, Haoqian Wu, Hanbang Liang, Yawen Huang, Yuexiang Li, Bernard Ghanem, Yefeng Zheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26062](https://doi.org/10.1609/aaai.v37i7.26062)

        **Abstract**:

        Generative Adversarial Networks (GANs) have shown compelling results in various tasks and applications in recent years. However, mode collapse remains a critical problem in GANs. In this paper, we propose a novel training pipeline to address the mode collapse issue of GANs. Different from existing methods, we propose to generalize the discriminator as feature embedding and maximize the entropy of distributions in the embedding space learned by the discriminator. Specifically, two regularization terms, i.e., Deep Local Linear Embedding (DLLE) and Deep Isometric feature Mapping (DIsoMap), are introduced to encourage the discriminator to learn the structural information embedded in the data, such that the embedding space learned by the discriminator can be well-formed. Based on the well-learned embedding space supported by the discriminator, a non-parametric entropy estimator is designed to efficiently maximize the entropy of embedding vectors, playing as an approximation of maximizing the entropy of the generated distribution. By improving the discriminator and maximizing the distance of the most similar samples in the embedding space, our pipeline effectively reduces the mode collapse without sacrificing the quality of generated samples. Extensive experimental results show the effectiveness of our method which outperforms the GAN baseline, MaF-GAN on CelebA (9.13 vs. 12.43 in FID) and surpasses the recent state-of-the-art energy-based model on the ANIMEFACE dataset (2.80 vs. 2.26 in Inception score).

        ----

        ## [993] Robust Representation Learning by Clustering with Bisimulation Metrics for Visual Reinforcement Learning with Distractions

        **Authors**: *Qiyuan Liu, Qi Zhou, Rui Yang, Jie Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26063](https://doi.org/10.1609/aaai.v37i7.26063)

        **Abstract**:

        Recent work has shown that representation learning plays a critical role in sample-efficient reinforcement learning (RL) from pixels. Unfortunately, in real-world scenarios, representation learning is usually fragile to task-irrelevant distractions such as variations in background or viewpoint. To tackle this problem, we propose a novel clustering-based approach, namely Clustering with Bisimulation Metrics (CBM), which learns robust representations by grouping visual observations in the latent space. Specifically, CBM alternates between two steps: (1) grouping observations by measuring their bisimulation distances to the learned prototypes; (2) learning a set of prototypes according to the current cluster assignments. Computing cluster assignments with bisimulation metrics enables CBM to capture task-relevant information, as bisimulation metrics quantify the behavioral similarity between observations. Moreover, CBM encourages the consistency of representations within each group, which facilitates filtering out task-irrelevant information and thus induces robust representations against distractions. An appealing feature is that CBM can achieve sample-efficient representation learning even if multiple distractions exist simultaneously. Experiments demonstrate that CBM significantly improves the sample efficiency of popular visual RL algorithms and achieves state-of-the-art performance on both multiple and single distraction settings. The code is available at https://github.com/MIRALab-USTC/RL-CBM.

        ----

        ## [994] Reliable Robustness Evaluation via Automatically Constructed Attack Ensembles

        **Authors**: *Shengcai Liu, Fu Peng, Ke Tang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26064](https://doi.org/10.1609/aaai.v37i7.26064)

        **Abstract**:

        Attack Ensemble (AE), which combines multiple attacks together, provides a reliable way to evaluate adversarial robustness. In practice, AEs are often constructed and tuned by human experts, which however tends to be sub-optimal and time-consuming. In this work, we present AutoAE, a conceptually simple approach for automatically constructing AEs. In brief, AutoAE repeatedly adds the attack and its iteration steps to the ensemble that maximizes ensemble improvement per additional iteration consumed. We show theoretically that AutoAE yields AEs provably within a constant factor of the optimal for a given defense. We then use AutoAE to construct two AEs for l∞ and l2 attacks, and apply them without any tuning or adaptation to 45 top adversarial defenses on the RobustBench leaderboard. In all except one cases we achieve equal or better (often the latter) robustness evaluation than existing AEs, and notably, in 29 cases we achieve better robustness evaluation than the best known one. Such performance of AutoAE shows itself as a reliable evaluation protocol for adversarial robustness, which further indicates the huge potential of automatic AE construction. Code is available at https://github.com/LeegerPENG/AutoAE.

        ----

        ## [995] Enhancing the Antidote: Improved Pointwise Certifications against Poisoning Attacks

        **Authors**: *Shijie Liu, Andrew C. Cullen, Paul Montague, Sarah M. Erfani, Benjamin I. P. Rubinstein*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26065](https://doi.org/10.1609/aaai.v37i7.26065)

        **Abstract**:

        Poisoning attacks can disproportionately influence model behaviour by making small changes to the training corpus. While defences against specific poisoning attacks do exist, they in general do not provide any guarantees, leaving them potentially countered by novel attacks. In contrast, by examining worst-case behaviours Certified Defences make it possible to provide guarantees of the robustness of a sample against adversarial attacks modifying a finite number of training samples, known as pointwise certification. We achieve this by exploiting both Differential Privacy and the Sampled Gaussian Mechanism to ensure the invariance of prediction for each testing instance against finite numbers of poisoned examples. In doing so, our model provides guarantees of adversarial robustness that are more than twice as large as those provided by prior certifications.

        ----

        ## [996] Safe Multi-View Deep Classification

        **Authors**: *Wei Liu, Yufei Chen, Xiaodong Yue, Changqing Zhang, Shaorong Xie*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26066](https://doi.org/10.1609/aaai.v37i7.26066)

        **Abstract**:

        Multi-view deep classification expects to obtain better classification performance than using a single view. However, due to the uncertainty and inconsistency of data sources, adding data views does not necessarily lead to the performance improvements in multi-view classification. How to avoid worsening classification performance when adding views is crucial for multi-view deep learning but rarely studied. To tackle this limitation, in this paper, we reformulate the multi-view classification problem from the perspective of safe learning and thereby propose a Safe Multi-view Deep Classification (SMDC) method, which can guarantee that the classification performance does not deteriorate when fusing multiple views. In the SMDC method, we dynamically integrate multiple views and estimate the inherent uncertainties among multiple views with different root causes based on evidence theory. Through minimizing the uncertainties, SMDC promotes the evidences from data views for correct classification, and in the meantime excludes the incorrect evidences to produce the safe multi-view classification results. Furthermore, we theoretically prove that in the safe multi-view classification, adding data views will certainly not increase the empirical risk of classification. The experiments on various kinds of multi-view datasets validate that the proposed SMDC method can achieve precise and safe classification results.

        ----

        ## [997] Tensor Compressive Sensing Fused Low-Rankness and Local-Smoothness

        **Authors**: *Xinling Liu, Jingyao Hou, Jiangjun Peng, Hailin Wang, Deyu Meng, Jianjun Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26067](https://doi.org/10.1609/aaai.v37i7.26067)

        **Abstract**:

        A plethora of previous studies indicates that making full use of multifarious intrinsic properties of primordial data is a valid pathway to recover original images from their degraded observations. Typically, both low-rankness and local-smoothness broadly exist in real-world tensor data such as hyperspectral images and videos. Modeling based on both properties has received a great deal of attention, whereas most studies concentrate on experimental performance, and theoretical investigations are still lacking. In this paper, we study the tensor compressive sensing problem based on the tensor correlated total variation, which is a new regularizer used to simultaneously capture both properties existing in the same dataset. The new regularizer has the outstanding advantage of not using a trade-off parameter to balance the two properties. The obtained theories provide a robust recovery guarantee, where the error bound shows that our model certainly benefits from both properties in ground-truth data adaptively. Moreover, based on the ADMM update procedure, we design an algorithm with a global convergence guarantee to solve this model. At last, we carry out experiments to apply our model to hyperspectral image and video restoration problems. The experimental results show that our method is prominently better than many other competing ones. Our code and Supplementary Material are available at https://github.com/fsliuxl/cs-tctv.

        ----

        ## [998] Coupling Artificial Neurons in BERT and Biological Neurons in the Human Brain

        **Authors**: *Xu Liu, Mengyue Zhou, Gaosheng Shi, Yu Du, Lin Zhao, Zihao Wu, David Liu, Tianming Liu, Xintao Hu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26068](https://doi.org/10.1609/aaai.v37i7.26068)

        **Abstract**:

        Linking computational natural language processing (NLP) models and neural responses to language in the human brain on the one hand facilitates the effort towards disentangling the neural representations underpinning language perception, on the other hand provides neurolinguistics evidence to evaluate and improve NLP models. Mappings of an NLP model’s representations of and the brain activities evoked by linguistic input are typically deployed to reveal this symbiosis. However, two critical problems limit its advancement: 1) The model’s representations (artificial neurons, ANs) rely on layer-level embeddings and thus lack fine-granularity; 2) The brain activities (biological neurons, BNs) are limited to neural recordings of isolated cortical unit (i.e., voxel/region) and thus lack integrations and interactions among brain functions. To address those problems, in this study, we 1) define ANs with fine-granularity in transformer-based NLP models (BERT in this study) and measure their temporal activations to input text sequences; 2) define BNs as functional brain networks (FBNs) extracted from functional magnetic resonance imaging (fMRI) data to capture functional interactions in the brain; 3) couple ANs and BNs by maximizing the synchronization of their temporal activations. Our experimental results demonstrate 1) The activations of ANs and BNs are significantly synchronized; 2) the ANs carry meaningful linguistic/semantic information and anchor to their BN signatures; 3) the anchored BNs are interpretable in a neurolinguistic context. Overall, our study introduces a novel, general, and effective framework to link transformer-based NLP models and neural activities in response to language and may provide novel insights for future studies such as brain-inspired evaluation and development of NLP models.

        ----

        ## [999] EASAL: Entity-Aware Subsequence-Based Active Learning for Named Entity Recognition

        **Authors**: *Yang Liu, Jinpeng Hu, Zhihong Chen, Xiang Wan, Tsung-Hui Chang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i7.26069](https://doi.org/10.1609/aaai.v37i7.26069)

        **Abstract**:

        Active learning is a critical technique for reducing labelling load by selecting the most informative data. Most previous works applied active learning on Named Entity Recognition (token-level task) similar to the text classification (sentence-level task). They failed to consider the heterogeneity of uncertainty within each sentence and required access to the entire sentence for the annotator when labelling. To overcome the mentioned limitations, in this paper, we allow the active learning algorithm to query subsequences within sentences and propose an Entity-Aware Subsequences-based Active Learning (EASAL) that utilizes an effective Head-Tail pointer to query one entity-aware subsequence for each sentence based on BERT. For other tokens outside this subsequence, we randomly select 30% of these tokens to be pseudo-labelled for training together where the model directly predicts their pseudo-labels. Experimental results on both news and biomedical datasets demonstrate the effectiveness of our proposed method. The code is released at https://github.com/lylylylylyly/EASAL.

        ----

        

[Go to the previous page](AAAI-2023-list04.md)

[Go to the next page](AAAI-2023-list06.md)

[Go to the catalog section](README.md)