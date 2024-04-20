## [800] Dist2Cycle: A Simplicial Neural Network for Homology Localization

**Authors**: *Alexandros Dimitrios Keros, Vidit Nanda, Kartic Subr*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20673](https://doi.org/10.1609/aaai.v36i7.20673)

**Abstract**:

Simplicial complexes can be viewed as high dimensional generalizations of graphs that explicitly encode multi-way ordered relations between vertices at different resolutions, all at once. This concept is central towards detection of higher dimensional topological features of data, features to which graphs, encoding only pairwise relationships, remain oblivious. While attempts have been made to extend Graph Neural Networks (GNNs) to a simplicial complex setting, the methods do not inherently exploit, or reason about, the underlying topological structure of the network. We propose a graph convolutional model for learning functions parametrized by the k-homological features of simplicial complexes. By spectrally manipulating their combinatorial k-dimensional Hodge Laplacians, the proposed model enables learning topological features of the underlying simplicial complexes, specifically, the distance of each k-simplex from the nearest "optimal" k-th homology generator, effectively providing an alternative to homology localization.

----

## [801] Same State, Different Task: Continual Reinforcement Learning without Interference

**Authors**: *Samuel Kessler, Jack Parker-Holder, Philip J. Ball, Stefan Zohren, Stephen J. Roberts*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20674](https://doi.org/10.1609/aaai.v36i7.20674)

**Abstract**:

Continual Learning (CL) considers the problem of training an agent sequentially on a set of tasks while seeking to retain performance on all previous tasks. A key challenge in CL is catastrophic forgetting, which arises when performance on a previously mastered task is reduced when learning a new task. While a variety of methods exist to combat forgetting, in some cases tasks are fundamentally incompatible with each other and thus cannot be learnt by a single policy. This can occur, in reinforcement learning (RL) when an agent may be rewarded for achieving different goals from the same observation. In this paper we formalize this "interference" as distinct from the problem of forgetting. We show that existing CL methods based on single neural network predictors with shared replay buffers fail in the presence of interference. Instead, we propose a simple method, OWL, to address this challenge. OWL learns a factorized policy, using shared feature extraction layers, but separate heads, each specializing on a new task. The separate heads in OWL are used to prevent interference. At test time, we formulate policy selection as a multi-armed bandit problem, and show it is possible to select the best policy for an unknown task using feedback from the environment. The use of bandit algorithms allows the OWL agent to constructively re-use different continually learnt policies at different times during an episode. We show in multiple RL environments that existing replay based CL methods fail, while OWL is able to achieve close to optimal performance when training sequentially.

----

## [802] Spatial Frequency Bias in Convolutional Generative Adversarial Networks

**Authors**: *Mahyar Khayatkhoei, Ahmed Elgammal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20675](https://doi.org/10.1609/aaai.v36i7.20675)

**Abstract**:

Understanding the capability of Generative Adversarial Networks (GANs) in learning the full spectrum of spatial frequencies, that is, beyond the low-frequency dominant spectrum of natural images, is critical for assessing the reliability of GAN-generated data in any detail-sensitive application. In this work, we show that the ability of convolutional GANs to learn an image distribution depends on the spatial frequency of the underlying carrier signal, that is, they have a bias against learning high spatial frequencies. Our findings are consistent with the recent observations of high-frequency artifacts in GAN-generated images, but further suggest that such artifacts are the consequence of an underlying bias. We also provide a theoretical explanation for this bias as the manifestation of linear dependencies present in the spectrum of filters of a typical generative Convolutional Neural Network (CNN). Finally, by proposing a proof-of-concept method that can effectively manipulate this bias towards other spatial frequencies, we show that the bias is not fixed and can be exploited to explicitly direct computational resources towards any specific spatial frequency of interest in a dataset, with minimal computational overhead.

----

## [803] The Effect of Manifold Entanglement and Intrinsic Dimensionality on Learning

**Authors**: *Daniel Kienitz, Ekaterina Komendantskaya, Michael A. Lones*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20676](https://doi.org/10.1609/aaai.v36i7.20676)

**Abstract**:

We empirically investigate the effect of class manifold entanglement and the intrinsic and extrinsic dimensionality of the data distribution on the sample complexity of supervised classification with deep ReLU networks. We separate the effect of entanglement and intrinsic dimensionality and show statistically for artificial and real-world image datasets that the intrinsic dimensionality and the entanglement have an interdependent effect on the sample complexity. Low levels of entanglement lead to low increases of the sample complexity when the intrinsic dimensionality is increased, while for high levels of entanglement the impact of the intrinsic dimensionality increases as well. Further, we show that in general the sample complexity is primarily due to the entanglement and only secondarily due to the intrinsic dimensionality of the data distribution.

----

## [804] A Computable Definition of the Spectral Bias

**Authors**: *Jonas Kiessling, Filip Thor*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20677](https://doi.org/10.1609/aaai.v36i7.20677)

**Abstract**:

Neural networks have a bias towards low frequency functions. This spectral bias has been the subject of several previous studies, both empirical and theoretical.
 Here we present a computable definition of the spectral bias based on a decomposition of the reconstruction error into a low and a high frequency component. The distinction between low and high frequencies is made in a way that allows for easy interpretation of the spectral bias. Furthermore, we present two methods for estimating the spectral bias. Method 1 relies on the use of the discrete Fourier transform to explicitly estimate the Fourier spectrum of the prediction residual, and Method 2 uses convolution to extract the low frequency components, where the convolution integral is estimated by Monte Carlo methods.
 The spectral bias depends on the distribution of the data, which is approximated with kernel density estimation when unknown. We devise a set of numerical experiments that confirm that low frequencies are learned first, a behavior quantified by our definition.

----

## [805] A Nested Bi-level Optimization Framework for Robust Few Shot Learning

**Authors**: *KrishnaTeja Killamsetty, Changbin Li, Chen Zhao, Feng Chen, Rishabh K. Iyer*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20678](https://doi.org/10.1609/aaai.v36i7.20678)

**Abstract**:

Model-Agnostic Meta-Learning (MAML), a popular gradient-based meta-learning framework, assumes that the contribution of each task or instance to the meta-learner is equal.Hence, it fails to address the domain shift between base and novel classes in few-shot learning. In this work, we propose a novel robust meta-learning algorithm, NESTEDMAML, which learns to assign weights to training tasks or instances. We con-sider weights as hyper-parameters and iteratively optimize them using a small set of validation tasks set in a nested bi-level optimization approach (in contrast to the standard bi-level optimization in MAML). We then applyNESTED-MAMLin the meta-training stage, which involves (1) several tasks sampled from a distribution different from the meta-test task distribution, or (2) some data samples with noisy labels.Extensive experiments on synthetic and real-world datasets demonstrate that NESTEDMAML efficiently mitigates the effects of ”unwanted” tasks or instances, leading to significant improvement over the state-of-the-art robust meta-learning methods.

----

## [806] Fast Monte-Carlo Approximation of the Attention Mechanism

**Authors**: *Hyunjun Kim, JeongGil Ko*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20679](https://doi.org/10.1609/aaai.v36i7.20679)

**Abstract**:

We introduce Monte-Carlo Attention (MCA), a randomized approximation method for reducing the computational cost of self-attention mechanisms in Transformer architectures. MCA exploits the fact that the importance of each token in an input sequence vary with respect to their attention scores; thus, some degree of error can be tolerable when encoding tokens with low attention. Using approximate matrix multiplication, MCA applies different error bounds to encode input tokens such that those with low attention scores are computed with relaxed precision, whereas errors of salient elements are minimized. MCA can operate in parallel with other attention optimization schemes and does not require model modification. We study the theoretical error bounds and demonstrate that MCA reduces attention complexity (in FLOPS) for various Transformer models by up to 11 in GLUE benchmarks without compromising model accuracy. Source code and appendix: https://github.com/eis-lab/monte-carlo-attention

----

## [807] Towards a Rigorous Evaluation of Time-Series Anomaly Detection

**Authors**: *Siwon Kim, Kukjin Choi, Hyun-Soo Choi, Byunghan Lee, Sungroh Yoon*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20680](https://doi.org/10.1609/aaai.v36i7.20680)

**Abstract**:

In recent years, proposed studies on time-series anomaly detection (TAD) report high F1 scores on benchmark TAD datasets, giving the impression of clear improvements in TAD. However, most studies apply a peculiar evaluation protocol called point adjustment (PA) before scoring. In this paper, we theoretically and experimentally reveal that the PA protocol has a great possibility of overestimating the detection performance; even a random anomaly score can easily turn into a state-of-the-art TAD method. Therefore, the comparison of TAD methods after applying the PA protocol can lead to misguided rankings. Furthermore, we question the potential of existing TAD methods by showing that an untrained model obtains comparable detection performance to the existing methods even when PA is forbidden. Based on our findings, we propose a new baseline and an evaluation protocol. We expect that our study will help a rigorous evaluation of TAD and lead to further improvement in future researches.

----

## [808] Introducing Symmetries to Black Box Meta Reinforcement Learning

**Authors**: *Louis Kirsch, Sebastian Flennerhag, Hado van Hasselt, Abram L. Friesen, Junhyuk Oh, Yutian Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20681](https://doi.org/10.1609/aaai.v36i7.20681)

**Abstract**:

Meta reinforcement learning (RL) attempts to discover new RL algorithms automatically from environment interaction. In so-called black-box approaches, the policy and the learning algorithm are jointly represented by a single neural network. These methods are very flexible, but they tend to underperform compared to human-engineered RL algorithms in terms of generalisation to new, unseen environments. In this paper, we explore the role of symmetries in meta-generalisation. We show that a recent successful meta RL approach that meta-learns an objective for backpropagation-based learning exhibits certain symmetries (specifically the reuse of the learning rule, and invariance to input and output permutations) that are not present in typical black-box meta RL systems. We hypothesise that these symmetries can play an important role in meta-generalisation. Building off recent work in black-box supervised meta learning, we develop a black-box meta RL system that exhibits these same symmetries. We show through careful experimentation that incorporating these symmetries can lead to algorithms with a greater ability to generalise to unseen action & observation spaces, tasks, and environments.

----

## [809] Directed Graph Auto-Encoders

**Authors**: *Georgios Kollias, Vasileios Kalantzis, Tsuyoshi Idé, Aurélie C. Lozano, Naoki Abe*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20682](https://doi.org/10.1609/aaai.v36i7.20682)

**Abstract**:

We introduce a new class of auto-encoders for directed graphs, motivated by a direct extension of the Weisfeiler-Leman algorithm to pairs of node labels. The proposed model learns pairs of interpretable latent representations for the nodes of directed graphs, and uses parameterized graph convolutional network (GCN) layers for its  encoder and an asymmetric inner product decoder. Parameters in the encoder control the weighting of representations exchanged between neighboring nodes. We demonstrate the ability of the proposed model to learn meaningful latent embeddings and achieve superior performance on the directed link prediction task on several popular network datasets.

----

## [810] HNO: High-Order Numerical Architecture for ODE-Inspired Deep Unfolding Networks

**Authors**: *Lin Kong, Wei Sun, Fanhua Shang, Yuanyuan Liu, Hongying Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20683](https://doi.org/10.1609/aaai.v36i7.20683)

**Abstract**:

Recently, deep unfolding networks (DUNs) based on optimization algorithms have received increasing attention, and their high efficiency has been confirmed by many experimental and theoretical results.  Since this type of networks combines model-based traditional optimization algorithms, they have high interpretability. In addition, ordinary differential equations (ODEs) are often used to explain deep neural networks, and provide some inspiration for designing innovative network models. In this paper, we transform DUNs into first-order ODE forms, and propose a high-order numerical architecture for ODE-inspired deep unfolding networks. To the best of our knowledge, this is the first work to establish the relationship between DUNs and ODEs. Moreover, we take two representative DUNs as examples, apply our architecture to them and design novel DUNs. In theory, we prove the existence, uniqueness of the solution and convergence of the proposed network, and also prove that our network obtains a fast linear convergence rate. Extensive experiments verify the effectiveness and advantages of our architecture.

----

## [811] Deep Reinforcement Learning Policies Learn Shared Adversarial Features across MDPs

**Authors**: *Ezgi Korkmaz*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20684](https://doi.org/10.1609/aaai.v36i7.20684)

**Abstract**:

The use of deep neural networks as function approximators has led to striking progress for reinforcement learning algorithms and applications. Yet the knowledge we have on decision boundary geometry and the loss landscape of neural policies is still quite limited. In this paper, we propose a framework to investigate the decision boundary and loss landscape similarities across states and across MDPs. We conduct experiments in various games from Arcade Learning Environment, and discover that high sensitivity directions for neural policies are correlated across MDPs. We argue that these high sensitivity directions support the hypothesis that non-robust features are shared across training environments of reinforcement learning agents. We believe our results reveal fundamental properties of the environments used in deep reinforcement learning training, and represent a tangible step towards building robust and reliable deep reinforcement learning agents.

----

## [812] Fast Approximations for Job Shop Scheduling: A Lagrangian Dual Deep Learning Method

**Authors**: *James Kotary, Ferdinando Fioretto, Pascal Van Hentenryck*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20685](https://doi.org/10.1609/aaai.v36i7.20685)

**Abstract**:

The Jobs Shop Scheduling problem (JSP) is a canonical combinatorial optimization problem that is routinely solved for a variety of industrial purposes. It models the optimal scheduling of multiple sequences of tasks, each under a fixed order of operations, in which individual tasks require exclusive access to a predetermined resource for a specified processing time. The problem is NP-hard and computationally challenging even for medium-sized instances. Motivated by the increased stochasticity in production chains, this paper explores a deep learning approach to deliver efficient and accurate approximations to the JSP. In particular, this paper proposes the design of a deep neural network architecture to exploit the problem structure, its integration with Lagrangian duality to capture the problem constraints, and a post-processing optimization, to guarantee solution feasibility. The resulting method, called JSP-DNN, is evaluated on hard JSP instances from the JSPLIB benchmark library and is shown to produce JSP approximations of high quality at negligible computational costs.

----

## [813] Learning Robust Policy against Disturbance in Transition Dynamics via State-Conservative Policy Optimization

**Authors**: *Yufei Kuang, Miao Lu, Jie Wang, Qi Zhou, Bin Li, Houqiang Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20686](https://doi.org/10.1609/aaai.v36i7.20686)

**Abstract**:

Deep reinforcement learning algorithms can perform poorly in real-world tasks due to the discrepancy between source and target environments. This discrepancy is commonly viewed as the disturbance in transition dynamics. Many existing algorithms learn robust policies by modeling the disturbance and applying it to source environments during training, which usually requires prior knowledge about the disturbance and control of simulators. However, these algorithms can fail in scenarios where the disturbance from target environments is unknown or is intractable to model in simulators. To tackle this problem, we propose a novel model-free actor-critic algorithm---namely, state-conservative policy optimization (SCPO)---to learn robust policies without modeling the disturbance in advance. Specifically, SCPO reduces the disturbance in transition dynamics to that in state space and then approximates it by a simple gradient-based regularizer. The appealing features of SCPO include that it is simple to implement and does not require additional knowledge about the disturbance or specially designed simulators. Experiments in several robot control tasks demonstrate that SCPO learns robust policies against the disturbance in transition dynamics.

----

## [814] Gradient Based Activations for Accurate Bias-Free Learning

**Authors**: *Vinod K. Kurmi, Rishabh Sharma, Yash Vardhan Sharma, Vinay P. Namboodiri*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20687](https://doi.org/10.1609/aaai.v36i7.20687)

**Abstract**:

Bias mitigation in machine learning models is imperative, yet challenging. While several approaches have been proposed, one view towards mitigating bias is through adversarial learning. A discriminator is used to identify the bias attributes such as gender, age or race in question. This discriminator is used adversarially to ensure that it cannot distinguish the bias attributes. The main drawback in such a model is that it directly introduces a trade-off with accuracy as the features that the discriminator deems to be sensitive for discrimination of bias could be correlated with classification. In this work we solve the problem. We show that a biased discriminator can actually be used to improve this bias-accuracy tradeoff. Specifically, this is achieved by using a feature masking approach using the discriminator's gradients. We ensure that the features favoured for the bias discrimination are de-emphasized and the unbiased features are enhanced during classification. We show that this simple approach works well to reduce bias as well as improve accuracy significantly. We evaluate the proposed model on standard benchmarks. We improve the accuracy of the adversarial methods while maintaining or even improving the unbiasness and also outperform several other recent methods.

----

## [815] TrustAL: Trustworthy Active Learning Using Knowledge Distillation

**Authors**: *Beong-woo Kwak, Youngwook Kim, Yu Jin Kim, Seung-won Hwang, Jinyoung Yeo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20688](https://doi.org/10.1609/aaai.v36i7.20688)

**Abstract**:

Active learning can be defined as iterations of data labeling, model training, and data acquisition, until sufficient labels are acquired. A traditional view of data acquisition is that, through iterations, knowledge from human labels and models is implicitly distilled to monotonically increase the accuracy and label consistency. Under this assumption, the most recently trained model is a good surrogate for the current labeled data, from which data acquisition is requested based on uncertainty/diversity. Our contribution is debunking this myth and proposing a new objective for distillation. First, we found example forgetting, which indicates the loss of knowledge learned across iterations. Second, for this reason, the last model is no longer the best teacher-- For mitigating such forgotten knowledge, we select one of its predecessor models as a teacher, by our proposed notion of "consistency". We show that this novel distillation is distinctive in the following three aspects; First, consistency ensures to avoid forgetting labels. Second, consistency improves both uncertainty/diversity of labeled data. Lastly, consistency redeems defective labels produced by human annotators.

----

## [816] Tight Neural Network Verification via Semidefinite Relaxations and Linear Reformulations

**Authors**: *Jianglin Lan, Yang Zheng, Alessio Lomuscio*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20689](https://doi.org/10.1609/aaai.v36i7.20689)

**Abstract**:

We present a novel semidefinite programming (SDP) relaxation that
enables tight and efficient verification of neural networks. The
tightness is achieved by combining SDP relaxations with valid linear
cuts, constructed by using the reformulation-linearisation technique
(RLT). The computational efficiency results from a layerwise SDP
formulation and an iterative algorithm for incrementally adding
RLT-generated linear cuts to the verification formulation.  The layer
RLT-SDP relaxation here presented is shown to produce the tightest SDP
relaxation for ReLU neural networks available in the literature. We
report experimental results based on MNIST neural networks showing
that the method outperforms the state-of-the-art methods while
maintaining acceptable computational overheads.  For networks of
approximately 10k nodes (1k, respectively), the proposed method
achieved an improvement in the ratio of certified robustness cases
from 0% to 82% (from 35% to 70%, respectively).

----

## [817] Learning Adversarial Markov Decision Processes with Delayed Feedback

**Authors**: *Tal Lancewicki, Aviv Rosenberg, Yishay Mansour*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20690](https://doi.org/10.1609/aaai.v36i7.20690)

**Abstract**:

Reinforcement learning typically assumes that agents observe feedback for their actions immediately, but in many real-world applications (like recommendation systems) feedback is observed in delay. This paper studies online learning in episodic Markov decision processes (MDPs) with unknown transitions, adversarially changing costs and unrestricted delayed feedback. That is, the costs and trajectory of episode k are revealed to the learner only in the end of episode k+dᵏ, where the delays dᵏ are neither identical nor bounded, and are chosen by an oblivious adversary. We present novel algorithms based on policy optimization that achieve near-optimal high-probability regret of (K+D)¹ᐟ² under full-information feedback, where K is the number of episodes and D=∑ₖ dᵏ is the total delay. Under bandit feedback, we prove similar (K+D)¹ᐟ² regret assuming the costs are stochastic, and (K+D)²ᐟ³ regret in the general case. We are the first to consider regret minimization in the important setting of MDPs with delayed feedback.

----

## [818] Learning Not to Learn: Nature versus Nurture In Silico

**Authors**: *Robert Tjarko Lange, Henning Sprekeler*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20691](https://doi.org/10.1609/aaai.v36i7.20691)

**Abstract**:

Animals are equipped with a rich innate repertoire of sensory, behavioral and motor skills, which allows them to interact with the world immediately after birth. At the same time, many behaviors are highly adaptive and can be tailored to specific environments by means of learning. In this work, we use mathematical analysis and the framework of memory-based meta-learning (or ’learning to learn’) to answer when it is beneficial to learn such an adaptive strategy and when to hard-code a heuristic behavior. We find that the interplay of ecological uncertainty, task complexity and the agents’ lifetime has crucial effects on the meta-learned amortized Bayesian inference performed by an agent. There exist two regimes: One in which meta-learning yields a learning algorithm that implements task-dependent information-integration and a second regime in which meta-learning imprints a heuristic or ’hard-coded’ behavior. Further analysis reveals that non-adaptive behaviors are not only optimal for aspects of the environment that are stable across individuals, but also in situations where an adaptation to the environment would in fact be highly beneficial, but could not be done quickly enough to be exploited within the remaining lifetime. Hard-coded behaviors should hence not only be those that always work, but also those that are too complex to be learned within a reasonable time frame.

----

## [819] Optimization for Classical Machine Learning Problems on the GPU

**Authors**: *Sören Laue, Mark Blacher, Joachim Giesen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20692](https://doi.org/10.1609/aaai.v36i7.20692)

**Abstract**:

Constrained optimization problems arise frequently in classical machine learning. There exist frameworks addressing constrained optimization, for instance, CVXPY and GENO. However, in contrast to deep learning frameworks, GPU support is limited. Here, we extend the GENO framework to also solve constrained optimization problems on the GPU. The framework allows the user to specify constrained optimization problems in an easy-to-read modeling language. A solver is then automatically generated from this specification. When run on the GPU, the solver outperforms state-of-the-art approaches like CVXPY combined with a GPU-accelerated solver such as cuOSQP or SCS by a few orders of magnitude.

----

## [820] Interpretable Clustering via Multi-Polytope Machines

**Authors**: *Connor Lawless, Jayant Kalagnanam, Lam M. Nguyen, Dzung T. Phan, Chandra Reddy*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20693](https://doi.org/10.1609/aaai.v36i7.20693)

**Abstract**:

Clustering is a popular unsupervised learning tool often used to discover groups within a larger population such as customer segments, or patient subtypes. However, despite its use as a tool for subgroup discovery and description few state-of-the-art algorithms provide any rationale or description behind the clusters found. We propose a novel approach for interpretable clustering that both clusters data points and constructs polytopes around the discovered clusters to explain them. Our framework allows for additional constraints on the polytopes including ensuring that the hyperplanes constructing the polytope are axis-parallel or sparse with integer coefficients. We formulate the problem of constructing clusters via polytopes as a Mixed-Integer Non-Linear Program (MINLP). To solve our formulation we propose a two phase approach where we first initialize clusters and polytopes using alternating minimization, and then use coordinate descent to boost clustering performance. We benchmark our approach on a suite of synthetic and real world clustering problems, where our algorithm outperforms state of the art interpretable and non-interpretable clustering algorithms.

----

## [821] Episodic Policy Gradient Training

**Authors**: *Hung Le, Majid Abdolshah, Thommen George Karimpanal, Kien Do, Dung Nguyen, Svetha Venkatesh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20694](https://doi.org/10.1609/aaai.v36i7.20694)

**Abstract**:

We introduce a novel training procedure for policy gradient methods wherein episodic memory is used to optimize the hyperparameters of reinforcement learning algorithms on-the-fly. Unlike other hyperparameter searches, we formulate hyperparameter scheduling as a standard Markov Decision Process and use episodic memory to store the outcome of used hyperparameters and their training contexts. At any policy update step, the policy learner refers to the stored experiences, and adaptively reconfigures its learning algorithm with the new hyperparameters determined by the memory. This mechanism, dubbed as Episodic Policy Gradient Training (EPGT), enables an episodic learning process, and jointly learns the policy and the learning algorithm's hyperparameters within a single run. Experimental results on both continuous and discrete environments demonstrate the advantage of using the proposed method in boosting the performance of various policy gradient algorithms.

----

## [822] Stability Verification in Stochastic Control Systems via Neural Network Supermartingales

**Authors**: *Mathias Lechner, Dorde Zikelic, Krishnendu Chatterjee, Thomas A. Henzinger*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20695](https://doi.org/10.1609/aaai.v36i7.20695)

**Abstract**:

We consider the problem of formally verifying almost-sure (a.s.) asymptotic stability in discrete-time nonlinear stochastic control systems. While verifying stability in deterministic control systems is extensively studied in the literature, verifying stability in stochastic control systems is an open problem. The few existing works on this topic either consider only specialized forms of stochasticity or make restrictive assumptions on the system, rendering them inapplicable to learning algorithms with neural network policies. 
 In this work, we present an approach for general nonlinear stochastic control problems with two novel aspects: (a) instead of classical stochastic extensions of Lyapunov functions, we use ranking supermartingales (RSMs) to certify a.s. asymptotic stability, and (b) we present a method for learning neural network RSMs. 
 We prove that our approach guarantees a.s. asymptotic stability of the system and
 provides the first method to obtain bounds on the stabilization time, which stochastic Lyapunov functions do not.
 Finally, we validate our approach experimentally on a set of nonlinear stochastic reinforcement learning environments with neural network policies.

----

## [823] Learning Losses for Strategic Classification

**Authors**: *Tosca Lechner, Ruth Urner*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20696](https://doi.org/10.1609/aaai.v36i7.20696)

**Abstract**:

Strategic classification, i.e. classification under possible strategic manipulations of features, has received a lot of attention from both the machine learning and the game theory community. Most works focus on analysing properties of the optimal decision rule under such manipulations. In our work we take a learning theoretic perspective, focusing on the sample complexity needed to learn a good decision rule which is robust to strategic manipulation. We perform this analysis by introducing a novel loss function, the strategic manipulation loss, which takes into account both the accuracy of the final decision rule and its vulnerability to manipulation. We analyse the sample complexity for a known graph of possible manipulations in terms of the complexity of the function class and the manipulation graph. Additionally, we initialize the study of learning under unknown manipulation capabilities of the involved agents. Using techniques from transfer learning theory, we define a similarity measure for manipulation graphs and show that learning outcomes are robust with respect to small changes in the manipulation graph. Lastly, we analyse the (sample complexity of) learning of the manipulation capability of agents with respect to this similarity measure, providing novel guarantees for strategic classification with respect to an unknown manipulation graph.

----

## [824] Differentially Private Normalizing Flows for Synthetic Tabular Data Generation

**Authors**: *Jaewoo Lee, Minjung Kim, Yonghyun Jeong, Youngmin Ro*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20697](https://doi.org/10.1609/aaai.v36i7.20697)

**Abstract**:

Normalizing flows have shown to be a promising approach to deep generative modeling due to their ability to exactly evaluate density --- other alternatives either implicitly model the density or use approximate surrogate density. In this work, we present a differentially private normalizing flow model for heterogeneous tabular data. Normalizing flows are in general not amenable to differentially private training because they require complex neural networks with larger depth (compared to other generative models) and use specialized architectures for which per-example gradient computation is difficult (or unknown). To reduce the parameter complexity, the proposed model introduces a conditional spline flow which simulates transformations at different stages depending on additional input and is shared among sub-flows. For privacy, we introduce two fine-grained gradient clipping strategies that provide a better signal-to-noise ratio and derive fast gradient clipping methods for layers with custom parameterization. Our empirical evaluations show that the proposed model preserves statistical properties of original dataset better than other baselines.

----

## [825] Multi-Head Modularization to Leverage Generalization Capability in Multi-Modal Networks

**Authors**: *Jun-Tae Lee, Hyunsin Park, Sungrack Yun, Simyung Chang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20698](https://doi.org/10.1609/aaai.v36i7.20698)

**Abstract**:

It has been crucial to leverage the rich information of multiple modalities in many tasks. Existing works have tried to design multi-modal networks with descent multi-modal fusion modules. Instead, we focus on improving generalization capability of multi-modal networks, especially the fusion module. Viewing the multi-modal data as different projections of information, we first observe that bad projection can cause poor generalization behaviors of multi-modal networks. Then, motivated by well-generalized network's low sensitivity to perturbation, we propose a novel multi-modal training method, multi-head modularization (MHM). We modularize a multi-modal network as a series of uni-modal embedding, multi-modal embedding, and task-specific head modules. Also, for training, we exploit multiple head modules learned with different datasets, swapping each other. From this, we can make the multi-modal embedding module robust to all the heads with different generalization behaviors. In testing phase, we select one of the head modules not to increase the computational cost. Owing to the perturbation of head modules, though including one selected head, the deployed network is more well-generalized compared to the simply end-to-end learned. We verify the effectiveness of MHM on various multi-modal tasks. We use the state-of-the-art methods as baselines, and show notable performance gain for all the baselines.

----

## [826] Fast and Efficient MMD-Based Fair PCA via Optimization over Stiefel Manifold

**Authors**: *Junghyun Lee, Gwangsu Kim, Mahbod Olfat, Mark Hasegawa-Johnson, Chang D. Yoo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20699](https://doi.org/10.1609/aaai.v36i7.20699)

**Abstract**:

This paper defines fair principal component analysis (PCA) as minimizing the maximum mean discrepancy (MMD) between the dimensionality-reduced conditional distributions of different protected classes. The incorporation of MMD naturally leads to an exact and tractable mathematical formulation of fairness with good statistical properties. We formulate the problem of fair PCA subject to MMD constraints as a non-convex optimization over the Stiefel manifold and solve it using the Riemannian Exact Penalty Method with Smoothing (REPMS). Importantly, we provide a local optimality guarantee and explicitly show the theoretical effect of each hyperparameter in practical settings, extending previous results. Experimental comparisons based on synthetic and UCI datasets show that our approach outperforms prior work in explained variance, fairness, and runtime.

----

## [827] Augmentation-Free Self-Supervised Learning on Graphs

**Authors**: *Namkyeong Lee, Junseok Lee, Chanyoung Park*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20700](https://doi.org/10.1609/aaai.v36i7.20700)

**Abstract**:

Inspired by the recent success of self-supervised methods applied on images, self-supervised learning on graph structured data has seen rapid growth especially centered on augmentation-based contrastive methods. However, we argue that without carefully designed augmentation techniques, augmentations on graphs may behave arbitrarily in that the underlying semantics of graphs can drastically change. As a consequence, the performance of existing augmentation-based methods is highly dependent on the choice of augmentation scheme, i.e., augmentation hyperparameters and combinations of augmentation. In this paper, we propose a novel augmentation-free self-supervised learning framework for graphs, named AFGRL. Specifically, we generate an alternative view of a graph by discovering nodes that share the local structural information and the global semantics with the graph. Extensive experiments towards various node-level tasks, i.e., node classification, clustering, and similarity search on various real-world datasets demonstrate the superiority of AFGRL. The source code for AFGRL is available at https://github.com/Namkyeong/AFGRL.

----

## [828] Fast and Robust Online Inference with Stochastic Gradient Descent via Random Scaling

**Authors**: *Sokbae Lee, Yuan Liao, Myung Hwan Seo, Youngki Shin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20701](https://doi.org/10.1609/aaai.v36i7.20701)

**Abstract**:

We develop a new method of online inference for a vector of parameters estimated by the Polyak-Ruppert averaging procedure of stochastic gradient descent (SGD) algorithms. We leverage insights from time series regression in econometrics and construct asymptotically pivotal statistics via random scaling. Our approach is fully operational with online data and is rigorously underpinned by a functional central limit theorem. Our proposed inference method has a couple of key advantages over the existing methods. First, the test statistic is computed in an online fashion with only SGD iterates and the critical values can be obtained without any resampling methods, thereby allowing for efficient implementation suitable for massive online data. Second, there is no need to estimate the asymptotic variance and our inference method is shown to be robust to changes in the tuning parameters for SGD algorithms in simulation experiments with synthetic data.

----

## [829] Diverse, Global and Amortised Counterfactual Explanations for Uncertainty Estimates

**Authors**: *Dan Ley, Umang Bhatt, Adrian Weller*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20702](https://doi.org/10.1609/aaai.v36i7.20702)

**Abstract**:

To interpret uncertainty estimates from differentiable probabilistic models, recent work has proposed generating a single Counterfactual Latent Uncertainty Explanation (CLUE) for a given data point where the model is uncertain. We broaden the exploration to examine δ-CLUE, the set of potential CLUEs within a δ ball of the original input in latent space. We study the diversity of such sets and find that many CLUEs are redundant; as such, we propose DIVerse CLUE (∇-CLUE), a set of CLUEs which each propose a distinct explanation as to how one can decrease the uncertainty associated with an input. We then further propose GLobal AMortised CLUE (GLAM-CLUE), a distinct, novel method which learns amortised mappings that apply to specific groups of uncertain inputs, taking them and efficiently transforming them in a single function call into inputs for which a model will be certain. Our experiments show that δ-CLUE, ∇-CLUE, and GLAM-CLUE all address shortcomings of CLUE and provide beneficial explanations of uncertainty estimates to practitioners.

----

## [830] Invariant Information Bottleneck for Domain Generalization

**Authors**: *Bo Li, Yifei Shen, Yezhen Wang, Wenzhen Zhu, Colorado Reed, Dongsheng Li, Kurt Keutzer, Han Zhao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20703](https://doi.org/10.1609/aaai.v36i7.20703)

**Abstract**:

Invariant risk minimization (IRM) has recently emerged as a promising alternative for domain generalization. Nevertheless, the loss function is difficult to optimize for nonlinear classifiers and the original optimization objective could fail when pseudo-invariant features and geometric skews exist. Inspired by IRM, in this paper we propose a novel formulation for domain generalization, dubbed invariant information bottleneck (IIB). IIB aims at minimizing invariant risks for nonlinear classifiers and simultaneously mitigating the impact of pseudo-invariant features and geometric skews. Specifically, we first present a novel formulation for invariant causal prediction via mutual information. Then we adopt the variational formulation of the mutual information to develop a tractable loss function for nonlinear classifiers. To overcome the failure modes of IRM, we propose to minimize the mutual information between the inputs and the corresponding representations. IIB significantly outperforms IRM on synthetic datasets, where the pseudo-invariant features and geometric skews occur, showing the effectiveness of proposed formulation in overcoming failure modes of IRM. Furthermore, experiments on DomainBed show that IIB outperforms 13 baselines by 0.9% on average across 7 real datasets.

----

## [831] Chunk Dynamic Updating for Group Lasso with ODEs

**Authors**: *Diyang Li, Bin Gu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20704](https://doi.org/10.1609/aaai.v36i7.20704)

**Abstract**:

Group Lasso is an important sparse regression method in machine learning which encourages selecting key explanatory factors in a grouped manner because of the use of L-2,1 norm. In real-world learning tasks, some chunks of data would be added into or removed from the training set in sequence due to the existence of new or obsolete historical data, which is normally called dynamic or lifelong learning scenario. However, most of existing algorithms of group Lasso are limited to offline updating, and only one is online algorithm which can only handle newly added samples inexactly. Due to the complexity of L-2,1 norm, how to achieve accurate chunk incremental and decremental learning efficiently for group Lasso is still an open question. To address this challenging problem, in this paper, we propose a novel accurate dynamic updating algorithm for group Lasso by utilizing the technique of Ordinary Differential Equations (ODEs), which can incorporate or eliminate a chunk of samples from original training set without retraining the model from scratch. Specifically, we introduce a new formulation to reparameterize the adjustment procedures of chunk incremental and decremental learning simultaneously. Based on the new formulation, we propose a path following algorithm for group Lasso regarding to the adjustment parameter. Importantly, we prove that our path following algorithm can exactly track the piecewise smooth solutions thanks to the technique of ODEs, so that the accurate chunk incremental and decremental learning can be achieved. Extensive experimental results not only confirm the effectiveness of proposed algorithm for the chunk incremental and decremental learning, but also validate its efficiency compared to the existing offline and online algorithms.

----

## [832] Policy Learning for Robust Markov Decision Process with a Mismatched Generative Model

**Authors**: *Jialian Li, Tongzheng Ren, Dong Yan, Hang Su, Jun Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20705](https://doi.org/10.1609/aaai.v36i7.20705)

**Abstract**:

In high-stake scenarios like medical treatment and auto-piloting, it's risky or even infeasible to collect online experimental data to train the agent. Simulation-based training can alleviate this issue, but may suffer from its inherent mismatches from the simulator and real environment. It is therefore imperative to utilize the simulator to learn a robust policy for the real-world deployment. In this work, we consider policy learning for Robust Markov Decision Processes (RMDP), where the agent tries to seek a robust policy with respect to unexpected perturbations on the environments. Specifically, we focus on the setting where the training environment can be characterized as a generative model and a constrained perturbation can be added to the model during testing. Our goal is to identify a near-optimal robust policy for the perturbed testing environment, which introduces additional technical difficulties as we need to simultaneously estimate the training environment uncertainty from samples and find the worst-case perturbation for testing. To solve this issue, we propose a generic method which formalizes the perturbation as an opponent to obtain a two-player zero-sum game, and further show that the Nash Equilibrium corresponds to the robust policy. We prove that, with a polynomial number of samples from the generative model, our algorithm can find a near-optimal robust policy with a high probability. Our method is able to deal with general perturbations under some mild assumptions and can also be extended to more complex problems like robust partial observable Markov decision process, thanks to the game-theoretical formulation.

----

## [833] A Fully Single Loop Algorithm for Bilevel Optimization without Hessian Inverse

**Authors**: *Junyi Li, Bin Gu, Heng Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20706](https://doi.org/10.1609/aaai.v36i7.20706)

**Abstract**:

In this paper, we propose a novel Hessian inverse free Fully Single Loop Algorithm (FSLA) for bilevel optimization problems. Classic algorithms for bilevel optimization admit a double loop structure which is computationally expensive. Recently, several single loop algorithms have been proposed with optimizing the inner and outer variable alternatively. However, these algorithms not yet achieve fully single loop. As they overlook the loop needed to evaluate the hyper-gradient for a given inner and outer state. In order to develop a fully single loop algorithm, we first study the structure of the hyper-gradient and identify a general approximation formulation of hyper-gradient computation that encompasses several previous common approaches, e.g. back-propagation through time, conjugate gradient, etc. Based on this formulation, we introduce a new state variable to maintain the historical hyper-gradient information. Combining our new formulation with the alternative update of the inner and outer variables, we propose an efficient fully single loop algorithm. We theoretically show that the error generated by the new state can be bounded and our algorithm converges. Finally, we verify the efficacy our algorithm empirically through multiple bilevel optimization based machine learning tasks. A long version of this paper can be found in: https://arxiv.org/abs/2112.04660.

----

## [834] A Hybrid Causal Structure Learning Algorithm for Mixed-Type Data

**Authors**: *Yan Li, Rui Xia, Chunchen Liu, Liang Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20707](https://doi.org/10.1609/aaai.v36i7.20707)

**Abstract**:

Inferring the causal structure of a set of random variables is a crucial problem in many disciplines of science. Over the past two decades, various approaches have been pro- posed for causal discovery from observational data. How- ever, most of the existing methods are designed for either purely discrete or continuous data, which limit their practical usage. In this paper, we target the problem of causal structure learning from observational mixed-type data. Although there are a few methods that are able to handle mixed-type data, they suffer from restrictions, such as linear assumption and poor scalability. To overcome these weaknesses, we formulate the causal mechanisms via mixed structure equation model and prove its identifiability under mild conditions. A novel locally consistent score, named CVMIC, is proposed for causal directed acyclic graph (DAG) structure learning. Moreover, we propose an efficient conditional independence test, named MRCIT, for mixed-type data, which is used in causal skeleton learning and final pruning to further improve the computational efficiency and precision of our model. Experimental results on both synthetic and real-world data demonstrate that our proposed hybrid model outperforms the other state-of-the-art methods. Our source code is available at https://github.com/DAMO-DI-ML/AAAI2022-HCM.

----

## [835] Sharp Analysis of Random Fourier Features in Classification

**Authors**: *Zhu Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20708](https://doi.org/10.1609/aaai.v36i7.20708)

**Abstract**:

We study the theoretical properties of random Fourier features classification with Lipschitz continuous loss functions such as support vector machine and logistic regression. Utilizing the regularity condition, we show for the first time that random Fourier features classification can achieve O(1/n^0.5) learning rate with only O(n^0.5) features, as opposed to O(n) features suggested by previous results. Our study covers the standard feature sampling method for which we reduce the number of features required, as well as a problem-dependent sampling method which further reduces the number of features while still keeping the optimal generalization property. Moreover, we prove that the random Fourier features classification can obtain a fast O(1/n) learning rate for both sampling schemes under Massart's low noise assumption. Our results demonstrate the potential effectiveness of random Fourier features approximation in reducing the computational complexity (roughly from O(n^3) in time and O(n^2) in space to O(n^2) and O(n^1.5) respectively) without having to trade-off the statistical prediction accuracy. In addition, the achieved trade-off in our analysis is at least the same as the optimal results in the literature under the worst case scenario and significantly improves the optimal results under benign regularity conditions.

----

## [836] Zeroth-Order Optimization for Composite Problems with Functional Constraints

**Authors**: *Zichong Li, Pin-Yu Chen, Sijia Liu, Songtao Lu, Yangyang Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20709](https://doi.org/10.1609/aaai.v36i7.20709)

**Abstract**:

In many real-world problems, first-order (FO) derivative evaluations are too expensive or even inaccessible. For solving these problems, zeroth-order (ZO) methods that only need function evaluations are often more efficient than FO methods or sometimes the only options. In this paper, we propose a novel zeroth-order inexact augmented Lagrangian method (ZO-iALM) to solve black-box optimization problems, which involve a composite (i.e., smooth+nonsmooth) objective and functional constraints. This appears to be the first work that develops an iALM-based ZO method for functional constrained optimization and meanwhile achieves query complexity results matching the best-known FO complexity results up to a factor of variable dimension. With an extensive experimental study, we show the effectiveness of our method. The applications of our method span from classical optimization problems to practical machine learning examples such as resource allocation in sensor networks and adversarial example generation.

----

## [837] Robust Graph-Based Multi-View Clustering

**Authors**: *Weixuan Liang, Xinwang Liu, Sihang Zhou, Jiyuan Liu, Siwei Wang, En Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20710](https://doi.org/10.1609/aaai.v36i7.20710)

**Abstract**:

Graph-based multi-view clustering (G-MVC) constructs a graphical representation of each view and then fuses them to a unified graph for clustering. Though demonstrating promising clustering performance in various applications, we observe that their formulations are usually non-convex, leading to a local optimum. In this paper, we propose a novel MVC algorithm termed robust graph-based multi-view clustering (RG-MVC) to address this issue. In particular, we define a min-max formulation for robust learning and then rewrite it as a convex and differentiable objective function whose convexity and differentiability are carefully proved. Thus, we can efficiently solve the resultant problem using a reduced gradient descent algorithm, and the corresponding solution is guaranteed to be globally optimal. As a consequence, although our algorithm is free of hyper-parameters, it has shown good robustness against noisy views. Extensive experiments on benchmark datasets verify the superiority of the proposed method against the compared state-of-the-art algorithms. Our codes and appendix are available at https://github.com/wx-liang/RG-MVC.

----

## [838] Conditional Local Convolution for Spatio-Temporal Meteorological Forecasting

**Authors**: *Haitao Lin, Zhangyang Gao, Yongjie Xu, Lirong Wu, Ling Li, Stan Z. Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20711](https://doi.org/10.1609/aaai.v36i7.20711)

**Abstract**:

Spatio-temporal forecasting is challenging attributing to the high nonlinearity in temporal dynamics as well as complex location-characterized patterns in spatial domains, especially in fields like weather forecasting. Graph convolutions are usually used for modeling the spatial dependency in meteorology to handle the irregular distribution of sensors' spatial location.  In this work, a novel graph-based convolution for imitating the meteorological flows is proposed to capture the local spatial patterns. Based on the assumption of smoothness of location-characterized patterns, we propose conditional local convolution whose shared kernel on nodes' local space is approximated by feedforward networks, with local representations of coordinate obtained by horizon maps into cylindrical-tangent space as its input. The established united standard of local coordinate system preserves the orientation on geography. We further propose the distance and orientation scaling terms to reduce the impacts of irregular spatial distribution. The convolution is embedded in a Recurrent Neural Network architecture to model the temporal dynamics, leading to the Conditional Local Convolution Recurrent Network (CLCRN). Our model is evaluated on real-world weather benchmark datasets, achieving state-of-the-art performance with obvious improvements. We conduct further analysis on local pattern visualization, model's framework choice, advantages of horizon maps and etc. The source code is available at https://github.com/BIRD-TAO/CLCRN.

----

## [839] On the Use of Unrealistic Predictions in Hundreds of Papers Evaluating Graph Representations

**Authors**: *Li-Chung Lin, Cheng-Hung Liu, Chih-Ming Chen, Kai-Chin Hsu, I-Feng Wu, Ming-Feng Tsai, Chih-Jen Lin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20712](https://doi.org/10.1609/aaai.v36i7.20712)

**Abstract**:

Prediction using the ground truth sounds like an oxymoron in machine learning. However, such an unrealistic setting was used in hundreds, if not thousands of papers in the area of finding graph representations. To evaluate the multi-label problem of node classification by using the obtained representations, many works assume that the number of labels of each test instance is known in the prediction stage. In practice such ground truth information is rarely available, but we point out that such an inappropriate setting is now ubiquitous in this research area. We detailedly investigate why the situation occurs. Our analysis indicates that with unrealistic information, the performance is likely over-estimated. To see why suitable predictions were not used, we identify difficulties in applying some multi-label techniques. For the use in future studies, we propose simple and effective settings without using practically unknown information. Finally, we take this chance to compare major graph representation learning methods on multi-label node classification.

----

## [840] Deep Unsupervised Hashing with Latent Semantic Components

**Authors**: *Qinghong Lin, Xiaojun Chen, Qin Zhang, Shaotian Cai, Wenzhe Zhao, Hongfa Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20713](https://doi.org/10.1609/aaai.v36i7.20713)

**Abstract**:

Deep unsupervised hashing has been appreciated in the regime of image retrieval.  However, most prior arts failed to detect the semantic components and their relationships behind the images, which makes them lack discriminative power. To make up the defect, we propose a novel Deep Semantic Components Hashing (DSCH), which involves a common sense that an image normally contains a bunch of semantic components with homology and co-occurrence relationships. Based on this prior, DSCH regards the semantic components as latent variables under the Expectation-Maximization framework and designs a two-step iterative algorithm with the objective of maximum likelihood of training data. Firstly, DSCH constructs a semantic component structure by uncovering the fine-grained semantics components of images with a Gaussian Mixture Modal~(GMM), where an image is represented as a mixture of multiple components, and the semantics co-occurrence are exploited. Besides, coarse-grained semantics components, are discovered by considering the homology relationships between fine-grained components, and the hierarchy organization is then constructed. Secondly, DSCH makes the images close to their semantic component centers at both fine-grained and coarse-grained levels, and also makes the images share similar semantic components close to each other. Extensive experiments on three benchmark datasets demonstrate that the proposed hierarchical semantic components indeed facilitate the hashing model to achieve superior performance.

----

## [841] SCRIB: Set-Classifier with Class-Specific Risk Bounds for Blackbox Models

**Authors**: *Zhen Lin, Lucas Glass, M. Brandon Westover, Cao Xiao, Jimeng Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20714](https://doi.org/10.1609/aaai.v36i7.20714)

**Abstract**:

Despite deep learning (DL) success in classification problems, DL classifiers do not provide a sound mechanism to decide when to refrain from predicting. Recent works tried to control the overall prediction risk with classification with rejection options. However, existing works overlook the different significance of different classes. We introduce Set-classifier with class-specific RIsk Bounds (SCRIB) to tackle this problem, assigning multiple labels to each example. Given the output of a black-box model on the validation set, SCRIB constructs a set-classifier that controls the class-specific prediction risks. The key idea is to reject when the set classifier returns more than one label. We validated SCRIB on several medical applications, including sleep staging on electroencephalogram(EEG) data, X-ray COVID image classification, and atrial fibrillation detection based on electrocardiogram (ECG) data.SCRIB obtained desirable class-specific risks, which are 35%-88% closer to the target risks than baseline methods.

----

## [842] RareGAN: Generating Samples for Rare Classes

**Authors**: *Zinan Lin, Hao Liang, Giulia Fanti, Vyas Sekar*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20715](https://doi.org/10.1609/aaai.v36i7.20715)

**Abstract**:

We study the problem of learning generative adversarial networks (GANs) for a rare class of an unlabeled dataset subject to a labeling budget. This problem is motivated from practical applications in domains including security (e.g., synthesizing packets for DNS amplification attacks), systems and networking (e.g., synthesizing workloads that trigger high resource usage), and machine learning (e.g., generating images from a rare class). Existing approaches are unsuitable, either requiring fully-labeled datasets or sacrificing the fidelity of the rare class for that of the common classes. We propose RareGAN, a novel synthesis of three key ideas: (1) extending conditional GANs to use labelled and unlabelled data for better generalization; (2) an active learning approach that requests the most useful labels; and (3) a weighted loss function to favor learning the rare class. We show that RareGAN achieves a better fidelity-diversity tradeoff on the rare class than prior work across different applications, budgets, rare class fractions, GAN losses, and architectures.

----

## [843] Conjugated Discrete Distributions for Distributional Reinforcement Learning

**Authors**: *Björn Lindenberg, Jonas Nordqvist, Karl-Olof Lindahl*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20716](https://doi.org/10.1609/aaai.v36i7.20716)

**Abstract**:

In this work we continue to build upon recent advances in reinforcement learning for finite Markov processes. A common approach among previous existing algorithms, both single-actor and distributed, is to either clip rewards or to apply a transformation method on Q-functions to handle a large variety of magnitudes in real discounted returns. We theoretically show that one of the most successful methods may not yield an optimal policy if we have a non-deterministic process. As a solution, we argue that distributional reinforcement learning lends itself to remedy this situation completely. By the introduction of a conjugated distributional operator we may handle a large class of transformations for real returns with guaranteed theoretical convergence. We propose an approximating single-actor algorithm based on this operator that trains agents directly on unaltered rewards using a proper distributional metric given by the Cramér distance. To evaluate its performance in a stochastic setting we train agents on a suite of 55 Atari 2600 games using sticky-actions and obtain state-of-the-art performance compared to other well-known algorithms in the Dopamine framework.

----

## [844] Lifelong Hyper-Policy Optimization with Multiple Importance Sampling Regularization

**Authors**: *Pierre Liotet, Francesco Vidaich, Alberto Maria Metelli, Marcello Restelli*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20717](https://doi.org/10.1609/aaai.v36i7.20717)

**Abstract**:

Learning in a lifelong setting, where the dynamics continually evolve, is a hard challenge for current reinforcement learning algorithms. Yet this would be a much needed feature for practical applications. 
 In this paper, we propose an approach which learns a hyper-policy, whose input is time, that outputs the parameters of the policy to be queried at that time. 
 This hyper-policy is trained to maximize the estimated future performance, efficiently reusing past data by means of importance sampling, at the cost of introducing a controlled bias. We combine the future performance estimate with the past performance to mitigate catastrophic forgetting.
 To avoid overfitting the collected data, we derive a differentiable variance bound that we embed as a penalization term. Finally, we empirically validate our approach, in comparison with state-of-the-art algorithms, on realistic environments, including water resource management and trading.

----

## [845] Learning Parameterized Task Structure for Generalization to Unseen Entities

**Authors**: *Anthony Z. Liu, Sungryull Sohn, Mahdi Qazwini, Honglak Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20718](https://doi.org/10.1609/aaai.v36i7.20718)

**Abstract**:

Real world tasks are hierarchical and compositional. Tasks can be composed of multiple subtasks (or sub-goals) that are dependent on each other. These subtasks are defined in terms of entities (e.g., "apple", "pear") that can be recombined to form new subtasks (e.g., "pickup apple", and "pickup pear"). To solve these tasks efficiently, an agent must infer subtask dependencies (e.g. an agent must execute "pickup apple" before "place apple in pot"), and generalize the inferred dependencies to new subtasks (e.g. "place apple in pot" is similar to "place apple in pan"). Moreover, an agent may also need to solve unseen tasks, which can involve unseen entities. To this end, we formulate parameterized subtask graph inference (PSGI), a method for modeling subtask dependencies using first-order logic with factored entities. To facilitate this, we learn parameter attributes in a zero-shot manner, which are used as quantifiers (e.g. is_pickable(X)) for the factored subtask graph. We show this approach accurately learns the latent structure on hierarchical and compositional tasks more efficiently than prior work, and show PSGI can generalize by modelling structure on subtasks unseen during adaptation.

----

## [846] Stationary Diffusion State Neural Estimation for Multiview Clustering

**Authors**: *Chenghua Liu, Zhuolin Liao, Yixuan Ma, Kun Zhan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20719](https://doi.org/10.1609/aaai.v36i7.20719)

**Abstract**:

Although many graph-based clustering methods attempt to model the stationary diffusion state in their objectives, their performance limits to using a predefined graph. We argue that the estimation of the stationary diffusion state can be achieved by gradient descent over neural networks. We specifically design the Stationary Diffusion State Neural Estimation (SDSNE) to exploit multiview structural graph information for co-supervised learning. We explore how to design a graph neural network specially for unsupervised multiview learning and integrate multiple graphs into a unified consensus graph by a shared self-attentional module. The view-shared self-attentional module utilizes the graph structure to learn a view-consistent global graph. Meanwhile, instead of using auto-encoder in most unsupervised learning graph neural networks, SDSNE uses a co-supervised strategy with structure information to supervise the model learning. The co-supervised strategy as the loss function guides SDSNE in achieving the stationary state. With the help of the loss and the self-attentional module, we learn to obtain a graph in which nodes in each connected component fully connect by the same weight. Experiments on several multiview datasets demonstrate effectiveness of SDSNE in terms of six clustering evaluation metrics.

----

## [847] Deep Amortized Relational Model with Group-Wise Hierarchical Generative Process

**Authors**: *Huafeng Liu, Tong Zhou, Jiaqi Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20720](https://doi.org/10.1609/aaai.v36i7.20720)

**Abstract**:

In this paper, we propose Deep amortized Relational Model (DaRM) with group-wise hierarchical generative process for community discovery and link prediction on relational data (e.g., graph, network). It provides an efficient neural relational model architecture by grouping nodes in a group-wise view rather than node-wise or edge-wise view. DaRM simultaneously learns what makes a group, how to divide nodes into groups, and how to adaptively control the number of groups. The dedicated group generative process is able to sufficiently exploit pair-wise or higher-order interactions between data points in both inter-group and intra-group, which is useful to sufficiently mine the hidden structure among data. A series of experiments have been conducted on both synthetic and real-world datasets. The experimental results demonstrated that DaRM can obtain high performance on both community detection and link prediction tasks.

----

## [848] Learn Goal-Conditioned Policy with Intrinsic Motivation for Deep Reinforcement Learning

**Authors**: *Jinxin Liu, Donglin Wang, Qiangxing Tian, Zhengyu Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20721](https://doi.org/10.1609/aaai.v36i7.20721)

**Abstract**:

It is of significance for an agent to autonomously explore the environment and learn a widely applicable and general-purpose goal-conditioned policy that can achieve diverse goals including images and text descriptions. Considering such perceptually-specific goals, one natural approach is to reward the agent with a prior non-parametric distance over the embedding spaces of states and goals. However, this may be infeasible in some situations, either because it is unclear how to choose suitable measurement, or because embedding (heterogeneous) goals and states is non-trivial. The key insight of this work is that we introduce a latent-conditioned policy to provide goals and intrinsic rewards for learning the goal-conditioned policy. As opposed to directly scoring current states with regards to goals, we obtain rewards by scoring current states with associated latent variables. We theoretically characterize the connection between our unsupervised objective and the multi-goal setting, and empirically demonstrate the effectiveness of our proposed method which substantially outperforms prior techniques in a variety of tasks.

----

## [849] Transformer with Memory Replay

**Authors**: *Rui Liu, Barzan Mozafari*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20722](https://doi.org/10.1609/aaai.v36i7.20722)

**Abstract**:

Transformers achieve state-of-the-art performance for natural language processing tasks by pre-training on large-scale text corpora. They are extremely compute-intensive and have very high sample complexity. Memory replay is a mechanism that remembers and reuses past examples by saving to and replaying from a memory buffer. It has been successfully used in reinforcement learning and GANs due to better sample efficiency. In this paper, we propose Transformer with Memory Replay, which integrates memory replay with transformer, making transformer more sample efficient. Experiments on GLUE and SQuAD benchmark datasets showed that Transformer with Memory Replay can achieve at least 1% point increase compared to the baseline transformer model when pre-trained with the same number of examples. Further, by adopting a careful design that reduces the wall-clock time overhead of memory replay, we also empirically achieve a better runtime efficiency.

----

## [850] Efficient One-Pass Multi-View Subspace Clustering with Consensus Anchors

**Authors**: *Suyuan Liu, Siwei Wang, Pei Zhang, Kai Xu, Xinwang Liu, Changwang Zhang, Feng Gao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20723](https://doi.org/10.1609/aaai.v36i7.20723)

**Abstract**:

Multi-view subspace clustering (MVSC) optimally integrates multiple graph structure information to improve clustering performance. Recently, many anchor-based variants are proposed to reduce the computational complexity of MVSC. Though achieving considerable acceleration, we observe that most of them adopt fixed anchor points separating from the subsequential anchor graph construction, which may adversely affect the clustering performance. In addition, post-processing is required to generate discrete clustering labels with additional time consumption. To address these issues, we propose a scalable and parameter-free MVSC method to directly output the clustering labels with optimal anchor graph, termed as Efficient One-pass Multi-view Subspace Clustering with Consensus Anchors (EOMSC-CA). Specially, we combine anchor learning and graph construction into a uniform framework to boost clustering performance. Meanwhile, by imposing a graph connectivity constraint, our algorithm directly outputs the clustering labels without any post-processing procedures as previous methods do. Our proposed EOMSC-CA is proven to be linear complexity respecting to the data size. The superiority of our EOMSC-CA over the effectiveness and efficiency is demonstrated by extensive experiments. Our code is publicly available at https://github.com/Tracesource/EOMSC-CA.

----

## [851] Trusted Multi-View Deep Learning with Opinion Aggregation

**Authors**: *Wei Liu, Xiaodong Yue, Yufei Chen, Thierry Denoeux*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20724](https://doi.org/10.1609/aaai.v36i7.20724)

**Abstract**:

Multi-view deep learning is performed based on the deep fusion of data from multiple sources, i.e. data with multiple views. However, due to the property differences and inconsistency of data sources, the deep learning results based on the fusion of multi-view data may be uncertain and unreliable. It is required to reduce the uncertainty in data fusion and implement the trusted multi-view deep learning. Aiming at the problem, we revisit the multi-view learning from the perspective of opinion aggregation and thereby devise a trusted multi-view deep learning method. Within this method, we adopt evidence theory to formulate the uncertainty of opinions as learning results from different data sources and measure the uncertainty of opinion aggregation as multi-view learning results through evidence accumulation. We prove that accumulating the evidences from multiple data views will decrease the uncertainty in multi-view deep learning and facilitate to achieve the trusted learning results. Experiments on various kinds of multi-view datasets verify the reliability and robustness of the proposed multi-view deep learning method.

----

## [852] Graph Convolutional Networks with Dual Message Passing for Subgraph Isomorphism Counting and Matching

**Authors**: *Xin Liu, Yangqiu Song*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20725](https://doi.org/10.1609/aaai.v36i7.20725)

**Abstract**:

Graph neural networks (GNNs) and message passing neural networks (MPNNs) have been proven to be expressive for subgraph structures in many applications. Some applications in heterogeneous graphs require explicit edge modeling, such as subgraph isomorphism counting and matching. However, existing message passing mechanisms are not designed well in theory. In this paper, we start from a particular edge-to-vertex transform and exploit the isomorphism property in the edge-to-vertex dual graphs. We prove that searching isomorphisms on the original graph is equivalent to searching on its dual graph. Based on this observation, we propose dual message passing neural networks (DMPNNs) to enhance the substructure representation learning in an asynchronous way for subgraph isomorphism counting and matching as well as unsupervised node classification. Extensive experiments demonstrate the robust performance of DMPNNs by combining both node and edge representation learning in synthetic and real heterogeneous graphs.

----

## [853] Deep Graph Clustering via Dual Correlation Reduction

**Authors**: *Yue Liu, Wenxuan Tu, Sihang Zhou, Xinwang Liu, Linxuan Song, Xihong Yang, En Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20726](https://doi.org/10.1609/aaai.v36i7.20726)

**Abstract**:

Deep graph clustering, which aims to reveal the underlying graph structure and divide the nodes into different groups, has attracted intensive attention in recent years. However, we observe that, in the process of node encoding, existing methods suffer from representation collapse which tends to map all data into the same representation. Consequently, the discriminative capability of the node representation is limited, leading to unsatisfied clustering performance. To address this issue, we propose a novel self-supervised deep graph clustering method termed Dual Correlation Reduction Network (DCRN) by reducing information correlation in a dual manner. Specifically, in our method, we first design a siamese network to encode samples. Then by forcing the cross-view sample correlation matrix and cross-view feature correlation matrix to approximate two identity matrices, respectively, we reduce the information correlation in the dual-level, thus improving the discriminative capability of the resulting features. Moreover, in order to alleviate representation collapse caused by over-smoothing in GCN, we introduce a propagation regularization term to enable the network to gain long-distance information with the shallow network structure. Extensive experimental results on six benchmark datasets demonstrate the effectiveness of the proposed DCRN against the existing state-of-the-art methods. The code of DCRN is available at https://github.com/yueliu1999/DCRN and a collection (papers, codes and, datasets) of deep graph clustering is shared at https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering on Github.

----

## [854] Optimistic Initialization for Exploration in Continuous Control

**Authors**: *Sam Lobel, Omer Gottesman, Cameron Allen, Akhil Bagaria, George Konidaris*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20727](https://doi.org/10.1609/aaai.v36i7.20727)

**Abstract**:

Optimistic initialization underpins many theoretically sound exploration schemes in tabular domains; however, in the deep function approximation setting, optimism can quickly disappear if initialized naively. We propose a framework for more effectively incorporating optimistic initialization into reinforcement learning for continuous control. Our approach uses metric information about the state-action space to estimate which transitions are still unexplored, and explicitly maintains the initial Q-value optimism for the corresponding state-action pairs. We also develop methods for efficiently approximating these training objectives, and for incorporating domain knowledge into the optimistic envelope to improve sample efficiency. We empirically evaluate these approaches on a variety of hard exploration problems in continuous control, where our method outperforms existing exploration techniques.

----

## [855] Fast and Data Efficient Reinforcement Learning from Pixels via Non-parametric Value Approximation

**Authors**: *Alexander Long, Alan Blair, Herke van Hoof*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20728](https://doi.org/10.1609/aaai.v36i7.20728)

**Abstract**:

We present Nonparametric Approximation of Inter-Trace returns (NAIT), a Reinforcement Learning algorithm for discrete action, pixel-based environments that is both highly sample and computation efficient. NAIT is a lazy-learning approach with an update that is equivalent to episodic Monte-Carlo on episode completion, but that allows the stable incorporation of rewards while an episode is ongoing. We make use of a fixed domain-agnostic representation, simple distance based exploration and a proximity graph-based lookup to facilitate extremely fast execution. We empirically evaluate NAIT on both the 26 and 57 game variants of ATARI100k where, despite its simplicity, it achieves competitive performance in the online setting with greater than 100x speedup in wall-time.

----

## [856] Frozen Pretrained Transformers as Universal Computation Engines

**Authors**: *Kevin Lu, Aditya Grover, Pieter Abbeel, Igor Mordatch*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20729](https://doi.org/10.1609/aaai.v36i7.20729)

**Abstract**:

We investigate the capability of a transformer pretrained on natural language to generalize to other modalities with minimal finetuning -- in particular, without finetuning of the self-attention and feedforward layers of the residual blocks. We consider such a model, which we call a Frozen Pretrained Transformer (FPT), and study finetuning it on a variety of sequence classification tasks spanning numerical computation, vision, and protein fold prediction. In contrast to prior works which investigate finetuning on the same modality as the pretraining dataset, we show that pretraining on natural language can improve performance and compute efficiency on non-language downstream tasks. Additionally, we perform an analysis of the architecture, comparing the performance of a random initialized transformer to a random LSTM. Combining the two insights, we find language-pretrained transformers can obtain strong performance on a variety of non-language tasks.

----

## [857] Adapt to Environment Sudden Changes by Learning a Context Sensitive Policy

**Authors**: *Fan-Ming Luo, Shengyi Jiang, Yang Yu, Zongzhang Zhang, Yi-Feng Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20730](https://doi.org/10.1609/aaai.v36i7.20730)

**Abstract**:

Dealing with real-world reinforcement learning (RL) tasks, we shall be aware that the environment may have sudden changes. We expect that a robust policy is able to handle such changes and adapt to the new environment rapidly.   
Context-based meta reinforcement learning aims at learning environment adaptable policies. These methods adopt a context encoder to perceive the environment on-the-fly, following which a contextual policy makes environment adaptive decisions according to the context. However, previous methods show lagged and unstable context extraction, which are hard to handle sudden changes well. This paper proposes an environment sensitive contextual policy learning (ESCP) approach, in order to improve both the sensitivity and the robustness of context encoding. ESCP is composed of three key components: variance minimization that forces a rapid and stable encoding of the environment context, relational matrix determinant maximization that avoids trivial solutions, and a history-truncated recurrent neural network model that avoids old memory interference. 
We use a grid-world task and 5 locomotion controlling tasks with changing parameters to empirically assess our algorithm. Experiment results show that in environments with both in-distribution and out-of-distribution parameter changes, ESCP can not only better recover the environment encoding, but also adapt more rapidly to the post-change environment (10x faster in the grid-world) while the return performance is kept or improved, compared with state-of-the-art meta RL methods.

----

## [858] Beyond Shared Subspace: A View-Specific Fusion for Multi-View Multi-Label Learning

**Authors**: *Gengyu Lyu, Xiang Deng, Yanan Wu, Songhe Feng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20731](https://doi.org/10.1609/aaai.v36i7.20731)

**Abstract**:

In multi-view multi-label learning (MVML), each instance is described by several heterogeneous feature representations and associated with multiple valid labels simultaneously. Although diverse MVML methods have been proposed over the last decade, most previous studies focus on leveraging the shared subspace across different views to represent the multi-view consensus information, while it is still an open issue whether such shared subspace representation is necessary when formulating the desired MVML model. In this paper, we propose a DeepGCN based View-Specific MVML method (D-VSM) which can bypass seeking for the shared subspace representation, and instead directly encoding the feature representation of each individual view through the deep GCN to couple with the information derived from the other views. Specifically, we first construct all instances under different feature representations into the corresponding feature graphs respectively, and then integrate them into a unified graph by integrating the different feature representations of each instance. Afterwards, the graph attention mechanism is adopted to aggregate and update all nodes on the unified graph to form structural representation for each instance, where both intra-view correlations and inter-view alignments have been jointly encoded to discover the underlying semantic relations. Finally, we derive a label confidence score for each instance by averaging the label confidence of its different feature representations with the multi-label soft margin loss. Extensive experiments have demonstrated that our proposed method significantly outperforms state-of-the-art methods.

----

## [859] Efficient Continuous Control with Double Actors and Regularized Critics

**Authors**: *Jiafei Lyu, Xiaoteng Ma, Jiangpeng Yan, Xiu Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20732](https://doi.org/10.1609/aaai.v36i7.20732)

**Abstract**:

How to obtain good value estimation is a critical problem in Reinforcement Learning (RL). Current value estimation methods in continuous control, such as DDPG and TD3, suffer from unnecessary over- or under- estimation. In this paper, we explore the potential of double actors, which has been neglected for a long time, for better value estimation in the continuous setting. First, we interestingly find that double actors improve the exploration ability of the agent. Next, we uncover the bias alleviation property of double actors in handling overestimation with single critic, and underestimation with double critics respectively. Finally, to mitigate the potentially pessimistic value estimate in double critics, we propose to regularize the critics under double actors architecture. Together, we present Double Actors Regularized Critics (DARC) algorithm. Extensive experiments on challenging continuous control benchmarks, MuJoCo and PyBullet, show that DARC significantly outperforms current baselines with higher average return and better sample efficiency.

----

## [860] Recursive Reasoning Graph for Multi-Agent Reinforcement Learning

**Authors**: *Xiaobai Ma, David Isele, Jayesh K. Gupta, Kikuo Fujimura, Mykel J. Kochenderfer*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20733](https://doi.org/10.1609/aaai.v36i7.20733)

**Abstract**:

Multi-agent reinforcement learning (MARL) provides an efficient way for simultaneously learning policies for multiple agents interacting with each other. However, in scenarios requiring complex interactions, existing algorithms can suffer from an inability to accurately anticipate the influence of self-actions on other agents. Incorporating an ability to reason about other agents' potential responses can allow an agent to formulate more effective strategies. This paper adopts a recursive reasoning model in a centralized-training-decentralized-execution framework to help learning agents better cooperate with or compete against others. 
 The proposed algorithm, referred to as the Recursive Reasoning Graph (R2G), shows state-of-the-art performance on multiple multi-agent particle and robotics games.

----

## [861] Sharp Restricted Isometry Property Bounds for Low-Rank Matrix Recovery Problems with Corrupted Measurements

**Authors**: *Ziye Ma, Yingjie Bi, Javad Lavaei, Somayeh Sojoudi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20734](https://doi.org/10.1609/aaai.v36i7.20734)

**Abstract**:

In this paper, we study a general low-rank matrix recovery problem with linear measurements corrupted by some noise. The objective is to understand under what conditions on the restricted isometry property (RIP) of the problem local search methods can find the ground truth with a small error. By analyzing the landscape of the non-convex problem, we first propose a global guarantee on the maximum distance between an arbitrary local minimizer and the ground truth under the assumption that the RIP constant is smaller than 1/2. We show that this distance shrinks to zero as the intensity of the noise reduces. Our new guarantee is sharp in terms of the RIP constant and is much stronger than the existing results. We then present a local guarantee for problems with an arbitrary RIP constant, which states that any local minimizer is either considerably close to the ground truth or far away from it. Next, we prove the strict saddle property, which guarantees the global convergence of the perturbed gradient descent method in polynomial time. The developed results demonstrate how the noise intensity and the RIP constant of the problem affect the landscape of the problem.

----

## [862] Cross-Lingual Adversarial Domain Adaptation for Novice Programming

**Authors**: *Ye Mao, Farzaneh Khoshnevisan, Thomas W. Price, Tiffany Barnes, Min Chi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20735](https://doi.org/10.1609/aaai.v36i7.20735)

**Abstract**:

Student modeling sits at the epicenter of adaptive learning technology. In contrast to the voluminous work on student modeling for well-defined domains such as algebra, there has been little research on student modeling in programming (SMP) due to data scarcity caused by the unbounded solution spaces of open-ended programming exercises. In this work, we focus on two essential SMP tasks: program classification and early prediction of student success and propose a Cross-Lingual Adversarial Domain Adaptation (CrossLing) framework that can leverage a large programming dataset to learn features that can improve SMP's build using a much smaller dataset in a different programming language. Our framework maintains one globally invariant latent representation across both datasets via an adversarial learning process, as well as allocating domain-specific models for each dataset to extract local latent representations that cannot and should not be united. By separating globally-shared representations from domain-specific representations, our framework outperforms existing state-of-the-art methods for both SMP tasks.

----

## [863] Hard to Forget: Poisoning Attacks on Certified Machine Unlearning

**Authors**: *Neil G. Marchant, Benjamin I. P. Rubinstein, Scott Alfeld*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20736](https://doi.org/10.1609/aaai.v36i7.20736)

**Abstract**:

The right to erasure requires removal of a user's information from data held by organizations, with rigorous interpretations extending to downstream products such as learned models. Retraining from scratch with the particular user's data omitted fully removes its influence on the resulting model, but comes with a high computational cost. Machine "unlearning" mitigates the cost incurred by full retraining: instead, models are updated incrementally, possibly only requiring retraining when approximation errors accumulate. Rapid progress has been made towards privacy guarantees on the indistinguishability of unlearned and retrained models, but current formalisms do not place practical bounds on computation. In this paper we demonstrate how an attacker can exploit this oversight, highlighting a novel attack surface introduced by machine unlearning. We consider an attacker aiming to increase the computational cost of data removal. We derive and empirically investigate a poisoning attack on certified machine unlearning where strategically designed training data triggers complete retraining when removed.

----

## [864] Exploring Safer Behaviors for Deep Reinforcement Learning

**Authors**: *Enrico Marchesini, Davide Corsi, Alessandro Farinelli*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20737](https://doi.org/10.1609/aaai.v36i7.20737)

**Abstract**:

We consider Reinforcement Learning (RL) problems where an agent attempts to maximize a reward signal while minimizing a cost function that models unsafe behaviors. Such formalization is addressed in the literature using constrained optimization on the cost, limiting the exploration and leading to a significant trade-off between cost and reward. In contrast, we propose a Safety-Oriented Search that complements Deep RL algorithms to bias the policy toward safety within an evolutionary cost optimization. We leverage evolutionary exploration benefits to design a novel concept of safe mutations that use visited unsafe states to explore safer actions. We further characterize the behaviors of the policies over desired specifics with a sample-based bound estimation, which makes prior verification analysis tractable in the training loop. Hence, driving the learning process towards safer regions of the policy space. Empirical evidence on the Safety Gym benchmark shows that we successfully avoid drawbacks on the return while improving the safety of the policy.

----

## [865] fGOT: Graph Distances Based on Filters and Optimal Transport

**Authors**: *Hermina Petric Maretic, Mireille El Gheche, Giovanni Chierchia, Pascal Frossard*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20738](https://doi.org/10.1609/aaai.v36i7.20738)

**Abstract**:

Graph comparison deals with identifying similarities and dissimilarities between graphs. A major obstacle is the unknown alignment of graphs, as well as the lack of accurate and inexpensive comparison metrics. In this work we introduce the filter graph distance. It is an optimal transport based distance which drives graph comparison through the probability distribution of filtered graph signals. 
 This creates a highly flexible distance, capable of prioritising different spectral information in observed graphs, offering a wide range of choices for a comparison metric. We tackle the problem of graph alignment by computing graph permutations that minimise our new filter distances, which implicitly solves the graph comparison problem. 
 We then propose a new approximate cost function that circumvents many computational difficulties inherent to graph comparison and permits the exploitation of fast algorithms such as mirror gradient descent, without grossly sacrificing the performance. We finally propose a novel algorithm derived from a stochastic version of mirror gradient descent, which accommodates the non-convexity of the alignment problem, offering a good trade-off between performance accuracy and speed. The experiments on graph alignment and classification show that the flexibility gained through filter graph distances can have a significant impact on performance, while the difference in speed offered by the approximation cost makes the framework applicable in practical settings.

----

## [866] When AI Difficulty Is Easy: The Explanatory Power of Predicting IRT Difficulty

**Authors**: *Fernando Martínez-Plumed, David Castellano Falcón, Carlos Monserrat Aranda, José Hernández-Orallo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20739](https://doi.org/10.1609/aaai.v36i7.20739)

**Abstract**:

One of challenges of artificial intelligence as a whole is robustness. Many issues such as adversarial examples, out of distribution performance, Clever Hans phenomena, and the wider areas of AI evaluation and explainable AI, have to do with the following question: Did the system fail because it is a hard instance or because something else? In this paper we address this question with a generic method for estimating IRT-based instance difficulty for a wide range of AI domains covering several areas, from supervised feature-based classification to automated reasoning. We show how to estimate difficulty systematically using off-the-shelf machine learning regression models. We illustrate the usefulness of this estimation for a range of applications.

----

## [867] Being Friends Instead of Adversaries: Deep Networks Learn from Data Simplified by Other Networks

**Authors**: *Simone Marullo, Matteo Tiezzi, Marco Gori, Stefano Melacci*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20740](https://doi.org/10.1609/aaai.v36i7.20740)

**Abstract**:

Amongst a variety of approaches aimed at making the learning procedure of neural networks more effective, the scientific community developed strategies to order the examples according to their estimated complexity, to distil knowledge from larger networks, or to exploit the principles behind adversarial machine learning.
 
A different idea has been recently proposed, named Friendly Training, which consists in altering the input data by adding an automatically estimated perturbation, with the goal of facilitating the learning process of a neural classifier. The transformation progressively fades-out as long as training proceeds, until it completely vanishes. In this work we revisit and extend this idea, introducing a radically different and novel approach inspired by the effectiveness of neural generators in the context of Adversarial Machine Learning. We propose an auxiliary multi-layer network that is responsible of altering the input data to make them easier to be handled by the classifier at the current stage of the training procedure. 
The auxiliary network is trained jointly with the neural classifier, thus intrinsically increasing the 'depth' of the classifier, and it is expected to spot general regularities in the data alteration process. The effect of the auxiliary network is progressively reduced up to the end of training, when it is fully dropped and the classifier is deployed for applications. We refer to this approach as Neural Friendly Training.
 
An extended experimental procedure involving several datasets and different neural architectures shows that Neural Friendly Training overcomes the originally proposed Friendly Training technique, improving the generalization of the classifier, especially in the case of noisy data.

----

## [868] An Experimental Design Approach for Regret Minimization in Logistic Bandits

**Authors**: *Blake Mason, Kwang-Sung Jun, Lalit Jain*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20741](https://doi.org/10.1609/aaai.v36i7.20741)

**Abstract**:

In this work we consider the problem of regret minimization for logistic bandits. The main challenge of logistic bandits is reducing the dependence on a potentially large problem dependent constant that can at worst scale exponentially with the norm of the unknown parameter vector. Previous works have applied self-concordance of the logistic function to remove this worst-case dependence providing regret guarantees that move the reduce the dependence on this worst case parameter to lower order terms with only polylogarithmic dependence on the main term and as well as linear dependence on the dimension of the unknown parameter. This work improves upon the prior art by 1) removing all scaling of the worst case term on the main term and 2) reducing the dependence on the dependence to scale with the square root of dimension in the fixed arm setting by employing an experimental design procedure. Our regret bound in fact takes a tighter instance (i.e., gap) dependent regret bound for the first time in logistic bandits. We also propose a new warmup sampling algorithm that can dramatically reduce the lower order term in the regret in general and prove that it can exponentially reduce the lower order term's dependency on the worst case parameter in some instances. Finally, we discuss the impact of the bias of the MLE on the logistic bandit problem in d dimensions, providing an example where d^2 lower order regret (cf., it is d for linear bandits) may not be improved as long as the MLE is used and how bias-corrected estimators may be used to make it closer to d.

----

## [869] Coordinate Descent on the Orthogonal Group for Recurrent Neural Network Training

**Authors**: *Estelle M. Massart, Vinayak Abrol*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20742](https://doi.org/10.1609/aaai.v36i7.20742)

**Abstract**:

We address the poor scalability of learning algorithms for orthogonal recurrent neural networks via the use of stochastic coordinate descent on the orthogonal group, leading to a cost per iteration that increases linearly with the number of recurrent states. This contrasts with the cubic dependency of typical feasible algorithms such as stochastic Riemannian gradient descent, which prohibits the use of big network architectures. Coordinate descent rotates successively two columns of the recurrent matrix. When the coordinate (i.e., indices of rotated columns) is selected uniformly at random at each iteration, we prove convergence of the algorithm under standard assumptions on the loss function, stepsize and minibatch noise. In addition, we numerically show that the Riemannian gradient has an approximately sparse structure. Leveraging this observation, we propose a variant of our proposed algorithm that relies on the Gauss-Southwell coordinate selection rule. Experiments on a benchmark recurrent neural network training problem show that the proposed approach is a very promising step towards the training of orthogonal recurrent neural networks with big architectures.

----

## [870] Curiosity-Driven Exploration via Latent Bayesian Surprise

**Authors**: *Pietro Mazzaglia, Ozan Çatal, Tim Verbelen, Bart Dhoedt*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20743](https://doi.org/10.1609/aaai.v36i7.20743)

**Abstract**:

The human intrinsic desire to pursue knowledge, also known as curiosity, is considered essential in the process of skill acquisition. With the aid of artificial curiosity, we could equip current techniques for control, such as Reinforcement Learning, with more natural exploration capabilities. A promising approach in this respect has consisted of using Bayesian surprise on model parameters, i.e. a metric for the difference between prior and posterior beliefs, to favour exploration. In this contribution, we propose to apply Bayesian surprise in a latent space representing the agent’s current understanding of the dynamics of the system, drastically reducing the computational costs. We extensively evaluate our method by measuring the agent's performance in terms of environment exploration, for continuous tasks, and looking at the game scores achieved, for video games. Our model is computationally cheap and compares positively with current state-of-the-art methods on several problems. We also investigate the effects caused by stochasticity in the environment, which is often a failure case for curiosity-driven agents. In this regime, the results suggest that our approach is resilient to stochastic transitions.

----

## [871] What Can We Learn Even from the Weakest? Learning Sketches for Programmatic Strategies

**Authors**: *Leandro C. Medeiros, David S. Aleixo, Levi H. S. Lelis*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20744](https://doi.org/10.1609/aaai.v36i7.20744)

**Abstract**:

In this paper we show that behavioral cloning can be used to learn effective sketches of programmatic strategies. We show that even the sketches learned by cloning the behavior of weak players can help the synthesis of programmatic strategies. This is because even weak players can provide helpful information, e.g., that a player must choose an action in their turn of the game. If behavioral cloning is not employed, the synthesizer needs to learn even the most basic information by playing the game, which can be computationally expensive. We demonstrate empirically the advantages of our sketch-learning approach with simulated annealing and UCT synthesizers. We evaluate our synthesizers in the games of Can't Stop and MicroRTS. The sketch-based synthesizers are able to learn stronger programmatic strategies than their original counterparts. Our synthesizers generate strategies of Can't Stop that defeat a traditional programmatic strategy for the game. They also synthesize strategies that defeat the best performing method from the latest MicroRTS competition.

----

## [872] Top-Down Deep Clustering with Multi-Generator GANs

**Authors**: *Daniel P. M. de Mello, Renato M. Assunção, Fabricio Murai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20745](https://doi.org/10.1609/aaai.v36i7.20745)

**Abstract**:

Deep clustering (DC) leverages the representation power of deep architectures to learn embedding spaces that are optimal for cluster analysis. This approach filters out low-level information irrelevant for clustering and has proven remarkably successful for high dimensional data spaces. Some DC methods employ Generative Adversarial Networks (GANs), motivated by the powerful latent representations these models are able to learn implicitly. In this work, we propose HC-MGAN, a new technique based on GANs with multiple generators (MGANs), which have not been explored for clustering. Our method is inspired by the observation that each generator of a MGAN tends to generate data that correlates with a sub-region of the real data distribution. We use this clustered generation to train a classifier for inferring from which generator a given image came from, thus providing a semantically meaningful clustering for the real distribution. Additionally, we design our method so that it is performed in a top-down hierarchical clustering tree, thus proposing the first hierarchical DC method, to the best of our knowledge. We conduct several experiments to evaluate the proposed method against recent DC methods, obtaining competitive results. Last, we perform an exploratory analysis of the hierarchical clustering tree that highlights how accurately it organizes the data in a hierarchy of semantically coherent patterns.

----

## [873] Temporal Knowledge Graph Completion Using Box Embeddings

**Authors**: *Johannes Messner, Ralph Abboud, Ismail Ilkan Ceylan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20746](https://doi.org/10.1609/aaai.v36i7.20746)

**Abstract**:

Knowledge graph completion is the task of inferring missing facts based on existing data in a knowledge graph. Temporal knowledge graph completion (TKGC) is an extension of this task to temporal knowledge graphs, where each fact is additionally associated with a time stamp. Current approaches for TKGC primarily build on existing embedding models which are developed for static knowledge graph completion, and extend these models to incorporate time, where the idea is to learn latent representations for entities, relations, and timestamps and then use the learned representations to predict missing facts at various time steps. In this paper, we propose BoxTE, a box embedding model for TKGC, building on the static knowledge graph embedding model BoxE. We show that BoxTE is fully expressive, and possesses strong inductive capacity in the temporal setting. We then empirically evaluate our model and show that it achieves state-of-the-art results on several TKGC benchmarks

----

## [874] An Evaluative Measure of Clustering Methods Incorporating Hyperparameter Sensitivity

**Authors**: *Siddhartha Mishra, Nicholas Monath, Michael Boratko, Ariel Kobren, Andrew McCallum*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20747](https://doi.org/10.1609/aaai.v36i7.20747)

**Abstract**:

Clustering algorithms are often evaluated using metrics which compare with ground-truth cluster assignments, such as Rand index and NMI. Algorithm performance may vary widely for different hyperparameters, however, and thus model selection based on optimal performance for these metrics is discordant with how these algorithms are applied in practice, where labels are unavailable and tuning is often more art than science. It is therefore desirable to compare clustering algorithms not only on their optimally tuned performance, but also some notion of how realistic it would be to obtain this performance in practice. We propose an evaluation of clustering methods capturing this ease-of-tuning by modeling the expected best clustering score under a given computation budget. To encourage the adoption of the proposed metric alongside classic clustering evaluations, we provide an extensible benchmarking framework. We perform an extensive empirical evaluation of our proposed metric on popular clustering algorithms over a large collection of datasets from different domains, and observe that our new metric leads to several noteworthy observations.

----

## [875] Simple Unsupervised Graph Representation Learning

**Authors**: *Yujie Mo, Liang Peng, Jie Xu, Xiaoshuang Shi, Xiaofeng Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20748](https://doi.org/10.1609/aaai.v36i7.20748)

**Abstract**:

In this paper, we propose a simple unsupervised graph representation learning method to conduct effective and efficient contrastive learning. Specifically, the proposed multiplet loss explores the complementary information between the structural information and neighbor information to enlarge the inter-class variation, as well as adds an upper bound loss to achieve the finite distance between positive embeddings and anchor embeddings for reducing the intra-class variation. As a result, both enlarging inter-class variation and reducing intra-class variation result in small generalization error, thereby obtaining an effective model. Furthermore, our method removes widely used data augmentation and discriminator from previous graph contrastive learning methods, meanwhile available to output low-dimensional embeddings, leading to an efficient model. Experimental results on various real-world datasets demonstrate the effectiveness and efficiency of our method, compared to state-of-the-art methods. The source codes are released at https://github.com/YujieMo/SUGRL.

----

## [876] The Role of Adaptive Optimizers for Honest Private Hyperparameter Selection

**Authors**: *Shubhankar Mohapatra, Sajin Sasy, Xi He, Gautam Kamath, Om Thakkar*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20749](https://doi.org/10.1609/aaai.v36i7.20749)

**Abstract**:

Hyperparameter optimization is a ubiquitous challenge in machine learning, and the performance of a trained model depends crucially upon their effective selection. While a rich set of tools exist for this purpose, there are currently no practical hyperparameter selection methods under the constraint of differential privacy (DP). We study honest hyperparameter selection for differentially private machine learning, in which the process of hyperparameter tuning is accounted for in the overall privacy budget. To this end, we i) show that standard composition tools outperform more advanced techniques in many settings, ii) empirically and theoretically demonstrate an intrinsic connection between the learning rate and clipping norm hyperparameters, iii) show that adaptive optimizers like DPAdam enjoy a significant advantage in the process of honest hyperparameter tuning, and iv) draw upon novel limiting behaviour of Adam in the DP setting to design a new and more efficient optimizer.

----

## [877] Learning Bayesian Networks in the Presence of Structural Side Information

**Authors**: *Ehsan Mokhtarian, Sina Akbari, Fateme Jamshidi, Jalal Etesami, Negar Kiyavash*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20750](https://doi.org/10.1609/aaai.v36i7.20750)

**Abstract**:

We study the problem of learning a Bayesian network (BN) of a set of variables when structural side information about the system is available. It is well known that learning the structure of a general BN is both computationally and statistically challenging. However, often in many applications, side information about the underlying structure can potentially reduce the learning complexity. In this paper, we develop a recursive constraint-based algorithm that efficiently incorporates such knowledge (i.e., side information) into the learning process. In particular, we study two types of structural side information about the underlying BN: (I) an upper bound on its clique number is known, or (II) it is diamond-free. We provide theoretical guarantees for the learning algorithms, including the worst-case number of tests required in each scenario. As a consequence of our work, we show that bounded treewidth BNs can be learned with polynomial complexity. Furthermore, we evaluate the performance and the scalability of our algorithms in both synthetic and real-world structures and show that they outperform the state-of-the-art structure learning algorithms.

----

## [878] Preemptive Image Robustification for Protecting Users against Man-in-the-Middle Adversarial Attacks

**Authors**: *Seungyong Moon, Gaon An, Hyun Oh Song*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20751](https://doi.org/10.1609/aaai.v36i7.20751)

**Abstract**:

Deep neural networks have become the driving force of modern image recognition systems. However, the vulnerability of neural networks against adversarial attacks poses a serious threat to the people affected by these systems. In this paper, we focus on a real-world threat model where a Man-in-the-Middle adversary maliciously intercepts and perturbs images web users upload online. This type of attack can raise severe ethical concerns on top of simple performance degradation. To prevent this attack, we devise a novel bi-level optimization algorithm that finds points in the vicinity of natural images that are robust to adversarial perturbations. Experiments on CIFAR-10 and ImageNet show our method can effectively robustify natural images within the given modification budget. We also show the proposed method can improve robustness when jointly used with randomized smoothing.

----

## [879] Provable Guarantees for Understanding Out-of-Distribution Detection

**Authors**: *Peyman Morteza, Yixuan Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20752](https://doi.org/10.1609/aaai.v36i7.20752)

**Abstract**:

Out-of-distribution (OOD) detection is important for deploying machine learning models in the real world, where test data from shifted distributions can naturally arise. While a plethora of algorithmic approaches have recently emerged for OOD detection, a critical gap remains in theoretical understanding. In this work, we develop an analytical framework that characterizes and unifies the theoretical understanding for OOD detection. Our analytical framework motivates a novel OOD detection method for neural networks, GEM, which demonstrates both theoretical and empirical superiority. In particular, on CIFAR-100 as in-distribution data, our method outperforms a competitive baseline by 16.57% (FPR95). Lastly, we formally provide provable guarantees and comprehensive analysis of our method, underpinning how various properties of data distribution affect the performance of OOD detection.

----

## [880] Constraint Sampling Reinforcement Learning: Incorporating Expertise for Faster Learning

**Authors**: *Tong Mu, Georgios Theocharous, David Arbour, Emma Brunskill*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20753](https://doi.org/10.1609/aaai.v36i7.20753)

**Abstract**:

Online reinforcement learning (RL) algorithms are often difficult to deploy in complex human-facing applications as they may learn slowly and have poor early performance. To address this, we introduce a practical algorithm for incorporating human insight to speed learning. Our algorithm, Constraint Sampling Reinforcement Learning (CSRL), incorporates prior domain knowledge as constraints/restrictions on the RL policy. It takes in multiple potential policy constraints to maintain robustness to misspecification of individual constraints while leveraging helpful ones to learn quickly. Given a base RL learning algorithm (ex. UCRL, DQN, Rainbow) we propose an upper confidence with elimination scheme that leverages the relationship between the constraints, and their observed performance, to adaptively switch among them. We instantiate our algorithm with DQN-type algorithms and UCRL as base algorithms, and evaluate our algorithm in four environments, including three simulators based on real data: recommendations, educational activity sequencing, and HIV treatment sequencing. In all cases, CSRL learns a good policy faster than baselines.

----

## [881] Unsupervised Reinforcement Learning in Multiple Environments

**Authors**: *Mirco Mutti, Mattia Mancassola, Marcello Restelli*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20754](https://doi.org/10.1609/aaai.v36i7.20754)

**Abstract**:

Several recent works have been dedicated to unsupervised reinforcement learning in a single environment, in which a policy is first pre-trained with unsupervised interactions, and then fine-tuned towards the optimal policy for several downstream supervised tasks defined over the same environment. Along this line, we address the problem of unsupervised reinforcement learning in a class of multiple environments, in which the policy is pre-trained with interactions from the whole class, and then fine-tuned for several tasks in any environment of the class. Notably, the problem is inherently multi-objective as we can trade off the pre-training objective between environments in many ways. In this work, we foster an exploration strategy that is sensitive to the most adverse cases within the class. Hence, we cast the exploration problem as the maximization of the mean of a critical percentile of the state visitation entropy induced by the exploration strategy over the class of environments. Then, we present a policy gradient algorithm, alphaMEPOL, to optimize the introduced objective through mediated interactions with the class. Finally, we empirically demonstrate the ability of the algorithm in learning to explore challenging classes of continuous environments and we show that reinforcement learning greatly benefits from the pre-trained exploration strategy w.r.t. learning from scratch.

----

## [882] Is Your Data Relevant?: Dynamic Selection of Relevant Data for Federated Learning

**Authors**: *Lokesh Nagalapatti, Ruhi Sharma Mittal, Ramasuri Narayanam*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20755](https://doi.org/10.1609/aaai.v36i7.20755)

**Abstract**:

Federated Learning (FL) is a machine learning paradigm in which multiple clients participate to collectively learn a global machine learning model at the central server. It is plausible that not all the data owned by each client is relevant to the server's learning objective. The updates incorporated from irrelevant data could be detrimental to the global model. The task of selecting relevant data is explored in traditional machine learning settings where the assumption is that all the data is available in one place. In FL settings, the data is distributed across multiple clients and the server can't introspect it. This precludes the application of traditional solutions to selecting relevant data here.
 In this paper, we propose an approach called Federated Learning with Relevant Data (FLRD), that facilitates clients to derive updates using relevant data. Each client learns a model called Relevant Data Selector (RDS) that is private to itself to do the selection. This in turn helps in building an effective global model.
 We perform experiments with multiple real-world datasets to demonstrate the efficacy of our solution. The results show (a) the capability of FLRD to identify relevant data samples at each client locally and (b) the superiority of the global model learned by FLRD over other baseline algorithms.

----

## [883] A Dynamic Meta-Learning Model for Time-Sensitive Cold-Start Recommendations

**Authors**: *Krishna Prasad Neupane, Ervine Zheng, Yu Kong, Qi Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20756](https://doi.org/10.1609/aaai.v36i7.20756)

**Abstract**:

We present a novel dynamic recommendation model that focuses on users who have interactions in the past but turn relatively inactive recently. Making effective recommendations to these time-sensitive cold-start users is critical to maintain the user base of a recommender system. Due to the sparse recent interactions, it is challenging to capture these users' current preferences precisely. Solely relying on their historical interactions may also lead to outdated recommendations misaligned with their recent interests. The proposed model leverages historical and current user-item interactions and dynamically factorizes a user's (latent) preference into time-specific and time-evolving representations that jointly affect user behaviors. These latent factors further interact with an optimized item embedding to achieve accurate and timely recommendations. Experiments over real-world data help demonstrate the effectiveness of the proposed time-sensitive cold-start recommendation model.

----

## [884] Out of Distribution Data Detection Using Dropout Bayesian Neural Networks

**Authors**: *André T. Nguyen, Fred Lu, Gary Lopez Munoz, Edward Raff, Charles Nicholas, James Holt*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20757](https://doi.org/10.1609/aaai.v36i7.20757)

**Abstract**:

We explore the utility of information contained within a dropout based Bayesian neural network (BNN) for the task of detecting out of distribution (OOD) data. We first show how previous attempts to leverage the randomized embeddings induced by the intermediate layers of a dropout BNN can fail due to the distance metric used. We introduce an alternative approach to measuring embedding uncertainty, and demonstrate how incorporating embedding uncertainty improves OOD data identification across three tasks: image classification, language classification, and malware detection.

----

## [885] Control-Oriented Model-Based Reinforcement Learning with Implicit Differentiation

**Authors**: *Evgenii Nikishin, Romina Abachi, Rishabh Agarwal, Pierre-Luc Bacon*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20758](https://doi.org/10.1609/aaai.v36i7.20758)

**Abstract**:

The shortcomings of maximum likelihood estimation in the context of model-based reinforcement learning have been highlighted by an increasing number of papers. When the model class is misspecified or has a limited representational capacity, model parameters with high likelihood might not necessarily result in high performance of the agent on a downstream control task. To alleviate this problem, we propose an end-to-end approach for model learning which directly optimizes the expected returns using implicit differentiation. We treat a value function that satisfies the Bellman optimality operator induced by the model as an implicit function of model parameters and show how to differentiate the function. We provide theoretical and empirical evidence highlighting the benefits of our approach in the model misspecification regime compared to likelihood-based methods.

----

## [886] Improving Evidential Deep Learning via Multi-Task Learning

**Authors**: *Dongpin Oh, Bonggun Shin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20759](https://doi.org/10.1609/aaai.v36i7.20759)

**Abstract**:

The Evidential regression network (ENet) estimates a continuous target and its predictive uncertainty without costly Bayesian model averaging. However, it is possible that the target is inaccurately predicted due to the gradient shrinkage problem of the original loss function of the ENet, the negative log marginal likelihood (NLL) loss. In this paper, the objective is to improve the prediction accuracy of the ENet while maintaining its efficient uncertainty estimation by resolving the gradient shrinkage problem. A multi-task learning (MTL) framework, referred to as MT-ENet, is proposed to accomplish this aim. In the MTL, we define the Lipschitz modified mean squared error (MSE) loss function as another loss and add it to the existing NLL loss. The Lipschitz modified MSE loss is designed to mitigate the gradient conflict with the NLL loss by dynamically adjusting its Lipschitz constant. By doing so, the Lipschitz MSE loss does not disturb the uncertainty estimation of the NLL loss. The MT-ENet enhances the predictive accuracy of the ENet without losing uncertainty estimation capability on the synthetic dataset and real-world benchmarks, including drug-target affinity (DTA) regression. Furthermore, the MT-ENet shows remarkable calibration and out-of-distribution detection capability on the DTA benchmarks.

----

## [887] Clustering Approach to Solve Hierarchical Classification Problem Complexity

**Authors**: *Aomar Osmani, Massinissa Hamidi, Pegah Alizadeh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20760](https://doi.org/10.1609/aaai.v36i7.20760)

**Abstract**:

In a large domain of classification problems for real applications, like human activity recognition, separable spaces between groups of concepts are easier to learn than each concept alone. This is because the search space biases required to separate groups of classes (or concepts) are more relevant than the ones needed to separate classes individually. For example, it is easier to learn the activities related to the body movements group (running, walking) versus "on-wheels" activities group (bicycling, driving a car), before learning more specific classes inside each of these groups. Despite the obvious interest of this approach, our theoretical analysis shows a high complexity for finding an exact solution. We propose in this paper an original approach based on the association of clustering and classification approaches to overcome this limitation. We propose a better approach to learn the concepts by grouping classes recursively rather than learning them class by class. We introduce an effective greedy algorithm and two theoretical measures, namely cohesion and dispersion, to evaluate the connection between the clusters and the classes. Extensive experiments on the SHL dataset show that our approach improves classification performances while reducing the number of instances used to learn each concept.

----

## [888] Random Tensor Theory for Tensor Decomposition

**Authors**: *Mohamed Ouerfelli, Mohamed Tamaazousti, Vincent Rivasseau*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20761](https://doi.org/10.1609/aaai.v36i7.20761)

**Abstract**:

We propose a new framework for tensor decomposition based on trace invariants, which are particular cases of tensor networks. In general, tensor networks are diagrams/graphs that specify a way to "multiply" a collection of tensors together to produce another tensor, matrix or scalar. The particularity of trace invariants is that the operation of multiplying copies of a certain input tensor that produces a scalar obeys specific symmetry constraints. In other words, the scalar resulting from this multiplication is invariant under some specific transformations of the involved tensor. We focus our study on the O(N)-invariant graphs, i.e. invariant under orthogonal transformations of the input tensor. The proposed approach is novel and versatile since it allows to address different theoretical and practical aspects of both CANDECOMP/PARAFAC (CP) and Tucker decomposition models. In particular we obtain several results: (i) we generalize the computational limit of Tensor PCA (a rank-one tensor decomposition) to the case of a tensor with axes of different dimensions (ii) we introduce new algorithms for both decomposition models (iii) we obtain theoretical guarantees for these algorithms and (iv) we show improvements with respect to state of the art on synthetic and real data which also highlights a promising potential for practical applications.

----

## [889] Bag Graph: Multiple Instance Learning Using Bayesian Graph Neural Networks

**Authors**: *Soumyasundar Pal, Antonios Valkanas, Florence Regol, Mark Coates*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20762](https://doi.org/10.1609/aaai.v36i7.20762)

**Abstract**:

Multiple Instance Learning (MIL) is a weakly supervised learning problem where the aim is to assign labels to sets or bags of instances, as opposed to traditional supervised learning where each instance is assumed to be independent and identically distributed (IID) and is to be labeled individually. Recent work has shown promising results for neural network models in the MIL setting. Instead of focusing on each instance, these models are trained in an end-to-end fashion to learn effective bag-level representations by suitably combining permutation invariant pooling techniques with neural architectures. In this paper, we consider modelling the interactions between bags using a graph and employ Graph Neural Networks (GNNs) to facilitate end-to-end learning. Since a meaningful graph representing dependencies between bags is rarely available, we propose to use a Bayesian GNN framework that can generate a likely graph structure for scenarios where there is uncertainty in the graph or when no graph is available. Empirical results demonstrate the efficacy of the proposed technique for several MIL benchmark tasks and a distribution regression task.

----

## [890] Competing Mutual Information Constraints with Stochastic Competition-Based Activations for Learning Diversified Representations

**Authors**: *Konstantinos P. Panousis, Anastasios Antoniadis, Sotirios Chatzis*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20763](https://doi.org/10.1609/aaai.v36i7.20763)

**Abstract**:

This work aims to address the long-established problem of learning diversified representations. To this end, we combine information-theoretic arguments with stochastic competition-based activations, namely Stochastic Local Winner-Takes-All (LWTA) units. In this context, we ditch the conventional deep architectures commonly used in Representation Learning, that rely on non-linear activations; instead, we replace them with sets of locally and stochastically competing linear units. In this setting, each network layer yields sparse outputs, determined by the outcome of the competition between units that are organized into blocks of competitors. We adopt stochastic arguments for the competition mechanism, which perform posterior sampling to determine the winner of each block. We further endow the considered networks with the ability to infer the sub-part of the network that is essential for modeling the data at hand; we impose appropriate stick-breaking priors to this end. To further enrich the information of the emerging representations, we resort to information-theoretic principles, namely the Information Competing Process (ICP). Then, all the components are tied together under the stochastic Variational Bayes framework for inference. We perform a thorough experimental investigation for our approach using benchmark datasets on image classification. As we experimentally show, the resulting networks yield significant discriminative representation learning abilities. In addition, the introduced paradigm allows for a principled investigation mechanism of the emerging intermediate network representations.

----

## [891] Blockwise Sequential Model Learning for Partially Observable Reinforcement Learning

**Authors**: *Giseung Park, Sungho Choi, Youngchul Sung*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20764](https://doi.org/10.1609/aaai.v36i7.20764)

**Abstract**:

This paper proposes a new sequential model learning architecture to solve partially observable Markov decision problems. Rather than compressing sequential information at every timestep as in conventional recurrent neural network-based methods, the proposed architecture generates a latent variable in each data block with a length of multiple timesteps and passes the most relevant information to the next block for policy optimization. The proposed blockwise sequential model is implemented based on self-attention, making the model capable of detailed sequential learning in partial observable settings. The proposed model builds an additional learning network to efficiently implement gradient estimation by using self-normalized importance sampling, which does not require the complex blockwise input data reconstruction in the model learning. Numerical results show that the proposed method significantly outperforms previous methods in various partially observable environments.

----

## [892] Deformable Graph Convolutional Networks

**Authors**: *Jinyoung Park, Sungdong Yoo, Jihwan Park, Hyunwoo J. Kim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20765](https://doi.org/10.1609/aaai.v36i7.20765)

**Abstract**:

Graph neural networks (GNNs) have significantly improved the representation power for graph-structured data. Despite of the recent success of GNNs, the graph convolution in most GNNs have two limitations. Since the graph convolution is performed in a small local neighborhood on the input graph, it is inherently incapable to capture long-range dependencies between distance nodes. In addition, when a node has neighbors that belong to different classes, i.e., heterophily, the aggregated messages from them often negatively affect representation learning. To address the two common problems of graph convolution, in this paper, we propose Deformable Graph Convolutional Networks (Deformable GCNs) that adaptively perform convolution in multiple latent spaces and capture short/long-range dependencies between nodes. Separated from node representations (features), our framework simultaneously learns the node positional embeddings (coordinates) to determine the relations between nodes in an end-to-end fashion. Depending on node position, the convolution kernels are deformed by deformation vectors and apply different transformations to its neighbor nodes. Our extensive experiments demonstrate that Deformable GCNs flexibly handles the heterophily and achieve the best performance in node classification tasks on six heterophilic graph datasets. Our code is publicly available at https://github.com/mlvlab/DeformableGCN.

----

## [893] Saliency Grafting: Innocuous Attribution-Guided Mixup with Calibrated Label Mixing

**Authors**: *Joonhyung Park, June Yong Yang, Jinwoo Shin, Sung Ju Hwang, Eunho Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20766](https://doi.org/10.1609/aaai.v36i7.20766)

**Abstract**:

The Mixup scheme suggests mixing a pair of samples to create an augmented training sample and has gained considerable attention recently for improving the generalizability of neural networks. A straightforward and widely used extension of Mixup is to combine with regional dropout-like methods: removing random patches from a sample and replacing it with the features from another sample. Albeit their simplicity and effectiveness, these methods are prone to create harmful samples due to their randomness. To address this issue, 'maximum saliency' strategies were recently proposed: they select only the most informative features to prevent such a phenomenon. However, they now suffer from lack of sample diversification as they always deterministically select regions with maximum saliency, injecting bias into the augmented data. In this paper, we present, a novel, yet simple Mixup-variant that captures the best of both worlds. Our idea is two-fold. By stochastically sampling the features and ‘grafting’ them onto another sample, our method effectively generates diverse yet meaningful samples. Its second ingredient is to produce the label of the grafted sample by mixing the labels in a saliency-calibrated fashion, which rectifies supervision misguidance introduced by the random sampling procedure. Our experiments under CIFAR, Tiny-ImageNet, and ImageNet datasets show that our scheme outperforms the current state-of-the-art augmentation strategies not only in terms of classification accuracy, but is also superior in coping under stress conditions such as data corruption and object occlusion.

----

## [894] Graph Transplant: Node Saliency-Guided Graph Mixup with Local Structure Preservation

**Authors**: *Joonhyung Park, Hajin Shim, Eunho Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20767](https://doi.org/10.1609/aaai.v36i7.20767)

**Abstract**:

Graph-structured datasets usually have irregular graph sizes and connectivities, rendering the use of recent data augmentation techniques, such as Mixup, difficult. To tackle this challenge, we present the first Mixup-like graph augmentation method called Graph Transplant, which mixes irregular graphs in data space. To be well defined on various scales of the graph, our method identifies the sub-structure as a mix unit that can preserve the local information. Since the mixup-based methods without special consideration of the context are prone to generate noisy samples, our method explicitly employs the node saliency information to select meaningful subgraphs and adaptively determine the labels. We extensively validate our method with diverse GNN architectures on multiple graph classification benchmark datasets from a wide range of graph domains of different sizes. Experimental results show the consistent superiority of our method over other basic data augmentation baselines. We also demonstrate that Graph Transplant enhances the performance in terms of robustness and model calibration.

----

## [895] CC-CERT: A Probabilistic Approach to Certify General Robustness of Neural Networks

**Authors**: *Mikhail Pautov, Nurislam Tursynbek, Marina Munkhoeva, Nikita Muravev, Aleksandr Petiushko, Ivan V. Oseledets*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20768](https://doi.org/10.1609/aaai.v36i7.20768)

**Abstract**:

In safety-critical machine learning applications, it is crucial to defend models against adversarial attacks --- small modifications of the input that change the predictions. Besides rigorously studied $\ell_p$-bounded additive perturbations, semantic perturbations (e.g. rotation, translation) raise a serious concern on deploying ML systems in real-world. Therefore, it is important to provide provable guarantees for deep learning models against semantically meaningful input transformations. In this paper, we propose a new universal probabilistic certification approach based on Chernoff-Cramer bounds that can be used in general attack settings. We estimate the probability of a model to fail if the attack is sampled from a certain distribution. Our theoretical findings are supported by experimental results on different datasets.

----

## [896] Covered Information Disentanglement: Model Transparency via Unbiased Permutation Importance

**Authors**: *João P. B. Pereira, Erik S. G. Stroes, Aeilko H. Zwinderman, Evgeni Levin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20769](https://doi.org/10.1609/aaai.v36i7.20769)

**Abstract**:

Model transparency is a prerequisite in many domains and an increasingly popular area in machine learning research.
 In the medical domain, for instance, unveiling the mechanisms behind a disease often has higher priority than the diagnostic itself since it might dictate or guide potential treatments and research directions. One of the most popular approaches to explain model global predictions is the permutation importance where the performance on permuted data is benchmarked against the baseline. However, this method and other related approaches will undervalue the importance of a feature in the presence of covariates since these cover part of its provided information. To address this issue, we propose Covered Information Disentanglement CID, a framework that considers all feature information overlap to correct the values provided by permutation importance. We further show how to compute CID efficiently when coupled with Markov random fields. We demonstrate its efficacy in adjusting permutation importance first on a controlled toy dataset and discuss its effect on real-world medical data.

----

## [897] On the Impossibility of Non-trivial Accuracy in Presence of Fairness Constraints

**Authors**: *Carlos Pinzón, Catuscia Palamidessi, Pablo Piantanida, Frank Valencia*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20770](https://doi.org/10.1609/aaai.v36i7.20770)

**Abstract**:

One of the main concerns about fairness in machine learning (ML) is that, in order to achieve it, one may have to trade off some accuracy. To overcome this issue, Hardt et al. proposed the notion of equality of opportunity (EO), which is compatible with maximal accuracy when the target label is deterministic with respect to the input features.
 
 In the probabilistic case, however, the issue is more complicated: It has been shown that under differential privacy constraints, there are data sources for which EO can only be achieved at the total detriment of accuracy, in the sense that a classifier that satisfies EO cannot be more accurate than a trivial (random guessing) classifier. 
 In our paper we strengthen this result by removing the privacy constraint. Namely, we show that for certain data sources, the most accurate classifier that satisfies EO is a trivial classifier. Furthermore, we study the trade-off between accuracy and EO loss (opportunity difference), and provide a sufficient condition on the data source under which EO and non-trivial accuracy are compatible.

----

## [898] Spiking Neural Networks with Improved Inherent Recurrence Dynamics for Sequential Learning

**Authors**: *Wachirawit Ponghiran, Kaushik Roy*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20771](https://doi.org/10.1609/aaai.v36i7.20771)

**Abstract**:

Spiking neural networks (SNNs) with leaky integrate and fire (LIF) neurons, can be operated in an event-driven manner and have internal states to retain information over time, providing opportunities for energy-efficient neuromorphic computing, especially on edge devices. Note, however, many representative works on SNNs do not fully demonstrate the usefulness of their inherent recurrence (membrane potential retaining information about the past) for sequential learning. Most of the works train SNNs to recognize static images by artificially expanded input representation in time through rate coding. We show that SNNs can be trained for practical sequential tasks by proposing modifications to a network of LIF neurons that enable internal states to learn long sequences and make their inherent recurrence resilient to the vanishing gradient problem. We then develop a training scheme to train the proposed SNNs with improved inherent recurrence dynamics. Our training scheme allows spiking neurons to produce multi-bit outputs (as opposed to binary spikes) which help mitigate the mismatch between a derivative of spiking neurons' activation function and a surrogate derivative used to overcome spiking neurons' non-differentiability. Our experimental results indicate that the proposed SNN architecture on TIMIT and LibriSpeech 100h speech recognition dataset yields accuracy comparable to that of LSTMs (within 1.10% and 0.36%, respectively), but with 2x fewer parameters than LSTMs. The sparse SNN outputs also lead to 10.13x and 11.14x savings in multiplication operations compared to GRUs, which are generally considered as a lightweight alternative to LSTMs, on TIMIT and LibriSpeech 100h datasets, respectively.

----

## [899] How Private Is Your RL Policy? An Inverse RL Based Analysis Framework

**Authors**: *Kritika Prakash, Fiza Husain, Praveen Paruchuri, Sujit Gujar*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20772](https://doi.org/10.1609/aaai.v36i7.20772)

**Abstract**:

Reinforcement Learning (RL) enables agents to learn how to perform various tasks from scratch. In domains like autonomous driving, recommendation systems, and more, optimal RL policies learned could cause a privacy breach if the policies memorize any part of the private reward. We study the set of existing differentially-private RL policies derived from various RL algorithms such as Value Iteration, Deep-Q Networks, and Vanilla Proximal Policy Optimization. We propose a new Privacy-Aware Inverse RL analysis framework (PRIL) that involves performing reward reconstruction as an adversarial attack on private policies that the agents may deploy. For this, we introduce the reward reconstruction attack, wherein we seek to reconstruct the original reward from a privacy-preserving policy using the Inverse RL algorithm. An adversary must do poorly at reconstructing the original reward function if the agent uses a tightly private policy. Using this framework, we empirically test the effectiveness of the privacy guarantee offered by the private algorithms on instances of the FrozenLake domain of varying complexities. Based on the analysis performed, we infer a gap between the current standard of privacy offered and the standard of privacy needed to protect reward functions in RL. We do so by quantifying the extent to which each private policy protects the reward function by measuring distances between the original and reconstructed rewards.

----

## [900] Detecting Misclassification Errors in Neural Networks with a Gaussian Process Model

**Authors**: *Xin Qiu, Risto Miikkulainen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20773](https://doi.org/10.1609/aaai.v36i7.20773)

**Abstract**:

As neural network classifiers are deployed in real-world applications, it is crucial that their failures can be detected reliably. One practical solution is to assign confidence scores to each prediction, then use these scores to filter out possible misclassifications. However, existing confidence metrics are not yet sufficiently reliable for this role. This paper presents a new framework that produces a quantitative metric for detecting misclassification errors. This framework, RED, builds an error detector on top of the base classifier and estimates uncertainty of the detection scores using Gaussian Processes. Experimental comparisons with other error detection methods on 125 UCI datasets demonstrate that this approach is effective. Further implementations on two probabilistic base classifiers and two large deep learning architecture in vision tasks further confirm that the method is robust and scalable. Third, an empirical analysis of RED with out-of-distribution and adversarial samples shows that the method can be used not only to detect errors but also to understand where they come from. RED can thereby be used to improve trustworthiness of neural network classifiers more broadly in the future.

----

## [901] DeepType 2: Superhuman Entity Linking, All You Need Is Type Interactions

**Authors**: *Jonathan Raiman*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20774](https://doi.org/10.1609/aaai.v36i7.20774)

**Abstract**:

Across multiple domains from computer vision to speech recognition, machine learning models have been shown to match or outperform human experts at recognition tasks. We lack such a comparison point for Entity Linking. We construct a human benchmark on two standard datasets (TAC KBP 2010 and AIDA (YAGO)) to measure human accuracy. We find that current systems still fall short of human performance.
 
 We present DeepType 2, a novel entity linking system that closes the gap. Our proposed approach overcomes shortcomings of previous type-based entity linking systems, and does not use pre-trained language models to reach this level. Three key innovations are responsible for DeepType 2's performance: 1) an abstracted representation of entities that favors shared learning and greater sample efficiency, 2) autoregressive entity features indicating type interactions (e.g. list type homogeneity, shared employers, geographical co-occurrence) with previous predictions that enable globally coherent document-wide predictions, 3) the entire model is trained end to end using a single entity-level maximum likelihood objective function. This is made possible by associating a context-specific score to each of the entity's abstract representation's sub-components (types), and summing these scores to form a candidate entity logit. In this paper, we explain how this factorization focuses the learning on the salient types of the candidate entities. Furthermore, we show how the scores can serve as a rationale for predictions.
 
  The key contributions of this work are twofold: 1) we create the first human performance benchmark on standard benchmarks in entity linking (TAC KBP 2010 and AIDA (YAGO)) which will be made publicly available to support further analyses, 2) we obtain a new state of the art on these datasets and are the first to outperform humans on our benchmark. We perform model ablations to measure the contribution of the different facets of our system. We also include an analysis of human and algorithmic errors to provide insights into the causes, notably originating from journalistic style and historical context.

----

## [902] Federated Nearest Neighbor Classification with a Colony of Fruit-Flies

**Authors**: *Parikshit Ram, Kaushik Sinha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20775](https://doi.org/10.1609/aaai.v36i7.20775)

**Abstract**:

The mathematical formalization of a neurological mechanism in the fruit-fly olfactory circuit as a locality sensitive hash (Flyhash) and bloom filter (FBF) has been recently proposed and "reprogrammed" for various learning tasks such as similarity search, outlier detection and text embeddings. We propose a novel reprogramming of this hash and bloom filter to emulate the canonical nearest neighbor classifier (NNC) in the challenging Federated Learning (FL) setup where training and test data are spread across parties and no data can leave their respective parties. Specifically, we utilize Flyhash and FBF to create the FlyNN classifier, and theoretically establish conditions where FlyNN matches NNC. We show how FlyNN is trained exactly in a FL setup with low communication overhead to produce FlyNNFL, and how it can be differentially private. Empirically, we demonstrate that (i) FlyNN matches NNC accuracy across 70 OpenML datasets, (ii) FlyNNFL training is highly scalable with low communication overhead, providing up to 8x speedup with 16 parties.

----

## [903] I-SEA: Importance Sampling and Expected Alignment-Based Deep Distance Metric Learning for Time Series Analysis and Embedding

**Authors**: *Sirisha Rambhatla, Zhengping Che, Yan Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20776](https://doi.org/10.1609/aaai.v36i7.20776)

**Abstract**:

Learning effective embeddings for potentially irregularly sampled time-series, evolving at different time scales, is fundamental for machine learning tasks such as classification and clustering. Task-dependent embeddings rely on similarities between data samples to learn effective geometries. However, many popular time-series similarity measures are not valid distance metrics, and as a result they do not reliably capture the intricate relationships between the multi-variate time-series data samples for learning effective embeddings. One of the primary ways to formulate an accurate distance metric is by forming distance estimates via Monte-Carlo-based expectation evaluations. However, the high-dimensionality of the underlying distribution, and the inability to sample from it, pose significant challenges. To this end, we develop an Importance Sampling based distance metric -- I-SEA -- which enjoys the properties of a metric while consistently achieving superior performance for machine learning tasks such as classification and representation learning. I-SEA leverages Importance Sampling and Non-parametric Density Estimation to adaptively estimate distances, enabling implicit estimation from the underlying high-dimensional distribution, resulting in improved accuracy and reduced variance. We theoretically establish the properties of I-SEA and demonstrate its capabilities via experimental evaluations on real-world healthcare datasets.

----

## [904] Saving Stochastic Bandits from Poisoning Attacks via Limited Data Verification

**Authors**: *Anshuka Rangi, Long Tran-Thanh, Haifeng Xu, Massimo Franceschetti*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20777](https://doi.org/10.1609/aaai.v36i7.20777)

**Abstract**:

This paper studies bandit algorithms under data poisoning attacks in a bounded reward setting. We consider a strong attacker model in which the attacker can observe both the selected actions and their corresponding rewards, and can contaminate the rewards with additive noise. We show that any bandit algorithm with regret O(log T) can be forced to suffer a regret O(T) with an expected amount of contamination O(log T). This amount of contamination is also necessary, as we prove that there exists an O(log T) regret bandit algorithm, specifically the classical UCB, that requires Omega(log T) amount of contamination to suffer regret Omega(T). To combat such poisoning attacks, our second main contribution is to propose verification based mechanisms, which use limited verification to access a limited number of uncontaminated rewards. In particular, for the case of unlimited verifications, we show that with O(log T) expected number of verifications, a simple modified version of the Explore-then-Commit type bandit algorithm can restore the order optimal O(log T) regret irrespective of the amount of contamination used by the attacker. 
 
We also provide a UCB-like verification scheme, called Secure-UCB, that also enjoys full recovery from any attacks, also with O(log T) expected number of verifications. To derive a matching lower bound on the number of verifications, we also prove that for any order-optimal bandit algorithm, this number of verifications O(log T) is necessary to recover the order-optimal regret. On the other hand, when the number of verifications is bounded above by a budget B, we propose a novel algorithm, Secure-BARBAR, which provably achieves O(min(C,T/sqrt(B))) regret with high probability against weak attackers (i.e., attackers who have to place the contamination before seeing the actual pulls of the bandit algorithm), where C is the total amount of contamination by the attacker, which breaks the known Omega(C) lower bound of the non-verified setting if C is large.

----

## [905] DISTREAL: Distributed Resource-Aware Learning in Heterogeneous Systems

**Authors**: *Martin Rapp, Ramin Khalili, Kilian Pfeiffer, Jörg Henkel*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20778](https://doi.org/10.1609/aaai.v36i7.20778)

**Abstract**:

We study the problem of distributed training of neural networks (NNs) on devices with heterogeneous, limited, and time-varying availability of computational resources. We present an adaptive, resource-aware, on-device learning mechanism, DISTREAL, which is able to fully and efficiently utilize the available resources on devices in a distributed manner, increasing the convergence speed. This is achieved with a dropout mechanism that dynamically adjusts the computational complexity of training an NN by randomly dropping filters of convolutional layers of the model. Our main contribution is the introduction of a design space exploration (DSE) technique, which finds Pareto-optimal per-layer dropout vectors with respect to resource requirements and convergence speed of the training. Applying this technique, each device is able to dynamically select the dropout vector that fits its available resource without requiring any assistance from the server. We implement our solution in a federated learning (FL) system, where the availability of computational resources varies both between devices and over time, and show through extensive evaluation that we are able to significantly increase the convergence speed over the state of the art without compromising on the final accuracy.

----

## [906] Sublinear Time Approximation of Text Similarity Matrices

**Authors**: *Archan Ray, Nicholas Monath, Andrew McCallum, Cameron Musco*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20779](https://doi.org/10.1609/aaai.v36i7.20779)

**Abstract**:

We study algorithms for approximating pairwise similarity matrices that arise in natural language processing. Generally, computing a similarity matrix for n data points requires Omega(n^2) similarity computations.
This quadratic scaling is a significant bottleneck, especially when similarities are computed via expensive functions, e.g., via transformer models.  Approximation methods reduce this quadratic complexity, often by using a small subset of exactly computed similarities to approximate the remainder of the complete pairwise similarity matrix.

Significant  work focuses on the efficient approximation of positive semidefinite (PSD) similarity matrices, which arise e.g., in kernel methods. However, much less is understood about indefinite (non-PSD) similarity matrices, which often  arise in  NLP. Motivated by the observation that many of these matrices are still somewhat close to PSD, we introduce a generalization of the popular Nystrom method to the indefinite setting. Our algorithm can be applied to any similarity matrix and runs in sublinear time in the size of the matrix, producing a rank-s approximation with just O(ns) similarity computations.

We show that our method, along with a simple variant of CUR decomposition, performs very well in approximating a variety of similarity matrices arising in NLP tasks. We demonstrate high accuracy of the approximated similarity matrices in tasks of document classification, sentence similarity, and cross-document coreference.

----

## [907] Decision-Dependent Risk Minimization in Geometrically Decaying Dynamic Environments

**Authors**: *Mitas Ray, Lillian J. Ratliff, Dmitriy Drusvyatskiy, Maryam Fazel*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20780](https://doi.org/10.1609/aaai.v36i7.20780)

**Abstract**:

This paper studies the problem of expected loss minimization given a data distribution that is dependent on the decision-maker's action and evolves dynamically in time according to a geometric decay process.  Novel algorithms for both the information setting in which the decision-maker has a first order gradient oracle and the setting in which they have simply a loss function oracle are introduced. The algorithms operate on the same underlying principle: the decision-maker deploys a fixed decision repeatedly over the length of an epoch,  thereby allowing the dynamically changing environment to sufficiently mix before updating the decision.  The iteration complexity in each of the settings is shown to match existing rates for first and zero order stochastic gradient methods up to logarithmic factors. The algorithms are evaluated on a ``semi-synthetic" example using real world data from the SFpark dynamic pricing pilot study; it is shown that the announced prices result in an improvement for the institution's objective (target occupancy), while achieving an overall reduction in parking rates.

----

## [908] On Causally Disentangled Representations

**Authors**: *Abbavaram Gowtham Reddy, Benin Godfrey L, Vineeth N. Balasubramanian*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20781](https://doi.org/10.1609/aaai.v36i7.20781)

**Abstract**:

Representation learners that disentangle factors of variation have already proven to be important in addressing various real world concerns such as fairness and interpretability. Initially consisting of unsupervised models with independence assumptions, more recently, weak supervision and correlated features have been explored, but without a causal view of the generative process. In contrast, we work under the regime of a causal generative process where generative factors are either independent or can be potentially confounded by a set of observed or unobserved confounders. We present an analysis of disentangled representations through the notion of disentangled causal process. We motivate the need for new metrics and datasets to study causal disentanglement and propose two evaluation metrics and a dataset. We show that our metrics capture the desiderata of disentangled causal process. Finally we perform an empirical study on state of the art disentangled representation learners using our metrics and dataset to evaluate them from causal perspective.

----

## [909] Conditional Loss and Deep Euler Scheme for Time Series Generation

**Authors**: *Carl Remlinger, Joseph Mikael, Romuald Elie*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20782](https://doi.org/10.1609/aaai.v36i7.20782)

**Abstract**:

We introduce three new generative models for time series that are based on Euler discretization of Stochastic Differential Equations (SDEs) and Wasserstein metrics. Two of these methods rely on the adaptation of generative adversarial networks (GANs) to time series. The third algorithm, called Conditional Euler Generator (CEGEN), minimizes a dedicated distance between the transition probability distributions over all time steps. In the context of Itô processes, we provide theoretical guarantees that minimizing this criterion implies accurate estimations of the drift and volatility parameters. Empirically, CEGEN outperforms state-of-the-art and GANs on both marginal and temporal dynamic metrics. Besides, correlation structures are accurately identified in high dimension. When few real data points are available, we verify the effectiveness of CEGEN when combined with transfer learning methods on model-based simulations. Finally, we illustrate the robustness of our methods on various real-world data sets.

----

## [910] Offline Reinforcement Learning as Anti-exploration

**Authors**: *Shideh Rezaeifar, Robert Dadashi, Nino Vieillard, Léonard Hussenot, Olivier Bachem, Olivier Pietquin, Matthieu Geist*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20783](https://doi.org/10.1609/aaai.v36i7.20783)

**Abstract**:

Offline Reinforcement Learning (RL) aims at learning an optimal control from a fixed dataset, without interactions with the system. An agent in this setting should avoid selecting actions whose consequences cannot be predicted from the data. This is the converse of exploration in RL, which favors such actions. We thus take inspiration from the literature on bonus-based exploration to design a new offline RL agent. The core idea is to subtract a prediction-based exploration bonus from the reward, instead of adding it for exploration. This allows the policy to stay close to the support of the dataset and practically extends some previous pessimism-based offline RL methods to a deep learning setting with arbitrary bonuses. We also connect this approach to a more common regularization of the learned policy towards the data. Instantiated with a bonus based on the prediction error of a variational autoencoder, we show that our simple agent is competitive with the state of the art on a set of continuous control locomotion and manipulation tasks.

----

## [911] Interpretable Neural Subgraph Matching for Graph Retrieval

**Authors**: *Indradyumna Roy, Venkata Sai Baba Reddy Velugoti, Soumen Chakrabarti, Abir De*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20784](https://doi.org/10.1609/aaai.v36i7.20784)

**Abstract**:

Given a query graph and a database of corpus graphs, a graph retrieval system aims to deliver the most relevant corpus graphs. Graph retrieval based on subgraph matching has a wide variety of applications, e.g., molecular fingerprint detection, circuit design, software analysis, and question answering. In such applications, a corpus graph is relevant to a query graph, if the query graph is (perfectly or approximately) a subgraph of the corpus graph. Existing neural graph retrieval models compare the node or graph embeddings of the query-corpus pairs, to compute the relevance scores between them. However, such models may not provide edge consistency between the query and corpus graphs. Moreover, they predominantly use symmetric relevance scores, which are not appropriate in the context of subgraph matching, since the underlying relevance score in subgraph search should be measured using the partial order induced by subgraph-supergraph relationship. Consequently, they show poor retrieval performance in the context of subgraph matching. In response, we propose ISONET, a novel interpretable neural edge alignment formulation, which is better able to learn the edge-consistent mapping necessary for subgraph matching. ISONET incorporates a new scoring mechanism which enforces an asymmetric relevance score, specifically tailored to subgraph matching. ISONET’s design enables it to directly identify the underlying subgraph in a corpus graph, which is relevant to the given query graph. Our experiments on diverse datasets show that ISONET outperforms recent graph retrieval formulations and systems. Additionally, ISONET can provide interpretable alignments between query-corpus graph pairs during inference, despite being trained only using binary relevance labels of whole graphs during training, without any fine-grained ground truth information about node or edge alignments.

----

## [912] FedSoft: Soft Clustered Federated Learning with Proximal Local Updating

**Authors**: *Yichen Ruan, Carlee Joe-Wong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20785](https://doi.org/10.1609/aaai.v36i7.20785)

**Abstract**:

Traditionally, clustered federated learning groups clients with the same data distribution into a cluster, so that every client is uniquely associated with one data distribution and helps train a model for this distribution. We relax this hard association assumption to soft clustered federated learning, which allows every local dataset to follow a mixture of multiple source distributions. We propose FedSoft, which trains both locally personalized models and high-quality cluster models in this setting. FedSoft limits client workload by using proximal updates to require the completion of only one optimization task from a subset of clients in every communication round. We show, analytically and empirically, that FedSoft effectively exploits similarities between the source distributions to learn personalized and cluster models that perform well.

----

## [913] Knowledge Distillation via Constrained Variational Inference

**Authors**: *Ardavan Saeedi, Yuria Utsumi, Li Sun, Kayhan Batmanghelich, Li-Wei H. Lehman*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20786](https://doi.org/10.1609/aaai.v36i7.20786)

**Abstract**:

Knowledge distillation has been used to capture the knowledge of a teacher model and distill it into a student model with some desirable characteristics such as being smaller, more efficient, or more generalizable. In this paper, we propose a framework for distilling the knowledge of a powerful discriminative model such as a neural network into commonly used graphical models known to be more interpretable (e.g., topic models, autoregressive Hidden Markov Models). Posterior of latent variables in these graphical models (e.g., topic proportions in topic models) is often used as feature representation for predictive tasks. However, these posterior-derived features are known to have poor predictive performance compared to the features learned via purely discriminative approaches. Our framework constrains variational inference for posterior variables in graphical models with a similarity preserving constraint. This constraint distills the knowledge of the discriminative model into the graphical model by ensuring that input pairs with (dis)similar representation in the teacher model also have (dis)similar representation in the student model. By adding this constraint to the variational inference scheme, we guide the graphical model to be a reasonable density model for the data while having predictive features which are as close as possible to those of a discriminative model. To make our framework applicable to a wide range of graphical models, we build upon the Automatic Differentiation Variational Inference (ADVI), a black-box inference framework for graphical models. We demonstrate the effectiveness of our framework on two real-world tasks of disease subtyping and disease trajectory modeling.

----

## [914] Hypergraph Modeling via Spectral Embedding Connection: Hypergraph Cut, Weighted Kernel k-Means, and Heat Kernel

**Authors**: *Shota Saito*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20787](https://doi.org/10.1609/aaai.v36i7.20787)

**Abstract**:

We propose a theoretical framework of multi-way similarity to model real-valued data into hypergraphs for clustering via spectral embedding.
 For graph cut based spectral clustering, it is common to model real-valued data into graph by modeling pairwise similarities using kernel function.
 This is because the kernel function has a theoretical connection to the graph cut.
 For problems where using multi-way similarities are more suitable than pairwise ones, it is natural to model as a hypergraph, which is generalization of a graph.
 However, although the hypergraph cut is well-studied, there is not yet established a hypergraph cut based framework to model multi-way similarity.
 In this paper, we formulate multi-way similarities by exploiting the theoretical foundation of kernel function.
 We show a theoretical connection between our formulation and hypergraph cut in two ways, generalizing both weighted kernel k-means and the heat kernel, by which we justify our formulation.
 We also provide a fast algorithm for spectral clustering.
 Our algorithm empirically shows better performance than existing graph and other heuristic modeling methods.

----

## [915] Reverse Differentiation via Predictive Coding

**Authors**: *Tommaso Salvatori, Yuhang Song, Zhenghua Xu, Thomas Lukasiewicz, Rafal Bogacz*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20788](https://doi.org/10.1609/aaai.v36i7.20788)

**Abstract**:

Deep learning has redefined AI thanks to the rise of artificial neural networks, which are inspired by neurological networks in the brain. Through the years, this dualism between AI and neuroscience has brought immense benefits to both fields, allowing neural networks to be used in a plethora of applications. Neural networks use an efficient implementation of reverse differentiation, called backpropagation (BP). This algorithm, however, is often criticized for its biological implausibility (e.g., lack of local update rules for the parameters). Therefore, biologically plausible learning methods that rely on predictive coding (PC), a framework for describing information processing in the brain, are increasingly studied. Recent works prove that these methods can approximate BP up to a certain margin on multilayer perceptrons (MLPs), and asymptotically on any other complex model, and that zero-divergence inference learning (Z-IL), a variant of PC, is able to exactly implement BP on MLPs. However, the recent literature shows also that there is no biologically plausible method yet that can exactly replicate the weight update of BP on complex models. To fill this gap, in this paper, we generalize (PC and) Z-IL by directly defining it on computational graphs, and show that it can perform exact reverse differentiation. What results is the first PC (and so biologically plausible) algorithm that is equivalent to BP in the way of updating parameters on any neural network, providing a bridge between the interdisciplinary research of neuroscience and deep learning. Furthermore, the above results in particular also immediately provide a novel local and parallel implementation of BP.

----

## [916] VACA: Designing Variational Graph Autoencoders for Causal Queries

**Authors**: *Pablo Sánchez-Martín, Miriam Rateike, Isabel Valera*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20789](https://doi.org/10.1609/aaai.v36i7.20789)

**Abstract**:

In this paper, we introduce VACA, a novel class of variational graph autoencoders for causal inference in the absence of hidden confounders, when only observational data and the causal graph are available. Without making any parametric assumptions, VACA mimics the necessary properties of a Structural Causal Model (SCM) to provide a flexible and practical framework for approximating interventions (do-operator) and abduction-action-prediction steps. As a result, and as shown by our empirical results, VACA accurately approximates the interventional and counterfactual distributions on diverse SCMs. Finally, we apply VACA to evaluate counterfactual fairness in fair classification problems, as well as to learn fair classifiers without compromising performance.

----

## [917] Verification of Neural-Network Control Systems by Integrating Taylor Models and Zonotopes

**Authors**: *Christian Schilling, Marcelo Forets, Sebastián Guadalupe*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20790](https://doi.org/10.1609/aaai.v36i7.20790)

**Abstract**:

We study the verification problem for closed-loop dynamical systems with neural-network controllers (NNCS). This problem is commonly reduced to computing the set of reachable states. When considering dynamical systems and neural networks in isolation, there exist precise approaches for that task based on set representations respectively called Taylor models and zonotopes. However, the combination of these approaches to NNCS is non-trivial because, when converting between the set representations, dependency information gets lost in each control cycle and the accumulated approximation error quickly renders the result useless. We present an algorithm to chain approaches based on Taylor models and zonotopes, yielding a precise reachability algorithm for NNCS. Because the algorithm only acts at the interface of the isolated approaches, it is applicable to general dynamical systems and neural networks and can benefit from future advances in these areas. Our implementation delivers state-of-the-art performance and is the first to successfully analyze all benchmark problems of an annual reachability competition for NNCS.

----

## [918] Scaling Up Influence Functions

**Authors**: *Andrea Schioppa, Polina Zablotskaia, David Vilar, Artem Sokolov*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20791](https://doi.org/10.1609/aaai.v36i8.20791)

**Abstract**:

We address efficient calculation of influence functions for tracking predictions back to the training data. We propose and analyze a new approach to speeding up the inverse Hessian calculation based on Arnoldi iteration. With this improvement, we achieve, to the best of our knowledge, the first successful implementation of influence functions that scales to full-size (language and vision) Transformer models with several hundreds of millions of parameters. We evaluate our approach in image classification and sequence-to-sequence tasks with tens to a hundred of millions of training examples. Our code is available at https://github.com/google-research/jax-influence.

----

## [919] Chaining Value Functions for Off-Policy Learning

**Authors**: *Simon Schmitt, John Shawe-Taylor, Hado van Hasselt*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20792](https://doi.org/10.1609/aaai.v36i8.20792)

**Abstract**:

To accumulate knowledge and improve its policy of behaviour, a reinforcement learning agent can learn `off-policy' about policies that differ from the policy used to generate its experience. This is important to learn counterfactuals, or because the experience was generated out of its own control. However, off-policy learning is non-trivial, and standard reinforcement-learning algorithms can be unstable and divergent.

In this paper we discuss a novel family of off-policy prediction algorithms which are convergent by construction. The idea is to first learn on-policy about the data-generating behaviour, and then bootstrap an off-policy value estimate on this on-policy estimate, thereby constructing a value estimate that is partially off-policy. This process can be repeated to build a chain of value functions, each time bootstrapping a new estimate on the previous estimate in the chain. Each step in the chain is stable and hence the complete algorithm is guaranteed to be stable. Under mild conditions this comes arbitrarily close to the off-policy TD solution when we increase the length of the chain. Hence it can compute the solution even in cases where off-policy TD diverges. 

We prove that the proposed scheme is convergent and corresponds to an iterative decomposition of the inverse key matrix. Furthermore it can be interpreted as estimating a novel objective -- that we call a `k-step expedition' -- of following the target policy for finitely many steps before continuing indefinitely with the behaviour policy. Empirically we evaluate the idea on challenging MDPs such as Baird's counter example and observe favourable results.

----

## [920] Graph Filtration Kernels

**Authors**: *Till Hendrik Schulz, Pascal Welke, Stefan Wrobel*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20793](https://doi.org/10.1609/aaai.v36i8.20793)

**Abstract**:

The majority of popular graph kernels is based on the concept of Haussler's R-convolution kernel and defines graph similarities in terms of mutual substructures. 
In this work, we enrich these similarity measures by considering graph filtrations:
Using meaningful orders on the set of edges, which allow to construct a sequence of nested graphs, we can consider a graph at multiple granularities. 
A key concept of our approach is to track graph features over the course of such graph resolutions. 
Rather than to simply compare frequencies of features in graphs, this allows for their comparison in terms of when and for how long they exist in the sequences. 
In this work, we propose a family of graph kernels that incorporate these existence intervals of features. 
While our approach can be applied to arbitrary graph features, we particularly highlight Weisfeiler-Lehman vertex labels, leading to efficient kernels. 
We show that using Weisfeiler-Lehman labels over certain filtrations strictly increases the expressive power over the ordinary Weisfeiler-Lehman procedure in terms of deciding graph isomorphism.
In fact, this result directly yields more powerful graph kernels based on such features and has implications to graph neural networks due to their close relationship to the Weisfeiler-Lehman method.
We empirically validate the expressive power of our graph kernels and show significant improvements over state-of-the-art graph kernels in terms of predictive performance on various real-world benchmark datasets.

----

## [921] Neural Networks Classify through the Class-Wise Means of Their Representations

**Authors**: *Mohamed El Amine Seddik, Mohamed Tamaazousti*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20794](https://doi.org/10.1609/aaai.v36i8.20794)

**Abstract**:

In this paper, based on an asymptotic analysis of the Softmax layer, we show that when training neural networks for classification tasks, the weight vectors corre sponding to each class of the Softmax layer tend to converge to the class-wise means computed at the representation layer (for specific choices of the representation activation). We further show some consequences of our findings to the context of transfer learning, essentially by proposing a simple yet effective initialization procedure that significantly accelerates the learning of the Softmax layer weights as the target domain gets closer to the source one. Experiments are notably performed on the datasets: MNIST, Fashion MNIST, Cifar10, and Cifar100 and using a standard CNN architecture.

----

## [922] Neuro-Symbolic Inductive Logic Programming with Logical Neural Networks

**Authors**: *Prithviraj Sen, Breno W. S. R. de Carvalho, Ryan Riegel, Alexander G. Gray*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20795](https://doi.org/10.1609/aaai.v36i8.20795)

**Abstract**:

Recent work on neuro-symbolic inductive logic programming has led to promising approaches that can learn explanatory rules from noisy, real-world data. While some proposals approximate logical operators with differentiable operators from fuzzy or real-valued logic that are parameter-free thus diminishing their capacity to fit the data, other approaches are only loosely based on logic making it difficult to interpret the learned ``rules". In this paper, we propose learning rules with the recently proposed logical neural networks (LNN). Compared to others, LNNs offer a strong connection to classical Boolean logic thus allowing for precise interpretation of learned rules while harboring parameters that can be trained with gradient-based optimization to effectively fit the data. We extend LNNs to induce rules in first-order logic. Our experiments on standard benchmarking tasks confirm that LNN rules are highly interpretable and can achieve comparable or higher accuracy due to their flexible parameterization.

----

## [923] Max-Margin Contrastive Learning

**Authors**: *Anshul Shah, Suvrit Sra, Rama Chellappa, Anoop Cherian*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20796](https://doi.org/10.1609/aaai.v36i8.20796)

**Abstract**:

Standard contrastive learning approaches usually require a large number of negatives for effective unsupervised learning and often exhibit slow convergence. We suspect this behavior is due to the suboptimal selection of negatives used for offering contrast to the positives. We counter this difficulty by taking inspiration from support vector machines (SVMs) to present max-margin contrastive learning (MMCL). Our approach selects negatives as the sparse support vectors obtained via a quadratic optimization problem, and contrastiveness is enforced by maximizing the decision margin. As SVM optimization can be computationally demanding, especially in an end-to-end setting, we present simplifications that alleviate the computational burden. We validate our approach on standard vision benchmark datasets, demonstrating better performance in unsupervised representation learning over state-of-the-art, while having better empirical convergence properties.

----

## [924] Learning to Transfer with von Neumann Conditional Divergence

**Authors**: *Ammar Shaker, Shujian Yu, Daniel Oñoro-Rubio*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20797](https://doi.org/10.1609/aaai.v36i8.20797)

**Abstract**:

The similarity of feature representations plays a pivotal role in the success of problems related to domain adaptation. Feature similarity includes both the invariance of marginal distributions and the closeness of conditional distributions given the desired response y (e.g., class labels). Unfortunately, traditional methods always learn such features without fully taking into consideration the information in y, which in turn may lead to a mismatch of the conditional distributions or the mixup of discriminative structures underlying data distributions. In this work, we introduce the recently proposed von Neumann conditional divergence to improve the transferability across multiple domains. We show that this new divergence is differentiable and eligible to easily quantify the functional dependence between features and y. Given multiple source tasks, we integrate this divergence to capture discriminative information in y and design novel learning objectives assuming those source tasks are observed either simultaneously or sequentially. In both scenarios, we obtain favorable performance against state-of-the-art methods in terms of smaller generalization error on new tasks and less catastrophic forgetting on source tasks (in the sequential setup).

----

## [925] Online Apprenticeship Learning

**Authors**: *Lior Shani, Tom Zahavy, Shie Mannor*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20798](https://doi.org/10.1609/aaai.v36i8.20798)

**Abstract**:

In Apprenticeship Learning (AL), we are given a Markov Decision Process (MDP) without access to the cost function. Instead, we observe trajectories sampled by an expert that acts according to some policy. The goal is to find a policy that matches the expert's performance on some predefined set of cost functions. 
We introduce an online variant of AL (Online Apprenticeship Learning; OAL), where the agent is expected to perform comparably to the expert while interacting with the environment. We show that the OAL problem can be effectively solved by combining two mirror descent based no-regret algorithms: one for policy optimization and another for learning the worst case cost. By employing optimistic exploration, we derive a convergent algorithm with O(sqrt(K)) regret, where K is the number of interactions with the MDP, and an additional linear error term that depends on the amount of expert trajectories available. Importantly, our algorithm avoids the need to solve an MDP at each iteration, making it more practical compared to prior AL methods. Finally, we implement a deep variant of our algorithm which shares some similarities to GAIL, but where the discriminator is replaced with the costs learned by OAL. Our simulations suggest that OAL performs well in high dimensional control problems.

----

## [926] HoD-Net: High-Order Differentiable Deep Neural Networks and Applications

**Authors**: *Siyuan Shen, Tianjia Shao, Kun Zhou, Chenfanfu Jiang, Feng Luo, Yin Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20799](https://doi.org/10.1609/aaai.v36i8.20799)

**Abstract**:

We introduce a deep architecture named HoD-Net to enable high-order differentiability for deep learning. HoD-Net is based on and generalizes the complex-step finite difference (CSFD) method. While similar to classic finite difference, CSFD approaches the derivative of a function from a higher-dimension complex domain, leading to highly accurate and robust differentiation computation without numerical stability issues. This method can be coupled with backpropagation and adjoint perturbation methods for an efficient calculation of high-order derivatives. We show how this numerical scheme can be leveraged in challenging deep learning problems, such as high-order network training, deep learning-based physics simulation, and neural differential equations.

----

## [927] Conditional Generative Model Based Predicate-Aware Query Approximation

**Authors**: *Nikhil Sheoran, Subrata Mitra, Vibhor Porwal, Siddharth Ghetia, Jatin Varshney, Tung Mai, Anup B. Rao, Vikas Maddukuri*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20800](https://doi.org/10.1609/aaai.v36i8.20800)

**Abstract**:

The goal of Approximate Query Processing (AQP) is to provide very fast but "accurate enough" results for costly aggregate queries thereby improving user experience in interactive exploration of large datasets. Recently proposed Machine-Learning-based AQP techniques can provide very low latency as query execution only involves model inference as compared to traditional query processing on database clusters. However, with increase in the number of filtering predicates (WHERE clauses), the approximation error significantly increases for these methods. Analysts often use queries with a large number of predicates for insights discovery. Thus, maintaining low approximation error is important to prevent analysts from drawing misleading conclusions. In this paper, we propose ELECTRA, a predicate-aware AQP system that can answer analytics-style queries with a large number of predicates with much smaller approximation errors. ELECTRA uses a conditional generative model that learns the conditional distribution of the data and at run-time generates a small (≈ 1000 rows) but representative sample, on which the query is executed to compute the approximate result. Our evaluations with four different baselines on three real-world datasets show that ELECTRA provides lower AQP error for large number of predicates compared to baselines.

----

## [928] Learning Bounded Context-Free-Grammar via LSTM and the Transformer: Difference and the Explanations

**Authors**: *Hui Shi, Sicun Gao, Yuandong Tian, Xinyun Chen, Jishen Zhao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20801](https://doi.org/10.1609/aaai.v36i8.20801)

**Abstract**:

Long Short-Term Memory (LSTM) and Transformers are two popular neural architectures used for natural language processing tasks. Theoretical results show that both are Turing-complete and can represent any context-free language (CFL).In practice, it is often observed that Transformer models have better representation power than LSTM. But the reason is barely understood. We study such practical differences between LSTM and Transformer and propose an explanation based on their latent space decomposition patterns. To achieve this goal, we introduce an oracle training paradigm, which forces the decomposition of the latent representation of LSTMand the Transformer and supervises with the transitions of the Pushdown Automaton (PDA) of the corresponding CFL. With the forced decomposition, we show that the performance upper bounds of LSTM and Transformer in learning CFL are close: both of them can simulate a stack and perform stack operation along with state transitions. However, the absence of forced decomposition leads to the failure of LSTM models to capture the stack and stack operations, while having a marginal impact on the Transformer model. Lastly, we connect the experiment on the prototypical PDA to a real-world parsing task to re-verify the conclusions

----

## [929] Shape Prior Guided Attack: Sparser Perturbations on 3D Point Clouds

**Authors**: *Zhenbo Shi, Zhi Chen, Zhenbo Xu, Wei Yang, Zhidong Yu, Liusheng Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20802](https://doi.org/10.1609/aaai.v36i8.20802)

**Abstract**:

Deep neural networks are extremely vulnerable to malicious input data. As 3D data is increasingly used in vision tasks such as robots, autonomous driving and drones, the internal robustness of the classification models for 3D point cloud has received widespread attention. In this paper, we propose a novel method named SPGA (Shape Prior Guided Attack) to generate adversarial point cloud examples. We use shape prior information to make perturbations sparser and thus achieve imperceptible attacks. In particular, we propose a Spatially Logical Block (SLB) to apply adversarial points through sliding in the oriented bounding box. Moreover, we design an algorithm called FOFA for this type of task, which further refines the adversarial attack in the process of breaking down complicated problems into sub-problems. Compared with the methods of global perturbation, our attack method consumes significantly fewer computations, making it more efficient. Most importantly of all, SPGA can generate examples with a higher attack success rate (even in a defensive situation), less perturbation budget and stronger transferability.

----

## [930] TRF: Learning Kernels with Tuned Random Features

**Authors**: *Alistair Shilton, Sunil Gupta, Santu Rana, Arun Kumar Anjanapura Venkatesh, Svetha Venkatesh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20803](https://doi.org/10.1609/aaai.v36i8.20803)

**Abstract**:

Random Fourier features (RFF) are a popular set of tools for constructing low-dimensional approximations of translation-invariant kernels, allowing kernel methods to be scaled to big data.  Apart from their computational advantages, by working in the spectral domain random Fourier features expose the translation invariant kernel as a density function that may, in principle, be manipulated directly to tune the kernel.  In this paper we propose selecting the density function from a reproducing kernel Hilbert space to allow us to search the space of all translation-invariant kernels.  Our approach, which we call tuned random features (TRF), achieves this by approximating the density function as the RKHS-norm regularised least-squares best fit to an unknown ``true'' optimal density function, resulting in a RFF formulation where kernel selection is reduced to regularised risk minimisation with a novel regulariser.  We derive bounds on the Rademacher complexity for our method showing that our random features approximation method converges to optimal kernel selection in the large N,D limit.  Finally, we prove experimental results for a variety of real-world learning problems, demonstrating the performance of our approach compared to comparable methods.

----

## [931] Estimation of Local Average Treatment Effect by Data Combination

**Authors**: *Kazuhiko Shinoda, Takahiro Hoshino*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20804](https://doi.org/10.1609/aaai.v36i8.20804)

**Abstract**:

It is important to estimate the local average treatment effect (LATE) when compliance with a treatment assignment is incomplete. The previously proposed methods for LATE estimation required all relevant variables to be jointly observed in a single dataset; however, it is sometimes difficult or even impossible to collect such data in many real-world problems for technical or privacy reasons. We consider a novel problem setting in which LATE, as a function of covariates, is nonparametrically identified from the combination of separately observed datasets. For estimation, we show that the direct least squares method, which was originally developed for estimating the average treatment effect under complete compliance, is applicable to our setting. However, model selection and hyperparameter tuning for the direct least squares estimator can be unstable in practice since it is defined as a solution to the minimax problem. We then propose a weighted least squares estimator that enables simpler model selection by avoiding the minimax objective formulation. Unlike the inverse probability weighted (IPW) estimator, the proposed estimator directly uses the pre-estimated weight without inversion, avoiding the problems caused by the IPW methods. We demonstrate the effectiveness of our method through experiments using synthetic and real-world datasets.

----

## [932] Constraint-Driven Explanations for Black-Box ML Models

**Authors**: *Aditya A. Shrotri, Nina Narodytska, Alexey Ignatiev, Kuldeep S. Meel, João Marques-Silva, Moshe Y. Vardi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20805](https://doi.org/10.1609/aaai.v36i8.20805)

**Abstract**:

The need to understand the inner workings of opaque Machine Learning models has prompted researchers to devise various types of post-hoc explanations. A large class of such explainers proceed in two phases: first perturb an input instance whose explanation is sought, and then generate an interpretable artifact to explain the prediction of the opaque model on that instance. Recently, Deutch and Frost proposed to use an additional input from the user: a set of constraints over the input space to guide the perturbation phase. While this approach affords the user the ability to tailor the explanation to their needs, striking a balance between flexibility, theoretical rigor and computational cost has remained an open challenge.
 
 We propose a novel constraint-driven explanation generation approach which simultaneously addresses these issues in a modular fashion. Our framework supports the use of expressive Boolean constraints giving the user more flexibility to specify the subspace to generate perturbations from. Leveraging advances in Formal Methods, we can theoretically guarantee strict adherence of the samples to the desired distribution. This also allows us to compute fidelity in a rigorous way, while scaling much better in practice. Our empirical study demonstrates concrete uses of our tool CLIME in obtaining more meaningful explanations with high fidelity.

----

## [933] Noise-Robust Learning from Multiple Unsupervised Sources of Inferred Labels

**Authors**: *Amila Silva, Ling Luo, Shanika Karunasekera, Christopher Leckie*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20806](https://doi.org/10.1609/aaai.v36i8.20806)

**Abstract**:

Deep Neural Networks (DNNs) generally require large-scale datasets for training. Since manually obtaining clean labels for large datasets is extremely expensive, unsupervised models based on domain-specific heuristics can be used to efficiently infer the labels for such datasets. However, the labels from such inferred sources are typically noisy, which could easily mislead and lessen the generalizability of DNNs. Most approaches proposed in the literature to address this problem assume the label noise depends only on the true class of an instance (i.e., class-conditional noise). However, this assumption is not realistic for the inferred labels as they are typically inferred based on the features of the instances. The few recent attempts to model such instance-dependent (i.e., feature-dependent) noise require auxiliary information about the label noise (e.g., noise rates or clean samples). This work proposes a theoretically motivated framework to correct label noise in the presence of multiple labels inferred from unsupervised models. The framework consists of two modules: (1) MULTI-IDNC, a novel approach to correct label noise that is instance-dependent yet not class-conditional; (2) MULTI-CCNC, which extends an existing class-conditional noise-robust approach to yield improved class-conditional noise correction using multiple noisy label sources. We conduct experiments using nine real-world datasets for three different classification tasks (images, text and graph nodes). Our results show that our approach achieves notable improvements (e.g., 6.4% in accuracy) against state-of-the-art baselines while dealing with both instance-dependent and class-conditional noise in inferred label sources.

----

## [934] QUILT: Effective Multi-Class Classification on Quantum Computers Using an Ensemble of Diverse Quantum Classifiers

**Authors**: *Daniel Silver, Tirthak Patel, Devesh Tiwari*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20807](https://doi.org/10.1609/aaai.v36i8.20807)

**Abstract**:

Quantum computers can theoretically have significant acceleration over classical computers; but, the near-future era of quantum computing is limited due to small number of qubits that are also error prone. QUILT is a framework for performing multi-class classification task designed to work effectively on current error-prone quantum computers. QUILT is evaluated with real quantum machines as well as with projected noise levels as quantum machines become more noise free. QUILT demonstrates up to 85% multi-class classification accuracy with the MNIST dataset on a five-qubit system.

----

## [935] EqGNN: Equalized Node Opportunity in Graphs

**Authors**: *Uriel Singer, Kira Radinsky*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20808](https://doi.org/10.1609/aaai.v36i8.20808)

**Abstract**:

Graph neural networks (GNNs), has been widely used for supervised learning tasks in graphs reaching state-of-the-art results. However, little work was dedicated to creating unbiased GNNs, i.e., where the classification is uncorrelated with sensitive attributes, such as race or gender. Some ignore the sensitive attributes or optimize for the criteria of statistical parity for fairness. However, it has been shown that neither approaches ensure fairness, but rather cripple the utility of the prediction task.
In this work, we present a GNN framework that allows optimizing representations for the notion of Equalized Odds fairness criteria. The architecture is composed of three components: (1) a GNN classifier predicting the utility class, (2) a sampler learning the distribution of the sensitive attributes of the nodes given their labels. It generates samples fed into a (3) discriminator that discriminates between true and sampled sensitive attributes using a novel ``permutation loss'' function. Using these components, we train a model to neglect information regarding the sensitive attribute only with respect to its label. To the best of our knowledge, we are the first to optimize GNNs for the equalized odds criteria. We evaluate our classifier over several graph datasets and sensitive attributes and show our algorithm reaches state-of-the-art results.

----

## [936] ApproxIFER: A Model-Agnostic Approach to Resilient and Robust Prediction Serving Systems

**Authors**: *Mahdi Soleymani, Ramy E. Ali, Hessam Mahdavifar, Amir Salman Avestimehr*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20809](https://doi.org/10.1609/aaai.v36i8.20809)

**Abstract**:

Due to the surge of cloud-assisted AI services, the problem of designing resilient prediction serving systems that can effectively cope with stragglers and minimize response delays has attracted much interest. The common approach for tackling this problem is replication which assigns the same prediction task to multiple workers. This approach, however, is  inefficient and incurs significant resource overheads. Hence, a learning-based approach known as parity model (ParM) has been recently proposed which learns models that can generate ``parities’’ for a group of predictions  to reconstruct the predictions of the slow/failed workers. While this learning-based approach is more resource-efficient than replication, it is tailored to the specific model hosted by the cloud and is particularly suitable for a small number of queries (typically less than four) and tolerating very few stragglers (mostly one). Moreover, ParM does not handle Byzantine adversarial workers. We propose a different approach, named Approximate Coded Inference (ApproxIFER), that does not require training any parity models, hence it is agnostic to the model hosted by the cloud and can be readily applied to different data domains and model architectures. Compared with earlier works, ApproxIFER can handle a general number of stragglers and scales significantly better with the  number of queries. Furthermore, ApproxIFER is robust against Byzantine workers. Our extensive experiments on a large number of datasets and model architectures show significant degraded mode accuracy improvement by up to 58% over ParM.

----

## [937] Feature Importance Explanations for Temporal Black-Box Models

**Authors**: *Akshay Sood, Mark Craven*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20810](https://doi.org/10.1609/aaai.v36i8.20810)

**Abstract**:

Models in the supervised learning framework may capture rich and complex representations over the features that are hard for humans to interpret. Existing methods to explain such models are often specific to architectures and data where the features do not have a time-varying component. In this work, we propose TIME, a method to explain models that are inherently temporal in nature. Our approach (i) uses a model-agnostic permutation-based approach to analyze global feature importance, (ii) identifies the importance of salient features with respect to their temporal ordering as well as localized windows of influence, and (iii) uses hypothesis testing to provide statistical rigor.

----

## [938] Reward-Weighted Regression Converges to a Global Optimum

**Authors**: *Miroslav Strupl, Francesco Faccio, Dylan R. Ashley, Rupesh Kumar Srivastava, Jürgen Schmidhuber*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20811](https://doi.org/10.1609/aaai.v36i8.20811)

**Abstract**:

Reward-Weighted Regression (RWR) belongs to a family of widely known iterative Reinforcement Learning algorithms based on the Expectation-Maximization framework. In this family, learning at each iteration consists of sampling a batch of trajectories using the current policy and fitting a new policy to maximize a return-weighted log-likelihood of actions. Although RWR is known to yield monotonic improvement of the policy under certain circumstances, whether and under which conditions RWR converges to the optimal policy have remained open questions. In this paper, we provide for the first time a proof that RWR converges to a global optimum when no function approximation is used, in a general compact setting. Furthermore, for the simpler case with finite state and action spaces we prove R-linear convergence of the state-value function to the optimum.

----

## [939] Gradient-Based Novelty Detection Boosted by Self-Supervised Binary Classification

**Authors**: *Jingbo Sun, Li Yang, Jiaxin Zhang, Frank Liu, Mahantesh Halappanavar, Deliang Fan, Yu Cao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20812](https://doi.org/10.1609/aaai.v36i8.20812)

**Abstract**:

Novelty detection aims to automatically identify out-of-distribution (OOD) data, without any prior knowledge of them. It is a critical step in data monitoring, behavior analysis and other applications, helping enable continual learning in the field. Conventional methods of OOD detection perform multi-variate analysis on an ensemble of data or features, and usually resort to the supervision with OOD data to improve the accuracy. In reality, such supervision is impractical as one cannot anticipate the anomalous data. In this paper, we propose a novel, self-supervised approach that does not rely on any pre-defined OOD data: (1) The new method evaluates the Mahalanobis distance of the gradients between the in-distribution and OOD data. (2) It is assisted by a self-supervised binary classifier to guide the label selection to generate the gradients, and maximize the Mahalanobis distance. In the evaluation with multiple datasets, such as CIFAR-10, CIFAR-100, SVHN and TinyImageNet, the proposed approach consistently outperforms state-of-the-art supervised and unsupervised methods in the area under the receiver operating characteristic (AUROC) and area under the precision-recall curve (AUPR) metrics. We further demonstrate that this detector is able to accurately learn one OOD class in continual learning.

----

## [940] Deterministic and Discriminative Imitation (D2-Imitation): Revisiting Adversarial Imitation for Sample Efficiency

**Authors**: *Mingfei Sun, Sam Devlin, Katja Hofmann, Shimon Whiteson*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20813](https://doi.org/10.1609/aaai.v36i8.20813)

**Abstract**:

Sample efficiency is crucial for imitation learning methods to be applicable in real-world applications. Many studies improve sample efficiency by extending adversarial imitation to be off-policy regardless of the fact that these off-policy extensions could either change the original objective or involve complicated optimization. We revisit the foundation of adversarial imitation and propose an off-policy sample efficient approach that requires no adversarial training or min-max optimization. Our formulation capitalizes on two key insights: (1) the similarity between the Bellman equation and the stationary state-action distribution equation allows us to derive a novel temporal difference (TD) learning approach; and (2) the use of a deterministic policy simplifies the TD learning. Combined, these insights yield a practical algorithm, Deterministic and Discriminative Imitation (D2-Imitation), which oper- ates by first partitioning samples into two replay buffers and then learning a deterministic policy via off-policy reinforcement learning. Our empirical results show that D2-Imitation is effective in achieving good sample efficiency, outperforming several off-policy extension approaches of adversarial imitation on many control tasks.

----

## [941] Exploiting Mixed Unlabeled Data for Detecting Samples of Seen and Unseen Out-of-Distribution Classes

**Authors**: *Yi-Xuan Sun, Wei Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20814](https://doi.org/10.1609/aaai.v36i8.20814)

**Abstract**:

Out-of-Distribution (OOD) detection is essential in real-world applications, which has attracted increasing attention in recent years. However, most existing OOD detection methods require many labeled In-Distribution (ID) data, causing a heavy labeling cost. In this paper, we focus on the more realistic scenario, where limited labeled data and abundant unlabeled data are available, and these unlabeled data are mixed with ID and OOD samples. We propose the Adaptive In-Out-aware Learning (AIOL) method, in which we employ the appropriate temperature to adaptively select potential ID and OOD samples from the mixed unlabeled data and consider the entropy over them for OOD detection. Moreover, since the test data in realistic applications may contain OOD samples whose classes are not in the mixed unlabeled data (we call them unseen OOD classes), data augmentation techniques are brought into the method to further improve the performance. The experiments are conducted on various benchmark datasets, which demonstrate the superiority of our method.

----

## [942] Generalized Equivariance and Preferential Labeling for GNN Node Classification

**Authors**: *Zeyu Sun, Wenjie Zhang, Lili Mou, Qihao Zhu, Yingfei Xiong, Lu Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20815](https://doi.org/10.1609/aaai.v36i8.20815)

**Abstract**:

Existing graph neural networks (GNNs) largely rely on node embeddings, which represent a node as a vector by its identity, type, or content. However, graphs with unattributed nodes widely exist in real-world applications (e.g., anonymized social networks). Previous GNNs either assign random labels to nodes (which introduces artefacts to the GNN) or assign one embedding to all nodes (which fails to explicitly distinguish one node from another). Further, when these GNNs are applied to unattributed node classification problems, they have an undesired equivariance property, which are fundamentally unable to address the data with multiple possible outputs. In this paper, we analyze the limitation of existing approaches to node classification problems. Inspired by our analysis, we propose a generalized equivariance property and a Preferential Labeling technique that satisfies the desired property asymptotically. Experimental results show that we achieve high performance in several unattributed node classification tasks.

----

## [943] Explainable and Local Correction of Classification Models Using Decision Trees

**Authors**: *Hirofumi Suzuki, Hiroaki Iwashita, Takuya Takagi, Keisuke Goto, Yuta Fujishige, Satoshi Hara*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20816](https://doi.org/10.1609/aaai.v36i8.20816)

**Abstract**:

In practical machine learning, models are frequently updated, or corrected, to adapt to new datasets. In this study, we pose two challenges to model correction. First, the effects of corrections to the end-users need to be described explicitly, similar to standard software where the corrections are described as release notes. Second, the amount of corrections need to be small so that the corrected models perform similarly to the old models. In this study, we propose the first model correction method for classification models that resolves these two challenges. Our idea is to use an additional decision tree to correct the output of the old models. Thanks to the explainability of decision trees, the corrections are describable to the end-users, which resolves the first challenge. We resolve the second challenge by incorporating the amount of corrections when training the additional decision tree so that the effects of corrections to be small. Experiments on real data confirm the effectiveness of the proposed method compared to existing correction methods.

----

## [944] Consistency Regularization for Adversarial Robustness

**Authors**: *Jihoon Tack, Sihyun Yu, Jongheon Jeong, Minseon Kim, Sung Ju Hwang, Jinwoo Shin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20817](https://doi.org/10.1609/aaai.v36i8.20817)

**Abstract**:

Adversarial training (AT) is currently one of the most successful methods to obtain the adversarial robustness of deep neural networks. However, the phenomenon of robust overfitting, i.e., the robustness starts to decrease significantly during AT, has been problematic, not only making practitioners consider a bag of tricks for a successful training, e.g., early stopping, but also incurring a significant generalization gap in the robustness. In this paper, we propose an effective regularization technique that prevents robust overfitting by optimizing an auxiliary `consistency' regularization loss during AT. Specifically, we discover that data augmentation is a quite effective tool to mitigate the overfitting in AT, and develop a regularization that forces the predictive distributions after attacking from two different augmentations of the same instance to be similar with each other. Our experimental results demonstrate that such a simple regularization technique brings significant improvements in the test robust accuracy of a wide range of AT methods. More remarkably, we also show that our method could significantly help the model to generalize its robustness against unseen adversaries, e.g., other types or larger perturbations compared to those used during training. Code is available at https://github.com/alinlab/consistency-adversarial.

----

## [945] Regularization Guarantees Generalization in Bayesian Reinforcement Learning through Algorithmic Stability

**Authors**: *Aviv Tamar, Daniel Soudry, Ev Zisselman*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20818](https://doi.org/10.1609/aaai.v36i8.20818)

**Abstract**:

In the Bayesian reinforcement learning (RL) setting, a prior distribution over the unknown problem parameters -- the rewards and transitions -- is assumed, and a policy that optimizes the (posterior) expected return is sought. A common approximation, which has been recently popularized as meta-RL, is to train the agent on a sample of N problem instances from the prior, with the hope that for large enough N, good generalization behavior to an unseen test instance will be obtained. 
 In this work, we study generalization in Bayesian RL under the probably approximately correct (PAC) framework, using the method of algorithmic stability. Our main contribution is showing that by adding regularization, the optimal policy becomes uniformly stable in an appropriate sense. Most stability results in the literature build on strong convexity of the regularized loss -- an approach that is not suitable for RL as Markov decision processes (MDPs) are not convex. Instead, building on recent results of fast convergence rates for mirror descent in regularized MDPs, we show that regularized MDPs satisfy a certain quadratic growth criterion, which is sufficient to establish stability. This result, which may be of independent interest, allows us to study the effect of regularization on generalization in the Bayesian RL setting.

----

## [946] FedProto: Federated Prototype Learning across Heterogeneous Clients

**Authors**: *Yue Tan, Guodong Long, Lu Liu, Tianyi Zhou, Qinghua Lu, Jing Jiang, Chengqi Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20819](https://doi.org/10.1609/aaai.v36i8.20819)

**Abstract**:

Heterogeneity across clients in federated learning (FL) usually hinders the optimization convergence and generalization performance when the aggregation of clients' knowledge occurs in the gradient space. For example, clients may differ in terms of data distribution, network latency, input/output space, and/or model architecture, which can easily lead to the misalignment of their local gradients. To improve the tolerance to heterogeneity, we propose a novel federated prototype learning (FedProto) framework in which the clients and server communicate the abstract class prototypes instead of the gradients. FedProto aggregates the local prototypes collected from different clients, and then sends the global prototypes back to all clients to regularize the training of local models. The training on each client aims to minimize the classification error on the local data while keeping the resulting local prototypes sufficiently close to the corresponding global ones. Moreover, we provide a theoretical analysis to the convergence rate of FedProto under non-convex objectives. In experiments, we propose a benchmark setting tailored for heterogeneous FL, with FedProto outperforming several recent FL approaches on multiple datasets.

----

## [947] What about Inputting Policy in Value Function: Policy Representation and Policy-Extended Value Function Approximator

**Authors**: *Hongyao Tang, Zhaopeng Meng, Jianye Hao, Chen Chen, Daniel Graves, Dong Li, Changmin Yu, Hangyu Mao, Wulong Liu, Yaodong Yang, Wenyuan Tao, Li Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20820](https://doi.org/10.1609/aaai.v36i8.20820)

**Abstract**:

We study Policy-extended Value Function Approximator (PeVFA) in Reinforcement Learning (RL), which extends conventional value function approximator (VFA) to take as input not only the state (and action) but also an explicit policy representation. Such an extension enables PeVFA to preserve values of multiple policies at the same time and brings an appealing characteristic, i.e., value generalization among policies. We formally analyze the value generalization under Generalized Policy Iteration (GPI). From theoretical and empirical lens, we show that generalized value estimates offered by PeVFA may have lower initial approximation error to true values of successive policies, which is expected to improve consecutive value approximation during GPI. Based on above clues, we introduce a new form of GPI with PeVFA which leverages the value generalization along policy improvement path. Moreover, we propose a representation learning framework for RL policy, providing several approaches to learn effective policy embeddings from policy network parameters or state-action pairs. In our experiments, we evaluate the efficacy of value generalization offered by PeVFA and policy representation learning in several OpenAI Gym continuous control tasks. For a representative instance of algorithm implementation, Proximal Policy Optimization (PPO) re-implemented under the paradigm of GPI with PeVFA achieves about 40% performance improvement on its vanilla counterpart in most environments.

----

## [948] Optimal Sampling Gaps for Adaptive Submodular Maximization

**Authors**: *Shaojie Tang, Jing Yuan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20821](https://doi.org/10.1609/aaai.v36i8.20821)

**Abstract**:

Running machine learning algorithms on large and rapidly growing volumes of data is often computationally expensive, one common trick to reduce the size of a data set, and thus reduce the computational cost of machine learning algorithms, is probability sampling. It creates a sampled data set by including each data point from the original data set with a known probability. Although the benefit of running machine learning algorithms on the reduced data set is obvious, one major concern is that the performance of the solution obtained from samples might be much worse than that of the optimal solution when using the full data set.  In this paper, we examine the performance loss caused by  probability sampling in the context of adaptive submodular maximization. We consider a simple probability sampling method which selects each data point with probability at least r.  If we set r=1, our problem reduces to finding a solution based on the original full data set. We define sampling gap as the largest ratio between the optimal solution obtained from the full data set and the optimal solution obtained from the samples,  over independence systems. Our main contribution is to show that if the sampling probability of each data point is at least r and the utility function is policywise submodular, then the sampling gap is both upper bounded and lower bounded by 1/r. We show that the property of policywise submodular can be found in a wide range of real-world applications, including pool-based active learning and adaptive viral marketing.

----

## [949] With False Friends Like These, Who Can Notice Mistakes?

**Authors**: *Lue Tao, Lei Feng, Jinfeng Yi, Songcan Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20822](https://doi.org/10.1609/aaai.v36i8.20822)

**Abstract**:

Adversarial examples crafted by an explicit adversary have attracted significant attention in machine learning. However, the security risk posed by a potential false friend has been largely overlooked. In this paper, we unveil the threat of hypocritical examples---inputs that are originally misclassified yet perturbed by a false friend to force correct predictions. While such perturbed examples seem harmless, we point out for the first time that they could be maliciously used to conceal the mistakes of a substandard (i.e., not as good as required) model during an evaluation. Once a deployer trusts the hypocritical performance and applies the "well-performed" model in real-world applications, unexpected failures may happen even in benign environments. More seriously, this security risk seems to be pervasive: we find that many types of substandard models are vulnerable to hypocritical examples across multiple datasets. Furthermore, we provide the first attempt to characterize the threat with a metric called hypocritical risk and try to circumvent it via several countermeasures. Results demonstrate the effectiveness of the countermeasures, while the risk remains non-negligible even after adaptive robust training.

----

## [950] Powering Finetuning in Few-Shot Learning: Domain-Agnostic Bias Reduction with Selected Sampling

**Authors**: *Ran Tao, Han Zhang, Yutong Zheng, Marios Savvides*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20823](https://doi.org/10.1609/aaai.v36i8.20823)

**Abstract**:

In recent works, utilizing a deep network trained on meta-training set serves as a strong baseline in few-shot learning. In this paper, we move forward to refine novel-class features by finetuning a trained deep network. Finetuning is designed to focus on reducing biases in novel-class feature distributions, which we define as two aspects: class-agnostic and class-specific biases. Class-agnostic bias is defined as the distribution shifting introduced by domain difference, which we propose Distribution Calibration Module(DCM) to reduce. DCM owes good property of eliminating domain difference and fast feature adaptation during optimization. Class-specific bias is defined as the biased estimation using a few samples in novel classes, which we propose Selected Sampling(SS) to reduce. Without inferring the actual class distribution, SS is designed by running sampling using proposal distributions around support-set samples. By powering finetuning with DCM and SS, we achieve state-of-the-art results on Meta-Dataset with consistent performance boosts over ten datasets from different domains. We believe our simple yet effective method demonstrates its possibility to be applied on practical few-shot applications.

----

## [951] SMINet: State-Aware Multi-Aspect Interests Representation Network for Cold-Start Users Recommendation

**Authors**: *Wanjie Tao, Yu Li, Liangyue Li, Zulong Chen, Hong Wen, Peilin Chen, Tingting Liang, Quan Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20824](https://doi.org/10.1609/aaai.v36i8.20824)

**Abstract**:

Online travel platforms (OTPs), e.g., bookings.com and Ctrip.com, deliver travel experiences to online users by providing travel-related products. Although much progress has been made, the state-of-the-arts for cold-start problems are largely sub-optimal for user representation, since they do not take into account the unique characteristics exhibited from user travel behaviors. In this work, we propose a State-aware Multi-aspect Interests representation Network (SMINet) for cold-start users recommendation at OTPs, which consists of a multi-aspect interests extractor, a co-attention layer, and a state-aware gating layer. The key component of the model is the multi-aspect interests extractor, which is able to extract representations for the user's multi-aspect interests. Furthermore, to learn the interactions between the user behaviors in the current session and the above multi-aspect interests, we carefully design a co-attention layer which allows the cross attentions between the two modules. Additionally, we propose a travel state-aware gating layer to attentively select the multi-aspect interests. The final user representation is obtained by fusing the three components. Comprehensive experiments conducted both offline and online demonstrate the superior performance of the proposed model at user representation, especially for cold-start users, compared with state-of-the-art methods.

----

## [952] SplitFed: When Federated Learning Meets Split Learning

**Authors**: *Chandra Thapa, Mahawaga Arachchige Pathum Chamikara, Seyit Camtepe, Lichao Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20825](https://doi.org/10.1609/aaai.v36i8.20825)

**Abstract**:

Federated learning (FL) and split learning (SL) are two popular distributed machine learning approaches. Both follow a model-to-data scenario; clients train and test machine learning models without sharing raw data. SL provides better model privacy than FL due to the machine learning model architecture split between clients and the server. Moreover, the split model makes SL a better option for resource-constrained environments. However, SL performs slower than FL due to the relay-based training across multiple clients. In this regard, this paper presents a novel approach, named splitfed learning (SFL), that amalgamates the two approaches eliminating their inherent drawbacks, along with a refined architectural configuration incorporating differential privacy and PixelDP to enhance data privacy and model robustness. Our analysis and empirical results demonstrate that (pure) SFL provides similar test accuracy and communication efficiency as SL while significantly decreasing its computation time per global epoch than in SL for multiple clients. Furthermore, as in SL, its communication efficiency over FL improves with the number of clients. Besides, the performance of SFL with privacy and robustness measures is further evaluated under extended experimental settings.

----

## [953] Listwise Learning to Rank Based on Approximate Rank Indicators

**Authors**: *Thibaut Thonet, Yagmur Gizem Cinar, Éric Gaussier, Minghan Li, Jean-Michel Renders*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20826](https://doi.org/10.1609/aaai.v36i8.20826)

**Abstract**:

We study here a way to approximate information retrieval metrics through a softmax-based approximation of the rank indicator function. Indeed, this latter function is a key component in the design of information retrieval metrics, as well as in the design of the ranking and sorting functions. Obtaining a good approximation for it thus opens the door to differentiable approximations of many evaluation measures that can in turn be used in neural end-to-end approaches. We first prove theoretically that the approximations proposed are of good quality, prior to validate them experimentally on both learning to rank and text-based information retrieval tasks.

----

## [954] PrivateMail: Supervised Manifold Learning of Deep Features with Privacy for Image Retrieval

**Authors**: *Praneeth Vepakomma, Julia Balla, Ramesh Raskar*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20827](https://doi.org/10.1609/aaai.v36i8.20827)

**Abstract**:

Differential Privacy offers strong guarantees such as immutable privacy under any post-processing. In this work, we propose a differentially private mechanism called PrivateMail
for performing supervised manifold learning. We then apply it to the use case of private image retrieval to obtain nearest matches to a client’s target image from a server’s database.
PrivateMail releases the target image as part of a differentially private manifold embedding. We give bounds on the global sensitivity of the manifold learning map in order to obfuscate and release embeddings with differential privacy inducing noise. We show that PrivateMail obtains a substantially better performance in terms of the privacy-utility trade off in comparison to several baselines on various datasets. We share code for applying PrivateMail at http://tiny.cc/PrivateMail.

----

## [955] Amortized Generation of Sequential Algorithmic Recourses for Black-Box Models

**Authors**: *Sahil Verma, Keegan Hines, John P. Dickerson*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20828](https://doi.org/10.1609/aaai.v36i8.20828)

**Abstract**:

Explainable machine learning (ML) has gained traction in recent years due to the increasing adoption of ML-based systems in many sectors. Algorithmic Recourses (ARs) provide "what if" feedback of the form "if an input datapoint were x' instead of x, then an ML-based system's output would be y' instead of y." Recourses are attractive due to their actionable feedback, amenability to existing legal frameworks, and fidelity to the underlying ML model. Yet, current recourse approaches are single shot that is, they assume x can change to x' in a single time period. We propose a novel stochastic-control-based approach that generates sequential recourses, that is, recourses that allow x to move stochastically and sequentially across intermediate states to a final state x'. Our approach is model agnostic and black box. Furthermore, the calculation of recourses is amortized such that once trained, it applies to multiple datapoints without the need for re-optimization. In addition to these primary characteristics, our approach admits optional desiderata such as adherence to the data manifold, respect for causal relations, and sparsity identified by past research as desirable properties of recourses. We evaluate our approach using three real-world datasets and show successful generation of sequential recourses that respect other recourse desiderata.

----

## [956] Robust Optimal Classification Trees against Adversarial Examples

**Authors**: *Daniël Vos, Sicco Verwer*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20829](https://doi.org/10.1609/aaai.v36i8.20829)

**Abstract**:

Decision trees are a popular choice of explainable model, but just like neural networks, they suffer from adversarial examples. Existing algorithms for fitting decision trees robust against adversarial examples are greedy heuristics and lack approximation guarantees. In this paper we propose ROCT, a collection of methods to train decision trees that are optimally robust against user-specified attack models. We show that the min-max optimization problem that arises in adversarial learning can be solved using a single minimization formulation for decision trees with 0-1 loss. We propose such formulations in Mixed-Integer Linear Programming and Maximum Satisfiability, which widely available solvers can optimize. We also present a method that determines the upper bound on adversarial accuracy for any model using bipartite matching. Our experimental results demonstrate that the existing heuristics achieve close to optimal scores while ROCT achieves state-of-the-art scores.

----

## [957] Spline-PINN: Approaching PDEs without Data Using Fast, Physics-Informed Hermite-Spline CNNs

**Authors**: *Nils Wandel, Michael Weinmann, Michael Neidlin, Reinhard Klein*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20830](https://doi.org/10.1609/aaai.v36i8.20830)

**Abstract**:

Partial Differential Equations (PDEs) are notoriously difficult to solve. In general, closed form solutions are not available and numerical approximation schemes are computationally expensive. In this paper, we propose to approach the solution of PDEs based on a novel technique that combines the advantages of two recently emerging machine learning based approaches.
 First, physics-informed neural networks (PINNs) learn continuous solutions of PDEs and can be trained with little to no ground truth data. However, PINNs do not generalize well to unseen domains. Second, convolutional neural networks provide fast inference and generalize but either require large amounts of training data or a physics-constrained loss based on finite differences that can lead to inaccuracies and discretization artifacts.
 We leverage the advantages of both of these approaches by using Hermite spline kernels in order to continuously interpolate a grid-based state representation that can be handled by a CNN. This allows for training without any precomputed training data using a physics-informed loss function only and provides fast, continuous solutions that generalize to unseen domains.
 We demonstrate the potential of our method at the examples of the incompressible Navier-Stokes equation and the damped wave equation. Our models are able to learn several intriguing phenomena such as Karman vortex streets, the Magnus effect, Doppler effect, interference patterns and wave reflections. Our quantitative assessment and an interactive real-time demo show that we are narrowing the gap in accuracy of unsupervised ML based methods to industrial solvers for computational fluid dynamics (CFD) while being orders of magnitude faster.

----

## [958] Context Uncertainty in Contextual Bandits with Applications to Recommender Systems

**Authors**: *Hao Wang, Yifei Ma, Hao Ding, Yuyang Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20831](https://doi.org/10.1609/aaai.v36i8.20831)

**Abstract**:

Recurrent neural networks have proven effective in modeling sequential user feedbacks for recommender systems. However, they usually focus solely on item relevance and fail to effectively explore diverse items for users, therefore harming the system performance in the long run. To address this problem, we propose a new type of recurrent neural networks, dubbed recurrent exploration networks (REN), to jointly perform representation learning and effective exploration in the latent space. REN tries to balance relevance and exploration while taking into account the uncertainty in the representations. Our theoretical analysis shows that REN can preserve the rate-optimal sublinear regret even when there exists uncertainty in the learned representations. Our empirical study demonstrates that REN can achieve satisfactory long-term rewards on both synthetic and real-world recommendation datasets, outperforming state-of-the-art models.

----

## [959] Demystifying Why Local Aggregation Helps: Convergence Analysis of Hierarchical SGD

**Authors**: *Jiayi Wang, Shiqiang Wang, Rong-Rong Chen, Mingyue Ji*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20832](https://doi.org/10.1609/aaai.v36i8.20832)

**Abstract**:

Hierarchical SGD (H-SGD) has emerged as a new distributed SGD algorithm for multi-level communication networks. In H-SGD, before each global aggregation, workers send their updated local models to local servers for aggregations. Despite recent research efforts, the effect of local aggregation on global convergence still lacks theoretical understanding. In this work, we first introduce a new notion of "upward" and "downward" divergences. We then use it to conduct a novel analysis to obtain a worst-case convergence upper bound for two-level H-SGD with non-IID data, non-convex objective function, and stochastic gradient. By extending this result to the case with random grouping, we observe that this convergence upper bound of H-SGD is between the upper bounds of two single-level local SGD settings, with the number of local iterations equal to the local and global update periods in H-SGD, respectively. We refer to this as the "sandwich behavior". Furthermore, we extend our analytical approach based on "upward" and "downward" divergences to study the convergence for the general case of H-SGD with more than two levels, where the "sandwich behavior" still holds. Our theoretical results provide key insights of why local aggregation can be beneficial in improving the convergence of H-SGD.

----

## [960] Learngene: From Open-World to Your Learning Task

**Authors**: *Qiu-Feng Wang, Xin Geng, Shuxia Lin, Shiyu Xia, Lei Qi, Ning Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20833](https://doi.org/10.1609/aaai.v36i8.20833)

**Abstract**:

Although deep learning has made significant progress on fixed large-scale datasets, it typically encounters challenges regarding improperly detecting unknown/unseen classes in the open-world scenario, over-parametrized, and overfitting small samples. Since biological systems can overcome the above difficulties very well, individuals inherit an innate gene from collective creatures that have evolved over hundreds of millions of years and then learn new skills through few examples. Inspired by this, we propose a practical collective-individual paradigm where an evolution (expandable) network is trained on sequential tasks and then recognize unknown classes in real-world. Moreover, the learngene, i.e., the gene for learning initialization rules of the target model, is proposed to inherit the meta-knowledge from the collective model and reconstruct a lightweight individual model on the target task. Particularly, a novel criterion is proposed to discover learngene in the collective model, according to the gradient information. Finally, the individual model is trained only with few samples on the target learning tasks. We demonstrate the effectiveness of our approach in an extensive empirical study and theoretical analysis.

----

## [961] Boosting Active Learning via Improving Test Performance

**Authors**: *Tianyang Wang, Xingjian Li, Pengkun Yang, Guosheng Hu, Xiangrui Zeng, Siyu Huang, Cheng-Zhong Xu, Min Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20834](https://doi.org/10.1609/aaai.v36i8.20834)

**Abstract**:

Central to active learning (AL) is what data should be selected for annotation. Existing works attempt to select highly uncertain or informative data for annotation. Nevertheless, it remains unclear how selected data impacts the test performance of the task model used in AL. In this work, we explore such an impact by theoretically proving that selecting unlabeled data of higher gradient norm leads to a lower upper-bound of test loss, resulting in a better test performance.  However, due to the lack of label information, directly computing gradient norm for unlabeled data is infeasible. To address this challenge, we propose two schemes, namely expected-gradnorm and entropy-gradnorm. The former computes the gradient norm by constructing an expected empirical loss while the latter constructs an unsupervised loss with entropy. Furthermore, we integrate the two schemes in a universal AL framework. We evaluate our method on classical image classification and semantic segmentation tasks. To demonstrate its competency in domain applications and its robustness to noise, we also validate our method on a cellular imaging analysis task, namely cryo-Electron Tomography subtomogram classification. Results demonstrate that our method achieves superior performance against the state of the art. We refer readers to https://arxiv.org/pdf/2112.05683.pdf for the full version of this paper which includes the appendix and source code link.

----

## [962] Efficient Algorithms for General Isotone Optimization

**Authors**: *Xiwen Wang, Jiaxi Ying, José Vinícius de Miranda Cardoso, Daniel P. Palomar*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20835](https://doi.org/10.1609/aaai.v36i8.20835)

**Abstract**:

Monotonicity is often a fundamental assumption involved in the modeling of a number of real-world applications. From an optimization perspective, monotonicity is formulated as partial order constraints among the optimization variables, commonly known as isotone optimization. In this paper, we develop an efficient, provable convergent algorithm for solving isotone optimization problems. The proposed algorithm is general in the sense that it can handle any arbitrary isotonic constraints and a wide range of objective functions. We evaluate our algorithm and state-of-the-art methods with experiments involving both synthetic and real-world data. The experimental results demonstrate that our algorithm is more efficient by one to four orders of magnitude than the state-of-the-art methods.

----

## [963] Efficient Causal Structure Learning from Multiple Interventional Datasets with Unknown Targets

**Authors**: *Yunxia Wang, Fuyuan Cao, Kui Yu, Jiye Liang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20836](https://doi.org/10.1609/aaai.v36i8.20836)

**Abstract**:

We consider the problem of reducing the false discovery rate in multiple high-dimensional interventional datasets under unknown targets. Traditional algorithms merged directly multiple causal graphs learned, which ignores the contradictions of different datasets, leading to lots of inconsistent directions of edges. For reducing the contradictory information, we propose a new algorithm, which first learns an interventional Markov equivalence class (I-MEC) before merging multiple graphs. It utilizes the full power of the constraints available in interventional data and combines ideas from local learning, intervention, and search-and-score techniques in a principled and effective way in different intervention experiments. Specifically, local learning on multiple datasets is used to build a causal skeleton. Perfect intervention destroys some possible triangles, leading to the identification of more possible V-structures. And then a theoretically correct I-MEC is learned. Search and scoring techniques based on the learned I-MEC further identify the remaining unoriented edges. Both theoretical analysis and experiments on benchmark Bayesian networks with the number of variables from 20 to 724 validate that the effectiveness of our algorithm in reducing the false discovery rate in high-dimensional interventional data.

----

## [964] Continual Learning through Retrieval and Imagination

**Authors**: *Zhen Wang, Liu Liu, Yiqun Duan, Dacheng Tao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20837](https://doi.org/10.1609/aaai.v36i8.20837)

**Abstract**:

Continual learning is an intellectual ability of artificial agents to learn new streaming labels from sequential data. The main impediment to continual learning is catastrophic forgetting, a severe performance degradation on previously learned tasks. Although simply replaying all previous data or continuously adding the model parameters could alleviate the issue, it is impractical in real-world applications due to the limited available resources. Inspired by the mechanism of the human brain to deepen its past impression, we propose a novel framework, Deep Retrieval and Imagination (DRI), which consists of two components: 1) an embedding network that constructs a unified embedding space without adding model parameters on the arrival of new tasks; and 2) a generative model to produce additional (imaginary) data based on the limited memory. By retrieving the past experiences and corresponding imaginary data, DRI distills knowledge and rebalances the embedding space to further mitigate forgetting. Theoretical analysis demonstrates that DRI can reduce the loss approximation error and improve the robustness through retrieval and imagination, bringing better generalizability to the network. Extensive experiments show that DRI performs significantly better than the existing state-of-the-art continual learning methods and effectively alleviates catastrophic forgetting.

----

## [965] Max-Min Grouped Bandits

**Authors**: *Zhenlin Wang, Jonathan Scarlett*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20838](https://doi.org/10.1609/aaai.v36i8.20838)

**Abstract**:

In this paper, we introduce a multi-armed bandit problem termed max-min grouped bandits, in which the arms are arranged in possibly-overlapping groups, and the goal is to find a group whose worst arm has the highest mean reward. This problem is of interest in applications such as recommendation systems, and is also closely related to widely-studied robust optimization problems. We present two algorithms based successive elimination and robust optimization, and derive upper bounds on the number of samples to guarantee finding a max-min optimal or near-optimal group, as well as an algorithm-independent lower bound. We discuss the degree of tightness of our bounds in various cases of interest, and the difficulties in deriving uniformly tight bounds.

----

## [966] Sample-Efficient Reinforcement Learning via Conservative Model-Based Actor-Critic

**Authors**: *Zhihai Wang, Jie Wang, Qi Zhou, Bin Li, Houqiang Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20839](https://doi.org/10.1609/aaai.v36i8.20839)

**Abstract**:

Model-based reinforcement learning algorithms, which aim to learn a model of the environment to make decisions, are more sample efficient than their model-free counterparts. The sample efficiency of model-based approaches relies on whether the model can well approximate the environment. However, learning an accurate model is challenging, especially in complex and noisy environments. To tackle this problem, we propose the conservative model-based actor-critic (CMBAC), a novel approach that achieves high sample efficiency without the strong reliance on accurate learned models. Specifically, CMBAC learns multiple estimates of the Q-value function from a set of inaccurate models and uses the average of the bottom-k estimates---a conservative estimate---to optimize the policy. An appealing feature of CMBAC is that the conservative estimates effectively encourage the agent to avoid unreliable “promising actions”---whose values are high in only a small fraction of the models. Experiments demonstrate that CMBAC significantly outperforms state-of-the-art approaches in terms of sample efficiency on several challenging control tasks, and the proposed method is more robust than previous methods in noisy environments.

----

## [967] Controlling Underestimation Bias in Reinforcement Learning via Quasi-median Operation

**Authors**: *Wei Wei, Yujia Zhang, Jiye Liang, Lin Li, Yuze Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20840](https://doi.org/10.1609/aaai.v36i8.20840)

**Abstract**:

How to get a good value estimation is one of the key problems in reinforcement learning (RL). Current off-policy methods, such as Maxmin Q-learning, TD3 and TADD, suffer from the underestimation problem when solving the overestimation problem. In this paper, we propose the Quasi-Median Operation, a novel way to mitigate the underestimation bias by selecting the quasi-median from multiple state-action values. Based on the quasi-median operation, we propose Quasi-Median Q-learning (QMQ) for the discrete action tasks and Quasi-Median Delayed Deep Deterministic Policy Gradient (QMD3) for the continuous action tasks. Theoretically, the underestimation bias of our method is improved while the estimation variance is significantly reduced compared to Maxmin Q-learning, TD3 and TADD. We conduct extensive experiments on the discrete and continuous action tasks, and results show that our method outperforms the state-of-the-art methods.

----

## [968] Symbolic Brittleness in Sequence Models: On Systematic Generalization in Symbolic Mathematics

**Authors**: *Sean Welleck, Peter West, Jize Cao, Yejin Choi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20841](https://doi.org/10.1609/aaai.v36i8.20841)

**Abstract**:

Neural sequence models trained with maximum likelihood estimation have led to breakthroughs in many tasks, where success is defined by the gap between training and test performance. However, their ability to achieve stronger forms of generalization remains unclear. We consider the problem of symbolic mathematical integration, as it requires generalizing systematically beyond the training set. We develop a methodology for evaluating generalization that takes advantage of the problem domain's structure and access to a verifier. Despite promising in-distribution performance of sequence-to-sequence models in this domain, we demonstrate challenges in achieving robustness, compositionality, and out-of-distribution generalization, through both carefully constructed manual test suites and a genetic algorithm that automatically finds large collections of failures in a controllable manner. Our investigation highlights the difficulty of generalizing well with the predominant modeling and learning approach, and the importance of evaluating beyond the test set, across different aspects of generalization.

----

## [969] Prune and Tune Ensembles: Low-Cost Ensemble Learning with Sparse Independent Subnetworks

**Authors**: *Tim Whitaker, Darrell Whitley*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20842](https://doi.org/10.1609/aaai.v36i8.20842)

**Abstract**:

Ensemble Learning is an effective method for improving generalization in machine learning. However, as state-of-the-art neural networks grow larger, the computational cost associated with training several independent networks becomes expensive. We introduce a fast, low-cost method for creating diverse ensembles of neural networks without needing to train multiple models from scratch. We do this by first training a single parent network. We then create child networks by cloning the parent and dramatically pruning the parameters of each child to create an ensemble of members with unique and diverse topologies. We then briefly train each child network for a small number of epochs, which now converge significantly faster when compared to training from scratch. We explore various ways to maximize diversity in the child networks, including the use of anti-random pruning and one-cycle tuning. This diversity enables "Prune and Tune" ensembles to achieve results that are competitive with traditional ensembles at a fraction of the training cost. We benchmark our approach against state of the art low-cost ensemble methods and display marked improvement in both accuracy and uncertainty estimation on CIFAR-10 and CIFAR-100.

----

## [970] PluGeN: Multi-Label Conditional Generation from Pre-trained Models

**Authors**: *Maciej Wolczyk, Magdalena Proszewska, Lukasz Maziarka, Maciej Zieba, Patryk Wielopolski, Rafal Kurczab, Marek Smieja*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20843](https://doi.org/10.1609/aaai.v36i8.20843)

**Abstract**:

Modern generative models achieve excellent quality in a variety of tasks including image or text generation and chemical molecule modeling. However, existing methods often lack the essential ability to generate examples with requested properties, such as the age of the person in the photo or the weight of the generated molecule. Incorporating such additional conditioning factors would require rebuilding the entire architecture and optimizing the parameters from scratch. Moreover, it is difficult to disentangle selected attributes so that to perform edits of only one attribute while leaving the others unchanged. To overcome these limitations we propose PluGeN (Plugin Generative Network), a simple yet effective generative technique that can be used as a plugin to pre-trained generative models. The idea behind our approach is to transform the entangled latent representation using a flow-based module into a multi-dimensional space where the values of each attribute are modeled as an independent one-dimensional distribution. In consequence, PluGeN can generate new samples with desired attributes as well as manipulate labeled attributes of existing examples. Due to the disentangling of the latent representation, we are even able to generate samples with rare or unseen combinations of attributes in the dataset, such as a young person with gray hair, men with make-up, or women with beards. We combined PluGeN with GAN and VAE models and applied it to conditional generation and manipulation of images and chemical molecule modeling. Experiments demonstrate that PluGeN preserves the quality of backbone models while adding the ability to control the values of labeled attributes. Implementation is available at https://github.com/gmum/plugen.

----

## [971] Structure Learning-Based Task Decomposition for Reinforcement Learning in Non-stationary Environments

**Authors**: *Honguk Woo, Gwangpyo Yoo, Minjong Yoo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20844](https://doi.org/10.1609/aaai.v36i8.20844)

**Abstract**:

Reinforcement learning (RL) agents empowered by deep neural networks have been considered a feasible solution to automate control functions in a cyber-physical system. 
 In this work, we consider an RL-based agent and address the issue of learning via continual interaction with a time-varying dynamic system modeled as a non-stationary Markov decision process (MDP). 
 We view such a non-stationary MDP as a time series of conventional MDPs that can be parameterized by hidden variables. To infer the hidden parameters, we present a task decomposition method that exploits CycleGAN-based structure learning. 
 This method enables the separation of time-variant tasks from a non-stationary MDP, establishing the task decomposition embedding specific to time-varying information. 
 To mitigate the adverse effect due to inherent noises of task embedding, we also leverage continual learning on sequential tasks by adapting the orthogonal gradient descent scheme with a sliding window.
 Through various experiments, we demonstrate that our approach renders the RL agent adaptable to time-varying dynamic environment conditions, outperforming other methods including state-of-the-art non-stationary MDP algorithms.

----

## [972] An Efficient Combinatorial Optimization Model Using Learning-to-Rank Distillation

**Authors**: *Honguk Woo, Hyunsung Lee, Sangwoo Cho*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20845](https://doi.org/10.1609/aaai.v36i8.20845)

**Abstract**:

Recently, deep reinforcement learning (RL) has proven its feasibility in solving combinatorial optimization problems (COPs). The learning-to-rank techniques have been studied in the field of information retrieval. While several COPs can be formulated as the prioritization of input items, as is common in the information retrieval, it has not been fully explored how the learning-to-rank techniques can be incorporated into deep RL for COPs. 
 In this paper, we present the learning-to-rank distillation-based COP framework, where a high-performance ranking policy obtained by RL for a COP can be distilled into a non-iterative, simple model, thereby achieving a low-latency COP solver. Specifically, we employ the approximated ranking distillation to render a score-based ranking model learnable via gradient descent. Furthermore, we use the efficient sequence sampling to improve the inference performance with a limited delay. 
 With the framework, we demonstrate that a distilled model not only achieves comparable performance to its respective, high-performance RL, but also provides several times faster inferences. We evaluate the framework with several COPs such as priority-based task scheduling and multidimensional knapsack, demonstrating the benefits of the framework in terms of inference latency and performance.

----

## [973] PUMA: Performance Unchanged Model Augmentation for Training Data Removal

**Authors**: *Ga Wu, Masoud Hashemi, Christopher Srinivasa*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20846](https://doi.org/10.1609/aaai.v36i8.20846)

**Abstract**:

Preserving the performance of a trained model while removing unique characteristics of marked training data points is challenging. Recent research usually suggests retraining a model from scratch with remaining training data or refining the model by reverting the model optimization on the marked data points. Unfortunately, aside from their computational inefficiency, those approaches inevitably hurt the resulting model's generalization ability since they remove not only unique characteristics but also discard shared (and possibly contributive) information. To address the performance degradation problem, this paper presents a novel approach called Performance Unchanged Model Augmentation (PUMA). The proposed PUMA framework explicitly models the influence of each training data point on the model's generalization ability with respect to various performance criteria. It then complements the negative impact of removing marked data by reweighting the remaining data optimally. To demonstrate the effectiveness of the PUMA framework, we compared it with multiple state-of-the-art data removal techniques in the experiments, where we show the PUMA can effectively and efficiently remove the unique characteristics of marked training data without retraining the model that can 1) fool a membership attack, and 2) resist performance degradation. In addition, as PUMA estimates the data importance during its operation, we show it could serve to debug mislabelled data points more efficiently than existing approaches.

----

## [974] Generalizing Reinforcement Learning through Fusing Self-Supervised Learning into Intrinsic Motivation

**Authors**: *Keyu Wu, Min Wu, Zhenghua Chen, Yuecong Xu, Xiaoli Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20847](https://doi.org/10.1609/aaai.v36i8.20847)

**Abstract**:

Despite the great potential of reinforcement learning (RL) in solving complex decision-making problems, generalization remains one of its key challenges, leading to difficulty in deploying learned RL policies to new environments. In this paper, we propose to improve the generalization of RL algorithms through fusing Self-supervised learning into Intrinsic Motivation (SIM). Specifically, SIM boosts representation learning through driving the cross-correlation matrix between the embeddings of augmented and non-augmented samples close to the identity matrix. This aims to increase the similarity between the embedding vectors of a sample and its augmented version while minimizing the redundancy between the components of these vectors. Meanwhile, the redundancy reduction based self-supervised loss is converted to an intrinsic reward to further improve generalization in RL via an auxiliary objective. As a general paradigm, SIM can be implemented on top of any RL algorithm. Extensive evaluations have been performed on a diversity of tasks. Experimental results demonstrate that SIM consistently outperforms the state-of-the-art methods and exhibits superior generalization capability and sample efficiency.

----

## [975] AdaLoss: A Computationally-Efficient and Provably Convergent Adaptive Gradient Method

**Authors**: *Xiaoxia Wu, Yuege Xie, Simon Shaolei Du, Rachel A. Ward*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20848](https://doi.org/10.1609/aaai.v36i8.20848)

**Abstract**:

We propose a computationally-friendly adaptive learning rate schedule, ``AdaLoss", which directly uses the information of the loss function to adjust the stepsize in gradient descent methods. We prove that this schedule enjoys linear convergence in linear regression. Moreover, we extend the to the non-convex regime, in the context of two-layer over-parameterized neural networks. If the width is sufficiently large (polynomially), then AdaLoss converges robustly to the global minimum in polynomial time. We numerically verify the theoretical results and extend the scope of the numerical experiments by considering applications in LSTM models for text clarification and policy gradients for control problems.

----

## [976] Towards Off-Policy Learning for Ranking Policies with Logged Feedback

**Authors**: *Teng Xiao, Suhang Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20849](https://doi.org/10.1609/aaai.v36i8.20849)

**Abstract**:

Probabilistic learning to rank (LTR) has been the dominating approach for optimizing the ranking metric, but cannot maximize long-term rewards. Reinforcement learning models have been proposed to maximize user long-term rewards by formulating the recommendation as a sequential decision-making problem, but could only achieve inferior accuracy compared to LTR counterparts, primarily due to the lack of online interactions and the characteristics of ranking. In this paper, we propose a new off-policy value ranking (VR) algorithm that can simultaneously maximize user long-term rewards and optimize the ranking metric offline for improved sample efficiency in a unified Expectation-Maximization (EM) framework. We theoretically and empirically show that the EM process guides the leaned policy to enjoy the benefit of integration of the future reward and ranking metric, and learn without any online interactions. Extensive offline and online experiments demonstrate the effectiveness of our methods

----

## [977] Active Learning for Domain Adaptation: An Energy-Based Approach

**Authors**: *Binhui Xie, Longhui Yuan, Shuang Li, Chi Harold Liu, Xinjing Cheng, Guoren Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20850](https://doi.org/10.1609/aaai.v36i8.20850)

**Abstract**:

Unsupervised domain adaptation has recently emerged as an effective paradigm for generalizing deep neural networks to new target domains. However, there is still enormous potential to be tapped to reach the fully supervised performance. In this paper, we present a novel active learning strategy to assist knowledge transfer in the target domain, dubbed active domain adaptation. We start from an observation that energy-based models exhibit free energy biases when training (source) and test (target) data come from different distributions. Inspired by this inherent mechanism, we empirically reveal that a simple yet efficient energy-based sampling strategy sheds light on selecting the most valuable target samples than existing approaches requiring particular architectures or computation of the distances. Our algorithm, Energy-based Active Domain Adaptation (EADA), queries groups of target data that incorporate both domain characteristic and instance uncertainty into every selection round. Meanwhile, by aligning the free energy of target data compact around the source domain via a regularization term, domain gap can be implicitly diminished. Through extensive experiments, we show that EADA surpasses state-of-the-art methods on well-known challenging benchmarks with substantial improvements, making it a useful option in the open world. Code is available at https://github.com/BIT-DA/EADA.

----

## [978] GearNet: Stepwise Dual Learning for Weakly Supervised Domain Adaptation

**Authors**: *Renchunzi Xie, Hongxin Wei, Lei Feng, Bo An*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20851](https://doi.org/10.1609/aaai.v36i8.20851)

**Abstract**:

This paper studies a weakly supervised domain adaptation (WSDA) problem, where we only have access to the source domain with noisy labels, from which we need to transfer useful information to the unlabeled target domain. Although there have been a few studies on this problem, most of them only exploit unidirectional relationships from the source domain to the target domain. In this paper, we propose a universal paradigm called GearNet to exploit bilateral relationships between the two domains. Specifically, we take the two domains as different inputs to train two models alternately, and a symmetrical Kullback-Leibler loss is used for selectively matching the predictions of the two models in the same domain. This interactive learning schema enables implicit label noise canceling and exploit correlations between the source and target domains. Therefore, our GearNet has the great potential to boost the performance of a wide range of existing WSDA methods. Comprehensive experimental results show that the performance of existing methods can be significantly improved by equipping with our GearNet.

----

## [979] Reinforcement Learning Augmented Asymptotically Optimal Index Policy for Finite-Horizon Restless Bandits

**Authors**: *Guojun Xiong, Jian Li, Rahul Singh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20852](https://doi.org/10.1609/aaai.v36i8.20852)

**Abstract**:

We study a finite-horizon restless multi-armed bandit problem with multiple actions, dubbed as R(MA)^2B. The state of each arm evolves according to a controlled Markov decision process (MDP), and the reward of pulling an arm depends on both the current state and action of the corresponding MDP. Since finding the optimal policy is typically intractable, we propose a computationally appealing index policy entitled Occupancy-Measured-Reward Index Policy for the finite-horizon R(MA)^2B. Our index policy is well-defined without the requirement of indexability condition and is provably asymptotically optimal as the number of arms tends to infinity. We then adopt a learning perspective where the system parameters are unknown, and propose R(MA)^2B-UCB, a generative model based reinforcement learning augmented algorithm that can fully exploit the structure of Occupancy-Measured-Reward Index Policy. Compared to existing algorithms, R(MA)^2B-UCB performs close to offline optimum, and achieves a sub-linear regret and a low computational complexity all at once. Experimental results show that R(MA)^2B-UCB outperforms existing algorithms in both regret and running time.

----

## [980] Coordinating Momenta for Cross-Silo Federated Learning

**Authors**: *An Xu, Heng Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20853](https://doi.org/10.1609/aaai.v36i8.20853)

**Abstract**:

Communication efficiency is crucial for federated learning (FL). Conducting local training steps in clients to reduce the communication frequency between clients and the server is a common method to address this issue. However, it leads to the client drift problem due to non-i.i.d. data distributions in different clients which severely deteriorates the performance. In this work, we propose a new method to improve the training performance in cross-silo FL via maintaining double momentum buffers. One momentum buffer tracks the server model updating direction, and the other tracks the local model updating direction. Moreover, we introduce a novel momentum fusion technique to coordinate the server and local momentum buffers. We also provide the first theoretical convergence analysis involving both the server and local standard momentum SGD. Extensive deep FL experimental results show a better training performance than FedAvg and existing standard momentum SGD variants.

----

## [981] Learning-Augmented Algorithms for Online Steiner Tree

**Authors**: *Chenyang Xu, Benjamin Moseley*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20854](https://doi.org/10.1609/aaai.v36i8.20854)

**Abstract**:

This paper considers the recently popular beyond-worst-case algorithm analysis model which integrates machine-learned predictions with online algorithm design. We consider the online Steiner tree problem in this model for both directed and undirected graphs. Steiner tree is known to have strong lower bounds in the online setting and any algorithm’s worst-case guarantee is far from desirable.
 
 This paper considers algorithms that predict which terminal arrives online. The predictions may be incorrect and the algorithms’ performance is parameterized by the number of incorrectly predicted terminals. These guarantees ensure that algorithms break through the online lower bounds with good predictions and the competitive ratio gracefully degrades as the prediction error grows. We then observe that the theory is predictive of what will occur empirically. We show on graphs where terminals are drawn from a distribution, the new online algorithms have strong performance even with modestly correct predictions.

----

## [982] Constraints Penalized Q-learning for Safe Offline Reinforcement Learning

**Authors**: *Haoran Xu, Xianyuan Zhan, Xiangyu Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20855](https://doi.org/10.1609/aaai.v36i8.20855)

**Abstract**:

We study the problem of safe offline reinforcement learning (RL), the goal is to learn a policy that maximizes long-term reward while satisfying safety constraints given only offline data, without further interaction with the environment. This problem is more appealing for real world RL applications, in which data collection is costly or dangerous. Enforcing constraint satisfaction is non-trivial, especially in offline settings, as there is a potential large discrepancy between the policy distribution and the data distribution, causing errors in estimating the value of safety constraints. We show that naïve approaches that combine techniques from safe RL and offline RL can only learn sub-optimal solutions. We thus develop a simple yet effective algorithm, Constraints Penalized Q-Learning (CPQ), to solve the problem. Our method admits the use of data generated by mixed behavior policies. We present a theoretical analysis and demonstrate empirically that our approach can learn robustly across a variety of benchmark control tasks, outperforming several baselines.

----

## [983] Deep Incomplete Multi-View Clustering via Mining Cluster Complementarity

**Authors**: *Jie Xu, Chao Li, Yazhou Ren, Liang Peng, Yujie Mo, Xiaoshuang Shi, Xiaofeng Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20856](https://doi.org/10.1609/aaai.v36i8.20856)

**Abstract**:

Incomplete multi-view clustering (IMVC) is an important unsupervised approach to group the multi-view data containing missing data in some views. Previous IMVC methods suffer from the following issues: (1) the inaccurate imputation or padding for missing data negatively affects the clustering performance, (2) the quality of features after fusion might be interfered by the low-quality views, especially the inaccurate imputed views. To avoid these issues, this work presents an imputation-free and fusion-free deep IMVC framework. First, the proposed method builds a deep embedding feature learning and clustering model for each view individually. Our method then nonlinearly maps the embedding features of complete data into a high-dimensional space to discover linear separability. Concretely, this paper provides an implementation of the high-dimensional mapping as well as shows the mechanism to mine the multi-view cluster complementarity. This complementary information is then transformed to the supervised information with high confidence, aiming to achieve the multi-view clustering consistency for the complete data and incomplete data. Furthermore, we design an EM-like optimization strategy to alternately promote feature learning and clustering. Extensive experiments on real-world multi-view datasets demonstrate that our method achieves superior clustering performance over state-of-the-art methods.

----

## [984] Linearity-Aware Subspace Clustering

**Authors**: *Yesong Xu, Shuo Chen, Jun Li, Jianjun Qian*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20857](https://doi.org/10.1609/aaai.v36i8.20857)

**Abstract**:

Obtaining a good similarity matrix is extremely important in subspace clustering. Current state-of-the-art methods learn the similarity matrix through self-expressive strategy. However, these methods directly adopt original samples as a set of basis to represent itself linearly. It is difficult to accurately describe the linear relation between samples in the real-world applications, and thus is hard to find an ideal similarity matrix. To better represent the linear relation of samples, we present a subspace clustering model, Linearity-Aware Subspace Clustering (LASC), which can consciously learn the similarity matrix by employing a linearity-aware metric. This is a new subspace clustering method that combines metric learning and subspace clustering into a joint learning framework. In our model, we first utilize the self-expressive strategy to obtain an initial subspace structure and discover a low-dimensional representation of the original data. Subsequently, we use the proposed metric to learn an intrinsic similarity matrix with linearity-aware on the obtained subspace. Based on such a learned similarity matrix, the inter-cluster distance becomes larger than the intra-cluster distances, and thus successfully obtaining a good subspace cluster result. In addition, to enrich the similarity matrix with more consistent knowledge, we adopt a collaborative learning strategy for self-expressive subspace learning and linearity-aware subspace learning. Moreover, we provide detailed mathematical analysis to show that the metric can properly characterize the linear correlation between samples.

----

## [985] Go Wider Instead of Deeper

**Authors**: *Fuzhao Xue, Ziji Shi, Futao Wei, Yuxuan Lou, Yong Liu, Yang You*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20858](https://doi.org/10.1609/aaai.v36i8.20858)

**Abstract**:

More transformer blocks with residual connections have recently achieved impressive results on various tasks. To achieve better performance with fewer trainable parameters, recent methods are proposed to go shallower by parameter sharing or model compressing along with the depth. However, weak modeling capacity limits their performance. Contrastively, going wider by inducing more trainable matrixes and parameters would produce a huge model requiring advanced parallelism to train and inference. 
 
 
 In this paper, we propose a parameter-efficient framework, going wider instead of deeper. Specially, following existing works, we adapt parameter sharing to compress along depth. But, such deployment would limit the performance. To maximize modeling capacity, we scale along model width by replacing feed-forward network (FFN) with mixture-of-experts (MoE). Across transformer blocks, instead of sharing normalization layers, we propose to use individual layernorms to transform various semantic representations in a more parameter-efficient way. To evaluate our plug-and-run framework, we design WideNet and conduct comprehensive experiments on popular computer vision and natural language processing benchmarks. On ImageNet-1K, our best model outperforms Vision Transformer (ViT) by 1.5% with 0.72 times trainable parameters. Using 0.46 times and 0.13 times parameters, our WideNet can still surpass ViT and ViT-MoE by 0.8% and 2.1%, respectively. On four natural language processing datasets, WideNet outperforms ALBERT by 1.8% on average and surpass BERT using factorized embedding parameterization by 0.8% with fewer parameters.

----

## [986] Seizing Critical Learning Periods in Federated Learning

**Authors**: *Gang Yan, Hao Wang, Jian Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20859](https://doi.org/10.1609/aaai.v36i8.20859)

**Abstract**:

Federated learning (FL) is a popular technique to train machine learning (ML) models with decentralized data. Extensive works have studied the performance of the global model; however, it is still unclear how the training process affects the final test accuracy. Exacerbating this problem is the fact that FL executions differ significantly from traditional ML with heterogeneous data characteristics across clients, involving more hyperparameters. In this work, we show that the final test accuracy of FL is dramatically affected by the early phase of the training process, i.e., FL exhibits critical learning periods, in which small gradient errors can have irrecoverable impact on the final test accuracy. To further explain this phenomenon, we generalize the trace of the Fisher Information Matrix (FIM) to FL and define a new notation called FedFIM, a quantity reflecting the local curvature of each clients from the beginning of the training in FL. Our findings suggest that the initial learning phase plays a critical role in understanding the FL performance. This is in contrast to many existing works which generally do not connect the final accuracy of FL to the early phase training.  Finally, seizing critical learning periods in FL is of independent interest and could be useful for other problems such as the choices of hyperparameters including but not limited to the number of client selected per round, batch size, so as to improve the performance of FL training and testing.

----

## [987] Learning to Identify Top Elo Ratings: A Dueling Bandits Approach

**Authors**: *Xue Yan, Yali Du, Binxin Ru, Jun Wang, Haifeng Zhang, Xu Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20860](https://doi.org/10.1609/aaai.v36i8.20860)

**Abstract**:

The Elo rating system is widely adopted to evaluate the skills of (chess) game and sports players. Recently it has been also integrated into machine learning algorithms in evaluating the performance of computerised AI agents. However, an accurate estimation of the Elo rating (for the top players) often requires many rounds of competitions, which can be expensive to carry out. In this paper, to minimize the number of comparisons and to improve the sample efficiency of the Elo evaluation (for top players), we propose an efficient online match scheduling algorithm. Specifically, we identify and match the top players through a dueling bandits framework and tailor the bandit algorithm to the gradient-based update of Elo. We show that it reduces the per-step memory and time complexity to constant, compared to the traditional likelihood maximization approaches requiring O(t) time. Our algorithm has a regret guarantee that is sublinear in the number of competition rounds and has been extended to the multidimensional Elo ratings for handling intransitive games. We empirically demonstrate that our method achieves superior convergence speed and time efficiency on a variety of gaming tasks.

----

## [988] Q-Ball: Modeling Basketball Games Using Deep Reinforcement Learning

**Authors**: *Chen Yanai, Adir Solomon, Gilad Katz, Bracha Shapira, Lior Rokach*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20861](https://doi.org/10.1609/aaai.v36i8.20861)

**Abstract**:

Basketball is one of the most popular types of sports in the world. Recent technological developments have made it possible to collect large amounts of data on the game, analyze it, and discover new insights. We propose a novel approach for modeling basketball games using deep reinforcement learning. By analyzing multiple aspects of both the players and the game, we are able to model the latent connections among players' movements, actions, and performance, into a single measure - the Q-Ball. Using Q-Ball, we are able to assign scores to the performance of both players and whole teams. Our approach has multiple practical applications, including evaluating and improving players' game decisions and producing tactical recommendations. We train and evaluate our approach on a large dataset of National Basketball Association games, and show that the Q-Ball is capable of accurately assessing the performance of players and teams. Furthermore, we show that Q-Ball is highly effective in recommending alternatives to players' actions.

----

## [989] Training a Resilient Q-network against Observational Interference

**Authors**: *Chao-Han Huck Yang, I-Te Danny Hung, Yi Ouyang, Pin-Yu Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20862](https://doi.org/10.1609/aaai.v36i8.20862)

**Abstract**:

Deep reinforcement learning (DRL) has demonstrated impressive performance in various gaming simulators and real-world applications.
 In practice, however, a DRL agent may receive faulty observation by abrupt interferences such as black-out, frozen-screen, and adversarial perturbation. How to design a resilient DRL algorithm against these rare but mission-critical and safety-crucial scenarios is an essential yet challenging task. In this paper, we consider a deep q-network (DQN) framework training with an auxiliary task of observational interferences such as artificial noises. Inspired by causal inference for observational interference, we propose a causal inference based DQN algorithm called causal inference Q-network (CIQ). We evaluate the performance of CIQ in several benchmark DQN environments with different types of interferences as auxiliary labels. Our experimental results show that the proposed CIQ method could achieve higher performance and more resilience against observational interferences.

----

## [990] Policy Optimization with Stochastic Mirror Descent

**Authors**: *Long Yang, Yu Zhang, Gang Zheng, Qian Zheng, Pengfei Li, Jianhang Huang, Gang Pan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20863](https://doi.org/10.1609/aaai.v36i8.20863)

**Abstract**:

Improving sample efficiency has been a longstanding goal in reinforcement learning. This paper proposes VRMPO algorithm: a sample efficient policy gradient method with stochastic mirror descent. In VRMPO, a novel variance-reduced policy gradient estimator is presented to improve sample efficiency. We prove that the proposed VRMPO needs only O(ε−3) sample trajectories to achieve an ε-approximate first-order stationary point, which matches the best sample complexity for policy optimization. Extensive empirical results demonstrate that VRMP outperforms the state-of-the-art policy gradient methods in various settings.

----

## [991] Graph Pointer Neural Networks

**Authors**: *Tianmeng Yang, Yujing Wang, Zhihan Yue, Yaming Yang, Yunhai Tong, Jing Bai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20864](https://doi.org/10.1609/aaai.v36i8.20864)

**Abstract**:

Graph Neural Networks (GNNs) have shown advantages in various graph-based applications. Most existing GNNs assume strong homophily of graph structure and apply permutation-invariant local aggregation of neighbors to learn a representation for each node. However, they fail to generalize to heterophilic graphs, where most neighboring nodes have different labels or features, and the relevant nodes are distant. Few recent studies attempt to address this problem by combining multiple hops of hidden representations of central nodes (i.e., multi-hop-based approaches) or sorting the neighboring nodes based on attention scores (i.e., ranking-based approaches). As a result, these approaches have some apparent limitations. On the one hand, multi-hop-based approaches do not explicitly distinguish relevant nodes from a large number of multi-hop neighborhoods, leading to a severe over-smoothing problem. On the other hand, ranking-based models do not joint-optimize node ranking with end tasks and result in sub-optimal solutions. In this work, we present Graph Pointer Neural Networks (GPNN) to tackle the challenges mentioned above. We leverage a pointer network to select the most relevant nodes from a large amount of multi-hop neighborhoods, which constructs an ordered sequence according to the relationship with the central node. 1D convolution is then applied to extract high-level features from the node sequence. The pointer-network-based ranker in GPNN is joint-optimized with other parts in an end-to-end manner. Extensive experiments are conducted on six public node classification datasets with heterophilic graphs. The results show that GPNN significantly improves the classification performance of state-of-the-art methods. In addition, analyses also reveal the privilege of the proposed GPNN in filtering out irrelevant neighbors and reducing over-smoothing.

----

## [992] LOGICDEF: An Interpretable Defense Framework against Adversarial Examples via Inductive Scene Graph Reasoning

**Authors**: *Yuan Yang, James Clayton Kerce, Faramarz Fekri*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20865](https://doi.org/10.1609/aaai.v36i8.20865)

**Abstract**:

Deep vision models have provided new capability across a spectrum of applications in transportation, manufacturing, agriculture, commerce, and security. However, recent studies have demonstrated that these models are vulnerable to adversarial attack, exposing a risk-of-use in critical applications where untrusted parties have access to the data environment or even directly to the sensor inputs. Existing adversarial defense methods are either limited to specific types of attacks or are too complex to be applied to practical vision models. More importantly, these methods rely on techniques that are not interpretable to humans. In this work, we argue that an effective defense should produce an explanation as to why the system is attacked, and by using a representation that is easily readable by a human user, e.g. a logic formalism. To this end, we propose logic adversarial defense (LogicDef), a defense framework that utilizes the scene graph of the image to provide a contextual structure for detecting and explaining object classification. Our framework first mines inductive logic rules from the extracted scene graph, and then uses these rules to construct a defense model that alerts the user when the vision model violates the consistency rules. The defense model is interpretable and its robustness is further enhanced by incorporating existing relational commonsense knowledge from projects such as ConceptNet. In order to handle the hierarchical nature of such relational reasoning, we use a curriculum learning approach based on object taxonomy, yielding additional improvements to training and performance.

----

## [993] Exploiting Invariance in Training Deep Neural Networks

**Authors**: *Chengxi Ye, Xiong Zhou, Tristan McKinney, Yanfeng Liu, Qinggang Zhou, Fedor Zhdanov*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20866](https://doi.org/10.1609/aaai.v36i8.20866)

**Abstract**:

Inspired by two basic mechanisms in animal visual systems, we introduce a feature transform technique that imposes invariance properties in the training of deep neural networks. The resulting algorithm requires less parameter tuning, trains well with an initial learning rate 1.0, and easily generalizes to different tasks. We enforce scale invariance with local statistics in the data to align similar samples at diverse scales. To accelerate convergence, we enforce a GL(n)-invariance property with global statistics extracted from a batch such that the gradient descent solution should remain invariant under basis change. Profiling analysis shows our proposed modifications takes 5% of the computations of the underlying convolution layer. Tested on convolutional networks and transformer networks, our proposed technique requires fewer iterations to train, surpasses all baselines by a large margin, seamlessly works on both small and large batch size training, and applies to different computer vision and language tasks.

----

## [994] Lifelong Generative Modelling Using Dynamic Expansion Graph Model

**Authors**: *Fei Ye, Adrian G. Bors*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20867](https://doi.org/10.1609/aaai.v36i8.20867)

**Abstract**:

Variational Autoencoders (VAEs) suffer from degenerated performance, when learning several successive tasks. This is caused by catastrophic forgetting. In order to address the knowledge loss, VAEs are using either Generative Replay (GR) mechanisms or Expanding Network Architectures (ENA). In this paper we study the forgetting behaviour of VAEs using a joint GR and ENA methodology, by deriving an upper bound on the negative marginal log-likelihood. This theoretical analysis provides new insights into how VAEs forget the previously learnt knowledge during lifelong learning. The analysis indicates the best performance achieved when considering model mixtures, under the ENA framework, where there are no restrictions on the number of components. However, an ENA-based approach may require an excessive number of parameters. This motivates us to propose a novel Dynamic Expansion Graph Model (DEGM). DEGM expands its architecture, according to the novelty associated with each new database, when compared to the information already learnt by the network from previous tasks. DEGM training optimizes knowledge structuring, characterizing the joint probabilistic representations corresponding to the past and more recently learned tasks. We demonstrate that DEGM guarantees optimal performance for each task while also minimizing the required number of parameters.

----

## [995] Stage Conscious Attention Network (SCAN): A Demonstration-Conditioned Policy for Few-Shot Imitation

**Authors**: *Jia-Fong Yeh, Chi-Ming Chung, Hung-Ting Su, Yi-Ting Chen, Winston H. Hsu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20868](https://doi.org/10.1609/aaai.v36i8.20868)

**Abstract**:

In few-shot imitation learning (FSIL), using behavioral cloning (BC) to solve unseen tasks with few expert demonstrations becomes a popular research direction. The following capabilities are essential in robotics applications: (1) Behaving in compound tasks that contain multiple stages. (2) Retrieving knowledge from few length-variant and misalignment demonstrations. (3) Learning from an expert different from the agent. No previous work can achieve these abilities at the same time. In this work, we conduct FSIL problem under the union of above settings and introduce a novel stage conscious attention network (SCAN) to retrieve knowledge from few demonstrations simultaneously. SCAN uses an attention module to identify each stage in length-variant demonstrations. Moreover, it is designed under demonstration-conditioned policy that learns the relationship between experts and agents. Experiment results show that SCAN can perform in complicated compound tasks without fine-tuning and provide the explainable visualization. Project page is at https://sites.google.com/view/scan-aaai2022.

----

## [996] BATUDE: Budget-Aware Neural Network Compression Based on Tucker Decomposition

**Authors**: *Miao Yin, Huy Phan, Xiao Zang, Siyu Liao, Bo Yuan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20869](https://doi.org/10.1609/aaai.v36i8.20869)

**Abstract**:

Model compression is very important for the efficient deployment of deep neural network (DNN) models on resource-constrained devices. Among various model compression approaches, high-order tensor decomposition is particularly attractive and useful because the decomposed model is very small and fully structured. For this category of approaches, tensor ranks are the most important hyper-parameters that directly determine the architecture and task performance of the compressed DNN models. However, as an NP-hard problem, selecting optimal tensor ranks under the desired budget is very challenging and the state-of-the-art studies suffer from unsatisfied compression performance and timing-consuming search procedures. To systematically address this fundamental problem, in this paper we propose BATUDE, a Budget-Aware TUcker DEcomposition-based compression approach that can efficiently calculate optimal tensor ranks via one-shot training. By integrating the rank selecting procedure to the DNN training process with a specified compression budget, the tensor ranks of the DNN models are learned from the data and thereby bringing very significant improvement on both compression ratio and classification accuracy for the compressed models. The experimental results on ImageNet dataset show that our method enjoys 0.33% top-5 higher accuracy with 2.52X less computational cost as compared to the uncompressed ResNet-18 model. For ResNet-50, the proposed approach enables 0.37% and 0.55% top-5 accuracy increase with 2.97X and 2.04X computational cost reduction, respectively, over the uncompressed model.

----

## [997] Distributed Randomized Sketching Kernel Learning

**Authors**: *Rong Yin, Yong Liu, Dan Meng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20870](https://doi.org/10.1609/aaai.v36i8.20870)

**Abstract**:

We investigate the statistical and computational requirements for distributed kernel ridge regression with randomized sketching (DKRR-RS) and successfully achieve the optimal learning rates with only a fraction of computations. More precisely, the proposed DKRR-RS combines sparse randomized sketching, divide-and-conquer and KRR to scale up kernel methods and successfully derives the same learning rate as the exact KRR with greatly reducing computational costs in expectation, at the basic setting, which outperforms previous state of the art solutions. Then, for the sake of the gap between theory and experiments, we derive the optimal learning rate in probability for DKRR-RS to reflect its generalization performance. Finally, to further improve the learning performance, we construct an efficient communication strategy for DKRR-RS and demonstrate the power of communications via theoretical assessment. An extensive experiment validates the effectiveness of DKRR-RS and the communication strategy on real datasets.

----

## [998] AutoGCL: Automated Graph Contrastive Learning via Learnable View Generators

**Authors**: *Yihang Yin, Qingzhong Wang, Siyu Huang, Haoyi Xiong, Xiang Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20871](https://doi.org/10.1609/aaai.v36i8.20871)

**Abstract**:

Contrastive learning has been widely applied to graph representation learning, where the view generators play a vital role in generating effective contrastive samples. Most of the existing contrastive learning methods employ pre-defined view generation methods, e.g., node drop or edge perturbation, which usually cannot adapt to input data or preserve the original semantic structures well. To address this issue, we propose a novel framework named Automated Graph Contrastive Learning (AutoGCL) in this paper. Specifically, AutoGCL employs a set of learnable graph view generators orchestrated by an auto augmentation strategy, where every graph view generator learns a probability distribution of graphs conditioned by the input. While the graph view generators in AutoGCL preserve the most representative structures of the original graph in generation of every contrastive sample, the auto augmentation learns policies to introduce adequate augmentation variances in the whole contrastive learning procedure. Furthermore, AutoGCL adopts a joint training strategy to train the learnable view generators, the graph encoder, and the classifier in an end-to-end manner, resulting in topological heterogeneity yet semantic similarity in the generation of contrastive samples. Extensive experiments on semi-supervised learning, unsupervised learning, and transfer learning demonstrate the superiority of our AutoGCL framework over the state-of-the-arts in graph contrastive learning. In addition, the visualization results further confirm that the learnable view generators can deliver more compact and semantically meaningful contrastive samples compared against the existing view generation methods. Our code is available at https://github.com/Somedaywilldo/AutoGCL.

----

## [999] BM-NAS: Bilevel Multimodal Neural Architecture Search

**Authors**: *Yihang Yin, Siyu Huang, Xiang Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20872](https://doi.org/10.1609/aaai.v36i8.20872)

**Abstract**:

Deep neural networks (DNNs) have shown superior performances on various multimodal learning problems. However, it often requires huge efforts to adapt DNNs to individual multimodal tasks by manually engineering unimodal features and designing multimodal feature fusion strategies. This paper proposes Bilevel Multimodal Neural Architecture Search (BM-NAS) framework, which makes the architecture of multimodal fusion models fully searchable via a bilevel searching scheme. At the upper level, BM-NAS selects the inter/intra-modal feature pairs from the pretrained unimodal backbones. At the lower level, BM-NAS learns the fusion strategy for each feature pair, which is a combination of predefined primitive operations. The primitive operations are elaborately designed and they can be flexibly combined to accommodate various effective feature fusion modules such as multi-head attention (Transformer) and Attention on Attention (AoA). Experimental results on three multimodal tasks demonstrate the effectiveness and efficiency of the proposed BM-NAS framework. BM-NAS achieves competitive performances with much less search time and fewer model parameters in comparison with the existing generalized multimodal NAS methods. Our code is available at https://github.com/Somedaywilldo/BM-NAS.

----



[Go to the previous page](AAAI-2022-list04.md)

[Go to the next page](AAAI-2022-list06.md)

[Go to the catalog section](README.md)