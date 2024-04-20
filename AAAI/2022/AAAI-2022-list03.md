## [400] I Can Find You! Boundary-Guided Separated Attention Network for Camouflaged Object Detection

**Authors**: *Hongwei Zhu, Peng Li, Haoran Xie, Xuefeng Yan, Dong Liang, Dapeng Chen, Mingqiang Wei, Jing Qin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20273](https://doi.org/10.1609/aaai.v36i3.20273)

**Abstract**:

Can you find me? By simulating how humans to discover the so-called 'perfectly'-camouflaged object, we present a novel boundary-guided separated attention network (call BSA-Net). Beyond the existing camouflaged object detection (COD) wisdom, BSA-Net utilizes two-stream separated attention modules to highlight the separator (or say the camouflaged object's boundary) between an image's background and foreground: the reverse attention stream helps erase the camouflaged object's interior to focus on the background, while the normal attention stream recovers the interior and thus pay more attention to the foreground; and both streams are followed by a boundary guider module and combined to strengthen the understanding of boundary. The core design of such separated attention is motivated by the COD procedure of humans: find the subtle difference between the foreground and background to delineate the boundary of a camouflaged object, then the boundary can help further enhance the COD accuracy. We validate on three benchmark datasets that the proposed BSA-Net is very beneficial to detect camouflaged objects with the blurred boundaries and similar colors/patterns with their backgrounds. Extensive results exhibit very clear COD improvements on our BSA-Net over sixteen SOTAs.

----

## [401] MoCaNet: Motion Retargeting In-the-Wild via Canonicalization Networks

**Authors**: *Wentao Zhu, Zhuoqian Yang, Ziang Di, Wayne Wu, Yizhou Wang, Chen Change Loy*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20274](https://doi.org/10.1609/aaai.v36i3.20274)

**Abstract**:

We present a novel framework that brings the 3D motion retargeting task from controlled environments to in-the-wild scenarios. In particular, our method is capable of retargeting body motion from a character in a 2D monocular video to a 3D character without using any motion capture system or 3D reconstruction procedure. It is designed to leverage massive online videos for unsupervised training, needless of 3D annotations or motion-body pairing information. The proposed method is built upon two novel canonicalization operations, structure canonicalization and view canonicalization. Trained with the canonicalization operations and the derived regularizations, our method learns to factorize a skeleton sequence into three independent semantic subspaces, i.e., motion, structure, and view angle. The disentangled representation enables motion retargeting from 2D to 3D with high precision. Our method achieves superior performance on motion transfer benchmarks with large body variations and challenging actions. Notably, the canonicalized skeleton sequence could serve as a disentangled and interpretable representation of human motion that benefits action analysis and motion retrieval.

----

## [402] Robust Depth Completion with Uncertainty-Driven Loss Functions

**Authors**: *Yufan Zhu, Weisheng Dong, Leida Li, Jinjian Wu, Xin Li, Guangming Shi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20275](https://doi.org/10.1609/aaai.v36i3.20275)

**Abstract**:

Recovering a dense depth image from sparse LiDAR scans is a challenging task. Despite the popularity of color-guided methods for sparse-to-dense depth completion, they treated pixels equally during optimization, ignoring the uneven distribution characteristics in the sparse depth map and the accumulated outliers in the synthesized ground truth. In this work, we introduce uncertainty-driven loss functions to improve the robustness of depth completion and handle the uncertainty in depth completion. Specifically, we propose an explicit uncertainty formulation for robust depth completion with Jeffrey's prior. A parametric uncertain-driven loss is introduced and translated to new loss functions that are robust to noisy or missing data. Meanwhile, we propose a multiscale joint prediction model that can simultaneously predict depth and uncertainty maps. The estimated uncertainty map is also used to perform adaptive prediction on the pixels with high uncertainty, leading to a residual map for refining the completion results. Our method has been tested on KITTI Depth Completion Benchmark and achieved the state-of-the-art robustness performance in terms of MAE, IMAE, and IRMSE metrics.

----

## [403] Efficient Model-Driven Network for Shadow Removal

**Authors**: *Yurui Zhu, Zeyu Xiao, Yanchi Fang, Xueyang Fu, Zhiwei Xiong, Zheng-Jun Zha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20276](https://doi.org/10.1609/aaai.v36i3.20276)

**Abstract**:

Deep Convolutional Neural Networks (CNNs) based methods have achieved significant breakthroughs in the task of single image shadow removal. However, the performance of these methods remains limited for several reasons. First, the existing shadow illumination model ignores the spatially variant property of the shadow images, hindering their further performance. Second, most deep CNNs based methods directly estimate the shadow free results from the input shadow images like a black box, thus losing the desired interpretability. To address these issues, we first propose a new shadow illumination model for the shadow removal task. This new shadow illumination model ensures the identity mapping among unshaded regions, and adaptively performs fine grained spatial mapping between shadow regions and their references. Then, based on the shadow illumination model, we reformulate the shadow removal task as a variational optimization problem. To effectively solve the variational problem, we design an iterative algorithm and unfold it into a deep network, naturally increasing the interpretability of the deep model. Experiments show that our method could achieve SOTA performance with less than half parameters, one-fifth of floating-point of operations (FLOPs), and over seventeen times faster than SOTA method (DHAN).

----

## [404] Learning Disentangled Classification and Localization Representations for Temporal Action Localization

**Authors**: *Zixin Zhu, Le Wang, Wei Tang, Ziyi Liu, Nanning Zheng, Gang Hua*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20277](https://doi.org/10.1609/aaai.v36i3.20277)

**Abstract**:

A common approach to Temporal Action Localization (TAL) is to generate action proposals and then perform action classification and localization on them. For each proposal, existing methods universally use a shared proposal-level representation for both tasks. However, our analysis indicates that this shared representation focuses on the most discriminative frames for classification, e.g., ``take-offs" rather than ``run-ups" in distinguishing ``high jump" and ``long jump", while frames most relevant to localization, such as the start and end frames of an action, are largely ignored. 
 In other words, such a shared representation can not simultaneously handle both classification and localization tasks well, and it makes precise TAL difficult. 
 To address this challenge, this paper disentangles the shared representation into classification and localization representations. The disentangled classification representation focuses on the most discriminative frames, and the disentangled localization representation focuses on the action phase as well as the action start and end. Our model could be divided into two sub-networks, i.e., the disentanglement network and the context-based aggregation network. The disentanglement network is an autoencoder to learn orthogonal hidden variables of classification and localization. The context-based aggregation network aggregates the classification and localization representations by modeling local and global contexts. We evaluate our proposed method on two popular benchmarks for TAL, which outperforms all state-of-the-art methods.

----

## [405] ACDNet: Adaptively Combined Dilated Convolution for Monocular Panorama Depth Estimation

**Authors**: *Chuanqing Zhuang, Zhengda Lu, Yiqun Wang, Jun Xiao, Ying Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20278](https://doi.org/10.1609/aaai.v36i3.20278)

**Abstract**:

Depth estimation is a crucial step for 3D reconstruction with panorama images in recent years. Panorama images maintain the complete spatial information but introduce distortion with equirectangular projection. In this paper, we propose an ACDNet based on the adaptively combined dilated convolution to predict the dense depth map for a monocular panoramic image. Specifically, we combine the convolution kernels with different dilations to extend the receptive field in the equirectangular projection. Meanwhile, we introduce an adaptive channel-wise fusion module to summarize the feature maps and get diverse attention areas in the receptive field along the channels. Due to the utilization of channel-wise attention in constructing the adaptive channel-wise fusion module, the network can capture and leverage the cross-channel contextual information efficiently. Finally, we conduct depth estimation experiments on three datasets (both virtual and real-world) and the experimental results demonstrate that our proposed ACDNet substantially outperforms the current state-of-the-art (SOTA) methods. Our codes and model parameters are accessed in https://github.com/zcq15/ACDNet.

----

## [406] Making Adversarial Examples More Transferable and Indistinguishable

**Authors**: *Junhua Zou, Yexin Duan, Boyu Li, Wu Zhang, Yu Pan, Zhisong Pan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20279](https://doi.org/10.1609/aaai.v36i3.20279)

**Abstract**:

Fast gradient sign attack series are popular methods that are used to generate adversarial examples. However, most of the approaches based on fast gradient sign attack series cannot balance the indistinguishability and transferability due to the limitations of the basic sign structure. To address this problem, we propose a method, called Adam Iterative Fast Gradient Tanh Method (AI-FGTM), to generate indistinguishable adversarial examples with high transferability. Besides, smaller kernels and dynamic step size are also applied to generate adversarial examples for further increasing the attack success rates. Extensive experiments on an ImageNet-compatible dataset show that our method generates more indistinguishable adversarial examples and achieves higher attack success rates without extra running time and resource. Our best transfer-based attack NI-TI-DI-AITM can fool six classic defense models with an average success rate of 89.3% and three advanced defense models with an average success rate of 82.7%, which are higher than the state-of-the-art gradient-based attacks. Additionally, our method can also reduce nearly 20% mean perturbation. We expect that our method will serve as a new baseline for generating adversarial examples with better transferability and indistinguishability.

----

## [407] Undercover Boolean Matrix Factorization with MaxSAT

**Authors**: *Florent Avellaneda, Roger Villemaire*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20280](https://doi.org/10.1609/aaai.v36i4.20280)

**Abstract**:

The k-undercover Boolean matrix factorization problem aims to approximate a m×n Boolean matrix X as the Boolean product of an m×k and a k×n matrices A◦B such that X is a cover of A◦B, i.e., no representation error is allowed on the 0’s entries of the matrix X. To infer an optimal and “block-optimal” k-undercover, we propose two exact methods based on MaxSAT encodings. From a theoretical standpoint, we prove that our method of inferring “block-optimal” k-undercover is a (1 - 1/e) ≈ 0.632 approximation for the optimal k-undercover problem. From a practical standpoint, experimental results indicate that our “block-optimal” k-undercover algorithm outperforms the state-of-the-art even when compared with algorithms for the more general k-undercover Boolean Matrix Factorization problem for which only minimizing reconstruction error is required.

----

## [408] Achieving Zero Constraint Violation for Constrained Reinforcement Learning via Primal-Dual Approach

**Authors**: *Qinbo Bai, Amrit Singh Bedi, Mridul Agarwal, Alec Koppel, Vaneet Aggarwal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20281](https://doi.org/10.1609/aaai.v36i4.20281)

**Abstract**:

Reinforcement learning is widely used in applications where one needs to perform sequential decisions while interacting with the environment. The problem becomes more challenging when the decision requirement includes satisfying some safety constraints. The problem is mathematically formulated as constrained Markov decision process (CMDP). In the literature, various algorithms are available to solve CMDP problems in a model-free manner to achieve epsilon-optimal cumulative reward with epsilon feasible policies. An epsilon-feasible policy implies that it suffers from constraint violation. An important question here is whether we can achieve epsilon-optimal cumulative reward with zero constraint violations or not. To achieve that, we advocate the use of a randomized primal-dual approach to solve the CMDP problems and propose a conservative stochastic primal-dual algorithm (CSPDA) which is shown to exhibit O(1/epsilon^2) sample complexity to achieve epsilon-optimal cumulative reward with zero constraint violations. In the prior works, the best available sample complexity for the epsilon-optimal policy with zero constraint violation is O(1/epsilon^5). Hence, the proposed algorithm provides a significant improvement compared to the state of the art.

----

## [409] GEQCA: Generic Qualitative Constraint Acquisition

**Authors**: *Mohamed-Bachir Belaid, Nassim Belmecheri, Arnaud Gotlieb, Nadjib Lazaar, Helge Spieker*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20282](https://doi.org/10.1609/aaai.v36i4.20282)

**Abstract**:

Many planning, scheduling or multi-dimensional packing problems involve the design of subtle logical combinations of temporal or spatial constraints. 
 On the one hand, the precise modelling of these constraints, which are formulated in various relation algebras, entails a number of possible logical combinations and requires expertise in constraint-based modelling.
 On the other hand, active constraint acquisition (CA) has been used successfully to support non-experienced users in learning conjunctive constraint networks through the generation of a sequence of queries. 
 In this paper, we propose GEACQ, which stands for Generic Qualitative Constraint Acquisition, an active CA method that learns qualitative constraints 
 via the concept of qualitative queries.
 GEACQ combines qualitative queries with time-bounded path consistency (PC) and background knowledge propagation to acquire the qualitative constraints of any scheduling or packing problem.
 We prove soundness, completeness and termination of GEACQ by exploiting the jointly exhaustive and pairwise disjoint property of qualitative calculus and we give an experimental evaluation that shows (i) the efficiency of our approach in learning temporal constraints and, (ii) the use of GEACQ on real scheduling instances.

----

## [410] Certified Symmetry and Dominance Breaking for Combinatorial Optimisation

**Authors**: *Bart Bogaerts, Stephan Gocht, Ciaran McCreesh, Jakob Nordström*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20283](https://doi.org/10.1609/aaai.v36i4.20283)

**Abstract**:

Symmetry and dominance breaking can be crucial for solving hard combinatorial search and optimisation problems, but the correctness of these techniques sometimes relies on subtle arguments. For this reason, it is desirable to produce efficient, machine-verifiable certificates that solutions have been computed correctly.
Building on the cutting planes proof system, we develop a certification method  for optimisation problems in which symmetry and dominance breaking are easily expressible. Our experimental evaluation demonstrates that we can efficiently verify fully general symmetry breaking in Boolean satisfiability (SAT) solving, thus providing, for the first time, a unified method to certify a range of advanced SAT techniques that also includes XOR and cardinality reasoning. In addition, we apply our method to maximum clique solving and constraint programming as a proof of concept that the approach applies to a wider range of combinatorial problems.

----

## [411] The Perils of Learning Before Optimizing

**Authors**: *Chris Cameron, Jason S. Hartford, Taylor Lundy, Kevin Leyton-Brown*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20284](https://doi.org/10.1609/aaai.v36i4.20284)

**Abstract**:

Formulating real-world optimization problems often begins with making predictions from historical data (e.g., an optimizer that aims to recommend fast routes relies upon travel-time predictions). Typically, learning the prediction model used to generate the optimization problem and solving that problem are performed in two separate stages. Recent work has showed how such prediction models can be learned end-to-end by differentiating through the optimization task. Such methods often yield empirical improvements, which are typically attributed to end-to-end making better error tradeoffs than the standard loss function used in a two-stage solution. We refine this explanation and more precisely characterize when end-to-end can improve performance. When prediction targets are stochastic, a two-stage solution must make an a priori choice about which statistics of the target distribution to model---we consider expectations over prediction targets---while an end-to-end solution can make this choice adaptively. We show that the performance gap between a two-stage and end-to-end approach is closely related to the \emph{price of correlation} concept in stochastic optimization and show the implications of some existing POC results for the predict-then-optimize problem. We then consider a novel and particularly practical setting, where multiple prediction targets are combined to obtain each of the objective function’s coefficients. We give explicit constructions where (1) two-stage performs unboundedly worse than end-to-end; and (2) two-stage is optimal. 
 We use simulations to experimentally quantify performance gaps and identify a wide range of real-world applications from the literature whose objective functions rely on multiple prediction targets, suggesting that end-to-end learning could yield significant improvements.

----

## [412] A Lyapunov-Based Methodology for Constrained Optimization with Bandit Feedback

**Authors**: *Semih Cayci, Yilin Zheng, Atilla Eryilmaz*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20285](https://doi.org/10.1609/aaai.v36i4.20285)

**Abstract**:

In a wide variety of applications including online advertising, contractual hiring, and wireless scheduling, the controller is constrained by a stringent budget constraint on the available resources, which are consumed in a random amount by each action, and a stochastic feasibility constraint that may impose important operational limitations on decision-making. In this work, we consider a general model to address such problems, where each action returns a random reward, cost, and penalty from an unknown joint distribution, and the decision-maker aims to maximize the total reward under a budget constraint B on the total cost and a stochastic constraint on the time-average penalty. We propose a novel low-complexity algorithm based on Lyapunov optimization methodology, named LyOn, and prove that for K arms it achieves square root of KBlog(B) regret and zero constraint-violation when B is sufficiently large. The low computational cost and sharp performance bounds of LyOn suggest that Lyapunov-based algorithm design methodology can be effective in solving constrained bandit optimization problems.

----

## [413] Resolving Inconsistencies in Simple Temporal Problems: A Parameterized Approach

**Authors**: *Konrad K. Dabrowski, Peter Jonsson, Sebastian Ordyniak, George Osipov*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20286](https://doi.org/10.1609/aaai.v36i4.20286)

**Abstract**:

The simple temporal problem (STP) is one of the most influential reasoning formalisms for representing temporal information in AI. We study the problem of resolving inconsistency of data encoded in the STP. We prove that the problem of identifying a maximally large consistent subset of data is NP-hard. In practical instances, it is reasonable to assume that the amount of erroneous data is small. We therefore parameterize by the number of constraints that need to be removed to achieve consistency. Using tools from parameterized complexity we design fixed-parameter tractable algorithms for two large fragments of the STP. Our main algorithmic results employ reductions to the Directed Subset Feedback Arc Set problem and  iterative compression combined with an efficient algorithm for the Edge Multicut problem. We complement our algorithmic results with hardness results that rule out fixed-parameter tractable algorithms for all remaining non-trivial fragments of the STP (under standard complexity-theoretic assumptions). Together, our results give a full classification of the classical and parameterized complexity of the problem.

----

## [414] Efficient Riemannian Meta-Optimization by Implicit Differentiation

**Authors**: *Xiaomeng Fan, Yuwei Wu, Zhi Gao, Yunde Jia, Mehrtash Harandi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20287](https://doi.org/10.1609/aaai.v36i4.20287)

**Abstract**:

To solve optimization problems with nonlinear constrains, the recently developed Riemannian meta-optimization methods show promise, which train neural networks as an optimizer to perform optimization on Riemannian manifolds.
 A key challenge is the heavy computational and memory burdens, because computing the meta-gradient with respect to the optimizer involves a series of time-consuming derivatives, and stores large computation graphs in memory.
 In this paper, we propose an efficient Riemannian meta-optimization method that decouples the complex computation scheme from the meta-gradient.
 We derive Riemannian implicit differentiation to compute the meta-gradient by establishing a link between Riemannian optimization and the implicit function theorem. As a result, the updating our optimizer is only related to the final two iterations, which in turn speeds up our method and reduces the memory footprint significantly. We theoretically study the computational load and memory footprint of our method for long optimization trajectories, and conduct an empirical study to demonstrate the benefits of the proposed method. Evaluations of three optimization problems on different Riemannian manifolds show that our method achieves state-of-the-art performance in terms of the convergence speed and the quality of optima.

----

## [415] Faster Algorithms for Weak Backdoors

**Authors**: *Serge Gaspers, Andrew Kaploun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20288](https://doi.org/10.1609/aaai.v36i4.20288)

**Abstract**:

A weak backdoor, or simply a backdoor, for a Boolean SAT formula F into a class of SAT formulae C is a partial truth assignment T such that F[T] is in C and satisfiability is preserved. The problem of finding a backdoor from class C1 into class C2, or WB(C1,C2), can be stated as follows: Given a formula F in C1, and a natural number k, determine whether there exists a backdoor for F into C2 assigning at most k variables. 
 
 The class 0-Val contains all Boolean formulae with at least one negative literal in each clause. We design a new algorithm for WB(3CNF, 0-Val) by reducing it to a local search variant of 3-SAT. We show that our algorithm runs in time O*(2.562^k), improving on the previous state-of-the-art of O*(2.85^k). Here, the O* notation is a variant of the big-O notation that allows to omit polynomial factors in the input size. 
 
 Next, we look at WB(3CNF, Null), where Null is the class consisting of the empty formula. This problem was known to have a trivial running time upper bound of O*(6^k) and can easily be solved in O*(3^k) time. We use a reduction to Conflict-Free-d-Hitting-Set to prove an upper bound of O*(2.2738^k), and also prove a lower bound of 2^o(k) assuming the Exponential Time Hypothesis. 
 
 Finally, Horn is the class of formulae with at most one positive literal per clause. We improve the previous O*(4.54^k) running time for WB(3CNF, Horn) problem to O*(4.17^k), by exploiting the structure of the SAT instance to give a novel proof of the non-existence of the slowest cases after a slight restructuring of the branching priorities.

----

## [416] A Divide and Conquer Algorithm for Predict+Optimize with Non-convex Problems

**Authors**: *Ali Ugur Guler, Emir Demirovic, Jeffrey Chan, James Bailey, Christopher Leckie, Peter J. Stuckey*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20289](https://doi.org/10.1609/aaai.v36i4.20289)

**Abstract**:

The predict+optimize problem combines machine learning and
 combinatorial optimization by predicting the problem coefficients first and then using these coefficients to solve the optimization problem. 
 While this problem can be solved in two separate stages, 
 recent research shows end to end models can achieve better results. 
 This requires differentiating through a discrete combinatorial function.
 Models that use differentiable surrogates are prone to approximation errors, while existing exact models are limited to dynamic programming, or they do not generalize well with scarce data. 
 In this work we propose a novel
 divide and conquer algorithm based on transition points to reason over exact optimization problems 
 and predict
 the coefficients using the optimization loss. Moreover, our model is not limited to dynamic programming problems.
 We also introduce a greedy version, which achieves similar
 results with less computation. 
 In comparison with other predict+optimize frameworks, we show our method outperforms existing exact frameworks and can reason over hard combinatorial problems better than surrogate methods.

----

## [417] Computing Diverse Shortest Paths Efficiently: A Theoretical and Experimental Study

**Authors**: *Tesshu Hanaka, Yasuaki Kobayashi, Kazuhiro Kurita, See Woo Lee, Yota Otachi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20290](https://doi.org/10.1609/aaai.v36i4.20290)

**Abstract**:

Finding diverse solutions in combinatorial problems recently has received considerable attention (Baste et al. 2020; Fomin et al. 2020; Hanaka et al. 2021). In this paper we study the following type of problems: given an integer k, the problem asks for k solutions such that the sum of pairwise (weighted) Hamming distances between these solutions is maximized. Such solutions are called diverse solutions. We present a polynomial-time algorithm for finding diverse shortest st-paths in weighted directed graphs. Moreover, we study the diverse version of other classical combinatorial problems such as diverse weighted matroid bases, diverse weighted arborescences, and diverse bipartite matchings. We show that these problems can be solved in polynomial time as well. To evaluate the practical performance of our algorithm for finding diverse shortest st-paths, we conduct a computational experiment with synthetic and real-world instances. The experiment shows that our algorithm successfully computes diverse solutions within reasonable computational time.

----

## [418] Optimizing Binary Decision Diagrams with MaxSAT for Classification

**Authors**: *Hao Hu, Marie-José Huguet, Mohamed Siala*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20291](https://doi.org/10.1609/aaai.v36i4.20291)

**Abstract**:

The growing interest in explainable artificial intelligence(XAI) for critical decision making motivates the need for interpretable machine learning (ML) models. In fact, due to their structure (especially with small sizes), these models are inherently understandable by humans. Recently, several exact methods for computing such models are proposed to overcome weaknesses of traditional heuristic methods by providing more compact models or better prediction quality. 

Despite their compressed representation of Boolean functions, Binary decision diagrams (BDDs) did not gain enough interest as other interpretable ML models. In this paper, we first propose SAT-based models for learning optimal BDDs (in terms of the number of features) that classify all input examples. Then, we lift the encoding to a MaxSAT model to learn optimal BDDs in limited depths, that maximize the number of examples correctly classified. Finally, we tackle the fragmentation problem by introducing a method to merge compatible subtrees for the BDDs found via the MaxSAT model. Our empirical study shows clear benefits of the proposed approach in terms of prediction quality and interpretability (i.e., lighter size) compared to the state-of-the-art approaches.

----

## [419] Using MaxSAT for Efficient Explanations of Tree Ensembles

**Authors**: *Alexey Ignatiev, Yacine Izza, Peter J. Stuckey, João Marques-Silva*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20292](https://doi.org/10.1609/aaai.v36i4.20292)

**Abstract**:

Tree ensembles (TEs) denote a prevalent machine learning model that do not offer guarantees of interpretability, that represent a challenge from the perspective of explainable artificial intelligence. Besides model agnostic approaches, recent work proposed to explain TEs with formally-defined explanations, which are computed with oracles for propositional satisfiability (SAT) and satisfiability modulo theories. The computation of explanations for TEs involves linear constraints to express the prediction. In practice, this deteriorates scalability of the underlying reasoners. Motivated by the inherent propositional nature of TEs, this paper proposes to circumvent the need for linear constraints and instead employ an optimization engine for pure propositional logic to efficiently handle the prediction. Concretely, the paper proposes to use a MaxSAT solver and exploit the objective function to determine a winning class. This is achieved by devising a propositional encoding for computing explanations of TEs. Furthermore, the paper proposes additional heuristics to improve the underlying MaxSAT solving procedure. Experimental results obtained on a wide range of publicly available datasets demonstrate that the proposed MaxSAT-based approach is either on par or outperforms the existing reasoning-based explainers, thus representing a robust and efficient alternative for computing formal explanations for TEs.

----

## [420] Finding Backdoors to Integer Programs: A Monte Carlo Tree Search Framework

**Authors**: *Elias B. Khalil, Pashootan Vaezipoor, Bistra Dilkina*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20293](https://doi.org/10.1609/aaai.v36i4.20293)

**Abstract**:

In Mixed Integer Linear Programming (MIP), a (strong) backdoor is a ``small" subset of an instance's integer variables with the following property: in a branch-and-bound procedure, the instance can be solved to global optimality by branching only on the variables in the backdoor. Constructing datasets of pre-computed backdoors for widely used MIP benchmark sets or particular problem families can enable new questions around novel structural properties of a MIP, or explain why a problem that is hard in theory can be solved efficiently in practice. Existing algorithms for finding backdoors rely on sampling candidate variable subsets in various ways, an approach which has demonstrated the existence of backdoors for some instances from MIPLIB2003 and MIPLIB2010. However, these algorithms fall short of consistently succeeding at the task due to an imbalance between exploration and exploitation. We propose BaMCTS, a Monte Carlo Tree Search framework for finding backdoors to MIPs. Extensive algorithmic engineering, hybridization with traditional MIP concepts, and close integration with the CPLEX solver have enabled our method to outperform baselines on MIPLIB2017 instances, finding backdoors more frequently and more efficiently.

----

## [421] Learning to Search in Local Branching

**Authors**: *Defeng Liu, Matteo Fischetti, Andrea Lodi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20294](https://doi.org/10.1609/aaai.v36i4.20294)

**Abstract**:

Finding high-quality solutions to mixed-integer linear programming problems (MILPs) is of great importance for many practical applications. In this respect, the refinement heuristic local branching (LB) has been proposed to produce improving solutions and has been highly influential for the development of local search methods in MILP. The algorithm iteratively explores a sequence of solution neighborhoods defined by the so-called local branching constraint, namely, a linear inequality limiting the distance from a reference solution. For a LB algorithm, the choice of the neighborhood size is critical to performance. Although it was initialized by a conservative value in the original LB scheme, our new observation is that the "best" size is strongly dependent on the particular MILP instance. In this work, we investigate the relation between the size of the search neighborhood and the behavior of the underlying LB algorithm, and we devise a leaning-based framework for guiding the neighborhood search of the LB heuristic. The framework consists of a two-phase strategy. For the first phase, a scaled regression model is trained to predict the size of the LB neighborhood at the first iteration through a regression task. In the second phase, we leverage reinforcement learning and devise a reinforced neighborhood search strategy to dynamically adapt the size at the subsequent iterations. We computationally show that the neighborhood size can indeed be learned, leading to improved performances and that the overall algorithm generalizes well both with respect to the instance size and, remarkably, across instances.

----

## [422] Analysis of Pure Literal Elimination Rule for Non-uniform Random (MAX) k-SAT Problem with an Arbitrary Degree Distribution

**Authors**: *Oleksii Omelchenko, Andrei A. Bulatov*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20295](https://doi.org/10.1609/aaai.v36i4.20295)

**Abstract**:

In this paper we analyse the performance of the pure literal elimination rule. We provide an equation that given an underlying degree distribution gives the number of clauses the pure literal elimination rule satisfies w.h.p. We also show how the distribution of variable degrees changes over time as the algorithm is being executed.

----

## [423] The SoftCumulative Constraint with Quadratic Penalty

**Authors**: *Yanick Ouellet, Claude-Guy Quimper*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20296](https://doi.org/10.1609/aaai.v36i4.20296)

**Abstract**:

The Cumulative constraint greatly contributes to the success of constraint programming at solving scheduling problems. The SoftCumulative, a version of the Cumulative where overloading the resource incurs a penalty is, however, less studied. We introduce a checker and a filtering algorithm for the SoftCumulative, which are inspired by the powerful energetic reasoning rule for the Cumulative. Both algorithms can be used with classic linear penalty function, but also with a quadratic penalty function, where the penalty of overloading the resource increases quadratically with the amount of the overload. We show that these algorithms are more general than existing algorithms and vastly outperform a decomposition of the SoftCumulative in practice.

----

## [424] Efficient Vertex-Oriented Polytopic Projection for Web-Scale Applications

**Authors**: *Rohan Ramanath, S. Sathiya Keerthi, Yao Pan, Konstantin Salomatin, Kinjal Basu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20297](https://doi.org/10.1609/aaai.v36i4.20297)

**Abstract**:

We consider applications involving a large set of instances of projecting points to polytopes. We develop an intuition guided by theoretical and empirical analysis to show that when these instances follow certain structures, a large majority of the projections lie on vertices of the polytopes. To do these projections efficiently we derive a vertex-oriented incremental algorithm to project a point onto any arbitrary polytope, as well as give specific algorithms to cater to simplex projection and polytopes where the unit box is cut by planes. Such settings are especially useful in web-scale applications such as optimal matching or allocation problems. Several such problems in internet marketplaces (e-commerce, ride-sharing, food delivery, professional services, advertising, etc.), can be formulated as Linear Programs (LP) with such polytope constraints that require a projection step in the overall optimization process. We show that in some of the very recent works, the polytopic projection is the most expensive step and our efficient projection algorithms help in gaining massive improvements in performance.

----

## [425] A Variant of Concurrent Constraint Programming on GPU

**Authors**: *Pierre Talbot, Frédéric G. Pinel, Pascal Bouvry*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20298](https://doi.org/10.1609/aaai.v36i4.20298)

**Abstract**:

The number of cores on graphical computing units (GPUs) is reaching thousands nowadays, whereas the clock speed of processors stagnates. Unfortunately, constraint programming solvers do not take advantage yet of GPU parallelism. One reason is that constraint solvers were primarily designed within the mental frame of sequential computation. To solve this issue, we take a step back and contribute to a simple, intrinsically parallel, lock-free and formally correct programming language based on concurrent constraint programming. We then re-examine parallel constraint solving on GPUs within this formalism, and develop Turbo, a simple constraint solver entirely programmed on GPUs. Turbo validates the correctness of our approach and compares positively to a parallel CPU-based solver.

----

## [426] Real-Time Driver-Request Assignment in Ridesourcing

**Authors**: *Hao Wang, Xiaohui Bei*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20299](https://doi.org/10.1609/aaai.v36i4.20299)

**Abstract**:

Online on-demand ridesourcing service has played a huge role in transforming urban transportation. A central function in most on-demand ridesourcing platforms is to dynamically assign drivers to rider requests that could balance the request waiting times and the driver pick-up distances. To deal with the online nature of this problem, existing literature either divides the time horizon into short windows and applies a static offline assignment algorithm within each window or assumes a fully online setting that makes decisions for each request immediately upon its arrival. In this paper, we propose a more realistic model for the driver-request assignment that bridges the above two settings together. Our model allows the requests to wait after their arrival but assumes that they may leave at any time following a quitting function. Under this model, we design an efficient algorithm for assigning available drivers to requests in real-time. Our algorithm is able to incorporate future estimated driver arrivals into consideration and make strategic waiting and matching decisions that could balance the waiting time and pick-up distance of the assignment. We prove that our algorithm is optimal ex-ante in the single-request setting, and demonstrate its effectiveness in the general multi-request setting through experiments on both synthetic and real-world datasets.

----

## [427] Encoding Multi-Valued Decision Diagram Constraints as Binary Constraint Trees

**Authors**: *Ruiwei Wang, Roland H. C. Yap*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20300](https://doi.org/10.1609/aaai.v36i4.20300)

**Abstract**:

Ordered Multi-valued Decision Diagram (MDD) is a compact representation used to model various constraints, such as regular constraints and table constraints. It can be particularly useful for representing ad-hoc problem specific constraints. Many algorithms have been proposed to enforce Generalized Arc Consistency (GAC) on MDD constraints. In this paper, we introduce a new compact representation called Binary Constraint Tree (BCT). We propose tree binary encodings to transform any MDD constraint into a BCT constraint. We also present a specialized algorithm enforcing GAC on the BCT constraint resulting from a MDD constraint. Experimental results on a large set of benchmarks show that the BCT GAC algorithm can significantly outperform state-of-the-art MDD as well as table GAC algorithms.

----

## [428] Sample Average Approximation for Stochastic Optimization with Dependent Data: Performance Guarantees and Tractability

**Authors**: *Yafei Wang, Bo Pan, Wei Tu, Peng Liu, Bei Jiang, Chao Gao, Wei Lu, Shangling Jui, Linglong Kong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20301](https://doi.org/10.1609/aaai.v36i4.20301)

**Abstract**:

Sample average approximation (SAA), a popular method for tractably solving stochastic optimization problems, enjoys strong asymptotic performance guarantees in settings with independent training samples. However, these guarantees are not known to hold generally with dependent samples, such as in online learning with time series data or distributed computing with Markovian training samples. In this paper, we show that SAA remains tractable when the distribution of unknown parameters is only observable through dependent instances and still enjoys asymptotic consistency and finite sample guarantees. Specifically, we provide a rigorous probability error analysis to derive 1 - beta confidence bounds for the out-of-sample performance of SAA estimators and show that these estimators are asymptotically consistent. We then, using monotone operator theory, study the performance of a class of stochastic first-order algorithms trained on a dependent source of data. We show that approximation error for these algorithms is bounded and concentrates around zero, and establish deviation bounds for iterates when the underlying stochastic process is phi-mixing. The algorithms presented can be used to handle numerically inconvenient loss functions such as the sum of a smooth and non-smooth function or of non-smooth functions with constraints. To illustrate the usefulness of our results, we present several stochastic versions of popular algorithms such as stochastic proximal gradient descent (S-PGD), stochastic relaxed Peaceman-Rachford splitting algorithms (S-rPRS), and numerical experiment.

----

## [429] A Provably-Efficient Model-Free Algorithm for Infinite-Horizon Average-Reward Constrained Markov Decision Processes

**Authors**: *Honghao Wei, Xin Liu, Lei Ying*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20302](https://doi.org/10.1609/aaai.v36i4.20302)

**Abstract**:

This paper presents a model-free reinforcement learning (RL) algorithm for infinite-horizon average-reward Constrained Markov Decision Processes (CMDPs). Considering a learning horizon K, which is sufficiently large, the proposed algorithm achieves sublinear regret and zero constraint violation. The bounds depend on the number of states S, the number of actions A, and two constants which are independent of the learning horizon K.

----

## [430] TextHoaxer: Budgeted Hard-Label Adversarial Attacks on Text

**Authors**: *Muchao Ye, Chenglin Miao, Ting Wang, Fenglong Ma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20303](https://doi.org/10.1609/aaai.v36i4.20303)

**Abstract**:

This paper focuses on a newly challenging setting in hard-label adversarial attacks on text data by taking the budget information into account. Although existing approaches can successfully generate adversarial examples in the hard-label setting, they follow an ideal assumption that the victim model does not restrict the number of queries. However, in real-world applications the query budget is usually tight or limited. Moreover, existing hard-label adversarial attack techniques use the genetic algorithm to optimize discrete text data by maintaining a number of adversarial candidates during optimization, which can lead to the problem of generating low-quality adversarial examples in the tight-budget setting. To solve this problem, in this paper, we propose a new method named TextHoaxer by formulating the budgeted hard-label adversarial attack task on text data as a gradient-based optimization problem of perturbation matrix in the continuous word embedding space. Compared with the genetic algorithm-based optimization, our solution only uses a single initialized adversarial example as the adversarial candidate for optimization, which significantly reduces the number of queries. The optimization is guided by a new objective function consisting of three terms, i.e., semantic similarity term, pair-wise perturbation constraint, and sparsity constraint. Semantic similarity term and pair-wise perturbation constraint can ensure the high semantic similarity of adversarial examples from both comprehensive text-level and individual word-level, while the sparsity constraint explicitly restricts the number of perturbed words, which is also helpful for enhancing the quality of generated text. We conduct extensive experiments on eight text datasets against three representative natural language models, and experimental results show that TextHoaxer can generate high-quality adversarial examples with higher semantic similarity and lower perturbation rate under the tight-budget setting.

----

## [431] Two Compacted Models for Efficient Model-Based Diagnosis

**Authors**: *Huisi Zhou, Dantong Ouyang, Xiangfu Zhao, Liming Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20304](https://doi.org/10.1609/aaai.v36i4.20304)

**Abstract**:

Model-based diagnosis (MBD) with multiple observations is complicated and difficult to manage over. In this paper, we proposed two new diagnosis models, namely, the Compacted Model with Multiple Observations (CMMO) and the
 Dominated-based Compacted Model with Multiple Observations (D-CMMO), to solve the problem in which a considerable amount of time is needed when multiple observations are given and more than one fault is injected. Three ideas are presented in this paper. First, we propose to encode MBD with each observation as a subsystem and share as many system variables as possible to compress the size of encoded clauses. Second, we utilize the notion of gate dominance in the CMMO approach to compute Top-Level Diagnosis with Compacted Model (CM-TLD) to reduce the solution space. Finally, we explore the performance of our model using three fault models. Experimental results on the ISCAS-85 benchmarks show that CMMO and D-CMMO perform better than
 the state-of-the-art algorithms.

----

## [432] Parameterized Approximation Algorithms for K-center Clustering and Variants

**Authors**: *Sayan Bandyapadhyay, Zachary Friggstad, Ramin Mousavi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20305](https://doi.org/10.1609/aaai.v36i4.20305)

**Abstract**:

k-center is one of the most popular clustering models. While it admits a simple 2-approximation in polynomial time in general metrics, the Euclidean version is NP-hard to approximate within a factor of 1.93, even in the plane, if one insists the dependence on k in the running time be polynomial. Without this restriction, a classic algorithm yields a 2^{O((klog k)/{epsilon})}dn-time (1+epsilon)-approximation for Euclidean k-center, where d is the dimension.
 
 In this work, we give a faster algorithm for small dimensions: roughly speaking an O^*(2^{O((1/epsilon)^{O(d)} k^{1-1/d} log k)})-time (1+epsilon)-approximation. In particular, the running time is roughly O^*(2^{O((1/epsilon)^{O(1)}sqrt{k}log k)}) in the plane. We complement our algorithmic result with a matching hardness lower bound. 
 
 We also consider a well-studied generalization of k-center, called Non-uniform k-center (NUkC), where we allow different radii clusters. NUkC is NP-hard to approximate within any factor, even in the Euclidean case. We design a 2^{O(klog k)}n^2 time 3-approximation for NUkC, and a 2^{O((klog k)/epsilon)}dn time (1+\epsilon)-approximation for Euclidean NUkC. The latter time bound matches the bound for k-center.

----

## [433] How to Find a Good Explanation for Clustering?

**Authors**: *Sayan Bandyapadhyay, Fedor V. Fomin, Petr A. Golovach, William Lochet, Nidhi Purohit, Kirill Simonov*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20306](https://doi.org/10.1609/aaai.v36i4.20306)

**Abstract**:

k-means and k-median clustering are powerful unsupervised machine learning techniques. However, due to complicated dependences on all the features, it is challenging to interpret the resulting cluster assignments. Moshkovitz, Dasgupta, Rashtchian, and  Frost  proposed an elegant model of explainable k-means and k-median clustering  in ICML 2020. In this model, a  decision tree with k leaves provides a straightforward characterization of the data set into clusters. 
 
 
 We study two natural algorithmic questions about explainable clustering.  (1) For a given clustering, how to find the ``best explanation'' by using a decision tree with k leaves?  (2) For a given set of points, how to find a decision tree with k leaves minimizing the k-means/median objective of the resulting explainable clustering?
To address the first question, we introduce a new model of explainable clustering. Our model, inspired by the notion of outliers in robust statistics, is the following. We are seeking a small number of points (outliers) whose removal makes the existing clustering well-explainable. For addressing the second question, we initiate the study of the model of Moshkovitz et al. from the perspective of multivariate complexity. Our rigorous algorithmic analysis sheds some light on the influence of parameters like the input size, dimension of the data, the number of outliers, the number of clusters, and the approximation ratio,  on the computational complexity of explainable clustering.

----

## [434] Regularizing Graph Neural Networks via Consistency-Diversity Graph Augmentations

**Authors**: *Deyu Bo, Binbin Hu, Xiao Wang, Zhiqiang Zhang, Chuan Shi, Jun Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20307](https://doi.org/10.1609/aaai.v36i4.20307)

**Abstract**:

Despite the remarkable performance of graph neural networks (GNNs) in semi-supervised learning, it is criticized for not making full use of unlabeled data and suffering from over-fitting. Recently, graph data augmentation, used to improve both accuracy and generalization of GNNs, has received considerable attentions. However, one fundamental question is how to evaluate the quality of graph augmentations in principle? In this paper, we propose two metrics, Consistency and Diversity, from the aspects of augmentation correctness and generalization. Moreover, we discover that existing augmentations fall into a dilemma between these two metrics. Can we find a graph augmentation satisfying both consistency and diversity? A well-informed answer can help us understand the mechanism behind graph augmentation and improve the performance of GNNs. To tackle this challenge, we analyze two representative semi-supervised learning algorithms: label propagation (LP) and consistency regularization (CR). We find that LP utilizes the prior knowledge of graphs to improve consistency and CR adopts variable augmentations to promote diversity. Based on this discovery, we treat neighbors as augmentations to capture the prior knowledge embodying homophily assumption, which promises a high consistency of augmentations. To further promote diversity, we randomly replace the immediate neighbors of each node with its remote neighbors. After that, a neighbor-constrained regularization is proposed to enforce the predictions of the augmented neighbors to be consistent with each other. Extensive experiments on five real-world graphs validate the superiority of our method in improving the accuracy and generalization of GNNs.

----

## [435] Two-Stage Octave Residual Network for End-to-End Image Compression

**Authors**: *Fangdong Chen, Yumeng Xu, Li Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20308](https://doi.org/10.1609/aaai.v36i4.20308)

**Abstract**:

Octave Convolution (OctConv) is a generic convolutional unit that has already achieved good performances in many computer vision tasks. Recent studies also have shown the potential of applying the OctConv in end-to-end image compression. However, considering the characteristic of image compression task, current works of OctConv may limit the performance of the image compression network due to the loss of spatial information caused by the sampling operations of inter-frequency communication. Besides, the correlation between multi-frequency latents produced by OctConv is not utilized in current architectures. In this paper, to address these problems, we propose a novel Two-stage Octave Residual (ToRes) block which strips the sampling operation from OctConv to strengthen the capability of preserving useful information. Moreover, to capture the redundancy between the multi-frequency latents, a context transfer module is designed. The results show that both ToRes block and the incorporation of context transfer module help to improve the Rate-Distortion performance, and the combination of these two strategies makes our model achieve the state-of-the-art performance and outperform the latest compression standard Versatile Video Coding (VVC) in terms of both PSNR and MS-SSIM.

----

## [436] DANets: Deep Abstract Networks for Tabular Data Classification and Regression

**Authors**: *Jintai Chen, Kuanlun Liao, Yao Wan, Danny Z. Chen, Jian Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20309](https://doi.org/10.1609/aaai.v36i4.20309)

**Abstract**:

Tabular data are ubiquitous in real world applications. Although many commonly-used neural components (e.g., convolution) and extensible neural networks (e.g., ResNet) have been developed by the machine learning community, few of them were effective for tabular data and few designs were adequately tailored for tabular data structures. In this paper, we propose a novel and flexible neural component for tabular data, called Abstract Layer (AbstLay), which learns to explicitly group correlative input features and generate higher-level features for semantics abstraction. Also, we design a structure re-parameterization method to compress the trained AbstLay, thus reducing the computational complexity by a clear margin in the reference phase. A special basic block is built using AbstLays, and we construct a family of Deep Abstract Networks (DANets) for tabular data classification and regression by stacking such blocks. In DANets, a special shortcut path is introduced to fetch information from raw tabular features, assisting feature interactions across different levels. Comprehensive experiments on seven real-world tabular datasets show that our AbstLay and DANets are effective for tabular data classification and regression, and the computational complexity is superior to competitive methods. Besides, we evaluate the performance gains of DANet as it goes deep, verifying the extendibility of our method. Our code is available at https://github.com/WhatAShot/DANet.

----

## [437] Fuzzy Logic Based Logical Query Answering on Knowledge Graphs

**Authors**: *Xuelu Chen, Ziniu Hu, Yizhou Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20310](https://doi.org/10.1609/aaai.v36i4.20310)

**Abstract**:

Answering complex First-Order Logical (FOL) queries on large-scale incomplete knowledge graphs (KGs) is an important yet challenging task. Recent advances embed logical queries and KG entities in the same space and conduct query answering via dense similarity search. However, most logical operators designed in previous studies do not satisfy the axiomatic system of classical logic, limiting their performance. Moreover, these logical operators are parameterized and thus require many complex FOL queries as training data, which are often arduous to collect or even inaccessible in most real-world KGs. We thus present FuzzQE, a fuzzy logic based logical query embedding framework for answering FOL queries over KGs. FuzzQE follows fuzzy logic to define logical operators in a principled and learning-free manner, where only entity and relation embeddings require learning. FuzzQE can further benefit from labeled complex logical queries for training. Extensive experiments on two benchmark datasets demonstrate that FuzzQE provides significantly better performance in answering FOL queries compared to state-of-the-art methods. In addition, FuzzQE trained with only KG link prediction can achieve comparable performance to those trained with extra complex query data.

----

## [438] TAG: Learning Timed Automata from Logs

**Authors**: *Lénaïg Cornanguer, Christine Largouët, Laurence Rozé, Alexandre Termier*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20311](https://doi.org/10.1609/aaai.v36i4.20311)

**Abstract**:

Event logs are often one of the main sources of information to understand the behavior of a system. While numerous approaches have extracted partial information from event logs, in this work, we aim at inferring a global model of a system from its event logs.
 We consider real-time systems, which can be modeled with Timed Automata: our approach is thus a Timed Automata learner. There is a handful of related work, however, they might require a lot of parameters or produce Timed Automata that either are undeterministic or lack precision. In contrast, our proposed approach, called TAG, requires only one parameter and learns a deterministic Timed Automaton having a good tradeoff between accuracy and complexity of the automata. This allows getting an interpretable and accurate global model of the real-time system considered. Our experiments compare our approach to the related work and demonstrate its merits.

----

## [439] Differentially Describing Groups of Graphs

**Authors**: *Corinna Coupette, Sebastian Dalleiger, Jilles Vreeken*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20312](https://doi.org/10.1609/aaai.v36i4.20312)

**Abstract**:

How does neural connectivity in autistic children differ from neural connectivity in healthy children or autistic youths? What patterns in global trade networks are shared across classes of goods, and how do these patterns change over time? Answering questions like these requires us to differentially describe groups of graphs: Given a set of graphs and a partition of these graphs into groups, discover what graphs in one group have in common, how they systematically differ from graphs in other groups, and how multiple groups of graphs are related. We refer to this task as graph group analysis, which seeks to describe similarities and differences between graph groups by means of statistically significant subgraphs. To perform graph group analysis, we introduce Gragra, which uses maximum entropy modeling to identify a non-redundant set of subgraphs with statistically significant associations to one or more graph groups. Through an extensive set of experiments on a wide range of synthetic and real-world graph groups, we confirm that Gragra works well in practice.

----

## [440] Molecular Contrastive Learning with Chemical Element Knowledge Graph

**Authors**: *Yin Fang, Qiang Zhang, Haihong Yang, Xiang Zhuang, Shumin Deng, Wen Zhang, Ming Qin, Zhuo Chen, Xiaohui Fan, Huajun Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20313](https://doi.org/10.1609/aaai.v36i4.20313)

**Abstract**:

Molecular representation learning contributes to multiple downstream tasks such as molecular property prediction and drug design. To properly represent molecules, graph contrastive learning is a promising paradigm as it utilizes self-supervision signals and has no requirements for human annotations. However, prior works fail to incorporate fundamental domain knowledge into graph semantics and thus ignore the correlations between atoms that have common attributes but are not directly connected by bonds. To address these issues, we construct a Chemical Element Knowledge Graph (KG) to summarize microscopic associations between elements and propose a novel Knowledge-enhanced Contrastive Learning (KCL) framework for molecular representation learning. KCL framework consists of three modules. The first module, knowledge-guided graph augmentation, augments the original molecular graph based on the Chemical Element KG. The second module, knowledge-aware graph representation, extracts molecular representations with a common graph encoder for the original molecular graph and a  Knowledge-aware Message Passing Neural Network (KMPNN) to encode complex information in the augmented molecular graph. The final module is a contrastive objective, where we maximize agreement between these two views of molecular graphs. Extensive experiments demonstrated that KCL obtained superior performances against state-of-the-art baselines on eight molecular datasets. Visualization experiments properly interpret what KCL has learned from atoms and attributes in the augmented molecular graphs.

----

## [441] Heterogeneity-Aware Twitter Bot Detection with Relational Graph Transformers

**Authors**: *Shangbin Feng, Zhaoxuan Tan, Rui Li, Minnan Luo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20314](https://doi.org/10.1609/aaai.v36i4.20314)

**Abstract**:

Twitter bot detection has become an important and challenging task to combat misinformation and protect the integrity of the online discourse. State-of-the-art approaches generally leverage the topological structure of the Twittersphere, while they neglect the heterogeneity of relations and influence among users. In this paper, we propose a novel bot detection framework to alleviate this problem, which leverages the topological structure of user-formed heterogeneous graphs and models varying influence intensity between users. Specifically, we construct a heterogeneous information network with users as nodes and diversified relations as edges. We then propose relational graph transformers to model heterogeneous influence between users and learn node representations. Finally, we use semantic attention networks to aggregate messages across users and relations and conduct heterogeneity-aware Twitter bot detection. Extensive experiments demonstrate that our proposal outperforms state-of-the-art methods on a comprehensive Twitter bot detection benchmark. Additional studies also bear out the effectiveness of our proposed relational graph transformers, semantic attention networks and the graph-based approach in general.

----

## [442] Subspace Differential Privacy

**Authors**: *Jie Gao, Ruobin Gong, Fang-Yi Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20315](https://doi.org/10.1609/aaai.v36i4.20315)

**Abstract**:

Many data applications have certain invariant constraints due to practical needs. Data curators who employ differential privacy need to respect such constraints on the sanitized data product as a primary utility requirement. Invariants challenge the formulation, implementation, and interpretation of privacy guarantees. 

We propose subspace differential privacy, to honestly characterize the dependence of the sanitized output on confidential aspects of the data. We discuss two design frameworks that convert well-known differentially private mechanisms, such as the Gaussian and the Laplace mechanisms, to subspace differentially private ones that respect the invariants specified by the curator. For linear queries, we discuss the design of near-optimal mechanisms that minimize the mean squared error. Subspace differentially private mechanisms rid the need for post-processing due to invariants, preserve transparency and statistical intelligibility of the output, and can be suitable for distributed implementation. We showcase the proposed mechanisms on the 2020 Census Disclosure Avoidance demonstration data, and a spatio-temporal dataset of mobile access point connections on a large university campus.

----

## [443] Orthogonal Graph Neural Networks

**Authors**: *Kai Guo, Kaixiong Zhou, Xia Hu, Yu Li, Yi Chang, Xin Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20316](https://doi.org/10.1609/aaai.v36i4.20316)

**Abstract**:

Graph neural networks (GNNs) have received tremendous attention due to their superiority in learning node representations. These models rely on message passing and feature transformation functions to encode the structural and feature information from neighbors. However, stacking more convolutional layers significantly decreases the performance of GNNs. Most recent studies attribute this limitation to the over-smoothing issue, where node embeddings converge to indistinguishable vectors. Through a number of experimental observations, we argue that the main factor degrading the performance is the unstable forward normalization and backward gradient resulted from the improper design of the feature transformation, especially for shallow GNNs where the over-smoothing has not happened. Therefore, we propose a novel orthogonal feature transformation, named Ortho-GConv, which could generally augment the existing GNN backbones to stabilize the model training and improve the model's generalization performance. Specifically, we maintain the orthogonality of the feature transformation comprehensively from three perspectives, namely hybrid weight initialization, orthogonal transformation, and orthogonal regularization. By equipping the existing GNNs (e.g. GCN, JKNet, GCNII) with Ortho-GConv, we demonstrate the generality of the orthogonal feature transformation to enable stable training, and show its effectiveness for node and graph classification tasks.

----

## [444] Learning Temporal Point Processes for Efficient Retrieval of Continuous Time Event Sequences

**Authors**: *Vinayak Gupta, Srikanta Bedathur, Abir De*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20317](https://doi.org/10.1609/aaai.v36i4.20317)

**Abstract**:

Recent developments in predictive modeling using marked temporal point processes (MTPPs) have enabled an accurate characterization of several real-world applications involving continuous-time event sequences (CTESs). However, the retrieval problem of such sequences remains largely unaddressed in literature. To tackle this, we propose NEUROSEQRET which learns to retrieve and rank a relevant set of continuous-time event sequences for a given query sequence, from a large corpus of sequences. More specifically, NEUROSEQRET first applies a trainable unwarping function on the query sequence, which makes it comparable with corpus sequences, especially when a relevant query-corpus pair has individually different attributes. Next, it feeds the unwarped query sequence and the corpus sequence into MTPP guided neural relevance models. We develop two variants of the relevance model which offer a tradeoff between accuracy and efficiency. We also propose an optimization framework to learn binary sequence embeddings from the relevance scores, suitable for the locality-sensitive hashing leading to a significant speedup in returning top-K results for a given query sequence. Our experiments with several datasets show the significant accuracy boost of NEUROSEQRET beyond several baselines, as well as the efficacy of our hashing mechanism.

----

## [445] GNN-Retro: Retrosynthetic Planning with Graph Neural Networks

**Authors**: *Peng Han, Peilin Zhao, Chan Lu, Junzhou Huang, Jiaxiang Wu, Shuo Shang, Bin Yao, Xiangliang Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20318](https://doi.org/10.1609/aaai.v36i4.20318)

**Abstract**:

Retrosynthetic planning plays an important role in the field of organic chemistry, which could generate a synthetic route for the target product. The synthetic route is a series of reactions which are started from the available molecules. The most challenging problem in the generation of the synthetic route is the large search space of the candidate reactions. Estimating the cost of candidate reactions has been proved effectively to prune the search space, which could achieve a higher accuracy with the same search iteration. And the estimation of one reaction is comprised of the estimations of all its reactants. So, how to estimate the cost of these reactants will directly influence the quality of results. To get a better performance, we propose a new framework, named GNN-Retro, for retrosynthetic planning problem by combining graph neural networks(GNN) and the latest search algorithm. The structure of GNN in our framework could incorporate the information of neighboring molecules, which will improve the estimation accuracy of our framework. The experiments on the USPTO dataset show that our framework could outperform the state-of-the-art methods with a large margin under the same settings.

----

## [446] Block Modeling-Guided Graph Convolutional Neural Networks

**Authors**: *Dongxiao He, Chundong Liang, Huixin Liu, Mingxiang Wen, Pengfei Jiao, Zhiyong Feng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20319](https://doi.org/10.1609/aaai.v36i4.20319)

**Abstract**:

Graph Convolutional Network (GCN) has shown remarkable potential of exploring graph representation. However, the GCN aggregating mechanism fails to generalize to networks with heterophily where most nodes have neighbors from different classes, which commonly exists in real-world networks. In order to make the propagation and aggregation mechanism of GCN suitable for both homophily and heterophily (or even their mixture), we introduce block modelling into the framework of GCN so that it can realize “block-guided classified aggregation”, and automatically learn the corresponding aggregation rules for neighbors of different classes. By incorporating block modelling into the aggregation process, GCN is able to automatically aggregate information from homophilic and heterophilic neighbors discriminately according to their homophily degree. We compared our algorithm with state-of-art methods which deal with the heterophily problem. Empirical results demonstrate the superiority of our new approach over existing methods in heterophilic datasets while maintaining a competitive performance in homophilic datasets.

----

## [447] CATN: Cross Attentive Tree-Aware Network for Multivariate Time Series Forecasting

**Authors**: *Hui He, Qi Zhang, Simeng Bai, Kun Yi, Zhendong Niu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20320](https://doi.org/10.1609/aaai.v36i4.20320)

**Abstract**:

Modeling complex hierarchical and grouped feature interaction in the multivariate time series data is indispensable to comprehend the data dynamics and predicting the future condition. The implicit feature interaction and high-dimensional data make multivariate forecasting very challenging. Many existing works did not put more emphasis on exploring explicit correlation among multiple time series data, and complicated models are designed to capture long- and short-range pattern with the aid of attention mechanism. In this work, we think that pre-defined graph or general learning method is difficult due to their irregular structure. Hence, we present CATN, an end-to-end model of Cross Attentive Tree-aware Network to jointly capture the inter-series correlation and intra-series temporal pattern. We first construct a tree structure to learn hierarchical and grouped correlation and design an embedding approach that can pass dynamic message to generalize implicit but interpretable cross features among multiple time series. Next in temporal aspect, we propose a multi-level dependency learning mechanism including global&local learning and cross attention mechanism, which can combine long-range dependencies, short-range dependencies as well as cross dependencies at different time steps. The extensive experiments on different datasets from real world show the effectiveness and robustness of the method we proposed when compared with existing state-of-the-art methods.

----

## [448] FPAdaMetric: False-Positive-Aware Adaptive Metric Learning for Session-Based Recommendation

**Authors**: *Jongwon Jeong, Jeong Choi, Hyunsouk Cho, Sehee Chung*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20321](https://doi.org/10.1609/aaai.v36i4.20321)

**Abstract**:

Modern recommendation systems are mostly based on implicit feedback data which can be quite noisy due to false positives (FPs) caused by many reasons, such as misclicks or quick curiosity. Numerous recommendation algorithms based on collaborative filtering have leveraged post-click user behavior (e.g., skip) to identify false positives. They effectively involved these false positives in the model supervision as negative-like signals. Yet, false positives had not been considered in existing session-based recommendation systems (SBRs) although they provide just as deleterious effects. 
 To resolve false positives in SBRs, we first introduce FP-Metric model which reformulates the objective of the session-based recommendation with FP constraints into metric learning regularization. In addition, we propose FP-AdaMetric that enhances the metric-learning regularization terms with an adaptive module that elaborately calculates the impact of FPs inside sequential patterns. We verify that FP-AdaMetric improves several session-based recommendation models' performances in terms of Hit Rate (HR), MRR, and NDCG on datasets from different domains including music, movie, and game. Furthermore, we show that the adaptive module plays a much more crucial role in FP-AdaMetric model than in other baselines.

----

## [449] STDEN: Towards Physics-Guided Neural Networks for Traffic Flow Prediction

**Authors**: *Jiahao Ji, Jingyuan Wang, Zhe Jiang, Jiawei Jiang, Hu Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20322](https://doi.org/10.1609/aaai.v36i4.20322)

**Abstract**:

High-performance traffic flow prediction model designing, a core technology of Intelligent Transportation System, is a long-standing but still challenging task for industrial and academic communities. The lack of integration between physical principles and data-driven models is an important reason for limiting the development of this field. In the literature, physics-based methods can usually provide a clear interpretation of the dynamic process of traffic flow systems but are with limited accuracy, while data-driven methods, especially deep learning with black-box structures, can achieve improved performance but can not be fully trusted due to lack of a reasonable physical basis. To bridge the gap between purely data-driven and physics-driven approaches, we propose a physics-guided deep learning model named Spatio-Temporal Differential Equation Network (STDEN), which casts the physical mechanism of traffic flow dynamics into a deep neural network framework. Specifically, we assume the traffic flow on road networks is driven by a latent potential energy field (like water flows are driven by the gravity field), and model the spatio-temporal dynamic process of the potential energy field as a differential equation network. STDEN absorbs both the performance advantage of data-driven models and the interpretability of physics-based models, so is named a physics-guided prediction model. Experiments on three real-world traffic datasets in Beijing show that our model outperforms state-of-the-art baselines by a significant margin. A case study further verifies that STDEN can capture the mechanism of urban traffic and generate accurate predictions with physical meaning. The proposed framework of differential equation network modeling may also cast light on other similar applications.

----

## [450] Naming the Most Anomalous Cluster in Hilbert Space for Structures with Attribute Information

**Authors**: *Janis Kalofolias, Jilles Vreeken*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20323](https://doi.org/10.1609/aaai.v36i4.20323)

**Abstract**:

We consider datasets consisting of arbitrarily structured entities (e.g., molecules,
 sequences, graphs, etc) whose similarity can be assessed with a reproducing ker-
 nel (or a family thereof). These entities are assumed to additionally have a
 set of named attributes (e.g.: number_of_atoms, stock_price, etc). These
 attributes can be used to classify the structured entities in discrete sets (e.g.,
 ‘number_of_atoms < 3’, ‘stock_price ≤ 100’, etc) and can effectively serve
 as Boolean predicates. Our goal is to use this side-information to provide explain-
 able kernel-based clustering. To this end, we propose a method which is able
 to find among all possible entity subsets that can be described as a conjunction
 of the available predicates either a) the optimal cluster within the Reproducing
 Kernel Hilbert Space, or b) the most anomalous subset within the same space.
 Our method works employs combinatorial optimisation via an adaptation of the
 Maximum-Mean-Discrepancy measure that captures the above intuition. Finally,
 we propose a criterion to select the optimal one out of a family of kernels in a
 way that preserves the available side-information. We provide several real world
 datasets that demonstrate the usefulness of our proposed method.

----

## [451] Meta-Learning for Online Update of Recommender Systems

**Authors**: *Minseok Kim, Hwanjun Song, Yooju Shin, Dongmin Park, Kijung Shin, Jae-Gil Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20324](https://doi.org/10.1609/aaai.v36i4.20324)

**Abstract**:

Online recommender systems should be always aligned with users' current interest to accurately suggest items that each user would like. Since user interest usually evolves over time, the update strategy should be flexible to quickly catch users' current interest from continuously generated new user-item interactions. Existing update strategies focus either on the importance of each user-item interaction or the learning rate for each recommender parameter, but such one-directional flexibility is insufficient to adapt to varying relationships between interactions and parameters. In this paper, we propose MeLON, a meta-learning based novel online recommender update strategy that supports two-directional flexibility. It is featured with an adaptive learning rate for each parameter-interaction pair for inducing a recommender to quickly learn users' up-to-date interest. The procedure of MeLON is optimized following a meta-learning approach: it learns how a recommender learns to generate the optimal learning rates for future updates. Specifically, MeLON first enriches the meaning of each interaction based on previous interactions and identifies the role of each parameter for the interaction; and then combines these two pieces of information to generate an adaptive learning rate. Theoretical analysis and extensive evaluation on three real-world online recommender datasets validate the effectiveness of MeLON.

----

## [452] The Triangle-Densest-K-Subgraph Problem: Hardness, Lovász Extension, and Application to Document Summarization

**Authors**: *Aritra Konar, Nicholas D. Sidiropoulos*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20325](https://doi.org/10.1609/aaai.v36i4.20325)

**Abstract**:

We introduce the triangle-densest-K-subgraph problem (TDKS) for undirected graphs: given a size parameter K, compute a subset of K vertices that maximizes the number of induced triangles. The problem corresponds to the simplest generalization of the edge based densest-K-subgraph problem (DKS) to the case of higher-order network motifs. We prove that TDKS is NP-hard and is not amenable to efficient approximation, in the worst-case. By judiciously exploiting the structure of the problem, we propose a relaxation algorithm for the purpose of obtaining high-quality, sub-optimal solutions. Our approach utilizes the fact that the cost function of TDKS is submodular to construct a convex relaxation for the problem based on the Lovász extension for submodular functions. We demonstrate that our approaches attain state-of-the-art performance on real-world graphs and can offer substantially improved exploration of the optimal density-size curve compared to sophisticated approximation baselines for DKS. We use document summarization to showcase why TDKS is a useful generalization of DKS.

----

## [453] Obtaining Calibrated Probabilities with Personalized Ranking Models

**Authors**: *Wonbin Kweon, SeongKu Kang, Hwanjo Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20326](https://doi.org/10.1609/aaai.v36i4.20326)

**Abstract**:

For personalized ranking models, the well-calibrated probability of an item being preferred by a user has great practical value.
 While existing work shows promising results in image classification, probability calibration has not been much explored for personalized ranking.
 In this paper, we aim to estimate the calibrated probability of how likely a user will prefer an item.
 We investigate various parametric distributions and propose two parametric calibration methods, namely Gaussian calibration and Gamma calibration.
 Each proposed method can be seen as a post-processing function that maps the ranking scores of pre-trained models to well-calibrated preference probabilities, without affecting the recommendation performance.
 We also design the unbiased empirical risk minimization framework that guides the calibration methods to learning of true preference probability from the biased user-item interaction dataset.
 Extensive evaluations with various personalized ranking models on real-world datasets show that both the proposed calibration methods and the unbiased empirical risk minimization significantly improve the calibration performance.

----

## [454] DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation

**Authors**: *Wendi Li, Xiao Yang, Weiqing Liu, Yingce Xia, Jiang Bian*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20327](https://doi.org/10.1609/aaai.v36i4.20327)

**Abstract**:

In many real-world scenarios, we often deal with streaming data that is sequentially collected over time. Due to the non-stationary nature of the environment, the streaming data distribution may change in unpredictable ways, which is known as the concept drift in the literature. To handle concept drift, previous methods first detect when/where the concept drift happens and then adapt models to fit the distribution of the latest data. However, there are still many cases that some underlying factors of environment evolution are predictable, making it possible to model the future concept drift trend of the streaming data, while such cases are not fully explored in previous work. In this paper, we propose a novel method DDG-DA, that can effectively forecast the evolution of data distribution and improve the performance of models. Specifically, we first train a predictor to estimate the future data distribution, then leverage it to generate training samples, and finally train models on the generated data. We conduct experiments on three real-world tasks (forecasting on stock price trend, electricity load and solar irradiance) and obtained significant improvement on multiple widely-used models.

----

## [455] Unsupervised Anomaly Detection by Robust Density Estimation

**Authors**: *Boyang Liu, Pang-Ning Tan, Jiayu Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20328](https://doi.org/10.1609/aaai.v36i4.20328)

**Abstract**:

Density estimation is a widely used method to perform unsupervised anomaly detection. By learning the density function, data points with relatively low densities are classified as anomalies. Unfortunately, the presence of anomalies in training data may significantly impact the density estimation process, thereby imposing significant challenges to the use of more sophisticated density estimation methods such as those based on deep neural networks. In this work, we propose RobustRealNVP, a deep density estimation framework that enhances the robustness of flow-based density estimation methods, enabling their application to unsupervised anomaly detection. 
 RobustRealNVP differs from existing flow-based models from two perspectives. First, RobustRealNVP discards data points with low estimated densities during optimization to prevent them from corrupting the density estimation process. Furthermore, it imposes Lipschitz regularization to the flow-based model to enforce smoothness in the estimated density function. We demonstrate the robustness of our algorithm against anomalies in training data from both theoretical and empirical perspectives. The results show that our algorithm achieves competitive results as compared to state-of-the-art unsupervised anomaly detection methods.

----

## [456] From One to All: Learning to Match Heterogeneous and Partially Overlapped Graphs

**Authors**: *Weijie Liu, Hui Qian, Chao Zhang, Jiahao Xie, Zebang Shen, Nenggan Zheng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20329](https://doi.org/10.1609/aaai.v36i4.20329)

**Abstract**:

Recent years have witnessed a flurry of research activity in graph matching, which aims at finding the correspondence of nodes across two graphs and lies at the heart of many artificial intelligence applications. However, matching heterogeneous graphs with partial overlap remains a challenging problem in real-world applications. This paper proposes the first practical learning-to-match method to meet this challenge. The proposed unsupervised method adopts a novel partial optimal transport paradigm to learn a transport plan and node embeddings simultaneously. In a from-one-to-all manner, the entire learning procedure is decomposed into a series of easy-to-solve sub-procedures, each of which only handles the alignment of a single type of nodes. A mechanism for searching the transport mass is also proposed. Experimental results demonstrate that the proposed method outperforms state-of-the-art graph matching methods.

----

## [457] TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal Knowledge Graphs

**Authors**: *Yushan Liu, Yunpu Ma, Marcel Hildebrandt, Mitchell Joblin, Volker Tresp*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20330](https://doi.org/10.1609/aaai.v36i4.20330)

**Abstract**:

Conventional static knowledge graphs model entities in relational data as nodes, connected by edges of specific relation types. However, information and knowledge evolve continuously, and temporal dynamics emerge, which are expected to influence future situations. In temporal knowledge graphs, time information is integrated into the graph by equipping each edge with a timestamp or a time range. Embedding-based methods have been introduced for link prediction on temporal knowledge graphs, but they mostly lack explainability and comprehensible reasoning chains. Particularly, they are usually not designed to deal with link forecasting -- event prediction involving future timestamps. We address the task of link forecasting on temporal knowledge graphs and introduce TLogic, an explainable framework that is based on temporal logical rules extracted via temporal random walks. We compare TLogic with state-of-the-art baselines on three benchmark datasets and show better overall performance while our method also provides explanations that preserve time consistency. Furthermore, in contrast to most state-of-the-art embedding-based methods, TLogic works well in the inductive setting where already learned rules are transferred to related datasets with a common vocabulary.

----

## [458] Transferring the Contamination Factor between Anomaly Detection Domains by Shape Similarity

**Authors**: *Lorenzo Perini, Vincent Vercruyssen, Jesse Davis*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20331](https://doi.org/10.1609/aaai.v36i4.20331)

**Abstract**:

Anomaly detection attempts to find examples in a dataset that do not conform to the expected behavior. Algorithms for this task assign an anomaly score to each example representing its degree of anomalousness. Setting a threshold on the anomaly scores enables converting these scores into a discrete prediction for each example. Setting an appropriate threshold is challenging in practice since anomaly detection is often treated as an unsupervised problem. A common approach is to set the threshold based on the dataset's contamination factor, i.e., the proportion of anomalous examples in the data. While the contamination factor may be known based on domain knowledge, it is often necessary to estimate it by labeling data. However, many anomaly detection problems involve monitoring multiple related, yet slightly different entities (e.g., a fleet of machines). Then, estimating the contamination factor for each dataset separately by labeling data would be extremely time-consuming. Therefore, this paper introduces a method for transferring the known contamination factor from one dataset (the source domain) to a related dataset where it is unknown (the target domain). Our approach does not require labeled target data and is based on modeling the shape of the distribution of the anomaly scores in both domains. We theoretically analyze how our method behaves when the (biased) target domain anomaly score distribution converges to its true one. Empirically, our method outperforms several baselines on real-world datasets.

----

## [459] Unifying Knowledge Base Completion with PU Learning to Mitigate the Observation Bias

**Authors**: *Jonas Schouterden, Jessa Bekker, Jesse Davis, Hendrik Blockeel*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20332](https://doi.org/10.1609/aaai.v36i4.20332)

**Abstract**:

Methods for Knowledge Base Completion (KBC) reason about a knowledge base (KB) in order to derive new facts that should be included in the KB. This is challenging for two reasons. First, KBs only contain positive examples. This complicates model evaluation which needs both positive and negative examples. Second, those facts that were selected to be included in the knowledge base, are most likely not an i.i.d. sample of the true facts, due to the way knowledge bases are constructed. In this paper, we focus on rule-based approaches, which traditionally address the first challenge by making assumptions that enable identifying negative examples, which in turn makes it possible to compute a rule's confidence or precision. However, they largely ignore the second challenge, which means that their estimates of a rule's confidence can be biased. This paper approaches rule-based KBC through the lens of PU-learning, which can cope with both challenges. We make three contributions.: (1) We provide a unifying view that formalizes the relationship between multiple existing confidences measures based on (i) what assumption they make about and (ii) how their accuracy 
 depends on the selection mechanism. (2) We introduce two new confidence measures that can mitigate known biases by using propensity scores that quantify how likely a fact is to be included the KB. (3) We show through theoretical and empirical analysis that taking the bias into account improves the confidence estimates, even when the propensity scores are not known exactly.

----

## [460] A Self-Supervised Mixed-Curvature Graph Neural Network

**Authors**: *Li Sun, Zhongbao Zhang, Junda Ye, Hao Peng, Jiawei Zhang, Sen Su, Philip S. Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20333](https://doi.org/10.1609/aaai.v36i4.20333)

**Abstract**:

Graph representation learning received increasing attentions in recent years. Most of the existing methods ignore the complexity of the graph structures and restrict graphs in a single constant-curvature representation space, which is only suitable to particular kinds of graph structure indeed. Additionally, these methods follow the supervised or semi-supervised learning paradigm, and thereby notably limit their deployment on the unlabeled graphs in real applications. To address these aforementioned limitations, we take the first attempt to study the self-supervised graph representation learning in the mixed-curvature spaces. In this paper, we present a novel Self-Supervised Mixed-Curvature Graph Neural Network (SelfMGNN). To capture the complex graph structures, we construct a mixed-curvature space via the Cartesian product of multiple Riemannian component spaces, and design hierarchical attention mechanisms for learning and fusing graph representations across these component spaces. To enable the self-supervised learning, we propose a novel dual contrastive approach. The constructed mixed-curvature space actually provides multiple Riemannian views for the contrastive learning. We introduce a Riemannian projector to reveal these views, and utilize a well-designed Riemannian discriminator for the single-view and cross-view contrastive learning within and across the Riemannian views. Finally, extensive experiments show that SelfMGNN captures the complex graph structures and outperforms state-of-the-art baselines.

----

## [461] MS-HGAT: Memory-Enhanced Sequential Hypergraph Attention Network for Information Diffusion Prediction

**Authors**: *Ling Sun, Yuan Rao, Xiangbo Zhang, Yuqian Lan, Shuanghe Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20334](https://doi.org/10.1609/aaai.v36i4.20334)

**Abstract**:

Predicting the diffusion cascades is a critical task to understand information spread on social networks. Previous methods usually focus on the order or structure of the infected users in a single cascade, thus ignoring the global dependencies of users and cascades, limiting the performance of prediction. Current strategies to introduce social networks only learn the social homogeneity among users, which is not enough to describe their interaction preferences, let alone the dynamic changes. To address the above issues, we propose a novel information diffusion prediction model named Memory-enhanced Sequential Hypergraph Attention Networks (MS-HGAT). Specifically, to introduce the global dependencies of users, we not only take advantages of their friendships, but also consider their interactions at the cascade level. Furthermore, to dynamically capture user' preferences, we divide the diffusion hypergraph into several sub graphs based on timestamps, develop Hypergraph Attention Networks to learn the sequential hypergraphs, and connect them with gated fusion strategy. In addition, a memory-enhanced embedding lookup module is proposed to capture the learned user representations into the cascade-specific embedding space, thus highlighting the feature interaction within the cascade. The experimental results over four realistic datasets demonstrate that MS-HGAT significantly outperforms the state-of-the-art diffusion prediction models in both Hits@K and MAP@k metrics.

----

## [462] Graph Structure Learning with Variational Information Bottleneck

**Authors**: *Qingyun Sun, Jianxin Li, Hao Peng, Jia Wu, Xingcheng Fu, Cheng Ji, Philip S. Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20335](https://doi.org/10.1609/aaai.v36i4.20335)

**Abstract**:

Graph Neural Networks (GNNs) have shown promising results on a broad spectrum of applications. Most empirical studies of GNNs directly take the observed graph as input, assuming the observed structure perfectly depicts the accurate and complete relations between nodes. However, graphs in the real-world are inevitably noisy or incomplete, which could even exacerbate the quality of graph representations. In this work, we propose a novel Variational Information Bottleneck guided Graph Structure Learning framework, namely VIB-GSL, in the perspective of information theory. VIB-GSL is the first attempt to advance the Information Bottleneck (IB) principle for graph structure learning, providing a more elegant and universal framework for mining underlying task-relevant relations. VIB-GSL learns an informative and compressive graph structure to distill the actionable information for specific downstream tasks. VIB-GSL deduces a variational approximation for irregular graph data to form a tractable IB objective function, which facilitates training stability. Extensive experimental results demonstrate that the superior effectiveness and robustness of the proposed VIB-GSL.

----

## [463] Heterogeneous Peer Effects in the Linear Threshold Model

**Authors**: *Christopher Tran, Elena Zheleva*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20336](https://doi.org/10.1609/aaai.v36i4.20336)

**Abstract**:

The Linear Threshold Model is a widely used model that describes how information diffuses through a social network. According to this model, an individual adopts an idea or product after the proportion of their neighbors who have adopted it reaches a certain threshold. Typical applications of the Linear Threshold Model assume that thresholds are either the same for all network nodes or randomly distributed, even though some people may be more susceptible to peer pressure than others. To address individual-level differences, we propose causal inference methods for estimating individual thresholds that can more accurately predict whether and when individuals will be affected by their peers. We introduce the concept of heterogeneous peer effects and develop a Structural Causal Model which corresponds to the Linear Threshold Model and supports heterogeneous peer effect identification and estimation. We develop two algorithms for individual threshold estimation, one based on causal trees and one based on causal meta-learners. Our experimental results on synthetic and real- world datasets show that our proposed models can better predict individual-level thresholds in the Linear Threshold Model and thus more precisely predict which nodes will get activated over time.

----

## [464] Exploring Relational Semantics for Inductive Knowledge Graph Completion

**Authors**: *Changjian Wang, Xiaofei Zhou, Shirui Pan, Linhua Dong, Zeliang Song, Ying Sha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20337](https://doi.org/10.1609/aaai.v36i4.20337)

**Abstract**:

Knowledge graph completion (KGC) aims to infer missing information in incomplete knowledge graphs (KGs). Most previous works only consider the transductive scenario where entities are existing in KGs, which cannot work effectively for the inductive scenario containing emerging entities. Recently some graph neural network-based methods have been proposed for inductive KGC by aggregating neighborhood information to capture some uncertainty semantics from the neighboring auxiliary triples. But these methods ignore the more general relational semantics underlying all the known triples that can provide richer information to represent emerging entities so as to satisfy the inductive scenario. In this paper, we propose a novel model called CFAG, which utilizes two granularity levels of relational semantics in a coarse-grained aggregator (CG-AGG) and a fine-grained generative adversarial net (FG-GAN), for inductive KGC. The CG-AGG firstly generates entity representations with multiple semantics through a hypergraph neural network-based global aggregator and a graph neural network-based local aggregator, and the FG-GAN further enhances entity representations with specific semantics through conditional generative adversarial nets. Experimental results on benchmark datasets show that our model outperforms state-of-the-art models for inductive KGC.

----

## [465] HAGEN: Homophily-Aware Graph Convolutional Recurrent Network for Crime Forecasting

**Authors**: *Chenyu Wang, Zongyu Lin, Xiaochen Yang, Jiao Sun, Mingxuan Yue, Cyrus Shahabi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20338](https://doi.org/10.1609/aaai.v36i4.20338)

**Abstract**:

The goal of the crime forecasting problem is to predict different types of crimes for each geographical region (like a neighborhood or censor tract) in the near future. Since nearby regions usually have similar socioeconomic characteristics which indicate similar crime patterns, recent state-of-the-art solutions constructed a distance-based region graph and utilized Graph Neural Network (GNN) techniques for crime forecasting, because the GNN techniques could effectively exploit the latent relationships between neighboring region nodes in the graph if the edges reveal high dependency or correlation. However, this distance-based pre-defined graph can not fully capture crime correlation between regions that are far from each other but share similar crime patterns. Hence, to make a more accurate crime prediction, the main challenge is to learn a better graph that reveals the dependencies between regions in crime occurrences and meanwhile captures the temporal patterns from historical crime records. To address these challenges, we propose an end-to-end graph convolutional recurrent network called HAGEN with several novel designs for crime prediction. Specifically, our framework could jointly capture the crime correlation between regions and the temporal crime dynamics by combining an adaptive region graph learning module with the Diffusion Convolution Gated Recurrent Unit (DCGRU). Based on the homophily assumption of GNN (i.e., graph convolution works better where neighboring nodes share the same label), we propose a homophily-aware constraint to regularize the optimization of the region graph so that neighboring region nodes on the learned graph share similar crime patterns, thus fitting the mechanism of diffusion convolution. Empirical experiments and comprehensive analysis on two real-world datasets showcase the effectiveness of HAGEN.

----

## [466] Calibrated Nonparametric Scan Statistics for Anomalous Pattern Detection in Graphs

**Authors**: *Chunpai Wang, Daniel B. Neill, Feng Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20339](https://doi.org/10.1609/aaai.v36i4.20339)

**Abstract**:

We propose a new approach, the calibrated nonparametric scan statistic (CNSS), for more accurate detection of anomalous patterns in large-scale, real-world graphs. Scan statistics identify connected subgraphs that are interesting or unexpected through maximization of a likelihood ratio statistic; in particular, nonparametric scan statistics (NPSSs) identify subgraphs with a higher than expected proportion of individually significant nodes. However, we show that recently proposed NPSS methods are miscalibrated, failing to account for the maximization of the statistic over the multiplicity of subgraphs. This results in both reduced detection power for subtle signals, and low precision of the detected subgraph even for stronger signals. Thus we develop a new statistical approach to recalibrate NPSSs, correctly adjusting for multiple hypothesis testing and taking the underlying graph structure into account. While the recalibration, based on randomization testing, is computationally expensive, we propose both an efficient (approximate) algorithm and new, closed-form lower bounds (on the expected maximum proportion of significant nodes for subgraphs of a given size, under the null hypothesis of no anomalous patterns). These advances, along with the integration of recent core-tree decomposition methods, enable CNSS to scale to large real-world graphs, with substantial improvement in the accuracy of detected subgraphs. Extensive experiments on both semi-synthetic and real-world datasets are demonstrated to validate the effectiveness of our proposed methods, in comparison with state-of-the-art counterparts.

----

## [467] Powerful Graph Convolutional Networks with Adaptive Propagation Mechanism for Homophily and Heterophily

**Authors**: *Tao Wang, Di Jin, Rui Wang, Dongxiao He, Yuxiao Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20340](https://doi.org/10.1609/aaai.v36i4.20340)

**Abstract**:

Graph Convolutional Networks (GCNs) have been widely applied in various fields due to their significant power on processing graph-structured data. Typical GCN and its variants work under a homophily assumption (i.e., nodes with same class are prone to connect to each other), while ignoring the heterophily which exists in many real-world networks (i.e., nodes with different classes tend to form edges). Existing methods deal with heterophily by mainly aggregating higher-order neighborhoods or combing the immediate representations, which leads to noise and irrelevant information in the result. But these methods did not change the propagation mechanism which works under homophily assumption (that is a fundamental part of GCNs). This makes it difficult to distinguish the representation of nodes from different classes. To address this problem, in this paper we design a novel propagation mechanism, which can automatically change the propagation and aggregation process according to homophily or heterophily between node pairs. To adaptively learn the propagation process, we introduce two measurements of homophily degree between node pairs, which is learned based on topological and attribute information, respectively. Then we incorporate the learnable homophily degree into the graph convolution framework, which is trained in an end-to-end schema, enabling it to go beyond the assumption of homophily. More importantly, we theoretically prove that our model can constrain the similarity of representations between nodes according to their homophily degree. Experiments on seven real-world datasets demonstrate that this new approach outperforms the state-of-the-art methods under heterophily or low homophily, and gains competitive performance under homophily.

----

## [468] ShuttleNet: Position-Aware Fusion of Rally Progress and Player Styles for Stroke Forecasting in Badminton

**Authors**: *Wei-Yao Wang, Hong-Han Shuai, Kai-Shiang Chang, Wen-Chih Peng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20341](https://doi.org/10.1609/aaai.v36i4.20341)

**Abstract**:

The increasing demand for analyzing the insights in sports has stimulated a line of productive studies from a variety of perspectives, e.g., health state monitoring, outcome prediction. In this paper, we focus on objectively judging what and where to return strokes, which is still unexplored in turn-based sports. By formulating stroke forecasting as a sequence prediction task, existing works can tackle the problem but fail to model information based on the characteristics of badminton. To address these limitations, we propose a novel Position-aware Fusion of Rally Progress and Player Styles framework (ShuttleNet) that incorporates rally progress and information of the players by two modified encoder-decoder extractors. Moreover, we design a fusion network to integrate rally contexts and contexts of the players by conditioning on information dependency and different positions. Extensive experiments on the badminton dataset demonstrate that ShuttleNet significantly outperforms the state-of-the-art methods and also empirically validates the feasibility of each component in ShuttleNet. On top of that, we provide an analysis scenario for the stroke forecasting problem.

----

## [469] Event-Aware Multimodal Mobility Nowcasting

**Authors**: *Zhaonan Wang, Renhe Jiang, Hao Xue, Flora D. Salim, Xuan Song, Ryosuke Shibasaki*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20342](https://doi.org/10.1609/aaai.v36i4.20342)

**Abstract**:

As a decisive part in the success of Mobility-as-a-Service (MaaS), spatio-temporal predictive modeling for crowd movements is a challenging task particularly considering scenarios where societal events drive mobility behavior deviated from the normality. While tremendous progress has been made to model high-level spatio-temporal regularities with deep learning, most, if not all of the existing methods are neither aware of the dynamic interactions among multiple transport modes nor adaptive to unprecedented volatility brought by potential societal events. In this paper, we are therefore motivated to improve the canonical spatio-temporal network (ST-Net) from two perspectives: (1) design a heterogeneous mobility information network (HMIN) to explicitly represent intermodality in multimodal mobility; (2) propose a memory-augmented dynamic filter generator (MDFG) to generate sequence-specific parameters in an on-the-fly fashion for various scenarios. The enhanced event-aware spatio-temporal network, namely EAST-Net, is evaluated on several real-world datasets with a wide variety and coverage of societal events. Both quantitative and qualitative experimental results verify the superiority of our approach compared with the state-of-the-art baselines. Code and data are published on https://github.com/underdoc-wang/EAST-Net.

----

## [470] Discovering Interpretable Data-to-Sequence Generators

**Authors**: *Boris Wiegand, Dietrich Klakow, Jilles Vreeken*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20343](https://doi.org/10.1609/aaai.v36i4.20343)

**Abstract**:

We study the problem of predicting an event sequence given some meta data. In particular, we are interested in learning easily interpretable models that can accurately generate a sequence based on an attribute vector. To this end, we propose to learn a sparse event-flow graph over the training sequences, and statistically robust rules that use meta data to determine which paths to follow. We formalize the problem in terms of the Minimum Description Length (MDL) principle, by which we identify the best model as the one that compresses the data best. As the resulting optimization problem is NP-hard, we propose the efficient ConSequence algorithm to discover good event-flow graphs from data. 
 Through an extensive set of experiments including a case study, we show that it ably discovers compact, interpretable and accurate models for the generation and prediction of event sequences from data, has a low sample complexity, and is particularly robust against noise.

----

## [471] DeepGPD: A Deep Learning Approach for Modeling Geospatio-Temporal Extreme Events

**Authors**: *Tyler Wilson, Pang-Ning Tan, Lifeng Luo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20344](https://doi.org/10.1609/aaai.v36i4.20344)

**Abstract**:

Geospatio-temporal data are pervasive across numerous application domains.These rich datasets can be harnessed to predict extreme events such as disease outbreaks, flooding, crime spikes, etc.
However, since the extreme events are rare, predicting them is a hard problem. Statistical methods based on extreme value theory provide a systematic way for modeling the distribution of extreme values. In particular, the generalized Pareto distribution (GPD) is useful for modeling the distribution of excess values above a certain threshold. However, applying such methods to large-scale geospatio-temporal data is a challenge due to the difficulty in capturing the complex spatial relationships between extreme events at multiple locations. This paper presents a deep learning framework for long-term prediction of the distribution of extreme values at different locations. We highlight its computational challenges and present a novel framework that combines convolutional neural networks with deep set and GPD. We demonstrate the effectiveness of our approach on a real-world dataset for modeling extreme climate events.

----

## [472] SmartIdx: Reducing Communication Cost in Federated Learning by Exploiting the CNNs Structures

**Authors**: *Donglei Wu, Xiangyu Zou, Shuyu Zhang, Haoyu Jin, Wen Xia, Binxing Fang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20345](https://doi.org/10.1609/aaai.v36i4.20345)

**Abstract**:

Top-k sparsification method is popular and powerful forreducing the communication cost in Federated Learning(FL). However, according to our experimental observation, it spends most of the total communication cost on the index of the selected parameters (i.e., their position informa-tion), which is inefficient for FL training. To solve this problem, we propose a FL compression algorithm for convolution neural networks (CNNs), called SmartIdx, by extending the traditional Top-k largest variation selection strategy intothe convolution-kernel-based selection, to reduce the proportion of the index in the overall communication cost and thusachieve a high compression ratio. The basic idea of SmartIdx is to improve the 1:1 proportion relationship betweenthe value and index of the parameters to n:1, by regarding the convolution kernel as the basic selecting unit in parameter selection, which can potentially deliver more informationto the parameter server under the limited network traffic. Tothis end, a set of rules are designed for judging which kernel should be selected and the corresponding packaging strategies are also proposed for further improving the compressionratio. Experiments on mainstream CNNs and datasets show that our proposed SmartIdx performs 2.5×−69.2× higher compression ratio than the state-of-the-art FL compression algorithms without degrading model performance.

----

## [473] Online Enhanced Semantic Hashing: Towards Effective and Efficient Retrieval for Streaming Multi-Modal Data

**Authors**: *Xiao-Ming Wu, Xin Luo, Yu-Wei Zhan, Chen-Lu Ding, Zhen-Duo Chen, Xin-Shun Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20346](https://doi.org/10.1609/aaai.v36i4.20346)

**Abstract**:

With the vigorous development of multimedia equipments and applications, efficient retrieval of large-scale multi-modal data has become a trendy research topic.  Thereinto, hashing has become a prevalent choice due to its retrieval efficiency and low storage cost. Although multi-modal hashing has drawn lots of attention in recent years, there still remain some problems. The first point is that existing methods are mainly designed in batch mode and not able to efficiently handle streaming multi-modal data. The second point is that all existing online multi-modal hashing methods fail to effectively handle unseen new classes which come continuously with streaming data chunks. In this paper, we propose a new model, termed Online enhAnced SemantIc haShing (OASIS). We design novel semantic-enhanced representation for data, which could help handle the new coming classes, and thereby construct the enhanced semantic objective function. An efficient and effective discrete online optimization algorithm is further proposed for OASIS. Extensive experiments show that our method can exceed the state-of-the-art models. For good reproducibility and benefiting the community, our code and data are already publicly available.

----

## [474] CoCoS: Enhancing Semi-supervised Learning on Graphs with Unlabeled Data via Contrastive Context Sharing

**Authors**: *Siyue Xie, Da Sun Handason Tam, Wing Cheong Lau*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20347](https://doi.org/10.1609/aaai.v36i4.20347)

**Abstract**:

Graph Neural Networks (GNNs) have recently become a popular framework for semi-supervised learning on graph-structured data. However, typical GNN models heavily rely on labeled data in the learning process, while ignoring or paying little attention to the data that are unlabeled but available. To make full use of available data, we propose a generic framework, Contrastive Context Sharing (CoCoS), to enhance the learning capacity of GNNs for semi-supervised tasks. By sharing the contextual information among nodes estimated to be in the same class, different nodes can be correlated even if they are unlabeled and remote from each other in the graph. Models can therefore learn different combinations of contextual patterns, which improves the robustness of node representations. Additionally, motivated by recent advances in self-supervised learning, we augment the context sharing strategy by integrating with contrastive learning, which naturally correlates intra-class and inter-class data. Such operations utilize all available data for training and effectively improve a model's learning capacity. CoCoS can be easily extended to a wide range of GNN-based models with little computational overheads. Extensive experiments show that CoCoS considerably enhances typical GNN models, especially when labeled data are sparse in a graph, and achieves state-of-the-art or competitive results in real-world public datasets. The code of CoCoS is available online.

----

## [475] Ensemble Semi-supervised Entity Alignment via Cycle-Teaching

**Authors**: *Kexuan Xin, Zequn Sun, Wen Hua, Bing Liu, Wei Hu, Jianfeng Qu, Xiaofang Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20348](https://doi.org/10.1609/aaai.v36i4.20348)

**Abstract**:

Entity alignment is to find identical entities in different knowledge graphs. Although embedding-based entity alignment has recently achieved remarkable progress, training data insufficiency remains a critical challenge. Conventional semi-supervised methods also suffer from the incorrect entity alignment in newly proposed training data. To resolve these issues, we design an iterative cycle-teaching framework for semi-supervised entity alignment. The key idea is to train multiple entity alignment models (called aligners) simultaneously and let each aligner iteratively teach its successor the proposed new entity alignment. We propose a diversity-aware alignment selection method to choose reliable entity alignment for each aligner. We also design a conflict resolution mechanism to resolve the alignment conflict when combining the new alignment of an aligner and that from its teacher. Besides, considering the influence of cycle-teaching order, we elaborately design a strategy to arrange the optimal order that can maximize the overall performance of multiple aligners. The cycle-teaching process can break the limitations of each model's learning capability and reduce the noise in new training data, leading to improved performance. Extensive experiments on benchmark datasets demonstrate the effectiveness of the proposed cycle-teaching framework, which significantly outperforms the state-of-the-art models when the training data is insufficient and the new entity alignment has much noise.

----

## [476] Unsupervised Adversarially Robust Representation Learning on Graphs

**Authors**: *Jiarong Xu, Yang Yang, Junru Chen, Xin Jiang, Chunping Wang, Jiangang Lu, Yizhou Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20349](https://doi.org/10.1609/aaai.v36i4.20349)

**Abstract**:

Unsupervised/self-supervised pre-training methods for graph representation learning have recently attracted increasing research interests, and they are shown to be able to generalize to various downstream applications. Yet, the adversarial robustness of such pre-trained graph learning models remains largely unexplored. More importantly, most existing defense techniques designed for end-to-end graph representation learning methods require pre-specified label definitions, and thus cannot be directly applied to the pre-training methods. In this paper, we propose an unsupervised defense technique to robustify pre-trained deep graph models, so that the perturbations on the input graph can be successfully identified and blocked before the model is applied to different downstream tasks. Specifically, we introduce a mutual information-based measure, graph representation vulnerability (GRV), to quantify the robustness of graph encoders on the representation space. We then formulate an optimization problem to learn the graph representation by carefully balancing the trade-off between the expressive power and the robustness (i.e., GRV) of the graph encoder. The discrete nature of graph topology and the joint space of graph data make the optimization problem intractable to solve. To handle the above difficulty and to reduce computational expense, we further relax the problem and thus provide an approximate solution. Additionally, we explore a provable connection between the robustness of the unsupervised graph encoder and that of models on downstream tasks. Extensive experiments demonstrate that even without access to labels and tasks, our model is still able to enhance robustness against adversarial attacks on three downstream tasks (node classification, link prediction, and community detection) by an average of +16.5% compared with existing methods.

----

## [477] Blindfolded Attackers Still Threatening: Strict Black-Box Adversarial Attacks on Graphs

**Authors**: *Jiarong Xu, Yizhou Sun, Xin Jiang, Yanhao Wang, Chunping Wang, Jiangang Lu, Yang Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20350](https://doi.org/10.1609/aaai.v36i4.20350)

**Abstract**:

Adversarial attacks on graphs have attracted considerable research interests. Existing works assume the attacker is either (partly) aware of the victim model, or able to send queries to it. These assumptions are, however, unrealistic. To bridge the gap between theoretical graph attacks and real-world scenarios, in this work, we propose a novel and more realistic setting: strict black-box graph attack, in which the attacker has no knowledge about the victim model at all and is not allowed to send any queries. To design such an attack strategy, we first propose a generic graph filter to unify different families of graph-based models. The strength of attacks can then be quantified by the change in the graph filter before and after attack. By maximizing this change, we are able to find an effective attack strategy, regardless of the underlying model. To solve this optimization problem, we also propose a relaxation technique and approximation theories to reduce the difficulty as well as the computational expense. Experiments demonstrate that, even with no exposure to the model, the Macro-F1 drops 6.4% in node classification and 29.5% in graph classification, which is a significant result compared with existent works.

----

## [478] PolygonE: Modeling N-ary Relational Data as Gyro-Polygons in Hyperbolic Space

**Authors**: *Shiyao Yan, Zequn Zhang, Xian Sun, Guangluan Xu, Shuchao Li, Qing Liu, Nayu Liu, Shensi Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20351](https://doi.org/10.1609/aaai.v36i4.20351)

**Abstract**:

N-ary relational knowledge base (KBs) embedding aims to map binary and beyond-binary facts into low-dimensional vector space simultaneously. Existing approaches typically decompose n-ary relational facts into subtuples (entity pairs, triples or quintuples, etc.), and they generally model n-ary relational KBs in Euclidean space. However, n-ary relational facts are semantically and structurally intact, decomposition leads to the loss of global information and undermines the semantical and structural integrity. Moreover, compared to the binary relational KBs, n-ary ones are characterized by more abundant and complicated hierarchy structures, which could not be well expressed in Euclidean space. To address the issues, we propose a gyro-polygon embedding approach to realize n-ary fact integrity keeping and hierarchy capturing, termed as PolygonE. Specifically, n-ary relational facts are modeled as gyro-polygons in the hyperbolic space, where we denote entities in facts as vertexes of gyro-polygons and relations as entity translocation operations. Importantly, we design a fact plausibility measuring strategy based on the vertex-gyrocentroid geodesic to optimize the relation-adjusted gyro-polygon. Extensive experiments demonstrate that PolygonE shows SOTA performance on all benchmark datasets, generalizability to binary data, and applicability to arbitrary arity fact. Finally, we also visualize the embedding to help comprehend PolygonE's awareness of hierarchies.

----

## [479] Cross-Task Knowledge Distillation in Multi-Task Recommendation

**Authors**: *Chenxiao Yang, Junwei Pan, Xiaofeng Gao, Tingyu Jiang, Dapeng Liu, Guihai Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20352](https://doi.org/10.1609/aaai.v36i4.20352)

**Abstract**:

Multi-task learning (MTL) has been widely used in recommender systems, wherein predicting each type of user feedback on items (e.g, click, purchase) are treated as individual tasks and jointly trained with a unified model. Our key observation is that the prediction results of each task may contain task-specific knowledge about user’s fine-grained preference towards items. While such knowledge could be transferred to benefit other tasks, it is being overlooked under the current MTL paradigm. This paper, instead, proposes a Cross-Task Knowledge Distillation framework that attempts to leverage prediction results of one task as supervised signals to teach another task. However, integrating MTL and KD in a proper manner is non-trivial due to several challenges including task conflicts, inconsistent magnitude and requirement of synchronous optimization. As countermeasures, we 1) introduce auxiliary tasks with quadruplet loss functions to capture cross-task fine-grained ranking information and avoid task conflicts, 2) design a calibrated distillation approach to align and distill knowledge from auxiliary tasks, and 3) propose a novel error correction mechanism to enable and facilitate synchronous training of teacher and student models. Comprehensive experiments are conducted to verify the effectiveness of our framework in real-world datasets.

----

## [480] Self-Supervised Graph Neural Networks via Diverse and Interactive Message Passing

**Authors**: *Liang Yang, Cheng Chen, Weixun Li, Bingxin Niu, Junhua Gu, Chuan Wang, Dongxiao He, Yuanfang Guo, Xiaochun Cao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20353](https://doi.org/10.1609/aaai.v36i4.20353)

**Abstract**:

By interpreting Graph Neural Networks (GNNs) as the message passing from the spatial perspective, their success is attributed to Laplacian smoothing. However, it also leads to serious over-smoothing issue by stacking many layers. Recently, many efforts have been paid to overcome this issue in semi-supervised learning. Unfortunately, it is more serious in unsupervised node representation learning task due to the lack of supervision information. Thus, most of the unsupervised or self-supervised GNNs often employ \textit{one-layer GCN} as the encoder. Essentially, the over-smoothing issue is caused by the over-simplification of the existing message passing, which possesses two intrinsic limits: blind message and uniform passing. In this paper, a novel Diverse and Interactive Message Passing (DIMP) is proposed for self-supervised learning by overcoming these limits. Firstly, to prevent the message from blindness and make it interactive between two connected nodes, the message is determined by both the two connected nodes instead of the attributes of one node. Secondly, to prevent the passing from uniformness and make it diverse over different attribute channels, different propagation weights are assigned to different elements in the message. To this end, a natural implementation of the message in DIMP is the element-wise product of the representations of two connected nodes. From the perspective of numerical optimization, the proposed DIMP is equivalent to performing an overlapping community detection via expectation-maximization (EM). Both the objective function of the community detection and the convergence of EM algorithm guarantee that DMIP can prevent from over-smoothing issue. Extensive evaluations on node-level and graph-level tasks demonstrate the superiority of DIMP on improving performance and overcoming over-smoothing issue.

----

## [481] Multi-Scale Distillation from Multiple Graph Neural Networks

**Authors**: *Chunhai Zhang, Jie Liu, Kai Dang, Wenzheng Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20354](https://doi.org/10.1609/aaai.v36i4.20354)

**Abstract**:

Knowledge Distillation (KD), which is an effective model compression and acceleration technique, has been successfully applied to graph neural networks (GNNs) recently. Existing approaches utilize a single GNN model as the teacher to distill knowledge. However, we notice that GNN models with different number of layers demonstrate different classification abilities on nodes with different degrees. On the one hand, for nodes with high degrees, their local structures are dense and complex, hence more message passing is needed. Therefore, GNN models with more layers perform better. On the other hand, for nodes with low degrees, whose local structures are relatively sparse and simple, the repeated message passing can easily lead to over-smoothing. Thus, GNN models with less layers are more suitable. However, existing single-teacher GNN knowledge distillation approaches which are based on a single GNN model, are sub-optimal. To this end, we propose a novel approach to distill multi-scale knowledge, which learns from multiple GNN teacher models with different number of layers to capture the topological semantic at different scales. Instead of learning from the teacher models equally, the proposed method automatically assigns proper weights for each teacher model via an attention mechanism which enables the student to select teachers for different local structures. Extensive experiments are conducted to evaluate the proposed method on four public datasets. The experimental results demonstrate the superiority of our proposed method over state-of-the-art methods. Our code is publicly available at https://github.com/NKU-IIPLab/MSKD.

----

## [482] Mind the Gap: Cross-Lingual Information Retrieval with Hierarchical Knowledge Enhancement

**Authors**: *Fuwei Zhang, Zhao Zhang, Xiang Ao, Dehong Gao, Fuzhen Zhuang, Yi Wei, Qing He*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20355](https://doi.org/10.1609/aaai.v36i4.20355)

**Abstract**:

Cross-Lingual Information Retrieval (CLIR) aims to rank the documents written in a language different from the user’s query. The intrinsic gap between different languages is an essential challenge for CLIR. In this paper, we introduce the multilingual knowledge graph (KG) to the CLIR task due to the sufficient information of entities in multiple languages. It is regarded as a “silver bullet” to simultaneously perform explicit alignment between queries and documents and also broaden the representations of queries. And we propose a model named CLIR with HIerarchical Knowledge Enhancement (HIKE) for our task. The proposed model encodes the textual information in queries, documents and the KG with multilingual BERT, and incorporates the KG information in the query-document matching process with a hierarchical information fusion mechanism. Particularly, HIKE first integrates the entities and their neighborhood in KG into query representations with a knowledge-level fusion, then combines the knowledge from both source and target languages to further mitigate the linguistic gap with a language-level fusion. Finally, experimental results demonstrate that HIKE achieves substantial improvements over state-of-the-art competitors.

----

## [483] Anisotropic Additive Quantization for Fast Inner Product Search

**Authors**: *Jin Zhang, Qi Liu, Defu Lian, Zheng Liu, Le Wu, Enhong Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20356](https://doi.org/10.1609/aaai.v36i4.20356)

**Abstract**:

Maximum Inner Product Search (MIPS) plays an important role in many applications ranging from information retrieval, recommender systems to natural language processing and machine learning. However, exhaustive MIPS is often expensive and impractical when there are a large number of candidate items. The state-of-the-art approximated MIPS is product quantization with a score-aware loss, which weighs more heavily on items with larger inner product scores. However, it is challenging to extend the score-aware loss for additive quantization due to parallel-orthogonal decomposition of residual error. Learning additive quantization with respect to this loss is important since additive quantization can achieve a lower approximation error than product quantization. To this end, we propose a quantization method called Anisotropic Additive Quantization to combine the score-aware anisotropic loss and additive quantization. To efficiently update the codebooks in this algorithm, we develop a new alternating optimization algorithm. The proposed algorithm is extensively evaluated on three real-world datasets. The experimental results show that it outperforms the state-of-the-art baselines with respect to approximate search accuracy while guaranteeing a similar retrieval efficiency.

----

## [484] Robust Heterogeneous Graph Neural Networks against Adversarial Attacks

**Authors**: *Mengmei Zhang, Xiao Wang, Meiqi Zhu, Chuan Shi, Zhiqiang Zhang, Jun Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20357](https://doi.org/10.1609/aaai.v36i4.20357)

**Abstract**:

Heterogeneous Graph Neural Networks (HGNNs) have drawn increasing attention in recent years and achieved outstanding performance in many tasks. However, despite their wide use, there is currently no understanding of their robustness to adversarial attacks. In this work, we first systematically study the robustness of HGNNs and show that they can be easily fooled by adding the adversarial edge between the target node and large-degree node (i.e., hub). Furthermore, we show two key reasons for such vulnerability of HGNNs: one is perturbation enlargement effect, i.e., HGNNs, failing to encode transiting probability, will enlarge the effect of the adversarial hub in comparison of GCNs, and the other is soft attention mechanism, i.e., such mechanism assigns positive attention values to obviously unreliable neighbors. Based on the two facts, we propose a novel robust HGNN framework RoHe against topology adversarial attacks by equipping an attention purifier, which can prune malicious neighbors based on topology and feature. Specifically, to eliminate the perturbation enlargement, we introduce the metapath-based transiting probability as the prior criterion of the purifier, restraining the confidence of malicious neighbors from the adversarial hub. Then the purifier learns to mask out neighbors with low confidence, thus can effectively alleviate the negative effect of malicious neighbors in the soft attention mechanism. Extensive experiments on different benchmark datasets for multiple HGNNs are conducted, where the considerable improvement of HGNNs under adversarial attacks will demonstrate the effectiveness and generalization ability of our defense framework.

----

## [485] Multi-Dimensional Prediction of Guild Health in Online Games: A Stability-Aware Multi-Task Learning Approach

**Authors**: *Chuang Zhao, Hongke Zhao, Runze Wu, Qilin Deng, Yu Ding, Jianrong Tao, Changjie Fan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20358](https://doi.org/10.1609/aaai.v36i4.20358)

**Abstract**:

Guild is the most important long-term virtual community and emotional bond in massively multiplayer online role-playing games (MMORPGs). It matters a lot to the player retention and game ecology how the guilds are going, e.g., healthy or not. The main challenge now is to characterize and predict the guild health in a quantitative, dynamic, and multi-dimensional manner based on complicated multi-media data streams. To this end, we propose a novel framework, namely Stability-Aware Multi-task Learning Approach(SAMLA) to address these challenges. Specifically, different media-specific modules are designed to extract information from multiple media types of tabular data, time seriescharacteristics, and heterogeneous graphs. To capture the dynamics of guild health, we introduce a representation encoder to provide a time series view of multi-media data that is used for task prediction. Inspiredby well-received theories on organization management, we delicately define five specific and quantitative dimensions of guild health and make parallel predictions based on a multi-task approach. Besides, we devise a novel auxiliary task, i.e.,the guild stability, to boost the performance of the guild health prediction task. Extensive experiments on a real-world large-scale MMORPG dataset verify that our proposed method outperforms the state-of-the-art methods in the task of organizational health characterization and prediction. Moreover, our work has been practically deployed in online MMORPG, and case studies clearly illustrate the significant value.

----

## [486] Multi-View Intent Disentangle Graph Networks for Bundle Recommendation

**Authors**: *Sen Zhao, Wei Wei, Ding Zou, Xianling Mao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20359](https://doi.org/10.1609/aaai.v36i4.20359)

**Abstract**:

Bundle recommendation aims to recommend the user a bundle of items as a whole. Previous models capture user’s preferences on both items and the association of items. Nevertheless, they usually neglect the diversity of user’s intents on adopting items and fail to disentangle user’s intents in representations. In the real scenario of bundle recommendation, a user’s intent may be naturally distributed in the different bundles of that user (Global view). And a bundle may contain multiple intents of a user (Local view). Each view has its advantages for intent disentangling: 1) In the global view, more items are involved to present each intent, which can demonstrate the user’s preference under each intent more clearly. 2) The local view can reveal the association between items under  each intent since the items within the same bundle are highly correlated to each other. To this end, in this paper we propose a novel model named Multi-view Intent Disentangle Graph Networks (MIDGN), which is capable of precisely and comprehensively capturing the diversity of user intent and items’ associations at the finer granularity. Specifically, MIDGN disentangles user’s intents from two different perspectives, respectively: 1) taking the Global view, MIDGN disentangles  the user’s intent coupled with inter-bundle items; 2) taking the Local view, MIDGN disentangles the user’s intent coupled with items within each bundle. Meanwhile, we compare user’s intents disentangled from different views by a contrast method to improve the learned intents. Extensive experiments are conducted on two benchmark datasets and MIDGN outperforms the state-of-the-art methods by over 10.7% and 26.8%, respectively.

----

## [487] Multi-Type Urban Crime Prediction

**Authors**: *Xiangyu Zhao, Wenqi Fan, Hui Liu, Jiliang Tang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20360](https://doi.org/10.1609/aaai.v36i4.20360)

**Abstract**:

Crime prediction plays an impactful role in enhancing public security and sustainable development of urban. With recent advances in data collection and integration technologies, a large amount of urban data with rich crime-related information and fine-grained spatio-temporal logs have been recorded. Such helpful information can boost our understandings of the temporal evolution and spatial factors of urban crimes and can enhance accurate crime prediction. However, the vast majority of existing crime prediction algorithms either do not distinguish different types of crime or treat each crime type separately, which fails to capture the intrinsic correlations among different types of crime. In this paper, we perform crime prediction exploiting the cross-type and spatio-temporal correlations of urban crimes. In particular, we verify the existence of correlations among different types of crime from temporal and spatial perspectives, and propose a coherent framework to mathematically model these correlations for crime prediction. Extensive experiments on real-world datasets validate the effectiveness of our framework.

----

## [488] Forecasting Asset Dependencies to Reduce Portfolio Risk

**Authors**: *Haoren Zhu, Shih-Yang Liu, Pengfei Zhao, Yingying Chen, Dik Lun Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20361](https://doi.org/10.1609/aaai.v36i4.20361)

**Abstract**:

Financial assets exhibit dependence structures, i.e., movements of their prices or returns show various correlations. Knowledge of assets’ price dependencies can help investors to create a diversified portfolio, aiming to reduce portfolio risk due to the high volatility of the financial market. Since asset dependency changes with time in complex patterns, asset dependency forecast is an essential problem in finance. In this paper, we organize pairwise assets dependencies in an Asset Dependency Matrix (ADM) and formulate the problem of assets dependencies forecast to predict the future ADM given a sequence of past ADMs. We propose a novel idea viewing a sequence of ADMs as a sequence of images to capture the spatial and temporal dependencies among the assets. Inspired by video prediction tasks, we develop a novel Asset Dependency Neural Network (ADNN) to tackle the ADM prediction problem. Experiments show that our proposed framework consistently outperforms baselines on both future ADM prediction and portfolio risk reduction tasks.

----

## [489] Defending Graph Convolutional Networks against Dynamic Graph Perturbations via Bayesian Self-Supervision

**Authors**: *Jun Zhuang, Mohammad Al Hasan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20362](https://doi.org/10.1609/aaai.v36i4.20362)

**Abstract**:

In recent years, plentiful evidence illustrates that Graph Convolutional Networks (GCNs) achieve extraordinary accomplishments on the node classification task. However, GCNs may be vulnerable to adversarial attacks on label-scarce dynamic graphs. Many existing works aim to strengthen the robustness of GCNs; for instance, adversarial training is used to shield GCNs against malicious perturbations. However, these works fail on dynamic graphs for which label scarcity is a pressing issue. To overcome label scarcity, self-training attempts to iteratively assign pseudo-labels to highly confident unlabeled nodes but such attempts may suffer serious degradation under dynamic graph perturbations. In this paper, we generalize noisy supervision as a kind of self-supervised learning method and then propose a novel Bayesian self-supervision model, namely GraphSS, to address the issue. Extensive experiments demonstrate that GraphSS can not only affirmatively alert the perturbations on dynamic graphs but also effectively recover the prediction of a node classifier when the graph is under such perturbations. These two advantages prove to be generalized over three classic GCNs across five public graph datasets.

----

## [490] Can Machines Read Coding Manuals Yet? - A Benchmark for Building Better Language Models for Code Understanding

**Authors**: *Ibrahim Abdelaziz, Julian Dolby, Jamie P. McCusker, Kavitha Srinivas*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20363](https://doi.org/10.1609/aaai.v36i4.20363)

**Abstract**:

Code understanding is an increasingly important application of Artificial Intelligence. A fundamental aspect of understanding code is understanding text about code, e.g., documentation  and forum discussions.  Pre-trained language models (e.g., BERT) are a popular approach for various NLP tasks, and there are now a variety of benchmarks, such as GLUE, to help improve the development of such models for natural language understanding. However, little is known about how well such models work on textual artifacts about code, and we are unaware of any systematic set of downstream tasks for such an evaluation.  In this paper, we derive a set of benchmarks (BLANCA - Benchmarks for LANguage models on Coding Artifacts) that assess code understanding based on tasks such as predicting the best answer to a question in a forum post, finding related forum posts, or predicting classes related in a hierarchy from class documentation.  We evaluate performance of current state-of-the-art language models on these tasks and show that there is significant improvement on each task from fine tuning.  We also show that multi-task training over BLANCA tasks help build better language models for code understanding.

----

## [491] No Task Left Behind: Multi-Task Learning of Knowledge Tracing and Option Tracing for Better Student Assessment

**Authors**: *Suyeong An, Junghoon Kim, Minsam Kim, Juneyoung Park*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20364](https://doi.org/10.1609/aaai.v36i4.20364)

**Abstract**:

Student assessment is one of the most fundamental tasks in the field of AI Education (AIEd). One of the most common approach to student assessment is Knowledge Tracing (KT), which evaluates a student's knowledge state by predicting whether the student will answer a given question correctly or not. However, in the context of multiple choice (polytomous) questions, conventional KT approaches are limited in that they only consider the binary (dichotomous) correctness label (i.e., correct or incorrect), and disregard the specific option chosen by the student. 
 Meanwhile, Option Tracing (OT) attempts to model a student by predicting which option they will choose for a given question, but overlooks the correctness information. In this paper, we propose Dichotomous-Polytomous Multi-Task Learning (DP-MTL), a multi-task learning framework that combines KT and OT for more precise student assessment. In particular, we show that the KT objective acts as a regularization term for OT in the DP-MTL framework, and propose an appropriate architecture for applying our method on top of existing deep learning-based KT models. We experimentally confirm that DP-MTL significantly improves both KT and OT performances, and also benefits downstream tasks such as Score Prediction (SP).

----

## [492] Diaformer: Automatic Diagnosis via Symptoms Sequence Generation

**Authors**: *Junying Chen, Dongfang Li, Qingcai Chen, Wenxiu Zhou, Xin Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20365](https://doi.org/10.1609/aaai.v36i4.20365)

**Abstract**:

Automatic diagnosis has attracted increasing attention but remains challenging due to multi-step reasoning. Recent works usually address it by reinforcement learning methods. However, these methods show low efficiency and require task-specific reward functions. Considering the conversation between doctor and patient allows doctors to probe for symptoms and make diagnoses, the diagnosis process can be naturally seen as the generation of a sequence including symptoms and diagnoses. Inspired by this, we reformulate automatic diagnosis as a symptoms Sequence Generation (SG) task and propose a simple but effective automatic Diagnosis model based on Transformer (Diaformer). We firstly design the symptom attention framework to learn the generation of symptom inquiry and the disease diagnosis. To alleviate the discrepancy between sequential generation and disorder of implicit symptoms, we further design three orderless training mechanisms. Experiments on three public datasets show that our model outperforms baselines on disease diagnosis by 1%, 6% and 11.5% with the highest training efficiency. Detailed analysis on symptom inquiry prediction demonstrates that the potential of applying symptoms sequence generation for automatic diagnosis.

----

## [493] Zero-Shot Audio Source Separation through Query-Based Learning from Weakly-Labeled Data

**Authors**: *Ke Chen, Xingjian Du, Bilei Zhu, Zejun Ma, Taylor Berg-Kirkpatrick, Shlomo Dubnov*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20366](https://doi.org/10.1609/aaai.v36i4.20366)

**Abstract**:

Deep learning techniques for separating audio into different sound sources face several challenges. Standard architectures require training separate models for different types of audio sources. Although some universal separators employ a single model to target multiple sources, they have difficulty generalizing to unseen sources. In this paper, we propose a three-component pipeline to train a universal audio source separator from a large, but weakly-labeled dataset: AudioSet. First, we propose a transformer-based sound event detection system for processing weakly-labeled training data. Second, we devise a query-based audio separation model that leverages this data for model training. Third, we design a latent embedding processor to encode queries that specify audio targets for separation, allowing for zero-shot generalization. Our approach uses a single model for source separation of multiple sound types, and relies solely on weakly-labeled data for training. In addition, the proposed audio separator can be used in a zero-shot setting, learning to separate types of audio sources that were never seen in training. To evaluate the separation performance, we test our model on MUSDB18, while training on the disjoint AudioSet. We further verify the zero-shot performance by conducting another experiment on audio source types that are held-out from training. The model achieves comparable Source-to-Distortion Ratio (SDR) performance to current supervised models in both cases.

----

## [494] DeepHardMark: Towards Watermarking Neural Network Hardware

**Authors**: *Joseph Clements, Yingjie Lao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20367](https://doi.org/10.1609/aaai.v36i4.20367)

**Abstract**:

This paper presents a framework for embedding watermarks into DNN hardware accelerators. Unlike previous works that have looked at protecting the algorithmic intellectual properties of deep learning systems, this work proposes a methodology for defending deep learning hardware. Our methodology embeds modifications into the hardware accelerator's functional blocks that can be revealed with the rightful owner's key DNN and corresponding key sample, verifying the legitimate owner. We propose an Lp-box ADMM based algorithm to co-optimize watermark's hardware overhead and impact on the design's algorithmic functionality. We evaluate the performance of the hardware watermarking scheme on popular image classifier models using various accelerator designs. Our results demonstrate that the proposed methodology effectively embeds watermarks while preserving the original functionality of the hardware architecture. Specifically, we can successfully embed watermarks into the deep learning hardware and reliably execute a ResNet ImageNet classifiers with an accuracy degradation of only 0.009%

----

## [495] A Unified Framework for Real Time Motion Completion

**Authors**: *Yinglin Duan, Yue Lin, Zhengxia Zou, Yi Yuan, Zhehui Qian, Bohan Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20368](https://doi.org/10.1609/aaai.v36i4.20368)

**Abstract**:

Motion completion, as a challenging and fundamental problem, is of great significance in film and game applications. For different motion completion application scenarios (in-betweening, in-filling, and blending), most previous methods deal with the completion problems with case-by-case methodology designs. In this work, we propose a simple but effective method to solve multiple motion completion problems under a unified framework and achieves a new state-of-the-art accuracy on LaFAN1 (+17% better than previous sota) under multiple evaluation settings. Inspired by the recent great success of self-attention-based transformer models, we consider the completion as a sequence-to-sequence prediction problem. Our method consists of three modules - a standard transformer encoder with self-attention that learns long-range dependencies of input motions, a trainable mixture embedding module that models temporal information and encodes different key-frame combinations in a unified form, and a new motion perceptual loss for better capturing high-frequency movements. Our method can predict multiple missing frames within a single forward propagation in real-time and get rid of the post-processing requirement. We also introduce a novel large-scale dance movement dataset for exploring the scaling capability of our method and its effectiveness in complex motion applications.

----

## [496] FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns

**Authors**: *Yitong Duan, Lei Wang, Qizhong Zhang, Jian Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20369](https://doi.org/10.1609/aaai.v36i4.20369)

**Abstract**:

As an asset pricing model in economics and finance, factor model has been widely used in quantitative investment. Towards building more effective factor models, recent years have witnessed the paradigm shift from linear models to more flexible nonlinear data-driven machine learning models. However, due to low signal-to-noise ratio of the financial data, it is quite challenging to learn effective factor models. In this paper, we propose a novel factor model, FactorVAE, as a probabilistic model with inherent randomness for noise modeling. Essentially, our model integrates the dynamic factor model (DFM) with the variational autoencoder (VAE) in machine learning, and we propose a prior-posterior learning method based on VAE, which can effectively guide the learning of model by approximating an optimal posterior factor model with future information. Particularly, considering that risk modeling is important for the noisy stock data, FactorVAE can estimate the variances from the distribution over the latent space of VAE, in addition to predicting returns. The experiments on the real stock market data demonstrate the effectiveness of FactorVAE, which outperforms various baseline methods.

----

## [497] AXM-Net: Implicit Cross-Modal Feature Alignment for Person Re-identification

**Authors**: *Ammarah Farooq, Muhammad Awais, Josef Kittler, Syed Safwan Khalid*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20370](https://doi.org/10.1609/aaai.v36i4.20370)

**Abstract**:

Cross-modal person re-identification (Re-ID) is critical for modern video surveillance systems. The key challenge is to align cross-modality representations conforming to semantic information present for a person and ignore background information. This work presents a novel convolutional neural network (CNN) based architecture designed to learn semantically aligned cross-modal visual and textual representations. The underlying building block, named AXM-Block, is a unified multi-layer network that dynamically exploits the multi-scale knowledge from both modalities and re-calibrates each modality according to shared semantics. To complement the convolutional design, contextual attention is applied in the text branch to manipulate long-term dependencies. Moreover, we propose a unique design to enhance visual part-based feature coherence and locality information. Our framework is novel in its ability to implicitly learn aligned semantics between modalities during the feature learning stage. The unified feature learning effectively utilizes textual data as a super-annotation signal for visual representation learning and automatically rejects irrelevant information. The entire AXM-Net is trained end-to-end on CUHK-PEDES data. We report results on two tasks, person search and cross-modal Re-ID. The AXM-Net outperforms the current state-of-the-art (SOTA) methods and achieves 64.44% Rank@1 on the CUHK-PEDES test set. It also outperforms by >10% for cross-viewpoint text-to-image Re-ID scenarios on CrossRe-ID and CUHK-SYSU datasets.

----

## [498] SCIR-Net: Structured Color Image Representation Based 3D Object Detection Network from Point Clouds

**Authors**: *Qingdong He, Hao Zeng, Yi Zeng, Yijun Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20371](https://doi.org/10.1609/aaai.v36i4.20371)

**Abstract**:

3D object detection from point clouds data has become an indispensable part in autonomous driving. Previous works for processing point clouds lie in either projection or voxelization. However, projection-based methods suffer from information loss while voxelization-based methods bring huge computation. In this paper, we propose to encode point clouds into structured color image representation (SCIR) and utilize 2D CNN to fulfill the 3D detection task. Specifically, we use the structured color image encoding module to convert the irregular 3D point clouds into a squared 2D tensor image, where each point corresponds to a spatial point in the 3D space. Furthermore, in order to fit for the Euclidean structure, we apply feature normalization to parameterize the 2D tensor image onto a regular dense color image. Then, we conduct repeated multi-scale fusion with different levels so as to augment the initial features and learn scale-aware feature representations for box prediction. Extensive experiments on KITTI benchmark, Waymo Open Dataset and more challenging nuScenes dataset show that our proposed method yields decent results and demonstrate the effectiveness of such representations for point clouds.

----

## [499] Learning and Dynamical Models for Sub-seasonal Climate Forecasting: Comparison and Collaboration

**Authors**: *Sijie He, Xinyan Li, Laurie Trenary, Benjamin A. Cash, Timothy DelSole, Arindam Banerjee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20372](https://doi.org/10.1609/aaai.v36i4.20372)

**Abstract**:

Sub-seasonal forecasting (SSF) is the prediction of key climate variables such as temperature and precipitation on the 2-week to 2-month time horizon. Skillful SSF would have substantial societal value in areas such as agricultural productivity, hydrology and water resource management, and emergency planning for extreme events such as droughts and wildfires. Despite its societal importance, SSF has stayed a challenging problem compared to both short-term weather forecasting and long-term seasonal forecasting. Recent studies have shown the potential of machine learning (ML) models to advance SSF. In this paper, for the first time, we perform a fine-grained comparison of a suite of modern ML models with start-of-the-art physics-based dynamical models from the Subseasonal Experiment (SubX) project for SSF in the western contiguous United States. Additionally, we explore mechanisms to enhance the ML models by using forecasts from dynamical models. Empirical results illustrate that, on average, ML models outperform dynamical models while the ML models tend to generate forecasts with conservative magnitude compared to the SubX models. Further, we illustrate that ML models make forecasting errors under extreme weather conditions, e.g., cold waves due to the polar vortex, highlighting the need for separate models for extreme events. Finally, we show that suitably incorporating dynamical model forecasts as inputs to ML models can substantially improve the forecasting performance of the ML models. The SSF dataset constructed for the work and code for the ML models are released along with the paper for the benefit of the artificial intelligence community.

----

## [500] Solving PDE-Constrained Control Problems Using Operator Learning

**Authors**: *Rakhoon Hwang, Jae Yong Lee, Jinyoung Shin, Hyung Ju Hwang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20373](https://doi.org/10.1609/aaai.v36i4.20373)

**Abstract**:

The modeling and control of complex physical systems are essential in real-world problems. We propose a novel framework that is generally applicable to solving PDE-constrained optimal control problems by introducing surrogate models for PDE solution operators with special regularizers. The procedure of the proposed framework is divided into two phases: solution operator learning for PDE constraints (Phase 1) and searching for optimal control (Phase 2). Once the surrogate model is trained in Phase 1, the optimal control can be inferred in Phase 2 without intensive computations. Our framework can be applied to both data-driven and data-free cases. We demonstrate the successful application of our method to various optimal control problems for different control variables with diverse PDE constraints from the Poisson equation to Burgers' equation.

----

## [501] Proxy Learning of Visual Concepts of Fine Art Paintings from Styles through Language Models

**Authors**: *Diana Kim, Ahmed Elgammal, Marian Mazzone*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20374](https://doi.org/10.1609/aaai.v36i4.20374)

**Abstract**:

We present a machine learning system that can quantify fine art paintings with a set of visual elements and principles of art. The formal analysis is fundamental for understanding art, but developing such a system is challenging. Paintings have high visual complexities, but it is also difficult to collect enough training data with direct labels. To resolve these practical limitations, we introduce a novel mechanism, called proxy learning, which learns visual concepts in paintings through their general relation to styles. This framework does not require any visual annotation, but only uses style labels and a general relationship between visual concepts and style. In this paper, we propose a novel proxy model and reformulate four pre-existing methods in the context of proxy learning.  Through quantitative and qualitative comparison, we evaluate these methods and compare their effectiveness in quantifying the artistic visual concepts, where the general relationship is estimated by language models; GloVe or BERT. The language modeling is a practical and scalable solution requiring no labeling, but it is inevitably imperfect. We demonstrate how the new proxy model is robust to the imperfection, while the other methods are sensitively affected by it.

----

## [502] SPATE-GAN: Improved Generative Modeling of Dynamic Spatio-Temporal Patterns with an Autoregressive Embedding Loss

**Authors**: *Konstantin Klemmer, Tianlin Xu, Beatrice Acciaio, Daniel B. Neill*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20375](https://doi.org/10.1609/aaai.v36i4.20375)

**Abstract**:

From ecology to atmospheric sciences, many academic disciplines deal with data characterized by intricate spatio-temporal complexities, the modeling of which often requires specialized approaches. Generative models of these data are of particular interest, as they enable a range of impactful downstream applications like simulation or creating synthetic training data. Recently, COT-GAN, a new GAN algorithm inspired by the theory of causal optimal transport (COT), was proposed in an attempt to improve generation of sequential data.  However, the task of learning complex patterns over time and space requires additional knowledge of the specific data structures. In this study, we propose a novel loss objective combined with COT-GAN based on an autoregressive embedding to reinforce the learning of spatio-temporal dynamics. We devise SPATE (spatio-temporal association), a new metric measuring spatio-temporal autocorrelation. We compute SPATE for real and synthetic data samples and use it to compute an embedding loss that considers space-time interactions, nudging the GAN to learn outputs that are faithful to the observed dynamics. We test our new SPATE-GAN on a diverse set of spatio-temporal patterns: turbulent flows, log-Gaussian Cox processes and global weather data. We show that our novel embedding loss improves performance without any changes to the architecture of the GAN backbone, highlighting our model's increased capacity for capturing autoregressive structures.

----

## [503] Intra-Inter Subject Self-Supervised Learning for Multivariate Cardiac Signals

**Authors**: *Xiang Lan, Dianwen Ng, Shenda Hong, Mengling Feng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20376](https://doi.org/10.1609/aaai.v36i4.20376)

**Abstract**:

Learning information-rich and generalizable representations effectively from unlabeled multivariate cardiac signals to identify abnormal heart rhythms (cardiac arrhythmias) is valuable in real-world clinical settings but often challenging due to its complex temporal dynamics. Cardiac arrhythmias can vary significantly in temporal patterns even for the same patient (i.e., intra subject difference). Meanwhile, the same type of cardiac arrhythmia can show different temporal patterns among different patients due to different cardiac structures (i.e., inter subject difference). In this paper, we address the challenges by proposing an Intra-Inter Subject Self-Supervised Learning (ISL) model that is customized for multivariate cardiac signals. Our proposed ISL model integrates medical knowledge into self-supervision to effectively learn from intra-inter subject differences. In intra subject self-supervision, ISL model first extracts heartbeat-level features from each subject using a channel-wise attentional CNN-RNN encoder. Then a stationarity test module is employed to capture the temporal dependencies between heartbeats. In inter subject self-supervision, we design a set of data augmentations according to the clinical characteristics of cardiac signals and perform contrastive learning among subjects to learn distinctive representations for various types of patients. Extensive experiments on three real-world datasets were conducted. In a semi-supervised transfer learning scenario, our pre-trained ISL model leads about 10% improvement over supervised training when only 1% labeled data is available, suggesting strong generalizability and robustness of the model.

----

## [504] GeomGCL: Geometric Graph Contrastive Learning for Molecular Property Prediction

**Authors**: *Shuangli Li, Jingbo Zhou, Tong Xu, Dejing Dou, Hui Xiong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20377](https://doi.org/10.1609/aaai.v36i4.20377)

**Abstract**:

Recently many efforts have been devoted to applying graph neural networks (GNNs) to molecular property prediction which is a fundamental task for computational drug and material discovery. One of major obstacles to hinder the successful prediction of molecular property by GNNs is the scarcity of labeled data. Though graph contrastive learning (GCL) methods have achieved extraordinary performance with insufficient labeled data, most focused on designing data augmentation schemes for general graphs. However, the fundamental property of a molecule could be altered with the augmentation method (like random perturbation) on molecular graphs. Whereas, the critical geometric information of molecules remains rarely explored under the current GNN and GCL architectures. To this end, we propose a novel graph contrastive learning method utilizing the geometry of the molecule across
2D and 3D views, which is named GeomGCL. Specifically, we first devise a dual-view geometric message passing network (GeomMPNN) to adaptively leverage the rich information of both 2D and 3D graphs of a molecule. The incorporation of geometric properties at different levels can greatly facilitate the molecular representation learning. Then a novel geometric graph contrastive scheme is designed to make both geometric views collaboratively supervise each other to improve the generalization ability of GeomMPNN. We evaluate GeomGCL on various downstream property prediction tasks via a finetune process. Experimental results on seven real-life molecular datasets demonstrate the effectiveness of our proposed GeomGCL against state-of-the-art baselines.

----

## [505] OAM: An Option-Action Reinforcement Learning Framework for Universal Multi-Intersection Control

**Authors**: *Enming Liang, Zicheng Su, Chilin Fang, Renxin Zhong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20378](https://doi.org/10.1609/aaai.v36i4.20378)

**Abstract**:

Efficient traffic signal control is an important means to alleviate urban traffic congestion. Reinforcement learning (RL) has shown great potentials in devising optimal signal plans that can adapt to dynamic traffic congestion. However, several challenges still need to be overcome. Firstly, a paradigm of state, action, and reward design is needed, especially for an optimality-guaranteed reward function. Secondly, the generalization of the RL algorithms is hindered by the varied topologies and physical properties of intersections. Lastly, enhancing the cooperation between intersections is needed for large network applications. To address these issues, the Option-Action RL framework for universal Multi-intersection control (OAM) is proposed. Based on the well-known cell transmission model, we first define a lane-cell-level state to better model the traffic flow propagation. Based on this physical queuing dynamics, we propose a regularized delay as the reward to facilitate temporal credit assignment while maintaining the equivalence with minimizing the average travel time. We then recapitulate the phase actions as the constrained combinations of lane options and design a universal neural network structure to realize model generalization to any intersection with any phase definition. The multiple-intersection cooperation is then rigorously discussed using the potential game theory.
 
 We test the OAM algorithm under four networks with different settings, including a city-level scenario with 2,048 intersections using synthetic and real-world datasets. The results show that the OAM can outperform the state-of-the-art controllers in reducing the average travel time.

----

## [506] End-to-End Line Drawing Vectorization

**Authors**: *Hanyuan Liu, Chengze Li, Xueting Liu, Tien-Tsin Wong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20379](https://doi.org/10.1609/aaai.v36i4.20379)

**Abstract**:

Vector graphics is broadly used in a variety of forms, such as illustrations, logos, posters, billboards, and printed ads. Despite its broad use, many artists still prefer to draw with pen and paper, which leads to a high demand of converting raster designs into the vector form. In particular, line drawing is a primary art and attracts many research efforts in automatically converting raster line drawings to vector form. However, the existing methods generally adopt a two-step approach, stroke segmentation and vectorization. Without vector guidance, the raster-based stroke segmentation frequently obtains unsatisfying segmentation results, such as over-grouped strokes and broken strokes. In this paper, we make an attempt in proposing an end-to-end vectorization method which directly generates vectorized stroke primitives from raster line drawing in one step. We propose a Transformer-based framework to perform stroke tracing like human does in an automatic stroke-by-stroke way with a novel stroke feature representation and multi-modal supervision to achieve vectorization with high quality and fidelity. Qualitative and quantitative evaluations show that our method achieves state of the art performance.

----

## [507] Context-Aware Health Event Prediction via Transition Functions on Dynamic Disease Graphs

**Authors**: *Chang Lu, Tian Han, Yue Ning*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20380](https://doi.org/10.1609/aaai.v36i4.20380)

**Abstract**:

With the wide application of electronic health records (EHR) in healthcare facilities, health event prediction with deep learning has gained more and more attention. A common feature of EHR data used for deep-learning-based predictions is historical diagnoses. Existing work mainly regards a diagnosis as an independent disease and does not consider clinical relations among diseases in a visit. Many machine learning approaches assume disease representations are static in different visits of a patient. However, in real practice, multiple diseases that are frequently diagnosed at the same time reflect hidden patterns that are conducive to prognosis. Moreover, the development of a disease is not static since some diseases can emerge or disappear and show various symptoms in different visits of a patient. To effectively utilize this combinational disease information and explore the dynamics of diseases, we propose a novel context-aware learning framework using transition functions on dynamic disease graphs. Specifically, we construct a global disease co-occurrence graph with multiple node properties for disease combinations. We design dynamic subgraphs for each patient's visit to leverage global and local contexts. We further define three diagnosis roles in each visit based on the variation of node properties to model disease transition processes. Experimental results on two real-world EHR datasets show that the proposed model outperforms state of the art in predicting health events.

----

## [508] Hyperverlet: A Symplectic Hypersolver for Hamiltonian Systems

**Authors**: *Frederik Baymler Mathiesen, Bin Yang, Jilin Hu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20381](https://doi.org/10.1609/aaai.v36i4.20381)

**Abstract**:

Hamiltonian systems represent an important class of dynamical systems such as pendulums, molecular dynamics, and cosmic systems. The choice of solvers is significant to the accuracy when simulating Hamiltonian systems, where symplectic solvers show great significance. Recent advances in neural network-based hypersolvers, though achieve competitive results, still lack the symplecity necessary for reliable simulations, especially over long time horizons. To alleviate this, we introduce Hyperverlet, a new hypersolver composing the traditional, symplectic velocity Verlet and symplectic neural network-based solvers. More specifically, we propose a parameterization of symplectic neural networks and prove that hyperbolic tangent is r-finite expanding the set of allowable activation functions for symplectic neural networks, improving the accuracy. Extensive experiments on a spring-mass and a pendulum system justify the design choices and suggest that Hyperverlet outperforms both traditional solvers and hypersolvers.

----

## [509] Learning Human Driving Behaviors with Sequential Causal Imitation Learning

**Authors**: *Kangrui Ruan, Xuan Di*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20382](https://doi.org/10.1609/aaai.v36i4.20382)

**Abstract**:

Learning human driving behaviors is an efficient approach for self-driving vehicles. Traditional Imitation Learning (IL) methods assume that the expert demonstrations follow Markov Decision Processes (MDPs). However, in reality, this assumption does not always hold true. Spurious correlation may exist through the paths of historical variables because of the existence of unobserved confounders. Accounting for the latent causal relationships from unobserved variables to outcomes, this paper proposes Sequential Causal Imitation Learning (SeqCIL) for imitating driver behaviors. We develop a sequential causal template that generalizes the default MDP settings to one with Unobserved Confounders (MDPUC-HD). Then we develop a sufficient graphical criterion to determine when ignoring causality leads to poor performances in MDPUC-HD. Through the framework of Adversarial Imitation Learning, we develop a procedure to imitate the expert policy by blocking π-backdoor paths at each time step. Our methods are evaluated on a synthetic dataset and a real-world highway driving dataset, both demonstrating that the proposed procedure significantly outperforms non-causal imitation learning methods.

----

## [510] EMVLight: A Decentralized Reinforcement Learning Framework for Efficient Passage of Emergency Vehicles

**Authors**: *Haoran Su, Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20383](https://doi.org/10.1609/aaai.v36i4.20383)

**Abstract**:

Emergency vehicles (EMVs) play a crucial role in responding to time-critical events such as medical emergencies and fire outbreaks in an urban area. The less time EMVs spend traveling through the traffic, the more likely it would help save people's lives and reduce property loss. To reduce the travel time of EMVs, prior work has used route optimization based on historical traffic-flow data and traffic signal pre-emption based on the optimal route. However, traffic signal pre-emption dynamically changes the traffic flow which, in turn, modifies the optimal route of an EMV. In addition, traffic signal pre-emption practices usually lead to significant disturbances in traffic flow and subsequently increase the travel time for non-EMVs. In this paper, we propose EMVLight, a decentralized reinforcement learning (RL) framework for simultaneous dynamic routing and traffic signal control. EMVLight extends Dijkstra's algorithm to efficiently update the optimal route for the EMVs in real-time as it travels through the traffic network. The decentralized RL agents learn network-level cooperative traffic signal phase strategies that not only reduce EMV travel time but also reduce the average travel time of non-EMVs in the network. This benefit has been demonstrated through comprehensive experiments with synthetic and real-world maps. These experiments show that EMVLight outperforms benchmark transportation engineering techniques and existing RL-based signal control methods.

----

## [511] Constrained Prescriptive Trees via Column Generation

**Authors**: *Shivaram Subramanian, Wei Sun, Youssef Drissi, Markus Ettl*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20384](https://doi.org/10.1609/aaai.v36i4.20384)

**Abstract**:

With the abundance of available data, many enterprises seek to implement data-driven prescriptive analytics to help them make informed decisions. These prescriptive policies need to satisfy operational constraints, and proactively eliminate rule conflicts, both of which are ubiquitous in practice. It is also desirable for them to be simple and interpretable, so they can be easily verified and implemented. Existing approaches from the literature center around constructing variants of prescriptive decision trees to generate interpretable policies. However, none of the existing methods is able to handle constraints. In this paper, we propose a scalable method that solves the constrained prescriptive policy generation problem. We introduce a novel path-based mixed-integer program (MIP) formulation which identifies a (near) optimal policy efficiently via column generation. The policy generated can be represented as a multiway-split tree which is more interpretable and informative than binary-split trees due to its shorter rules. We demonstrate the efficacy of our method with extensive computational experiments on both synthetic and real datasets.

----

## [512] DDGCN: Dual Dynamic Graph Convolutional Networks for Rumor Detection on Social Media

**Authors**: *Mengzhu Sun, Xi Zhang, Jiaqi Zheng, Guixiang Ma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20385](https://doi.org/10.1609/aaai.v36i4.20385)

**Abstract**:

Detecting rumors on social media has become particular important due to the rapid dissemination and adverse impacts on our lives. Though a set of rumor detection models have exploited the message propagation structural or temporal information, they seldom model them altogether to enjoy the best of both worlds. Moreover, the dynamics of knowledge information associated with the comments are not involved, either. To this end, we propose a novel Dual-Dynamic Graph Convolutional Networks, termed as DDGCN, which can model the dynamics of messages in propagation as well as the dynamics of the background knowledge from Knowledge graphs in one unified framework. Specifically, two Graph Convolutional Networks are adopted to capture the above two types of structure information at different time stages, which are then combined with a temporal fusing unit. This allows for learning the dynamic event representations in a more fine-grained manner, and incrementally aggregating them to capture the cascading effect for better rumor detection. Extensive experiments on two public real-world datasets demonstrate that our proposal yields significant improvements compared to strong baselines and can detect rumors at early stages.

----

## [513] Contact-Distil: Boosting Low Homologous Protein Contact Map Prediction by Self-Supervised Distillation

**Authors**: *Qin Wang, Jiayang Chen, Yuzhe Zhou, Yu Li, Liangzhen Zheng, Sheng Wang, Zhen Li, Shuguang Cui*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20386](https://doi.org/10.1609/aaai.v36i4.20386)

**Abstract**:

Accurate protein contact map prediction (PCMP) is essential for precise protein structure estimation and further biological studies. Recent works achieve significant performance on this task with high quality multiple sequence alignment (MSA). However, the PCMP accuracy drops dramatically while only poor MSA (e.g., absolute MSA count less than 10) is available. Therefore, in this paper, we propose the Contact-Distil to improve the low homologous PCMP accuracy through knowledge distillation on a self-supervised model. Particularly, two pre-trained transformers are exploited to learn the high quality and low quality MSA representation in parallel for the teacher and student model correspondingly. Besides, the co-evolution information is further extracted from pure sequence through a pretrained ESM-1b model, which provides auxiliary knowledge to improve student performance. Extensive experiments show Contact-Distil outperforms previous state-of-the-arts by large margins on CAMEO-L dataset for low homologous PCMP, i.e., around 13.3% and 9.5% improvements against Alphafold2 and MSA Transformer respectively when MSA count less than 10.

----

## [514] EtinyNet: Extremely Tiny Network for TinyML

**Authors**: *Kunran Xu, Yishi Li, Huawei Zhang, Rui Lai, Lin Gu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20387](https://doi.org/10.1609/aaai.v36i4.20387)

**Abstract**:

There are many AI applications in high-income countries because their implementation depends on expensive GPU cards (~2000$) and reliable power supply (~200W). To deploy AI in resource-poor settings on cheaper (~20$) and low-power devices (<1W), key modifications are required to adapt neural networks for Tiny machine learning (TinyML). In this paper, for putting CNNs into storage limited devices, we developed efficient tiny models with only hundreds of KB parameters. Toward this end, we firstly design a parameter-efficient tiny architecture by introducing dense linear depthwise block. Then, a novel adaptive scale quantization (ASQ) method is proposed for further quantizing tiny models in aggressive low-bit while retaining the accuracy. With the optimized architecture and 4-bit ASQ, we present a family of ultralightweight networks, named EtinyNet, that achieves 57.0% ImageNet top-1 accuracy with an extremely tiny model size of 340KB. When deployed on an off-the-shelf commercial microcontroller for object detection tasks, EtinyNet achieves state-of-the-art 56.4% mAP on Pascal VOC. Furthermore, the experimental results on Xilinx compact FPGA indicate that EtinyNet achieves prominent low power of 620mW, about 5.6x lower than existing FPGA designs. The code and demo are in https://github.com/aztc/EtinyNet

----

## [515] RepBin: Constraint-Based Graph Representation Learning for Metagenomic Binning

**Authors**: *Hansheng Xue, Vijini Mallawaarachchi, Yujia Zhang, Vaibhav Rajan, Yu Lin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20388](https://doi.org/10.1609/aaai.v36i4.20388)

**Abstract**:

Mixed communities of organisms are found in many environments -- from the human gut to marine ecosystems -- and can have profound impact on human health and the environment. Metagenomics studies the genomic material of such communities through high-throughput sequencing that yields DNA subsequences for subsequent analysis. A fundamental problem in the standard workflow, called binning, is to discover clusters, of genomic subsequences, associated with the constituent organisms. Inherent noise in the subsequences, various biological constraints that need to be imposed on them and the skewed cluster size distribution exacerbate the difficulty of this unsupervised learning problem. In this paper, we present a new formulation using a graph where the nodes are subsequences and edges represent homophily information. In addition, we model biological constraints providing heterophilous signal about nodes that cannot be clustered together. We solve the binning problem by developing new algorithms for (i) graph representation learning that preserves both homophily relations and heterophily constraints (ii) constraint-based graph clustering method that addresses the problems of skewed cluster size distribution. Extensive experiments, on real and synthetic datasets, demonstrate that our approach, called RepBin, outperforms a wide variety of competing methods. Our constraint-based graph representation learning and clustering methods, that may be useful in other domains as well, advance the state-of-the-art in both metagenomics binning and graph representation learning.

----

## [516] NSGZero: Efficiently Learning Non-exploitable Policy in Large-Scale Network Security Games with Neural Monte Carlo Tree Search

**Authors**: *Wanqi Xue, Bo An, Chai Kiat Yeo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20389](https://doi.org/10.1609/aaai.v36i4.20389)

**Abstract**:

How resources are deployed to secure critical targets in networks can be modelled by Network Security Games (NSGs). While recent advances in deep learning (DL) provide a powerful approach to dealing with large-scale NSGs, DL methods such as NSG-NFSP suffer from the problem of data inefficiency. Furthermore, due to centralized control, they cannot scale to scenarios with a large number of resources. In this paper, we propose a novel DL-based method, NSGZero, to learn a non-exploitable policy in NSGs. NSGZero improves data efficiency by performing planning with neural Monte Carlo Tree Search (MCTS). Our main contributions are threefold. First, we design deep neural networks (DNNs) to perform neural MCTS in NSGs. Second, we enable neural MCTS with decentralized control, making NSGZero applicable to NSGs with many resources. Third, we provide an efficient learning paradigm, to achieve joint training of the DNNs in NSGZero. Compared to state-of-the-art algorithms, our method achieves significantly better data efficiency and scalability.

----

## [517] RID-Noise: Towards Robust Inverse Design under Noisy Environments

**Authors**: *Jia-Qi Yang, Ke-Bin Fan, Hao Ma, De-Chuan Zhan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20390](https://doi.org/10.1609/aaai.v36i4.20390)

**Abstract**:

From an engineering perspective, a design should not only perform well in an ideal condition, but should also resist noises. Such a design methodology, namely robust design, has been widely implemented in the industry for product quality control. However, classic robust design requires a lot of evaluations for a single design target, while the results of these evaluations could not be reused for a new target. To achieve data-efficient robust design, we propose Robust Inverse Design under Noise (RID-Noise), which can utilize existing data to train a conditional invertible neural network. Specifically, we estimate the robustness of a design parameter by its predictability, measured by the prediction error of a forward neural network. We also define a sample-wise weight, which can be used in the maximum weighted likelihood estimation of an inverse model based on a conditional invertible neural network. With the visual results from experiments, we clearly justify how RID-Noise works by learning the distribution and robustness from data. Further experiments on several real-world benchmark tasks with noises confirm that our method is more effective than other state-of-the-art inverse design methods. Code and supplementary is publicly available at https://github.com/ThyrixYang/rid-noise-aaai22

----

## [518] Deepfake Network Architecture Attribution

**Authors**: *Tianyun Yang, Ziyao Huang, Juan Cao, Lei Li, Xirong Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20391](https://doi.org/10.1609/aaai.v36i4.20391)

**Abstract**:

With the rapid progress of generation technology, it has become necessary to attribute the origin of fake images. Existing works on fake image attribution perform multi-class classification on several Generative Adversarial Network (GAN) models and obtain high accuracies. While encouraging, these works are restricted to model-level attribution, only capable of handling images generated by seen models with a specific seed, loss and dataset, which is limited in real-world scenarios when fake images may be generated by privately trained models. This motivates us to ask whether it is possible to attribute fake images to the source models' architectures even if they are finetuned or retrained under different configurations. In this work, we present the first study on Deepfake Network Architecture Attribution to attribute fake images on architecture-level. Based on an observation that GAN architecture is likely to leave globally consistent fingerprints while traces left by model weights vary in different regions, we provide a simple yet effective solution named by DNA-Det for this problem. Extensive experiments on multiple cross-test setups and a large-scale dataset demonstrate the effectiveness of DNA-Det.

----

## [519] ZINB-Based Graph Embedding Autoencoder for Single-Cell RNA-Seq Interpretations

**Authors**: *Zhuohan Yu, Yifu Lu, Yunhe Wang, Fan Tang, Ka-Chun Wong, Xiangtao Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20392](https://doi.org/10.1609/aaai.v36i4.20392)

**Abstract**:

Single-cell RNA sequencing (scRNA-seq) provides high-throughput information about the genome-wide gene expression levels at the single-cell resolution, bringing a precise understanding on the transcriptome of individual cells. Unfortunately, the rapidly growing scRNA-seq data and the prevalence of dropout events pose substantial challenges for cell type annotation. Here, we propose a single-cell model-based deep graph embedding clustering (scTAG) method, which simultaneously learns cell–cell topology representations and identifies cell clusters based on deep graph convolutional network. scTAG integrates the zero-inflated negative binomial (ZINB) model into a topology adaptive graph convolutional autoencoder to learn the low-dimensional latent representation and adopts Kullback–Leibler (KL) divergence for the clustering tasks. By simultaneously optimizing the clustering loss, ZINB loss, and the cell graph reconstruction loss, scTAG jointly optimizes cluster label assignment and feature learning with the topological structures preserved in an end-to-end manner. Extensive experiments on 16 single-cell RNA-seq datasets from diverse yet representative single-cell sequencing platforms demonstrate the superiority of scTAG over various state-of-the-art clustering methods.

----

## [520] DeepThermal: Combustion Optimization for Thermal Power Generating Units Using Offline Reinforcement Learning

**Authors**: *Xianyuan Zhan, Haoran Xu, Yue Zhang, Xiangyu Zhu, Honglei Yin, Yu Zheng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20393](https://doi.org/10.1609/aaai.v36i4.20393)

**Abstract**:

Optimizing the combustion efficiency of a thermal power generating unit (TPGU) is a highly challenging and critical task in the energy industry. We develop a new data-driven AI system, namely DeepThermal, to optimize the combustion control strategy for TPGUs. At its core, is a new model-based offline reinforcement learning (RL) framework, called MORE, which leverages historical operational data of a TGPU to solve a highly complex constrained Markov decision process problem via purely offline training. In DeepThermal, we first learn a data-driven combustion process simulator from the offline dataset. The RL agent of MORE is then trained by combining real historical data as well as carefully filtered and processed simulation data through a novel restrictive exploration scheme. DeepThermal has been successfully deployed in four large coal-fired thermal power plants in China. Real-world experiments show that DeepThermal effectively improves the combustion efficiency of TPGUs. We also report the superior performance of MORE by comparing with the state-of-the-art algorithms on the standard offline RL benchmarks.

----

## [521] AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via End-to-End Reinforcement Learning

**Authors**: *Enmin Zhao, Renye Yan, Jinqiu Li, Kai Li, Junliang Xing*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20394](https://doi.org/10.1609/aaai.v36i4.20394)

**Abstract**:

Heads-up no-limit Texas hold’em (HUNL) is the quintessential game with imperfect information. Representative priorworks like DeepStack and Libratus heavily rely on counter-factual regret minimization (CFR) and its variants to tackleHUNL. However, the prohibitive computation cost of CFRiteration makes it difficult for subsequent researchers to learnthe CFR model in HUNL and apply it in other practical applications. In this work, we present AlphaHoldem, a high-performance and lightweight HUNL AI obtained with an end-to-end self-play reinforcement learning framework. The proposed framework adopts a pseudo-siamese architecture to directly learn from the input state information to the output actions by competing the learned model with its different historical versions. The main technical contributions include anovel state representation of card and betting information, amultitask self-play training loss function, and a new modelevaluation and selection metric to generate the final model.In a study involving 100,000 hands of poker, AlphaHoldemdefeats Slumbot and DeepStack using only one PC with threedays training. At the same time, AlphaHoldem only takes 2.9milliseconds for each decision-making using only a singleGPU, more than 1,000 times faster than DeepStack. We release the history data among among AlphaHoldem, Slumbot,and top human professionals in the author’s GitHub repository to facilitate further studies in this direction.

----

## [522] Hierarchical Multi-Supervision Multi-Interaction Graph Attention Network for Multi-Camera Pedestrian Trajectory Prediction

**Authors**: *Guoliang Zhao, Yuxun Zhou, Zhanbo Xu, Yadong Zhou, Jiang Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20395](https://doi.org/10.1609/aaai.v36i4.20395)

**Abstract**:

Pedestrian trajectory prediction has become an essential underpinning in various human-centric applications including but not limited to autonomous vehicles, intelligent surveillance system and social robotics. Previous research endeavors mainly focus on single camera trajectory prediction (SCTP), while the problem of multi-camera trajectory prediction (MCTP) is often overly simplified into predicting presence in the next camera. This paper addresses MCTP from a more realistic yet challenging perspective, by redefining the task as a joint estimation of both future destination and possible trajectory. As such, two major efforts are devoted to facilitating related research and advancing modeling techniques. Firstly, we establish a comprehensive multi-camera Scenes Pedestrian Trajectory Dataset (mcScenes), which is collected from a real-world multi-camera space combined with thorough human interaction annotations and carefully designed evaluation metrics. Secondly, we propose a novel joint prediction framework, namely HM3GAT, for the MCTP task by building a tailored network architecture. The core idea behind HM3GAT is a fusion of topological and trajectory information that are mutually beneficial to the prediction of each task, achieved by deeply customized networks. The proposed framework is comprehensively evaluated on the mcScenes dataset with multiple ablation experiments. Status-of-the-art SCTP models are adopted as baselines to further validate the advantages of our method in terms of both information fusion and technical improvement. The mcScenes dataset, the HM3GAT, and alternative models are made publicly available for interested readers.

----

## [523] 6DCNN with Roto-Translational Convolution Filters for Volumetric Data Processing

**Authors**: *Dmitrii Zhemchuzhnikov, Ilia Igashov, Sergei Grudinin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20396](https://doi.org/10.1609/aaai.v36i4.20396)

**Abstract**:

In this work, we introduce 6D Convolutional Neural Network (6DCNN) designed to tackle the problem of detecting relative positions and orientations of local patterns when processing three-dimensional volumetric data. 6DCNN also includes SE(3)-equivariant message-passing and nonlinear activation operations constructed in the Fourier space. Working in the Fourier space allows significantly reducing the computational complexity of our operations. We demonstrate the properties of the 6D convolution and its efficiency in the recognition of spatial patterns. We also assess the 6DCNN model on several datasets from the recent CASP protein structure prediction challenges. Here, 6DCNN improves over the baseline architecture and also outperforms the state of the art.

----

## [524] Deeply Tensor Compressed Transformers for End-to-End Object Detection

**Authors**: *Peining Zhen, Ziyang Gao, Tianshu Hou, Yuan Cheng, Hai-Bao Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20397](https://doi.org/10.1609/aaai.v36i4.20397)

**Abstract**:

DEtection TRansformer (DETR) is a recently proposed method that streamlines the detection pipeline and achieves competitive results against two-stage detectors such as Faster-RCNN. The DETR models get rid of complex anchor generation and post-processing procedures thereby making the detection pipeline more intuitive. However, the numerous redundant parameters in transformers make the DETR models computation and storage intensive, which seriously hinder them to be deployed on the resources-constrained devices. In this paper, to obtain a compact end-to-end detection framework, we propose to deeply compress the transformers with low-rank tensor decomposition. The basic idea of the tensor-based compression is to represent the large-scale weight matrix in one network layer with a chain of low-order matrices. Furthermore, we propose a gated multi-head attention (GMHA) module to mitigate the accuracy drop of the tensor-compressed DETR models. In GMHA, each attention head has an independent gate to determine the passed attention value. The redundant attention information can be suppressed by adopting the normalized gates. Lastly, to obtain fully compressed DETR models, a low-bitwidth quantization technique is introduced for further reducing the model storage size. Based on the proposed methods, we can achieve significant parameter and model size reduction while maintaining high detection performance. We conduct extensive experiments on the COCO dataset to validate the effectiveness of our tensor-compressed (tensorized) DETR models. The experimental results show that we can attain 3.7 times full model compression with 482 times feed forward network (FFN) parameter reduction and only 0.6 points accuracy drop.

----

## [525] Dynamic Manifold Learning for Land Deformation Forecasting

**Authors**: *Fan Zhou, Rongfan Li, Qiang Gao, Goce Trajcevski, Kunpeng Zhang, Ting Zhong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20398](https://doi.org/10.1609/aaai.v36i4.20398)

**Abstract**:

Landslides refer to occurrences of massive ground movements due to geological (and meteorological) factors, and can have disastrous impact on property, economy, and even lead to loss of life. The advances of remote sensing provide accurate and continuous terrain monitoring, enabling the study and analysis of land deformation which, in turn, can be used for possible landslides forecast. Prior studies either rely on independent observations for displacement prediction or model static land characteristics without considering the subtle interactions between different locations and the dynamic changes of the surface conditions. We present DyLand -- Dynamic Manifold Learning with Normalizing Flows for Land deformation prediction -- a novel framework for learning dynamic structures of terrain surface and improving the performance of land deformation prediction. DyLand models the spatial connections of InSAR measurements and estimates conditional distributions of deformations on the terrain manifold with a novel normalizing flow-based method. Instead of modeling the stable terrains, it incorporates surface permutations and captures the innate dynamics of the land surface while allowing for tractable likelihood estimates on the manifold. Our extensive evaluations on curated InSAR datasets from continuous monitoring of slopes prone to landslides show that DyLand outperforms existing bechmarking models.

----

## [526] Fully Adaptive Framework: Neural Computerized Adaptive Testing for Online Education

**Authors**: *Yan Zhuang, Qi Liu, Zhenya Huang, Zhi Li, Shuanghong Shen, Haiping Ma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i4.20399](https://doi.org/10.1609/aaai.v36i4.20399)

**Abstract**:

Computerized Adaptive Testing (CAT) refers to an efficient and personalized test mode in online education, aiming to accurately measure student proficiency level on the required subject/domain. The key component of CAT is the "adaptive" question selection algorithm, which automatically selects the best suited question for student based on his/her current estimated proficiency, reducing test length. Existing algorithms rely on some manually designed and pre-fixed informativeness/uncertainty metrics of question for selections, which is labor-intensive and not sufficient for capturing complex relations between students and questions. In this paper, we propose a fully adaptive framework named Neural Computerized Adaptive Testing (NCAT), which formally redefines CAT as a reinforcement learning problem and directly learns selection algorithm from real-world data. Specifically, a bilevel optimization is defined and simplified under CAT's application scenarios to make the algorithm learnable. Furthermore, to address the CAT task effectively, we tackle it as an equivalent reinforcement learning problem and propose an attentive neural policy to model complex non-linear interactions. Extensive experiments on real-world datasets demonstrate the effectiveness and robustness of NCAT compared with several state-of-the-art methods.

----

## [527] An Algorithmic Introduction to Savings Circles

**Authors**: *Rediet Abebe, Adam Eck, Christian Ikeokwu, Sam Taggart*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20400](https://doi.org/10.1609/aaai.v36i5.20400)

**Abstract**:

Rotating savings and credit associations (roscas) are informal financial organizations common in settings where communities have reduced access to formal financial institutions. In a rosca, a fixed group of participants regularly contribute sums of money to a pot. This pot is then allocated periodically using lottery, aftermarket, or auction mechanisms. Roscas are empirically well-studied in economics. They are, however, challenging to study theoretically due to their dynamic nature. Typical economic analyses of roscas stop at coarse ordinal welfare comparisons to other credit allocation mechanisms, leaving much of roscas' ubiquity unexplained. In this work, we take an algorithmic perspective on the study of roscas. Building on techniques from the price of anarchy literature, we present worst-case welfare approximation guarantees. We further experimentally compare the welfare of outcomes as key features of the environment vary. These cardinal welfare analyses further rationalize the prevalence of roscas. We conclude by discussing several other promising avenues.

----

## [528] Locally Fair Partitioning

**Authors**: *Pankaj K. Agarwal, Shao-Heng Ko, Kamesh Munagala, Erin Taylor*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20401](https://doi.org/10.1609/aaai.v36i5.20401)

**Abstract**:

We model the societal task of redistricting political districts as a partitioning problem: Given a set of n points in the plane, each belonging to one of two parties, and a parameter k, our goal is to compute a partition P of the plane into regions so that each region contains roughly  s = n/k points. P should satisfy a notion of  "local" fairness, which is related to the notion of core, a well-studied concept in cooperative game theory. A region is associated with the majority party in that region, and a point is unhappy in P if it belongs to the minority party. A group D of roughly s contiguous points is called a deviating group with respect to P if majority of points in D are unhappy in P. The partition P is locally fair if there is no deviating group with respect to P.

This paper focuses on a restricted case when points lie in 1D. The problem is non-trivial even in this case. We consider both adversarial and "beyond worst-case" settings for this problem. For the former, we characterize the input parameters for which a locally fair partition always exists; we also show that a locally fair partition may not exist for certain parameters. We then consider input models where there are "runs" of red and blue points. For such clustered inputs, we show that a locally fair partition may not exist for certain values of s, but an approximate locally fair partition exists if we allow some regions to have smaller sizes. We finally present a polynomial-time algorithm for computing a locally fair partition if one exists.

----

## [529] Maximizing Nash Social Welfare in 2-Value Instances

**Authors**: *Hannaneh Akrami, Bhaskar Ray Chaudhury, Martin Hoefer, Kurt Mehlhorn, Marco Schmalhofer, Golnoosh Shahkarami, Giovanna Varricchio, Quentin Vermande, Ernest van Wijland*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20402](https://doi.org/10.1609/aaai.v36i5.20402)

**Abstract**:

We consider the problem of maximizing the Nash social welfare when allocating a set G of indivisible goods to a set N of agents. We study instances, in which all agents have 2-value additive valuations: The value of every agent for every good is either p or q, where p and q are integers and p2. 
 
In terms of approximation, we present positive and negative results for general p and q. We show that our algorithm obtains an approximation ratio of at most 1.0345. Moreover, we prove that the problem is APX-hard, with a lower bound of 1.000015 achieved at p/q = 4/5.

----

## [530] Truth-Tracking via Approval Voting: Size Matters

**Authors**: *Tahar Allouche, Jérôme Lang, Florian Yger*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20403](https://doi.org/10.1609/aaai.v36i5.20403)

**Abstract**:

Epistemic social choice aims at unveiling a hidden ground truth given votes, which are interpreted as noisy signals about it. We consider here a simple setting where votes consist of approval ballots: each voter approves a set of alternatives which they believe can possibly be the ground truth. Based on the intuitive idea that more reliable votes contain fewer alternatives, we define several noise models that are approval voting variants of the Mallows model. The likelihood-maximizing alternative is then characterized as the winner of a weighted approval rule, where the weight of a ballot decreases with its cardinality. We have conducted an experiment on three image annotation datasets; they conclude that rules based on our noise model outperform standard approval voting; the best performance is obtained by a variant of the Condorcet noise model.

----

## [531] Dimensionality and Coordination in Voting: The Distortion of STV

**Authors**: *Ioannis Anagnostides, Dimitris Fotakis, Panagiotis Patsilinakos*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20404](https://doi.org/10.1609/aaai.v36i5.20404)

**Abstract**:

We study the performance of voting mechanisms from a utilitarian standpoint, under the recently introduced framework of metric-distortion, offering new insights along two main lines. First, if d represents the doubling dimension of the metric space, we show that the distortion of STV is O(d log log m), where m represents the number of candidates. For doubling metrics this implies an exponential improvement over the lower bound for general metrics, and as a special case it effectively answers a question left open by Skowron and Elkind (AAAI '17) regarding the distortion of STV under low-dimensional Euclidean spaces. More broadly, this constitutes the first nexus between the performance of any voting rule and the ``intrinsic dimensionality'' of the underlying metric space. We also establish a nearly-matching lower bound, refining the construction of Skowron and Elkind. Moreover, motivated by the efficiency of STV, we investigate whether natural learning rules can lead to low-distortion outcomes. Specifically, we introduce simple, deterministic and decentralized exploration/exploitation dynamics, and we show that they converge to a candidate with O(1) distortion.

----

## [532] Fair and Truthful Giveaway Lotteries

**Authors**: *Tal Arbiv, Yonatan Aumann*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20405](https://doi.org/10.1609/aaai.v36i5.20405)

**Abstract**:

We consider a setting where a large number of agents are all interested in attending some public resource of limited capacity. Attendance is thus allotted by lottery. If agents arrive individually, then randomly choosing the agents – one by one - is a natural, fair and efficient solution. We consider the case where agents are organized in groups (e.g. families, friends), the members of each of which must all be admitted together. We study the question of how best to design such lotteries. We first establish the desired properties of such lotteries, in terms of fairness and efficiency, and define the appropriate notions of strategy proofness (providing that agents cannot gain by misrepresenting the true groups, e.g. joining or splitting groups). We establish inter-relationships between the different properties, proving properties that cannot be fulfilled simultaneously (e.g. leximin optimality and strong group stratagy proofness). Our main contribution is a polynomial mechanism for the problem, which guarantees many of the desired properties, including: leximin optimality, Pareto-optimality, anonymity, group strategy proofness, and adjunctive strategy proofness (which provides that no benefit can be obtained by registering additional - uninterested or bogus - individuals). The mechanism approximates the utilitarian optimum to within a factor of 2, which, we prove, is optimal for any mechanism that guarantees any one of the following properties: egalitarian welfare optimality, leximin optimality, envyfreeness, and adjunctive strategy proofness.

----

## [533] Universal and Tight Online Algorithms for Generalized-Mean Welfare

**Authors**: *Siddharth Barman, Arindam Khan, Arnab Maiti*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20406](https://doi.org/10.1609/aaai.v36i5.20406)

**Abstract**:

We study fair and efficient allocation of divisible goods, in an online manner, among n agents. The goods arrive online in a sequence of T time periods. The agents' values for a good are revealed only after its arrival, and the online algorithm needs to fractionally allocate the good, immediately and irrevocably, among the agents. Towards a unifying treatment of fairness and economic efficiency objectives, we develop an algorithmic framework for finding online allocations to maximize the generalized mean of the values received by the agents. In particular, working with the assumption that each agent's value for the grand bundle of goods is appropriately scaled, we address online maximization of p-mean welfare. Parameterized by an exponent term p in (-infty, 1], these means encapsulate a range of welfare functions, including social welfare (p=1), egalitarian welfare (p to -infty), and Nash social welfare (p to 0).

We present a simple algorithmic template that takes a threshold as input and, with judicious choices for this threshold, leads to both universal and tailored competitive guarantees. First, we show that one can compute online a single allocation that O (sqrt(n) log n)-approximates the optimal p-mean welfare for all p <= 1. The existence of such a universal allocation is interesting in and of itself. Moreover, this universal guarantee achieves essentially tight competitive ratios for specific values of p. 

Next, we obtain improved competitive ratios for different ranges of p by executing our algorithm with p-specific thresholds, e.g., we provide O(log^3 n)-competitive ratio for all p in (-1/(log 2n),1). 

We complement our positive results by establishing lower bounds to show that our guarantees are essentially tight for a wide range of the exponent parameter.

----

## [534] Truthful and Fair Mechanisms for Matroid-Rank Valuations

**Authors**: *Siddharth Barman, Paritosh Verma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20407](https://doi.org/10.1609/aaai.v36i5.20407)

**Abstract**:

We study the problem of allocating indivisible goods among strategic agents. We focus on settings wherein monetary transfers are not available and each agent's private valuation is a submodular function with binary marginals, i.e., the agents' valuations are matroid-rank functions. In this setup, we establish a notable dichotomy between two of the most well-studied fairness notions in discrete fair division; specifically, between envy-freeness up to one good (EF1) and maximin shares (MMS).   

First, we show that a known Pareto-efficient mechanism is group strategy-proof for finding EF1 allocations, under matroid-rank valuations. The group strategy-proofness guarantee strengthens an existing result that establishes truthfulness (individually for each agent) in the same context. Our result also generalizes prior work from binary additive valuations to the matroid-rank case.  

Next, we establish that an analogous positive result cannot be achieved for MMS, even when considering truthfulness on an individual level. Specifically, we prove that, for matroid-rank valuations, there does not exist a truthful mechanism that is index oblivious, Pareto efficient, and maximin fair.   

For establishing our results, we develop a characterization of truthful mechanisms for matroid-rank functions. This characterization in fact holds for a broader class of valuations (specifically, holds for binary XOS functions) and might be of independent interest.

----

## [535] Truthful Cake Sharing

**Authors**: *Xiaohui Bei, Xinhang Lu, Warut Suksompong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20408](https://doi.org/10.1609/aaai.v36i5.20408)

**Abstract**:

The classic cake cutting problem concerns the fair allocation of a heterogeneous resource among interested agents. In this paper, we study a public goods variant of the problem, where instead of competing with one another for the cake, the agents all share the same subset of the cake which must be chosen subject to a length constraint. We focus on the design of truthful and fair mechanisms in the presence of strategic agents who have piecewise uniform utilities over the cake. On the one hand, we show that the leximin solution is truthful and moreover maximizes an egalitarian welfare measure among all truthful and position oblivious mechanisms. On the other hand, we demonstrate that the maximum Nash welfare solution is truthful for two agents but not in general. Our results assume that mechanisms can block each agent from accessing parts that the agent does not claim to desire; we provide an impossibility result when blocking is not allowed.

----

## [536] The Secretary Problem with Competing Employers on Random Edge Arrivals

**Authors**: *Xiaohui Bei, Shengyu Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20409](https://doi.org/10.1609/aaai.v36i5.20409)

**Abstract**:

The classic secretary problem concerns the problem of an employer facing a random sequence of candidates and making online hiring decisions to try to hire the best candidate. In this paper, we study a game-theoretic generalization of the secretary problem where a set of employers compete with each other to hire the best candidate. Different from previous secretary market models, our model assumes that the sequence of candidates arriving at each employer is uniformly random but independent from other sequences. We consider two versions of this secretary game where employers can have adaptive or non-adaptive strategies, and provide characterizations of the best response and Nash equilibrium of each game.

----

## [537] Almost Full EFX Exists for Four Agents

**Authors**: *Ben Berger, Avi Cohen, Michal Feldman, Amos Fiat*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20410](https://doi.org/10.1609/aaai.v36i5.20410)

**Abstract**:

The existence of EFX allocations of goods is a major open problem in fair division, even for additive valuations. The current state of the art is that no setting where EFX allocations are impossible is known, and yet, existence results are known only for very restricted settings, such as: (i) agents with identical valuations, (ii) 2 agents, and (iii) 3 agents with additive valuations. It is also known that EFX exists if one can leave n-1 items unallocated, where n is the number of agents.
 
 We develop new techniques that allow us to push the boundaries of the enigmatic EFX problem beyond these known results, and (arguably) to simplify proofs of earlier results. Our main result is that every setting with 4 additive agents admits an EFX allocation that leaves at most a single item unallocated. Beyond our main result, we introduce a new class of valuations, termed nice cancelable, which includes additive, unit-demand, budget-additive and multiplicative valuations, among others. Using our new techniques, we show that both our results and previous results for additive valuations extend to nice cancelable valuations.

----

## [538] Sequential Blocked Matching

**Authors**: *Nicholas Bishop, Hau Chan, Debmalya Mandal, Long Tran-Thanh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20411](https://doi.org/10.1609/aaai.v36i5.20411)

**Abstract**:

We consider a sequential blocked matching (SBM) model where strategic agents repeatedly report ordinal preferences over a set of services to a central planner. The planner's goal is to elicit agents' true preferences and design a policy that matches services to agents in order to maximize the expected social welfare with the added constraint that  each matched service can be blocked or unavailable for a number of time periods. Naturally, SBM models the repeated allocation of reusable services to a set of agents where each allocated service becomes unavailable for a fixed duration. 

We first consider the offline SBM setting, where the strategic agents are aware of their true preferences. We measure the performance of any policy by distortion, the worst-case multiplicative approximation guaranteed by any policy. For the setting with s services, we establish lower bounds of Ω(s) and Ω(√s) on the distortions of any deterministic and randomised mechanisms, respectively. We complement these results by providing approximately truthful, measured by incentive ratio, deterministic and randomised policies based on random serial dictatorship which match our lower bounds. Our results show that there is a significant improvement if one considers the class of randomised policies.  Finally, we consider the online SBM setting with bandit feedback where each agent is initially unaware of her true preferences, and the planner must facilitate each agent in the learning of their preferences through the matching of services over time. We design an approximately truthful mechanism based on the explore-then-commit paradigm, which achieves logarithmic dynamic approximate regret.

----

## [539] Combating Collusion Rings Is Hard but Possible

**Authors**: *Niclas Boehmer, Robert Bredereck, André Nichterlein*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20412](https://doi.org/10.1609/aaai.v36i5.20412)

**Abstract**:

A recent report of Littmann published in the Communications of the ACM outlines the existence and the fatal impact of collusion rings in academic peer reviewing. We introduce and analyze the problem Cycle-Free Reviewing that aims at finding a review assignment without the following kind of collusion ring: A sequence of reviewers each reviewing a paper authored by the next reviewer in the sequence (with the last reviewer reviewing a paper of the first), thus creating a review cycle where each reviewer gives favorable reviews. As a result, all papers in that cycle have a high chance of acceptance independent of their respective scientific merit.

We observe that review assignments computed using a standard Linear Programming approach typically admit many short review cycles. On the negative side, we show that Cycle-Free Reviewing is NP-hard in various restricted cases (i.e., when every author is qualified to review all papers and one wants to prevent that authors review each other's or their own papers or when every author has only one paper and is only qualified to review few papers). On the positive side, among others, we show that, in some realistic settings, an assignment without any review cycles of small length always exists. This result also gives rise to an efficient heuristic for computing (weighted) cycle-free review assignments, which we show to be of excellent quality in practice.

----

## [540] Theory of and Experiments on Minimally Invasive Stability Preservation in Changing Two-Sided Matching Markets

**Authors**: *Niclas Boehmer, Klaus Heeger, Rolf Niedermeier*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20413](https://doi.org/10.1609/aaai.v36i5.20413)

**Abstract**:

Following up on purely theoretical work, we contribute further theoretical insights into adapting stable two-sided matchings to change. Moreover, we perform extensive empirical studies hinting at numerous practically useful properties. Our theoretical extensions include the study of new problems (that is, incremental variants of Almost Stable Marriage and Hospital Residents), focusing on their (parameterized) computational complexity and the equivalence of various change types (thus simplifying algorithmic and complexity-theoretic studies for various natural change scenarios). Our experimental findings reveal, for instance, that allowing the new matching to be blocked by a few pairs significantly decreases the difference between the old and the new matching.

----

## [541] A Calculus for Computing Structured Justifications for Election Outcomes

**Authors**: *Arthur Boixel, Ulle Endriss, Ronald de Haan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20414](https://doi.org/10.1609/aaai.v36i5.20414)

**Abstract**:

In the context of social choice theory, we develop a tableau-based calculus for reasoning about voting rules. This calculus can be used to obtain structured explanations for why a given set of axioms justifies a given election outcome for a given profile of voter preferences. We then show how to operationalise this calculus, using a combination of SAT solving and answer set programming, to arrive at a flexible framework for presenting human-readable justifications to users.

----

## [542] Single-Agent Dynamics in Additively Separable Hedonic Games

**Authors**: *Felix Brandt, Martin Bullinger, Leo Tappe*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20415](https://doi.org/10.1609/aaai.v36i5.20415)

**Abstract**:

The formation of stable coalitions is a central concern in multiagent systems. A considerable stream of research defines stability via the absence of beneficial deviations by single agents. Such deviations require an agent to improve her utility by joining another coalition while possibly imposing further restrictions on the consent of the agents in the welcoming as well as the abandoned coalition. While most of the literature focuses on unanimous consent, we also study consent decided by majority vote, and introduce two new stability notions that can be seen as local variants of popularity. We investigate these notions in additively separable hedonic games by pinpointing boundaries to computational complexity depending on the type of consent and restrictions on the utility functions. The latter restrictions shed new light on well-studied classes of games based on the appreciation of friends or the aversion to enemies. Many of our positive results follow from the Deviation Lemma, a general combinatorial observation, which can be leveraged to prove the convergence of simple and natural single-agent dynamics under fairly general conditions.

----

## [543] On Improving Resource Allocations by Sharing

**Authors**: *Robert Bredereck, Andrzej Kaczmarczyk, Junjie Luo, Rolf Niedermeier, Florian Sachse*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20416](https://doi.org/10.1609/aaai.v36i5.20416)

**Abstract**:

Given an initial resource allocation, where some agents may envy others or where a different distribution of resources might lead to higher social welfare, our goal is to improve the allocation without reassigning resources. We consider a sharing concept allowing resources being shared with social network neighbors of the resource owners. To this end, we introduce a formal model that allows a central authority to compute an optimal sharing between neighbors based on an initial allocation. Advocating this point of view, we focus on the most basic scenario where a resource may be shared by two neighbors in a social network and each agent can participate in a bounded number of sharings. We present algorithms for optimizing utilitarian and egalitarian social welfare of allocations and for reducing the number of envious agents. In particular, we examine the computational complexity with respect to several natural parameters. Furthermore, we study cases with restricted social network structures and, among others, devise polynomial-time algorithms in path- and tree-like (hierarchical) social networks.

----

## [544] Liquid Democracy with Ranked Delegations

**Authors**: *Markus Brill, Théo Delemazure, Anne-Marie George, Martin Lackner, Ulrike Schmidt-Kraepelin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20417](https://doi.org/10.1609/aaai.v36i5.20417)

**Abstract**:

Liquid democracy is a novel paradigm for collective decision-making that gives agents the choice between casting a direct vote or delegating their vote to another agent. We consider a generalization of the standard liquid democracy setting by allowing agents to specify multiple potential delegates, together with a preference ranking among them. This generalization increases the number of possible delegation paths and enables higher participation rates because fewer votes are lost due to delegation cycles or abstaining agents. In order to implement this generalization of liquid democracy, we need to find a principled way of choosing between multiple delegation paths. In this paper, we provide a thorough axiomatic analysis of the space of delegation rules, i.e., functions assigning a feasible delegation path to each delegating agent. In particular, we prove axiomatic characterizations as well as an impossibility result for delegation rules. We also analyze requirements on delegation rules that have been suggested by practitioners, and introduce novel rules with attractive properties. By performing an extensive experimental analysis on synthetic as well as real-world data, we compare delegation rules with respect to several quantitative criteria relating to the chosen paths and the resulting distribution of voting power. Our experiments reveal that delegation rules can be aligned on a spectrum reflecting an inherent trade-off between competing objectives.

----

## [545] Individual Representation in Approval-Based Committee Voting

**Authors**: *Markus Brill, Jonas Israel, Evi Micha, Jannik Peters*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20418](https://doi.org/10.1609/aaai.v36i5.20418)

**Abstract**:

When selecting multiple candidates based on approval preferences of agents, the proportional representation of agents' opinions is an important and well-studied desideratum. Existing criteria for evaluating the representativeness of outcomes focus on groups of agents and demand that sufficiently large and cohesive groups are "represented" in the sense that candidates approved by some group members are selected. Crucially, these criteria say nothing about the representation of individual agents, even if these agents are members of groups that deserve representation. In this paper, we formalize the concept of individual representation (IR) and explore to which extent, and under which circumstances, it can be achieved. We show that checking whether an IR outcome exists is computationally intractable, and we verify that all common approval-based voting rules may fail to provide IR even in cases where this is possible. We then focus on domain restrictions and establish an interesting contrast between "voter interval" and "candidate interval" preferences. This contrast can also be observed in our experimental results, where we analyze the attainability of IR for realistic preference profiles.

----

## [546] The Metric Distortion of Multiwinner Voting

**Authors**: *Ioannis Caragiannis, Nisarg Shah, Alexandros A. Voudouris*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20419](https://doi.org/10.1609/aaai.v36i5.20419)

**Abstract**:

We extend the recently introduced framework of metric distortion to multiwinner voting. In this framework, n agents and m alternatives are located in an underlying metric space. The exact distances between agents and alternatives are unknown. Instead, each agent provides a ranking of the alternatives, ordered from the closest to the farthest. Typically, the goal is to select a single alternative that approximately minimizes the total distance from the agents, and the worst-case approximation ratio is termed distortion. In the case of multiwinner voting, the goal is to select a committee of k alternatives that (approximately) minimizes the total cost to all agents. We consider the scenario where the cost of an agent for a committee is her distance from the q-th closest alternative in the committee. We reveal a surprising trichotomy on the distortion of multiwinner voting rules in terms of k and q: The distortion is unbounded when q <= k/3, asymptotically linear in the number of agents when k/3 < q <= k/2, and constant when q > k/2.

----

## [547] A Little Charity Guarantees Fair Connected Graph Partitioning

**Authors**: *Ioannis Caragiannis, Evi Micha, Nisarg Shah*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20420](https://doi.org/10.1609/aaai.v36i5.20420)

**Abstract**:

Motivated by fair division applications, we study a fair connected graph partitioning problem, in which an undirected graph with m nodes must be divided between n agents such that each agent receives a connected subgraph and the partition is fair. We study approximate versions of two fairness criteria: \alpha-proportionality requires that each agent receive a subgraph with at least (1/\alpha)*m/n nodes, and \alpha-balancedness requires that the ratio between the sizes of the largest and smallest subgraphs be at most \alpha. Unfortunately, there exist simple examples in which no partition is reasonably proportional or balanced. To circumvent this, we introduce the idea of charity. We show that by "donating" just n-1 nodes, we can guarantee the existence of 2-proportional and almost 2-balanced partitions (and find them in polynomial time), and that this result is almost tight. More generally, we chart the tradeoff between the size of charity and the approximation of proportionality or balancedness we can guarantee.

----

## [548] Truthful Aggregation of Budget Proposals with Proportionality Guarantees

**Authors**: *Ioannis Caragiannis, George Christodoulou, Nicos Protopapas*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20421](https://doi.org/10.1609/aaai.v36i5.20421)

**Abstract**:

We study a participatory budgeting problem, where a set of strategic agents wish to split a divisible budget among different projects by aggregating their proposals on a single division. Unfortunately, the straightforward rule that divides the budget proportionally is susceptible to manipulation. Recently, a class of truthful mechanisms has been proposed, namely the moving phantom mechanisms. One such mechanism satisfies the proportionality property, in the sense that in the extreme case where all agents prefer a single project to receive the whole amount, the budget is assigned proportionally.


While proportionality is a naturally desired property, it is defined over a limited type of preference profiles. To address this, we expand the notion of proportionality, by proposing a quantitative framework that evaluates a budget aggregation mechanism according to its worst-case distance from the proportional allocation. Crucially, this is defined for every preference profile. We study this measure on the class of moving phantom mechanisms, and we provide approximation guarantees. For two projects, we show that the Uniform Phantom mechanism is optimal among all truthful mechanisms. For three projects, we propose a new, proportional mechanism that is optimal among all moving phantom mechanisms. Finally, we provide impossibility results regarding the approximability of moving phantom mechanisms.

----

## [549] The Complexity of Learning Approval-Based Multiwinner Voting Rules

**Authors**: *Ioannis Caragiannis, Karl Fehrs*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20422](https://doi.org/10.1609/aaai.v36i5.20422)

**Abstract**:

We study the PAC learnability of multiwinner voting, focusing on the class of approval-based committee scoring (ABCS) rules. These are voting rules applied on profiles with approval ballots, where each voter approves some of the candidates. According to ABCS rules, each committee of k candidates collects from each voter a score, that depends on the size of the voter's ballot and on the size of its intersection with the committee. Then, committees of maximum score are the winning ones. Our goal is to learn a target rule (i.e., to learn the corresponding scoring function) using information about the winning committees of a small number of sampled profiles. Despite the existence of exponentially many outcomes compared to single-winner elections, we show that the sample complexity is still low: a polynomial number of samples carries enough information for learning the target rule with high confidence and accuracy. Unfortunately, even simple tasks that need to be solved for learning from these samples are intractable. We prove that deciding whether there exists some ABCS rule that makes a given committee winning in a given profile is a computationally hard problem. Our results extend to the class of sequential Thiele rules, which have received attention due to their simplicity.

----

## [550] Efficiency of Ad Auctions with Price Displaying

**Authors**: *Matteo Castiglioni, Diodato Ferraioli, Nicola Gatti, Alberto Marchesi, Giulia Romano*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20423](https://doi.org/10.1609/aaai.v36i5.20423)

**Abstract**:

Most economic reports suggest that almost half of the market value unlocked by artificial intelligence (AI) by the next decade (about 9 trillion USD per year) will be in marketing&sales. In particular, AI will allow the optimization of more and more intricate economic settings in which multiple different activities can be automated jointly. A relatively recent example is that one of ad auctions in which similar products or services are displayed together with their price, thus merging advertising and pricing in a unique website. This is the case, e.g., of Google Hotel Ads and TripAdvisor. More precisely, as in a classical ad auction, the ranking of the ads depends on the advertisers' bids, while, differently from classical ad auctions, the price is displayed together with the ad, so as to provide a direct comparison among the prices and thus dramatically affect the behavior of the users. This paper investigates how displaying prices and ads together conditions the properties of the main economic mechanisms such as VCG and GSP. Initially, we focus on the direct-revelation mechanism, showing that prices are chosen by the mechanisms once given the advertisers' reports. We also provide an efficient algorithm to compute the optimal allocation given the private information reported by the advertisers. Then, with both VCG and GSP payments, we show the inefficiency in terms of Price of Anarchy (PoA) and Stability (PoS) over the social welfare and mechanism's revenue when the advertisers choose the prices. The main results show that, with both VCG and GSP, PoS over the revenue may be unbounded even with two slots, while PoA over the social welfare may be as large as the number of slots. Finally, we show that, under some assumptions, simple modifications to VCG and GSP allow us to obtain a better PoS over the revenue.

----

## [551] Signaling in Posted Price Auctions

**Authors**: *Matteo Castiglioni, Giulia Romano, Alberto Marchesi, Nicola Gatti*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20424](https://doi.org/10.1609/aaai.v36i5.20424)

**Abstract**:

We study single-item single-unit Bayesian posted price auctions, where buyers arrive sequentially and their valuations for the item being sold depend on a random, unknown state of nature. The seller has complete knowledge of the actual state and can send signals to the buyers so as to disclose information about it. For instance, the state of nature may reflect the condition and/or some particular features of the item, which are known to the seller only. The problem faced by the seller is about how to partially disclose information about the state so as to maximize revenue. Unlike classical signaling problems, in this setting, the seller must also correlate the signals being sent to the buyers with some price proposals for them. This introduces additional challenges compared to standard settings. We consider two cases: the one where the seller can only send signals publicly visible to all buyers, and the case in which the seller can privately send a different signal to each buyer. As a first step, we prove that, in both settings, the problem of maximizing the seller's revenue does not admit an FPTAS unless P=NP, even for basic instances with a single buyer. As a result, in the rest of the paper, we focus on designing PTASs. In order to do so, we first introduce a unifying framework encompassing both public and private signaling, whose core result is a decomposition lemma that allows focusing on a finite set of possible buyers' posteriors. This forms the basis on which our PTASs are developed. In particular, in the public signaling setting, our PTAS employs some ad hoc techniques based on linear programming, while our PTAS for the private setting relies on the ellipsoid method to solve an exponentially-sized LP in polynomial time. In the latter case, we need a custom approximate separation oracle, which we implement with a dynamic programming approach.

----

## [552] Weighted Fairness Notions for Indivisible Items Revisited

**Authors**: *Mithun Chakraborty, Erel Segal-Halevi, Warut Suksompong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20425](https://doi.org/10.1609/aaai.v36i5.20425)

**Abstract**:

We revisit the setting of fairly allocating indivisible items when agents have different weights representing their entitlements. First, we propose a parameterized family of relaxations for weighted envy-freeness and the same for weighted proportionality; the parameters indicate whether smaller-weight or larger-weight agents should be given a higher priority. We show that each notion in these families can always be satisfied, but any two cannot necessarily be fulfilled simultaneously. We then introduce an intuitive weighted generalization of maximin share fairness and establish the optimal approximation of it that can be guaranteed. Furthermore, we characterize the implication relations between the various weighted fairness notions introduced in this and prior work, and relate them to the lower and upper quota axioms from apportionment.

----

## [553] Pizza Sharing Is PPA-Hard

**Authors**: *Argyrios Deligkas, John Fearnley, Themistoklis Melissourgos*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20426](https://doi.org/10.1609/aaai.v36i5.20426)

**Abstract**:

We study the computational complexity of computing solutions for the
 straight-cut and square-cut pizza sharing problems. We show that finding an
 approximate solution is PPA-hard for the straight-cut problem, and
 PPA-complete for the square-cut problem, while finding an exact solution for
 the square-cut problem is FIXP-hard and in BU. Our PPA-hardness results apply
 even when all mass distributions are unions of non-overlapping squares, and our
 FIXP-hardness result applies even when all mass distributions are unions of
 weighted squares and right-angled triangles. We also prove that decision variants
 of the square-cut problem are hard: the approximate problem is
 NP-complete, and the exact problem is ETR-complete.

----

## [554] Heterogeneous Facility Location with Limited Resources

**Authors**: *Argyrios Deligkas, Aris Filos-Ratsikas, Alexandros A. Voudouris*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20427](https://doi.org/10.1609/aaai.v36i5.20427)

**Abstract**:

We initiate the study of the heterogeneous facility location problem with limited resources. We mainly focus on the fundamental case where a set of agents are positioned in the line segment [0,1] and have approval preferences over two available facilities. A mechanism takes as input the positions and the preferences of the agents, and chooses to locate a single facility based on this information. We study mechanisms that aim to maximize the social welfare (the total utility the agents derive from facilities they approve), under the constraint of incentivizing the agents to truthfully report their positions and preferences. We consider three different settings depending on the level of agent-related information that is public or private. For each setting, we design deterministic and randomized strategyproof mechanisms that achieve a good approximation of the optimal social welfare, and complement these with nearly-tight impossibility results.

----

## [555] Complexity of Deliberative Coalition Formation

**Authors**: *Edith Elkind, Abheek Ghosh, Paul Goldberg*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20428](https://doi.org/10.1609/aaai.v36i5.20428)

**Abstract**:

Elkind et al. (AAAI'21) introduced a model for deliberative coalition formation, where a community wishes to identify a strongly supported proposal from a space of alternatives, in order to change the status quo. In their model, agents and proposals are points in a metric space, agents' preferences are determined by distances, and agents deliberate by dynamically forming coalitions around proposals that they prefer over the status quo. The deliberation process operates via k-compromise transitions, where agents from k (current) coalitions come together to form a larger coalition in order to support a (perhaps new) proposal, possibly leaving behind some of the dissenting agents from their old coalitions. A deliberation succeeds if it terminates by identifying a proposal with the largest possible support. For deliberation in d dimensions, Elkind et al. consider two variants of their model: in the Euclidean model, proposals and agent locations are points in R^d and the distance is measured according to ||...||_2; and in the hypercube model, proposals and agent locations are vertices of the d-dimensional hypercube and the metric is the Hamming distance. They show that in the Euclidean model 2-compromises are guaranteed to succeed, but in the hypercube model for deliberation to succeed it may be necessary to use k-compromises with k >= d. 
We complement their analysis by 
 (1) proving that in both models it is hard to find a proposal with a high degree of support, and even a 2-compromise transition may be hard to compute;
 (2) showing that a sequence of 2-compromise transitions may be exponentially long;
 (3) strengthening the lower bound on the size of the compromise for the d-hypercube model from d to 2^Ω(d).

----

## [556] The Price of Justified Representation

**Authors**: *Edith Elkind, Piotr Faliszewski, Ayumi Igarashi, Pasin Manurangsi, Ulrike Schmidt-Kraepelin, Warut Suksompong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20429](https://doi.org/10.1609/aaai.v36i5.20429)

**Abstract**:

In multiwinner approval voting, the goal is to select k-member committees based on voters' approval ballots. A well-studied concept of proportionality in this context is the justified representation (JR) axiom, which demands that no large cohesive group of voters remains unrepresented. However, the JR axiom may conflict with other desiderata, such as coverage (maximizing the number of voters who approve at least one committee member) or social welfare (maximizing the number of approvals obtained by committee members). In this work, we investigate the impact of imposing the JR axiom (as well as the more demanding EJR axiom) on social welfare and coverage. Our approach is threefold: we derive worst-case bounds on the loss of welfare/coverage that is caused by imposing JR, study the computational complexity of finding 'good' committees that provide JR (obtaining a hardness result, an approximation algorithm, and an exact algorithm for one-dimensional preferences), and examine this setting empirically on several synthetic datasets.

----

## [557] The Complexity of Subelection Isomorphism Problems

**Authors**: *Piotr Faliszewski, Krzysztof Sornat, Stanislaw Szufa*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20430](https://doi.org/10.1609/aaai.v36i5.20430)

**Abstract**:

We study extensions of the Election Isomorphism problem, focused on the existence of isomorphic subelections. Specifically, we propose the Subelection Isomorphism and the Maximum Common Subelection problems and study their computational complexity and approximability. Using our problems in experiments, we provide some insights into the nature of several statistical models of elections.

----

## [558] Fast Payoff Matrix Sparsification Techniques for Structured Extensive-Form Games

**Authors**: *Gabriele Farina, Tuomas Sandholm*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20431](https://doi.org/10.1609/aaai.v36i5.20431)

**Abstract**:

The practical scalability of many optimization algorithms for large extensive-form games is often limited by the games' huge payoff matrices. To ameliorate the issue, Zhang and Sandholm recently proposed a sparsification technique that factorizes the payoff matrix A into a sparser object A = Â + UVᵀ, where the total combined number of nonzeros of Â, U, and V, is significantly smaller. Such a factorization can be used in place of the original payoff matrix in many optimization algorithm, such as interior-point and second-order methods, thus increasing the size of games that can be handled. Their technique significantly sparsifies poker (end)games, standard benchmarks used in computational game theory, AI, and more broadly. We show that the existence of extremely sparse factorizations in poker games can be tied to their particular Kronecker-product structure. We clarify how such structure arises and introduce the connection between that structure and sparsification. By leveraging such structure, we give two ways of computing strong sparsifications of poker games (as well as any other game with a similar structure) that are i) orders of magnitude faster to compute, ii) more numerically stable, and iii) produce a dramatically smaller number of nonzeros than the prior technique. Our techniques enable—for the first time—effective computation of high-precision Nash equilibria and strategies subject to constraints on the amount of allowed randomization. Furthermore, they significantly speed up parallel first-order game-solving algorithms; we show state-of-the-art speed on a GPU.

----

## [559] Two-Price Equilibrium

**Authors**: *Michal Feldman, Galia Shabtai, Aner Wolfenfeld*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20432](https://doi.org/10.1609/aaai.v36i5.20432)

**Abstract**:

Walrasian equilibrium is a prominent market equilibrium notion, but rarely exists in markets with indivisible items.
 We introduce a new market equilibrium notion, called two-price equilibrium (2PE). A 2PE is a relaxation of Walrasian equilibrium, where instead of a single price per item, every item has two prices: one for the item's owner and a (possibly) higher one for all other buyers. 
 Thus, a 2PE is given by a tuple (S,p_high,p_low) of an allocation S and two price vectors p_high,p_low, where every buyer i is maximally happy with her bundle S_i, given prices p_low for items in S_i and prices p_high for all other items. 
 2PE generalizes previous market equilibrium notions, such as conditional equilibrium, and is related to relaxed equilibrium notions like endowment equilibrium. 
 We define the discrepancy of a 2PE --- a measure of distance from Walrasian equilibrium --- as the sum of differences p_high_j-p_low_j over all items (normalized by social welfare).
 We show that the social welfare degrades gracefully with the discrepancy; namely, the social welfare of a 2PE with discrepancy d is at least a fraction 1/d+1 of the optimal welfare.
 We use this to establish welfare guarantees for markets with subadditive valuations over identical items.
 In particular, we show that every such market admits a 2PE with at least 1/7 of the optimal welfare.
 This is in contrast to Walrasian equilibrium or conditional equilibrium which may not even exist.
 Our techniques provide new insights regarding valuation functions over identical items, which we also use to characterize instances that admit a WE.

----

## [560] Algorithmic Bayesian Persuasion with Combinatorial Actions

**Authors**: *Kaito Fujii, Shinsaku Sakaue*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20433](https://doi.org/10.1609/aaai.v36i5.20433)

**Abstract**:

Bayesian persuasion is a model for understanding strategic information revelation: an agent with an informational advantage, called a sender, strategically discloses information by sending signals to another agent, called a receiver. In algorithmic Bayesian persuasion, we are interested in efficiently designing the sender's signaling schemes that lead the receiver to take action in favor of the sender. This paper studies algorithmic Bayesian-persuasion settings where the receiver's feasible actions are specified by combinatorial constraints, e.g., matroids or paths in graphs. We first show that constant-factor approximation is NP-hard even in some special cases of matroids or paths. We then propose a polynomial-time algorithm for general matroids by assuming the number of states of nature to be a constant. We finally consider a relaxed notion of persuasiveness, called CCE-persuasiveness, and present a sufficient condition for polynomial-time approximability.

----

## [561] Bayesian Persuasion in Sequential Decision-Making

**Authors**: *Jiarui Gan, Rupak Majumdar, Goran Radanovic, Adish Singla*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20434](https://doi.org/10.1609/aaai.v36i5.20434)

**Abstract**:

We study a dynamic model of Bayesian persuasion in sequential decision-making settings. An informed principal observes an external parameter of the world and advises an uninformed agent about actions to take over time. The agent takes actions in each time step based on the current state, the principal's advice/signal, and beliefs about the external parameter. The action of the agent updates the state according to a stochastic process. The model arises naturally in many applications, e.g., an app (the principal) can advice the user (the agent) on possible choices between actions based on additional real-time information the app has. We study the problem of designing a signaling strategy from the principal's point of view. We show that the principal has an optimal strategy against a myopic agent, who only optimizes their rewards locally, and the optimal strategy can be computed in polynomial time. In contrast, it is NP-hard to approximate an optimal policy against a far-sighted agent. Further, we show that if the principal has the power to threaten the agent by not providing future signals, then we can efficiently design a threat-based strategy. This strategy guarantees the principal's payoff as if playing against an agent who is far-sighted but myopic to future signals.

----

## [562] Hedonic Diversity Games: A Complexity Picture with More than Two Colors

**Authors**: *Robert Ganian, Thekla Hamm, Dusan Knop, Simon Schierreich, Ondrej Suchý*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20435](https://doi.org/10.1609/aaai.v36i5.20435)

**Abstract**:

Hedonic diversity games are a variant of the classical Hedonic games designed to better model a variety of questions concerning diversity and fairness. Previous works mainly targeted the case with two diversity classes (represented as colors in the model) and provided a set of initial complexity-theoretic and existential results concerning Nash and Individually stable outcomes. Here, we design new algorithms accompanied with lower bounds which provide a full parameterized-complexity picture for computing Nash and Individually stable outcomes with respect to the most natural parameterizations of the problem. Crucially, our results hold for general Hedonic diversity games where the number of colors is not necessarily restricted to two, and show that---apart from two trivial cases---a necessary condition for tractability in this setting is that the number of colors is bounded by the parameter. Moreover, for the special case of two colors we resolve an open question asked in previous work~(Boehmer and Elkind, AAAI 2020).

----

## [563] Fair and Efficient Allocations of Chores under Bivalued Preferences

**Authors**: *Jugal Garg, Aniket Murhekar, John Qin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20436](https://doi.org/10.1609/aaai.v36i5.20436)

**Abstract**:

We study the problem of fair and efficient allocation of a set of indivisible chores to agents with additive cost functions. We consider the popular fairness notion of envy-freeness up to one good (EF1) with the efficiency notion of Pareto-optimality (PO). While it is known that EF1+PO allocations exists and can be computed in pseudo-polynomial time in the case of goods, the same problem is open for chores.
 
 Our first result is a strongly polynomial-time algorithm for computing an EF1+PO allocation for bivalued instances, where agents have (at most) two disutility values for the chores. To the best of our knowledge, this is the first non-trivial class of chores to admit an EF1+PO allocation and an efficient algorithm for its computation. 
 
 We also study the problem of computing an envy-free (EF) and PO allocation for the case of divisible chores. While the existence of EF+PO allocation is known via competitive equilibrium with equal incomes, its efficient computation is open. Our second result shows that for bivalued instances, an EF+PO allocation can be computed in strongly polynomial-time.

----

## [564] Secretary Matching with Vertex Arrivals and No Rejections

**Authors**: *Mohak Goyal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20437](https://doi.org/10.1609/aaai.v36i5.20437)

**Abstract**:

Most prior work on online matching problems has been with the flexibility of keeping some vertices unmatched. We study three related online matching problems with the constraint of matching every vertex, i.e., with no rejections. We adopt a model in which vertices arrive in a uniformly random order and the edge-weights are arbitrary positive numbers. For the capacitated online bipartite matching problem in which the vertices of one side of the graph are offline and those of the other side arrive online, we give a 4.62-competitive algorithm when the capacity of each offline vertex is 2. For the online general (non-bipartite) matching problem, where all vertices arrive online, we give a 3.34-competitive algorithm. We also study the online roommate matching problem, in which each room (offline vertex) holds 2 persons (online vertices). Persons derive non-negative additive utilities from their room as well as roommate. In this model, with the goal of maximizing the sum of utilities of all persons, we give a 7.96-competitive algorithm. This is an improvement over the 24.72 approximation factor in prior work.

----

## [565] Machine-Learned Prediction Equilibrium for Dynamic Traffic Assignment

**Authors**: *Lukas Graf, Tobias Harks, Kostas Kollias, Michael Markl*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20438](https://doi.org/10.1609/aaai.v36i5.20438)

**Abstract**:

We study a dynamic traffic assignment model, where agents base their instantaneous routing decisions on real-time delay predictions. We formulate a mathematically concise model and derive properties of the predictors that ensure a dynamic prediction equilibrium exists. We demonstrate the versatility of our framework by showing that it subsumes the well-known full information and instantaneous information models, in addition to admitting further realistic predictors as special cases. We complement our theoretical analysis by an experimental study, in which we systematically compare the induced average travel times of different predictors, including a machine-learning model trained on data gained from previously computed equilibrium flows, both on a synthetic and a real road network.

----

## [566] Multi-Leader Congestion Games with an Adversary

**Authors**: *Tobias Harks, Mona Henle, Max Klimm, Jannik Matuschke, Anja Schedel*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20439](https://doi.org/10.1609/aaai.v36i5.20439)

**Abstract**:

We study a multi-leader single-follower congestion game where multiple users (leaders) choose one resource out of a set of resources and, after observing the realized loads, an adversary (single-follower) attacks the resources with maximum loads causing additional costs for the leaders. For the resulting strategic game among the leaders, we show that pure Nash equilibria fail to exist and therefore, we consider approximate equilibria instead.
 
 As our first main result, we show that the existence of a K-approximate equilibrium can always be guaranteed, where K (approximately equal to 1.1974) is the unique solution of a cubic polynomial equation. To this end, we give a polynomial time combinatorial algorithm which computes a K-approximate equilibrium. The factor K is tight, meaning that there is an instance that does not admit an A-approximate equilibrium for any A < K. Thus A = K is the smallest possible value of A such that the existence of an A-approximate equilibrium can be guaranteed for any instance of the considered game. 
 
 Secondly, we focus on approximate equilibria of a given fixed instance. We show how to compute efficiently a best approximate equilibrium, that is, with smallest possible A among all A-approximate equilibria of the given instance.

----

## [567] Approval-Based Committee Voting under Incomplete Information

**Authors**: *Aviram Imber, Jonas Israel, Markus Brill, Benny Kimelfeld*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20440](https://doi.org/10.1609/aaai.v36i5.20440)

**Abstract**:

We investigate approval-based committee voting with incomplete information about the approval preferences of voters. We consider several models of incompleteness where each voter partitions the set of candidates into approved, disapproved, and unknown candidates, possibly with ordinal preference constraints among candidates in the latter category. This captures scenarios where voters have not evaluated all candidates and/or it is unknown where voters draw the threshold between approved and disapproved candidates. We study the complexity of some fundamental computational problems for a number of classic approval-based committee voting rules including Proportional Approval Voting and Chamberlin-Courant. These problems include that of determining whether a given set of candidates is a possible or necessary winning committee and whether it forms a committee that possibly or necessarily satisfies representation axioms. We also consider the problem whether a given candidate is possibly or necessarily a member of the winning committee.

----

## [568] Reforming an Envy-Free Matching

**Authors**: *Takehiro Ito, Yuni Iwamasa, Naonori Kakimura, Naoyuki Kamiyama, Yusuke Kobayashi, Yuta Nozaki, Yoshio Okamoto, Kenta Ozeki*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20441](https://doi.org/10.1609/aaai.v36i5.20441)

**Abstract**:

We consider the problem of reforming an envy-free matching when each agent is assigned a single item. Given an envy-free matching, we consider an operation to exchange the item of an agent with an unassigned item preferred by the agent that results in another envy-free matching. We repeat this operation as long as we can. We prove that the resulting envy-free matching is uniquely determined up to the choice of an initial envy-free matching, and can be found in polynomial time. We call the resulting matching a reformist envy-free matching, and then we study a shortest sequence to obtain the reformist envy-free matching from an initial envy-free matching. We prove that a shortest sequence is computationally hard to obtain even when each agent accepts at most four items and each item is accepted by at most three agents. On the other hand, we give polynomial-time algorithms when each agent accepts at most three items or each item is accepted by at most two agents. Inapproximability and fixed-parameter (in)tractability are also discussed.

----

## [569] The Complexity of Proportionality Degree in Committee Elections

**Authors**: *Lukasz Janeczko, Piotr Faliszewski*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20442](https://doi.org/10.1609/aaai.v36i5.20442)

**Abstract**:

Over the last few years, researchers have put significant effort into understanding of the notion of proportional representation in committee election. In particular, recently they have proposed the notion of proportionality degree. We study the complexity of computing committees with a given proportionality degree and of testing if a given committee provides a particular one. This way, we complement recent studies that mostly focused on the notion of (extended) justified representation.  We also study the problems of testing if a cohesive group of a given size exists and of counting such groups.

----

## [570] Worst-Case Voting When the Stakes Are High

**Authors**: *Anson Kahng, Gregory Kehne*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20443](https://doi.org/10.1609/aaai.v36i5.20443)

**Abstract**:

We study the additive distortion of social choice functions in the implicit utilitarian model, and argue that it is a more appropriate metric than multiplicative distortion when an alternative that confers significant social welfare may exist (i.e., when the stakes are high). We define a randomized analog of positional scoring rules, and present a rule which is asymptotically optimal within this class as the number of alternatives increases. We then show that the instance-optimal social choice function can be efficiently computed. Next, we take a beyond-worst-case view, bounding the additive distortion of prominent voting rules as a function of the best welfare attainable in an instance. Lastly, we evaluate the additive distortion of a range of rules on real-world election data.

----

## [571] PageRank for Edges: Axiomatic Characterization

**Authors**: *Natalia Kucharczuk, Tomasz Was, Oskar Skibski*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20444](https://doi.org/10.1609/aaai.v36i5.20444)

**Abstract**:

Edge centrality measures are functions that evaluate the importance of edges in a network. They can be used to assess the role of a backlink for the popularity of a website as well as the importance of a flight in virus spreading. Various node centralities have been translated to apply for edges, including Edge Betweenness, Eigenedge (edge version of eigenvector centrality), and Edge PageRank. With this paper, we initiate the discussion on the axiomatic properties of edge centrality measures. We do it by proposing an axiomatic characterization of Edge PageRank. Our characterization is the first characterization of any edge centrality measures in the literature.

----

## [572] Safe Subgame Resolving for Extensive Form Correlated Equilibrium

**Authors**: *Chun Kai Ling, Fei Fang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20445](https://doi.org/10.1609/aaai.v36i5.20445)

**Abstract**:

Correlated Equilibrium is a solution concept that is more general than Nash Equilibrium (NE) and can lead to outcomes with better social welfare. However, its natural extension to the sequential setting, the Extensive Form Correlated Equilibrium (EFCE), requires a quadratic amount of space to solve, even in restricted settings without randomness in nature. To alleviate these concerns, we apply subgame resolving, a technique extremely successful in finding NE in zero-sum games to solving general-sum EFCEs. Subgame resolving refines a correlation plan in an online manner: instead of solving for the full game upfront, it only solves for strategies in subgames that are reached in actual play, resulting in significant computational gains. In this paper, we (i) lay out the foundations to quantify the quality of a refined strategy, in terms of the social welfare and exploitability of correlation plans, (ii) show that EFCEs possess a sufficient amount of independence between subgames to perform resolving efficiently, and (iii) provide two algorithms for resolving, one using linear programming and the other based on regret minimization. Both methods guarantee safety, i.e., they will never be counterproductive. Our methods are the first time an online method has been applied to the correlated, general-sum setting.

----

## [573] The Semi-random Likelihood of Doctrinal Paradoxes

**Authors**: *Ao Liu, Lirong Xia*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20446](https://doi.org/10.1609/aaai.v36i5.20446)

**Abstract**:

When aggregating logically interconnected judgements from n agents, the result might be logically inconsistent. This phenomenon is known as the doctrinal paradox, which plays a central role in the field of judgement aggregation. Previous work has mostly focused on the worst-case analysis of the doctrinal paradox, leading to many impossibility results. Little is known about its likelihood of occurrence in practical settings,  except for the study under certain distributions by List in 2005. In this paper, we characterize the likelihood of the doctrinal paradox under a  general and realistic model called semi-random social choice framework (proposed by Xia in 2020). In the framework, agents' ground truth judgements can be arbitrarily correlated, while the noises are independent.  Our main theorem states that under mild conditions, the semi-random likelihood of the doctrinal paradox is either 0, exp(-Θ(n)), Θ(n\^~(-0.5)) or Θ(1). This not only answers open questions by List in 2005, but also draws clear lines between situations with frequent paradoxes and with vanishing paradoxes.

----

## [574] Is There a Strongest Die in a Set of Dice with the Same Mean Pips?

**Authors**: *Shang Lu, Shuji Kijima*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20447](https://doi.org/10.1609/aaai.v36i5.20447)

**Abstract**:

Jan-ken, a.k.a. rock-paper-scissors, is a cerebrated example of a non-transitive game with three (pure) strategies, rock, paper and scissors. Interestingly, any Jan-ken generalized to four strategies contains at least one useless strategy unless it allows a tie between distinct pure strategies. Non-transitive dice could be a stochastic analogue of Jan-ken: the stochastic transitivity does not hold on some sets of dice, e.g., Efron's dice. Including the non-transitive dice, this paper is interested in dice sets which do not contain a useless die.
  In particular, we are concerned with the existence of a strongest (or weakest, symmetrically) die in a dice set under the two conditions that (1) any number appears on at most one die and at most one side, i.e., no tie break between two distinct dice, and (2) the mean pips of dice are the same. We firstly prove that a strongest die never exist if a set of n dice of m-sided is given as a partition of the set of numbers {1,…,mn}. Next, we show some sufficient conditions that a strongest die exists in a dice set which is not a partition of a set of numbers. We also give some algorithms to find a strongest die in a dice set which includes given dice.

----

## [575] Choices Are Not Independent: Stackelberg Security Games with Nested Quantal Response Models

**Authors**: *Tien Mai, Arunesh Sinha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20448](https://doi.org/10.1609/aaai.v36i5.20448)

**Abstract**:

The quantal response (QR) model is widely used in Stackelberg security games (SSG) to model a bounded rational adversary. The QR model is a model of human response from among a large variety of prominent models known as discrete choice models. QR is the simplest type of discrete choice models and does not capture commonly observed phenomenon such as correlation among choices. We introduce the nested QR adversary model (based on nested logit model in discrete choice theory) in SSG which addresses shortcoming of the QR model. We present tractable approximation of the resulting equilibrium problem with nested QR adversary. We do so by deriving an interesting property of the equilibrium problem, namely a loosely coupled split into nested problems that mirrors the nested decision making by the adversary in the nested QR model. We show that each separate nested problem can be approximated efficiently and that the loosely coupled overall problem can be solved approximately by formulating it as a discretized version of a continuous dynamic program. Finally, we conduct experiments that show the scalability and parallelizability of our approach, as well as advantages of the nested QR model.

----

## [576] Strictly Proper Contract Functions Can Be Arbitrage-Free

**Authors**: *Eric Neyman, Tim Roughgarden*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20449](https://doi.org/10.1609/aaai.v36i5.20449)

**Abstract**:

We consider mechanisms for truthfully eliciting probabilistic predictions from a group of experts. The standard approach --- using a proper scoring rule to separately reward each expert --- is not robust to collusion: experts may collude to misreport their beliefs in a way that guarantees them a larger total reward no matter the eventual outcome. It is a long-standing open question whether there is a truthful elicitation mechanism that makes any such collusion (also called "arbitrage") impossible. We resolve this question positively, exhibiting a class of strictly proper arbitrage-free contract functions. These contract functions have two parts: one ensures that the total reward of a coalition of experts depends only on the average of their reports; the other ensures that changing this average report hurts the experts under at least one outcome.

----

## [577] Characterization of Incentive Compatibility of an Ex-ante Constrained Player

**Authors**: *Bonan Ni, Pingzhong Tang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20450](https://doi.org/10.1609/aaai.v36i5.20450)

**Abstract**:

We consider a variant of the standard Bayesian mechanism, where players evaluate their outcomes and constraints in an ex-ante manner. Such a model captures a major form of modern online advertising where an advertiser is concerned with her/his expected utility over a time period and her/his type may change over time. We are interested in the incentive compatibility (IC) problem of such Bayesian mechanism. Under very mild conditions on the mechanism environments, we give a full characterization of IC via the taxation principle and show, perhaps surprisingly, that such IC mechanisms are fully characterized by the so-called auto-bidding mechanisms, which are pervasively fielded in the online advertising industry.

----

## [578] Online Elicitation of Necessarily Optimal Matchings

**Authors**: *Jannik Peters*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20451](https://doi.org/10.1609/aaai.v36i5.20451)

**Abstract**:

In this paper, we study the problem of eliciting preferences of agents in the house allocation model. For this we build on a recently introduced model  and focus on the task of eliciting preferences to find matchings which are necessarily optimal, i.e., optimal under all possible completions of the elicited preferences.  In particular, we investigate the elicitation of necessarily Pareto optimal (NPO) and necessarily rank-maximal (NRM) matchings. Most importantly, we answer an open question and give an online algorithm for eliciting an NRM matching in the next-best query model which is  3/2-competitive, i.e., it takes at most 3/2 as many queries as an optimal algorithm. Besides this, we extend this field of research by introducing two new natural models of elicitation and by studying both the complexity of determining whether a necessarily optimal matching exists in them, and by giving online algorithms for these models.

----

## [579] Generalized Dynamic Cognitive Hierarchy Models for Strategic Driving Behavior

**Authors**: *Atrisha Sarkar, Kate Larson, Krzysztof Czarnecki*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20452](https://doi.org/10.1609/aaai.v36i5.20452)

**Abstract**:

While there has been an increasing focus on the use of game theoretic models for autonomous driving, empirical evidence shows that there are still open questions around dealing with the challenges of common knowledge assumptions as well as modeling bounded rationality. To address some of these practical challenges, we develop a framework of generalized dynamic cognitive hierarchy for both modelling naturalistic human driving behavior as well as behavior planning for autonomous vehicles (AV). This framework is built upon a rich model of level-0 behavior through the use of automata strategies, an interpretable notion of bounded rationality through safety and maneuver satisficing, and a robust response for planning. Based on evaluation on two large naturalistic datasets as well as simulation of critical traffic scenarios, we show that i) automata strategies are well suited for level-0 behavior in a dynamic level-k framework, and ii) the proposed robust response to a heterogeneous population of strategic and non-strategic reasoners can be an effective approach for game theoretic planning in AV.

----

## [580] Improved Maximin Guarantees for Subadditive and Fractionally Subadditive Fair Allocation Problem

**Authors**: *Masoud Seddighin, Saeed Seddighin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20453](https://doi.org/10.1609/aaai.v36i5.20453)

**Abstract**:

In this work, we study the maximin share fairness notion for allocation of indivisible goods in the subadditive and fractionally subadditive settings. While previous work refutes the possibility of obtaining an allocation which is better than 1/2-MMS, the only positive result for the subadditive setting states that when the number of items is equal to m, there always exists an Ω(1/log m)-\MMS allocation. Since the number of items may be larger than the number of agents (n), such a bound can only imply a weak bound of Ω(1/(n log n))-MMS allocation in general.
 
 In this work, we improve this gap exponentially to an Ω(1/(log n log log n))-MMS guarantee. In addition to this, we prove that when the valuation functions are fractionally subadditive, a 1/4.6-MMS allocation is guaranteed to exist. This also improves upon the previous bound of 1/5-MMS guarantee for the fractionally subadditive setting.

----

## [581] Proportional Public Decisions

**Authors**: *Piotr Skowron, Adrian Górecki*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20454](https://doi.org/10.1609/aaai.v36i5.20454)

**Abstract**:

We consider a setting where a group of individuals needs to make a number of independent decisions. The decisions should proportionally represent the views of the voters. We formulate new criteria of proportionality and analyse two rules, Proportional Approval Voting and the Method of Equal Shares, that are inspired by the corresponding approval-based committee election rules. We prove that the two rules provide very strong proportionality guarantees when applied to the setting of public decisions.

----

## [582] Online Task Assignment Problems with Reusable Resources

**Authors**: *Hanna Sumita, Shinji Ito, Kei Takemura, Daisuke Hatano, Takuro Fukunaga, Naonori Kakimura, Ken-ichi Kawarabayashi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20455](https://doi.org/10.1609/aaai.v36i5.20455)

**Abstract**:

We study online task assignment problem with reusable resources, motivated by practical applications such as ridesharing, crowdsourcing and job hiring. In the problem, we are given a set of offline vertices (agents), and, at each time, an online vertex (task) arrives randomly according to a known time-dependent distribution. Upon arrival, we assign the task to agents immediately and irrevocably. The goal of the problem is to maximize the expected total profit produced by completed tasks. The key features of our problem are (1) an agent is reusable, i.e., an agent comes back to the market after completing the assigned task, (2) an agent may reject the assigned task to stay the market, and (3) a task may accommodate multiple agents. The setting generalizes that of existing work in which an online task is assigned to one agent under (1).

In this paper, we propose an online algorithm that is 1/2-competitive for the above setting, which is tight. Moreover, when each agent can reject assigned tasks at most Δ times, the algorithm is shown to have the competitive ratio Δ/(3Δ-1), which is at least 1/3. We also evaluate our proposed algorithm with numerical experiments.

----

## [583] Iterative Calculus of Voting under Plurality

**Authors**: *Fabricio Vasselai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20456](https://doi.org/10.1609/aaai.v36i5.20456)

**Abstract**:

We formalize a voting model for plurality elections that combines Iterative Voting and Calculus of Voting. Each iteration, autonomous agents simultaneously maximize the utility they expect from candidates. Agents are aware of neither other individuals’ preferences or choices, nor of the distribution of preferences. They know only of candidates’ latest vote shares and with that calculate expected rewards from each candidate, pondering the probability that voting for each would alter the election. We define the general form of those pivotal probabilities, then we derive efficient exact and approximated calculations. Lastly, we prove formally the model converges with asymptotically large electorates and show via simulations that it nearly always converges even with very few agents.

----

## [584] Coordinating Followers to Reach Better Equilibria: End-to-End Gradient Descent for Stackelberg Games

**Authors**: *Kai Wang, Lily Xu, Andrew Perrault, Michael K. Reiter, Milind Tambe*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20457](https://doi.org/10.1609/aaai.v36i5.20457)

**Abstract**:

A growing body of work in game theory extends the traditional Stackelberg game to settings with one leader and multiple followers who play a Nash equilibrium. Standard approaches for computing equilibria in these games reformulate the followers' best response as constraints in the leader's optimization problem. These reformulation approaches can sometimes be effective, but make limiting assumptions on the followers' objectives and the equilibrium reached by followers, e.g., uniqueness, optimism, or pessimism. To overcome these limitations, we run gradient descent to update the leader's strategy by differentiating through the equilibrium reached by followers. Our approach generalizes to any stochastic equilibrium selection procedure that chooses from multiple equilibria, where we compute the stochastic gradient by back-propagating through a sampled Nash equilibrium using the solution to a partial differential equation to establish the unbiasedness of the stochastic gradient. Using the unbiased gradient estimate, we implement the gradient-based approach to solve three Stackelberg problems with multiple followers. Our approach consistently outperforms existing baselines to achieve higher utility for the leader.

----

## [585] Multi-Unit Auction in Social Networks with Budgets

**Authors**: *Mingyu Xiao, Yuchao Song, Bakh Khoussainov*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20458](https://doi.org/10.1609/aaai.v36i5.20458)

**Abstract**:

We study multi-unit auctions in social networks, where each buyer has a fixed budget and can spread the sale information to the network neighbors. We design a mechanism encouraging buyers to report their valuations truthfully and spread the sale information. Our design uses the idea of the clinching mechanism to decide the transaction price and can be viewed as a network version of the mechanism. Most of the previous clinching mechanisms search for the transaction prices by increasing the current price. Our mechanism directly computes the transaction prices in polynomial time. Furthermore, the mechanism applies a technique to iteratively activate new buyers in the network. This ensures utility preservations of the buyers and benefits the seller. We prove key properties of our mechanism, such as no-positive-transfers, individual rationality, incentive compatibility, non-wastefulness and social welfare preservation.

----

## [586] The Strange Role of Information Asymmetry in Auctions - Does More Accurate Value Estimation Benefit a Bidder?

**Authors**: *Haifeng Xu, Ruggiero Cavallo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20459](https://doi.org/10.1609/aaai.v36i5.20459)

**Abstract**:

We study the second-price auction in which bidders have asymmetric information regarding the item’s value. Each bidder’s value for the item depends on a private component and a public component. While each bidder observes their own private component, they hold different and asymmetric information about the public component. We characterize the equilibrium of this auction game and study how the asymmetric bidder information affects their equilibrium bidding strategies. We also discover multiple surprisingly counter-intuitive equilibrium phenomena. For instance, a bidder may be better off if she is less informed regarding the public component. Conversely, a bidder may sometimes be worse off if she obtains more accurate estimation about the auctioned item. Our results suggest that efforts devoted by bidders to improve their value estimations, as widely seen in today’s online advertising auctions, may not always be to their benefit.

----

## [587] AutoCFR: Learning to Design Counterfactual Regret Minimization Algorithms

**Authors**: *Hang Xu, Kai Li, Haobo Fu, Qiang Fu, Junliang Xing*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20460](https://doi.org/10.1609/aaai.v36i5.20460)

**Abstract**:

Counterfactual regret minimization (CFR) is the most commonly used algorithm to approximately solving two-player zero-sum imperfect-information games (IIGs). In recent years, a series of novel CFR variants such as CFR+, Linear CFR, DCFR have been proposed and have significantly improved the convergence rate of the vanilla CFR. However, most of these new variants are hand-designed by researchers through trial and error based on different motivations, which generally requires a tremendous amount of efforts and insights. This work proposes to meta-learn novel CFR algorithms through evolution to ease the burden of manual algorithm design. We first design a search language that is rich enough to represent many existing hand-designed CFR variants. We then exploit a scalable regularized evolution algorithm with a bag of acceleration techniques to efficiently search over the combinatorial space of algorithms defined by this language. The learned novel CFR algorithm can generalize to new IIGs not seen during training and performs on par with or better than existing state-of-the-art CFR variants. The code is available at https://github.com/rpSebastian/AutoCFR.

----

## [588] Team Correlated Equilibria in Zero-Sum Extensive-Form Games via Tree Decompositions

**Authors**: *Brian Hu Zhang, Tuomas Sandholm*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20461](https://doi.org/10.1609/aaai.v36i5.20461)

**Abstract**:

Despite the many recent practical and theoretical breakthroughs in computational game theory, equilibrium finding in extensive-form team games remains a significant challenge. While NP-hard in the worst case, there are provably efficient algorithms for certain families of team game. In particular, if the game has common external information, also known as A-loss recall---informally, actions played by non-team members (i.e., the opposing team or nature) are either unknown to the entire team, or common knowledge within the team---then polynomial-time algorithms exist. In this paper, we devise a completely new algorithm for solving team games. It uses a tree decomposition of the constraint system representing each team's strategy to reduce the number and degree of constraints required for correctness (tightness of the mathematical program). Our approach has the bags of the tree decomposition correspond to team-public states---that is, minimal sets of nodes (that is, states of the team) such that, upon reaching the set, it is common knowledge among the players on the team that the set has been reached. Our algorithm reduces the problem of solving team games to a linear program with at most O(NW^(w+1)) nonzero entries in the constraint matrix, where N is the size of the game tree, w is a parameter that depends on the amount of uncommon external information, and W is the treewidth of the tree decomposition. In public-action games, our program size is bounded by the tighter 2^(O(nt))N for teams of n players with t types each. Our algorithm is based on a new way to write a custom, concise tree decomposition, and its fast run time does not assume that the decomposition has small treewidth. Since our algorithm describes the polytope of correlated strategies directly, we get equilibrium finding in correlated strategies for free---instead of, say, having to run a double oracle algorithm. We show via experiments on a standard suite of games that our algorithm achieves state-of-the-art performance on all benchmark game classes except one. We also present, to our knowledge, the first experiments for this setting where both teams have more than one member.

----

## [589] Planning with Participation Constraints

**Authors**: *Hanrui Zhang, Yu Cheng, Vincent Conitzer*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20462](https://doi.org/10.1609/aaai.v36i5.20462)

**Abstract**:

We pose and study the problem of planning in Markov decision processes (MDPs), subject to participation constraints as studied in mechanism design. In this problem, a planner must work with a self-interested agent on a given MDP. Each action in the MDP provides an immediate reward to the planner and a (possibly different) reward to the agent. The agent has no control in choosing the actions, but has the option to end the entire process at any time. The goal of the planner is to find a policy that maximizes her cumulative reward, taking into consideration the agent's ability to terminate.
    
We give a fully polynomial-time approximation scheme for this problem.  En route, we present polynomial-time algorithms for computing (exact) optimal policies for important special cases of this problem, including when the time horizon is constant, or when the MDP exhibits a "definitive decisions" property. We illustrate our algorithms with two different game-theoretic applications: the problem of assigning rides in ride-sharing and the problem of designing screening policies. Our results imply efficient algorithms for computing (approximately) optimal policies in both applications.

----

## [590] "I Don't Think So": Summarizing Policy Disagreements for Agent Comparison

**Authors**: *Yotam Amitai, Ofra Amir*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20463](https://doi.org/10.1609/aaai.v36i5.20463)

**Abstract**:

With Artificial Intelligence on the rise, human interaction with autonomous agents becomes more frequent. Effective human-agent collaboration requires users to understand the agent's behavior, as failing to do so may cause reduced productivity, misuse or frustration. Agent strategy summarization methods are used to describe the strategy of an agent to users through demonstrations. A summary's objective is to maximize the user's understanding of the agent's aptitude by showcasing its behaviour in a selected set of world states. While shown to be useful, we show that current methods are limited when tasked with comparing between agents, as each summary is independently generated for a specific agent. In this paper, we propose a novel method for generating dependent and contrastive summaries that emphasize the differences between agent policies by identifying states in which the agents disagree on the best course of action. We conducted user studies to assess the usefulness of disagreement-based summaries for identifying superior agents and conveying agent differences. Results show disagreement-based summaries lead to improved user performance compared to summaries generated using HIGHLIGHTS, a strategy summarization algorithm which generates summaries for each agent independently.

----

## [591] Explain, Edit, and Understand: Rethinking User Study Design for Evaluating Model Explanations

**Authors**: *Siddhant Arora, Danish Pruthi, Norman M. Sadeh, William W. Cohen, Zachary C. Lipton, Graham Neubig*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20464](https://doi.org/10.1609/aaai.v36i5.20464)

**Abstract**:

In attempts to "explain" predictions of machine learning models, researchers have proposed hundreds of techniques for attributing predictions to features that are deemed important. While these attributions are often claimed to hold the potential to improve human "understanding" of the models, surprisingly little work explicitly evaluates progress towards this aspiration. In this paper, we conduct a crowdsourcing study, where participants interact with deception detection models that have been trained to distinguish between genuine and fake hotel reviews. They are challenged both to simulate the model on fresh reviews, and to edit reviews with the goal of lowering the probability of the originally predicted class. Successful manipulations would lead to an adversarial example. During the training (but not the test) phase, input spans are highlighted to communicate salience. Through our evaluation, we observe that for a linear bag-of-words model, participants with access to the feature coefficients during training are able to cause a larger reduction in model confidence in the testing phase when compared to the no-explanation control. For the BERT-based classifier, popular local explanations do not improve their ability to reduce the model confidence over the no-explanation case. Remarkably, when the explanation for the BERT model is given by the (global) attributions of a linear model trained to imitate the BERT model, people can effectively manipulate the model.

----

## [592] Role of Human-AI Interaction in Selective Prediction

**Authors**: *Elizabeth Bondi, Raphael Koster, Hannah Sheahan, Martin J. Chadwick, Yoram Bachrach, A. Taylan Cemgil, Ulrich Paquet, Krishnamurthy Dvijotham*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20465](https://doi.org/10.1609/aaai.v36i5.20465)

**Abstract**:

Recent work has shown the potential benefit of  selective prediction systems that can learn to defer to a human when the predictions of the AI are unreliable, particularly to improve the reliability of AI systems in high-stakes applications like healthcare or conservation. However, most prior work assumes that human behavior remains unchanged when they solve a prediction task as part of a human-AI team as opposed to by themselves. We show that this is not the case by performing experiments to quantify human-AI interaction in the context of selective prediction. In particular, we study the impact of communicating different types of information to humans about the AI system's decision to defer. Using real-world conservation data and a selective prediction system that improves expected accuracy over that of the human or AI system working individually, we show that this messaging has a significant impact on the accuracy of human judgements. Our results study two components of the messaging strategy: 1) Whether humans are informed about the prediction of the AI system and 2) Whether they are informed about the decision of the selective prediction system to defer. By manipulating these messaging components, we show that it is possible to significantly boost human performance by informing the human of the decision to defer, but not revealing the  prediction of the AI. We therefore show that it is vital to consider how the decision to defer is communicated to a human when designing selective prediction systems, and that the composite accuracy of a human-AI team must be carefully evaluated using a human-in-the-loop framework.

----

## [593] How General-Purpose Is a Language Model? Usefulness and Safety with Human Prompters in the Wild

**Authors**: *Pablo Antonio Moreno Casares, Bao Sheng Loe, John Burden, Seán Ó hÉigeartaigh, José Hernández-Orallo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20466](https://doi.org/10.1609/aaai.v36i5.20466)

**Abstract**:

The new generation of language models is reported to solve some extraordinary tasks the models were never trained for specifically, in few-shot or zero-shot settings. However, these reports usually cherry-pick the tasks, use the best prompts, and unwrap or extract the solutions leniently even if they are followed by nonsensical text. In sum, they are specialised results for one domain, a particular way of using the models and interpreting the results. In this paper, we present a novel theoretical evaluation framework and a distinctive experimental study assessing language models as general-purpose systems when used directly by human prompters --- in the wild. For a useful and safe interaction in these increasingly more common conditions, we need to understand when the model fails because of a lack of capability or a misunderstanding of the user's intents. Our results indicate that language models such as GPT-3 have limited understanding of the human command; far from becoming general-purpose systems in the wild.

----

## [594] Adversarial Learning from Crowds

**Authors**: *Pengpeng Chen, Hailong Sun, Yongqiang Yang, Zhijun Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20467](https://doi.org/10.1609/aaai.v36i5.20467)

**Abstract**:

Learning from Crowds (LFC) seeks to induce a high-quality classifier from training instances, which are linked to a range of possible noisy annotations from crowdsourcing workers under their various levels of skills and their own preconditions. Recent studies on LFC focus on designing new methods to improve the performance of the classifier trained from crowdsourced labeled data.
 To this day, however, there remain under-explored security aspects of LFC systems. In this work, we seek to bridge this gap. We first show that LFC models are vulnerable to adversarial examples---small changes to input data can cause classifiers to make prediction mistakes. Second, we propose an approach, A-LFC for training a robust classifier from crowdsourced labeled data. Our empirical results on three real-world datasets show that the proposed approach can substantially improve the performance of the trained classifier even with the existence of adversarial examples. On average, A-LFC has 10.05% and 11.34% higher test robustness than the state-of-the-art in the white-box and black-box attack settings, respectively.

----

## [595] FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles

**Authors**: *Ana Lucic, Harrie Oosterhuis, Hinda Haned, Maarten de Rijke*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20468](https://doi.org/10.1609/aaai.v36i5.20468)

**Abstract**:

Model interpretability has become an important problem in machine learning (ML) due to the increased effect algorithmic decisions have on humans. 
 Counterfactual explanations can help users understand not only why ML models make certain decisions, but also how these decisions can be changed. 
 We frame the problem of finding counterfactual explanations as an optimization task and extend previous work that could only be applied to differentiable models. 
 In order to accommodate non-differentiable models such as tree ensembles, we use probabilistic model approximations in the optimization framework.
 We introduce an approximation technique that is effective for finding counterfactual explanations for predictions of the original model and show that our counterfactual examples are significantly closer to the original instances than those produced by other methods specifically designed for tree ensembles.

----

## [596] Teaching Humans When to Defer to a Classifier via Exemplars

**Authors**: *Hussein Mozannar, Arvind Satyanarayan, David A. Sontag*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20469](https://doi.org/10.1609/aaai.v36i5.20469)

**Abstract**:

Expert decision makers are starting to rely on data-driven automated agents to assist them with various tasks. For this collaboration to perform properly, the human decision maker must have a mental model of when and when not to rely on the agent. In this work, we aim to ensure that human decision makers learn a valid mental model of the agent's strengths and weaknesses. To accomplish this goal, we propose an exemplar-based teaching strategy where humans solve a set of selected examples and with our help generalize from them to the domain. We present a novel parameterization of the human's mental model of the AI that applies a nearest neighbor rule in local regions surrounding the teaching examples. Using this model, we derive a near-optimal strategy for selecting a representative teaching set. We validate the benefits of our teaching strategy on a multi-hop question answering task with an interpretable AI model using crowd workers. We find that when workers draw the right lessons from the teaching stage, their task performance improves. We furthermore validate our method on a set of synthetic experiments.

----

## [597] Deceptive Decision-Making under Uncertainty

**Authors**: *Yagiz Savas, Christos K. Verginis, Ufuk Topcu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20470](https://doi.org/10.1609/aaai.v36i5.20470)

**Abstract**:

We study the design of autonomous agents that are capable of deceiving outside observers about their intentions while carrying out tasks in stochastic, complex environments. By modeling the agent's behavior as a Markov decision process, we consider a setting where the agent aims to reach one of multiple potential goals while deceiving outside observers about its true goal. We propose a novel approach to model observer predictions based on the principle of maximum entropy and to efficiently generate deceptive strategies via linear programming. The proposed approach enables the agent to exhibit a variety of tunable deceptive behaviors while ensuring the satisfaction of probabilistic constraints on the behavior. We evaluate the performance of the proposed approach via comparative user studies and present a case study on the streets of Manhattan, New York, using real travel time distributions.

----

## [598] On Optimizing Interventions in Shared Autonomy

**Authors**: *Weihao Tan, David Koleczek, Siddhant Pradhan, Nicholas Perello, Vivek Chettiar, Vishal Rohra, Aaslesha Rajaram, Soundararajan Srinivasan, H. M. Sajjad Hossain, Yash Chandak*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20471](https://doi.org/10.1609/aaai.v36i5.20471)

**Abstract**:

Shared autonomy refers to approaches for enabling an autonomous agent to collaborate with a human with the aim of improving human performance. However, besides improving performance, it may often also be beneficial that the agent concurrently accounts for preserving the user’s experience or satisfaction of collaboration. In order to address this additional goal, we examine approaches for improving the user experience by constraining the number of interventions by the autonomous agent. We propose two model-free reinforcement learning methods that can account for both hard and soft constraints on the number of interventions. We show that not only does our method outperform the existing baseline, but also eliminates the need to manually tune a black-box hyperparameter for controlling the level of assistance. We also provide an in-depth analysis of intervention scenarios in order to further illuminate system understanding.

----

## [599] Open Vocabulary Electroencephalography-to-Text Decoding and Zero-Shot Sentiment Classification

**Authors**: *Zhenhailong Wang, Heng Ji*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20472](https://doi.org/10.1609/aaai.v36i5.20472)

**Abstract**:

State-of-the-art brain-to-text systems have achieved great success in decoding language directly from brain signals using neural networks. However, current approaches are limited to small closed vocabularies which are far from enough for natural communication. In addition, most of the high-performing approaches require data from invasive devices (e.g., ECoG). In this paper, we extend the problem to open vocabulary Electroencephalography(EEG)-To-Text Sequence-To-Sequence decoding and zero-shot sentence sentiment classification on natural reading tasks. We hypothesis that the human brain functions as a special text encoder and propose a novel framework leveraging pre-trained language models (e.g., BART). Our model achieves a 40.1% BLEU-1 score on EEG-To-Text decoding and a 55.6% F1 score on zero-shot EEG-based ternary sentiment classification, which significantly outperforms supervised baselines. Furthermore, we show that our proposed model can handle data from various subjects and sources, showing great potential for a high-performance open vocabulary brain-to-text system once sufficient data is available. The code is made publicly available for research purpose at https://github.com/MikeWangWZHL/EEG-To-Text.

----



[Go to the previous page](AAAI-2022-list02.md)

[Go to the next page](AAAI-2022-list04.md)

[Go to the catalog section](README.md)