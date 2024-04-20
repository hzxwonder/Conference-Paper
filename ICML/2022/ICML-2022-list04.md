## [600] Constrained Variational Policy Optimization for Safe Reinforcement Learning

**Authors**: *Zuxin Liu, Zhepeng Cen, Vladislav Isenbaev, Wei Liu, Zhiwei Steven Wu, Bo Li, Ding Zhao*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22b.html](https://proceedings.mlr.press/v162/liu22b.html)

**Abstract**:

Safe reinforcement learning (RL) aims to learn policies that satisfy certain constraints before deploying them to safety-critical applications. Previous primal-dual style approaches suffer from instability issues and lack optimality guarantees. This paper overcomes the issues from the perspective of probabilistic inference. We introduce a novel Expectation-Maximization approach to naturally incorporate constraints during the policy learning: 1) a provable optimal non-parametric variational distribution could be computed in closed form after a convex optimization (E-step); 2) the policy parameter is improved within the trust region based on the optimal variational distribution (M-step). The proposed algorithm decomposes the safe RL problem into a convex optimization phase and a supervised learning phase, which yields a more stable training performance. A wide range of experiments on continuous robotic tasks shows that the proposed method achieves significantly better constraint satisfaction performance and better sample efficiency than baselines. The code is available at https://github.com/liuzuxin/cvpo-safe-rl.

----

## [601] Benefits of Overparameterized Convolutional Residual Networks: Function Approximation under Smoothness Constraint

**Authors**: *Hao Liu, Minshuo Chen, Siawpeng Er, Wenjing Liao, Tong Zhang, Tuo Zhao*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22c.html](https://proceedings.mlr.press/v162/liu22c.html)

**Abstract**:

Overparameterized neural networks enjoy great representation power on complex data, and more importantly yield sufficiently smooth output, which is crucial to their generalization and robustness. Most existing function approximation theories suggest that with sufficiently many parameters, neural networks can well approximate certain classes of functions in terms of the function value. The neural network themselves, however, can be highly nonsmooth. To bridge this gap, we take convolutional residual networks (ConvResNets) as an example, and prove that large ConvResNets can not only approximate a target function in terms of function value, but also exhibit sufficient first-order smoothness. Moreover, we extend our theory to approximating functions supported on a low-dimensional manifold. Our theory partially justifies the benefits of using deep and wide networks in practice. Numerical experiments on adversarial robust image classification are provided to support our theory.

----

## [602] Boosting Graph Structure Learning with Dummy Nodes

**Authors**: *Xin Liu, Jiayang Cheng, Yangqiu Song, Xin Jiang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22d.html](https://proceedings.mlr.press/v162/liu22d.html)

**Abstract**:

With the development of graph kernels and graph representation learning, many superior methods have been proposed to handle scalability and oversmoothing issues on graph structure learning. However, most of those strategies are designed based on practical experience rather than theoretical analysis. In this paper, we use a particular dummy node connecting to all existing vertices without affecting original vertex and edge properties. We further prove that such the dummy node can help build an efficient monomorphic edge-to-vertex transform and an epimorphic inverse to recover the original graph back. It also indicates that adding dummy nodes can preserve local and global structures for better graph representation learning. We extend graph kernels and graph neural networks with dummy nodes and conduct experiments on graph classification and subgraph isomorphism matching tasks. Empirical results demonstrate that taking graphs with dummy nodes as input significantly boosts graph structure learning, and using their edge-to-vertex graphs can also achieve similar results. We also discuss the gain of expressive power from the dummy in neural networks.

----

## [603] Equivalence Analysis between Counterfactual Regret Minimization and Online Mirror Descent

**Authors**: *Weiming Liu, Huacong Jiang, Bin Li, Houqiang Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22e.html](https://proceedings.mlr.press/v162/liu22e.html)

**Abstract**:

Follow-the-Regularized-Leader (FTRL) and Online Mirror Descent (OMD) are regret minimization algorithms for Online Convex Optimization (OCO), they are mathematically elegant but less practical in solving Extensive-Form Games (EFGs). Counterfactual Regret Minimization (CFR) is a technique for approximating Nash equilibria in EFGs. CFR and its variants have a fast convergence rate in practice, but their theoretical results are not satisfactory. In recent years, researchers have been trying to link CFRs with OCO algorithms, which may provide new theoretical results and inspire new algorithms. However, existing analysis is restricted to local decision points. In this paper, we show that CFRs with Regret Matching and Regret Matching+ are equivalent to special cases of FTRL and OMD, respectively. According to these equivalences, a new FTRL and a new OMD algorithm, which can be considered as extensions of vanilla CFR and CFR+, are derived. The experimental results show that the two variants converge faster than conventional FTRL and OMD, even faster than vanilla CFR and CFR+ in some EFGs.

----

## [604] Deep Probability Estimation

**Authors**: *Sheng Liu, Aakash Kaku, Weicheng Zhu, Matan Leibovich, Sreyas Mohan, Boyang Yu, Haoxiang Huang, Laure Zanna, Narges Razavian, Jonathan Niles-Weed, Carlos Fernandez-Granda*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22f.html](https://proceedings.mlr.press/v162/liu22f.html)

**Abstract**:

Reliable probability estimation is of crucial importance in many real-world applications where there is inherent (aleatoric) uncertainty. Probability-estimation models are trained on observed outcomes (e.g. whether it has rained or not, or whether a patient has died or not), because the ground-truth probabilities of the events of interest are typically unknown. The problem is therefore analogous to binary classification, with the difference that the objective is to estimate probabilities rather than predicting the specific outcome. This work investigates probability estimation from high-dimensional data using deep neural networks. There exist several methods to improve the probabilities generated by these models but they mostly focus on model (epistemic) uncertainty. For problems with inherent uncertainty, it is challenging to evaluate performance without access to ground-truth probabilities. To address this, we build a synthetic dataset to study and compare different computable metrics. We evaluate existing methods on the synthetic data as well as on three real-world probability estimation tasks, all of which involve inherent uncertainty: precipitation forecasting from radar images, predicting cancer patient survival from histopathology images, and predicting car crashes from dashcam videos. We also give a theoretical analysis of a model for high-dimensional probability estimation which reproduces several of the phenomena evinced in our experiments. Finally, we propose a new method for probability estimation using neural networks, which modifies the training process to promote output probabilities that are consistent with empirical probabilities computed from the data. The method outperforms existing approaches on most metrics on the simulated as well as real-world data.

----

## [605] Gating Dropout: Communication-efficient Regularization for Sparsely Activated Transformers

**Authors**: *Rui Liu, Young Jin Kim, Alexandre Muzio, Hany Hassan*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22g.html](https://proceedings.mlr.press/v162/liu22g.html)

**Abstract**:

Sparsely activated transformers, such as Mixture of Experts (MoE), have received great interest due to their outrageous scaling capability which enables dramatical increases in model size without significant increases in computational cost. To achieve this, MoE models replace the feedforward sub-layer with Mixture-of-Experts sub-layer in transformers and use a gating network to route each token to its assigned experts. Since the common practice for efficient training of such models requires distributing experts and tokens across different machines, this routing strategy often incurs huge cross-machine communication cost because tokens and their assigned experts likely reside in different machines. In this paper, we propose Gating Dropout, which allows tokens to ignore the gating network and stay at their local machines, thus reducing the cross-machine communication. Similar to traditional dropout, we also show that Gating Dropout has a regularization effect during training, resulting in improved generalization performance. We validate the effectiveness of Gating Dropout on multilingual machine translation tasks. Our results demonstrate that Gating Dropout improves a state-of-the-art MoE model with faster wall-clock time convergence rates and better BLEU scores for a variety of model sizes and datasets.

----

## [606] Simplex Neural Population Learning: Any-Mixture Bayes-Optimality in Symmetric Zero-sum Games

**Authors**: *Siqi Liu, Marc Lanctot, Luke Marris, Nicolas Heess*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22h.html](https://proceedings.mlr.press/v162/liu22h.html)

**Abstract**:

Learning to play optimally against any mixture over a diverse set of strategies is of important practical interests in competitive games. In this paper, we propose simplex-NeuPL that satisfies two desiderata simultaneously: i) learning a population of strategically diverse basis policies, represented by a single conditional network; ii) using the same network, learn best-responses to any mixture over the simplex of basis policies. We show that the resulting conditional policies incorporate prior information about their opponents effectively, enabling near optimal returns against arbitrary mixture policies in a game with tractable best-responses. We verify that such policies behave Bayes-optimally under uncertainty and offer insights in using this flexibility at test time. Finally, we offer evidence that learning best-responses to any mixture policies is an effective auxiliary task for strategic exploration, which, by itself, can lead to more performant populations.

----

## [607] Rethinking Attention-Model Explainability through Faithfulness Violation Test

**Authors**: *Yibing Liu, Haoliang Li, Yangyang Guo, Chenqi Kong, Jing Li, Shiqi Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22i.html](https://proceedings.mlr.press/v162/liu22i.html)

**Abstract**:

Attention mechanisms are dominating the explainability of deep models. They produce probability distributions over the input, which are widely deemed as feature-importance indicators. However, in this paper, we find one critical limitation in attention explanations: weakness in identifying the polarity of feature impact. This would be somehow misleading – features with higher attention weights may not faithfully contribute to model predictions; instead, they can impose suppression effects. With this finding, we reflect on the explainability of current attention-based techniques, such as Attention $\bigodot$ Gradient and LRP-based attention explanations. We first propose an actionable diagnostic methodology (henceforth faithfulness violation test) to measure the consistency between explanation weights and the impact polarity. Through the extensive experiments, we then show that most tested explanation methods are unexpectedly hindered by the faithfulness violation issue, especially the raw attention. Empirical analyses on the factors affecting violation issues further provide useful observations for adopting explanation methods in attention models.

----

## [608] Optimization-Derived Learning with Essential Convergence Analysis of Training and Hyper-training

**Authors**: *Risheng Liu, Xuan Liu, Shangzhi Zeng, Jin Zhang, Yixuan Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22j.html](https://proceedings.mlr.press/v162/liu22j.html)

**Abstract**:

Recently, Optimization-Derived Learning (ODL) has attracted attention from learning and vision areas, which designs learning models from the perspective of optimization. However, previous ODL approaches regard the training and hyper-training procedures as two separated stages, meaning that the hyper-training variables have to be fixed during the training process, and thus it is also impossible to simultaneously obtain the convergence of training and hyper-training variables. In this work, we design a Generalized Krasnoselskii-Mann (GKM) scheme based on fixed-point iterations as our fundamental ODL module, which unifies existing ODL methods as special cases. Under the GKM scheme, a Bilevel Meta Optimization (BMO) algorithmic framework is constructed to solve the optimal training and hyper-training variables together. We rigorously prove the essential joint convergence of the fixed-point iteration for training and the process of optimizing hyper-parameters for hyper-training, both on the approximation quality, and on the stationary analysis. Experiments demonstrate the efficiency of BMO with competitive performance on sparse coding and real-world applications such as image deconvolution and rain streak removal.

----

## [609] Deep Neural Network Fusion via Graph Matching with Applications to Model Ensemble and Federated Learning

**Authors**: *Chang Liu, Chenfei Lou, Runzhong Wang, Alan Yuhan Xi, Li Shen, Junchi Yan*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22k.html](https://proceedings.mlr.press/v162/liu22k.html)

**Abstract**:

Model fusion without accessing training data in machine learning has attracted increasing interest due to the practical resource-saving and data privacy issues. During the training process, the neural weights of each model can be randomly permuted, and we have to align the channels of each layer before fusing them. Regrading the channels as nodes and weights as edges, aligning the channels to maximize weight similarity is a challenging NP-hard assignment problem. Due to its quadratic assignment nature, we formulate the model fusion problem as a graph matching task, considering the second-order similarity of model weights instead of previous work merely formulating model fusion as a linear assignment problem. For the rising problem scale and multi-model consistency issues, we propose an efficient graduated assignment-based model fusion method, dubbed GAMF, which iteratively updates the matchings in a consistency-maintaining manner. We apply GAMF to tackle the compact model ensemble task and federated learning task on MNIST, CIFAR-10, CIFAR-100, and Tiny-Imagenet. The performance shows the efficacy of our GAMF compared to state-of-the-art baselines.

----

## [610] Welfare Maximization in Competitive Equilibrium: Reinforcement Learning for Markov Exchange Economy

**Authors**: *Zhihan Liu, Miao Lu, Zhaoran Wang, Michael I. Jordan, Zhuoran Yang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22l.html](https://proceedings.mlr.press/v162/liu22l.html)

**Abstract**:

We study a bilevel economic system, which we refer to as a Markov exchange economy (MEE), from the point of view of multi-agent reinforcement learning (MARL). An MEE involves a central planner and a group of self-interested agents. The goal of the agents is to form a Competitive Equilibrium (CE), where each agent myopically maximizes her own utility at each step. The goal of the central planner is to steer the system so as to maximize social welfare, which is defined as the sum of the utilities of all agents. Working in a setting in which the utility function and the system dynamics are both unknown, we propose to find the socially optimal policy and the CE from data via both online and offline variants of MARL. Concretely, we first devise a novel suboptimality metric specifically tailored to MEE, such that minimizing such a metric certifies globally optimal policies for both the planner and the agents. Second, in the online setting, we propose an algorithm, dubbed as \texttt{MOLM}, which combines the optimism principle for exploration with subgame CE seeking. Our algorithm can readily incorporate general function approximation tools for handling large state spaces and achieves a sublinear regret. Finally, we adapt the algorithm to an offline setting based on the pessimism principle and establish an upper bound on the suboptimality.

----

## [611] Generating 3D Molecules for Target Protein Binding

**Authors**: *Meng Liu, Youzhi Luo, Kanji Uchino, Koji Maruhashi, Shuiwang Ji*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22m.html](https://proceedings.mlr.press/v162/liu22m.html)

**Abstract**:

A fundamental problem in drug discovery is to design molecules that bind to specific proteins. To tackle this problem using machine learning methods, here we propose a novel and effective framework, known as GraphBP, to generate 3D molecules that bind to given proteins by placing atoms of specific types and locations to the given binding site one by one. In particular, at each step, we first employ a 3D graph neural network to obtain geometry-aware and chemically informative representations from the intermediate contextual information. Such context includes the given binding site and atoms placed in the previous steps. Second, to preserve the desirable equivariance property, we select a local reference atom according to the designed auxiliary classifiers and then construct a local spherical coordinate system. Finally, to place a new atom, we generate its atom type and relative location w.r.t. the constructed local coordinate system via a flow model. We also consider generating the variables of interest sequentially to capture the underlying dependencies among them. Experiments demonstrate that our GraphBP is effective to generate 3D molecules with binding ability to target protein binding sites. Our implementation is available at https://github.com/divelab/GraphBP.

----

## [612] Communication-efficient Distributed Learning for Large Batch Optimization

**Authors**: *Rui Liu, Barzan Mozafari*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22n.html](https://proceedings.mlr.press/v162/liu22n.html)

**Abstract**:

Many communication-efficient methods have been proposed for distributed learning, whereby gradient compression is used to reduce the communication cost. However, given recent advances in large batch optimization (e.g., large batch SGD and its variant LARS with layerwise adaptive learning rates), the compute power of each machine is being fully utilized. This means, in modern distributed learning, the per-machine computation cost is no longer negligible compared to the communication cost. In this paper, we propose new gradient compression methods for large batch optimization, JointSpar and its variant JointSpar-LARS with layerwise adaptive learning rates, that jointly reduce both the computation and the communication cost. To achieve this, we take advantage of the redundancy in the gradient computation, unlike the existing methods compute all coordinates of the gradient vector, even if some coordinates are later dropped for communication efficiency. JointSpar and its variant further reduce the training time by avoiding the wasted computation on dropped coordinates. While computationally more efficient, we prove that JointSpar and its variant also maintain the same convergence rates as their respective baseline methods. Extensive experiments show that, by reducing the time per iteration, our methods converge faster than state-of-the-art compression methods in terms of wall-clock time.

----

## [613] Adaptive Accelerated (Extra-)Gradient Methods with Variance Reduction

**Authors**: *Zijian Liu, Ta Duy Nguyen, Alina Ene, Huy L. Nguyen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22o.html](https://proceedings.mlr.press/v162/liu22o.html)

**Abstract**:

In this paper, we study the finite-sum convex optimization problem focusing on the general convex case. Recently, the study of variance reduced (VR) methods and their accelerated variants has made exciting progress. However, the step size used in the existing VR algorithms typically depends on the smoothness parameter, which is often unknown and requires tuning in practice. To address this problem, we propose two novel adaptive VR algorithms: Adaptive Variance Reduced Accelerated Extra-Gradient (AdaVRAE) and Adaptive Variance Reduced Accelerated Gradient (AdaVRAG). Our algorithms do not require knowledge of the smoothness parameter. AdaVRAE uses $\mathcal{O}\left(n\log\log n+\sqrt{\frac{n\beta}{\epsilon}}\right)$ and AdaVRAG uses $\mathcal{O}\left(n\log\log n+\sqrt{\frac{n\beta\log\beta}{\epsilon}}\right)$ gradient evaluations to attain an $\mathcal{O}(\epsilon)$-suboptimal solution, where $n$ is the number of functions in the finite sum and $\beta$ is the smoothness parameter. This result matches the best-known convergence rate of non-adaptive VR methods and it improves upon the convergence of the state of the art adaptive VR method, AdaSVRG. We demonstrate the superior performance of our algorithms compared with previous methods in experiments on real-world datasets.

----

## [614] REvolveR: Continuous Evolutionary Models for Robot-to-robot Policy Transfer

**Authors**: *Xingyu Liu, Deepak Pathak, Kris Kitani*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22p.html](https://proceedings.mlr.press/v162/liu22p.html)

**Abstract**:

A popular paradigm in robotic learning is to train a policy from scratch for every new robot. This is not only inefficient but also often impractical for complex robots. In this work, we consider the problem of transferring a policy across two different robots with significantly different parameters such as kinematics and morphology. Existing approaches that train a new policy by matching the action or state transition distribution, including imitation learning methods, fail due to optimal action and/or state distribution being mismatched in different robots. In this paper, we propose a novel method named REvolveR of using continuous evolutionary models for robotic policy transfer implemented in a physics simulator. We interpolate between the source robot and the target robot by finding a continuous evolutionary change of robot parameters. An expert policy on the source robot is transferred through training on a sequence of intermediate robots that gradually evolve into the target robot. Experiments on a physics simulator show that the proposed continuous evolutionary model can effectively transfer the policy across robots and achieve superior sample efficiency on new robots. The proposed method is especially advantageous in sparse reward settings where exploration can be significantly reduced.

----

## [615] Kill a Bird with Two Stones: Closing the Convergence Gaps in Non-Strongly Convex Optimization by Directly Accelerated SVRG with Double Compensation and Snapshots

**Authors**: *Yuanyuan Liu, Fanhua Shang, Weixin An, Hongying Liu, Zhouchen Lin*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22q.html](https://proceedings.mlr.press/v162/liu22q.html)

**Abstract**:

Recently, some accelerated stochastic variance reduction algorithms such as Katyusha and ASVRG-ADMM achieve faster convergence than non-accelerated methods such as SVRG and SVRG-ADMM. However, there are still some gaps between the oracle complexities and their lower bounds. To fill in these gaps, this paper proposes a novel Directly Accelerated stochastic Variance reductIon (DAVIS) algorithm with two Snapshots for non-strongly convex (non-SC) unconstrained problems. Our theoretical results show that DAVIS achieves the optimal convergence rate O(1/(nS^2)) and optimal gradient complexity O(n+\sqrt{nL/\epsilon}), which is identical to its lower bound. To the best of our knowledge, this is the first directly accelerated algorithm that attains the optimal lower bound and improves the convergence rate from O(1/S^2) to O(1/(nS^2)). Moreover, we extend DAVIS and theoretical results to non-SC problems with a structured regularizer, and prove that the proposed algorithm with double-snapshots also attains the optimal convergence rate O(1/(nS)) and optimal oracle complexity O(n+L/\epsilon) for such problems, and it is at least a factor n/S faster than existing accelerated stochastic algorithms, where n\gg S in general.

----

## [616] Learning Markov Games with Adversarial Opponents: Efficient Algorithms and Fundamental Limits

**Authors**: *Qinghua Liu, Yuanhao Wang, Chi Jin*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22r.html](https://proceedings.mlr.press/v162/liu22r.html)

**Abstract**:

An ideal strategy in zero-sum games should not only grant the player an average reward no less than the value of Nash equilibrium, but also exploit the (adaptive) opponents when they are suboptimal. While most existing works in Markov games focus exclusively on the former objective, it remains open whether we can achieve both objectives simultaneously. To address this problem, this work studies no-regret learning in Markov games with adversarial opponents when competing against the best fixed policy in hindsight. Along this direction, we present a new complete set of positive and negative results: When the policies of the opponents are revealed at the end of each episode, we propose new efficient algorithms achieving $\sqrt{K}$ regret bounds when either (1) the baseline policy class is small or (2) the opponent’s policy class is small. This is complemented with an exponential lower bound when neither conditions are true. When the policies of the opponents are not revealed, we prove a statistical hardness result even in the most favorable scenario when both above conditions are true. Our hardness result is much stronger than the existing hardness results which either only involve computational hardness, or require further restrictions on the algorithms.

----

## [617] Local Augmentation for Graph Neural Networks

**Authors**: *Songtao Liu, Rex Ying, Hanze Dong, Lanqing Li, Tingyang Xu, Yu Rong, Peilin Zhao, Junzhou Huang, Dinghao Wu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22s.html](https://proceedings.mlr.press/v162/liu22s.html)

**Abstract**:

Graph Neural Networks (GNNs) have achieved remarkable performance on graph-based tasks. The key idea for GNNs is to obtain informative representation through aggregating information from local neighborhoods. However, it remains an open question whether the neighborhood information is adequately aggregated for learning representations of nodes with few neighbors. To address this, we propose a simple and efficient data augmentation strategy, local augmentation, to learn the distribution of the node representations of the neighbors conditioned on the central node’s representation and enhance GNN’s expressive power with generated features. Local augmentation is a general framework that can be applied to any GNN model in a plug-and-play manner. It samples feature vectors associated with each node from the learned conditional distribution as additional input for the backbone model at each training iteration. Extensive experiments and analyses show that local augmentation consistently yields performance improvement when applied to various GNN architectures across a diverse set of benchmarks. For example, experiments show that plugging in local augmentation to GCN and GAT improves by an average of 3.4% and 1.6% in terms of test accuracy on Cora, Citeseer, and Pubmed. Besides, our experimental results on large graphs (OGB) show that our model consistently improves performance over backbones. Code is available at https://github.com/SongtaoLiu0823/LAGNN.

----

## [618] Asking for Knowledge (AFK): Training RL Agents to Query External Knowledge Using Language

**Authors**: *Iou-Jen Liu, Xingdi Yuan, Marc-Alexandre Côté, Pierre-Yves Oudeyer, Alexander G. Schwing*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22t.html](https://proceedings.mlr.press/v162/liu22t.html)

**Abstract**:

To solve difficult tasks, humans ask questions to acquire knowledge from external sources. In contrast, classical reinforcement learning agents lack such an ability and often resort to exploratory behavior. This is exacerbated as few present-day environments support querying for knowledge. In order to study how agents can be taught to query external knowledge via language, we first introduce two new environments: the grid-world-based Q-BabyAI and the text-based Q-TextWorld. In addition to physical interactions, an agent can query an external knowledge source specialized for these environments to gather information. Second, we propose the ‘Asking for Knowledge’ (AFK) agent, which learns to generate language commands to query for meaningful knowledge that helps solve the tasks. AFK leverages a non-parametric memory, a pointer mechanism and an episodic exploration bonus to tackle (1) irrelevant information, (2) a large query language space, (3) delayed reward for making meaningful queries. Extensive experiments demonstrate that the AFK agent outperforms recent baselines on the challenging Q-BabyAI and Q-TextWorld environments.

----

## [619] Learning from Demonstration: Provably Efficient Adversarial Policy Imitation with Linear Function Approximation

**Authors**: *Zhihan Liu, Yufeng Zhang, Zuyue Fu, Zhuoran Yang, Zhaoran Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22u.html](https://proceedings.mlr.press/v162/liu22u.html)

**Abstract**:

In generative adversarial imitation learning (GAIL), the agent aims to learn a policy from an expert demonstration so that its performance cannot be discriminated from the expert policy on a certain predefined reward set. In this paper, we study GAIL in both online and offline settings with linear function approximation, where both the transition and reward function are linear in the feature maps. Besides the expert demonstration, in the online setting the agent can interact with the environment, while in the offline setting the agent only accesses an additional dataset collected by a prior. For online GAIL, we propose an optimistic generative adversarial policy imitation algorithm (OGAPI) and prove that OGAPI achieves $\widetilde{\mathcal{O}}(\sqrt{H^4d^3K}+\sqrt{H^3d^2K^2/N_1})$ regret. Here $N_1$ represents the number of trajectories of the expert demonstration, $d$ is the feature dimension, and $K$ is the number of episodes. For offline GAIL, we propose a pessimistic generative adversarial policy imitation algorithm (PGAPI). We also obtain the optimality gap of PGAPI, achieving the minimax lower bound in the utilization of the additional dataset. Assuming sufficient coverage on the additional dataset, we show that PGAPI achieves $\widetilde{\mathcal{O}}(\sqrt{H^4d^2/K}+\sqrt{H^4d^3/N_2}+\sqrt{H^3d^2/N_1})$ optimality gap. Here $N_2$ represents the number of trajectories of the additional dataset with sufficient coverage.

----

## [620] GACT: Activation Compressed Training for Generic Network Architectures

**Authors**: *Xiaoxuan Liu, Lianmin Zheng, Dequan Wang, Yukuo Cen, Weize Chen, Xu Han, Jianfei Chen, Zhiyuan Liu, Jie Tang, Joey Gonzalez, Michael W. Mahoney, Alvin Cheung*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22v.html](https://proceedings.mlr.press/v162/liu22v.html)

**Abstract**:

Training large neural network (NN) models requires extensive memory resources, and Activation Compression Training (ACT) is a promising approach to reduce training memory footprint. This paper presents GACT, an ACT framework to support a broad range of machine learning tasks for generic NN architectures with limited domain knowledge. By analyzing a linearized version of ACT’s approximate gradient, we prove the convergence of GACT without prior knowledge on operator type or model architecture. To make training stable, we propose an algorithm that decides the compression ratio for each tensor by estimating its impact on the gradient at run time. We implement GACT as a PyTorch library that readily applies to any NN architecture. GACT reduces the activation memory for convolutional NNs, transformers, and graph NNs by up to 8.1x, enabling training with a 4.2x to 24.7x larger batch size, with negligible accuracy loss.

----

## [621] Robust Training under Label Noise by Over-parameterization

**Authors**: *Sheng Liu, Zhihui Zhu, Qing Qu, Chong You*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22w.html](https://proceedings.mlr.press/v162/liu22w.html)

**Abstract**:

Recently, over-parameterized deep networks, with increasingly more network parameters than training samples, have dominated the performances of modern machine learning. However, when the training data is corrupted, it has been well-known that over-parameterized networks tend to overfit and do not generalize. In this work, we propose a principled approach for robust training of over-parameterized deep networks in classification tasks where a proportion of training labels are corrupted. The main idea is yet very simple: label noise is sparse and incoherent with the network learned from clean data, so we model the noise and learn to separate it from the data. Specifically, we model the label noise via another sparse over-parameterization term, and exploit implicit algorithmic regularizations to recover and separate the underlying corruptions. Remarkably, when trained using such a simple method in practice, we demonstrate state-of-the-art test accuracy against label noise on a variety of real datasets. Furthermore, our experimental results are corroborated by theory on simplified linear models, showing that exact separation between sparse noise and low-rank data can be achieved under incoherent conditions. The work opens many interesting directions for improving over-parameterized models by using sparse over-parameterization and implicit regularization. Code is available at https://github.com/shengliu66/SOP.

----

## [622] Plan Your Target and Learn Your Skills: Transferable State-Only Imitation Learning via Decoupled Policy Optimization

**Authors**: *Minghuan Liu, Zhengbang Zhu, Yuzheng Zhuang, Weinan Zhang, Jianye Hao, Yong Yu, Jun Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22x.html](https://proceedings.mlr.press/v162/liu22x.html)

**Abstract**:

Recent progress in state-only imitation learning extends the scope of applicability of imitation learning to real-world settings by relieving the need for observing expert actions. However, existing solutions only learn to extract a state-to-action mapping policy from the data, without considering how the expert plans to the target. This hinders the ability to leverage demonstrations and limits the flexibility of the policy. In this paper, we introduce Decoupled Policy Optimization (DePO), which explicitly decouples the policy as a high-level state planner and an inverse dynamics model. With embedded decoupled policy gradient and generative adversarial training, DePO enables knowledge transfer to different action spaces or state transition dynamics, and can generalize the planner to out-of-demonstration state regions. Our in-depth experimental analysis shows the effectiveness of DePO on learning a generalized target state planner while achieving the best imitation performance. We demonstrate the appealing usage of DePO for transferring across different tasks by pre-training, and the potential for co-training agents with various skills.

----

## [623] On the Impossibility of Learning to Cooperate with Adaptive Partner Strategies in Repeated Games

**Authors**: *Robert Tyler Loftin, Frans A. Oliehoek*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/loftin22a.html](https://proceedings.mlr.press/v162/loftin22a.html)

**Abstract**:

Learning to cooperate with other agents is challenging when those agents also possess the ability to adapt to our own behavior. Practical and theoretical approaches to learning in cooperative settings typically assume that other agents’ behaviors are stationary, or else make very specific assumptions about other agents’ learning processes. The goal of this work is to understand whether we can reliably learn to cooperate with other agents without such restrictive assumptions, which are unlikely to hold in real-world applications. Our main contribution is a set of impossibility results, which show that no learning algorithm can reliably learn to cooperate with all possible adaptive partners in a repeated matrix game, even if that partner is guaranteed to cooperate with some stationary strategy. Motivated by these results, we then discuss potential alternative assumptions which capture the idea that an adaptive partner will only adapt rationally to our behavior.

----

## [624] AutoIP: A United Framework to Integrate Physics into Gaussian Processes

**Authors**: *Da Long, Zheng Wang, Aditi S. Krishnapriyan, Robert M. Kirby, Shandian Zhe, Michael W. Mahoney*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/long22a.html](https://proceedings.mlr.press/v162/long22a.html)

**Abstract**:

Physical modeling is critical for many modern science and engineering applications. From a data science or machine learning perspective, where more domain-agnostic, data-driven models are pervasive, physical knowledge {—} often expressed as differential equations {—} is valuable in that it is complementary to data, and it can potentially help overcome issues such as data sparsity, noise, and inaccuracy. In this work, we propose a simple, yet powerful and general framework {—} AutoIP, for Automatically Incorporating Physics {—} that can integrate all kinds of differential equations into Gaussian Processes (GPs) to enhance prediction accuracy and uncertainty quantification. These equations can be linear or nonlinear, spatial, temporal, or spatio-temporal, complete or incomplete with unknown source terms, and so on. Based on kernel differentiation, we construct a GP prior to sample the values of the target function, equation related derivatives, and latent source functions, which are all jointly from a multivariate Gaussian distribution. The sampled values are fed to two likelihoods: one to fit the observations, and the other to conform to the equation. We use the whitening method to evade the strong dependency between the sampled function values and kernel parameters, and we develop a stochastic variational learning algorithm. AutoIP shows improvement upon vanilla GPs in both simulation and several real-world applications, even using rough, incomplete equations.

----

## [625] Bayesian Model Selection, the Marginal Likelihood, and Generalization

**Authors**: *Sanae Lotfi, Pavel Izmailov, Gregory W. Benton, Micah Goldblum, Andrew Gordon Wilson*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lotfi22a.html](https://proceedings.mlr.press/v162/lotfi22a.html)

**Abstract**:

How do we compare between hypotheses that are entirely consistent with observations? The marginal likelihood (aka Bayesian evidence), which represents the probability of generating our observations from a prior, provides a distinctive approach to this foundational question, automatically encoding Occam’s razor. Although it has been observed that the marginal likelihood can overfit and is sensitive to prior assumptions, its limitations for hyperparameter learning and discrete model comparison have not been thoroughly investigated. We first revisit the appealing properties of the marginal likelihood for learning constraints and hypothesis testing. We then highlight the conceptual and practical issues in using the marginal likelihood as a proxy for generalization. Namely, we show how marginal likelihood can be negatively correlated with generalization, with implications for neural architecture search, and can lead to both underfitting and overfitting in hyperparameter learning. We provide a partial remedy through a conditional marginal likelihood, which we show is more aligned with generalization, and practically valuable for large-scale hyperparameter learning, such as in deep kernel learning.

----

## [626] Feature Learning and Signal Propagation in Deep Neural Networks

**Authors**: *Yizhang Lou, Chris E. Mingard, Soufiane Hayou*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lou22a.html](https://proceedings.mlr.press/v162/lou22a.html)

**Abstract**:

Recent work by Baratin et al. (2021) sheds light on an intriguing pattern that occurs during the training of deep neural networks: some layers align much more with data compared to other layers (where the alignment is defined as the normalize euclidean product of the tangent features matrix and the data labels matrix). The curve of the alignment as a function of layer index (generally) exhibits a ascent-descent pattern where the maximum is reached for some hidden layer. In this work, we provide the first explanation for this phenomenon. We introduce the Equilibrium Hypothesis which connects this alignment pattern to signal propagation in deep neural networks. Our experiments demonstrate an excellent match with the theoretical predictions.

----

## [627] Fluctuations, Bias, Variance & Ensemble of Learners: Exact Asymptotics for Convex Losses in High-Dimension

**Authors**: *Bruno Loureiro, Cédric Gerbelot, Maria Refinetti, Gabriele Sicuro, Florent Krzakala*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/loureiro22a.html](https://proceedings.mlr.press/v162/loureiro22a.html)

**Abstract**:

From the sampling of data to the initialisation of parameters, randomness is ubiquitous in modern Machine Learning practice. Understanding the statistical fluctuations engendered by the different sources of randomness in prediction is therefore key to understanding robust generalisation. In this manuscript we develop a quantitative and rigorous theory for the study of fluctuations in an ensemble of generalised linear models trained on different, but correlated, features in high-dimensions. In particular, we provide a complete description of the asymptotic joint distribution of the empirical risk minimiser for generic convex loss and regularisation in the high-dimensional limit. Our result encompasses a rich set of classification and regression tasks, such as the lazy regime of overparametrised neural networks, or equivalently the random features approximation of kernels. While allowing to study directly the mitigating effect of ensembling (or bagging) on the bias-variance decomposition of the test error, our analysis also helps disentangle the contribution of statistical fluctuations, and the singular role played by the interpolation threshold that are at the roots of the “double-descent” phenomenon.

----

## [628] A Single-Loop Gradient Descent and Perturbed Ascent Algorithm for Nonconvex Functional Constrained Optimization

**Authors**: *Songtao Lu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lu22a.html](https://proceedings.mlr.press/v162/lu22a.html)

**Abstract**:

Nonconvex constrained optimization problems can be used to model a number of machine learning problems, such as multi-class Neyman-Pearson classification and constrained Markov decision processes. However, such kinds of problems are challenging because both the objective and constraints are possibly nonconvex, so it is difficult to balance the reduction of the loss value and reduction of constraint violation. Although there are a few methods that solve this class of problems, all of them are double-loop or triple-loop algorithms, and they require oracles to solve some subproblems up to certain accuracy by tuning multiple hyperparameters at each iteration. In this paper, we propose a novel gradient descent and perturbed ascent (GDPA) algorithm to solve a class of smooth nonconvex inequality constrained problems. The GDPA is a primal-dual algorithm, which only exploits the first-order information of both the objective and constraint functions to update the primal and dual variables in an alternating way. The key feature of the proposed algorithm is that it is a single-loop algorithm, where only two step-sizes need to be tuned. We show that under a mild regularity condition GDPA is able to find Karush-Kuhn-Tucker (KKT) points of nonconvex functional constrained problems with convergence rate guarantees. To the best of our knowledge, it is the first single-loop algorithm that can solve the general nonconvex smooth problems with nonconvex inequality constraints. Numerical results also showcase the superiority of GDPA compared with the best-known algorithms (in terms of both stationarity measure and feasibility of the obtained solutions).

----

## [629] Additive Gaussian Processes Revisited

**Authors**: *Xiaoyu Lu, Alexis Boukouvalas, James Hensman*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lu22b.html](https://proceedings.mlr.press/v162/lu22b.html)

**Abstract**:

Gaussian Process (GP) models are a class of flexible non-parametric models that have rich representational power. By using a Gaussian process with additive structure, complex responses can be modelled whilst retaining interpretability. Previous work showed that additive Gaussian process models require high-dimensional interaction terms. We propose the orthogonal additive kernel (OAK), which imposes an orthogonality constraint on the additive functions, enabling an identifiable, low-dimensional representation of the functional relationship. We connect the OAK kernel to functional ANOVA decomposition, and show improved convergence rates for sparse computation methods. With only a small number of additive low-dimensional terms, we demonstrate the OAK model achieves similar or better predictive performance compared to black-box models, while retaining interpretability.

----

## [630] ModLaNets: Learning Generalisable Dynamics via Modularity and Physical Inductive Bias

**Authors**: *Yupu Lu, Shijie Lin, Guanqi Chen, Jia Pan*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lu22c.html](https://proceedings.mlr.press/v162/lu22c.html)

**Abstract**:

Deep learning models are able to approximate one specific dynamical system but struggle at learning generalisable dynamics, where dynamical systems obey the same laws of physics but contain different numbers of elements (e.g., double- and triple-pendulum systems). To relieve this issue, we proposed the Modular Lagrangian Network (ModLaNet), a structural neural network framework with modularity and physical inductive bias. This framework models the energy of each element using modularity and then construct the target dynamical system via Lagrangian mechanics. Modularity is beneficial for reusing trained networks and reducing the scale of networks and datasets. As a result, our framework can learn from the dynamics of simpler systems and extend to more complex ones, which is not feasible using other relevant physics-informed neural networks. We examine our framework for modelling double-pendulum or three-body systems with small training datasets, where our models achieve the best data efficiency and accuracy performance compared with counterparts. We also reorganise our models as extensions to model multi-pendulum and multi-body systems, demonstrating the intriguing reusable feature of our framework.

----

## [631] Model-Free Opponent Shaping

**Authors**: *Christopher Lu, Timon Willi, Christian A. Schröder de Witt, Jakob N. Foerster*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lu22d.html](https://proceedings.mlr.press/v162/lu22d.html)

**Abstract**:

In general-sum games the interaction of self-interested learning agents commonly leads to collectively worst-case outcomes, such as defect-defect in the iterated prisoner’s dilemma (IPD). To overcome this, some methods, such as Learning with Opponent-Learning Awareness (LOLA), directly shape the learning process of their opponents. However, these methods are myopic since only a small number of steps can be anticipated, are asymmetric since they treat other agents as naive learners, and require the use of higher-order derivatives, which are calculated through white-box access to an opponent’s differentiable learning algorithm. To address these issues, we propose Model-Free Opponent Shaping (M-FOS). M-FOS learns in a meta-game in which each meta-step is an episode of the underlying game. The meta-state consists of the policies in the underlying game and the meta-policy produces a new policy to be used in the next episode. M-FOS then uses generic model-free optimisation methods to learn meta-policies that accomplish long-horizon opponent shaping. Empirically, M-FOS near-optimally exploits naive learners and other, more sophisticated algorithms from the literature. For example, to the best of our knowledge, it is the first method to learn the well-known ZD extortion strategy in the IPD. In the same settings, M-FOS leads to socially optimal outcomes under meta-self-play. Finally, we show that M-FOS can be scaled to high-dimensional settings.

----

## [632] Multi-slots Online Matching with High Entropy

**Authors**: *Xingyu Lu, Qintong Wu, Wenliang Zhong*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lu22e.html](https://proceedings.mlr.press/v162/lu22e.html)

**Abstract**:

Online matching with diversity and fairness pursuit, a common building block in the recommendation and advertising, can be modeled as constrained convex programming with high entropy. While most existing approaches are based on the “single slot” assumption (i.e., assigning one item per iteration), they cannot be directly applied to cases with multiple slots, e.g., stock-aware top-N recommendation and advertising at multiple places. Particularly, the gradient computation and resource allocation are both challenging under this setting due to the absence of a closed-form solution. To overcome these obstacles, we develop a novel algorithm named Online subGradient descent for Multi-slots Allocation (OG-MA). It uses an efficient pooling algorithm to compute closed-form of the gradient then performs a roulette swapping for allocation, yielding a sub-linear regret with linear cost per iteration. Extensive experiments on synthetic and industrial data sets demonstrate that OG-MA is a fast and promising method for multi-slots online matching.

----

## [633] Maximum Likelihood Training for Score-based Diffusion ODEs by High Order Denoising Score Matching

**Authors**: *Cheng Lu, Kaiwen Zheng, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lu22f.html](https://proceedings.mlr.press/v162/lu22f.html)

**Abstract**:

Score-based generative models have excellent performance in terms of generation quality and likelihood. They model the data distribution by matching a parameterized score network with first-order data score functions. The score network can be used to define an ODE (“score-based diffusion ODE”) for exact likelihood evaluation. However, the relationship between the likelihood of the ODE and the score matching objective is unclear. In this work, we prove that matching the first-order score is not sufficient to maximize the likelihood of the ODE, by showing a gap between the maximum likelihood and score matching objectives. To fill up this gap, we show that the negative likelihood of the ODE can be bounded by controlling the first, second, and third-order score matching errors; and we further present a novel high-order denoising score matching method to enable maximum likelihood training of score-based diffusion ODEs. Our algorithm guarantees that the higher-order matching error is bounded by the training error and the lower-order errors. We empirically observe that by high-order score matching, score-based diffusion ODEs achieve better likelihood on both synthetic data and CIFAR-10, while retaining the high generation quality.

----

## [634] Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering

**Authors**: *Ekdeep Singh Lubana, Chi Ian Tang, Fahim Kawsar, Robert P. Dick, Akhil Mathur*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lubana22a.html](https://proceedings.mlr.press/v162/lubana22a.html)

**Abstract**:

Federated learning is generally used in tasks where labels are readily available (e.g., next word prediction). Relaxing this constraint requires design of unsupervised learning techniques that can support desirable properties for federated training: robustness to statistical/systems heterogeneity, scalability with number of participants, and communication efficiency. Prior work on this topic has focused on directly extending centralized self-supervised learning techniques, which are not designed to have the properties listed above. To address this situation, we propose Orchestra, a novel unsupervised federated learning technique that exploits the federation’s hierarchy to orchestrate a distributed clustering task and enforce a globally consistent partitioning of clients’ data into discriminable clusters. We show the algorithmic pipeline in Orchestra guarantees good generalization performance under a linear probe, allowing it to outperform alternative techniques in a broad range of conditions, including variation in heterogeneity, number of clients, participation ratio, and local epochs.

----

## [635] A Rigorous Study of Integrated Gradients Method and Extensions to Internal Neuron Attributions

**Authors**: *Daniel Lundström, Tianjian Huang, Meisam Razaviyayn*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lundstrom22a.html](https://proceedings.mlr.press/v162/lundstrom22a.html)

**Abstract**:

As deep learning (DL) efficacy grows, concerns for poor model explainability grow also. Attribution methods address the issue of explainability by quantifying the importance of an input feature for a model prediction. Among various methods, Integrated Gradients (IG) sets itself apart by claiming other methods failed to satisfy desirable axioms, while IG and methods like it uniquely satisfy said axioms. This paper comments on fundamental aspects of IG and its applications/extensions: 1) We identify key differences between IG function spaces and the supporting literature’s function spaces which problematize previous claims of IG uniqueness. We show that with the introduction of an additional axiom, non-decreasing positivity, the uniqueness claims can be established. 2) We address the question of input sensitivity by identifying function classes where IG is/is not Lipschitz in the attributed input. 3) We show that axioms for single-baseline methods have analogous properties for methods with probability distribution baselines. 4) We introduce a computationally efficient method of identifying internal neurons that contribute to specified regions of an IG attribution map. Finally, we present experimental results validating this method.

----

## [636] BAMDT: Bayesian Additive Semi-Multivariate Decision Trees for Nonparametric Regression

**Authors**: *Zhao Tang Luo, Huiyan Sang, Bani K. Mallick*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/luo22a.html](https://proceedings.mlr.press/v162/luo22a.html)

**Abstract**:

Bayesian additive regression trees (BART; Chipman et al., 2010) have gained great popularity as a flexible nonparametric function estimation and modeling tool. Nearly all existing BART models rely on decision tree weak learners with axis-parallel univariate split rules to partition the Euclidean feature space into rectangular regions. In practice, however, many regression problems involve features with multivariate structures (e.g., spatial locations) possibly lying in a manifold, where rectangular partitions may fail to respect irregular intrinsic geometry and boundary constraints of the structured feature space. In this paper, we develop a new class of Bayesian additive multivariate decision tree models that combine univariate split rules for handling possibly high dimensional features without known multivariate structures and novel multivariate split rules for features with multivariate structures in each weak learner. The proposed multivariate split rules are built upon stochastic predictive spanning tree bipartition models on reference knots, which are capable of achieving highly flexible nonlinear decision boundaries on manifold feature spaces while enabling efficient dimension reduction computations. We demonstrate the superior performance of the proposed method using simulation data and a Sacramento housing price data set.

----

## [637] Disentangled Federated Learning for Tackling Attributes Skew via Invariant Aggregation and Diversity Transferring

**Authors**: *Zhengquan Luo, Yunlong Wang, Zilei Wang, Zhenan Sun, Tieniu Tan*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/luo22b.html](https://proceedings.mlr.press/v162/luo22b.html)

**Abstract**:

Attributes skew hinders the current federated learning (FL) frameworks from consistent optimization directions among the clients, which inevitably leads to performance reduction and unstable convergence. The core problems lie in that: 1) Domain-specific attributes, which are non-causal and only locally valid, are indeliberately mixed into global aggregation. 2) The one-stage optimizations of entangled attributes cannot simultaneously satisfy two conflicting objectives, i.e., generalization and personalization. To cope with these, we proposed disentangled federated learning (DFL) to disentangle the domain-specific and cross-invariant attributes into two complementary branches, which are trained by the proposed alternating local-global optimization independently. Importantly, convergence analysis proves that the FL system can be stably converged even if incomplete client models participate in the global aggregation, which greatly expands the application scope of FL. Extensive experiments verify that DFL facilitates FL with higher performance, better interpretability, and faster convergence rate, compared with SOTA FL methods on both manually synthesized and realistic attributes skew datasets.

----

## [638] Channel Importance Matters in Few-Shot Image Classification

**Authors**: *Xu Luo, Jing Xu, Zenglin Xu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/luo22c.html](https://proceedings.mlr.press/v162/luo22c.html)

**Abstract**:

Few-Shot Learning (FSL) requires vision models to quickly adapt to brand-new classification tasks with a shift in task distribution. Understanding the difficulties posed by this task distribution shift is central to FSL. In this paper, we show that a simple channel-wise feature transformation may be the key to unraveling this secret from a channel perspective. When facing novel few-shot tasks in the test-time datasets, this transformation can greatly improve the generalization ability of learned image representations, while being agnostic to the choice of datasets and training algorithms. Through an in-depth analysis of this transformation, we find that the difficulty of representation transfer in FSL stems from the severe channel bias problem of image representations: channels may have different importance in different tasks, while convolutional neural networks are likely to be insensitive, or respond incorrectly to such a shift. This points out a core problem of the generalization ability of modern vision systems which needs further attention in the future.

----

## [639] Learning Dynamics and Generalization in Deep Reinforcement Learning

**Authors**: *Clare Lyle, Mark Rowland, Will Dabney, Marta Kwiatkowska, Yarin Gal*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lyle22a.html](https://proceedings.mlr.press/v162/lyle22a.html)

**Abstract**:

Solving a reinforcement learning (RL) problem poses two competing challenges: fitting a potentially discontinuous value function, and generalizing well to new observations. In this paper, we analyze the learning dynamics of temporal difference algorithms to gain novel insight into the tension between these two objectives. We show theoretically that temporal difference learning encourages agents to fit non-smooth components of the value function early in training, and at the same time induces the second-order effect of discouraging generalization. We corroborate these findings in deep RL agents trained on a range of environments, finding that neural networks trained using temporal difference algorithms on dense reward tasks exhibit weaker generalization between states than randomly initialized networks and networks trained with policy gradient methods. Finally, we investigate how post-training policy distillation may avoid this pitfall, and show that this approach improves generalization to novel environments in the ProcGen suite and improves robustness to input perturbations.

----

## [640] On Finite-Sample Identifiability of Contrastive Learning-Based Nonlinear Independent Component Analysis

**Authors**: *Qi Lyu, Xiao Fu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lyu22a.html](https://proceedings.mlr.press/v162/lyu22a.html)

**Abstract**:

Nonlinear independent component analysis (nICA) aims at recovering statistically independent latent components that are mixed by unknown nonlinear functions. Central to nICA is the identifiability of the latent components, which had been elusive until very recently. Specifically, Hyvärinen et al. have shown that the nonlinearly mixed latent components are identifiable (up to often inconsequential ambiguities) under a generalized contrastive learning (GCL) formulation, given that the latent components are independent conditioned on a certain auxiliary variable. The GCL-based identifiability of nICA is elegant, and establishes interesting connections between nICA and popular unsupervised/self-supervised learning paradigms in representation learning, causal learning, and factor disentanglement. However, existing identifiability analyses of nICA all build upon an unlimited sample assumption and the use of ideal universal function learners—which creates a non-negligible gap between theory and practice. Closing the gap is a nontrivial challenge, as there is a lack of established “textbook” routine for finite sample analysis of such unsupervised problems. This work puts forth a finite-sample identifiability analysis of GCL-based nICA. Our analytical framework judiciously combines the properties of the GCL loss function, statistical generalization analysis, and numerical differentiation. Our framework also takes the learning function’s approximation error into consideration, and reveals an intuitive trade-off between the complexity and expressiveness of the employed function learner. Numerical experiments are used to validate the theorems.

----

## [641] Pessimism meets VCG: Learning Dynamic Mechanism Design via Offline Reinforcement Learning

**Authors**: *Boxiang Lyu, Zhaoran Wang, Mladen Kolar, Zhuoran Yang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lyu22b.html](https://proceedings.mlr.press/v162/lyu22b.html)

**Abstract**:

Dynamic mechanism design has garnered significant attention from both computer scientists and economists in recent years. By allowing agents to interact with the seller over multiple rounds, where agents’ reward functions may change with time and are state-dependent, the framework is able to model a rich class of real-world problems. In these works, the interaction between agents and sellers is often assumed to follow a Markov Decision Process (MDP). We focus on the setting where the reward and transition functions of such an MDP are not known a priori, and we are attempting to recover the optimal mechanism using an a priori collected data set. In the setting where the function approximation is employed to handle large state spaces, with only mild assumptions on the expressiveness of the function class, we are able to design a dynamic mechanism using offline reinforcement learning algorithms. Moreover, learned mechanisms approximately have three key desiderata: efficiency, individual rationality, and truthfulness. Our algorithm is based on the pessimism principle and only requires a mild assumption on the coverage of the offline data set. To the best of our knowledge, our work provides the first offline RL algorithm for dynamic mechanism design without assuming uniform coverage.

----

## [642] Versatile Offline Imitation from Observations and Examples via Regularized State-Occupancy Matching

**Authors**: *Yecheng Jason Ma, Andrew Shen, Dinesh Jayaraman, Osbert Bastani*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ma22a.html](https://proceedings.mlr.press/v162/ma22a.html)

**Abstract**:

We propose State Matching Offline DIstribution Correction Estimation (SMODICE), a novel and versatile regression-based offline imitation learning algorithm derived via state-occupancy matching. We show that the SMODICE objective admits a simple optimization procedure through an application of Fenchel duality and an analytic solution in tabular MDPs. Without requiring access to expert actions, SMODICE can be effectively applied to three offline IL settings: (i) imitation from observations (IfO), (ii) IfO with dynamics or morphologically mismatched expert, and (iii) example-based reinforcement learning, which we show can be formulated as a state-occupancy matching problem. We extensively evaluate SMODICE on both gridworld environments as well as on high-dimensional offline benchmarks. Our results demonstrate that SMODICE is effective for all three problem settings and significantly outperforms prior state-of-art.

----

## [643] Quantification and Analysis of Layer-wise and Pixel-wise Information Discarding

**Authors**: *Haotian Ma, Hao Zhang, Fan Zhou, Yinqing Zhang, Quanshi Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ma22b.html](https://proceedings.mlr.press/v162/ma22b.html)

**Abstract**:

This paper presents a method to explain how the information of each input variable is gradually discarded during the forward propagation in a deep neural network (DNN), which provides new perspectives to explain DNNs. We define two types of entropy-based metrics, i.e. (1) the discarding of pixel-wise information used in the forward propagation, and (2) the uncertainty of the input reconstruction, to measure input information contained by a specific layer from two perspectives. Unlike previous attribution metrics, the proposed metrics ensure the fairness of comparisons between different layers of different DNNs. We can use these metrics to analyze the efficiency of information processing in DNNs, which exhibits strong connections to the performance of DNNs. We analyze information discarding in a pixel-wise manner, which is different from the information bottleneck theory measuring feature information w.r.t. the sample distribution. Experiments have shown the effectiveness of our metrics in analyzing classic DNNs and explaining existing deep-learning techniques. The code is available at https://github.com/haotianSustc/deepinfo.

----

## [644] Interpretable Neural Networks with Frank-Wolfe: Sparse Relevance Maps and Relevance Orderings

**Authors**: *Jan MacDonald, Mathieu Besançon, Sebastian Pokutta*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/macdonald22a.html](https://proceedings.mlr.press/v162/macdonald22a.html)

**Abstract**:

We study the effects of constrained optimization formulations and Frank-Wolfe algorithms for obtaining interpretable neural network predictions. Reformulating the Rate-Distortion Explanations (RDE) method for relevance attribution as a constrained optimization problem provides precise control over the sparsity of relevance maps. This enables a novel multi-rate as well as a relevance-ordering variant of RDE that both empirically outperform standard RDE and other baseline methods in a well-established comparison test. We showcase several deterministic and stochastic variants of the Frank-Wolfe algorithm and their effectiveness for RDE.

----

## [645] A Tighter Analysis of Spectral Clustering, and Beyond

**Authors**: *Peter Macgregor, He Sun*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/macgregor22a.html](https://proceedings.mlr.press/v162/macgregor22a.html)

**Abstract**:

This work studies the classical spectral clustering algorithm which embeds the vertices of some graph G=(V_G, E_G) into R^k using k eigenvectors of some matrix of G, and applies k-means to partition V_G into k clusters. Our first result is a tighter analysis on the performance of spectral clustering, and explains why it works under some much weaker condition than the ones studied in the literature. For the second result, we show that, by applying fewer than k eigenvectors to construct the embedding, spectral clustering is able to produce better output for many practical instances; this result is the first of its kind in spectral clustering. Besides its conceptual and theoretical significance, the practical impact of our work is demonstrated by the empirical analysis on both synthetic and real-world data sets, in which spectral clustering produces comparable or better results with fewer than k eigenvectors.

----

## [646] Zero-Shot Reward Specification via Grounded Natural Language

**Authors**: *Parsa Mahmoudieh, Deepak Pathak, Trevor Darrell*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mahmoudieh22a.html](https://proceedings.mlr.press/v162/mahmoudieh22a.html)

**Abstract**:

Reward signals in reinforcement learning are expensive to design and often require access to the true state which is not available in the real world. Common alternatives are usually demonstrations or goal images which can be labor-intensive to collect. On the other hand, text descriptions provide a general, natural, and low-effort way of communicating the desired task. However, prior works in learning text-conditioned policies still rely on rewards that are defined using either true state or labeled expert demonstrations. We use recent developments in building large-scale visuolanguage models like CLIP to devise a framework that generates the task reward signal just from goal text description and raw pixel observations which is then used to learn the task policy. We evaluate the proposed framework on control and robotic manipulation tasks. Finally, we distill the individual task policies into a single goal text conditioned policy that can generalize in a zero-shot manner to new tasks with unseen objects and unseen goal text descriptions.

----

## [647] Feature selection using e-values

**Authors**: *Subhabrata Majumdar, Snigdhansu Chatterjee*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/majumdar22a.html](https://proceedings.mlr.press/v162/majumdar22a.html)

**Abstract**:

In the context of supervised learning, we introduce the concept of e-value. An e-value is a scalar quantity that represents the proximity of the sampling distribution of parameter estimates in a model trained on a subset of features to that of the model trained on all features (i.e. the full model). Under general conditions, a rank ordering of e-values separates models that contain all essential features from those that do not. For a p-dimensional feature space, this requires fitting only the full model and evaluating p+1 models, as opposed to the traditional requirement of fitting and evaluating 2^p models. The above e-values framework is applicable to a wide range of parametric models. We use data depths and a fast resampling-based algorithm to implement a feature selection procedure, providing consistency results. Through experiments across several model settings and synthetic and real datasets, we establish that the e-values can be a promising general alternative to existing model-specific methods of feature selection.

----

## [648] SSL Enables Learning from Sparse Rewards in Image-Goal Navigation

**Authors**: *Arjun Majumdar, Gunnar A. Sigurdsson, Robinson Piramuthu, Jesse Thomason, Dhruv Batra, Gaurav S. Sukhatme*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/majumdar22b.html](https://proceedings.mlr.press/v162/majumdar22b.html)

**Abstract**:

Visual navigation through novel environments is an essential skill for intelligent household robots. Reinforcement learning (RL) with dense reward shaping is the standard approach for such visual navigation problems. Dense rewards are used despite the fact that they penalize exploration—a key skill for navigating in new environments— because of the commonly held belief that sparse rewards do not work. In this paper, we present a surprising finding. We demonstrate that combining sparse rewards with self-supervised learning (SSL) not only makes them work, but also outperforms dense rewards, which is the first result of this kind. We apply our approach alongside data augmentation to train a general-purpose agent for image-goal navigation (ImageNav), in which the goal is presented as an image captured from the target location. These techniques improve navigation success from 64% to 72% (+8), path efficiency—measured by success weighted by path length (SPL)—from 0.50 to 0.62 (+0.12), and also lead to a +0.06 absolute improvement in SPL over the state-of-the-art.

----

## [649] Knowledge-Grounded Self-Rationalization via Extractive and Natural Language Explanations

**Authors**: *Bodhisattwa Prasad Majumder, Oana Camburu, Thomas Lukasiewicz, Julian J. McAuley*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/majumder22a.html](https://proceedings.mlr.press/v162/majumder22a.html)

**Abstract**:

Models that generate extractive rationales (i.e., subsets of features) or natural language explanations (NLEs) for their predictions are important for explainable AI. While an extractive rationale provides a quick view of the features most responsible for a prediction, an NLE allows for a comprehensive description of the decision-making process behind a prediction. However, current models that generate the best extractive rationales or NLEs often fall behind the state-of-the-art (SOTA) in terms of task performance. In this work, we bridge this gap by introducing RExC, a self-rationalizing framework that grounds its predictions and two complementary types of explanations (NLEs and extractive rationales) in background knowledge. Our framework improves over previous methods by: (i) reaching SOTA task performance while also providing explanations, (ii) providing two types of explanations, while existing models usually provide only one type, and (iii) beating by a large margin the previous SOTA in terms of quality of both types of explanations. Furthermore, a perturbation analysis in RExC shows a high degree of association between explanations and predictions, a necessary property of faithful explanations.

----

## [650] Nonparametric Involutive Markov Chain Monte Carlo

**Authors**: *Carol Mak, Fabian Zaiser, Luke Ong*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mak22a.html](https://proceedings.mlr.press/v162/mak22a.html)

**Abstract**:

A challenging problem in probabilistic programming is to develop inference algorithms that work for arbitrary programs in a universal probabilistic programming language (PPL). We present the nonparametric involutive Markov chain Monte Carlo (NP-iMCMC) algorithm as a method for constructing MCMC inference algorithms for nonparametric models expressible in universal PPLs. Building on the unifying involutive MCMC framework, and by providing a general procedure for driving state movement between dimensions, we show that NP-iMCMC can generalise numerous existing iMCMC algorithms to work on nonparametric models. We prove the correctness of the NP-iMCMC sampler. Our empirical study shows that the existing strengths of several iMCMC algorithms carry over to their nonparametric extensions. Applying our method to the recently proposed Nonparametric HMC, an instance of (Multiple Step) NP-iMCMC, we have constructed several nonparametric extensions (all of which new) that exhibit significant performance improvements.

----

## [651] Architecture Agnostic Federated Learning for Neural Networks

**Authors**: *Disha Makhija, Xing Han, Nhat Ho, Joydeep Ghosh*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/makhija22a.html](https://proceedings.mlr.press/v162/makhija22a.html)

**Abstract**:

With growing concerns regarding data privacy and rapid increase in data volume, Federated Learning (FL) has become an important learning paradigm. However, jointly learning a deep neural network model in a FL setting proves to be a non-trivial task because of the complexities associated with the neural networks, such as varied architectures across clients, permutation invariance of the neurons, and presence of non-linear transformations in each layer. This work introduces a novel framework, Federated Heterogeneous Neural Networks (FedHeNN), that allows each client to build a personalised model without enforcing a common architecture across clients. This allows each client to optimize with respect to local data and compute constraints, while still benefiting from the learnings of other (potentially more powerful) clients. The key idea of FedHeNN is to use the instance-level representations obtained from peer clients to guide the simultaneous training on each client. The extensive experimental results demonstrate that the FedHeNN framework is capable of learning better performing models on clients in both the settings of homogeneous and heterogeneous architectures across clients.

----

## [652] Robustness in Multi-Objective Submodular Optimization: a Quantile Approach

**Authors**: *Cédric Malherbe, Kevin Scaman*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/malherbe22a.html](https://proceedings.mlr.press/v162/malherbe22a.html)

**Abstract**:

The optimization of multi-objective submodular systems appears in a wide variety of applications. However, there are currently very few techniques which are able to provide a robust allocation to such systems. In this work, we propose to design and analyse novel algorithms for the robust allocation of submodular systems through lens of quantile maximization. We start by observing that identifying an exact solution for this problem is computationally intractable. To tackle this issue, we propose a proxy for the quantile function using a softmax formulation, and show that this proxy is well suited to submodular optimization. Based on this relaxation, we propose a novel and simple algorithm called SOFTSAT. Theoretical properties are provided for this algorithm as well as novel approximation guarantees. Finally, we provide numerical experiments showing the efficiency of our algorithm with regards to state-of-the-art methods in a test bed of real-world applications, and show that SOFTSAT is particularly robust and well-suited to online scenarios.

----

## [653] More Efficient Sampling for Tensor Decomposition With Worst-Case Guarantees

**Authors**: *Osman Asif Malik*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/malik22a.html](https://proceedings.mlr.press/v162/malik22a.html)

**Abstract**:

Recent papers have developed alternating least squares (ALS) methods for CP and tensor ring decomposition with a per-iteration cost which is sublinear in the number of input tensor entries for low-rank decomposition. However, the per-iteration cost of these methods still has an exponential dependence on the number of tensor modes when parameters are chosen to achieve certain worst-case guarantees. In this paper, we propose sampling-based ALS methods for the CP and tensor ring decompositions whose cost does not have this exponential dependence, thereby significantly improving on the previous state-of-the-art. We provide a detailed theoretical analysis and also apply the methods in a feature extraction experiment.

----

## [654] Unaligned Supervision for Automatic Music Transcription in The Wild

**Authors**: *Ben Maman, Amit H. Bermano*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/maman22a.html](https://proceedings.mlr.press/v162/maman22a.html)

**Abstract**:

Multi-instrument Automatic Music Transcription (AMT), or the decoding of a musical recording into semantic musical content, is one of the holy grails of Music Information Retrieval. Current AMT approaches are restricted to piano and (some) guitar recordings, due to difficult data collection. In order to overcome data collection barriers, previous AMT approaches attempt to employ musical scores in the form of a digitized version of the same song or piece. The scores are typically aligned using audio features and strenuous human intervention to generate training labels. We introduce Note$_{EM}$, a method for simultaneously training a transcriber and aligning the scores to their corresponding performances, in a fully-automated process. Using this unaligned supervision scheme, complemented by pseudo-labels and pitch shift augmentation, our method enables training on in-the-wild recordings with unprecedented accuracy and instrumental variety. Using only synthetic data and unaligned supervision, we report SOTA note-level accuracy of the MAPS dataset, and large favorable margins on cross-dataset evaluations. We also demonstrate robustness and ease of use; we report comparable results when training on a small, easily obtainable, self-collected dataset, and we propose alternative labeling to the MusicNet dataset, which we show to be more accurate. Our project page is available at https://benadar293.github.io.

----

## [655] Decision-Focused Learning: Through the Lens of Learning to Rank

**Authors**: *Jayanta Mandi, Víctor Bucarey, Maxime Mulamba Ke Tchomba, Tias Guns*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mandi22a.html](https://proceedings.mlr.press/v162/mandi22a.html)

**Abstract**:

In the last years decision-focused learning framework, also known as predict-and-optimize, have received increasing attention. In this setting, the predictions of a machine learning model are used as estimated cost coefficients in the objective function of a discrete combinatorial optimization problem for decision making. Decision-focused learning proposes to train the ML models, often neural network models, by directly optimizing the quality of decisions made by the optimization solvers. Based on a recent work that proposed a noise contrastive estimation loss over a subset of the solution space, we observe that decision-focused learning can more generally be seen as a learning-to-rank problem, where the goal is to learn an objective function that ranks the feasible points correctly. This observation is independent of the optimization method used and of the form of the objective function. We develop pointwise, pairwise and listwise ranking loss functions, which can be differentiated in closed form given a subset of solutions. We empirically investigate the quality of our generic methods compared to existing decision-focused learning approaches with competitive results. Furthermore, controlling the subset of solutions allows controlling the runtime considerably, with limited effect on regret.

----

## [656] Differentially Private Coordinate Descent for Composite Empirical Risk Minimization

**Authors**: *Paul Mangold, Aurélien Bellet, Joseph Salmon, Marc Tommasi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mangold22a.html](https://proceedings.mlr.press/v162/mangold22a.html)

**Abstract**:

Machine learning models can leak information about the data used to train them. To mitigate this issue, Differentially Private (DP) variants of optimization algorithms like Stochastic Gradient Descent (DP-SGD) have been designed to trade-off utility for privacy in Empirical Risk Minimization (ERM) problems. In this paper, we propose Differentially Private proximal Coordinate Descent (DP-CD), a new method to solve composite DP-ERM problems. We derive utility guarantees through a novel theoretical analysis of inexact coordinate descent. Our results show that, thanks to larger step sizes, DP-CD can exploit imbalance in gradient coordinates to outperform DP-SGD. We also prove new lower bounds for composite DP-ERM under coordinate-wise regularity assumptions, that are nearly matched by DP-CD. For practical implementations, we propose to clip gradients using coordinate-wise thresholds that emerge from our theory, avoiding costly hyperparameter tuning. Experiments on real and synthetic data support our results, and show that DP-CD compares favorably with DP-SGD.

----

## [657] Refined Convergence Rates for Maximum Likelihood Estimation under Finite Mixture Models

**Authors**: *Tudor A. Manole, Nhat Ho*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/manole22a.html](https://proceedings.mlr.press/v162/manole22a.html)

**Abstract**:

We revisit the classical problem of deriving convergence rates for the maximum likelihood estimator (MLE) in finite mixture models. The Wasserstein distance has become a standard loss function for the analysis of parameter estimation in these models, due in part to its ability to circumvent label switching and to accurately characterize the behaviour of fitted mixture components with vanishing weights. However, the Wasserstein distance is only able to capture the worst-case convergence rate among the remaining fitted mixture components. We demonstrate that when the log-likelihood function is penalized to discourage vanishing mixing weights, stronger loss functions can be derived to resolve this shortcoming of the Wasserstein distance. These new loss functions accurately capture the heterogeneity in convergence rates of fitted mixture components, and we use them to sharpen existing pointwise and uniform convergence rates in various classes of mixture models. In particular, these results imply that a subset of the components of the penalized MLE typically converge significantly faster than could have been anticipated from past work. We further show that some of these conclusions extend to the traditional MLE. Our theoretical findings are supported by a simulation study to illustrate these improved convergence rates.

----

## [658] On Improving Model-Free Algorithms for Decentralized Multi-Agent Reinforcement Learning

**Authors**: *Weichao Mao, Lin Yang, Kaiqing Zhang, Tamer Basar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mao22a.html](https://proceedings.mlr.press/v162/mao22a.html)

**Abstract**:

Multi-agent reinforcement learning (MARL) algorithms often suffer from an exponential sample complexity dependence on the number of agents, a phenomenon known as the curse of multiagents. We address this challenge by investigating sample-efficient model-free algorithms in decentralized MARL, and aim to improve existing algorithms along this line. For learning (coarse) correlated equilibria in general-sum Markov games, we propose stage-based V-learning algorithms that significantly simplify the algorithmic design and analysis of recent works, and circumvent a rather complicated no-weighted-regret bandit subroutine. For learning Nash equilibria in Markov potential games, we propose an independent policy gradient algorithm with a decentralized momentum-based variance reduction technique. All our algorithms are decentralized in that each agent can make decisions based on only its local information. Neither communication nor centralized coordination is required during learning, leading to a natural generalization to a large number of agents. Finally, we provide numerical simulations to corroborate our theoretical findings.

----

## [659] On the Effects of Artificial Data Modification

**Authors**: *Antonia Marcu, Adam Prügel-Bennett*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/marcu22a.html](https://proceedings.mlr.press/v162/marcu22a.html)

**Abstract**:

Data distortion is commonly applied in vision models during both training (e.g methods like MixUp and CutMix) and evaluation (e.g. shape-texture bias and robustness). This data modification can introduce artificial information. It is often assumed that the resulting artefacts are detrimental to training, whilst being negligible when analysing models. We investigate these assumptions and conclude that in some cases they are unfounded and lead to incorrect results. Specifically, we show current shape bias identification methods and occlusion robustness measures are biased and propose a fairer alternative for the latter. Subsequently, through a series of experiments we seek to correct and strengthen the community’s perception of how augmenting affects learning of vision models. Based on our empirical results we argue that the impact of the artefacts must be understood and exploited rather than eliminated.

----

## [660] Personalized Federated Learning through Local Memorization

**Authors**: *Othmane Marfoq, Giovanni Neglia, Richard Vidal, Laetitia Kameni*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/marfoq22a.html](https://proceedings.mlr.press/v162/marfoq22a.html)

**Abstract**:

Federated learning allows clients to collaboratively learn statistical models while keeping their data local. Federated learning was originally used to train a unique global model to be served to all clients, but this approach might be sub-optimal when clients’ local data distributions are heterogeneous. In order to tackle this limitation, recent personalized federated learning methods train a separate model for each client while still leveraging the knowledge available at other clients. In this work, we exploit the ability of deep neural networks to extract high quality vectorial representations (embeddings) from non-tabular data, e.g., images and text, to propose a personalization mechanism based on local memorization. Personalization is obtained by interpolating a collectively trained global model with a local $k$-nearest neighbors (kNN) model based on the shared representation provided by the global model. We provide generalization bounds for the proposed approach in the case of binary classification, and we show on a suite of federated datasets that this approach achieves significantly higher accuracy and fairness than state-of-the-art methods.

----

## [661] Nested Bandits

**Authors**: *Matthieu Martin, Panayotis Mertikopoulos, Thibaud Rahier, Houssam Zenati*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/martin22a.html](https://proceedings.mlr.press/v162/martin22a.html)

**Abstract**:

In many online decision processes, the optimizing agent is called to choose between large numbers of alternatives with many inherent similarities; in turn, these similarities imply closely correlated losses that may confound standard discrete choice models and bandit algorithms. We study this question in the context of nested bandits, a class of adversarial multi-armed bandit problems where the learner seeks to minimize their regret in the presence of a large number of distinct alternatives with a hierarchy of embedded (non-combinatorial) similarities. In this setting, optimal algorithms based on the exponential weights blueprint (like Hedge, EXP3, and their variants) may incur significant regret because they tend to spend excessive amounts of time exploring irrelevant alternatives with similar, suboptimal costs. To account for this, we propose a nested exponential weights (NEW) algorithm that performs a layered exploration of the learner’s set of alternatives based on a nested, step-by-step selection method. In so doing, we obtain a series of tight bounds for the learner’s regret showing that online learning problems with a high degree of similarity between alternatives can be resolved efficiently, without a red bus / blue bus paradox occurring.

----

## [662] Closed-Form Diffeomorphic Transformations for Time Series Alignment

**Authors**: *Iñigo Martinez, Elisabeth Viles, Igor G. Olaizola*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/martinez22a.html](https://proceedings.mlr.press/v162/martinez22a.html)

**Abstract**:

Time series alignment methods call for highly expressive, differentiable and invertible warping functions which preserve temporal topology, i.e diffeomorphisms. Diffeomorphic warping functions can be generated from the integration of velocity fields governed by an ordinary differential equation (ODE). Gradient-based optimization frameworks containing diffeomorphic transformations require to calculate derivatives to the differential equation’s solution with respect to the model parameters, i.e. sensitivity analysis. Unfortunately, deep learning frameworks typically lack automatic-differentiation-compatible sensitivity analysis methods; and implicit functions, such as the solution of ODE, require particular care. Current solutions appeal to adjoint sensitivity methods, ad-hoc numerical solvers or ResNet’s Eulerian discretization. In this work, we present a closed-form expression for the ODE solution and its gradient under continuous piecewise-affine (CPA) velocity functions. We present a highly optimized implementation of the results on CPU and GPU. Furthermore, we conduct extensive experiments on several datasets to validate the generalization ability of our model to unseen data for time-series joint alignment. Results show significant improvements both in terms of efficiency and accuracy.

----

## [663] SPECTRE: Spectral Conditioning Helps to Overcome the Expressivity Limits of One-shot Graph Generators

**Authors**: *Karolis Martinkus, Andreas Loukas, Nathanaël Perraudin, Roger Wattenhofer*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/martinkus22a.html](https://proceedings.mlr.press/v162/martinkus22a.html)

**Abstract**:

We approach the graph generation problem from a spectral perspective by first generating the dominant parts of the graph Laplacian spectrum and then building a graph matching these eigenvalues and eigenvectors. Spectral conditioning allows for direct modeling of the global and local graph structure and helps to overcome the expressivity and mode collapse issues of one-shot graph generators. Our novel GAN, called SPECTRE, enables the one-shot generation of much larger graphs than previously possible with one-shot models. SPECTRE outperforms state-of-the-art deep autoregressive generators in terms of modeling fidelity, while also avoiding expensive sequential generation and dependence on node ordering. A case in point, in sizable synthetic and real-world graphs SPECTRE achieves a 4-to-170 fold improvement over the best competitor that does not overfit and is 23-to-30 times faster than autoregressive generators.

----

## [664] Modular Conformal Calibration

**Authors**: *Charles Marx, Shengjia Zhao, Willie Neiswanger, Stefano Ermon*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/marx22a.html](https://proceedings.mlr.press/v162/marx22a.html)

**Abstract**:

Uncertainty estimates must be calibrated (i.e., accurate) and sharp (i.e., informative) in order to be useful. This has motivated a variety of methods for recalibration, which use held-out data to turn an uncalibrated model into a calibrated model. However, the applicability of existing methods is limited due to their assumption that the original model is also a probabilistic model. We introduce a versatile class of algorithms for recalibration in regression that we call modular conformal calibration (MCC). This framework allows one to transform any regression model into a calibrated probabilistic model. The modular design of MCC allows us to make simple adjustments to existing algorithms that enable well-behaved distribution predictions. We also provide finite-sample calibration guarantees for MCC algorithms. Our framework recovers isotonic recalibration, conformal calibration, and conformal interval prediction, implying that our theoretical results apply to those methods as well. Finally, we conduct an empirical study of MCC on 17 regression datasets. Our results show that new algorithms designed in our framework achieve near-perfect calibration and improve sharpness relative to existing methods.

----

## [665] Continual Repeated Annealed Flow Transport Monte Carlo

**Authors**: *Alexander G. de G. Matthews, Michael Arbel, Danilo Jimenez Rezende, Arnaud Doucet*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/matthews22a.html](https://proceedings.mlr.press/v162/matthews22a.html)

**Abstract**:

We propose Continual Repeated Annealed Flow Transport Monte Carlo (CRAFT), a method that combines a sequential Monte Carlo (SMC) sampler (itself a generalization of Annealed Importance Sampling) with variational inference using normalizing flows. The normalizing flows are directly trained to transport between annealing temperatures using a KL divergence for each transition. This optimization objective is itself estimated using the normalizing flow/SMC approximation. We show conceptually and using multiple empirical examples that CRAFT improves on Annealed Flow Transport Monte Carlo (Arbel et al., 2021), on which it builds and also on Markov chain Monte Carlo (MCMC) based Stochastic Normalizing Flows (Wu et al., 2020). By incorporating CRAFT within particle MCMC, we show that such learnt samplers can achieve impressively accurate results on a challenging lattice field theory example.

----

## [666] How to Stay Curious while avoiding Noisy TVs using Aleatoric Uncertainty Estimation

**Authors**: *Augustine N. Mavor-Parker, Kimberly A. Young, Caswell Barry, Lewis D. Griffin*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mavor-parker22a.html](https://proceedings.mlr.press/v162/mavor-parker22a.html)

**Abstract**:

When extrinsic rewards are sparse, artificial agents struggle to explore an environment. Curiosity, implemented as an intrinsic reward for prediction errors, can improve exploration but it is known to fail when faced with action-dependent noise sources (‘noisy TVs’). In an attempt to make exploring agents robust to Noisy TVs, we present a simple solution: aleatoric mapping agents (AMAs). AMAs are a novel form of curiosity that explicitly ascertain which state transitions of the environment are unpredictable, even if those dynamics are induced by the actions of the agent. This is achieved by generating separate forward predictions for the mean and aleatoric uncertainty of future states, with the aim of reducing intrinsic rewards for those transitions that are unpredictable. We demonstrate that in a range of environments AMAs are able to circumvent action-dependent stochastic traps that immobilise conventional curiosity driven agents. Furthermore, we demonstrate empirically that other common exploration approaches—previously thought to be immune to agent-induced randomness—can be trapped by stochastic dynamics.

----

## [667] How to Steer Your Adversary: Targeted and Efficient Model Stealing Defenses with Gradient Redirection

**Authors**: *Mantas Mazeika, Bo Li, David A. Forsyth*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mazeika22a.html](https://proceedings.mlr.press/v162/mazeika22a.html)

**Abstract**:

Model stealing attacks present a dilemma for public machine learning APIs. To protect financial investments, companies may be forced to withhold important information about their models that could facilitate theft, including uncertainty estimates and prediction explanations. This compromise is harmful not only to users but also to external transparency. Model stealing defenses seek to resolve this dilemma by making models harder to steal while preserving utility for benign users. However, existing defenses have poor performance in practice, either requiring enormous computational overheads or severe utility trade-offs. To meet these challenges, we present a new approach to model stealing defenses called gradient redirection. At the core of our approach is a provably optimal, efficient algorithm for steering an adversary’s training updates in a targeted manner. Combined with improvements to surrogate networks and a novel coordinated defense strategy, our gradient redirection defense, called GRAD^2, achieves small utility trade-offs and low computational overhead, outperforming the best prior defenses. Moreover, we demonstrate how gradient redirection enables reprogramming the adversary with arbitrary behavior, which we hope will foster work on new avenues of defense.

----

## [668] Quant-BnB: A Scalable Branch-and-Bound Method for Optimal Decision Trees with Continuous Features

**Authors**: *Rahul Mazumder, Xiang Meng, Haoyue Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mazumder22a.html](https://proceedings.mlr.press/v162/mazumder22a.html)

**Abstract**:

Decision trees are one of the most useful and popular methods in the machine learning toolbox. In this paper, we consider the problem of learning optimal decision trees, a combinatorial optimization problem that is challenging to solve at scale. A common approach in the literature is to use greedy heuristics, which may not be optimal. Recently there has been significant interest in learning optimal decision trees using various approaches (e.g., based on integer programming, dynamic programming)—to achieve computational scalability, most of these approaches focus on classification tasks with binary features. In this paper, we present a new discrete optimization method based on branch-and-bound (BnB) to obtain optimal decision trees. Different from existing customized approaches, we consider both regression and classification tasks with continuous features. The basic idea underlying our approach is to split the search space based on the quantiles of the feature distribution—leading to upper and lower bounds for the underlying optimization problem along the BnB iterations. Our proposed algorithm Quant-BnB shows significant speedups compared to existing approaches for shallow optimal trees on various real datasets.

----

## [669] Optimizing Tensor Network Contraction Using Reinforcement Learning

**Authors**: *Eli A. Meirom, Haggai Maron, Shie Mannor, Gal Chechik*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/meirom22a.html](https://proceedings.mlr.press/v162/meirom22a.html)

**Abstract**:

Quantum Computing (QC) stands to revolutionize computing, but is currently still limited. To develop and test quantum algorithms today, quantum circuits are often simulated on classical computers. Simulating a complex quantum circuit requires computing the contraction of a large network of tensors. The order (path) of contraction can have a drastic effect on the computing cost, but finding an efficient order is a challenging combinatorial optimization problem. We propose a Reinforcement Learning (RL) approach combined with Graph Neural Networks (GNN) to address the contraction ordering problem. The problem is extremely challenging due to the huge search space, the heavy-tailed reward distribution, and the challenging credit assignment. We show how a carefully implemented RL-agent that uses a GNN as the basic policy construct can address these challenges and obtain significant improvements over state-of-the-art techniques in three varieties of circuits, including the largest scale networks used in contemporary QC.

----

## [670] Causal Transformer for Estimating Counterfactual Outcomes

**Authors**: *Valentyn Melnychuk, Dennis Frauen, Stefan Feuerriegel*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/melnychuk22a.html](https://proceedings.mlr.press/v162/melnychuk22a.html)

**Abstract**:

Estimating counterfactual outcomes over time from observational data is relevant for many applications (e.g., personalized medicine). Yet, state-of-the-art methods build upon simple long short-term memory (LSTM) networks, thus rendering inferences for complex, long-range dependencies challenging. In this paper, we develop a novel Causal Transformer for estimating counterfactual outcomes over time. Our model is specifically designed to capture complex, long-range dependencies among time-varying confounders. For this, we combine three transformer subnetworks with separate inputs for time-varying covariates, previous treatments, and previous outcomes into a joint network with in-between cross-attentions. We further develop a custom, end-to-end training procedure for our Causal Transformer. Specifically, we propose a novel counterfactual domain confusion loss to address confounding bias: it aims to learn adversarial balanced representations, so that they are predictive of the next outcome but non-predictive of the current treatment assignment. We evaluate our Causal Transformer based on synthetic and real-world datasets, where it achieves superior performance over current baselines. To the best of our knowledge, this is the first work proposing transformer-based architecture for estimating counterfactual outcomes from longitudinal data.

----

## [671] Steerable 3D Spherical Neurons

**Authors**: *Pavlo Melnyk, Michael Felsberg, Mårten Wadenbäck*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/melnyk22a.html](https://proceedings.mlr.press/v162/melnyk22a.html)

**Abstract**:

Emerging from low-level vision theory, steerable filters found their counterpart in prior work on steerable convolutional neural networks equivariant to rigid transformations. In our work, we propose a steerable feed-forward learning-based approach that consists of neurons with spherical decision surfaces and operates on point clouds. Such spherical neurons are obtained by conformal embedding of Euclidean space and have recently been revisited in the context of learning representations of point sets. Focusing on 3D geometry, we exploit the isometry property of spherical neurons and derive a 3D steerability constraint. After training spherical neurons to classify point clouds in a canonical orientation, we use a tetrahedron basis to quadruplicate the neurons and construct rotation-equivariant spherical filter banks. We then apply the derived constraint to interpolate the filter bank outputs and, thus, obtain a rotation-invariant network. Finally, we use a synthetic point set and real-world 3D skeleton data to verify our theoretical findings. The code is available at https://github.com/pavlo-melnyk/steerable-3d-neurons.

----

## [672] Transformers are Meta-Reinforcement Learners

**Authors**: *Luckeciano C. Melo*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/melo22a.html](https://proceedings.mlr.press/v162/melo22a.html)

**Abstract**:

The transformer architecture and variants presented a remarkable success across many machine learning tasks in recent years. This success is intrinsically related to the capability of handling long sequences and the presence of context-dependent weights from the attention mechanism. We argue that these capabilities suit the central role of a Meta-Reinforcement Learning algorithm. Indeed, a meta-RL agent needs to infer the task from a sequence of trajectories. Furthermore, it requires a fast adaptation strategy to adapt its policy for a new task - which can be achieved using the self-attention mechanism. In this work, we present TrMRL (Transformers for Meta-Reinforcement Learning), a meta-RL agent that mimics the memory reinstatement mechanism using the transformer architecture. It associates the recent past of working memories to build an episodic memory recursively through the transformer layers. We show that the self-attention computes a consensus representation that minimizes the Bayes Risk at each layer and provides meaningful features to compute the best actions. We conducted experiments in high-dimensional continuous control environments for locomotion and dexterous manipulation. Results show that TrMRL presents comparable or superior asymptotic performance, sample efficiency, and out-of-distribution generalization compared to the baselines in these environments.

----

## [673] ButterflyFlow: Building Invertible Layers with Butterfly Matrices

**Authors**: *Chenlin Meng, Linqi Zhou, Kristy Choi, Tri Dao, Stefano Ermon*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/meng22a.html](https://proceedings.mlr.press/v162/meng22a.html)

**Abstract**:

Normalizing flows model complex probability distributions using maps obtained by composing invertible layers. Special linear layers such as masked and 1{\texttimes}1 convolutions play a key role in existing architectures because they increase expressive power while having tractable Jacobians and inverses. We propose a new family of invertible linear layers based on butterfly layers, which are known to theoretically capture complex linear structures including permutations and periodicity, yet can be inverted efficiently. This representational power is a key advantage of our approach, as such structures are common in many real-world datasets. Based on our invertible butterfly layers, we construct a new class of normalizing flow mod- els called ButterflyFlow. Empirically, we demonstrate that ButterflyFlows not only achieve strong density estimation results on natural images such as MNIST, CIFAR-10, and ImageNet-32{\texttimes}32, but also obtain significantly better log-likelihoods on structured datasets such as galaxy images and MIMIC-III patient cohorts{—}all while being more efficient in terms of memory and computation than relevant baselines.

----

## [674] In defense of dual-encoders for neural ranking

**Authors**: *Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Seungyeon Kim, Sashank J. Reddi, Sanjiv Kumar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/menon22a.html](https://proceedings.mlr.press/v162/menon22a.html)

**Abstract**:

Transformer-based models such as BERT have proven successful in information retrieval problem, which seek to identify relevant documents for a given query. There are two broad flavours of such models: cross-attention (CA) models, which learn a joint embedding for the query and document, and dual-encoder (DE) models, which learn separate embeddings for the query and document. Empirically, CA models are often found to be more accurate, which has motivated a series of works seeking to bridge this gap. However, a more fundamental question remains less explored: does this performance gap reflect an inherent limitation in the capacity of DE models, or a limitation in the training of such models? And does such an understanding suggest a principled means of improving DE models? In this paper, we study these questions, with three contributions. First, we establish theoretically that with a sufficiently large embedding dimension, DE models have the capacity to model a broad class of score distributions. Second, we show empirically that on real-world problems, DE models may overfit to spurious correlations in the training set, and thus under-perform on test samples. To mitigate this behaviour, we propose a suitable distillation strategy, and confirm its practical efficacy on the MSMARCO-Passage and Natural Questions benchmarks.

----

## [675] Equivariant Quantum Graph Circuits

**Authors**: *Péter Mernyei, Konstantinos Meichanetzidis, Ismail Ilkan Ceylan*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mernyei22a.html](https://proceedings.mlr.press/v162/mernyei22a.html)

**Abstract**:

We investigate quantum circuits for graph representation learning, and propose equivariant quantum graph circuits (EQGCs), as a class of parameterized quantum circuits with strong relational inductive bias for learning over graph-structured data. Conceptually, EQGCs serve as a unifying framework for quantum graph representation learning, allowing us to define several interesting subclasses which subsume existing proposals. In terms of the representation power, we prove that the studied subclasses of EQGCs are universal approximators for functions over the bounded graph domain. This theoretical perspective on quantum graph machine learning methods opens many directions for further work, and could lead to models with capabilities beyond those of classical approaches. We empirically verify the expressive power of EQGCs through a dedicated experiment on synthetic data, and additionally observe that the performance of EQGCs scales well with the depth of the model and does not suffer from barren plateu issues.

----

## [676] Stochastic Rising Bandits

**Authors**: *Alberto Maria Metelli, Francesco Trovò, Matteo Pirola, Marcello Restelli*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/metelli22a.html](https://proceedings.mlr.press/v162/metelli22a.html)

**Abstract**:

This paper is in the field of stochastic Multi-Armed Bandits (MABs), i.e., those sequential selection techniques able to learn online using only the feedback given by the chosen option (a.k.a. arm). We study a particular case of the rested and restless bandits in which the arms’ expected payoff is monotonically non-decreasing. This characteristic allows designing specifically crafted algorithms that exploit the regularity of the payoffs to provide tight regret bounds. We design an algorithm for the rested case (R-ed-UCB) and one for the restless case (R-less-UCB), providing a regret bound depending on the properties of the instance and, under certain circumstances, of $\widetilde{\mathcal{O}}(T^{\frac{2}{3}})$. We empirically compare our algorithms with state-of-the-art methods for non-stationary MABs over several synthetically generated tasks and an online model selection problem for a real-world dataset. Finally, using synthetic and real-world data, we illustrate the effectiveness of the proposed approaches compared with state-of-the-art algorithms for the non-stationary bandits.

----

## [677] Minimizing Control for Credit Assignment with Strong Feedback

**Authors**: *Alexander Meulemans, Matilde Tristany Farinha, Maria R. Cervera, João Sacramento, Benjamin F. Grewe*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/meulemans22a.html](https://proceedings.mlr.press/v162/meulemans22a.html)

**Abstract**:

The success of deep learning ignited interest in whether the brain learns hierarchical representations using gradient-based learning. However, current biologically plausible methods for gradient-based credit assignment in deep neural networks need infinitesimally small feedback signals, which is problematic in biologically realistic noisy environments and at odds with experimental evidence in neuroscience showing that top-down feedback can significantly influence neural activity. Building upon deep feedback control (DFC), a recently proposed credit assignment method, we combine strong feedback influences on neural activity with gradient-based learning and show that this naturally leads to a novel view on neural network optimization. Instead of gradually changing the network weights towards configurations with low output loss, weight updates gradually minimize the amount of feedback required from a controller that drives the network to the supervised output label. Moreover, we show that the use of strong feedback in DFC allows learning forward and feedback connections simultaneously, using learning rules fully local in space and time. We complement our theoretical results with experiments on standard computer-vision benchmarks, showing competitive performance to backpropagation as well as robustness to noise. Overall, our work presents a fundamentally novel view of learning as control minimization, while sidestepping biologically unrealistic assumptions.

----

## [678] A Dynamical System Perspective for Lipschitz Neural Networks

**Authors**: *Laurent Meunier, Blaise Delattre, Alexandre Araujo, Alexandre Allauzen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/meunier22a.html](https://proceedings.mlr.press/v162/meunier22a.html)

**Abstract**:

The Lipschitz constant of neural networks has been established as a key quantity to enforce the robustness to adversarial examples. In this paper, we tackle the problem of building $1$-Lipschitz Neural Networks. By studying Residual Networks from a continuous time dynamical system perspective, we provide a generic method to build $1$-Lipschitz Neural Networks and show that some previous approaches are special cases of this framework. Then, we extend this reasoning and show that ResNet flows derived from convex potentials define $1$-Lipschitz transformations, that lead us to define the Convex Potential Layer (CPL). A comprehensive set of experiments on several datasets demonstrates the scalability of our architecture and the benefits as an $\ell_2$-provable defense against adversarial examples. Our code is available at \url{https://github.com/MILES-PSL/Convex-Potential-Layer}

----

## [679] Distribution Regression with Sliced Wasserstein Kernels

**Authors**: *Dimitri Meunier, Massimiliano Pontil, Carlo Ciliberto*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/meunier22b.html](https://proceedings.mlr.press/v162/meunier22b.html)

**Abstract**:

The problem of learning functions over spaces of probabilities - or distribution regression - is gaining significant interest in the machine learning community. The main challenge in these settings is to identify a suitable representation capturing all relevant properties of a distribution. The well-established approach in this sense is to use kernel mean embeddings, which lift kernel-induced similarity on the input domain at the probability level. This strategy effectively tackles the two-stage sampling nature of the problem, enabling one to derive estimators with strong statistical guarantees, such as universal consistency and excess risk bounds. However, kernel mean embeddings implicitly hinge on the maximum mean discrepancy (MMD), a metric on probabilities, which is not the most suited to capture geometrical relations between distributions. In contrast, optimal transport (OT) metrics, are potentially more appealing. In this work, we propose an OT-based estimator for distribution regression. We build on the Sliced Wasserstein distance to obtain an OT-based representation. We study the theoretical properties of a kernel ridge regression estimator based on such representation, for which we prove universal consistency and excess risk bounds. Preliminary experiments complement our theoretical findings by showing the effectiveness of the proposed approach and compare it with MMD-based estimators.

----

## [680] Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism

**Authors**: *Siqi Miao, Mia Liu, Pan Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/miao22a.html](https://proceedings.mlr.press/v162/miao22a.html)

**Abstract**:

Interpretable graph learning is in need as many scientific applications depend on learning models to collect insights from graph-structured data. Previous works mostly focused on using post-hoc approaches to interpret pre-trained models (graph neural networks in particular). They argue against inherently interpretable models because the good interpretability of these models is often at the cost of their prediction accuracy. However, those post-hoc methods often fail to provide stable interpretation and may extract features that are spuriously correlated with the task. In this work, we address these issues by proposing Graph Stochastic Attention (GSAT). Derived from the information bottleneck principle, GSAT injects stochasticity to the attention weights to block the information from task-irrelevant graph components while learning stochasticity-reduced attention to select task-relevant subgraphs for interpretation. The selected subgraphs provably do not contain patterns that are spuriously correlated with the task under some assumptions. Extensive experiments on eight datasets show that GSAT outperforms the state-of-the-art methods by up to 20% in interpretation AUC and 5% in prediction accuracy. Our code is available at https://github.com/Graph-COM/GSAT.

----

## [681] Modeling Structure with Undirected Neural Networks

**Authors**: *Tsvetomila Mihaylova, Vlad Niculae, André F. T. Martins*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mihaylova22a.html](https://proceedings.mlr.press/v162/mihaylova22a.html)

**Abstract**:

Neural networks are powerful function estimators, leading to their status as a paradigm of choice for modeling structured data. However, unlike other structured representations that emphasize the modularity of the problem {–} e.g., factor graphs {–} neural networks are usually monolithic mappings from inputs to outputs, with a fixed computation order. This limitation prevents them from capturing different directions of computation and interaction between the modeled variables. In this paper, we combine the representational strengths of factor graphs and of neural networks, proposing undirected neural networks (UNNs): a flexible framework for specifying computations that can be performed in any order. For particular choices, our proposed models subsume and extend many existing architectures: feed-forward, recurrent, self-attention networks, auto-encoders, and networks with implicit layers. We demonstrate the effectiveness of undirected neural architectures, both unstructured and structured, on a range of tasks: tree-constrained dependency parsing, convolutional image classification, and sequence completion with attention. By varying the computation order, we show how a single UNN can be used both as a classifier and a prototype generator, and how it can fill in missing parts of an input sequence, making them a promising field for further research.

----

## [682] Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Models

**Authors**: *Beren Millidge, Tommaso Salvatori, Yuhang Song, Thomas Lukasiewicz, Rafal Bogacz*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/millidge22a.html](https://proceedings.mlr.press/v162/millidge22a.html)

**Abstract**:

A large number of neural network models of associative memory have been proposed in the literature. These include the classical Hopfield networks (HNs), sparse distributed memories (SDMs), and more recently the modern continuous Hopfield networks (MCHNs), which possess close links with self-attention in machine learning. In this paper, we propose a general framework for understanding the operation of such memory networks as a sequence of three operations: similarity, separation, and projection. We derive all these memory models as instances of our general framework with differing similarity and separation functions. We extend the mathematical framework of Krotov et al (2020) to express general associative memory models using neural network dynamics with local computation, and derive a general energy function that is a Lyapunov function of the dynamics. Finally, using our framework, we empirically investigate the capacity of using different similarity functions for these associative memory models, beyond the dot product similarity measure, and demonstrate empirically that Euclidean or Manhattan distance similarity metrics perform substantially better in practice on many tasks, enabling a more robust retrieval and higher memory capacity than existing models.

----

## [683] Learning Stochastic Shortest Path with Linear Function Approximation

**Authors**: *Yifei Min, Jiafan He, Tianhao Wang, Quanquan Gu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/min22a.html](https://proceedings.mlr.press/v162/min22a.html)

**Abstract**:

We study the stochastic shortest path (SSP) problem in reinforcement learning with linear function approximation, where the transition kernel is represented as a linear mixture of unknown models. We call this class of SSP problems as linear mixture SSPs. We propose a novel algorithm with Hoeffding-type confidence sets for learning the linear mixture SSP, which can attain an $\tilde{\mathcal{O}}(d B_{\star}^{1.5}\sqrt{K/c_{\min}})$ regret. Here $K$ is the number of episodes, $d$ is the dimension of the feature mapping in the mixture model, $B_{\star}$ bounds the expected cumulative cost of the optimal policy, and $c_{\min}>0$ is the lower bound of the cost function. Our algorithm also applies to the case when $c_{\min} = 0$, and an $\tilde{\mathcal{O}}(K^{2/3})$ regret is guaranteed. To the best of our knowledge, this is the first algorithm with a sublinear regret guarantee for learning linear mixture SSP. Moreover, we design a refined Bernstein-type confidence set and propose an improved algorithm, which provably achieves an $\tilde{\mathcal{O}}(d B_{\star}\sqrt{K/c_{\min}})$ regret. In complement to the regret upper bounds, we also prove a lower bound of $\Omega(dB_{\star} \sqrt{K})$. Hence, our improved algorithm matches the lower bound up to a $1/\sqrt{c_{\min}}$ factor and poly-logarithmic factors, achieving a near-optimal regret guarantee.

----

## [684] Prioritized Training on Points that are Learnable, Worth Learning, and not yet Learnt

**Authors**: *Sören Mindermann, Jan Markus Brauner, Muhammed Razzak, Mrinank Sharma, Andreas Kirsch, Winnie Xu, Benedikt Höltgen, Aidan N. Gomez, Adrien Morisot, Sebastian Farquhar, Yarin Gal*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mindermann22a.html](https://proceedings.mlr.press/v162/mindermann22a.html)

**Abstract**:

Training on web-scale data can take months. But much computation and time is wasted on redundant and noisy points that are already learnt or not learnable. To accelerate training, we introduce Reducible Holdout Loss Selection (RHO-LOSS), a simple but principled technique which selects approximately those points for training that most reduce the model’s generalization loss. As a result, RHO-LOSS mitigates the weaknesses of existing data selection methods: techniques from the optimization literature typically select "hard" (e.g. high loss) points, but such points are often noisy (not learnable) or less task-relevant. Conversely, curriculum learning prioritizes "easy" points, but such points need not be trained on once learned. In contrast, RHO-LOSS selects points that are learnable, worth learning, and not yet learnt. RHO-LOSS trains in far fewer steps than prior art, improves accuracy, and speeds up training on a wide range of datasets, hyperparameters, and architectures (MLPs, CNNs, and BERT). On the large web-scraped image dataset Clothing-1M, RHO-LOSS trains in 18x fewer steps and reaches 2% higher final accuracy than uniform data shuffling.

----

## [685] POEM: Out-of-Distribution Detection with Posterior Sampling

**Authors**: *Yifei Ming, Ying Fan, Yixuan Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ming22a.html](https://proceedings.mlr.press/v162/ming22a.html)

**Abstract**:

Out-of-distribution (OOD) detection is indispensable for machine learning models deployed in the open world. Recently, the use of an auxiliary outlier dataset during training (also known as outlier exposure) has shown promising performance. As the sample space for potential OOD data can be prohibitively large, sampling informative outliers is essential. In this work, we propose a novel posterior sampling based outlier mining framework, POEM, which facilitates efficient use of outlier data and promotes learning a compact decision boundary between ID and OOD data for improved detection. We show that POEM establishes state-of-the-art performance on common benchmarks. Compared to the current best method that uses a greedy sampling strategy, POEM improves the relative performance by 42.0% and 24.2% (FPR95) on CIFAR-10 and CIFAR-100, respectively. We further provide theoretical insights on the effectiveness of POEM for OOD detection.

----

## [686] A Simple Reward-free Approach to Constrained Reinforcement Learning

**Authors**: *Sobhan Miryoosefi, Chi Jin*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/miryoosefi22a.html](https://proceedings.mlr.press/v162/miryoosefi22a.html)

**Abstract**:

In constrained reinforcement learning (RL), a learning agent seeks to not only optimize the overall reward but also satisfy the additional safety, diversity, or budget constraints. Consequently, existing constrained RL solutions require several new algorithmic ingredients that are notably different from standard RL. On the other hand, reward-free RL is independently developed in the unconstrained literature, which learns the transition dynamics without using the reward information, and thus naturally capable of addressing RL with multiple objectives under the common dynamics. This paper bridges reward-free RL and constrained RL. Particularly, we propose a simple meta-algorithm such that given any reward-free RL oracle, the approachability and constrained RL problems can be directly solved with negligible overheads in sample complexity. Utilizing the existing reward-free RL solvers, our framework provides sharp sample complexity results for constrained RL in the tabular MDP setting, matching the best existing results up to a factor of horizon dependence; our framework directly extends to a setting of tabular two-player Markov games, and gives a new result for constrained RL with linear function approximation.

----

## [687] Wide Neural Networks Forget Less Catastrophically

**Authors**: *Seyed-Iman Mirzadeh, Arslan Chaudhry, Dong Yin, Huiyi Hu, Razvan Pascanu, Dilan Görür, Mehrdad Farajtabar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mirzadeh22a.html](https://proceedings.mlr.press/v162/mirzadeh22a.html)

**Abstract**:

A primary focus area in continual learning research is alleviating the "catastrophic forgetting" problem in neural networks by designing new algorithms that are more robust to the distribution shifts. While the recent progress in continual learning literature is encouraging, our understanding of what properties of neural networks contribute to catastrophic forgetting is still limited. To address this, instead of focusing on continual learning algorithms, in this work, we focus on the model itself and study the impact of "width" of the neural network architecture on catastrophic forgetting, and show that width has a surprisingly significant effect on forgetting. To explain this effect, we study the learning dynamics of the network from various perspectives such as gradient orthogonality, sparsity, and lazy training regime. We provide potential explanations that are consistent with the empirical results across different architectures and continual learning benchmarks.

----

## [688] Proximal and Federated Random Reshuffling

**Authors**: *Konstantin Mishchenko, Ahmed Khaled, Peter Richtárik*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mishchenko22a.html](https://proceedings.mlr.press/v162/mishchenko22a.html)

**Abstract**:

Random Reshuffling (RR), also known as Stochastic Gradient Descent (SGD) without replacement, is a popular and theoretically grounded method for finite-sum minimization. We propose two new algorithms: Proximal and Federated Random Reshuffling (ProxRR and FedRR). The first algorithm, ProxRR, solves composite finite-sum minimization problems in which the objective is the sum of a (potentially non-smooth) convex regularizer and an average of $n$ smooth objectives. ProxRR evaluates the proximal operator once per epoch only. When the proximal operator is expensive to compute, this small difference makes ProxRR up to $n$ times faster than algorithms that evaluate the proximal operator in every iteration, such as proximal (stochastic) gradient descent. We give examples of practical optimization tasks where the proximal operator is difficult to compute and ProxRR has a clear advantage. One such task is federated or distributed optimization, where the evaluation of the proximal operator corresponds to communication across the network. We obtain our second algorithm, FedRR, as a special case of ProxRR applied to federated optimization, and prove it has a smaller communication footprint than either distributed gradient descent or Local SGD. Our theory covers both constant and decreasing stepsizes, and allows for importance resampling schemes that can improve conditioning, which may be of independent interest. Our theory covers both convex and nonconvex regimes. Finally, we corroborate our results with experiments on real data sets.

----

## [689] ProxSkip: Yes! Local Gradient Steps Provably Lead to Communication Acceleration! Finally!

**Authors**: *Konstantin Mishchenko, Grigory Malinovsky, Sebastian U. Stich, Peter Richtárik*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mishchenko22b.html](https://proceedings.mlr.press/v162/mishchenko22b.html)

**Abstract**:

We introduce ProxSkip—a surprisingly simple and provably efficient method for minimizing the sum of a smooth ($f$) and an expensive nonsmooth proximable ($\psi$) function. The canonical approach to solving such problems is via the proximal gradient descent (ProxGD) algorithm, which is based on the evaluation of the gradient of $f$ and the prox operator of $\psi$ in each iteration. In this work we are specifically interested in the regime in which the evaluation of prox is costly relative to the evaluation of the gradient, which is the case in many applications. ProxSkip allows for the expensive prox operator to be skipped in most iterations: while its iteration complexity is $\mathcal{O}(\kappa \log \nicefrac{1}{\varepsilon})$, where $\kappa$ is the condition number of $f$, the number of prox evaluations is $\mathcal{O}(\sqrt{\kappa} \log \nicefrac{1}{\varepsilon})$ only. Our main motivation comes from federated learning, where evaluation of the gradient operator corresponds to taking a local GD step independently on all devices, and evaluation of prox corresponds to (expensive) communication in the form of gradient averaging. In this context, ProxSkip offers an effective acceleration of communication complexity. Unlike other local gradient-type methods, such as FedAvg, SCAFFOLD, S-Local-GD and FedLin, whose theoretical communication complexity is worse than, or at best matching, that of vanilla GD in the heterogeneous data regime, we obtain a provable and large improvement without any heterogeneity-bounding assumptions.

----

## [690] Fast Convex Optimization for Two-Layer ReLU Networks: Equivalent Model Classes and Cone Decompositions

**Authors**: *Aaron Mishkin, Arda Sahiner, Mert Pilanci*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mishkin22a.html](https://proceedings.mlr.press/v162/mishkin22a.html)

**Abstract**:

We develop fast algorithms and robust software for convex optimization of two-layer neural networks with ReLU activation functions. Our work leverages a convex re-formulation of the standard weight-decay penalized training problem as a set of group-l1-regularized data-local models, where locality is enforced by polyhedral cone constraints. In the special case of zero-regularization, we show that this problem is exactly equivalent to unconstrained optimization of a convex "gated ReLU" network. For problems with non-zero regularization, we show that convex gated ReLU models obtain data-dependent approximation bounds for the ReLU training problem. To optimize the convex re-formulations, we develop an accelerated proximal gradient method and a practical augmented Lagrangian solver. We show that these approaches are faster than standard training heuristics for the non-convex problem, such as SGD, and outperform commercial interior-point solvers. Experimentally, we verify our theoretical results, explore the group-l1 regularization path, and scale convex optimization for neural networks to image classification on MNIST and CIFAR-10.

----

## [691] Memory-Based Model Editing at Scale

**Authors**: *Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, Chelsea Finn*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mitchell22a.html](https://proceedings.mlr.press/v162/mitchell22a.html)

**Abstract**:

Even the largest neural networks make errors, and once-correct predictions can become invalid as the world changes. Model editors make local updates to the behavior of base (pre-trained) models to inject updated knowledge or correct undesirable behaviors. Existing model editors have shown promise, but also suffer from insufficient expressiveness: they struggle to accurately model an edit’s intended scope (examples affected by the edit), leading to inaccurate predictions for test inputs loosely related to the edit, and they often fail altogether after many edits. As a higher-capacity alternative, we propose Semi-Parametric Editing with a Retrieval-Augmented Counterfactual Model (SERAC), which stores edits in an explicit memory and learns to reason over them to modulate the base model’s predictions as needed. To enable more rigorous evaluation of model editors, we introduce three challenging language model editing problems based on question answering, fact-checking, and dialogue generation. We find that only SERAC achieves high performance on all three problems, consistently outperforming existing approaches to model editing by a significant margin. Code, data, and additional project information will be made available at https://sites.google.com/view/serac-editing.

----

## [692] Invariant Ancestry Search

**Authors**: *Phillip B. Mogensen, Nikolaj Thams, Jonas Peters*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mogensen22a.html](https://proceedings.mlr.press/v162/mogensen22a.html)

**Abstract**:

Recently, methods have been proposed that exploit the invariance of prediction models with respect to changing environments to infer subsets of the causal parents of a response variable. If the environments influence only few of the underlying mechanisms, the subset identified by invariant causal prediction (ICP), for example, may be small, or even empty. We introduce the concept of minimal invariance and propose invariant ancestry search (IAS). In its population version, IAS outputs a set which contains only ancestors of the response and is a superset of the output of ICP. When applied to data, corresponding guarantees hold asymptotically if the underlying test for invariance has asymptotic level and power. We develop scalable algorithms and perform experiments on simulated and real data.

----

## [693] Differentially Private Community Detection for Stochastic Block Models

**Authors**: *Mohamed S. Mohamed, Dung Nguyen, Anil Vullikanti, Ravi Tandon*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mohamed22a.html](https://proceedings.mlr.press/v162/mohamed22a.html)

**Abstract**:

The goal of community detection over graphs is to recover underlying labels/attributes of users (e.g., political affiliation) given the connectivity between users. There has been significant recent progress on understanding the fundamental limits of community detection when the graph is generated from a stochastic block model (SBM). Specifically, sharp information theoretic limits and efficient algorithms have been obtained for SBMs as a function of $p$ and $q$, which represent the intra-community and inter-community connection probabilities. In this paper, we study the community detection problem while preserving the privacy of the individual connections between the vertices. Focusing on the notion of $(\epsilon, \delta)$-edge differential privacy (DP), we seek to understand the fundamental tradeoffs between $(p, q)$, DP budget $(\epsilon, \delta)$, and computational efficiency for exact recovery of community labels. To this end, we present and analyze the associated information-theoretic tradeoffs for three differentially private community recovery mechanisms: a) stability based mechanism; b) sampling based mechanisms; and c) graph perturbation mechanisms. Our main findings are that stability and sampling based mechanisms lead to a superior tradeoff between $(p,q)$ and the privacy budget $(\epsilon, \delta)$; however this comes at the expense of higher computational complexity. On the other hand, albeit low complexity, graph perturbation mechanisms require the privacy budget $\epsilon$ to scale as $\Omega(\log(n))$ for exact recovery.

----

## [694] A Multi-objective / Multi-task Learning Framework Induced by Pareto Stationarity

**Authors**: *Michinari Momma, Chaosheng Dong, Jia Liu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/momma22a.html](https://proceedings.mlr.press/v162/momma22a.html)

**Abstract**:

Multi-objective optimization (MOO) and multi-task learning (MTL) have gained much popularity with prevalent use cases such as production model development of regression / classification / ranking models with MOO, and training deep learning models with MTL. Despite the long history of research in MOO, its application to machine learning requires development of solution strategy, and algorithms have recently been developed to solve specific problems such as discovery of any Pareto optimal (PO) solution, and that with a particular form of preference. In this paper, we develop a novel and generic framework to discover a PO solution with multiple forms of preferences. It allows us to formulate a generic MOO / MTL problem to express a preference, which is solved to achieve both alignment with the preference and PO, at the same time. Specifically, we apply the framework to solve the weighted Chebyshev problem and an extension of that. The former is known as a method to discover the Pareto front, the latter helps to find a model that outperforms an existing model with only one run. Experimental results demonstrate not only the method achieves competitive performance with existing methods, but also it allows us to achieve the performance from different forms of preferences.

----

## [695] EqR: Equivariant Representations for Data-Efficient Reinforcement Learning

**Authors**: *Arnab Kumar Mondal, Vineet Jain, Kaleem Siddiqi, Siamak Ravanbakhsh*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mondal22a.html](https://proceedings.mlr.press/v162/mondal22a.html)

**Abstract**:

We study a variety of notions of equivariance as an inductive bias in Reinforcement Learning (RL). In particular, we propose new mechanisms for learning representations that are equivariant to both the agent’s action, as well as symmetry transformations of the state-action pairs. Whereas prior work on exploiting symmetries in deep RL can only incorporate predefined linear transformations, our approach allows non-linear symmetry transformations of state-action pairs to be learned from the data. This is achieved through 1) equivariant Lie algebraic parameterization of state and action encodings, 2) equivariant latent transition models, and 3) the incorporation of symmetry-based losses. We demonstrate the advantages of our method, which we call Equivariant representations for RL (EqR), for Atari games in a data-efficient setting limited to 100K steps of interactions with the environment.

----

## [696] Feature and Parameter Selection in Stochastic Linear Bandits

**Authors**: *Ahmadreza Moradipari, Berkay Turan, Yasin Abbasi-Yadkori, Mahnoosh Alizadeh, Mohammad Ghavamzadeh*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/moradipari22a.html](https://proceedings.mlr.press/v162/moradipari22a.html)

**Abstract**:

We study two model selection settings in stochastic linear bandits (LB). In the first setting, which we refer to as feature selection, the expected reward of the LB problem is in the linear span of at least one of $M$ feature maps (models). In the second setting, the reward parameter of the LB problem is arbitrarily selected from $M$ models represented as (possibly) overlapping balls in $\mathbb R^d$. However, the agent only has access to misspecified models, i.e., estimates of the centers and radii of the balls. We refer to this setting as parameter selection. For each setting, we develop and analyze a computationally efficient algorithm that is based on a reduction from bandits to full-information problems. This allows us to obtain regret bounds that are not worse (up to a $\sqrt{\log M}$ factor) than the case where the true model is known. This is the best reported dependence on the number of models $M$ in these settings. Finally, we empirically show the effectiveness of our algorithms using synthetic and real-world experiments.

----

## [697] Power-Law Escape Rate of SGD

**Authors**: *Takashi Mori, Ziyin Liu, Kangqiao Liu, Masahito Ueda*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mori22a.html](https://proceedings.mlr.press/v162/mori22a.html)

**Abstract**:

Stochastic gradient descent (SGD) undergoes complicated multiplicative noise for the mean-square loss. We use this property of SGD noise to derive a stochastic differential equation (SDE) with simpler additive noise by performing a random time change. Using this formalism, we show that the log loss barrier $\Delta\log L=\log[L(\theta^s)/L(\theta^*)]$ between a local minimum $\theta^*$ and a saddle $\theta^s$ determines the escape rate of SGD from the local minimum, contrary to the previous results borrowing from physics that the linear loss barrier $\Delta L=L(\theta^s)-L(\theta^*)$ decides the escape rate. Our escape-rate formula strongly depends on the typical magnitude $h^*$ and the number $n$ of the outlier eigenvalues of the Hessian. This result explains an empirical fact that SGD prefers flat minima with low effective dimensions, giving an insight into implicit biases of SGD.

----

## [698] Rethinking Fano's Inequality in Ensemble Learning

**Authors**: *Terufumi Morishita, Gaku Morio, Shota Horiguchi, Hiroaki Ozaki, Nobuo Nukaga*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/morishita22a.html](https://proceedings.mlr.press/v162/morishita22a.html)

**Abstract**:

We propose a fundamental theory on ensemble learning that evaluates a given ensemble system by a well-grounded set of metrics. Previous studies used a variant of Fano’s inequality of information theory and derived a lower bound of the classification error rate on the basis of the accuracy and diversity of models. We revisit the original Fano’s inequality and argue that the studies did not take into account the information lost when multiple model predictions are combined into a final prediction. To address this issue, we generalize the previous theory to incorporate the information loss. Further, we empirically validate and demonstrate the proposed theory through extensive experiments on actual systems. The theory reveals the strengths and weaknesses of systems on each metric, which will push the theoretical understanding of ensemble learning and give us insights into designing systems.

----

## [699] SpeqNets: Sparsity-aware permutation-equivariant graph networks

**Authors**: *Christopher Morris, Gaurav Rattan, Sandra Kiefer, Siamak Ravanbakhsh*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/morris22a.html](https://proceedings.mlr.press/v162/morris22a.html)

**Abstract**:

While message-passing graph neural networks have clear limitations in approximating permutation-equivariant functions over graphs or general relational data, more expressive, higher-order graph neural networks do not scale to large graphs. They either operate on $k$-order tensors or consider all $k$-node subgraphs, implying an exponential dependence on $k$ in memory requirements, and do not adapt to the sparsity of the graph. By introducing new heuristics for the graph isomorphism problem, we devise a class of universal, permutation-equivariant graph networks, which, unlike previous architectures, offer a fine-grained control between expressivity and scalability and adapt to the sparsity of the graph. These architectures lead to vastly reduced computation times compared to standard higher-order graph networks in the supervised node- and graph-level classification and regression regime while significantly improving standard graph neural network and graph kernel architectures in terms of predictive performance.

----

## [700] CtrlFormer: Learning Transferable State Representation for Visual Control via Transformer

**Authors**: *Yao Mark Mu, Shoufa Chen, Mingyu Ding, Jianyu Chen, Runjian Chen, Ping Luo*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mu22a.html](https://proceedings.mlr.press/v162/mu22a.html)

**Abstract**:

Transformer has achieved great successes in learning vision and language representation, which is general across various downstream tasks. In visual control, learning transferable state representation that can transfer between different control tasks is important to reduce the training sample size. However, porting Transformer to sample-efficient visual control remains a challenging and unsolved problem. To this end, we propose a novel Control Transformer (CtrlFormer), possessing many appealing benefits that prior arts do not have. Firstly, CtrlFormer jointly learns self-attention mechanisms between visual tokens and policy tokens among different control tasks, where multitask representation can be learned and transferred without catastrophic forgetting. Secondly, we carefully design a contrastive reinforcement learning paradigm to train CtrlFormer, enabling it to achieve high sample efficiency, which is important in control problems. For example, in the DMControl benchmark, unlike recent advanced methods that failed by producing a zero score in the “Cartpole” task after transfer learning with 100k samples, CtrlFormer can achieve a state-of-the-art score with only 100k samples while maintaining the performance of previous tasks. The code and models are released in our project homepage.

----

## [701] Generalized Beliefs for Cooperative AI

**Authors**: *Darius Muglich, Luisa M. Zintgraf, Christian A. Schröder de Witt, Shimon Whiteson, Jakob N. Foerster*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/muglich22a.html](https://proceedings.mlr.press/v162/muglich22a.html)

**Abstract**:

Self-play is a common method for constructing solutions in Markov games that can yield optimal policies in collaborative settings. However, these policies often adopt highly-specialized conventions that make playing with a novel partner difficult. To address this, recent approaches rely on encoding symmetry and convention-awareness into policy training, but these require strong environmental assumptions and can complicate policy training. To overcome this, we propose moving the learning of conventions to the belief space. Specifically, we propose a belief learning paradigm that can maintain beliefs over rollouts of policies not seen at training time, and can thus decode and adapt to novel conventions at test time. We show how to leverage this belief model for both search and training of a best response over a pool of policies to greatly improve zero-shot coordination. We also show how our paradigm promotes explainability and interpretability of nuanced agent conventions.

----

## [702] Bounding the Width of Neural Networks via Coupled Initialization A Worst Case Analysis

**Authors**: *Alexander Munteanu, Simon Omlor, Zhao Song, David P. Woodruff*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/munteanu22a.html](https://proceedings.mlr.press/v162/munteanu22a.html)

**Abstract**:

A common method in training neural networks is to initialize all the weights to be independent Gaussian vectors. We observe that by instead initializing the weights into independent pairs, where each pair consists of two identical Gaussian vectors, we can significantly improve the convergence analysis. While a similar technique has been studied for random inputs [Daniely, NeurIPS 2020], it has not been analyzed with arbitrary inputs. Using this technique, we show how to significantly reduce the number of neurons required for two-layer ReLU networks, both in the under-parameterized setting with logistic loss, from roughly $\gamma^{-8}$ [Ji and Telgarsky, ICLR 2020] to $\gamma^{-2}$, where $\gamma$ denotes the separation margin with a Neural Tangent Kernel, as well as in the over-parameterized setting with squared loss, from roughly $n^4$ [Song and Yang, 2019] to $n^2$, implicitly also improving the recent running time bound of [Brand, Peng, Song and Weinstein, ITCS 2021]. For the under-parameterized setting we also prove new lower bounds that improve upon prior work, and that under certain assumptions, are best possible.

----

## [703] Constants Matter: The Performance Gains of Active Learning

**Authors**: *Stephen O. Mussmann, Sanjoy Dasgupta*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mussmann22a.html](https://proceedings.mlr.press/v162/mussmann22a.html)

**Abstract**:

Within machine learning, active learning studies the gains in performance made possible by adaptively selecting data points to label. In this work, we show through upper and lower bounds, that for a simple benign setting of well-specified logistic regression on a uniform distribution over a sphere, the expected excess error of both active learning and random sampling have the same inverse proportional dependence on the number of samples. Importantly, due to the nature of lower bounds, any more general setting does not allow a better dependence on the number of samples. Additionally, we show a variant of uncertainty sampling can achieve a faster rate of convergence than random sampling by a factor of the Bayes error, a recent empirical observation made by other work. Qualitatively, this work is pessimistic with respect to the asymptotic dependence on the number of samples, but optimistic with respect to finding performance gains in the constants.

----

## [704] On the Generalization Analysis of Adversarial Learning

**Authors**: *Waleed Mustafa, Yunwen Lei, Marius Kloft*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mustafa22a.html](https://proceedings.mlr.press/v162/mustafa22a.html)

**Abstract**:

Many recent studies have highlighted the susceptibility of virtually all machine-learning models to adversarial attacks. Adversarial attacks are imperceptible changes to an input example of a given prediction model. Such changes are carefully designed to alter the otherwise correct prediction of the model. In this paper, we study the generalization properties of adversarial learning. In particular, we derive high-probability generalization bounds on the adversarial risk in terms of the empirical adversarial risk, the complexity of the function class and the adversarial noise set. Our bounds are generally applicable to many models, losses, and adversaries. We showcase its applicability by deriving adversarial generalization bounds for the multi-class classification setting and various prediction models (including linear models and Deep Neural Networks). We also derive optimistic adversarial generalization bounds for the case of smooth losses. These are the first fast-rate bounds valid for adversarial deep learning to the best of our knowledge.

----

## [705] Universal and data-adaptive algorithms for model selection in linear contextual bandits

**Authors**: *Vidya K. Muthukumar, Akshay Krishnamurthy*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/muthukumar22a.html](https://proceedings.mlr.press/v162/muthukumar22a.html)

**Abstract**:

Model selection in contextual bandits is an important complementary problem to regret minimization with respect to a fixed model class. We consider the simplest non-trivial instance of model-selection: distinguishing a simple multi-armed bandit problem from a linear contextual bandit problem. Even in this instance, current state-of-the-art methods explore in a suboptimal manner and require strong "feature-diversity" conditions. In this paper, we introduce new algorithms that a) explore in a data-adaptive manner, and b) provide model selection guarantees of the form O(d^{\alpha} T^{1 - \alpha}) with no feature diversity conditions whatsoever, where d denotes the dimension of the linear model and T denotes the total number of rounds. The first algorithm enjoys a "best-of-both-worlds" property, recovering two prior results that hold under distinct distributional assumptions, simultaneously. The second removes distributional assumptions altogether, expanding the scope for tractable model selection. Our approach extends to model selection among nested linear contextual bandits under some additional assumptions.

----

## [706] The Importance of Non-Markovianity in Maximum State Entropy Exploration

**Authors**: *Mirco Mutti, Riccardo De Santi, Marcello Restelli*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/mutti22a.html](https://proceedings.mlr.press/v162/mutti22a.html)

**Abstract**:

In the maximum state entropy exploration framework, an agent interacts with a reward-free environment to learn a policy that maximizes the entropy of the expected state visitations it is inducing. Hazan et al. (2019) noted that the class of Markovian stochastic policies is sufficient for the maximum state entropy objective, and exploiting non-Markovianity is generally considered pointless in this setting. In this paper, we argue that non-Markovianity is instead paramount for maximum state entropy exploration in a finite-sample regime. Especially, we recast the objective to target the expected entropy of the induced state visitations in a single trial. Then, we show that the class of non-Markovian deterministic policies is sufficient for the introduced objective, while Markovian policies suffer non-zero regret in general. However, we prove that the problem of finding an optimal non-Markovian policy is NP-hard. Despite this negative result, we discuss avenues to address the problem in a tractable way and how non-Markovian exploration could benefit the sample efficiency of online reinforcement learning in future works.

----

## [707] PAC-Net: A Model Pruning Approach to Inductive Transfer Learning

**Authors**: *Sanghoon Myung, In Huh, Wonik Jang, Jae Myung Choe, Jisu Ryu, Daesin Kim, Kee-Eung Kim, Changwook Jeong*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/myung22a.html](https://proceedings.mlr.press/v162/myung22a.html)

**Abstract**:

Inductive transfer learning aims to learn from a small amount of training data for the target task by utilizing a pre-trained model from the source task. Most strategies that involve large-scale deep learning models adopt initialization with the pre-trained model and fine-tuning for the target task. However, when using over-parameterized models, we can often prune the model without sacrificing the accuracy of the source task. This motivates us to adopt model pruning for transfer learning with deep learning models. In this paper, we propose PAC-Net, a simple yet effective approach for transfer learning based on pruning. PAC-Net consists of three steps: Prune, Allocate, and Calibrate (PAC). The main idea behind these steps is to identify essential weights for the source task, fine-tune on the source task by updating the essential weights, and then calibrate on the target task by updating the remaining redundant weights. Under the various and extensive set of inductive transfer learning experiments, we show that our method achieves state-of-the-art performance by a large margin.

----

## [708] AutoSNN: Towards Energy-Efficient Spiking Neural Networks

**Authors**: *Byunggook Na, Jisoo Mok, Seongsik Park, Dongjin Lee, Hyeokjun Choe, Sungroh Yoon*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/na22a.html](https://proceedings.mlr.press/v162/na22a.html)

**Abstract**:

Spiking neural networks (SNNs) that mimic information transmission in the brain can energy-efficiently process spatio-temporal information through discrete and sparse spikes, thereby receiving considerable attention. To improve accuracy and energy efficiency of SNNs, most previous studies have focused solely on training methods, and the effect of architecture has rarely been studied. We investigate the design choices used in the previous studies in terms of the accuracy and number of spikes and figure out that they are not best-suited for SNNs. To further improve the accuracy and reduce the spikes generated by SNNs, we propose a spike-aware neural architecture search framework called AutoSNN. We define a search space consisting of architectures without undesirable design choices. To enable the spike-aware architecture search, we introduce a fitness that considers both the accuracy and number of spikes. AutoSNN successfully searches for SNN architectures that outperform hand-crafted SNNs in accuracy and energy efficiency. We thoroughly demonstrate the effectiveness of AutoSNN on various datasets including neuromorphic datasets.

----

## [709] Implicit Bias of the Step Size in Linear Diagonal Neural Networks

**Authors**: *Mor Shpigel Nacson, Kavya Ravichandran, Nathan Srebro, Daniel Soudry*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nacson22a.html](https://proceedings.mlr.press/v162/nacson22a.html)

**Abstract**:

Focusing on diagonal linear networks as a model for understanding the implicit bias in underdetermined models, we show how the gradient descent step size can have a large qualitative effect on the implicit bias, and thus on generalization ability. In particular, we show how using large step size for non-centered data can change the implicit bias from a "kernel" type behavior to a "rich" (sparsity-inducing) regime — even when gradient flow, studied in previous works, would not escape the "kernel" regime. We do so by using dynamic stability, proving that convergence to dynamically stable global minima entails a bound on some weighted $\ell_1$-norm of the linear predictor, i.e. a "rich" regime. We prove this leads to good generalization in a sparse regression setting.

----

## [710] DNNR: Differential Nearest Neighbors Regression

**Authors**: *Youssef Nader, Leon Sixt, Tim Landgraf*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nader22a.html](https://proceedings.mlr.press/v162/nader22a.html)

**Abstract**:

K-nearest neighbors (KNN) is one of the earliest and most established algorithms in machine learning. For regression tasks, KNN averages the targets within a neighborhood which poses a number of challenges: the neighborhood definition is crucial for the predictive performance as neighbors might be selected based on uninformative features, and averaging does not account for how the function changes locally. We propose a novel method called Differential Nearest Neighbors Regression (DNNR) that addresses both issues simultaneously: during training, DNNR estimates local gradients to scale the features; during inference, it performs an n-th order Taylor approximation using estimated gradients. In a large-scale evaluation on over 250 datasets, we find that DNNR performs comparably to state-of-the-art gradient boosting methods and MLPs while maintaining the simplicity and transparency of KNN. This allows us to derive theoretical error bounds and inspect failures. In times that call for transparency of ML models, DNNR provides a good balance between performance and interpretability.

----

## [711] Overcoming Oscillations in Quantization-Aware Training

**Authors**: *Markus Nagel, Marios Fournarakis, Yelysei Bondarenko, Tijmen Blankevoort*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nagel22a.html](https://proceedings.mlr.press/v162/nagel22a.html)

**Abstract**:

When training neural networks with simulated quantization, we observe that quantized weights can, rather unexpectedly, oscillate between two grid-points. The importance of this effect and its impact on quantization-aware training (QAT) are not well-understood or investigated in literature. In this paper, we delve deeper into the phenomenon of weight oscillations and show that it can lead to a significant accuracy degradation due to wrongly estimated batch-normalization statistics during inference and increased noise during training. These effects are particularly pronounced in low-bit ($\leq$ 4-bits) quantization of efficient networks with depth-wise separable layers, such as MobileNets and EfficientNets. In our analysis we investigate several previously proposed QAT algorithms and show that most of these are unable to overcome oscillations. Finally, we propose two novel QAT algorithms to overcome oscillations during training: oscillation dampening and iterative weight freezing. We demonstrate that our algorithms achieve state-of-the-art accuracy for low-bit (3 & 4 bits) weight and activation quantization of efficient architectures, such as MobileNetV2, MobileNetV3, and EfficentNet-lite on ImageNet. Our source code is available at https://github.com/qualcomm-ai-research/oscillations-qat.

----

## [712] Strategic Representation

**Authors**: *Vineet Nair, Ganesh Ghalme, Inbal Talgam-Cohen, Nir Rosenfeld*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nair22a.html](https://proceedings.mlr.press/v162/nair22a.html)

**Abstract**:

Humans have come to rely on machines for reducing excessive information to manageable representations. But this reliance can be abused – strategic machines might craft representations that manipulate their users. How can a user make good choices based on strategic representations? We formalize this as a learning problem, and pursue algorithms for decision-making that are robust to manipulation. In our main setting of interest, the system represents attributes of an item to the user, who then decides whether or not to consume. We model this interaction through the lens of strategic classification (Hardt et al. 2016), reversed: the user, who learns, plays first; and the system, which responds, plays second. The system must respond with representations that reveal ‘nothing but the truth’ but need not reveal the entire truth. Thus, the user faces the problem of learning set functions under strategic subset selection, which presents distinct algorithmic and statistical challenges. Our main result is a learning algorithm that minimizes error despite strategic representations, and our theoretical analysis sheds light on the trade-off between learning effort and susceptibility to manipulation.

----

## [713] Improving Ensemble Distillation With Weight Averaging and Diversifying Perturbation

**Authors**: *Giung Nam, Hyungi Lee, Byeongho Heo, Juho Lee*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nam22a.html](https://proceedings.mlr.press/v162/nam22a.html)

**Abstract**:

Ensembles of deep neural networks have demonstrated superior performance, but their heavy computational cost hinders applying them for resource-limited environments. It motivates distilling knowledge from the ensemble teacher into a smaller student network, and there are two important design choices for this ensemble distillation: 1) how to construct the student network, and 2) what data should be shown during training. In this paper, we propose a weight averaging technique where a student with multiple subnetworks is trained to absorb the functional diversity of ensemble teachers, but then those subnetworks are properly averaged for inference, giving a single student network with no additional inference cost. We also propose a perturbation strategy that seeks inputs from which the diversities of teachers can be better transferred to the student. Combining these two, our method significantly improves upon previous methods on various image classification tasks.

----

## [714] Measuring Representational Robustness of Neural Networks Through Shared Invariances

**Authors**: *Vedant Nanda, Till Speicher, Camila Kolling, John P. Dickerson, Krishna P. Gummadi, Adrian Weller*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nanda22a.html](https://proceedings.mlr.press/v162/nanda22a.html)

**Abstract**:

A major challenge in studying robustness in deep learning is defining the set of “meaningless” perturbations to which a given Neural Network (NN) should be invariant. Most work on robustness implicitly uses a human as the reference model to define such perturbations. Our work offers a new view on robustness by using another reference NN to define the set of perturbations a given NN should be invariant to, thus generalizing the reliance on a reference “human NN” to any NN. This makes measuring robustness equivalent to measuring the extent to which two NNs share invariances. We propose a measure called \stir, which faithfully captures the extent to which two NNs share invariances. \stir re-purposes existing representation similarity measures to make them suitable for measuring shared invariances. Using our measure, we are able to gain insights about how shared invariances vary with changes in weight initialization, architecture, loss functions, and training dataset. Our implementation is available at: \url{https://github.com/nvedant07/STIR}.

----

## [715] Tight and Robust Private Mean Estimation with Few Users

**Authors**: *Shyam Narayanan, Vahab S. Mirrokni, Hossein Esfandiari*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/narayanan22a.html](https://proceedings.mlr.press/v162/narayanan22a.html)

**Abstract**:

In this work, we study high-dimensional mean estimation under user-level differential privacy, and design an $(\varepsilon,\delta)$-differentially private mechanism using as few users as possible. In particular, we provide a nearly optimal trade-off between the number of users and the number of samples per user required for private mean estimation, even when the number of users is as low as $O(\frac{1}{\varepsilon}\log\frac{1}{\delta})$. Interestingly, this bound on the number of users is independent of the dimension (though the number of samples per user is allowed to depend polynomially on the dimension), unlike the previous work that requires the number of users to depend polynomially on the dimension. This resolves a problem first proposed by Amin et al. (2019). Moreover, our mechanism is robust against corruptions in up to $49%$ of the users. Finally, our results also apply to optimal algorithms for privately learning discrete distributions with few users, answering a question of Liu et al. (2020), and a broader range of problems such as stochastic convex optimization and a variant of stochastic gradient descent via a reduction to differentially private mean estimation.

----

## [716] Fast Aquatic Swimmer Optimization with Differentiable Projective Dynamics and Neural Network Hydrodynamic Models

**Authors**: *Elvis Nava, John Z. Zhang, Mike Yan Michelis, Tao Du, Pingchuan Ma, Benjamin F. Grewe, Wojciech Matusik, Robert Kevin Katzschmann*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nava22a.html](https://proceedings.mlr.press/v162/nava22a.html)

**Abstract**:

Aquatic locomotion is a classic fluid-structure interaction (FSI) problem of interest to biologists and engineers. Solving the fully coupled FSI equations for incompressible Navier-Stokes and finite elasticity is computationally expensive. Optimizing robotic swimmer design within such a system generally involves cumbersome, gradient-free procedures on top of the already costly simulation. To address this challenge we present a novel, fully differentiable hybrid approach to FSI that combines a 2D direct numerical simulation for the deformable solid structure of the swimmer and a physics-constrained neural network surrogate to capture hydrodynamic effects of the fluid. For the deformable solid simulation of the swimmer’s body, we use state-of-the-art techniques from the field of computer graphics to speed up the finite-element method (FEM). For the fluid simulation, we use a U-Net architecture trained with a physics-based loss function to predict the flow field at each time step. The pressure and velocity field outputs from the neural network are sampled around the boundary of our swimmer using an immersed boundary method (IBM) to compute its swimming motion accurately and efficiently. We demonstrate the computational efficiency and differentiability of our hybrid simulator on a 2D carangiform swimmer. Due to differentiability, the simulator can be used for computational design of controls for soft bodies immersed in fluids via direct gradient-based optimization.

----

## [717] Multi-Task Learning as a Bargaining Game

**Authors**: *Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, Ethan Fetaya*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/navon22a.html](https://proceedings.mlr.press/v162/navon22a.html)

**Abstract**:

In Multi-task learning (MTL), a joint model is trained to simultaneously make predictions for several tasks. Joint training reduces computation costs and improves data efficiency; however, since the gradients of these different tasks may conflict, training a joint model for MTL often yields lower performance than its corresponding single-task counterparts. A common method for alleviating this issue is to combine per-task gradients into a joint update direction using a particular heuristic. In this paper, we propose viewing the gradients combination step as a bargaining game, where tasks negotiate to reach an agreement on a joint direction of parameter update. Under certain assumptions, the bargaining problem has a unique solution, known as the Nash Bargaining Solution, which we propose to use as a principled approach to multi-task learning. We describe a new MTL optimization procedure, Nash-MTL, and derive theoretical guarantees for its convergence. Empirically, we show that Nash-MTL achieves state-of-the-art results on multiple MTL benchmarks in various domains.

----

## [718] Variational Inference for Infinitely Deep Neural Networks

**Authors**: *Achille Nazaret, David M. Blei*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nazaret22a.html](https://proceedings.mlr.press/v162/nazaret22a.html)

**Abstract**:

We introduce the unbounded depth neural network (UDN), an infinitely deep probabilistic model that adapts its complexity to the training data. The UDN contains an infinite sequence of hidden layers and places an unbounded prior on a truncation L, the layer from which it produces its data. Given a dataset of observations, the posterior UDN provides a conditional distribution of both the parameters of the infinite neural network and its truncation. We develop a novel variational inference algorithm to approximate this posterior, optimizing a distribution of the neural network weights and of the truncation depth L, and without any upper limit on L. To this end, the variational family has a special structure: it models neural network weights of arbitrary depth, and it dynamically creates or removes free variational parameters as its distribution of the truncation is optimized. (Unlike heuristic approaches to model search, it is solely through gradient-based optimization that this algorithm explores the space of truncations.) We study the UDN on real and synthetic data. We find that the UDN adapts its posterior depth to the dataset complexity; it outperforms standard neural networks of similar computational complexity; and it outperforms other approaches to infinite-depth neural networks.

----

## [719] Stable Conformal Prediction Sets

**Authors**: *Eugène Ndiaye*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ndiaye22a.html](https://proceedings.mlr.press/v162/ndiaye22a.html)

**Abstract**:

When one observes a sequence of variables $(x_1, y_1), \ldots, (x_n, y_n)$, Conformal Prediction (CP) is a methodology that allows to estimate a confidence set for $y_{n+1}$ given $x_{n+1}$ by merely assuming that the distribution of the data is exchangeable. CP sets have guaranteed coverage for any finite population size $n$. While appealing, the computation of such a set turns out to be infeasible in general, \eg when the unknown variable $y_{n+1}$ is continuous. The bottleneck is that it is based on a procedure that readjusts a prediction model on data where we replace the unknown target by all its possible values in order to select the most probable one. This requires computing an infinite number of models, which often makes it intractable. In this paper, we combine CP techniques with classical algorithmic stability bounds to derive a prediction set computable with a single model fit. We demonstrate that our proposed confidence set does not lose any coverage guarantees while avoiding the need for data splitting as currently done in the literature. We provide some numerical experiments to illustrate the tightness of our estimation when the sample size is sufficiently large, on both synthetic and real datasets.

----

## [720] Discovering Generalizable Spatial Goal Representations via Graph-based Active Reward Learning

**Authors**: *Aviv Netanyahu, Tianmin Shu, Joshua B. Tenenbaum, Pulkit Agrawal*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/netanyahu22a.html](https://proceedings.mlr.press/v162/netanyahu22a.html)

**Abstract**:

In this work, we consider one-shot imitation learning for object rearrangement tasks, where an AI agent needs to watch a single expert demonstration and learn to perform the same task in different environments. To achieve a strong generalization, the AI agent must infer the spatial goal specification for the task. However, there can be multiple goal specifications that fit the given demonstration. To address this, we propose a reward learning approach, Graph-based Equivalence Mappings (GEM), that can discover spatial goal representations that are aligned with the intended goal specification, enabling successful generalization in unseen environments. Specifically, GEM represents a spatial goal specification by a reward function conditioned on i) a graph indicating important spatial relationships between objects and ii) state equivalence mappings for each edge in the graph indicating invariant properties of the corresponding relationship. GEM combines inverse reinforcement learning and active reward learning to efficiently improve the reward function by utilizing the graph structure and domain randomization enabled by the equivalence mappings. We conducted experiments with simulated oracles and with human subjects. The results show that GEM can drastically improve the generalizability of the learned goal representations over strong baselines.

----

## [721] Sublinear-Time Clustering Oracle for Signed Graphs

**Authors**: *Stefan Neumann, Pan Peng*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/neumann22a.html](https://proceedings.mlr.press/v162/neumann22a.html)

**Abstract**:

Social networks are often modeled using signed graphs, where vertices correspond to users and edges have a sign that indicates whether an interaction between users was positive or negative. The arising signed graphs typically contain a clear community structure in the sense that the graph can be partitioned into a small number of polarized communities, each defining a sparse cut and indivisible into smaller polarized sub-communities. We provide a local clustering oracle for signed graphs with such a clear community structure, that can answer membership queries, i.e., “Given a vertex $v$, which community does $v$ belong to?”, in sublinear time by reading only a small portion of the graph. Formally, when the graph has bounded maximum degree and the number of communities is at most $O(\log n)$, then with $\tilde{O}(\sqrt{n}\operatorname{poly}(1/\varepsilon))$ preprocessing time, our oracle can answer each membership query in $\tilde{O}(\sqrt{n}\operatorname{poly}(1/\varepsilon))$ time, and it correctly classifies a $(1-\varepsilon)$-fraction of vertices w.r.t. a set of hidden planted ground-truth communities. Our oracle is desirable in applications where the clustering information is needed for only a small number of vertices. Previously, such local clustering oracles were only known for unsigned graphs; our generalization to signed graphs requires a number of new ideas and gives a novel spectral analysis of the behavior of random walks with signs. We evaluate our algorithm for constructing such an oracle and answering membership queries on both synthetic and real-world datasets, validating its performance in practice.

----

## [722] Improved Regret for Differentially Private Exploration in Linear MDP

**Authors**: *Dung Daniel T. Ngo, Giuseppe Vietri, Steven Wu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ngo22a.html](https://proceedings.mlr.press/v162/ngo22a.html)

**Abstract**:

We study privacy-preserving exploration in sequential decision-making for environments that rely on sensitive data such as medical records. In particular, we focus on solving the problem of reinforcement learning (RL) subject to the constraint of (joint) differential privacy in the linear MDP setting, where both dynamics and rewards are given by linear functions. Prior work on this problem due to (Luyo et al., 2021) achieves a regret rate that has a dependence of O(K^{3/5}) on the number of episodes K. We provide a private algorithm with an improved regret rate with an optimal dependence of O($\sqrt{}$K) on the number of episodes. The key recipe for our stronger regret guarantee is the adaptivity in the policy update schedule, in which an update only occurs when sufficient changes in the data are detected. As a result, our algorithm benefits from low switching cost and only performs O(log(K)) updates, which greatly reduces the amount of privacy noise. Finally, in the most prevalent privacy regimes where the privacy parameter \epsilon is a constant, our algorithm incurs negligible privacy cost{—}in comparison with the existing non-private regret bounds, the additional regret due to privacy appears in lower-order terms.

----

## [723] A Framework for Learning to Request Rich and Contextually Useful Information from Humans

**Authors**: *Khanh X. Nguyen, Yonatan Bisk, Hal Daumé III*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nguyen22a.html](https://proceedings.mlr.press/v162/nguyen22a.html)

**Abstract**:

When deployed, AI agents will encounter problems that are beyond their autonomous problem-solving capabilities. Leveraging human assistance can help agents overcome their inherent limitations and robustly cope with unfamiliar situations. We present a general interactive framework that enables an agent to request and interpret rich, contextually useful information from an assistant that has knowledge about the task and the environment. We demonstrate the practicality of our framework on a simulated human-assisted navigation problem. Aided with an assistance-requesting policy learned by our method, a navigation agent achieves up to a 7{\texttimes} improvement in success rate on tasks that take place in previously unseen environments, compared to fully autonomous behavior. We show that the agent can take advantage of different types of information depending on the context, and analyze the benefits and challenges of learning the assistance-requesting policy when the assistant can recursively decompose tasks into subtasks.

----

## [724] Transformer Neural Processes: Uncertainty-Aware Meta Learning Via Sequence Modeling

**Authors**: *Tung Nguyen, Aditya Grover*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nguyen22b.html](https://proceedings.mlr.press/v162/nguyen22b.html)

**Abstract**:

Neural Processes (NPs) are a popular class of approaches for meta-learning. Similar to Gaussian Processes (GPs), NPs define distributions over functions and can estimate uncertainty in their predictions. However, unlike GPs, NPs and their variants suffer from underfitting and often have intractable likelihoods, which limit their applications in sequential decision making. We propose Transformer Neural Processes (TNPs), a new member of the NP family that casts uncertainty-aware meta learning as a sequence modeling problem. We learn TNPs via an autoregressive likelihood-based objective and instantiate it with a novel transformer-based architecture that respects the inductive biases inherent to the problem structure, such as invariance to the observed data points and equivariance to the unobserved points. We further design knobs within the TNP architecture to tradeoff the increase in expressivity of the decoding distribution with extra computation. Empirically, we show that TNPs achieve state-of-the-art performance on various benchmark problems, outperforming all previous NP variants on meta regression, image completion, contextual multi-armed bandits, and Bayesian optimization.

----

## [725] Improving Transformers with Probabilistic Attention Keys

**Authors**: *Tam Minh Nguyen, Tan Minh Nguyen, Dung D. D. Le, Duy Khuong Nguyen, Viet-Anh Tran, Richard G. Baraniuk, Nhat Ho, Stanley J. Osher*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nguyen22c.html](https://proceedings.mlr.press/v162/nguyen22c.html)

**Abstract**:

Multi-head attention is a driving force behind state-of-the-art transformers, which achieve remarkable performance across a variety of natural language processing (NLP) and computer vision tasks. It has been observed that for many applications, those attention heads learn redundant embedding, and most of them can be removed without degrading the performance of the model. Inspired by this observation, we propose Transformer with a Mixture of Gaussian Keys (Transformer-MGK), a novel transformer architecture that replaces redundant heads in transformers with a mixture of keys at each head. These mixtures of keys follow a Gaussian mixture model and allow each attention head to focus on different parts of the input sequence efficiently. Compared to its conventional transformer counterpart, Transformer-MGK accelerates training and inference, has fewer parameters, and requires fewer FLOPs to compute while achieving comparable or better accuracy across tasks. Transformer-MGK can also be easily extended to use with linear attention. We empirically demonstrate the advantage of Transformer-MGK in a range of practical applications, including language modeling and tasks that involve very long sequences. On the Wikitext-103 and Long Range Arena benchmark, Transformer-MGKs with 4 heads attain comparable or better performance to the baseline transformers with 8 heads.

----

## [726] On Transportation of Mini-batches: A Hierarchical Approach

**Authors**: *Khai Nguyen, Dang Nguyen, Quoc Dinh Nguyen, Tung Pham, Hung Bui, Dinh Phung, Trung Le, Nhat Ho*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nguyen22d.html](https://proceedings.mlr.press/v162/nguyen22d.html)

**Abstract**:

Mini-batch optimal transport (m-OT) has been successfully used in practical applications that involve probability measures with a very high number of supports. The m-OT solves several smaller optimal transport problems and then returns the average of their costs and transportation plans. Despite its scalability advantage, the m-OT does not consider the relationship between mini-batches which leads to undesirable estimation. Moreover, the m-OT does not approximate a proper metric between probability measures since the identity property is not satisfied. To address these problems, we propose a novel mini-batch scheme for optimal transport, named Batch of Mini-batches Optimal Transport (BoMb-OT), that finds the optimal coupling between mini-batches and it can be seen as an approximation to a well-defined distance on the space of probability measures. Furthermore, we show that the m-OT is a limit of the entropic regularized version of the BoMb-OT when the regularized parameter goes to infinity. Finally, we carry out experiments on various applications including deep generative models, deep domain adaptation, approximate Bayesian computation, color transfer, and gradient flow to show that the BoMb-OT can be widely applied and performs well in various applications.

----

## [727] Improving Mini-batch Optimal Transport via Partial Transportation

**Authors**: *Khai Nguyen, Dang Nguyen, The-Anh Vu-Le, Tung Pham, Nhat Ho*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nguyen22e.html](https://proceedings.mlr.press/v162/nguyen22e.html)

**Abstract**:

Mini-batch optimal transport (m-OT) has been widely used recently to deal with the memory issue of OT in large-scale applications. Despite their practicality, m-OT suffers from misspecified mappings, namely, mappings that are optimal on the mini-batch level but are partially wrong in the comparison with the optimal transportation plan between the original measures. Motivated by the misspecified mappings issue, we propose a novel mini-batch method by using partial optimal transport (POT) between mini-batch empirical measures, which we refer to as mini-batch partial optimal transport (m-POT). Leveraging the insight from the partial transportation, we explain the source of misspecified mappings from the m-OT and motivate why limiting the amount of transported masses among mini-batches via POT can alleviate the incorrect mappings. Finally, we carry out extensive experiments on various applications such as deep domain adaptation, partial domain adaptation, deep generative model, color transfer, and gradient flow to demonstrate the favorable performance of m-POT compared to current mini-batch methods.

----

## [728] Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs

**Authors**: *Tianwei Ni, Benjamin Eysenbach, Ruslan Salakhutdinov*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ni22a.html](https://proceedings.mlr.press/v162/ni22a.html)

**Abstract**:

Many problems in RL, such as meta-RL, robust RL, generalization in RL, and temporal credit assignment, can be cast as POMDPs. In theory, simply augmenting model-free RL with memory-based architectures, such as recurrent neural networks, provides a general approach to solving all types of POMDPs. However, prior work has found that such recurrent model-free RL methods tend to perform worse than more specialized algorithms that are designed for specific types of POMDPs. This paper revisits this claim. We find that careful architecture and hyperparameter decisions can often yield a recurrent model-free implementation that performs on par with (and occasionally substantially better than) more sophisticated recent techniques. We compare to 21 environments from 6 prior specialized methods and find that our implementation achieves greater sample efficiency and asymptotic performance than these methods on 18/21 environments. We also release a simple and efficient implementation of recurrent model-free RL for future work to use as a baseline for POMDPs.

----

## [729] Optimal Estimation of Policy Gradient via Double Fitted Iteration

**Authors**: *Chengzhuo Ni, Ruiqi Zhang, Xiang Ji, Xuezhou Zhang, Mengdi Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ni22b.html](https://proceedings.mlr.press/v162/ni22b.html)

**Abstract**:

Policy gradient (PG) estimation becomes a challenge when we are not allowed to sample with the target policy but only have access to a dataset generated by some unknown behavior policy. Conventional methods for off-policy PG estimation often suffer from either significant bias or exponentially large variance. In this paper, we propose the double Fitted PG estimation (FPG) algorithm. FPG can work with an arbitrary policy parameterization, assuming access to a Bellman-complete value function class. In the case of linear value function approximation, we provide a tight finite-sample upper bound on policy gradient estimation error, that is governed by the amount of distribution mismatch measured in feature space. We also establish the asymptotic normality of FPG estimation error with a precise covariance characterization, which is further shown to be statistically optimal with a matching Cramer-Rao lower bound. Empirically, we evaluate the performance of FPG on both policy gradient estimation and policy optimization, using either softmax tabular or ReLU policy networks. Under various metrics, our results show that FPG significantly outperforms existing off-policy PG estimation methods based on importance sampling and variance reduction techniques.

----

## [730] GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models

**Authors**: *Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, Mark Chen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nichol22a.html](https://proceedings.mlr.press/v162/nichol22a.html)

**Abstract**:

Diffusion models have recently been shown to generate high-quality synthetic images, especially when paired with a guidance technique to trade off diversity for fidelity. We explore diffusion models for the problem of text-conditional image synthesis and compare two different guidance strategies: CLIP guidance and classifier-free guidance. We find that the latter is preferred by human evaluators for both photorealism and caption similarity, and often produces photorealistic samples. Samples from a 3.5 billion parameter text-conditional diffusion model using classifier-free guidance are favored by human evaluators to those from DALL-E, even when the latter uses expensive CLIP reranking. Additionally, we find that our models can be fine-tuned to perform image inpainting, enabling powerful text-driven image editing. We train a smaller model on a filtered dataset and release the code and weights at https://github.com/openai/glide-text2im.

----

## [731] Diffusion Models for Adversarial Purification

**Authors**: *Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, Animashree Anandkumar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nie22a.html](https://proceedings.mlr.press/v162/nie22a.html)

**Abstract**:

Adversarial purification refers to a class of defense methods that remove adversarial perturbations using a generative model. These methods do not make assumptions on the form of attack and the classification model, and thus can defend pre-existing classifiers against unseen threats. However, their performance currently falls behind adversarial training methods. In this work, we propose DiffPure that uses diffusion models for adversarial purification: Given an adversarial example, we first diffuse it with a small amount of noise following a forward diffusion process, and then recover the clean image through a reverse generative process. To evaluate our method against strong adaptive attacks in an efficient and scalable way, we propose to use the adjoint method to compute full gradients of the reverse generative process. Extensive experiments on three image datasets including CIFAR-10, ImageNet and CelebA-HQ with three classifier architectures including ResNet, WideResNet and ViT demonstrate that our method achieves the state-of-the-art results, outperforming current adversarial training and adversarial purification methods, often by a large margin.

----

## [732] The Primacy Bias in Deep Reinforcement Learning

**Authors**: *Evgenii Nikishin, Max Schwarzer, Pierluca D'Oro, Pierre-Luc Bacon, Aaron C. Courville*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nikishin22a.html](https://proceedings.mlr.press/v162/nikishin22a.html)

**Abstract**:

This work identifies a common flaw of deep reinforcement learning (RL) algorithms: a tendency to rely on early interactions and ignore useful evidence encountered later. Because of training on progressively growing datasets, deep RL agents incur a risk of overfitting to earlier experiences, negatively affecting the rest of the learning process. Inspired by cognitive science, we refer to this effect as the primacy bias. Through a series of experiments, we dissect the algorithmic aspects of deep RL that exacerbate this bias. We then propose a simple yet generally-applicable mechanism that tackles the primacy bias by periodically resetting a part of the agent. We apply this mechanism to algorithms in both discrete (Atari 100k) and continuous action (DeepMind Control Suite) domains, consistently improving their performance.

----

## [733] Causal Conceptions of Fairness and their Consequences

**Authors**: *Hamed Nilforoshan, Johann D. Gaebler, Ravi Shroff, Sharad Goel*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nilforoshan22a.html](https://proceedings.mlr.press/v162/nilforoshan22a.html)

**Abstract**:

Recent work highlights the role of causality in designing equitable decision-making algorithms. It is not immediately clear, however, how existing causal conceptions of fairness relate to one another, or what the consequences are of using these definitions as design principles. Here, we first assemble and categorize popular causal definitions of algorithmic fairness into two broad families: (1) those that constrain the effects of decisions on counterfactual disparities; and (2) those that constrain the effects of legally protected characteristics, like race and gender, on decisions. We then show, analytically and empirically, that both families of definitions almost always—in a measure theoretic sense—result in strongly Pareto dominated decision policies, meaning there is an alternative, unconstrained policy favored by every stakeholder with preferences drawn from a large, natural class. For example, in the case of college admissions decisions, policies constrained to satisfy causal fairness definitions would be disfavored by every stakeholder with neutral or positive preferences for both academic preparedness and diversity. Indeed, under a prominent definition of causal fairness, we prove the resulting policies require admitting all students with the same probability, regardless of academic qualifications or group membership. Our results highlight formal limitations and potential adverse consequences of common mathematical notions of causal fairness.

----

## [734] Efficient Test-Time Model Adaptation without Forgetting

**Authors**: *Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, Mingkui Tan*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/niu22a.html](https://proceedings.mlr.press/v162/niu22a.html)

**Abstract**:

Test-time adaptation provides an effective means of tackling the potential distribution shift between model training and inference, by dynamically updating the model at test time. This area has seen fast progress recently, at the effectiveness of handling test shifts. Nonetheless, prior methods still suffer two key limitations: 1) these methods rely on performing backward computation for each test sample, which takes a considerable amount of time; and 2) these methods focus on improving the performance on out-of-distribution test samples and ignore that the adaptation on test data may result in a catastrophic forgetting issue, \ie, the performance on in-distribution test samples may degrade. To address these issues, we propose an efficient anti-forgetting test-time adaptation (EATA) method. Specifically, we devise a sample-efficient entropy minimization loss to exclude uninformative samples out of backward computation, which improves the overall efficiency and meanwhile boosts the out-of-distribution accuracy. Afterward, we introduce a regularization loss to ensure that critical model weights tend to be preserved during adaptation, thereby alleviating the forgetting issue. Extensive experiments on CIFAR-10-C, ImageNet-C, and ImageNet-R verify the effectiveness and superiority of our EATA.

----

## [735] Generative Trees: Adversarial and Copycat

**Authors**: *Richard Nock, Mathieu Guillame-Bert*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nock22a.html](https://proceedings.mlr.press/v162/nock22a.html)

**Abstract**:

While Generative Adversarial Networks (GANs) achieve spectacular results on unstructured data like images, there is still a gap on tabular data, data for which state of the art supervised learning still favours decision tree (DT)-based models. This paper proposes a new path forward for the generation of tabular data, exploiting decades-old understanding of the supervised task’s best components for DT induction, from losses (properness), models (tree-based) to algorithms (boosting). The properness condition on the supervised loss – which postulates the optimality of Bayes rule – leads us to a variational GAN-style loss formulation which is tight when discriminators meet a calibration property trivially satisfied by DTs, and, under common assumptions about the supervised loss, yields "one loss to train against them all" for the generator: the $\chi^2$. We then introduce tree-based generative models, generative trees (GTs), meant to mirror on the generative side the good properties of DTs for classifying tabular data, with a boosting-compliant adversarial training algorithm for GTs. We also introduce copycat training, in which the generator copies at run time the underlying tree (graph) of the discriminator DT and completes it for the hardest discriminative task, with boosting compliant convergence. We test our algorithms on tasks including fake/real distinction and missing data imputation.

----

## [736] Path-Aware and Structure-Preserving Generation of Synthetically Accessible Molecules

**Authors**: *Juhwan Noh, Dae-Woong Jeong, Kiyoung Kim, Sehui Han, Moontae Lee, Honglak Lee, Yousung Jung*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/noh22a.html](https://proceedings.mlr.press/v162/noh22a.html)

**Abstract**:

Computational chemistry aims to autonomously design specific molecules with target functionality. Generative frameworks provide useful tools to learn continuous representations of molecules in a latent space. While modelers could optimize chemical properties, many generated molecules are not synthesizable. To design synthetically accessible molecules that preserve main structural motifs of target molecules, we propose a reaction-embedded and structure-conditioned variational autoencoder. As the latent space jointly encodes molecular structures and their reaction routes, our new sampling method that measures the path-informed structural similarity allows us to effectively generate structurally analogous synthesizable molecules. When targeting out-of-domain as well as in-domain seed structures, our model generates structurally and property-wisely similar molecules equipped with well-defined reaction paths. By focusing on the important region in chemical space, we also demonstrate that our model can design new molecules with even higher activity than the seed molecules.

----

## [737] Utilizing Expert Features for Contrastive Learning of Time-Series Representations

**Authors**: *Manuel T. Nonnenmacher, Lukas Oldenburg, Ingo Steinwart, David Reeb*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/nonnenmacher22a.html](https://proceedings.mlr.press/v162/nonnenmacher22a.html)

**Abstract**:

We present an approach that incorporates expert knowledge for time-series representation learning. Our method employs expert features to replace the commonly used data transformations in previous contrastive learning approaches. We do this since time-series data frequently stems from the industrial or medical field where expert features are often available from domain experts, while transformations are generally elusive for time-series data. We start by proposing two properties that useful time-series representations should fulfill and show that current representation learning approaches do not ensure these properties. We therefore devise ExpCLR, a novel contrastive learning approach built on an objective that utilizes expert features to encourage both properties for the learned representation. Finally, we demonstrate on three real-world time-series datasets that ExpCLR surpasses several state-of-the-art methods for both unsupervised and semi-supervised representation learning.

----

## [738] Tranception: Protein Fitness Prediction with Autoregressive Transformers and Inference-time Retrieval

**Authors**: *Pascal Notin, Mafalda Dias, Jonathan Frazer, Javier Marchena-Hurtado, Aidan N. Gomez, Debora S. Marks, Yarin Gal*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/notin22a.html](https://proceedings.mlr.press/v162/notin22a.html)

**Abstract**:

The ability to accurately model the fitness landscape of protein sequences is critical to a wide range of applications, from quantifying the effects of human variants on disease likelihood, to predicting immune-escape mutations in viruses and designing novel biotherapeutic proteins. Deep generative models of protein sequences trained on multiple sequence alignments have been the most successful approaches so far to address these tasks. The performance of these methods is however contingent on the availability of sufficiently deep and diverse alignments for reliable training. Their potential scope is thus limited by the fact many protein families are hard, if not impossible, to align. Large language models trained on massive quantities of non-aligned protein sequences from diverse families address these problems and show potential to eventually bridge the performance gap. We introduce Tranception, a novel transformer architecture leveraging autoregressive predictions and retrieval of homologous sequences at inference to achieve state-of-the-art fitness prediction performance. Given its markedly higher performance on multiple mutants, robustness to shallow alignments and ability to score indels, our approach offers significant gain of scope over existing approaches. To enable more rigorous model testing across a broader range of protein families, we develop ProteinGym – an extensive set of multiplexed assays of variant effects, substantially increasing both the number and diversity of assays compared to existing benchmarks.

----

## [739] Fast Finite Width Neural Tangent Kernel

**Authors**: *Roman Novak, Jascha Sohl-Dickstein, Samuel S. Schoenholz*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/novak22a.html](https://proceedings.mlr.press/v162/novak22a.html)

**Abstract**:

The Neural Tangent Kernel (NTK), defined as the outer product of the neural network (NN) Jacobians, has emerged as a central object of study in deep learning. In the infinite width limit, the NTK can sometimes be computed analytically and is useful for understanding training and generalization of NN architectures. At finite widths, the NTK is also used to better initialize NNs, compare the conditioning across models, perform architecture search, and do meta-learning. Unfortunately, the finite width NTK is notoriously expensive to compute, which severely limits its practical utility. We perform the first in-depth analysis of the compute and memory requirements for NTK computation in finite width networks. Leveraging the structure of neural networks, we further propose two novel algorithms that change the exponent of the compute and memory requirements of the finite width NTK, dramatically improving efficiency. Our algorithms can be applied in a black box fashion to any differentiable function, including those implementing neural networks. We open-source our implementations within the Neural Tangents package at https://github.com/google/neural-tangents.

----

## [740] Multicoated Supermasks Enhance Hidden Networks

**Authors**: *Yasuyuki Okoshi, Ángel López García-Arias, Kazutoshi Hirose, Kota Ando, Kazushi Kawamura, Thiem Van Chu, Masato Motomura, Jaehoon Yu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/okoshi22a.html](https://proceedings.mlr.press/v162/okoshi22a.html)

**Abstract**:

Hidden Networks (Ramanujan et al., 2020) showed the possibility of finding accurate subnetworks within a randomly weighted neural network by training a connectivity mask, referred to as supermask. We show that the supermask stops improving even though gradients are not zero, thus underutilizing backpropagated information. To address this we propose a method that extends Hidden Networks by training an overlay of multiple hierarchical supermasks{—}a multicoated supermask. This method shows that using multiple supermasks for a single task achieves higher accuracy without additional training cost. Experiments on CIFAR-10 and ImageNet show that Multicoated Supermasks enhance the tradeoff between accuracy and model size. A ResNet-101 using a 7-coated supermask outperforms its Hidden Networks counterpart by 4%, matching the accuracy of a dense ResNet-50 while being an order of magnitude smaller.

----

## [741] Generalized Leverage Scores: Geometric Interpretation and Applications

**Authors**: *Bruno Ordozgoiti, Antonis Matakos, Aristides Gionis*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ordozgoiti22a.html](https://proceedings.mlr.press/v162/ordozgoiti22a.html)

**Abstract**:

In problems involving matrix computations, the concept of leverage has found a large number of applications. In particular, leverage scores, which relate the columns of a matrix to the subspaces spanned by its leading singular vectors, are helpful in revealing column subsets to approximately factorize a matrix with quality guarantees. As such, they provide a solid foundation for a variety of machine-learning methods. In this paper we extend the definition of leverage scores to relate the columns of a matrix to arbitrary subsets of singular vectors. We establish a precise connection between column and singular-vector subsets, by relating the concepts of leverage scores and principal angles between subspaces. We employ this result to design approximation algorithms with provable guarantees for two well-known problems: generalized column subset selection and sparse canonical correlation analysis. We run numerical experiments to provide further insight on the proposed methods. The novel bounds we derive improve our understanding of fundamental concepts in matrix approximations. In addition, our insights may serve as building blocks for further contributions.

----

## [742] Practical Almost-Linear-Time Approximation Algorithms for Hybrid and Overlapping Graph Clustering

**Authors**: *Lorenzo Orecchia, Konstantinos Ameranis, Charalampos E. Tsourakakis, Kunal Talwar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/orecchia22a.html](https://proceedings.mlr.press/v162/orecchia22a.html)

**Abstract**:

Detecting communities in real-world networks and clustering similarity graphs are major data mining tasks with a wide range of applications in graph mining, collaborative filtering, and bioinformatics. In many such applications, overwhelming empirical evidence suggests that communities and clusters are naturally overlapping, i.e., the boundary of a cluster may contain both edges across clusters and nodes that are shared with other clusters, calling for novel hybrid graph partitioning algorithms (HGP). While almost-linear-time approximation algorithms are known for edge-boundary-based graph partitioning, little progress has been made on fast algorithms for HGP, even in the special case of vertex-boundary-based graph partitioning. In this work, we introduce a frame-work based on two novel clustering objectives, which naturally extend the well-studied notion of conductance to clusters with hybrid vertex-and edge-boundary structure. Our main algorithmic contributions are almost-linear-time algorithms O(log n)-approximation algorithms for both these objectives. To this end, we show that the cut-matching framework of (Khandekar et al., 2014) can be significantly extended to incorporate hybrid partitions. Crucially, we implement our approximation algorithm to produce both hybrid partitions and optimality certificates for large graphs, easily scaling to tens of millions of edges, and test our implementation on real-world datasets against other competitive baselines.

----

## [743] Anticorrelated Noise Injection for Improved Generalization

**Authors**: *Antonio Orvieto, Hans Kersting, Frank Proske, Francis R. Bach, Aurélien Lucchi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/orvieto22a.html](https://proceedings.mlr.press/v162/orvieto22a.html)

**Abstract**:

Injecting artificial noise into gradient descent (GD) is commonly employed to improve the performance of machine learning models. Usually, uncorrelated noise is used in such perturbed gradient descent (PGD) methods. It is, however, not known if this is optimal or whether other types of noise could provide better generalization performance. In this paper, we zoom in on the problem of correlating the perturbations of consecutive PGD steps. We consider a variety of objective functions for which we find that GD with anticorrelated perturbations ("Anti-PGD") generalizes significantly better than GD and standard (uncorrelated) PGD. To support these experimental findings, we also derive a theoretical analysis that demonstrates that Anti-PGD moves to wider minima, while GD and PGD remain stuck in suboptimal regions or even diverge. This new connection between anticorrelated noise and generalization opens the field to novel ways to exploit noise for training machine learning models.

----

## [744] Scalable Deep Gaussian Markov Random Fields for General Graphs

**Authors**: *Joel Oskarsson, Per Sidén, Fredrik Lindsten*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/oskarsson22a.html](https://proceedings.mlr.press/v162/oskarsson22a.html)

**Abstract**:

Machine learning methods on graphs have proven useful in many applications due to their ability to handle generally structured data. The framework of Gaussian Markov Random Fields (GMRFs) provides a principled way to define Gaussian models on graphs by utilizing their sparsity structure. We propose a flexible GMRF model for general graphs built on the multi-layer structure of Deep GMRFs, originally proposed for lattice graphs only. By designing a new type of layer we enable the model to scale to large graphs. The layer is constructed to allow for efficient training using variational inference and existing software frameworks for Graph Neural Networks. For a Gaussian likelihood, close to exact Bayesian inference is available for the latent field. This allows for making predictions with accompanying uncertainty estimates. The usefulness of the proposed model is verified by experiments on a number of synthetic and real world datasets, where it compares favorably to other both Bayesian and deep learning methods.

----

## [745] Zero-shot AutoML with Pretrained Models

**Authors**: *Ekrem Öztürk, Fabio Ferreira, Hadi S. Jomaa, Lars Schmidt-Thieme, Josif Grabocka, Frank Hutter*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ozturk22a.html](https://proceedings.mlr.press/v162/ozturk22a.html)

**Abstract**:

Given a new dataset D and a low compute budget, how should we choose a pre-trained model to fine-tune to D, and set the fine-tuning hyperparameters without risking overfitting, particularly if D is small? Here, we extend automated machine learning (AutoML) to best make these choices. Our domain-independent meta-learning approach learns a zero-shot surrogate model which, at test time, allows to select the right deep learning (DL) pipeline (including the pre-trained model and fine-tuning hyperparameters) for a new dataset D given only trivial meta-features describing D such as image resolution or the number of classes. To train this zero-shot model, we collect performance data for many DL pipelines on a large collection of datasets and meta-train on this data to minimize a pairwise ranking objective. We evaluate our approach under the strict time limit of the vision track of the ChaLearn AutoDL challenge benchmark, clearly outperforming all challenge contenders.

----

## [746] History Compression via Language Models in Reinforcement Learning

**Authors**: *Fabian Paischer, Thomas Adler, Vihang Patil, Angela Bitto-Nemling, Markus Holzleitner, Sebastian Lehner, Hamid Eghbal-Zadeh, Sepp Hochreiter*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/paischer22a.html](https://proceedings.mlr.press/v162/paischer22a.html)

**Abstract**:

In a partially observable Markov decision process (POMDP), an agent typically uses a representation of the past to approximate the underlying MDP. We propose to utilize a frozen Pretrained Language Transformer (PLT) for history representation and compression to improve sample efficiency. To avoid training of the Transformer, we introduce FrozenHopfield, which automatically associates observations with pretrained token embeddings. To form these associations, a modern Hopfield network stores these token embeddings, which are retrieved by queries that are obtained by a random but fixed projection of observations. Our new method, HELM, enables actor-critic network architectures that contain a pretrained language Transformer for history representation as a memory module. Since a representation of the past need not be learned, HELM is much more sample efficient than competitors. On Minigrid and Procgen environments HELM achieves new state-of-the-art results. Our code is available at https://github.com/ml-jku/helm.

----

## [747] A Study on the Ramanujan Graph Property of Winning Lottery Tickets

**Authors**: *Bithika Pal, Arindam Biswas, Sudeshna Kolay, Pabitra Mitra, Biswajit Basu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pal22a.html](https://proceedings.mlr.press/v162/pal22a.html)

**Abstract**:

Winning lottery tickets refer to sparse subgraphs of deep neural networks which have classification accuracy close to the original dense networks. Resilient connectivity properties of such sparse networks play an important role in their performance. The attempt is to identify a sparse and yet well-connected network to guarantee unhindered information flow. Connectivity in a graph is best characterized by its spectral expansion property. Ramanujan graphs are robust expanders which lead to sparse but highly-connected networks, and thus aid in studying the winning tickets. A feedforward neural network consists of a sequence of bipartite graphs representing its layers. We analyze the Ramanujan graph property of such bipartite layers in terms of their spectral characteristics using the Cheeger’s inequality for irregular graphs. It is empirically observed that the winning ticket networks preserve the Ramanujan graph property and achieve a high accuracy even when the layers are sparse. Accuracy and robustness to noise start declining as many of the layers lose the property. Next we find a robust winning lottery ticket by pruning individual layers while retaining their respective Ramanujan graph property. This strategy is observed to improve the performance of existing network pruning algorithms.

----

## [748] On Learning Mixture of Linear Regressions in the Non-Realizable Setting

**Authors**: *Soumyabrata Pal, Arya Mazumdar, Rajat Sen, Avishek Ghosh*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pal22b.html](https://proceedings.mlr.press/v162/pal22b.html)

**Abstract**:

While mixture of linear regressions (MLR) is a well-studied topic, prior works usually do not analyze such models for prediction error. In fact, prediction and loss are not well-defined in the context of mixtures. In this paper, first we show that MLR can be used for prediction where instead of predicting a label, the model predicts a list of values (also known as list-decoding). The list size is equal to the number of components in the mixture, and the loss function is defined to be minimum among the losses resulted by all the component models. We show that with this definition, a solution of the empirical risk minimization (ERM) achieves small probability of prediction error. This begs for an algorithm to minimize the empirical risk for MLR, which is known to be computationally hard. Prior algorithmic works in MLR focus on the realizable setting, i.e., recovery of parameters when data is probabilistically generated by a mixed linear (noisy) model. In this paper we show that a version of the popular expectation minimization (EM) algorithm finds out the best fit lines in a dataset even when a realizable model is not assumed, under some regularity conditions on the dataset and the initial points, and thereby provides a solution for the ERM. We further provide an algorithm that runs in polynomial time in the number of datapoints, and recovers a good approximation of the best fit lines. The two algorithms are experimentally compared.

----

## [749] Plan Better Amid Conservatism: Offline Multi-Agent Reinforcement Learning with Actor Rectification

**Authors**: *Ling Pan, Longbo Huang, Tengyu Ma, Huazhe Xu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pan22a.html](https://proceedings.mlr.press/v162/pan22a.html)

**Abstract**:

Conservatism has led to significant progress in offline reinforcement learning (RL) where an agent learns from pre-collected datasets. However, as many real-world scenarios involve interaction among multiple agents, it is important to resolve offline RL in the multi-agent setting. Given the recent success of transferring online RL algorithms to the multi-agent setting, one may expect that offline RL algorithms will also transfer to multi-agent settings directly. Surprisingly, we empirically observe that conservative offline RL algorithms do not work well in the multi-agent setting—the performance degrades significantly with an increasing number of agents. Towards mitigating the degradation, we identify a key issue that non-concavity of the value function makes the policy gradient improvements prone to local optima. Multiple agents exacerbate the problem severely, since the suboptimal policy by any agent can lead to uncoordinated global failure. Following this intuition, we propose a simple yet effective method, Offline Multi-Agent RL with Actor Rectification (OMAR), which combines the first-order policy gradients and zeroth-order optimization methods to better optimize the conservative value functions over the actor parameters. Despite the simplicity, OMAR achieves state-of-the-art results in a variety of multi-agent control tasks.

----

## [750] A Unified Weight Initialization Paradigm for Tensorial Convolutional Neural Networks

**Authors**: *Yu Pan, Zeyong Su, Ao Liu, Jingquan Wang, Nannan Li, Zenglin Xu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pan22b.html](https://proceedings.mlr.press/v162/pan22b.html)

**Abstract**:

Tensorial Convolutional Neural Networks (TCNNs) have attracted much research attention for their power in reducing model parameters or enhancing the generalization ability. However, exploration of TCNNs is hindered even from weight initialization methods. To be specific, general initialization methods, such as Xavier or Kaiming initialization, usually fail to generate appropriate weights for TCNNs. Meanwhile, although there are ad-hoc approaches for specific architectures (e.g., Tensor Ring Nets), they are not applicable to TCNNs with other tensor decomposition methods (e.g., CP or Tucker decomposition). To address this problem, we propose a universal weight initialization paradigm, which generalizes Xavier and Kaiming methods and can be widely applicable to arbitrary TCNNs. Specifically, we first present the Reproducing Transformation to convert the backward process in TCNNs to an equivalent convolution process. Then, based on the convolution operators in the forward and backward processes, we build a unified paradigm to control the variance of features and gradients in TCNNs. Thus, we can derive fan-in and fan-out initialization for various TCNNs. We demonstrate that our paradigm can stabilize the training of TCNNs, leading to faster convergence and better results.

----

## [751] Robustness and Accuracy Could Be Reconcilable by (Proper) Definition

**Authors**: *Tianyu Pang, Min Lin, Xiao Yang, Jun Zhu, Shuicheng Yan*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pang22a.html](https://proceedings.mlr.press/v162/pang22a.html)

**Abstract**:

The trade-off between robustness and accuracy has been widely studied in the adversarial literature. Although still controversial, the prevailing view is that this trade-off is inherent, either empirically or theoretically. Thus, we dig for the origin of this trade-off in adversarial training and find that it may stem from the improperly defined robust error, which imposes an inductive bias of local invariance — an overcorrection towards smoothness. Given this, we advocate employing local equivariance to describe the ideal behavior of a robust model, leading to a self-consistent robust error named SCORE. By definition, SCORE facilitates the reconciliation between robustness and accuracy, while still handling the worst-case uncertainty via robust optimization. By simply substituting KL divergence with variants of distance metrics, SCORE can be efficiently minimized. Empirically, our models achieve top-rank performance on RobustBench under AutoAttack. Besides, SCORE provides instructive insights for explaining the overfitting phenomenon and semantic input gradients observed on robust models.

----

## [752] Towards Coherent and Consistent Use of Entities in Narrative Generation

**Authors**: *Pinelopi Papalampidi, Kris Cao, Tomás Kociský*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/papalampidi22a.html](https://proceedings.mlr.press/v162/papalampidi22a.html)

**Abstract**:

Large pre-trained language models (LMs) have demonstrated impressive capabilities in generating long, fluent text; however, there is little to no analysis on their ability to maintain entity coherence and consistency. In this work, we focus on the end task of narrative generation and systematically analyse the long-range entity coherence and consistency in generated stories. First, we propose a set of automatic metrics for measuring model performance in terms of entity usage. Given these metrics, we quantify the limitations of current LMs. Next, we propose augmenting a pre-trained LM with a dynamic entity memory in an end-to-end manner by using an auxiliary entity-related loss for guiding the reads and writes to the memory. We demonstrate that the dynamic entity memory increases entity coherence according to both automatic and human judgment and helps preserving entity-related information especially in settings with a limited context window. Finally, we also validate that our automatic metrics are correlated with human ratings and serve as a good indicator of the quality of generated stories.

----

## [753] Constrained Discrete Black-Box Optimization using Mixed-Integer Programming

**Authors**: *Theodore P. Papalexopoulos, Christian Tjandraatmadja, Ross Anderson, Juan Pablo Vielma, David Belanger*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/papalexopoulos22a.html](https://proceedings.mlr.press/v162/papalexopoulos22a.html)

**Abstract**:

Discrete black-box optimization problems are challenging for model-based optimization (MBO) algorithms, such as Bayesian optimization, due to the size of the search space and the need to satisfy combinatorial constraints. In particular, these methods require repeatedly solving a complex discrete global optimization problem in the inner loop, where popular heuristic inner-loop solvers introduce approximations and are difficult to adapt to combinatorial constraints. In response, we propose NN+MILP, a general discrete MBO framework using piecewise-linear neural networks as surrogate models and mixed-integer linear programming (MILP) to optimize the acquisition function. MILP provides optimality guarantees and a versatile declarative language for domain-specific constraints. We test our approach on a range of unconstrained and constrained problems, including DNA binding, constrained binary quadratic problems from the MINLPLib benchmark, and the NAS-Bench-101 neural architecture search benchmark. NN+MILP surpasses or matches the performance of black-box algorithms tailored to the constraints at hand, with global optimization of the acquisition problem running in a few minutes using only standard software packages and hardware.

----

## [754] A Theoretical Comparison of Graph Neural Network Extensions

**Authors**: *Pál András Papp, Roger Wattenhofer*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/papp22a.html](https://proceedings.mlr.press/v162/papp22a.html)

**Abstract**:

We study and compare different Graph Neural Network extensions that increase the expressive power of GNNs beyond the Weisfeiler-Leman test. We focus on (i) GNNs based on higher order WL methods, (ii) GNNs that preprocess small substructures in the graph, (iii) GNNs that preprocess the graph up to a small radius, and (iv) GNNs that slightly perturb the graph to compute an embedding. We begin by presenting a simple improvement for this last extension that strictly increases the expressive power of this GNN variant. Then, as our main result, we compare the expressiveness of these extensions to each other through a series of example constructions that can be distinguished by one of the extensions, but not by another one. We also show negative examples that are particularly challenging for each of the extensions, and we prove several claims about the ability of these extensions to count cliques and cycles in the graph.

----

## [755] Validating Causal Inference Methods

**Authors**: *Harsh Parikh, Carlos Varjao, Louise Xu, Eric Tchetgen Tchetgen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/parikh22a.html](https://proceedings.mlr.press/v162/parikh22a.html)

**Abstract**:

The fundamental challenge of drawing causal inference is that counterfactual outcomes are not fully observed for any unit. Furthermore, in observational studies, treatment assignment is likely to be confounded. Many statistical methods have emerged for causal inference under unconfoundedness conditions given pre-treatment covariates, including propensity score-based methods, prognostic score-based methods, and doubly robust methods. Unfortunately for applied researchers, there is no ‘one-size-fits-all’ causal method that can perform optimally universally. In practice, causal methods are primarily evaluated quantitatively on handcrafted simulated data. Such data-generative procedures can be of limited value because they are typically stylized models of reality. They are simplified for tractability and lack the complexities of real-world data. For applied researchers, it is critical to understand how well a method performs for the data at hand. Our work introduces a deep generative model-based framework, Credence, to validate causal inference methods. The framework’s novelty stems from its ability to generate synthetic data anchored at the empirical distribution for the observed sample, and therefore virtually indistinguishable from the latter. The approach allows the user to specify ground truth for the form and magnitude of causal effects and confounding bias as functions of covariates. Thus simulated data sets are used to evaluate the potential performance of various causal estimation methods when applied to data similar to the observed sample. We demonstrate Credence’s ability to accurately assess the relative performance of causal estimation techniques in an extensive simulation study and two real-world data applications from Lalonde and Project STAR studies.

----

## [756] The Unsurprising Effectiveness of Pre-Trained Vision Models for Control

**Authors**: *Simone Parisi, Aravind Rajeswaran, Senthil Purushwalkam, Abhinav Gupta*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/parisi22a.html](https://proceedings.mlr.press/v162/parisi22a.html)

**Abstract**:

Recent years have seen the emergence of pre-trained representations as a powerful abstraction for AI applications in computer vision, natural language, and speech. However, policy learning for control is still dominated by a tabula-rasa learning paradigm, with visuo-motor policies often trained from scratch using data from deployment environments. In this context, we revisit and study the role of pre-trained visual representations for control, and in particular representations trained on large-scale computer vision datasets. Through extensive empirical evaluation in diverse control domains (Habitat, DeepMind Control, Adroit, Franka Kitchen), we isolate and study the importance of different representation training methods, data augmentations, and feature hierarchies. Overall, we find that pre-trained visual representations can be competitive or even better than ground-truth state representations to train control policies. This is in spite of using only out-of-domain data from standard vision datasets, without any in-domain data from the deployment environments.

----

## [757] Learning Symmetric Embeddings for Equivariant World Models

**Authors**: *Jung Yeon Park, Ondrej Biza, Linfeng Zhao, Jan-Willem van de Meent, Robin Walters*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/park22a.html](https://proceedings.mlr.press/v162/park22a.html)

**Abstract**:

Incorporating symmetries can lead to highly data-efficient and generalizable models by defining equivalence classes of data samples related by transformations. However, characterizing how transformations act on input data is often difficult, limiting the applicability of equivariant models. We propose learning symmetric embedding networks (SENs) that encode an input space (e.g. images), where we do not know the effect of transformations (e.g. rotations), to a feature space that transforms in a known manner under these operations. This network can be trained end-to-end with an equivariant task network to learn an explicitly symmetric representation. We validate this approach in the context of equivariant transition models with 3 distinct forms of symmetry. Our experiments demonstrate that SENs facilitate the application of equivariant networks to data with complex symmetry representations. Moreover, doing so can yield improvements in accuracy and generalization relative to both fully-equivariant and non-equivariant baselines.

----

## [758] Blurs Behave Like Ensembles: Spatial Smoothings to Improve Accuracy, Uncertainty, and Robustness

**Authors**: *Namuk Park, Songkuk Kim*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/park22b.html](https://proceedings.mlr.press/v162/park22b.html)

**Abstract**:

Neural network ensembles, such as Bayesian neural networks (BNNs), have shown success in the areas of uncertainty estimation and robustness. However, a crucial challenge prohibits their use in practice. BNNs require a large number of predictions to produce reliable results, leading to a significant increase in computational cost. To alleviate this issue, we propose spatial smoothing, a method that ensembles neighboring feature map points of convolutional neural networks. By simply adding a few blur layers to the models, we empirically show that spatial smoothing improves accuracy, uncertainty estimation, and robustness of BNNs across a whole range of ensemble sizes. In particular, BNNs incorporating spatial smoothing achieve high predictive performance merely with a handful of ensembles. Moreover, this method also can be applied to canonical deterministic neural networks to improve the performances. A number of evidences suggest that the improvements can be attributed to the stabilized feature maps and the smoothing of the loss landscape. In addition, we provide a fundamental explanation for prior works {—} namely, global average pooling, pre-activation, and ReLU6 {—} by addressing them as special cases of spatial smoothing. These not only enhance accuracy, but also improve uncertainty estimation and robustness by making the loss landscape smoother in the same manner as spatial smoothing. The code is available at https://github.com/xxxnell/spatial-smoothing.

----

## [759] Exact Optimal Accelerated Complexity for Fixed-Point Iterations

**Authors**: *Jisun Park, Ernest K. Ryu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/park22c.html](https://proceedings.mlr.press/v162/park22c.html)

**Abstract**:

Despite the broad use of fixed-point iterations throughout applied mathematics, the optimal convergence rate of general fixed-point problems with nonexpansive nonlinear operators has not been established. This work presents an acceleration mechanism for fixed-point iterations with nonexpansive operators, contractive operators, and nonexpansive operators satisfying a Hölder-type growth condition. We then provide matching complexity lower bounds to establish the exact optimality of the acceleration mechanisms in the nonexpansive and contractive setups. Finally, we provide experiments with CT imaging, optimal transport, and decentralized optimization to demonstrate the practical effectiveness of the acceleration mechanism.

----

## [760] Kernel Methods for Radial Transformed Compositional Data with Many Zeros

**Authors**: *Junyoung Park, Changwon Yoon, Cheolwoo Park, Jeongyoun Ahn*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/park22d.html](https://proceedings.mlr.press/v162/park22d.html)

**Abstract**:

Compositional data analysis with a high proportion of zeros has gained increasing popularity, especially in chemometrics and human gut microbiomes research. Statistical analyses of this type of data are typically carried out via a log-ratio transformation after replacing zeros with small positive values. We should note, however, that this procedure is geometrically improper, as it causes anomalous distortions through the transformation. We propose a radial transformation that does not require zero substitutions and more importantly results in essential equivalence between domains before and after the transformation. We show that a rich class of kernels on hyperspheres can successfully define a kernel embedding for compositional data based on this equivalence. To the best of our knowledge, this is the first work that theoretically establishes the availability of the extensive library of kernel-based machine learning methods for compositional data. The applicability of the proposed approach is demonstrated with kernel principal component analysis.

----

## [761] Evolving Curricula with Regret-Based Environment Design

**Authors**: *Jack Parker-Holder, Minqi Jiang, Michael Dennis, Mikayel Samvelyan, Jakob N. Foerster, Edward Grefenstette, Tim Rocktäschel*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/parker-holder22a.html](https://proceedings.mlr.press/v162/parker-holder22a.html)

**Abstract**:

Training generally-capable agents with reinforcement learning (RL) remains a significant challenge. A promising avenue for improving the robustness of RL agents is through the use of curricula. One such class of methods frames environment design as a game between a student and a teacher, using regret-based objectives to produce environment instantiations (or levels) at the frontier of the student agent’s capabilities. These methods benefit from theoretical robustness guarantees at equilibrium, yet they often struggle to find effective levels in challenging design spaces in practice. By contrast, evolutionary approaches incrementally alter environment complexity, resulting in potentially open-ended learning, but often rely on domain-specific heuristics and vast amounts of computational resources. This work proposes harnessing the power of evolution in a principled, regret-based curriculum. Our approach, which we call Adversarially Compounding Complexity by Editing Levels (ACCEL), seeks to constantly produce levels at the frontier of an agent’s capabilities, resulting in curricula that start simple but become increasingly complex. ACCEL maintains the theoretical benefits of prior regret-based methods, while providing significant empirical gains in a diverse set of environments. An interactive version of this paper is available at https://accelagent.github.io.

----

## [762] Neural Language Models are not Born Equal to Fit Brain Data, but Training Helps

**Authors**: *Alexandre Pasquiou, Yair Lakretz, John T. Hale, Bertrand Thirion, Christophe Pallier*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pasquiou22a.html](https://proceedings.mlr.press/v162/pasquiou22a.html)

**Abstract**:

Neural Language Models (NLMs) have made tremendous advances during the last years, achieving impressive performance on various linguistic tasks. Capitalizing on this, studies in neuroscience have started to use NLMs to study neural activity in the human brain during language processing. However, many questions remain unanswered regarding which factors determine the ability of a neural language model to capture brain activity (aka its ’brain score’). Here, we make first steps in this direction and examine the impact of test loss, training corpus and model architecture (comparing GloVe, LSTM, GPT-2 and BERT), on the prediction of functional Magnetic Resonance Imaging time-courses of participants listening to an audiobook. We find that (1) untrained versions of each model already explain significant amount of signal in the brain by capturing similarity in brain responses across identical words, with the untrained LSTM outperforming the transformer-based models, being less impacted by the effect of context; (2) that training NLP models improves brain scores in the same brain regions irrespective of the model’s architecture; (3) that Perplexity (test loss) is not a good predictor of brain score; (4) that training data have a strong influence on the outcome and, notably, that off-the-shelf models may lack statistical power to detect brain activations. Overall, we outline the impact of model-training choices, and suggest good practices for future studies aiming at explaining the human language system using neural language models.

----

## [763] A new similarity measure for covariate shift with applications to nonparametric regression

**Authors**: *Reese Pathak, Cong Ma, Martin J. Wainwright*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pathak22a.html](https://proceedings.mlr.press/v162/pathak22a.html)

**Abstract**:

We study covariate shift in the context of nonparametric regression. We introduce a new measure of distribution mismatch between the source and target distributions using the integrated ratio of probabilities of balls at a given radius. We use the scaling of this measure with respect to the radius to characterize the minimax rate of estimation over a family of H{รถ}lder continuous functions under covariate shift. In comparison to the recently proposed notion of transfer exponent, this measure leads to a sharper rate of convergence and is more fine-grained. We accompany our theory with concrete instances of covariate shift that illustrate this sharp difference.

----

## [764] Align-RUDDER: Learning From Few Demonstrations by Reward Redistribution

**Authors**: *Vihang Patil, Markus Hofmarcher, Marius-Constantin Dinu, Matthias Dorfer, Patrick M. Blies, Johannes Brandstetter, José Antonio Arjona-Medina, Sepp Hochreiter*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/patil22a.html](https://proceedings.mlr.press/v162/patil22a.html)

**Abstract**:

Reinforcement learning algorithms require many samples when solving complex hierarchical tasks with sparse and delayed rewards. For such complex tasks, the recently proposed RUDDER uses reward redistribution to leverage steps in the Q-function that are associated with accomplishing sub-tasks. However, often only few episodes with high rewards are available as demonstrations since current exploration strategies cannot discover them in reasonable time. In this work, we introduce Align-RUDDER, which utilizes a profile model for reward redistribution that is obtained from multiple sequence alignment of demonstrations. Consequently, Align-RUDDER employs reward redistribution effectively and, thereby, drastically improves learning on few demonstrations. Align-RUDDER outperforms competitors on complex artificial tasks with delayed rewards and few demonstrations. On the Minecraft ObtainDiamond task, Align-RUDDER is able to mine a diamond, though not frequently. Code is available at github.com/ml-jku/align-rudder.

----

## [765] POET: Training Neural Networks on Tiny Devices with Integrated Rematerialization and Paging

**Authors**: *Shishir G. Patil, Paras Jain, Prabal Dutta, Ion Stoica, Joseph Gonzalez*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/patil22b.html](https://proceedings.mlr.press/v162/patil22b.html)

**Abstract**:

Fine-tuning models on edge devices like mobile phones would enable privacy-preserving personalization over sensitive data. However, edge training has historically been limited to relatively small models with simple architectures because training is both memory and energy intensive. We present POET, an algorithm to enable training large neural networks on memory-scarce battery-operated edge devices. POET jointly optimizes the integrated search search spaces of rematerialization and paging, two algorithms to reduce the memory consumption of backpropagation. Given a memory budget and a run-time constraint, we formulate a mixed-integer linear program (MILP) for energy-optimal training. Our approach enables training significantly larger models on embedded devices while reducing energy consumption while not modifying mathematical correctness of backpropagation. We demonstrate that it is possible to fine-tune both ResNet-18 and BERT within the memory constraints of a Cortex-M class embedded device while outperforming current edge training methods in energy efficiency. POET is an open-source project available at https://github.com/ShishirPatil/poet

----

## [766] Learning to Cut by Looking Ahead: Cutting Plane Selection via Imitation Learning

**Authors**: *Max B. Paulus, Giulia Zarpellon, Andreas Krause, Laurent Charlin, Chris J. Maddison*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/paulus22a.html](https://proceedings.mlr.press/v162/paulus22a.html)

**Abstract**:

Cutting planes are essential for solving mixed-integer linear problems (MILPs), because they facilitate bound improvements on the optimal solution value. For selecting cuts, modern solvers rely on manually designed heuristics that are tuned to gauge the potential effectiveness of cuts. We show that a greedy selection rule explicitly looking ahead to select cuts that yield the best bound improvement delivers strong decisions for cut selection – but is too expensive to be deployed in practice. In response, we propose a new neural architecture (NeuralCut) for imitation learning on the lookahead expert. Our model outperforms standard baselines for cut selection on several synthetic MILP benchmarks. Experiments on a realistic B&C solver further validate our approach, and exhibit the potential of learning methods in this setting.

----

## [767] Neural Network Pruning Denoises the Features and Makes Local Connectivity Emerge in Visual Tasks

**Authors**: *Franco Pellegrini, Giulio Biroli*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pellegrini22a.html](https://proceedings.mlr.press/v162/pellegrini22a.html)

**Abstract**:

Pruning methods can considerably reduce the size of artificial neural networks without harming their performance and in some cases they can even uncover sub-networks that, when trained in isolation, match or surpass the test accuracy of their dense counterparts. Here, we characterize the inductive bias that pruning imprints in such "winning lottery tickets": focusing on visual tasks, we analyze the architecture resulting from iterative magnitude pruning of a simple fully connected network. We show that the surviving node connectivity is local in input space, and organized in patterns reminiscent of the ones found in convolutional networks. We investigate the role played by data and tasks in shaping the architecture of the pruned sub-network. We find that pruning performances, and the ability to sift out the noise and make local features emerge, improve by increasing the size of the training set, and the semantic value of the data. We also study different pruning procedures, and find that iterative magnitude pruning is particularly effective in distilling meaningful connectivity out of features present in the original task. Our results suggest the possibility to automatically discover new and efficient architectural inductive biases in other datasets and tasks.

----

## [768] Branchformer: Parallel MLP-Attention Architectures to Capture Local and Global Context for Speech Recognition and Understanding

**Authors**: *Yifan Peng, Siddharth Dalmia, Ian R. Lane, Shinji Watanabe*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/peng22a.html](https://proceedings.mlr.press/v162/peng22a.html)

**Abstract**:

Conformer has proven to be effective in many speech processing tasks. It combines the benefits of extracting local dependencies using convolutions and global dependencies using self-attention. Inspired by this, we propose a more flexible, interpretable and customizable encoder alternative, Branchformer, with parallel branches for modeling various ranged dependencies in end-to-end speech processing. In each encoder layer, one branch employs self-attention or its variant to capture long-range dependencies, while the other branch utilizes an MLP module with convolutional gating (cgMLP) to extract local relationships. We conduct experiments on several speech recognition and spoken language understanding benchmarks. Results show that our model outperforms both Transformer and cgMLP. It also matches with or outperforms state-of-the-art results achieved by Conformer. Furthermore, we show various strategies to reduce computation thanks to the two-branch architecture, including the ability to have variable inference complexity in a single trained model. The weights learned for merging branches indicate how local and global dependencies are utilized in different layers, which benefits model designing.

----

## [769] Pocket2Mol: Efficient Molecular Sampling Based on 3D Protein Pockets

**Authors**: *Xingang Peng, Shitong Luo, Jiaqi Guan, Qi Xie, Jian Peng, Jianzhu Ma*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/peng22b.html](https://proceedings.mlr.press/v162/peng22b.html)

**Abstract**:

Deep generative models have achieved tremendous success in designing novel drug molecules in recent years. A new thread of works have shown potential in advancing the specificity and success rate of in silico drug design by considering the structure of protein pockets. This setting posts fundamental computational challenges in sampling new chemical compounds that could satisfy multiple geometrical constraints imposed by pockets. Previous sampling algorithms either sample in the graph space or only consider the 3D coordinates of atoms while ignoring other detailed chemical structures such as bond types and functional groups. To address the challenge, we develop an E(3)-equivariant generative network composed of two modules: 1) a new graph neural network capturing both spatial and bonding relationships between atoms of the binding pockets and 2) a new efficient algorithm which samples new drug candidates conditioned on the pocket representations from a tractable distribution without relying on MCMC. Experimental results demonstrate that molecules sampled from Pocket2Mol achieve significantly better binding affinity and other drug properties such as drug-likeness and synthetic accessibility.

----

## [770] Differentiable Top-k Classification Learning

**Authors**: *Felix Petersen, Hilde Kuehne, Christian Borgelt, Oliver Deussen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/petersen22a.html](https://proceedings.mlr.press/v162/petersen22a.html)

**Abstract**:

The top-k classification accuracy is one of the core metrics in machine learning. Here, k is conventionally a positive integer, such as 1 or 5, leading to top-1 or top-5 training objectives. In this work, we relax this assumption and optimize the model for multiple k simultaneously instead of using a single k. Leveraging recent advances in differentiable sorting and ranking, we propose a family of differentiable top-k cross-entropy classification losses. This allows training while not only considering the top-1 prediction, but also, e.g., the top-2 and top-5 predictions. We evaluate the proposed losses for fine-tuning on state-of-the-art architectures, as well as for training from scratch. We find that relaxing k not only produces better top-5 accuracies, but also leads to top-1 accuracy improvements. When fine-tuning publicly available ImageNet models, we achieve a new state-of-the-art for these models.

----

## [771] Multi-scale Feature Learning Dynamics: Insights for Double Descent

**Authors**: *Mohammad Pezeshki, Amartya Mitra, Yoshua Bengio, Guillaume Lajoie*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pezeshki22a.html](https://proceedings.mlr.press/v162/pezeshki22a.html)

**Abstract**:

An intriguing phenomenon that arises from the high-dimensional learning dynamics of neural networks is the phenomenon of “double descent”. The more commonly studied aspect of this phenomenon corresponds to model-wise double descent where the test error exhibits a second descent with increasing model complexity, beyond the classical U-shaped error curve. In this work, we investigate the origins of the less studied epoch-wise double descent in which the test error undergoes two non-monotonous transitions, or descents as the training time increases. We study a linear teacher-student setup exhibiting epoch-wise double descent similar to that in deep neural networks. In this setting, we derive closed-form analytical expressions describing the generalization error in terms of low-dimensional scalar macroscopic variables. We find that double descent can be attributed to distinct features being learned at different scales: as fast-learning features overfit, slower-learning features start to fit, resulting in a second descent in test error. We validate our findings through numerical simulations where our theory accurately predicts empirical findings and remains consistent with observations in deep neural networks.

----

## [772] A Differential Entropy Estimator for Training Neural Networks

**Authors**: *Georg Pichler, Pierre Jean A. Colombo, Malik Boudiaf, Günther Koliander, Pablo Piantanida*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pichler22a.html](https://proceedings.mlr.press/v162/pichler22a.html)

**Abstract**:

Mutual Information (MI) has been widely used as a loss regularizer for training neural networks. This has been particularly effective when learn disentangled or compressed representations of high dimensional data. However, differential entropy (DE), another fundamental measure of information, has not found widespread use in neural network training. Although DE offers a potentially wider range of applications than MI, off-the-shelf DE estimators are either non differentiable, computationally intractable or fail to adapt to changes in the underlying distribution. These drawbacks prevent them from being used as regularizers in neural networks training. To address shortcomings in previously proposed estimators for DE, here we introduce KNIFE, a fully parameterized, differentiable kernel-based estimator of DE. The flexibility of our approach also allows us to construct KNIFE-based estimators for conditional (on either discrete or continuous variables) DE, as well as MI. We empirically validate our method on high-dimensional synthetic data and further apply it to guide the training of neural networks for real-world tasks. Our experiments on a large variety of tasks, including visual domain adaptation, textual fair classification, and textual fine-tuning demonstrate the effectiveness of KNIFE-based estimation. Code can be found at https://github.com/g-pichler/knife.

----

## [773] Federated Learning with Partial Model Personalization

**Authors**: *Krishna Pillutla, Kshitiz Malik, Abdelrahman Mohamed, Michael G. Rabbat, Maziar Sanjabi, Lin Xiao*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pillutla22a.html](https://proceedings.mlr.press/v162/pillutla22a.html)

**Abstract**:

We consider two federated learning algorithms for training partially personalized models, where the shared and personal parameters are updated either simultaneously or alternately on the devices. Both algorithms have been proposed in the literature, but their convergence properties are not fully understood, especially for the alternating variant. We provide convergence analyses of both algorithms in the general nonconvex setting with partial participation and delineate the regime where one dominates the other. Our experiments on real-world image, text, and speech datasets demonstrate that (a) partial personalization can obtain most of the benefits of full model personalization with a small fraction of personal parameters, and, (b) the alternating update algorithm outperforms the simultaneous update algorithm by a small but consistent margin.

----

## [774] Deep Networks on Toroids: Removing Symmetries Reveals the Structure of Flat Regions in the Landscape Geometry

**Authors**: *Fabrizio Pittorino, Antonio Ferraro, Gabriele Perugini, Christoph Feinauer, Carlo Baldassi, Riccardo Zecchina*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pittorino22a.html](https://proceedings.mlr.press/v162/pittorino22a.html)

**Abstract**:

We systematize the approach to the investigation of deep neural network landscapes by basing it on the geometry of the space of implemented functions rather than the space of parameters. Grouping classifiers into equivalence classes, we develop a standardized parameterization in which all symmetries are removed, resulting in a toroidal topology. On this space, we explore the error landscape rather than the loss. This lets us derive a meaningful notion of the flatness of minimizers and of the geodesic paths connecting them. Using different optimization algorithms that sample minimizers with different flatness we study the mode connectivity and relative distances. Testing a variety of state-of-the-art architectures and benchmark datasets, we confirm the correlation between flatness and generalization performance; we further show that in function space flatter minima are closer to each other and that the barriers along the geodesics connecting them are small. We also find that minimizers found by variants of gradient descent can be connected by zero-error paths composed of two straight lines in parameter space, i.e. polygonal chains with a single bend. We observe similar qualitative results in neural networks with binary weights and activations, providing one of the first results concerning the connectivity in this setting. Our results hinge on symmetry removal, and are in remarkable agreement with the rich phenomenology described by some recent analytical studies performed on simple shallow models.

----

## [775] Geometric Multimodal Contrastive Representation Learning

**Authors**: *Petra Poklukar, Miguel Vasco, Hang Yin, Francisco S. Melo, Ana Paiva, Danica Kragic*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/poklukar22a.html](https://proceedings.mlr.press/v162/poklukar22a.html)

**Abstract**:

Learning representations of multimodal data that are both informative and robust to missing modalities at test time remains a challenging problem due to the inherent heterogeneity of data obtained from different channels. To address it, we present a novel Geometric Multimodal Contrastive (GMC) representation learning method consisting of two main components: i) a two-level architecture consisting of modality-specific base encoders, allowing to process an arbitrary number of modalities to an intermediate representation of fixed dimensionality, and a shared projection head, mapping the intermediate representations to a latent representation space; ii) a multimodal contrastive loss function that encourages the geometric alignment of the learned representations. We experimentally demonstrate that GMC representations are semantically rich and achieve state-of-the-art performance with missing modality information on three different learning problems including prediction and reinforcement learning tasks.

----

## [776] Constrained Offline Policy Optimization

**Authors**: *Nicholas Polosky, Bruno C. da Silva, Madalina Fiterau, Jithin Jagannath*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/polosky22a.html](https://proceedings.mlr.press/v162/polosky22a.html)

**Abstract**:

In this work we introduce Constrained Offline Policy Optimization (COPO), an offline policy optimization algorithm for learning in MDPs with cost constraints. COPO is built upon a novel offline cost-projection method, which we formally derive and analyze. Our method improves upon the state-of-the-art in offline constrained policy optimization by explicitly accounting for distributional shift and by offering non-asymptotic confidence bounds on the cost of a policy. These formal properties are superior to those of existing techniques, which only guarantee convergence to a point estimate. We formally analyze our method and empirically demonstrate that it achieves state-of-the-art performance on discrete and continuous control problems, while offering the aforementioned improved, stronger, and more robust theoretical guarantees.

----

## [777] Offline Meta-Reinforcement Learning with Online Self-Supervision

**Authors**: *Vitchyr H. Pong, Ashvin Nair, Laura M. Smith, Catherine Huang, Sergey Levine*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pong22a.html](https://proceedings.mlr.press/v162/pong22a.html)

**Abstract**:

Meta-reinforcement learning (RL) methods can meta-train policies that adapt to new tasks with orders of magnitude less data than standard RL, but meta-training itself is costly and time-consuming. If we can meta-train on offline data, then we can reuse the same static dataset, labeled once with rewards for different tasks, to meta-train policies that adapt to a variety of new tasks at meta-test time. Although this capability would make meta-RL a practical tool for real-world use, offline meta-RL presents additional challenges beyond online meta-RL or standard offline RL settings. Meta-RL learns an exploration strategy that collects data for adapting, and also meta-trains a policy that quickly adapts to data from a new task. Since this policy was meta-trained on a fixed, offline dataset, it might behave unpredictably when adapting to data collected by the learned exploration strategy, which differs systematically from the offline data and thus induces distributional shift. We propose a hybrid offline meta-RL algorithm, which uses offline data with rewards to meta-train an adaptive policy, and then collects additional unsupervised online data, without any reward labels to bridge this distribution shift. By not requiring reward labels for online collection, this data can be much cheaper to collect. We compare our method to prior work on offline meta-RL on simulated robot locomotion and manipulation tasks and find that using additional unsupervised online data collection leads to a dramatic improvement in the adaptive capabilities of the meta-trained policies, matching the performance of fully online meta-RL on a range of challenging domains that require generalization to new tasks.

----

## [778] Debiaser Beware: Pitfalls of Centering Regularized Transport Maps

**Authors**: *Aram-Alexandre Pooladian, Marco Cuturi, Jonathan Niles-Weed*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pooladian22a.html](https://proceedings.mlr.press/v162/pooladian22a.html)

**Abstract**:

Estimating optimal transport (OT) maps (a.k.a. Monge maps) between two measures P and Q is a problem fraught with computational and statistical challenges. A promising approach lies in using the dual potential functions obtained when solving an entropy-regularized OT problem between samples P_n and Q_n, which can be used to recover an approximately optimal map. The negentropy penalization in that scheme introduces, however, an estimation bias that grows with the regularization strength. A well-known remedy to debias such estimates, which has gained wide popularity among practitioners of regularized OT, is to center them, by subtracting auxiliary problems involving P_n and itself, as well as Q_n and itself. We do prove that, under favorable conditions on P and Q, debiasing can yield better approximations to the Monge map. However, and perhaps surprisingly, we present a few cases in which debiasing is provably detrimental in a statistical sense, notably when the regularization strength is large or the number of samples is small. These claims are validated experimentally on synthetic and real datasets, and should reopen the debate on whether debiasing is needed when using entropic OT.

----

## [779] Adaptive Second Order Coresets for Data-efficient Machine Learning

**Authors**: *Omead Pooladzandi, David Davini, Baharan Mirzasoleiman*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/pooladzandi22a.html](https://proceedings.mlr.press/v162/pooladzandi22a.html)

**Abstract**:

Training machine learning models on massive datasets incurs substantial computational costs. To alleviate such costs, there has been a sustained effort to develop data-efficient training methods that can carefully select subsets of the training examples that generalize on par with the full training data. However, existing methods are limited in providing theoretical guarantees for the quality of the models trained on the extracted subsets, and may perform poorly in practice. We propose AdaCore, a method that leverages the geometry of the data to extract subsets of the training examples for efficient machine learning. The key idea behind our method is to dynamically approximate the curvature of the loss function via an exponentially-averaged estimate of the Hessian to select weighted subsets (coresets) that provide a close approximation of the full gradient preconditioned with the Hessian. We prove rigorous guarantees for the convergence of various first and second-order methods applied to the subsets chosen by AdaCore. Our extensive experiments show that AdaCore extracts coresets with higher quality compared to baselines and speeds up training of convex and non-convex machine learning models, such as logistic regression and neural networks, by over 2.9x over the full data and 4.5x over random subsets.

----

## [780] On the Practicality of Deterministic Epistemic Uncertainty

**Authors**: *Janis Postels, Mattia Segù, Tao Sun, Luca Daniel Sieber, Luc Van Gool, Fisher Yu, Federico Tombari*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/postels22a.html](https://proceedings.mlr.press/v162/postels22a.html)

**Abstract**:

A set of novel approaches for estimating epistemic uncertainty in deep neural networks with a single forward pass has recently emerged as a valid alternative to Bayesian Neural Networks. On the premise of informative representations, these deterministic uncertainty methods (DUMs) achieve strong performance on detecting out-of-distribution (OOD) data while adding negligible computational costs at inference time. However, it remains unclear whether DUMs are well calibrated and can seamlessly scale to real-world applications - both prerequisites for their practical deployment. To this end, we first provide a taxonomy of DUMs, and evaluate their calibration under continuous distributional shifts. Then, we extend them to semantic segmentation. We find that, while DUMs scale to realistic vision tasks and perform well on OOD detection, the practicality of current methods is undermined by poor calibration under distributional shifts.

----

## [781] A Simple Guard for Learned Optimizers

**Authors**: *Isabeau Prémont-Schwarz, Jaroslav Vitku, Jan Feyereisl*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/premont-schwarz22a.html](https://proceedings.mlr.press/v162/premont-schwarz22a.html)

**Abstract**:

If the trend of learned components eventually outperforming their hand-crafted version continues, learned optimizers will eventually outperform hand-crafted optimizers like SGD or Adam. Even if learned optimizers (L2Os) eventually outpace hand-crafted ones in practice however, they are still not provably convergent and might fail out of distribution. These are the questions addressed here. Currently, learned optimizers frequently outperform generic hand-crafted optimizers (such as gradient descent) at the beginning of learning but they generally plateau after some time while the generic algorithms continue to make progress and often overtake the learned algorithm as Aesop’s tortoise which overtakes the hare. L2Os also still have a difficult time generalizing out of distribution. \cite{heaton_safeguarded_2020} proposed Safeguarded L2O (GL2O) which can take a learned optimizer and safeguard it with a generic learning algorithm so that by conditionally switching between the two, the resulting algorithm is provably convergent. We propose a new class of Safeguarded L2O, called Loss-Guarded L2O (LGL2O), which is both conceptually simpler and computationally less expensive. The guarding mechanism decides solely based on the expected future loss value of both optimizers. Furthermore, we show theoretical proof of LGL2O’s convergence guarantee and empirical results comparing to GL2O and other baselines showing that it combines the best of both L2O and SGD and that in practice converges much better than GL2O.

----

## [782] Hardness and Algorithms for Robust and Sparse Optimization

**Authors**: *Eric Price, Sandeep Silwal, Samson Zhou*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/price22a.html](https://proceedings.mlr.press/v162/price22a.html)

**Abstract**:

We explore algorithms and limitations for sparse optimization problems such as sparse linear regression and robust linear regression. The goal of the sparse linear regression problem is to identify a small number of key features, while the goal of the robust linear regression problem is to identify a small number of erroneous measurements. Specifically, the sparse linear regression problem seeks a $k$-sparse vector $x\in\mathbb{R}^d$ to minimize $\|Ax-b\|_2$, given an input matrix $A\in\mathbb{R}^{n\times d}$ and a target vector $b\in\mathbb{R}^n$, while the robust linear regression problem seeks a set $S$ that ignores at most $k$ rows and a vector $x$ to minimize $\|(Ax-b)_S\|_2$. We first show bicriteria, NP-hardness of approximation for robust regression building on the work of \cite{ODonnellWZ15} which implies a similar result for sparse regression. We further show fine-grained hardness of robust regression through a reduction from the minimum-weight $k$-clique conjecture. On the positive side, we give an algorithm for robust regression that achieves arbitrarily accurate additive error and uses runtime that closely matches the lower bound from the fine-grained hardness result, as well as an algorithm for sparse regression with similar runtime. Both our upper and lower bounds rely on a general reduction from robust linear regression to sparse regression that we introduce. Our algorithms, inspired by the 3SUM problem, use approximate nearest neighbor data structures and may be of independent interest for solving sparse optimization problems. For instance, we demonstrate that our techniques can also be used for the well-studied sparse PCA problem.

----

## [783] Nonlinear Feature Diffusion on Hypergraphs

**Authors**: *Konstantin Prokopchik, Austin R. Benson, Francesco Tudisco*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/prokopchik22a.html](https://proceedings.mlr.press/v162/prokopchik22a.html)

**Abstract**:

Hypergraphs are a common model for multiway relationships in data, and hypergraph semi-supervised learning is the problem of assigning labels to all nodes in a hypergraph, given labels on just a few nodes. Diffusions and label spreading are classical techniques for semi-supervised learning in the graph setting, and there are some standard ways to extend them to hypergraphs. However, these methods are linear models, and do not offer an obvious way of incorporating node features for making predictions. Here, we develop a nonlinear diffusion process on hypergraphs that spreads both features and labels following the hypergraph structure. Even though the process is nonlinear, we show global convergence to a unique limiting point for a broad class of nonlinearities and we show that such limit is the global minimum of a new regularized semi-supervised learning loss function which aims at reducing a generalized form of variance of the nodes across the hyperedges. The limiting point serves as a node embedding from which we make predictions with a linear model. Our approach is competitive with state-of-the-art graph and hypergraph neural networks, and also takes less time to train.

----

## [784] Universal Joint Approximation of Manifolds and Densities by Simple Injective Flows

**Authors**: *Michael Puthawala, Matti Lassas, Ivan Dokmanic, Maarten V. de Hoop*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/puthawala22a.html](https://proceedings.mlr.press/v162/puthawala22a.html)

**Abstract**:

We study approximation of probability measures supported on n-dimensional manifolds embedded in R^m by injective flows—neural networks composed of invertible flows and injective layers. We show that in general, injective flows between R^n and R^m universally approximate measures supported on images of extendable embeddings, which are a subset of standard embeddings: when the embedding dimension m is small, topological obstructions may preclude certain manifolds as admissible targets. When the embedding dimension is sufficiently large, m >= 3n+1, we use an argument from algebraic topology known as the clean trick to prove that the topological obstructions vanish and injective flows universally approximate any differentiable embedding. Along the way we show that the studied injective flows admit efficient projections on the range, and that their optimality can be established "in reverse," resolving a conjecture made in Brehmer & Cranmer 2020.

----

## [785] The Teaching Dimension of Regularized Kernel Learners

**Authors**: *Hong Qian, Xu-Hui Liu, Chen-Xi Su, Aimin Zhou, Yang Yu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/qian22a.html](https://proceedings.mlr.press/v162/qian22a.html)

**Abstract**:

Teaching dimension (TD) is a fundamental theoretical property for understanding machine teaching algorithms. It measures the sample complexity of teaching a target hypothesis to a learner. The TD of linear learners has been studied extensively, whereas the results of teaching non-linear learners are rare. A recent result investigates the TD of polynomial and Gaussian kernel learners. Unfortunately, the theoretical bounds therein show that the TD is high when teaching those non-linear learners. Inspired by the fact that regularization can reduce the learning complexity in machine learning, a natural question is whether the similar fact happens in machine teaching. To answer this essential question, this paper proposes a unified theoretical framework termed STARKE to analyze the TD of regularized kernel learners. On the basis of STARKE, we derive a generic result of any type of kernels. Furthermore, we disclose that the TD of regularized linear and regularized polynomial kernel learners can be strictly reduced. For regularized Gaussian kernel learners, we reveal that, although their TD is infinite, their epsilon-approximate TD can be exponentially reduced compared with that of the unregularized learners. The extensive experimental results of teaching the optimization-based learners verify the theoretical findings.

----

## [786] ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers

**Authors**: *Kaizhi Qian, Yang Zhang, Heting Gao, Junrui Ni, Cheng-I Lai, David D. Cox, Mark Hasegawa-Johnson, Shiyu Chang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/qian22b.html](https://proceedings.mlr.press/v162/qian22b.html)

**Abstract**:

Self-supervised learning in speech involves training a speech representation network on a large-scale unannotated speech corpus, and then applying the learned representations to downstream tasks. Since the majority of the downstream tasks of SSL learning in speech largely focus on the content information in speech, the most desirable speech representations should be able to disentangle unwanted variations, such as speaker variations, from the content. However, disentangling speakers is very challenging, because removing the speaker information could easily result in a loss of content as well, and the damage of the latter usually far outweighs the benefit of the former. In this paper, we propose a new SSL method that can achieve speaker disentanglement without severe loss of content. Our approach is adapted from the HuBERT framework, and incorporates disentangling mechanisms to regularize both the teacher labels and the learned representations. We evaluate the benefit of speaker disentanglement on a set of content-related downstream tasks, and observe a consistent and notable performance advantage of our speaker-disentangled representations.

----

## [787] Interventional Contrastive Learning with Meta Semantic Regularizer

**Authors**: *Wenwen Qiang, Jiangmeng Li, Changwen Zheng, Bing Su, Hui Xiong*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/qiang22a.html](https://proceedings.mlr.press/v162/qiang22a.html)

**Abstract**:

Contrastive learning (CL)-based self-supervised learning models learn visual representations in a pairwise manner. Although the prevailing CL model has achieved great progress, in this paper, we uncover an ever-overlooked phenomenon: When the CL model is trained with full images, the performance tested in full images is better than that in foreground areas; when the CL model is trained with foreground areas, the performance tested in full images is worse than that in foreground areas. This observation reveals that backgrounds in images may interfere with the model learning semantic information and their influence has not been fully eliminated. To tackle this issue, we build a Structural Causal Model (SCM) to model the background as a confounder. We propose a backdoor adjustment-based regularization method, namely Interventional Contrastive Learning with Meta Semantic Regularizer (ICL-MSR), to perform causal intervention towards the proposed SCM. ICL-MSR can be incorporated into any existing CL methods to alleviate background distractions from representation learning. Theoretically, we prove that ICL-MSR achieves a tighter error bound. Empirically, our experiments on multiple benchmark datasets demonstrate that ICL-MSR is able to improve the performances of different state-of-the-art CL methods.

----

## [788] Sample-Efficient Reinforcement Learning with loglog(T) Switching Cost

**Authors**: *Dan Qiao, Ming Yin, Ming Min, Yu-Xiang Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/qiao22a.html](https://proceedings.mlr.press/v162/qiao22a.html)

**Abstract**:

We study the problem of reinforcement learning (RL) with low (policy) switching cost {—} a problem well-motivated by real-life RL applications in which deployments of new policies are costly and the number of policy updates must be low. In this paper, we propose a new algorithm based on stage-wise exploration and adaptive policy elimination that achieves a regret of $\widetilde{O}(\sqrt{H^4S^2AT})$ while requiring a switching cost of $O(HSA \log\log T)$. This is an exponential improvement over the best-known switching cost $O(H^2SA\log T)$ among existing methods with $\widetilde{O}(\mathrm{poly}(H,S,A)\sqrt{T})$ regret. In the above, $S,A$ denotes the number of states and actions in an $H$-horizon episodic Markov Decision Process model with unknown transitions, and $T$ is the number of steps. As a byproduct of our new techniques, we also derive a reward-free exploration algorithm with a switching cost of $O(HSA)$. Furthermore, we prove a pair of information-theoretical lower bounds which say that (1) Any no-regret algorithm must have a switching cost of $\Omega(HSA)$; (2) Any $\widetilde{O}(\sqrt{T})$ regret algorithm must incur a switching cost of $\Omega(HSA\log\log T)$. Both our algorithms are thus optimal in their switching costs.

----

## [789] Generalizing to Evolving Domains with Latent Structure-Aware Sequential Autoencoder

**Authors**: *Tiexin Qin, Shiqi Wang, Haoliang Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/qin22a.html](https://proceedings.mlr.press/v162/qin22a.html)

**Abstract**:

Domain generalization aims to improve the generalization capability of machine learning systems to out-of-distribution (OOD) data. Existing domain generalization techniques embark upon stationary and discrete environments to tackle the generalization issue caused by OOD data. However, many real-world tasks in non-stationary environments (e.g., self-driven car system, sensor measures) involve more complex and continuously evolving domain drift, which raises new challenges for the problem of domain generalization. In this paper, we formulate the aforementioned setting as the problem of evolving domain generalization. Specifically, we propose to introduce a probabilistic framework called Latent Structure-aware Sequential Autoencoder (LSSAE) to tackle the problem of evolving domain generalization via exploring the underlying continuous structure in the latent space of deep neural networks, where we aim to identify two major factors namely covariate shift and concept shift accounting for distribution shift in non-stationary environments. Experimental results on both synthetic and real-world datasets show that LSSAE can lead to superior performances based on the evolving domain generalization setting.

----

## [790] Graph Neural Architecture Search Under Distribution Shifts

**Authors**: *Yijian Qin, Xin Wang, Ziwei Zhang, Pengtao Xie, Wenwu Zhu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/qin22b.html](https://proceedings.mlr.press/v162/qin22b.html)

**Abstract**:

Graph neural architecture search has shown great potentials for automatically designing graph neural network (GNN) architectures for graph classification tasks. However, when there is a distribution shift between training and testing graphs, the existing approaches fail to deal with the problem of adapting to unknown test graph structures since they only search for a fixed architecture for all graphs. To solve this problem, we propose a novel GRACES model which is able to generalize under distribution shifts through tailoring a customized GNN architecture suitable for each graph instance with unknown distribution. Specifically, we design a self-supervised disentangled graph encoder to characterize invariant factors hidden in diverse graph structures. Then, we propose a prototype-based architecture customization strategy to generate the most suitable GNN architecture weights in a continuous space for each graph instance. We further propose a customized super-network to share weights among different architectures for the sake of efficient training. Extensive experiments on both synthetic and real-world datasets demonstrate that our proposed GRACES model can adapt to diverse graph structures and achieve state-of-the-art performance for graph classification tasks under distribution shifts.

----

## [791] Spectral Representation of Robustness Measures for Optimization Under Input Uncertainty

**Authors**: *Jixiang Qing, Tom Dhaene, Ivo Couckuyt*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/qing22a.html](https://proceedings.mlr.press/v162/qing22a.html)

**Abstract**:

We study the inference of mean-variance robustness measures to quantify input uncertainty under the Gaussian Process (GP) framework. These measures are widely used in applications where the robustness of the solution is of interest, for example, in engineering design. While the variance is commonly used to characterize the robustness, Bayesian inference of the variance using GPs is known to be challenging. In this paper, we propose a Spectral Representation of Robustness Measures based on the GP’s spectral representation, i.e., an analytical approach to approximately infer both robustness measures for normal and uniform input uncertainty distributions. We present two approximations based on different Fourier features and compare their accuracy numerically. To demonstrate their utility and efficacy in robust Bayesian Optimization, we integrate the analytical robustness measures in three standard acquisition functions for various robust optimization formulations. We show their competitive performance on numerical benchmarks and real-life applications.

----

## [792] Large-scale Stochastic Optimization of NDCG Surrogates for Deep Learning with Provable Convergence

**Authors**: *Zi-Hao Qiu, Quanqi Hu, Yongjian Zhong, Lijun Zhang, Tianbao Yang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/qiu22a.html](https://proceedings.mlr.press/v162/qiu22a.html)

**Abstract**:

NDCG, namely Normalized Discounted Cumulative Gain, is a widely used ranking metric in information retrieval and machine learning. However, efficient and provable stochastic methods for maximizing NDCG are still lacking, especially for deep models. In this paper, we propose a principled approach to optimize NDCG and its top-$K$ variant. First, we formulate a novel compositional optimization problem for optimizing the NDCG surrogate, and a novel bilevel compositional optimization problem for optimizing the top-$K$ NDCG surrogate. Then, we develop efficient stochastic algorithms with provable convergence guarantees for the non-convex objectives. Different from existing NDCG optimization methods, the per-iteration complexity of our algorithms scales with the mini-batch size instead of the number of total items. To improve the effectiveness for deep learning, we further propose practical strategies by using initial warm-up and stop gradient operator. Experimental results on multiple datasets demonstrate that our methods outperform prior ranking approaches in terms of NDCG. To the best of our knowledge, this is the first time that stochastic algorithms are proposed to optimize NDCG with a provable convergence guarantee. Our proposed methods are implemented in the LibAUC library at https://libauc.org.

----

## [793] Latent Outlier Exposure for Anomaly Detection with Contaminated Data

**Authors**: *Chen Qiu, Aodong Li, Marius Kloft, Maja Rudolph, Stephan Mandt*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/qiu22b.html](https://proceedings.mlr.press/v162/qiu22b.html)

**Abstract**:

Anomaly detection aims at identifying data points that show systematic deviations from the majority of data in an unlabeled dataset. A common assumption is that clean training data (free of anomalies) is available, which is often violated in practice. We propose a strategy for training an anomaly detector in the presence of unlabeled anomalies that is compatible with a broad class of models. The idea is to jointly infer binary labels to each datum (normal vs. anomalous) while updating the model parameters. Inspired by outlier exposure (Hendrycks et al., 2018) that considers synthetically created, labeled anomalies, we thereby use a combination of two losses that share parameters: one for the normal and one for the anomalous data. We then iteratively proceed with block coordinate updates on the parameters and the most likely (latent) labels. Our experiments with several backbone models on three image datasets, 30 tabular data sets, and a video anomaly detection benchmark showed consistent and significant improvements over the baselines.

----

## [794] Contrastive UCB: Provably Efficient Contrastive Self-Supervised Learning in Online Reinforcement Learning

**Authors**: *Shuang Qiu, Lingxiao Wang, Chenjia Bai, Zhuoran Yang, Zhaoran Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/qiu22c.html](https://proceedings.mlr.press/v162/qiu22c.html)

**Abstract**:

In view of its power in extracting feature representation, contrastive self-supervised learning has been successfully integrated into the practice of (deep) reinforcement learning (RL), leading to efficient policy learning on various applications. Despite its tremendous empirical successes, the understanding of contrastive learning for RL remains elusive. To narrow such a gap, we study contrastive-learning empowered RL for a class of Markov decision processes (MDPs) and Markov games (MGs) with low-rank transitions. For both models, we propose to extract the correct feature representations of the low-rank model by minimizing a contrastive loss. Moreover, under the online setting, we propose novel upper confidence bound (UCB)-type algorithms that incorporate such a contrastive loss with online RL algorithms for MDPs or MGs. We further theoretically prove that our algorithm recovers the true representations and simultaneously achieves sample efficiency in learning the optimal policy and Nash equilibrium in MDPs and MGs. We also provide empirical studies to demonstrate the efficacy of the UCB-based contrastive learning method for RL. To the best of our knowledge, we provide the first provably efficient online RL algorithm that incorporates contrastive learning for representation learning.

----

## [795] Fast and Provable Nonconvex Tensor RPCA

**Authors**: *Haiquan Qiu, Yao Wang, Shaojie Tang, Deyu Meng, Quanming Yao*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/qiu22d.html](https://proceedings.mlr.press/v162/qiu22d.html)

**Abstract**:

In this paper, we study nonconvex tensor robust principal component analysis (RPCA) based on the $t$-SVD. We first propose an alternating projection method, i.e., APT, which converges linearly to the ground-truth under the incoherence conditions of tensors. However, as the projection to the low-rank tensor space in APT can be slow, we further propose to speedup such a process by utilizing the property of the tangent space of low-rank. The resulting algorithm, i.e., EAPT, is not only more efficient than APT but also keeps the linear convergence. Compared with existing tensor RPCA works, the proposed method, especially EAPT, is not only more effective due to the recovery guarantee and adaption in the transformed (frequency) domain but also more efficient due to faster convergence rate and lower iteration complexity. These benefits are also empirically verified both on synthetic data, and real applications, e.g., hyperspectral image denoising and video background subtraction.

----

## [796] Generalized Federated Learning via Sharpness Aware Minimization

**Authors**: *Zhe Qu, Xingyu Li, Rui Duan, Yao Liu, Bo Tang, Zhuo Lu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/qu22a.html](https://proceedings.mlr.press/v162/qu22a.html)

**Abstract**:

Federated Learning (FL) is a promising framework for performing privacy-preserving, distributed learning with a set of clients. However, the data distribution among clients often exhibits non-IID, i.e., distribution shift, which makes efficient optimization difficult. To tackle this problem, many FL algorithms focus on mitigating the effects of data heterogeneity across clients by increasing the performance of the global model. However, almost all algorithms leverage Empirical Risk Minimization (ERM) to be the local optimizer, which is easy to make the global model fall into a sharp valley and increase a large deviation of parts of local clients. Therefore, in this paper, we revisit the solutions to the distribution shift problem in FL with a focus on local learning generality. To this end, we propose a general, effective algorithm, \texttt{FedSAM}, based on Sharpness Aware Minimization (SAM) local optimizer, and develop a momentum FL algorithm to bridge local and global models, \texttt{MoFedSAM}. Theoretically, we show the convergence analysis of these two algorithms and demonstrate the generalization bound of \texttt{FedSAM}. Empirically, our proposed algorithms substantially outperform existing FL studies and significantly decrease the learning deviation.

----

## [797] Particle Transformer for Jet Tagging

**Authors**: *Huilin Qu, Congqiao Li, Sitian Qian*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/qu22b.html](https://proceedings.mlr.press/v162/qu22b.html)

**Abstract**:

Jet tagging is a critical yet challenging classification task in particle physics. While deep learning has transformed jet tagging and significantly improved performance, the lack of a large-scale public dataset impedes further enhancement. In this work, we present JetClass, a new comprehensive dataset for jet tagging. The JetClass dataset consists of 100 M jets, about two orders of magnitude larger than existing public datasets. A total of 10 types of jets are simulated, including several types unexplored for tagging so far. Based on the large dataset, we propose a new Transformer-based architecture for jet tagging, called Particle Transformer (ParT). By incorporating pairwise particle interactions in the attention mechanism, ParT achieves higher tagging performance than a plain Transformer and surpasses the previous state-of-the-art, ParticleNet, by a large margin. The pre-trained ParT models, once fine-tuned, also substantially enhance the performance on two widely adopted jet tagging benchmarks. The dataset, code and models are publicly available at https://github.com/jet-universe/particle_transformer.

----

## [798] Winning the Lottery Ahead of Time: Efficient Early Network Pruning

**Authors**: *John Rachwan, Daniel Zügner, Bertrand Charpentier, Simon Geisler, Morgane Ayle, Stephan Günnemann*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rachwan22a.html](https://proceedings.mlr.press/v162/rachwan22a.html)

**Abstract**:

Pruning, the task of sparsifying deep neural networks, received increasing attention recently. Although state-of-the-art pruning methods extract highly sparse models, they neglect two main challenges: (1) the process of finding these sparse models is often very expensive; (2) unstructured pruning does not provide benefits in terms of GPU memory, training time, or carbon emissions. We propose Early Compression via Gradient Flow Preservation (EarlyCroP), which efficiently extracts state-of-the-art sparse models before or early in training addressing challenge (1), and can be applied in a structured manner addressing challenge (2). This enables us to train sparse networks on commodity GPUs whose dense versions would be too large, thereby saving costs and reducing hardware requirements. We empirically show that EarlyCroP outperforms a rich set of baselines for many tasks (incl. classification, regression) and domains (incl. computer vision, natural language processing, and reinforcment learning). EarlyCroP leads to accuracy comparable to dense training while outperforming pruning baselines.

----

## [799] Convergence of Uncertainty Sampling for Active Learning

**Authors**: *Anant Raj, Francis R. Bach*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/raj22a.html](https://proceedings.mlr.press/v162/raj22a.html)

**Abstract**:

Uncertainty sampling in active learning is heavily used in practice to reduce the annotation cost. However, there has been no wide consensus on the function to be used for uncertainty estimation in binary classification tasks and convergence guarantees of the corresponding active learning algorithms are not well understood. The situation is even more challenging for multi-category classification. In this work, we propose an efficient uncertainty estimator for binary classification which we also extend to multiple classes, and provide a non-asymptotic rate of convergence for our uncertainty sampling based active learning algorithm in both cases under no-noise conditions (i.e., linearly separable data). We also extend our analysis to the noisy case and provide theoretical guarantees for our algorithm under the influence of noise in the task of binary and multi-class classification.

----



[Go to the previous page](ICML-2022-list03.md)

[Go to the next page](ICML-2022-list05.md)

[Go to the catalog section](README.md)