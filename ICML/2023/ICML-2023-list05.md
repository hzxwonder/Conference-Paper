## [800] Prototype-oriented unsupervised anomaly detection for multivariate time series

**Authors**: *Yuxin Li, Wenchao Chen, Bo Chen, Dongsheng Wang, Long Tian, Mingyuan Zhou*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23d.html](https://proceedings.mlr.press/v202/li23d.html)

**Abstract**:

Unsupervised anomaly detection (UAD) of multivariate time series (MTS) aims to learn robust representations of normal multivariate temporal patterns. Existing UAD methods try to learn a fixed set of mappings for each MTS, entailing expensive computation and limited model adaptation. To address this pivotal issue, we propose a prototype-oriented UAD (PUAD) method under a probabilistic framework. Specifically, instead of learning the mappings for each MTS, the proposed PUAD views multiple MTSs as the distribution over a group of prototypes, which are extracted to represent a diverse set of normal patterns. To learn and regulate the prototypes, PUAD introduces a reconstruction-based unsupervised anomaly detection approach, which incorporates a prototype-oriented optimal transport method into a Transformer-powered probabilistic dynamical generative framework. Leveraging meta-learned transferable prototypes, PUAD can achieve high model adaptation capacity for new MTSs. Experiments on five public MTS datasets all verify the effectiveness of the proposed UAD method.

----

## [801] Learning Preconditioners for Conjugate Gradient PDE Solvers

**Authors**: *Yichen Li, Peter Yichen Chen, Tao Du, Wojciech Matusik*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23e.html](https://proceedings.mlr.press/v202/li23e.html)

**Abstract**:

Efficient numerical solvers for partial differential equations empower science and engineering. One commonly employed numerical solver is the preconditioned conjugate gradient (PCG) algorithm, whose performance is largely affected by the preconditioner quality. However, designing high-performing preconditioner with traditional numerical methods is highly non-trivial, often requiring problem-specific knowledge and meticulous matrix operations. We present a new method that leverages learning-based approach to obtain an approximate matrix factorization to the system matrix to be used as a preconditioner in the context of PCG solvers. Our high-level intuition comes from the shared property between preconditioners and network-based PDE solvers that excels at obtaining approximate solutions at a low computational cost. Such observation motivates us to represent preconditioners as graph neural networks (GNNs). In addition, we propose a new loss function that rewrites traditional preconditioner metrics to incorporate inductive bias from PDE data distributions, enabling effective training of high-performing preconditioners. We conduct extensive experiments to demonstrate the efficacy and generalizability of our proposed approach on solving various 2D and 3D linear second-order PDEs.

----

## [802] Parallel Q-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation

**Authors**: *Zechu Li, Tao Chen, Zhang-Wei Hong, Anurag Ajay, Pulkit Agrawal*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23f.html](https://proceedings.mlr.press/v202/li23f.html)

**Abstract**:

Reinforcement learning is time-consuming for complex tasks due to the need for large amounts of training data. Recent advances in GPU-based simulation, such as Isaac Gym, have sped up data collection thousands of times on a commodity GPU. Most prior works have used on-policy methods like PPO due to their simplicity and easy-to-scale nature. Off-policy methods are more sample-efficient, but challenging to scale, resulting in a longer wall-clock training time. This paper presents a novel Parallel Q-Learning (PQL) scheme that outperforms PPO in terms of wall-clock time and maintains superior sample efficiency. The driving force lies in the parallelization of data collection, policy function learning, and value function learning. Different from prior works on distributed off-policy learning, such as Apex, our scheme is designed specifically for massively parallel GPU-based simulation and optimized to work on a single workstation. In experiments, we demonstrate the capability of scaling up Q-learning methods to tens of thousands of parallel environments and investigate important factors that can affect learning speed, including the number of parallel environments, exploration strategies, batch size, GPU models, etc. The code is available at https://github.com/Improbable-AI/pql.

----

## [803] Minimum Width of Leaky-ReLU Neural Networks for Uniform Universal Approximation

**Authors**: *Li'ang Li, Yifei Duan, Guanghua Ji, Yongqiang Cai*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23g.html](https://proceedings.mlr.press/v202/li23g.html)

**Abstract**:

The study of universal approximation properties (UAP) for neural networks (NN) has a long history. When the network width is unlimited, only a single hidden layer is sufficient for UAP. In contrast, when the depth is unlimited, the width for UAP needs to be not less than the critical width $w^*_{\min}=\max(d_x,d_y)$, where $d_x$ and $d_y$ are the dimensions of the input and output, respectively. Recently, (Cai, 2022) shows that a leaky-ReLU NN with this critical width can achieve UAP for $L^p$ functions on a compact domain $\mathcal{K}$, i.e., the UAP for $L^p(\mathcal{K},\mathbb{R}^{d_y})$. This paper examines a uniform UAP for the function class $C(\mathcal{K},\mathbb{R}^{d_y})$ and gives the exact minimum width of the leaky-ReLU NN as $w_{\min}=\max(d_x+1,d_y)+1_{d_y=d_x+1}$, which involves the effects of the output dimensions. To obtain this result, we propose a novel lift-flow-discretization approach that shows that the uniform UAP has a deep connection with topological theory.

----

## [804] FAIRER: Fairness as Decision Rationale Alignment

**Authors**: *Tianlin Li, Qing Guo, Aishan Liu, Mengnan Du, Zhiming Li, Yang Liu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23h.html](https://proceedings.mlr.press/v202/li23h.html)

**Abstract**:

Deep neural networks (DNNs) have made significant progress, but often suffer from fairness issues, as deep models typically show distinct accuracy differences among certain subgroups (e.g., males and females). Existing research addresses this critical issue by employing fairness-aware loss functions to constrain the last-layer outputs and directly regularize DNNs. Although the fairness of DNNs is improved, it is unclear how the trained network makes a fair prediction, which limits future fairness improvements. In this paper, we investigate fairness from the perspective of decision rationale and define the parameter parity score to characterize the fair decision process of networks by analyzing neuron influence in various subgroups. Extensive empirical studies show that the unfair issue could arise from the unaligned decision rationales of subgroups. Existing fairness regularization terms fail to achieve decision rationale alignment because they only constrain last-layer outputs while ignoring intermediate neuron alignment. To address the issue, we formulate the fairness as a new task, i.e., decision rationale alignment that requires DNNs’ neurons to have consistent responses on subgroups at both intermediate processes and the final prediction. To make this idea practical during optimization, we relax the naive objective function and propose gradient-guided parity alignment, which encourages gradient-weighted consistency of neurons across subgroups. Extensive experiments on a variety of datasets show that our method can significantly enhance fairness while sustaining a high level of accuracy and outperforming other approaches by a wide margin.

----

## [805] RACE: Improve Multi-Agent Reinforcement Learning with Representation Asymmetry and Collaborative Evolution

**Authors**: *Pengyi Li, Jianye Hao, Hongyao Tang, Yan Zheng, Xian Fu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23i.html](https://proceedings.mlr.press/v202/li23i.html)

**Abstract**:

Multi-Agent Reinforcement Learning (MARL) has demonstrated its effectiveness in learning collaboration, but it often struggles with low-quality reward signals and high non-stationarity. In contrast, Evolutionary Algorithm (EA) has shown better convergence, robustness, and signal quality insensitivity. This paper introduces a hybrid framework, Representation Asymmetry and Collaboration Evolution (RACE), which combines EA and MARL for efficient collaboration. RACE maintains a MARL team and a population of EA teams. To enable efficient knowledge sharing and policy exploration, RACE decomposes the policies of different teams controlling the same agent into a shared nonlinear observation representation encoder and individual linear policy representations. To address the partial observation issue, we introduce Value-Aware Mutual Information Maximization to enhance the shared representation with useful information about superior global states. EA evolves the population using novel agent-level crossover and mutation operators, offering diverse experiences for MARL. Concurrently, MARL optimizes its policies and injects them into the population for evolution. The experiments on challenging continuous and discrete tasks demonstrate that RACE significantly improves the basic algorithms, consistently outperforming other algorithms. Our code is available at https://github.com/yeshenpy/RACE.

----

## [806] Adversarial Collaborative Learning on Non-IID Features

**Authors**: *Qinbin Li, Bingsheng He, Dawn Song*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23j.html](https://proceedings.mlr.press/v202/li23j.html)

**Abstract**:

Federated Learning (FL) has been a popular approach to enable collaborative learning on multiple parties without exchanging raw data. However, the model performance of FL may degrade a lot due to non-IID data. While many FL algorithms focus on non-IID labels, FL on non-IID features has largely been overlooked. Different from typical FL approaches, the paper proposes a new learning concept called ADCOL (Adversarial Collaborative Learning) for non-IID features. Instead of adopting the widely used model-averaging scheme, ADCOL conducts training in an adversarial way: the server aims to train a discriminator to distinguish the representations of the parties, while the parties aim to generate a common representation distribution. Our experiments show that ADCOL achieves better performance than state-of-the-art FL algorithms on non-IID features.

----

## [807] Near-optimal Conservative Exploration in Reinforcement Learning under Episode-wise Constraints

**Authors**: *Donghao Li, Ruiquan Huang, Cong Shen, Jing Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23k.html](https://proceedings.mlr.press/v202/li23k.html)

**Abstract**:

This paper investigates conservative exploration in reinforcement learning where the performance of the learning agent is guaranteed to be above a certain threshold throughout the learning process. It focuses on the tabular episodic Markov Decision Process (MDP) setting that has finite states and actions. With the knowledge of an existing safe baseline policy, an algorithms termed as StepMix is proposed to balance the exploitation and exploration while ensuring that the conservative constraint is never violated in each episode with high probability. StepMix features a unique design of a mixture policy that adaptively and smoothly interpolates between the baseline policy and the optimistic policy. Theoretical analysis shows that StepMix achieves near-optimal regret order as in the constraint-free setting, indicating that obeying the stringent episode-wise conservative constraint does not compromise the learning performance. Besides, a randomization based EpsMix algorithm is also proposed and shown the achieve the same performance as StepMix. The algorithm design and theoretical analysis are further extended to the setting where the baseline policy is not given a priori but must be learned from an offline dataset, and it is proved that similar conservative guarantee and regret can be achieved if the offline dataset is sufficiently large. Experiment results corroborate the theoretical analysis and demonstrate the effectiveness of the proposed conservative exploration strategies.

----

## [808] Transformers as Algorithms: Generalization and Stability in In-context Learning

**Authors**: *Yingcong Li, Muhammed Emrullah Ildiz, Dimitris Papailiopoulos, Samet Oymak*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23l.html](https://proceedings.mlr.press/v202/li23l.html)

**Abstract**:

In-context learning (ICL) is a type of prompting where a transformer model operates on a sequence of (input, output) examples and performs inference on-the-fly. In this work, we formalize in-context learning as an algorithm learning problem where a transformer model implicitly constructs a hypothesis function at inference-time. We first explore the statistical aspects of this abstraction through the lens of multitask learning: We obtain generalization bounds for ICL when the input prompt is (1) a sequence of i.i.d. (input, label) pairs or (2) a trajectory arising from a dynamical system. The crux of our analysis is relating the excess risk to the stability of the algorithm implemented by the transformer. We characterize when transformer/attention architecture provably obeys the stability condition and also provide empirical verification. For generalization on unseen tasks, we identify an inductive bias phenomenon in which the transfer learning risk is governed by the task complexity and the number of MTL tasks in a highly predictable manner. Finally, we provide numerical evaluations that (1) demonstrate transformers can indeed implement near-optimal algorithms on classical regression problems with i.i.d. and dynamic data, (2) provide insights on stability, and (3) verify our theoretical predictions.

----

## [809] Improving Hyperparameter Learning under Approximate Inference in Gaussian Process Models

**Authors**: *Rui Li, S. T. John, Arno Solin*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23m.html](https://proceedings.mlr.press/v202/li23m.html)

**Abstract**:

Approximate inference in Gaussian process (GP) models with non-conjugate likelihoods gets entangled with the learning of the model hyperparameters. We improve hyperparameter learning in GP models and focus on the interplay between variational inference (VI) and the learning target. While VI’s lower bound to the marginal likelihood is a suitable objective for inferring the approximate posterior, we show that a direct approximation of the marginal likelihood as in Expectation Propagation (EP) is a better learning objective for hyperparameter optimization. We design a hybrid training procedure to bring the best of both worlds: it leverages conjugate-computation VI for inference and uses an EP-like marginal likelihood approximation for hyperparameter learning. We compare VI, EP, Laplace approximation, and our proposed training procedure and empirically demonstrate the effectiveness of our proposal across a wide range of data sets.

----

## [810] Local Vertex Colouring Graph Neural Networks

**Authors**: *Shouheng Li, Dongwoo Kim, Qing Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23n.html](https://proceedings.mlr.press/v202/li23n.html)

**Abstract**:

In recent years, there has been a significant amount of research focused on expanding the expressivity of Graph Neural Networks (GNNs) beyond the Weisfeiler-Lehman (1-WL) framework. While many of these studies have yielded advancements in expressivity, they have frequently come at the expense of decreased efficiency or have been restricted to specific types of graphs. In this study, we investigate the expressivity of GNNs from the perspective of graph search. Specifically, we propose a new vertex colouring scheme and demonstrate that classical search algorithms can efficiently compute graph representations that extend beyond the 1-WL. We show the colouring scheme inherits useful properties from graph search that can help solve problems like graph biconnectivity. Furthermore, we show that under certain conditions, the expressivity of GNNs increases hierarchically with the radius of the search neighbourhood. To further investigate the proposed scheme, we develop a new type of GNN based on two search strategies, breadth-first search and depth-first search, highlighting the graph properties they can capture on top of 1-WL. Our code is available at https://github.com/seanli3/lvc.

----

## [811] Analysis of Error Feedback in Federated Non-Convex Optimization with Biased Compression: Fast Convergence and Partial Participation

**Authors**: *Xiaoyun Li, Ping Li*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23o.html](https://proceedings.mlr.press/v202/li23o.html)

**Abstract**:

In practical federated learning (FL) systems, the communication cost between the clients and the central server can often be a bottleneck. In this paper, we focus on biased gradient compression in non-convex FL problems. In the classical distributed learning, the method of error feedback (EF) is a common technique to remedy the downsides of biased gradient compression, but the performance of EF in FL still lacks systematic investigation. In this work, we study a compressed FL scheme with error feedback, named Fed-EF, with two variants depending on the global model optimizer. While directly applying biased compression in FL leads to poor convergence, we show that Fed-EF is able to match the convergence rate of the full-precision FL counterpart with a linear speedup w.r.t. the number of clients. Experiments verify that Fed-EF achieves the same performance as the full-precision FL approach, at the substantially reduced communication cost. Moreover, we develop a new analysis of the EF under partial participation (PP), an important scenario in FL. Under PP, the convergence rate of Fed-EF exhibits an extra slow-down factor due to a so-called “stale error compensation” effect, which is also justified in our experiments. Our results provide insights on a theoretical limitation of EF, and possible directions for improvements.

----

## [812] How Do Transformers Learn Topic Structure: Towards a Mechanistic Understanding

**Authors**: *Yuchen Li, Yuanzhi Li, Andrej Risteski*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23p.html](https://proceedings.mlr.press/v202/li23p.html)

**Abstract**:

While the successes of transformers across many domains are indisputable, accurate understanding of the learning mechanics is still largely lacking. Their capabilities have been probed on benchmarks which include a variety of structured and reasoning tasks—but mathematical understanding is lagging substantially behind. Recent lines of work have begun studying representational aspects of this question: that is, the size/depth/complexity of attention-based networks to perform certain tasks. However, there is no guarantee the learning dynamics will converge to the constructions proposed. In our paper, we provide fine-grained mechanistic understanding of how transformers learn “semantic structure”, understood as capturing co-occurrence structure of words. Precisely, we show, through a combination of mathematical analysis and experiments on Wikipedia data and synthetic data modeled by Latent Dirichlet Allocation (LDA), that the embedding layer and the self-attention layer encode the topical structure. In the former case, this manifests as higher average inner product of embeddings between same-topic words. In the latter, it manifests as higher average pairwise attention between same-topic words. The mathematical results involve several assumptions to make the analysis tractable, which we verify on data, and might be of independent interest as well.

----

## [813] BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

**Authors**: *Junnan Li, Dongxu Li, Silvio Savarese, Steven C. H. Hoi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23q.html](https://proceedings.mlr.press/v202/li23q.html)

**Abstract**:

The cost of vision-and-language pre-training has become increasingly prohibitive due to end-to-end training of large-scale models. This paper proposes BLIP-2, a generic and efficient pre-training strategy that bootstraps vision-language pre-training from off-the-shelf frozen pre-trained image encoders and frozen large language models. BLIP-2 bridges the modality gap with a lightweight Querying Transformer, which is pre-trained in two stages. The first stage bootstraps vision-language representation learning from a frozen image encoder. The second stage bootstraps vision-to-language generative learning from a frozen language model. BLIP-2 achieves state-of-the-art performance on various vision-language tasks, despite having significantly fewer trainable parameters than existing methods. For example, our model outperforms Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters. We also demonstrate the model’s emerging capabilities of zero-shot image-to-text generation that can follow natural language instructions.

----

## [814] Nearly Optimal Algorithms with Sublinear Computational Complexity for Online Kernel Regression

**Authors**: *Junfan Li, Shizhong Liao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23r.html](https://proceedings.mlr.press/v202/li23r.html)

**Abstract**:

The trade-off between regret and computational cost is a fundamental problem for online kernel regression, and previous algorithms worked on the trade-off can not keep optimal regret bounds at a sublinear computational complexity. In this paper, we propose two new algorithms, AOGD-ALD and NONS-ALD, which can keep nearly optimal regret bounds at a sublinear computational complexity, and give sufficient conditions under which our algorithms work. Both algorithms dynamically maintain a group of nearly orthogonal basis used to approximate the kernel mapping, and keep nearly optimal regret bounds by controlling the approximate error. The number of basis depends on the approximate error and the decay rate of eigenvalues of the kernel matrix. If the eigenvalues decay exponentially, then AOGD-ALD and NONS-ALD separately achieves a regret of $O(\sqrt{L(f)})$ and $O(\mathrm{d}_{\mathrm{eff}}(\mu)\ln{T})$ at a computational complexity in $O(\ln^2{T})$. If the eigenvalues decay polynomially with degree $p\geq 1$, then our algorithms keep the same regret bounds at a computational complexity in $o(T)$ in the case of $p>4$ and $p\geq 10$, respectively. $L(f)$ is the cumulative losses of $f$ and $\mathrm{d}_{\mathrm{eff}}(\mu)$ is the effective dimension of the problem. The two regret bounds are nearly optimal and are not comparable.

----

## [815] Revisiting Weighted Aggregation in Federated Learning with Neural Networks

**Authors**: *Zexi Li, Tao Lin, Xinyi Shang, Chao Wu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23s.html](https://proceedings.mlr.press/v202/li23s.html)

**Abstract**:

In federated learning (FL), weighted aggregation of local models is conducted to generate a global model, and the aggregation weights are normalized (the sum of weights is 1) and proportional to the local data sizes. In this paper, we revisit the weighted aggregation process and gain new insights into the training dynamics of FL. First, we find that the sum of weights can be smaller than 1, causing global weight shrinking effect (analogous to weight decay) and improving generalization. We explore how the optimal shrinking factor is affected by clients’ data heterogeneity and local epochs. Second, we dive into the relative aggregation weights among clients to depict the clients’ importance. We develop client coherence to study the learning dynamics and find a critical point that exists. Before entering the critical point, more coherent clients play more essential roles in generalization. Based on the above insights, we propose an effective method for Federated Learning with Learnable Aggregation Weights, named as FedLAW. Extensive experiments verify that our method can improve the generalization of the global model by a large margin on different datasets and models.

----

## [816] Distribution-dependent McDiarmid-type Inequalities for Functions of Unbounded Interaction

**Authors**: *Shaojie Li, Yong Liu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23t.html](https://proceedings.mlr.press/v202/li23t.html)

**Abstract**:

The concentration of measure inequalities serves an essential role in statistics and machine learning. This paper gives unbounded analogues of the McDiarmid-type exponential inequalities for three popular classes of distributions, namely sub-Gaussian, sub-exponential and heavy-tailed distributions. The inequalities in the sub-Gaussian and sub-exponential cases are distribution-dependent compared with the recent results, and the inequalities in the heavy-tailed case are not available in the previous works. The usefulness of the inequalities is illustrated through applications to the sample mean, U-statistics and V-statistics.

----

## [817] Optimal Convergence Rates for Agnostic Nyström Kernel Learning

**Authors**: *Jian Li, Yong Liu, Weiping Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23u.html](https://proceedings.mlr.press/v202/li23u.html)

**Abstract**:

Nyström low-rank approximation has shown great potential in processing large-scale kernel matrix and neural networks. However, there lacks a unified analysis for Nyström approximation, and the asymptotical minimax optimality for Nyström methods usually require a strict condition, assuming that the target regression lies exactly in the hypothesis space. In this paper, to tackle these problems, we provide a refined generalization analysis for Nyström approximation in the agnostic setting, where the target regression may be out of the hypothesis space. Specifically, we show Nyström approximation can still achieve the capacity-dependent optimal rates in the agnostic setting. To this end, we first prove the capacity-dependent optimal guarantees of Nyström approximation with the standard uniform sampling, which covers both loss functions and applies to some agnostic settings. Then, using data-dependent sampling, for example, leverage scores sampling, we derive the capacity-dependent optimal rates that apply to the whole range of the agnostic setting. To our best knowledge, the capacity-dependent optimality for the whole range of the agnostic setting is first achieved and novel in Nyström approximation.

----

## [818] Reconstructive Neuron Pruning for Backdoor Defense

**Authors**: *Yige Li, Xixiang Lyu, Xingjun Ma, Nodens Koren, Lingjuan Lyu, Bo Li, Yu-Gang Jiang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23v.html](https://proceedings.mlr.press/v202/li23v.html)

**Abstract**:

Deep neural networks (DNNs) have been found to be vulnerable to backdoor attacks, raising security concerns about their deployment in mission-critical applications. While existing defense methods have demonstrated promising results, it is still not clear how to effectively remove backdoor-associated neurons in backdoored DNNs. In this paper, we propose a novel defense called Reconstructive Neuron Pruning (RNP) to expose and prune backdoor neurons via an unlearning and then recovering process. Specifically, RNP first unlearns the neurons by maximizing the model’s error on a small subset of clean samples and then recovers the neurons by minimizing the model’s error on the same data. In RNP, unlearning is operated at the neuron level while recovering is operated at the filter level, forming an asymmetric reconstructive learning procedure. We show that such an asymmetric process on only a few clean samples can effectively expose and prune the backdoor neurons implanted by a wide range of attacks, achieving a new state-of-the-art defense performance. Moreover, the unlearned model at the intermediate step of our RNP can be directly used to improve other backdoor defense tasks including backdoor removal, trigger recovery, backdoor label detection, and backdoor sample detection. Code is available at https://github.com/bboylyg/RNP.

----

## [819] Meta Learning of Interface Conditions for Multi-Domain Physics-Informed Neural Networks

**Authors**: *Shibo Li, Michael Penwarden, Yiming Xu, Conor Tillinghast, Akil Narayan, Mike Kirby, Shandian Zhe*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23w.html](https://proceedings.mlr.press/v202/li23w.html)

**Abstract**:

Physics-informed neural networks (PINNs) are emerging as popular mesh-free solvers for partial differential equations (PDEs). Recent extensions decompose the domain, apply different PINNs to solve the problem in each subdomain, and stitch the subdomains at the interface. Thereby, they can further alleviate the problem complexity, reduce the computational cost, and allow parallelization. However, the performance of multi-domain PINNs is sensitive to the choice of the interface conditions. While quite a few conditions have been proposed, there is no suggestion about how to select the conditions according to specific problems. To address this gap, we propose META Learning of Interface Conditions (METALIC), a simple, efficient yet powerful approach to dynamically determine appropriate interface conditions for solving a family of parametric PDEs. Specifically, we develop two contextual multi-arm bandit (MAB) models. The first one applies to the entire training course, and online updates a Gaussian process (GP) reward that given the PDE parameters and interface conditions predicts the performance. We prove a sub-linear regret bound for both UCB and Thompson sampling, which in theory guarantees the effectiveness of our MAB. The second one partitions the training into two stages, one is the stochastic phase and the other deterministic phase; we update a GP reward for each phase to enable different condition selections at the two stages to further bolster the flexibility and performance. We have shown the advantage of METALIC on four bench-mark PDE families.

----

## [820] Deep Anomaly Detection under Labeling Budget Constraints

**Authors**: *Aodong Li, Chen Qiu, Marius Kloft, Padhraic Smyth, Stephan Mandt, Maja Rudolph*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23x.html](https://proceedings.mlr.press/v202/li23x.html)

**Abstract**:

Selecting informative data points for expert feedback can significantly improve the performance of anomaly detection (AD) in various contexts, such as medical diagnostics or fraud detection. In this paper, we determine a set of theoretical conditions under which anomaly scores generalize from labeled queries to unlabeled data. Motivated by these results, we propose a data labeling strategy with optimal data coverage under labeling budget constraints. In addition, we propose a new learning framework for semi-supervised AD. Extensive experiments on image, tabular, and video data sets show that our approach results in state-of-the-art semi-supervised AD performance under labeling budget constraints.

----

## [821] On the Initialization of Graph Neural Networks

**Authors**: *Jiahang Li, Yakun Song, Xiang Song, David Wipf*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23y.html](https://proceedings.mlr.press/v202/li23y.html)

**Abstract**:

Graph Neural Networks (GNNs) have displayed considerable promise in graph representation learning across various applications. The core learning process requires the initialization of model weight matrices within each GNN layer, which is typically accomplished via classic initialization methods such as Xavier initialization. However, these methods were originally motivated to stabilize the variance of hidden embeddings and gradients across layers of Feedforward Neural Networks (FNNs) and Convolutional Neural Networks (CNNs) to avoid vanishing gradients and maintain steady information flow. In contrast, within the GNN context classical initializations disregard the impact of the input graph structure and message passing on variance. In this paper, we analyze the variance of forward and backward propagation across GNN layers and show that the variance instability of GNN initializations comes from the combined effect of the activation function, hidden dimension, graph structure and message passing. To better account for these influence factors, we propose a new initialization method for Variance Instability Reduction within GNN Optimization (Virgo), which naturally tends to equate forward and backward variances across successive layers. We conduct comprehensive experiments on 15 datasets to show that Virgo can lead to superior model performance and more stable variance at initialization on node classification, link prediction and graph classification tasks.

----

## [822] Federated Adversarial Learning: A Framework with Convergence Analysis

**Authors**: *Xiaoxiao Li, Zhao Song, Jiaming Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23z.html](https://proceedings.mlr.press/v202/li23z.html)

**Abstract**:

Federated learning (FL) is a trending training paradigm to utilize decentralized training data. FL allows clients to update model parameters locally for several epochs, then share them to a global model for aggregation. This training paradigm with multi-local step updating before aggregation exposes unique vulnerabilities to adversarial attacks. Adversarial training is a popular and effective method to improve the robustness of networks against adversaries. In this work, we formulate a general form of federated adversarial learning (FAL) that is adapted from adversarial learning in the centralized setting. On the client side of FL training, FAL has an inner loop to generate adversarial samples for adversarial training and an outer loop to update local model parameters. On the server side, FAL aggregates local model updates and broadcast the aggregated model. We design a global robust training loss and formulate FAL training as a min-max optimization problem. Unlike the convergence analysis in classical centralized training that relies on the gradient direction, it is significantly harder to analyze the convergence in FAL for three reasons: 1) the complexity of min-max optimization, 2) model not updating in the gradient direction due to the multi-local updates on the client-side before aggregation and 3) inter-client heterogeneity. We address these challenges by using appropriate gradient approximation and coupling techniques and present the convergence analysis in the over-parameterized regime. Our main result theoretically shows that the minimum loss under our algorithm can converge to $\epsilon$ small with chosen learning rate and communication rounds. It is noteworthy that our analysis is feasible for non-IID clients.

----

## [823] How Powerful are Shallow Neural Networks with Bandlimited Random Weights?

**Authors**: *Ming Li, Sho Sonoda, Feilong Cao, Yu Guang Wang, Jiye Liang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23aa.html](https://proceedings.mlr.press/v202/li23aa.html)

**Abstract**:

We investigate the expressive power of depth-2 bandlimited random neural networks. A random net is a neural network where the hidden layer parameters are frozen with random assignment, and only the output layer parameters are trained by loss minimization. Using random weights for a hidden layer is an effective method to avoid non-convex optimization in standard gradient descent learning. It has also been adopted in recent deep learning theories. Despite the well-known fact that a neural network is a universal approximator, in this study, we mathematically show that when hidden parameters are distributed in a bounded domain, the network may not achieve zero approximation error. In particular, we derive a new nontrivial approximation error lower bound. The proof utilizes the technique of ridgelet analysis, a harmonic analysis method designed for neural networks. This method is inspired by fundamental principles in classical signal processing, specifically the idea that signals with limited bandwidth may not always be able to perfectly reconstruct the original signal. We corroborate our theoretical results with various simulation studies, and generally, two main take-home messages are offered: (i) Not any distribution for selecting random weights is feasible to build a universal approximator; (ii) A suitable assignment of random weights exists but to some degree is associated with the complexity of the target function.

----

## [824] Efficient Quantum Algorithms for Quantum Optimal Control

**Authors**: *Xiantao Li, Chunhao Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23ab.html](https://proceedings.mlr.press/v202/li23ab.html)

**Abstract**:

In this paper, we present efficient quantum algorithms that are exponentially faster than classical algorithms for solving the quantum optimal control problem. This problem involves finding the control variable that maximizes a physical quantity at time $T$, where the system is governed by a time-dependent Schrödinger equation. This type of control problem also has an intricate relation with machine learning. Our algorithms are based on a time-dependent Hamiltonian simulation method and a fast gradient-estimation algorithm. We also provide a comprehensive error analysis to quantify the total error from various steps, such as the finite-dimensional representation of the control function, the discretization of the Schrödinger equation, the numerical quadrature, and optimization. Our quantum algorithms require fault-tolerant quantum computers.

----

## [825] Low-Switching Policy Gradient with Exploration via Online Sensitivity Sampling

**Authors**: *Yunfan Li, Yiran Wang, Yu Cheng, Lin Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23ac.html](https://proceedings.mlr.press/v202/li23ac.html)

**Abstract**:

Policy optimization methods are powerful algorithms in Reinforcement Learning (RL) for their flexibility to deal with policy parameterization and ability to handle model misspecification. However, these methods usually suffer from slow convergence rates and poor sample complexity. Hence it is important to design provably sample efficient algorithms for policy optimization. Yet, recent advances for this problems have only been successful in tabular and linear setting, whose benign structures cannot be generalized to non-linearly parameterized policies. In this paper, we address this problem by leveraging recent advances in value-based algorithms, including bounded eluder-dimension and online sensitivity sampling, to design a low-switching sample-efficient policy optimization algorithm, LPO, with general non-linear function approximation. We show that, our algorithm obtains an $\varepsilon$-optimal policy with only $\widetilde{O}(\frac{\text{poly}(d)}{\varepsilon^3})$ samples, where $\varepsilon$ is the suboptimality gap and $d$ is a complexity measure of the function class approximating the policy. This drastically improves previously best-known sample bound for policy optimization algorithms, $\widetilde{O}(\frac{\text{poly}(d)}{\varepsilon^8})$. Moreover, we empirically test our theory with deep neural nets to show the benefits of the theoretical inspiration.

----

## [826] Hierarchical Diffusion for Offline Decision Making

**Authors**: *Wenhao Li, Xiangfeng Wang, Bo Jin, Hongyuan Zha*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23ad.html](https://proceedings.mlr.press/v202/li23ad.html)

**Abstract**:

Offline reinforcement learning typically introduces a hierarchical structure to solve the long-horizon problem so as to address its thorny issue of variance accumulation. Problems of deadly triad, limited data and reward sparsity, however, still remain, rendering the design of effective, hierarchical offline RL algorithms for general-purpose policy learning a formidable challenge. In this paper, we first formulate the problem of offline long-horizon decision-$\mathbf{M}$ak$\mathbf{I}$ng from the perspective of conditional generative modeling by incorporating goals into the control-as-inference graphic models. A $\mathbf{H}$ierarchical trajectory-level $\mathbf{D}$iffusion probabilistic model is then proposed with classifier-free guidance. HDMI employs a cascade framework that utilizes the reward-conditional goal diffuser for the subgoal discovery and the goal-conditional trajectory diffuser for generating the corresponding action sequence of subgoals. Planning-based subgoal extraction and transformer-based diffusion are employed to deal with the sub-optimal data pollution and long-range subgoal dependencies in the goal diffusion. Numerical experiments verify the advantages of HDMI on long-horizon decision-making compared to SOTA offline RL methods and conditional generative models.

----

## [827] Divide and Conquer Dynamic Programming: An Almost Linear Time Change Point Detection Methodology in High Dimensions

**Authors**: *Wanshan Li, Daren Wang, Alessandro Rinaldo*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23ae.html](https://proceedings.mlr.press/v202/li23ae.html)

**Abstract**:

We develop a novel, general and computationally efficient framework, called Divide and Conquer Dynamic Programming (DCDP), for localizing change points in time series data with high-dimensional features. DCDP deploys a class of greedy algorithms that are applicable to a broad variety of high-dimensional statistical models and can enjoy almost linear computational complexity. We investigate the performance of DCDP in three commonly studied change point settings in high dimensions: the mean model, the Gaussian graphical model, and the linear regression model. In all three cases, we derive non-asymptotic bounds for the accuracy of the DCDP change point estimators. We demonstrate that the DCDP procedures consistently estimate the change points with sharp, and in some cases, optimal rates while incurring significantly smaller computational costs than the best available algorithms. Our findings are supported by extensive numerical experiments on both synthetic and real data.

----

## [828] Architecture-Agnostic Masked Image Modeling - From ViT back to CNN

**Authors**: *Siyuan Li, Di Wu, Fang Wu, Zelin Zang, Stan Z. Li*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23af.html](https://proceedings.mlr.press/v202/li23af.html)

**Abstract**:

Masked image modeling, an emerging self-supervised pre-training method, has shown impressive success across numerous downstream vision tasks with Vision transformers. Its underlying idea is simple: a portion of the input image is masked out and then reconstructed via a pre-text task. However, the working principle behind MIM is not well explained, and previous studies insist that MIM primarily works for the Transformer family but is incompatible with CNNs. In this work, we observe that MIM essentially teaches the model to learn better middle-order interactions among patches for more generalized feature extraction. We then propose an Architecture-Agnostic Masked Image Modeling framework (A$^2$MIM), which is compatible with both Transformers and CNNs in a unified way. Extensive experiments on popular benchmarks show that A$^2$MIM learns better representations without explicit design and endows the backbone model with the stronger capability to transfer to various downstream tasks.

----

## [829] Learning Antidote Data to Individual Unfairness

**Authors**: *Peizhao Li, Ethan Xia, Hongfu Liu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23ag.html](https://proceedings.mlr.press/v202/li23ag.html)

**Abstract**:

Fairness is essential for machine learning systems deployed in high-stake applications. Among all fairness notions, individual fairness, deriving from a consensus that ‘similar individuals should be treated similarly,’ is a vital notion to describe fair treatment for individual cases. Previous studies typically characterize individual fairness as a prediction-invariant problem when perturbing sensitive attributes on samples, and solve it by Distributionally Robust Optimization (DRO) paradigm. However, such adversarial perturbations along a direction covering sensitive information used in DRO do not consider the inherent feature correlations or innate data constraints, therefore could mislead the model to optimize at off-manifold and unrealistic samples. In light of this drawback, in this paper, we propose to learn and generate antidote data that approximately follows the data distribution to remedy individual unfairness. These generated on-manifold antidote data can be used through a generic optimization procedure along with original training data, resulting in a pure pre-processing approach to individual unfairness, or can also fit well with the in-processing DRO paradigm. Through extensive experiments on multiple tabular datasets, we demonstrate our method resists individual unfairness at a minimal or zero cost to predictive utility compared to baselines.

----

## [830] Propensity Matters: Measuring and Enhancing Balancing for Recommendation

**Authors**: *Haoxuan Li, Yanghao Xiao, Chunyuan Zheng, Peng Wu, Peng Cui*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23ah.html](https://proceedings.mlr.press/v202/li23ah.html)

**Abstract**:

Propensity-based weighting methods have been widely studied and demonstrated competitive performance in debiased recommendations. Nevertheless, there are still many questions to be addressed. How to estimate the propensity more conducive to debiasing performance? Which metric is more reasonable to measure the quality of the learned propensities? Is it better to make the cross-entropy loss as small as possible when learning propensities? In this paper, we first discuss the potential problems of the previously widely adopted metrics for learned propensities, and propose balanced-mean-squared-error (BMSE) metric for debiased recommendations. Based on BMSE, we propose IPS-V2 and DR-V2 as the estimators of unbiased loss, and theoretically show that IPS-V2 and DR-V2 have greater propensity balancing and smaller variance without sacrificing additional bias. We further propose a co-training method for learning balanced representation and unbiased prediction. Extensive experiments are conducted on three real-world datasets including a large industrial dataset, and the results show that our approach boosts the balancing property and results in enhanced debiasing performance.

----

## [831] GraphCleaner: Detecting Mislabelled Samples in Popular Graph Learning Benchmarks

**Authors**: *Yuwen Li, Miao Xiong, Bryan Hooi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23ai.html](https://proceedings.mlr.press/v202/li23ai.html)

**Abstract**:

Label errors have been found to be prevalent in popular text, vision, and audio datasets, which heavily influence the safe development and evaluation of machine learning algorithms. Despite increasing efforts towards improving the quality of generic data types, such as images and texts, the problem of mislabel detection in graph data remains underexplored. To bridge the gap, we explore mislabelling issues in popular real-world graph datasets and propose GraphCleaner, a post-hoc method to detect and correct these mislabelled nodes in graph datasets. GraphCleaner combines the novel ideas of 1) Synthetic Mislabel Dataset Generation, which seeks to generate realistic mislabels; and 2) Neighborhood-Aware Mislabel Detection, where neighborhood dependency is exploited in both labels and base classifier predictions. Empirical evaluations on 6 datasets and 6 experimental settings demonstrate that GraphCleaner outperforms the closest baseline, with an average improvement of $0.14$ in F1 score, and $0.16$ in MCC. On real-data case studies, GraphCleaner detects real and previously unknown mislabels in popular graph benchmarks: PubMed, Cora, CiteSeer and OGB-arxiv; we find that at least 6.91% of PubMed data is mislabelled or ambiguous, and simply removing these mislabelled data can boost evaluation performance from 86.71% to 89.11%.

----

## [832] SMURF-THP: Score Matching-based UnceRtainty quantiFication for Transformer Hawkes Process

**Authors**: *Zichong Li, Yanbo Xu, Simiao Zuo, Haoming Jiang, Chao Zhang, Tuo Zhao, Hongyuan Zha*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23aj.html](https://proceedings.mlr.press/v202/li23aj.html)

**Abstract**:

Transformer Hawkes process models have shown to be successful in modeling event sequence data. However, most of the existing training methods rely on maximizing the likelihood of event sequences, which involves calculating some intractable integral. Moreover, the existing methods fail to provide uncertainty quantification for model predictions, e.g., confidence interval for the predicted event’s arrival time. To address these issues, we propose SMURF-THP, a score-based method for learning Transformer Hawkes process and quantifying prediction uncertainty. Specifically, SMURF-THP learns the score function of the event’s arrival time based on a score-matching objective that avoids the intractable computation. With such a learnt score function, we can sample arrival time of events from the predictive distribution. This naturally allows for the quantification of uncertainty by computing confidence intervals over the generated samples. We conduct extensive experiments in both event type prediction and uncertainty quantification on time of arrival. In all the experiments, SMURF-THP outperforms existing likelihood-based methods in confidence calibration while exhibiting comparable prediction accuracy.

----

## [833] Horizon-free Learning for Markov Decision Processes and Games: Stochastically Bounded Rewards and Improved Bounds

**Authors**: *Shengshi Li, Lin Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23ak.html](https://proceedings.mlr.press/v202/li23ak.html)

**Abstract**:

Horizon dependence is an important difference between reinforcement learning and other machine learning paradigms. Yet, existing results tackling the (exact) horizon dependence either assume that the reward is bounded per step, introducing unfair comparison, or assume strict total boundedness that requires the sum of rewards to be bounded almost surely – allowing only restricted noise on the reward observation. This paper addresses these limitations by introducing a new relaxation – expected boundedness on rewards, where we allow the reward to be stochastic with only boundedness on the expected sum – opening the door to study horizon-dependence with a much broader set of reward functions with noises. We establish a novel generic algorithm that achieves no-horizon dependence in terms of sample complexity for both Markov Decision Processes (MDP) and Games, via reduction to a good-conditioned auxiliary Markovian environment, in which only “important” state-action pairs are preserved. The algorithm takes only $\tilde{O}(\frac{S^2A}{\epsilon^2})$ episodes interacting with such an environment to achieve an $\epsilon$-optimal policy/strategy (with high probability), improving (zhang, 2022) (which only applies to MDPs with deterministic rewards). Here $S$ is the number of states and $A$ is the number of actions, and the bound is independent of the horizon $H$.

----

## [834] Transcendental Idealism of Planner: Evaluating Perception from Planning Perspective for Autonomous Driving

**Authors**: *Weixin Li, Xiaodong Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23al.html](https://proceedings.mlr.press/v202/li23al.html)

**Abstract**:

Evaluating the performance of perception modules in autonomous driving is one of the most critical tasks in developing the complex intelligent system. While module-level unit test metrics adopted from traditional computer vision tasks are feasible to some extent, it remains far less explored to measure the impact of perceptual noise on the driving quality of autonomous vehicles in a consistent and holistic manner. In this work, we propose a principled framework that provides a coherent and systematic understanding of the impact an error in the perception module imposes on an autonomous agent’s planning that actually controls the vehicle. Specifically, the planning process is formulated as expected utility maximisation, where all input signals from upstream modules jointly provide a world state description, and the planner strives for the optimal action by maximising the expected utility determined by both world states and actions. We show that, under practical conditions, the objective function can be represented as an inner product between the world state description and the utility function in a Hilbert space. This geometric interpretation enables a novel way to analyse the impact of noise in world state estimation on planning and leads to a universal metric for evaluating perception. The whole framework resembles the idea of transcendental idealism in the classical philosophical literature, which gives the name to our approach.

----

## [835] Learning for Edge-Weighted Online Bipartite Matching with Robustness Guarantees

**Authors**: *Pengfei Li, Jianyi Yang, Shaolei Ren*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23am.html](https://proceedings.mlr.press/v202/li23am.html)

**Abstract**:

Many problems, such as online ad display, can be formulated as online bipartite matching. The crucial challenge lies in the nature of sequentially-revealed online item information, based on which we make irreversible matching decisions at each step. While numerous expert online algorithms have been proposed with bounded worst-case competitive ratios, they may not offer satisfactory performance in average cases. On the other hand, reinforcement learning (RL) has been applied to improve the average performance, but it lacks robustness and can perform arbitrarily poorly. In this paper, we propose a novel RL-based approach to edge-weighted online bipartite matching with robustness guarantees (LOMAR), achieving both good average-case and worst-case performance. The key novelty of LOMAR is a new online switching operation which, based on a judicious condition to hedge against future uncertainties, decides whether to follow the expert’s decision or the RL decision for each online item. We prove that for any $\rho\in[0,1]$, LOMAR is $\rho$-competitive against any given expert online algorithm. To improve the average performance, we train the RL policy by explicitly considering the online switching operation. Finally, we run empirical experiments to demonstrate the advantages of LOMAR compared to existing baselines.

----

## [836] FedVS: Straggler-Resilient and Privacy-Preserving Vertical Federated Learning for Split Models

**Authors**: *Songze Li, Duanyi Yao, Jin Liu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23an.html](https://proceedings.mlr.press/v202/li23an.html)

**Abstract**:

In a vertical federated learning (VFL) system consisting of a central server and many distributed clients, the training data are vertically partitioned such that different features are privately stored on different clients. The problem of split VFL is to train a model split between the server and the clients. This paper aims to address two major challenges in split VFL: 1) performance degradation due to straggling clients during training; and 2) data and model privacy leakage from clients’ uploaded data embeddings. We propose FedVS to simultaneously address these two challenges. The key idea of FedVS is to design secret sharing schemes for the local data and models, such that information-theoretical privacy against colluding clients and curious server is guaranteed, and the aggregation of all clients’ embeddings is reconstructed losslessly, via decrypting computation shares from the non-straggling clients. Extensive experiments on various types of VFL datasets (including tabular, CV, and multi-view) demonstrate the universal advantages of FedVS in straggler mitigation and privacy protection over baseline protocols.

----

## [837] Achieving Hierarchy-Free Approximation for Bilevel Programs with Equilibrium Constraints

**Authors**: *Jiayang Li, Jing Yu, Boyi Liu, Yu Nie, Zhaoran Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23ao.html](https://proceedings.mlr.press/v202/li23ao.html)

**Abstract**:

In this paper, we develop an approximation scheme for solving bilevel programs with equilibrium constraints, which are generally difficult to solve. Among other things, calculating the first-order derivative in such a problem requires differentiation across the hierarchy, which is computationally intensive, if not prohibitive. To bypass the hierarchy, we propose to bound such bilevel programs, equivalent to multiple-followers Stackelberg games, with two new hierarchy-free problems: a $T$-step Cournot game and a $T$-step monopoly model. Since they are standard equilibrium or optimization problems, both can be efficiently solved via first-order methods. Importantly, we show that the bounds provided by these problems — the upper bound by the $T$-step Cournot game and the lower bound by the $T$-step monopoly model — can be made arbitrarily tight by increasing the step parameter $T$ for a wide range of problems. We prove that a small $T$ usually suffices under appropriate conditions to reach an approximation acceptable for most practical purposes. Eventually, the analytical insights are highlighted through numerical examples.

----

## [838] LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation

**Authors**: *Yixiao Li, Yifan Yu, Qingru Zhang, Chen Liang, Pengcheng He, Weizhu Chen, Tuo Zhao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23ap.html](https://proceedings.mlr.press/v202/li23ap.html)

**Abstract**:

Transformer models have achieved remarkable results in various natural language tasks, but they are often prohibitively large, requiring massive memories and computational resources. To re- duce the size and complexity of these models, we propose LoSparse (Low-Rank and Sparse ap- proximation), a novel model compression tech- nique that approximates a weight matrix by the sum of a low-rank matrix and a sparse matrix. Our method combines the advantages of both low- rank approximations and pruning, while avoid- ing their limitations. Low-rank approximation compresses the coherent and expressive parts in neurons, while pruning removes the incoherent and non-expressive parts in neurons. Pruning enhances the diversity of low-rank approxima- tions, and low-rank approximation prevents prun- ing from losing too many expressive neurons. We evaluate our method on natural language under- standing, question answering, and natural lan- guage generation tasks. We show that it signif- icantly outperforms existing compression meth- ods. Our code is publicly available at https: //github.com/yxli2123/LoSparse

----

## [839] Nesterov Meets Optimism: Rate-Optimal Separable Minimax Optimization

**Authors**: *Chris Junchi Li, Huizhuo Yuan, Gauthier Gidel, Quanquan Gu, Michael I. Jordan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23aq.html](https://proceedings.mlr.press/v202/li23aq.html)

**Abstract**:

We propose a new first-order optimization algorithm — AcceleratedGradient-OptimisticGradient (AG-OG) Descent Ascent—for separable convex-concave minimax optimization. The main idea of our algorithm is to carefully leverage the structure of the minimax problem, performing Nesterov acceleration on the individual component and optimistic gradient on the coupling component. Equipped with proper restarting, we show that AG-OG achieves the optimal convergence rate (up to a constant) for a variety of settings, including bilinearly coupled strongly convex-strongly concave minimax optimization (bi-SC-SC), bilinearly coupled convex-strongly concave minimax optimization (bi-C-SC), and bilinear games. We also extend our algorithm to the stochastic setting and achieve the optimal convergence rate in both bi-SC-SC and bi-C-SC settings. AG-OG is the first single-call algorithm with optimal convergence rates in both deterministic and stochastic settings for bilinearly coupled minimax optimization problems.

----

## [840] Alternating Local Enumeration (TnALE): Solving Tensor Network Structure Search with Fewer Evaluations

**Authors**: *Chao Li, Junhua Zeng, Chunmei Li, Cesar F. Caiafa, Qibin Zhao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23ar.html](https://proceedings.mlr.press/v202/li23ar.html)

**Abstract**:

Tensor network (TN) is a powerful framework in machine learning, but selecting a good TN model, known as TN structure search (TN-SS), is a challenging and computationally intensive task. The recent approach TNLS (Li et al., 2022) showed promising results for this task. However, its computational efficiency is still unaffordable, requiring too many evaluations of the objective function. We propose TnALE, a surprisingly simple algorithm that updates each structure-related variable alternately by local enumeration, greatly reducing the number of evaluations compared to TNLS. We theoretically investigate the descent steps for TNLS and TnALE, proving that both the algorithms can achieve linear convergence up to a constant if a sufficient reduction of the objective is reached in each neighborhood. We further compare the evaluation efficiency of TNLS and TnALE, revealing that $\Omega(2^K)$ evaluations are typically required in TNLS for reaching the objective reduction, while ideally $O(KR)$ evaluations are sufficient in TnALE, where $K$ denotes the dimension of search space and $R$ reflects the “low-rankness” of the neighborhood. Experimental results verify that TnALE can find practically good TN structures with vastly fewer evaluations than the state-of-the-art algorithms.

----

## [841] Understanding the Complexity Gains of Single-Task RL with a Curriculum

**Authors**: *Qiyang Li, Yuexiang Zhai, Yi Ma, Sergey Levine*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23as.html](https://proceedings.mlr.press/v202/li23as.html)

**Abstract**:

Reinforcement learning (RL) problems can be challenging without well-shaped rewards. Prior work on provably efficient RL methods generally proposes to address this issue with dedicated exploration strategies. However, another way to tackle this challenge is to reformulate it as a multi-task RL problem, where the task space contains not only the challenging task of interest but also easier tasks that implicitly function as a curriculum. Such a reformulation opens up the possibility of running existing multi-task RL methods as a more efficient alternative to solving a single challenging task from scratch. In this work, we provide a theoretical framework that reformulates a single-task RL problem as a multi-task RL problem defined by a curriculum. Under mild regularity conditions on the curriculum, we show that sequentially solving each task in the multi-task RL problem is more computationally efficient than solving the original single-task problem, without any explicit exploration bonuses or other exploration strategies. We also show that our theoretical insights can be translated into an effective practical learning algorithm that can accelerate curriculum learning on simulated robotic tasks.

----

## [842] Does a Neural Network Really Encode Symbolic Concepts?

**Authors**: *Mingjie Li, Quanshi Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23at.html](https://proceedings.mlr.press/v202/li23at.html)

**Abstract**:

Recently, a series of studies have tried to extract interactions between input variables modeled by a DNN and define such interactions as concepts encoded by the DNN. However, strictly speaking, there still lacks a solid guarantee whether such interactions indeed represent meaningful concepts. Therefore, in this paper, we examine the trustworthiness of interaction concepts from four perspectives. Extensive empirical studies have verified that a well-trained DNN usually encodes sparse, transferable, and discriminative concepts, which is partially aligned with human intuition. The code is released at https://github.com/sjtu-xai-lab/interaction-concept.

----

## [843] Cooperative Open-ended Learning Framework for Zero-Shot Coordination

**Authors**: *Yang Li, Shao Zhang, Jichen Sun, Yali Du, Ying Wen, Xinbing Wang, Wei Pan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23au.html](https://proceedings.mlr.press/v202/li23au.html)

**Abstract**:

Zero-shot coordination in cooperative artificial intelligence (AI) remains a significant challenge, which means effectively coordinating with a wide range of unseen partners. Previous algorithms have attempted to address this challenge by optimizing fixed objectives within a population to improve strategy or behaviour diversity. However, these approaches can result in a loss of learning and an inability to cooperate with certain strategies within the population, known as cooperative incompatibility. To address this issue, we propose the Cooperative Open-ended LEarning (COLE) framework, which constructs open-ended objectives in cooperative games with two players from the perspective of graph theory to assess and identify the cooperative ability of each strategy. We further specify the framework and propose a practical algorithm that leverages knowledge from game theory and graph theory. Furthermore, an analysis of the learning process of the algorithm shows that it can efficiently overcome cooperative incompatibility. The experimental results in the Overcooked game environment demonstrate that our method outperforms current state-of-the-art methods when coordinating with different-level partners. Our demo is available at https://sites.google.com/view/cole-2023.

----

## [844] Offline Reinforcement Learning with Closed-Form Policy Improvement Operators

**Authors**: *Jiachen Li, Edwin Zhang, Ming Yin, Qinxun Bai, Yu-Xiang Wang, William Yang Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23av.html](https://proceedings.mlr.press/v202/li23av.html)

**Abstract**:

Behavior constrained policy optimization has been demonstrated to be a successful paradigm for tackling Offline Reinforcement Learning. By exploiting historical transitions, a policy is trained to maximize a learned value function while constrained by the behavior policy to avoid a significant distributional shift. In this paper, we propose our closed-form policy improvement operators. We make a novel observation that the behavior constraint naturally motivates the use of first-order Taylor approximation, leading to a linear approximation of the policy objective. Additionally, as practical datasets are usually collected by heterogeneous policies, we model the behavior policies as a Gaussian Mixture and overcome the induced optimization difficulties by leveraging the LogSumExp’s lower bound and Jensen’s Inequality, giving rise to a closed-form policy improvement operator. We instantiate both one-step and iterative offline RL algorithms with our novel policy improvement operators and empirically demonstrate their effectiveness over state-of-the-art algorithms on the standard D4RL benchmark. Our code is available at https://cfpi-icml23.github.io/.

----

## [845] Optimal Arms Identification with Knapsacks

**Authors**: *Shaoang Li, Lan Zhang, Yingqi Yu, Xiangyang Li*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23aw.html](https://proceedings.mlr.press/v202/li23aw.html)

**Abstract**:

Best Arm Identification (BAI) is a general online pure exploration framework to identify optimal decisions among candidates via sequential interactions. We pioneer the Optimal Arms identification with Knapsacks (OAK) problem, which extends the BAI setting to model the resource consumption. We present a novel OAK algorithm and prove the upper bound of our algorithm by exploring the relationship between selecting optimal actions and the structure of the feasible region. Our analysis introduces a new complexity measure, which builds a bridge between the OAK setting and bandits with knapsacks problem. We establish the instance-dependent lower bound for the OAK problem based on the new complexity measure. Our results show that the proposed algorithm achieves a near-optimal probability bound for the OAK problem. In addition, we demonstrate that our algorithm recovers or improves the state-of-the-art upper bounds for several special cases, including the simple OAK setting and some classical pure exploration problems.

----

## [846] Internally Rewarded Reinforcement Learning

**Authors**: *Mengdi Li, Xufeng Zhao, Jae Hee Lee, Cornelius Weber, Stefan Wermter*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23ax.html](https://proceedings.mlr.press/v202/li23ax.html)

**Abstract**:

We study a class of reinforcement learning problems where the reward signals for policy learning are generated by a discriminator that is dependent on and jointly optimized with the policy. This interdependence between the policy and the discriminator leads to an unstable learning process because reward signals from an immature discriminator are noisy and impede policy learning, and conversely, an under-optimized policy impedes discriminator learning. We call this learning setting $\textit{Internally Rewarded Reinforcement Learning}$ (IRRL) as the reward is not provided directly by the environment but $\textit{internally}$ by the discriminator. In this paper, we formally formulate IRRL and present a class of problems that belong to IRRL. We theoretically derive and empirically analyze the effect of the reward function in IRRL and based on these analyses propose the clipped linear reward function. Experimental results show that the proposed reward function can consistently stabilize the training process by reducing the impact of reward noise, which leads to faster convergence and higher performance compared with baselines in diverse tasks.

----

## [847] Trustworthy Policy Learning under the Counterfactual No-Harm Criterion

**Authors**: *Haoxuan Li, Chunyuan Zheng, Yixiao Cao, Zhi Geng, Yue Liu, Peng Wu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23ay.html](https://proceedings.mlr.press/v202/li23ay.html)

**Abstract**:

Trustworthy policy learning has significant importance in making reliable and harmless treatment decisions for individuals. Previous policy learning approaches aim at the well-being of subgroups by maximizing the utility function (e.g., conditional average causal effects, post-view click-through&conversion rate in recommendations), however, individual-level counterfactual no-harm criterion has rarely been discussed. In this paper, we first formalize the counterfactual no-harm criterion for policy learning from a principal stratification perspective. Next, we propose a novel upper bound for the fraction negatively affected by the policy and show the consistency and asymptotic normality of the estimator. Based on the estimators for the policy utility and harm upper bounds, we further propose a policy learning approach that satisfies the counterfactual no-harm criterion, and prove its consistency to the optimal policy reward for parametric and non-parametric policy classes, respectively. Extensive experiments are conducted to show the effectiveness of the proposed policy learning approach for satisfying the counterfactual no-harm criterion.

----

## [848] Structured Cooperative Learning with Graphical Model Priors

**Authors**: *Shuangtong Li, Tianyi Zhou, Xinmei Tian, Dacheng Tao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/li23az.html](https://proceedings.mlr.press/v202/li23az.html)

**Abstract**:

We study how to train personalized models for different tasks on decentralized devices with limited local data. We propose "Structured Cooperative Learning (SCooL)", in which a cooperation graph across devices is generated by a graphical model prior to automatically coordinate mutual learning between devices. By choosing graphical models enforcing different structures, we can derive a rich class of existing and novel decentralized learning algorithms via variational inference. In particular, we show three instantiations of SCooL that adopt Dirac distribution, stochastic block model (SBM), and attention as the prior generating cooperation graphs. These EM-type algorithms alternate between updating the cooperation graph and cooperative learning of local models. They can automatically capture the cross-task correlations among devices by only monitoring their model updating in order to optimize the cooperation graph. We evaluate SCooL and compare it with existing decentralized learning methods on an extensive set of benchmarks, on which SCooL always achieves the highest accuracy of personalized models and significantly outperforms other baselines on communication efficiency. Our code is available at https://github.com/ShuangtongLi/SCooL.

----

## [849] Low Complexity Homeomorphic Projection to Ensure Neural-Network Solution Feasibility for Optimization over (Non-)Convex Set

**Authors**: *Enming Liang, Minghua Chen, Steven H. Low*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liang23a.html](https://proceedings.mlr.press/v202/liang23a.html)

**Abstract**:

There has been growing interest in employing neural network (NN) to directly solve constrained optimization problems with low run-time complexity. However, it is non-trivial to ensure NN solutions strictly satisfying problem constraints due to inherent NN prediction errors. Existing feasibility-ensuring methods either are computationally expensive or lack performance guarantee. In this paper, we propose homeomorphic projection as a low-complexity scheme to guarantee NN solution feasibility for optimization over a general set homeomorphic to a unit ball, covering all compact convex sets and certain classes of nonconvex sets. The idea is to (i) learn a minimum distortion homeomorphic mapping between the constraint set and a unit ball using an invertible NN (INN), and then (ii) perform a simple bisection operation concerning the unit ball so that the INN-mapped final solution is feasible with respect to the constraint set with minor distortion-induced optimality loss. We prove the feasibility guarantee and bound the optimality loss under mild conditions. Simulation results, including those for non-convex AC-OPF problems in power grid operation, show that homeomorphic projection outperforms existing methods in solution feasibility and run-time complexity, while achieving similar optimality loss.

----

## [850] Consistency of Multiple Kernel Clustering

**Authors**: *Weixuan Liang, Xinwang Liu, Yong Liu, Chuan Ma, Yunping Zhao, Zhe Liu, En Zhu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liang23b.html](https://proceedings.mlr.press/v202/liang23b.html)

**Abstract**:

Consistency plays an important role in learning theory. However, in multiple kernel clustering (MKC), the consistency of kernel weights has not been sufficiently investigated. In this work, we fill this gap with a non-asymptotic analysis on the consistency of kernel weights of a novel method termed SimpleMKKM. Under the assumptions of the eigenvalue gap, we give an infinity norm bound as $\widetilde{\mathcal{O}}(k/\sqrt{n})$, where $k$ is the number of clusters and $n$ is the number of samples. On this basis, we establish an upper bound for the excess clustering risk. Moreover, we study the difference of the kernel weights learned from $n$ samples and $r$ points sampled without replacement, and derive its upper bound as $\widetilde{\mathcal{O}}(k\cdot\sqrt{1/r-1/n})$. Based on the above results, we propose a novel strategy with Nyström method to enable SimpleMKKM to handle large-scale datasets with a theoretical learning guarantee. Finally, extensive experiments are conducted to verify the theoretical results and the effectiveness of the proposed large-scale strategy.

----

## [851] A Distribution Optimization Framework for Confidence Bounds of Risk Measures

**Authors**: *Hao Liang, Zhi-Quan Luo*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liang23c.html](https://proceedings.mlr.press/v202/liang23c.html)

**Abstract**:

We present a distribution optimization framework that significantly improves confidence bounds for various risk measures compared to previous methods. Our framework encompasses popular risk measures such as the entropic risk measure, conditional value at risk (CVaR), spectral risk measure, distortion risk measure, equivalent certainty, and rank-dependent expected utility, which are well established in risk-sensitive decision-making literature. To achieve this, we introduce two estimation schemes based on concentration bounds derived from the empirical distribution, specifically using either the Wasserstein distance or the supremum distance. Unlike traditional approaches that add or subtract a confidence radius from the empirical risk measures, our proposed schemes evaluate a specific transformation of the empirical distribution based on the distance. Consequently, our confidence bounds consistently yield tighter results compared to previous methods. We further verify the efficacy of the proposed framework by providing tighter problem-dependent regret bound for the CVaR bandit.

----

## [852] Accuracy on the Curve: On the Nonlinear Correlation of ML Performance Between Data Subpopulations

**Authors**: *Weixin Liang, Yining Mao, Yongchan Kwon, Xinyu Yang, James Zou*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liang23d.html](https://proceedings.mlr.press/v202/liang23d.html)

**Abstract**:

Understanding the performance of machine learning (ML) models across diverse data distributions is critically important for reliable applications. Despite recent empirical studies positing a near-perfect linear correlation between in-distribution (ID) and out-of-distribution (OOD) accuracies, we empirically demonstrate that this correlation is more nuanced under subpopulation shifts. Through rigorous experimentation and analysis across a variety of datasets, models, and training epochs, we demonstrate that OOD performance often has a nonlinear correlation with ID performance in subpopulation shifts. Our findings, which contrast previous studies that have posited a linear correlation in model performance during distribution shifts, reveal a "moon shape" correlation (parabolic uptrend curve) between the test performance on the majority subpopulation and the minority subpopulation. This non-trivial nonlinear correlation holds across model architectures, hyperparameters, training durations, and the imbalance between subpopulations. Furthermore, we found that the nonlinearity of this "moon shape" is causally influenced by the degree of spurious correlations in the training data. Our controlled experiments show that stronger spurious correlation in the training data creates more nonlinear performance correlation. We provide complementary experimental and theoretical analyses for this phenomenon, and discuss its implications for ML reliability and fairness. Our work highlights the importance of understanding the nonlinear effects of model improvement on performance in different subpopulations, and has the potential to inform the development of more equitable and responsible machine learning models.

----

## [853] AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners

**Authors**: *Zhixuan Liang, Yao Mu, Mingyu Ding, Fei Ni, Masayoshi Tomizuka, Ping Luo*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liang23e.html](https://proceedings.mlr.press/v202/liang23e.html)

**Abstract**:

Diffusion models have demonstrated their powerful generative capability in many tasks, with great potential to serve as a paradigm for offline reinforcement learning. However, the quality of the diffusion model is limited by the insufficient diversity of training data, which hinders the performance of planning and the generalizability to new tasks. This paper introduces AdaptDiffuser, an evolutionary planning method with diffusion that can self-evolve to improve the diffusion model hence a better planner, not only for seen tasks but can also adapt to unseen tasks. AdaptDiffuser enables the generation of rich synthetic expert data for goal-conditioned tasks using guidance from reward gradients. It then selects high-quality data via a discriminator to finetune the diffusion model, which improves the generalization ability to unseen tasks. Empirical experiments on two benchmark environments and two carefully designed unseen tasks in KUKA industrial robot arm and Maze2D environments demonstrate the effectiveness of AdaptDiffuser. For example, AdaptDiffuser not only outperforms the previous art Diffuser by 20.8% on Maze2D and 7.5% on MuJoCo locomotion, but also adapts better to new tasks, e.g., KUKA pick-and-place, by 27.9% without requiring additional expert data. More visualization results and demo videos could be found on our project page.

----

## [854] Learning Compiler Pass Orders using Coreset and Normalized Value Prediction

**Authors**: *Youwei Liang, Kevin Stone, Ali Shameli, Chris Cummins, Mostafa Elhoushi, Jiadong Guo, Benoit Steiner, Xiaomeng Yang, Pengtao Xie, Hugh James Leather, Yuandong Tian*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liang23f.html](https://proceedings.mlr.press/v202/liang23f.html)

**Abstract**:

Finding the optimal pass sequence of compilation can lead to a significant reduction in program size. Prior works on compilation pass ordering have two major drawbacks. They either require an excessive budget (in terms of the number of compilation passes) at compile time or fail to generalize to unseen programs. In this work, instead of predicting passes sequentially, we directly learn a policy on the pass sequence space, which outperforms the default -Oz flag by an average of 4.5% over a large collection (4683) of unseen code repositories from diverse domains across 14 datasets. To achieve this, we first identify a small set (termed coreset) of pass sequences that generally optimize the size of most programs. Then, a policy is learned to pick the optimal sequences by predicting the normalized values of the pass sequences in the coreset. Our results demonstrate that existing human-designed compiler passes can be improved with a simple yet effective technique that leverages pass sequence space which contains dense rewards, while approaches operating on the individual pass space may suffer from issues of sparse reward, and do not generalize well to held-out programs from different domains. Website: https://rlcompopt.github.io.

----

## [855] Adversarial Example Does Good: Preventing Painting Imitation from Diffusion Models via Adversarial Examples

**Authors**: *Chumeng Liang, Xiaoyu Wu, Yang Hua, Jiaru Zhang, Yiming Xue, Tao Song, Zhengui Xue, Ruhui Ma, Haibing Guan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liang23g.html](https://proceedings.mlr.press/v202/liang23g.html)

**Abstract**:

Recently, Diffusion Models (DMs) boost a wave in AI for Art yet raise new copyright concerns, where infringers benefit from using unauthorized paintings to train DMs and generate novel paintings in a similar style. To address these emerging copyright violations, in this paper, we are the first to explore and propose to utilize adversarial examples for DMs to protect human-created artworks. Specifically, we first build a theoretical framework to define and evaluate the adversarial examples for DMs. Then, based on this framework, we design a novel algorithm to generate these adversarial examples, named AdvDM, which exploits a Monte-Carlo estimation of adversarial examples for DMs by optimizing upon different latent variables sampled from the reverse process of DMs. Extensive experiments show that the generated adversarial examples can effectively hinder DMs from extracting their features. Therefore, our method can be a powerful tool for human artists to protect their copyright against infringers equipped with DM-based AI-for-Art applications. The code of our method is available on GitHub: https://github.com/mist-project/mist.git.

----

## [856] CLUSTSEG: Clustering for Universal Segmentation

**Authors**: *James Chenhao Liang, Tianfei Zhou, Dongfang Liu, Wenguan Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liang23h.html](https://proceedings.mlr.press/v202/liang23h.html)

**Abstract**:

We present CLUSTSEG, a general, transformer-based framework that tackles different image segmentation tasks ($i.e.,$ superpixel, semantic, instance, and panoptic) through a unified, neural clustering scheme. Regarding queries as cluster centers, CLUSTSEG is innovative in two aspects: 1) cluster centers are initialized in heterogeneous ways so as to pointedly address task-specific demands ($e.g.,$ instance- or category-level distinctiveness), yet without modifying the architecture; and 2) pixel-cluster assignment, formalized in a cross-attention fashion, is alternated with cluster center update, yet without learning additional parameters. These innovations closely link CLUSTSEG to EM clustering and make it a transparent and powerful framework that yields superior results across the above segmentation tasks.

----

## [857] Conformal Inference is (almost) Free for Neural Networks Trained with Early Stopping

**Authors**: *Ziyi Liang, Yanfei Zhou, Matteo Sesia*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liang23i.html](https://proceedings.mlr.press/v202/liang23i.html)

**Abstract**:

Early stopping based on hold-out data is a popular regularization technique designed to mitigate overfitting and increase the predictive accuracy of neural networks. Models trained with early stopping often provide relatively accurate predictions, but they generally still lack precise statistical guarantees unless they are further calibrated using independent hold-out data. This paper addresses the above limitation with conformalized early stopping: a novel method that combines early stopping with conformal calibration while efficiently recycling the same hold-out data. This leads to models that are both accurate and able to provide exact predictive inferences without multiple data splits nor overly conservative adjustments. Practical implementations are developed for different learning tasks—outlier detection, multi-class classification, regression—and their competitive performance is demonstrated on real data.

----

## [858] Less is More: Task-aware Layer-wise Distillation for Language Model Compression

**Authors**: *Chen Liang, Simiao Zuo, Qingru Zhang, Pengcheng He, Weizhu Chen, Tuo Zhao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liang23j.html](https://proceedings.mlr.press/v202/liang23j.html)

**Abstract**:

Layer-wise distillation is a powerful tool to compress large models (i.e. teacher models) into small ones (i.e., student models). The student distills knowledge from the teacher by mimicking the hidden representations of the teacher at every intermediate layer. However, layer-wise distillation is difficult. Since the student has a smaller model capacity than the teacher, it is often under-fitted. Furthermore, the hidden representations of the teacher contain redundant information that the student does not necessarily need for the target task’s learning. To address these challenges, we propose a novel Task-aware layEr-wise Distillation (TED). TED designs task-aware filters to align the hidden representations of the student and the teacher at each layer. The filters select the knowledge that is useful for the target task from the hidden representations. As such, TED reduces the knowledge gap between the two models and helps the student to fit better on the target task. We evaluate TED in two scenarios: continual pre-training and fine-tuning. TED demonstrates significant and consistent improvements over existing distillation methods in both scenarios. Code is available at https://github.com/cliang1453/task-aware-distillation.

----

## [859] Statistical Inference and A/B Testing for First-Price Pacing Equilibria

**Authors**: *Luofeng Liao, Christian Kroer*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liao23a.html](https://proceedings.mlr.press/v202/liao23a.html)

**Abstract**:

We initiate the study of statistical inference and A/B testing for first-price pacing equilibria (FPPE). The FPPE model captures the dynamics resulting from large-scale first-price auction markets where buyers use pacing-based budget management. Such markets arise in the context of internet advertising, where budgets are prevalent. We propose a statistical framework for the FPPE model, in which a limit FPPE with a continuum of items models the long-run steady-state behavior of the auction platform, and an observable FPPE consisting of a finite number of items provides the data to estimate primitives of the limit FPPE, such as revenue, Nash social welfare (a fair metric of efficiency), and other parameters of interest. We develop central limit theorems and asymptotically valid confidence intervals. Furthermore, we establish the asymptotic local minimax optimality of our estimators. We then show that the theory can be used for conducting statistically valid A/B testing on auction platforms. Numerical simulations verify our central limit theorems, and empirical coverage rates for our confidence intervals agree with our theory.

----

## [860] Supervised Metric Learning to Rank for Retrieval via Contextual Similarity Optimization

**Authors**: *Christopher Liao, Theodoros Tsiligkaridis, Brian Kulis*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liao23b.html](https://proceedings.mlr.press/v202/liao23b.html)

**Abstract**:

There is extensive interest in metric learning methods for image retrieval. Many metric learning loss functions focus on learning a correct ranking of training samples, but strongly overfit semantically inconsistent labels and require a large amount of data. To address these shortcomings, we propose a new metric learning method, called contextual loss, which optimizes contextual similarity in addition to cosine similarity. Our contextual loss implicitly enforces semantic consistency among neighbors while converging to the correct ranking. We empirically show that the proposed loss is more robust to label noise, and is less prone to overfitting even when a large portion of train data is withheld. Extensive experiments demonstrate that our method achieves a new state-of-the-art across four image retrieval benchmarks and multiple different evaluation settings. Code is available at: https://github.com/Chris210634/metric-learning-using-contextual-similarity

----

## [861] Revisiting Domain Randomization via Relaxed State-Adversarial Policy Optimization

**Authors**: *Yun-Hsuan Lien, Ping-Chun Hsieh, Yu-Shuen Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lien23a.html](https://proceedings.mlr.press/v202/lien23a.html)

**Abstract**:

Domain randomization (DR) is widely used in reinforcement learning (RL) to bridge the gap between simulation and reality by maximizing its average returns under the perturbation of environmental parameters. However, even the most complex simulators cannot capture all details in reality due to finite domain parameters and simplified physical models. Additionally, the existing methods often assume that the distribution of domain parameters belongs to a specific family of probability functions, such as normal distributions, which may not be correct. To overcome these limitations, we propose a new approach to DR by rethinking it from the perspective of adversarial state perturbation, without the need for reconfiguring the simulator or relying on prior knowledge about the environment. We also address the issue of over-conservatism that can occur when perturbing agents to the worst states during training by introducing a Relaxed State-Adversarial Algorithm that simultaneously maximizes the average-case and worst-case returns. We evaluate our method by comparing it to state-of-the-art methods, providing experimental results and theoretical proofs to verify its effectiveness. Our source code and appendix are available at https://github.com/sophialien/RAPPO.

----

## [862] Variational Open-Domain Question Answering

**Authors**: *Valentin Liévin, Andreas Geert Motzfeldt, Ida Riis Jensen, Ole Winther*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lievin23a.html](https://proceedings.mlr.press/v202/lievin23a.html)

**Abstract**:

Retrieval-augmented models have proven to be effective in natural language processing tasks, yet there remains a lack of research on their optimization using variational inference. We introduce the Variational Open-Domain (VOD) framework for end-to-end training and evaluation of retrieval-augmented models, focusing on open-domain question answering and language modelling. The VOD objective, a self-normalized estimate of the Rényi variational bound, approximates the task marginal likelihood and is evaluated under samples drawn from an auxiliary sampling distribution (cached retriever and/or approximate posterior). It remains tractable, even for retriever distributions defined on large corpora. We demonstrate VOD’s versatility by training reader-retriever BERT-sized models on multiple-choice medical exam questions. On the MedMCQA dataset, we outperform the domain-tuned Med-PaLM by +5.3% despite using 2.500$\times$ fewer parameters. Our retrieval-augmented BioLinkBERT model scored 62.9% on the MedMCQA and 55.0% on the MedQA-USMLE. Last, we show the effectiveness of our learned retriever component in the context of medical semantic search.

----

## [863] Generating Novel, Designable, and Diverse Protein Structures by Equivariantly Diffusing Oriented Residue Clouds

**Authors**: *Yeqing Lin, Mohammed AlQuraishi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23a.html](https://proceedings.mlr.press/v202/lin23a.html)

**Abstract**:

Proteins power a vast array of functional processes in living cells. The capability to create new proteins with designed structures and functions would thus enable the engineering of cellular behavior and development of protein-based therapeutics and materials. Structure-based protein design aims to find structures that are designable (can be realized by a protein sequence), novel (have dissimilar geometry from natural proteins), and diverse (span a wide range of geometries). While advances in protein structure prediction have made it possible to predict structures of novel protein sequences, the combinatorially large space of sequences and structures limits the practicality of search-based methods. Generative models provide a compelling alternative, by implicitly learning the low-dimensional structure of complex data distributions. Here, we leverage recent advances in denoising diffusion probabilistic models and equivariant neural networks to develop Genie, a generative model of protein structures that performs discrete-time diffusion using a cloud of oriented reference frames in 3D space. Through in silico evaluations, we demonstrate that Genie generates protein backbones that are more designable, novel, and diverse than existing models. This indicates that Genie is capturing key aspects of the distribution of protein structure space and facilitates protein design with high success rates. Code for generating new proteins and training new versions of Genie is available at https://github.com/aqlaboratory/genie.

----

## [864] Hyperbolic Diffusion Embedding and Distance for Hierarchical Representation Learning

**Authors**: *Ya-Wei Eileen Lin, Ronald R. Coifman, Gal Mishne, Ronen Talmon*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23b.html](https://proceedings.mlr.press/v202/lin23b.html)

**Abstract**:

Finding meaningful representations and distances of hierarchical data is important in many fields. This paper presents a new method for hierarchical data embedding and distance. Our method relies on combining diffusion geometry, a central approach to manifold learning, and hyperbolic geometry. Specifically, using diffusion geometry, we build multi-scale densities on the data, aimed to reveal their hierarchical structure, and then embed them into a product of hyperbolic spaces. We show theoretically that our embedding and distance recover the underlying hierarchical structure. In addition, we demonstrate the efficacy of the proposed method and its advantages compared to existing methods on graph embedding benchmarks and hierarchical datasets.

----

## [865] Simplifying Momentum-based Positive-definite Submanifold Optimization with Applications to Deep Learning

**Authors**: *Wu Lin, Valentin Duruisseaux, Melvin Leok, Frank Nielsen, Mohammad Emtiyaz Khan, Mark Schmidt*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23c.html](https://proceedings.mlr.press/v202/lin23c.html)

**Abstract**:

Riemannian submanifold optimization with momentum is computationally challenging because, to ensure that the iterates remain on the submanifold, we often need to solve difficult differential equations. Here, we simplify such difficulties for a class of structured symmetric positive-definite matrices with the affine-invariant metric. We do so by proposing a generalized version of the Riemannian normal coordinates that dynamically orthonormalizes the metric and locally converts the problem into an unconstrained problem in the Euclidean space. We use our approach to simplify existing approaches for structured covariances and develop matrix-inverse-free $2^\text{nd}$-order optimizers for deep learning in low precision settings.

----

## [866] Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Paragraph Denoise

**Authors**: *Zhenghao Lin, Yeyun Gong, Yelong Shen, Tong Wu, Zhihao Fan, Chen Lin, Nan Duan, Weizhu Chen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23d.html](https://proceedings.mlr.press/v202/lin23d.html)

**Abstract**:

In this paper, we introduce a novel dIffusion language modEl pre-training framework for text generation, which we call GENIE. GENIE is a large-scale pre-trained diffusion language model that consists of an encoder and a diffusion-based decoder, which can generate text by gradually transforming a random noise sequence into a coherent text sequence. To pre-train GENIE on a large-scale language corpus, we design a new continuous paragraph denoise objective, which encourages the diffusion-decoder to reconstruct a clean text paragraph from a corrupted version, while preserving the semantic and syntactic coherence. We evaluate GENIE on four downstream text generation benchmarks, namely XSum, CNN/DailyMail, Gigaword, and CommonGen. Our experimental results show that GENIE achieves comparable performance with the state-of-the-art autoregressive models on these benchmarks, and generates more diverse text samples. The code and models of GENIE are available at https://github.com/microsoft/ProphetNet/tree/master/GENIE.

----

## [867] Self-supervised Neural Factor Analysis for Disentangling Utterance-level Speech Representations

**Authors**: *Weiwei Lin, Chenhang He, Man-Wai Mak, Youzhi Tu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23e.html](https://proceedings.mlr.press/v202/lin23e.html)

**Abstract**:

Self-supervised learning (SSL) speech models such as wav2vec and HuBERT have demonstrated state-of-the-art performance on automatic speech recognition (ASR) and proved to be extremely useful in low label-resource settings. However, the success of SSL models has yet to transfer to utterance-level tasks such as speaker, emotion, and language recognition, which still require supervised fine-tuning of the SSL models to obtain good performance. We argue that the problem is caused by the lack of disentangled representations and an utterance-level learning objective for these tasks. Inspired by how HuBERT uses clustering to discover hidden acoustic units, we formulate a factor analysis (FA) model that uses the discovered hidden acoustic units to align the SSL features. The underlying utterance-level representations are disentangled using probabilistic inference on the aligned features. Furthermore, the variational lower bound derived from the FA model provides an utterance-level objective, allowing error gradients to be backpropagated to the Transformer layers to learn highly discriminative acoustic units. When used in conjunction with HuBERT’s masked prediction training, our models outperform the current best model, WavLM, on all utterance-level non-semantic tasks on the SUPERB benchmark with only 20% of labeled data.

----

## [868] Theory on Forgetting and Generalization of Continual Learning

**Authors**: *Sen Lin, Peizhong Ju, Yingbin Liang, Ness B. Shroff*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23f.html](https://proceedings.mlr.press/v202/lin23f.html)

**Abstract**:

Continual learning (CL), which aims to learn a sequence of tasks, has attracted significant recent attention. However, most work has focused on the experimental performance of CL, and theoretical studies of CL are still limited. In particular, there is a lack of understanding on what factors are important and how they affect "catastrophic forgetting" and generalization performance. To fill this gap, our theoretical analysis, under overparameterized linear models, provides the first-known explicit form of the expected forgetting and generalization error for a general CL setup with an arbitrary number of tasks. Further analysis of such a key result yields a number of theoretical explanations about how overparameterization, task similarity, and task ordering affect both forgetting and generalization error of CL. More interestingly, by conducting experiments on real datasets using deep neural networks (DNNs), we show that some of these insights even go beyond the linear models and can be carried over to practical setups. In particular, we use concrete examples to show that our results not only explain some interesting empirical observations in recent studies, but also motivate better practical algorithm designs of CL.

----

## [869] Accelerated Cyclic Coordinate Dual Averaging with Extrapolation for Composite Convex Optimization

**Authors**: *Cheuk Yin Lin, Chaobing Song, Jelena Diakonikolas*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23g.html](https://proceedings.mlr.press/v202/lin23g.html)

**Abstract**:

Exploiting partial first-order information in a cyclic way is arguably the most natural strategy to obtain scalable first-order methods. However, despite their wide use in practice, cyclic schemes are far less understood from a theoretical perspective than their randomized counterparts. Motivated by a recent success in analyzing an extrapolated cyclic scheme for generalized variational inequalities, we propose an Accelerated Cyclic Coordinate Dual Averaging with Extrapolation (A-CODER) method for composite convex optimization, where the objective function can be expressed as the sum of a smooth convex function accessible via a gradient oracle and a convex, possibly nonsmooth, function accessible via a proximal oracle. We show that A-CODER attains the optimal convergence rate with improved dependence on the number of blocks compared to prior work. Furthermore, for the setting where the smooth component of the objective function is expressible in a finite sum form, we introduce a variance-reduced variant of A-CODER, VR-A-CODER, with state-of-the-art complexity guarantees. Finally, we demonstrate the effectiveness of our algorithms through numerical experiments.

----

## [870] Safe Offline Reinforcement Learning with Real-Time Budget Constraints

**Authors**: *Qian Lin, Bo Tang, Zifan Wu, Chao Yu, Shangqin Mao, Qianlong Xie, Xingxing Wang, Dong Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23h.html](https://proceedings.mlr.press/v202/lin23h.html)

**Abstract**:

Aiming at promoting the safe real-world deployment of Reinforcement Learning (RL), research on safe RL has made significant progress in recent years. However, most existing works in the literature still focus on the online setting where risky violations of the safety budget are likely to be incurred during training. Besides, in many realworld applications, the learned policy is required to respond to dynamically determined safety budgets (i.e., constraint threshold) in real time. In this paper, we target at the above real-time budget constraint problem under the offline setting, and propose Trajectory-based REal-time Budget Inference (TREBI) as a novel solution that approaches this problem from the perspective of trajectory distribution. Theoretically, we prove an error bound of the estimation on the episodic reward and cost under the offline setting and thus provide a performance guarantee for TREBI. Empirical results on a wide range of simulation tasks and a real-world large-scale advertising application demonstrate the capability of TREBI in solving real-time budget constraint problems under offline settings.

----

## [871] Probabilistic Unrolling: Scalable, Inverse-Free Maximum Likelihood Estimation for Latent Gaussian Models

**Authors**: *Alexander Lin, Bahareh Tolooshams, Yves F. Atchadé, Demba E. Ba*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23i.html](https://proceedings.mlr.press/v202/lin23i.html)

**Abstract**:

Latent Gaussian models have a rich history in statistics and machine learning, with applications ranging from factor analysis to compressed sensing to time series analysis. The classical method for maximizing the likelihood of these models is the expectation-maximization (EM) algorithm. For problems with high-dimensional latent variables and large datasets, EM scales poorly because it needs to invert as many large covariance matrices as the number of data points. We introduce probabilistic unrolling, a method that combines Monte Carlo sampling with iterative linear solvers to circumvent matrix inversion. Our theoretical analyses reveal that unrolling and backpropagation through the iterations of the solver can accelerate gradient estimation for maximum likelihood estimation. In experiments on simulated and real data, we demonstrate that probabilistic unrolling learns latent Gaussian models up to an order of magnitude faster than gradient EM, with minimal losses in model performance.

----

## [872] Fast Online Value-Maximizing Prediction Sets with Conformal Cost Control

**Authors**: *Zhen Lin, Shubhendu Trivedi, Cao Xiao, Jimeng Sun*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23j.html](https://proceedings.mlr.press/v202/lin23j.html)

**Abstract**:

Many real-world multi-label prediction problems involve set-valued predictions that must satisfy specific requirements dictated by downstream usage. We focus on a typical scenario where such requirements, separately encoding value and cost, compete with each other. For instance, a hospital might expect a smart diagnosis system to capture as many severe, often co-morbid, diseases as possible (the value), while maintaining strict control over incorrect predictions (the cost). We present a general pipeline, dubbed as FavMac, to maximize the value while controlling the cost in such scenarios. FavMac can be combined with almost any multi-label classifier, affording distribution-free theoretical guarantees on cost control. Moreover, unlike prior works, FavMac can handle real-world large-scale applications via a carefully designed online update mechanism, which is of independent interest. Our methodological and theoretical contributions are supported by experiments on several healthcare tasks and synthetic datasets - FavMac furnishes higher value compared with several variants and baselines while maintaining strict cost control.

----

## [873] Unveiling The Mask of Position-Information Pattern Through the Mist of Image Features

**Authors**: *Chieh Hubert Lin, Hung-Yu Tseng, Hsin-Ying Lee, Maneesh Kumar Singh, Ming-Hsuan Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23k.html](https://proceedings.mlr.press/v202/lin23k.html)

**Abstract**:

Recent studies have shown that paddings in convolutional neural networks encode absolute position information which can negatively affect the model performance for certain tasks. However, existing metrics for quantifying the strength of positional information remain unreliable and frequently lead to erroneous results. To address this issue, we propose novel metrics for measuring and visualizing the encoded positional information. We formally define the encoded information as Position-information Pattern from Padding (PPP) and conduct a series of experiments to study its properties as well as its formation. The proposed metrics measure the presence of positional information more reliably than the existing metrics based on PosENet and tests in F-Conv. We also demonstrate that for any extant (and proposed) padding schemes, PPP is primarily a learning artifact and is less dependent on the characteristics of the underlying padding schemes.

----

## [874] Fair yet Asymptotically Equal Collaborative Learning

**Authors**: *Xiaoqiang Lin, Xinyi Xu, See-Kiong Ng, Chuan-Sheng Foo, Bryan Kian Hsiang Low*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23l.html](https://proceedings.mlr.press/v202/lin23l.html)

**Abstract**:

In collaborative learning with streaming data, nodes (e.g., organizations) jointly and continuously learn a machine learning (ML) model by sharing the latest model updates computed from their latest streaming data. For the more resourceful nodes to be willing to share their model updates, they need to be fairly incentivized. This paper explores an incentive design that guarantees fairness so that nodes receive rewards commensurate to their contributions. Our approach leverages an explore-then-exploit formulation to estimate the nodes’ contributions (i.e., exploration) for realizing our theoretically guaranteed fair incentives (i.e., exploitation). However, we observe a "rich get richer" phenomenon arising from the existing approaches to guarantee fairness and it discourages the participation of the less resourceful nodes. To remedy this, we additionally preserve asymptotic equality, i.e., less resourceful nodes achieve equal performance eventually to the more resourceful/“rich” nodes. We empirically demonstrate in two settings with real-world streaming data: federated online incremental learning and federated reinforcement learning, that our proposed approach outperforms existing baselines in fairness and learning performance while remaining competitive in preserving equality.

----

## [875] Efficient Approximations of Complete Interatomic Potentials for Crystal Property Prediction

**Authors**: *Yuchao Lin, Keqiang Yan, Youzhi Luo, Yi Liu, Xiaoning Qian, Shuiwang Ji*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23m.html](https://proceedings.mlr.press/v202/lin23m.html)

**Abstract**:

We study property prediction for crystal materials. A crystal structure consists of a minimal unit cell that is repeated infinitely in 3D space. How to accurately represent such repetitive structures in machine learning models remains unresolved. Current methods construct graphs by establishing edges only between nearby nodes, thereby failing to faithfully capture infinite repeating patterns and distant interatomic interactions. In this work, we propose several innovations to overcome these limitations. First, we propose to model physics-principled interatomic potentials directly instead of only using distances as in many existing methods. These potentials include the Coulomb potential, London dispersion potential, and Pauli repulsion potential. Second, we model the complete set of potentials among all atoms, instead of only between nearby atoms as in existing methods. This is enabled by our approximations of infinite potential summations with provable error bounds. We further develop efficient algorithms to compute the approximations. Finally, we propose to incorporate our computations of complete interatomic potentials into message passing neural networks for representation learning. We perform experiments on the JARVIS and Materials Project benchmarks for evaluation. Results show that the use of interatomic potentials and complete interatomic potentials leads to consistent performance improvements with reasonable computational costs. Our code is publicly available as part of the AIRS library (https://github.com/divelab/AIRS).

----

## [876] Continuation Path Learning for Homotopy Optimization

**Authors**: *Xi Lin, Zhiyuan Yang, Xiaoyuan Zhang, Qingfu Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lin23n.html](https://proceedings.mlr.press/v202/lin23n.html)

**Abstract**:

Homotopy optimization is a traditional method to deal with a complicated optimization problem by solving a sequence of easy-to-hard surrogate subproblems. However, this method can be very sensitive to the continuation schedule design and might lead to a suboptimal solution to the original problem. In addition, the intermediate solutions, often ignored by classic homotopy optimization, could be useful for many real-world applications. In this work, we propose a novel model-based approach to learn the whole continuation path for homotopy optimization, which contains infinite intermediate solutions for any surrogate subproblems. Rather than the classic unidirectional easy-to-hard optimization, our method can simultaneously optimize the original problem and all surrogate subproblems in a collaborative manner. The proposed model also supports the real-time generation of any intermediate solution, which could be desirable for many applications. Experimental studies on different problems show that our proposed method can significantly improve the performance of homotopy optimization and provide extra helpful information to support better decision-making.

----

## [877] Speed-Oblivious Online Scheduling: Knowing (Precise) Speeds is not Necessary

**Authors**: *Alexander Lindermayr, Nicole Megow, Martin Rapp*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lindermayr23a.html](https://proceedings.mlr.press/v202/lindermayr23a.html)

**Abstract**:

We consider online scheduling on unrelated (heterogeneous) machines in a speed-oblivious setting, where an algorithm is unaware of the exact job-dependent processing speeds. We show strong impossibility results for clairvoyant and non-clairvoyant algorithms and overcome them in models inspired by practical settings: (i) we provide competitive learning-augmented algorithms, assuming that (possibly erroneous) predictions on the speeds are given, and (ii) we provide competitive algorithms for the speed-ordered model, where a single global order of machines according to their unknown job-dependent speeds is known. We prove strong theoretical guarantees and evaluate our findings on a representative heterogeneous multi-core processor. These seem to be the first empirical results for scheduling algorithms with predictions that are evaluated in a non-synthetic hardware environment.

----

## [878] Graph Mixup with Soft Alignments

**Authors**: *Hongyi Ling, Zhimeng Jiang, Meng Liu, Shuiwang Ji, Na Zou*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ling23a.html](https://proceedings.mlr.press/v202/ling23a.html)

**Abstract**:

We study graph data augmentation by mixup, which has been used successfully on images. A key operation of mixup is to compute a convex combination of a pair of inputs. This operation is straightforward for grid-like data, such as images, but challenging for graph data. The key difficulty lies in the fact that different graphs typically have different numbers of nodes, and thus there lacks a node-level correspondence between graphs. In this work, we propose S-Mixup, a simple yet effective mixup method for graph classification by soft alignments. Specifically, given a pair of graphs, we explicitly obtain node-level correspondence via computing a soft assignment matrix to match the nodes between two graphs. Based on the soft assignments, we transform the adjacency and node feature matrices of one graph, so that the transformed graph is aligned with the other graph. In this way, any pair of graphs can be mixed directly to generate an augmented graph. We conduct systematic experiments to show that S-Mixup can improve the performance and generalization of graph neural networks (GNNs) on various graph classification tasks. In addition, we show that S-Mixup can increase the robustness of GNNs against noisy labels. Our code is publicly available as part of the DIG package (https://github.com/divelab/DIG).

----

## [879] Deep Graph Representation Learning and Optimization for Influence Maximization

**Authors**: *Chen Ling, Junji Jiang, Junxiang Wang, My T. Thai, Renhao Xue, James Song, Meikang Qiu, Liang Zhao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ling23b.html](https://proceedings.mlr.press/v202/ling23b.html)

**Abstract**:

Influence maximization (IM) is formulated as selecting a set of initial users from a social network to maximize the expected number of influenced users. Researchers have made great progresses to design various traditional methods, yet both theoretical design and performance gain are close to their limits. In the past few years, learning-based IM methods have emerged to achieve stronger generalization ability to unknown graphs than traditional ones. However, the development of learning-based IM methods is still limited by fundamental obstacles, including 1) the difficulty of effectively solving the objective function; 2) the difficulty of characterizing the diversified and underlying diffusion patterns; and 3) the difficulty of adapting the solution under various node-centrality-constrained IM variants. To cope with the above challenges, we design a novel framework DeepIM to generatively characterize the latent representation of seed sets, and we propose to learn the diversified information diffusion pattern in a data-driven and end-to-end manner. Finally, we design a novel objective function to infer optimal seed sets under flexible node-centrality-based budget constraints. Extensive analyses are conducted over both synthetic and real-world datasets to demonstrate the overall performance of DeepIM.

----

## [880] Emergent Agentic Transformer from Chain of Hindsight Experience

**Authors**: *Hao Liu, Pieter Abbeel*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23a.html](https://proceedings.mlr.press/v202/liu23a.html)

**Abstract**:

Large transformer models powered by diverse data and model scale have dominated natural language modeling and computer vision and pushed the frontier of multiple AI areas. In reinforcement learning (RL), despite many efforts into transformer-based policies, a key limitation, however, is that current transformer-based policies cannot learn by directly combining information from multiple sub-optimal trials. In this work, we address this issue using recently proposed chain of hindsight to relabel experience, where we train a transformer on a sequence of trajectory experience ascending sorted according to their total rewards. Our method consists of relabelling target return of each trajectory to the maximum total reward among in sequence of trajectories and training an autoregressive model to predict actions conditioning on past states, actions, rewards, target returns, and task completion tokens, the resulting model, Agentic Transformer (AT), can learn to improve upon itself both at training and test time. As we show on D4RL and ExoRL benchmarks, to the best our knowledge, this is the first time that a simple transformer-based model performs competitively with both temporal-difference and imitation-learning-based approaches, even from sub-optimal data. Our Agentic Transformer also shows a promising scaling trend that bigger models consistently improve results.

----

## [881] Shapley Based Residual Decomposition for Instance Analysis

**Authors**: *Tommy Liu, Amanda S. Barnard*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23b.html](https://proceedings.mlr.press/v202/liu23b.html)

**Abstract**:

In this paper, we introduce the idea of decomposing the residuals of regression with respect to the data instances instead of features. This allows us to determine the effects of each individual instance on the model and each other, and in doing so makes for a model-agnostic method of identifying instances of interest. In doing so, we can also determine the appropriateness of the model and data in the wider context of a given study. The paper focuses on the possible applications that such a framework brings to the relatively unexplored field of instance analysis in the context of Explainable AI tasks.

----

## [882] Learning Representations without Compositional Assumptions

**Authors**: *Tennison Liu, Jeroen Berrevoets, Zhaozhi Qian, Mihaela van der Schaar*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23c.html](https://proceedings.mlr.press/v202/liu23c.html)

**Abstract**:

This paper addresses unsupervised representation learning on tabular data containing multiple views generated by distinct sources of measurement. Traditional methods, which tackle this problem using the multi-view framework, are constrained by predefined assumptions that assume feature sets share the same information and representations should learn globally shared factors. However, this assumption is not always valid for real-world tabular datasets with complex dependencies between feature sets, resulting in localized information that is harder to learn. To overcome this limitation, we propose a data-driven approach that learns feature set dependencies by representing feature sets as graph nodes and their relationships as learnable edges. Furthermore, we introduce $\texttt{LEGATO}$, a novel hierarchical graph autoencoder that learns a smaller, latent graph to aggregate information from multiple views dynamically. This approach results in latent graph components that specialize in capturing localized information from different regions of the input, leading to superior downstream performance.

----

## [883] Byzantine-Robust Learning on Heterogeneous Data via Gradient Splitting

**Authors**: *Yuchen Liu, Chen Chen, Lingjuan Lyu, Fangzhao Wu, Sai Wu, Gang Chen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23d.html](https://proceedings.mlr.press/v202/liu23d.html)

**Abstract**:

Federated learning has exhibited vulnerabilities to Byzantine attacks, where the Byzantine attackers can send arbitrary gradients to a central server to destroy the convergence and performance of the global model. A wealth of robust AGgregation Rules (AGRs) have been proposed to defend against Byzantine attacks. However, Byzantine clients can still circumvent robust AGRs when data is non-Identically and Independently Distributed (non-IID). In this paper, we first reveal the root causes of performance degradation of current robust AGRs in non-IID settings: the curse of dimensionality and gradient heterogeneity. In order to address this issue, we propose GAS, a GrAdient Splitting approach that can successfully adapt existing robust AGRs to non-IID settings. We also provide a detailed convergence analysis when the existing robust AGRs are combined with GAS. Experiments on various real-world datasets verify the efficacy of our proposed GAS. The implementation code is provided in https://github.com/YuchenLiu-a/byzantine-gas.

----

## [884] Towards Constituting Mathematical Structures for Learning to Optimize

**Authors**: *Jialin Liu, Xiaohan Chen, Zhangyang Wang, Wotao Yin, HanQin Cai*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23e.html](https://proceedings.mlr.press/v202/liu23e.html)

**Abstract**:

Learning to Optimize (L2O), a technique that utilizes machine learning to learn an optimization algorithm automatically from data, has gained arising attention in recent years. A generic L2O approach parameterizes the iterative update rule and learns the update direction as a black-box network. While the generic approach is widely applicable, the learned model can overfit and may not generalize well to out-of-distribution test sets. In this paper, we derive the basic mathematical conditions that successful update rules commonly satisfy. Consequently, we propose a novel L2O model with a mathematics-inspired structure that is broadly applicable and generalized well to out-of-distribution problems. Numerical simulations validate our theoretical findings and demonstrate the superior empirical performance of the proposed L2O model.

----

## [885] AudioLDM: Text-to-Audio Generation with Latent Diffusion Models

**Authors**: *Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo P. Mandic, Wenwu Wang, Mark D. Plumbley*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23f.html](https://proceedings.mlr.press/v202/liu23f.html)

**Abstract**:

Text-to-audio (TTA) systems have recently gained attention for their ability to synthesize general audio based on text descriptions. However, previous studies in TTA have limited generation quality with high computational costs. In this study, we propose AudioLDM, a TTA system that is built on a latent space to learn continuous audio representations from contrastive language-audio pretraining (CLAP) embeddings. The pretrained CLAP models enable us to train LDMs with audio embeddings while providing text embeddings as the condition during sampling. By learning the latent representations of audio signals without modelling the cross-modal relationship, AudioLDM improves both generation quality and computational efficiency. Trained on AudioCaps with a single GPU, AudioLDM achieves state-of-the-art TTA performance compared to other open-sourced systems, measured by both objective and subjective metrics. AudioLDM is also the first TTA system that enables various text-guided audio manipulations (e.g., style transfer) in a zero-shot fashion. Our implementation and demos are available at https://audioldm.github.io.

----

## [886] Identifiability of Label Noise Transition Matrix

**Authors**: *Yang Liu, Hao Cheng, Kun Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23g.html](https://proceedings.mlr.press/v202/liu23g.html)

**Abstract**:

The noise transition matrix plays a central role in the problem of learning with noisy labels. Among many other reasons, a large number of existing solutions rely on the knowledge of it. Identifying and estimating the transition matrix without ground truth labels is a critical and challenging task. When label noise transition depends on each instance, the problem of identifying the instance-dependent noise transition matrix becomes substantially more challenging. Despite recently proposed solutions for learning from instance-dependent noisy labels, the literature lacks a unified understanding of when such a problem remains identifiable. The goal of this paper is to characterize the identifiability of the label noise transition matrix. Building on Kruskal’s identifiability results, we are able to show the necessity of multiple noisy labels in identifying the noise transition matrix at the instance level. We further instantiate the results to explain the successes of the state-of-the-art solutions and how additional assumptions alleviated the requirement of multiple noisy labels. Our result reveals that disentangled features improve identification. This discovery led us to an approach that improves the estimation of the transition matrix using properly disentangled features. Code is available at https://github.com/UCSC-REAL/Identifiability.

----

## [887] A Group Symmetric Stochastic Differential Equation Model for Molecule Multi-modal Pretraining

**Authors**: *Shengchao Liu, Weitao Du, Zhi-Ming Ma, Hongyu Guo, Jian Tang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23h.html](https://proceedings.mlr.press/v202/liu23h.html)

**Abstract**:

Molecule pretraining has quickly become the go-to schema to boost the performance of AI-based drug discovery. Naturally, molecules can be represented as 2D topological graphs or 3D geometric point clouds. Although most existing pertaining methods focus on merely the single modality, recent research has shown that maximizing the mutual information (MI) between such two modalities enhances the molecule representation ability. Meanwhile, existing molecule multi-modal pretraining approaches approximate MI based on the representation space encoded from the topology and geometry, thus resulting in the loss of critical structural information of molecules. To address this issue, we propose MoleculeSDE. MoleculeSDE leverages group symmetric (e.g., SE(3)-equivariant and reflection-antisymmetric) stochastic differential equation models to generate the 3D geometries from 2D topologies, and vice versa, directly in the input space. It not only obtains tighter MI bound but also enables prosperous downstream tasks than the previous work. By comparing with 17 pretraining baselines, we empirically verify that MoleculeSDE can learn an expressive representation with state-of-the-art performance on 26 out of 32 downstream tasks.

----

## [888] Using Perturbation to Improve Goodness-of-Fit Tests based on Kernelized Stein Discrepancy

**Authors**: *Xing Liu, Andrew B. Duncan, Axel Gandy*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23i.html](https://proceedings.mlr.press/v202/liu23i.html)

**Abstract**:

Kernelized Stein discrepancy (KSD) is a score-based discrepancy widely used in goodness-of-fit tests. It can be applied even when the target distribution has an unknown normalising factor, such as in Bayesian analysis. We show theoretically and empirically that the KSD test can suffer from low power when the target and the alternative distributions have the same well-separated modes but differ in mixing proportions. We propose to perturb the observed sample via Markov transition kernels, with respect to which the target distribution is invariant. This allows us to then employ the KSD test on the perturbed sample. We provide numerical evidence that with suitably chosen transition kernels the proposed approach can lead to substantially higher power than the KSD test.

----

## [889] Cones: Concept Neurons in Diffusion Models for Customized Generation

**Authors**: *Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, Yang Cao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23j.html](https://proceedings.mlr.press/v202/liu23j.html)

**Abstract**:

Human brains respond to semantic features of presented stimuli with different neurons. This raises the question of whether deep neural networks admit a similar behavior pattern. To investigate this phenomenon, this paper identifies a small cluster of neurons associated with a specific subject in a diffusion model. We call those neurons the concept neurons. They can be identified by statistics of network gradients to a stimulation connected with the given subject. The concept neurons demonstrate magnetic properties in interpreting and manipulating generation results. Shutting them can directly yield the related subject contextualized in different scenes. Concatenating multiple clusters of concept neurons can vividly generate all related concepts in a single image. Our method attains impressive performance for multi-subject customization, even four or more subjects. For large-scale applications, the concept neurons are environmentally friendly as we only need to store a sparse cluster of int index instead of dense float32 parameter values, reducing storage consumption by 90% compared with previous customized generation methods. Extensive qualitative and quantitative studies on diverse scenarios show the superiority of our method in interpreting and manipulating diffusion models.

----

## [890] Opponent-Limited Online Search for Imperfect Information Games

**Authors**: *Weiming Liu, Haobo Fu, Qiang Fu, Wei Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23k.html](https://proceedings.mlr.press/v202/liu23k.html)

**Abstract**:

In recent years, online search has been playing an increasingly important role in imperfect information games (IIGs). Previous online search is known as common-knowledge subgame solving, which has to consider all the states in a common-knowledge closure. This is only computationally tolerable for medium size games, such as poker. To handle larger games, order-1 Knowledge-Limited Subgame Solving (1-KLSS) only considers the states in a knowledge-limited closure, which results in a much smaller subgame. However, 1-KLSS is unsafe. In this paper, we first extend 1-KLSS to Safe-1-KLSS and prove its safeness. To make Safe-1-KLSS applicable to even larger games, we propose Opponent-Limited Subgame Solving (OLSS) to limit how the opponent reaches a subgame and how it acts in the subgame. Limiting the opponent’s strategy dramatically reduces the subgame size and improves the efficiency of subgame solving while still preserving some safety in the limit. Experiments in medium size poker show that Safe-1-KLSS and OLSS are orders of magnitude faster than previous common-knowledge subgame solving. Also, OLSS significantly improves the online performance in a two-player Mahjong game, whose game size prohibits the use of previous common-knowledge subgame-solving methods.

----

## [891] Towards Robust and Safe Reinforcement Learning with Benign Off-policy Data

**Authors**: *Zuxin Liu, Zijian Guo, Zhepeng Cen, Huan Zhang, Yihang Yao, Hanjiang Hu, Ding Zhao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23l.html](https://proceedings.mlr.press/v202/liu23l.html)

**Abstract**:

Previous work demonstrates that the optimal safe reinforcement learning policy in a noise-free environment is vulnerable and could be unsafe under observational attacks. While adversarial training effectively improves robustness and safety, collecting samples by attacking the behavior agent online could be expensive or prohibitively dangerous in many applications. We propose the robuSt vAriational ofF-policy lEaRning (SAFER) approach, which only requires benign training data without attacking the agent. SAFER obtains an optimal non-parametric variational policy distribution via convex optimization and then uses it to improve the parameterized policy robustly via supervised learning. The two-stage policy optimization facilitates robust training, and extensive experiments on multiple robot platforms show the efficiency of SAFER in learning a robust and safe policy: achieving the same reward with much fewer constraint violations during training than on-policy baselines.

----

## [892] Constrained Decision Transformer for Offline Safe Reinforcement Learning

**Authors**: *Zuxin Liu, Zijian Guo, Yihang Yao, Zhepeng Cen, Wenhao Yu, Tingnan Zhang, Ding Zhao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23m.html](https://proceedings.mlr.press/v202/liu23m.html)

**Abstract**:

Safe reinforcement learning (RL) trains a constraint satisfaction policy by interacting with the environment. We aim to tackle a more challenging problem: learning a safe policy from an offline dataset. We study the offline safe RL problem from a novel multi-objective optimization perspective and propose the $\epsilon$-reducible concept to characterize problem difficulties. The inherent trade-offs between safety and task performance inspire us to propose the constrained decision transformer (CDT) approach, which can dynamically adjust the trade-offs during deployment. Extensive experiments show the advantages of the proposed method in learning an adaptive, safe, robust, and high-reward policy. CDT outperforms its variants and strong offline safe RL baselines by a large margin with the same hyperparameters across all tasks, while keeping the zero-shot adaptation capability to different constraint thresholds, making our approach more suitable for real-world RL under constraints.

----

## [893] Understanding and Defending Patched-based Adversarial Attacks for Vision Transformer

**Authors**: *Liang Liu, Yanan Guo, Youtao Zhang, Jun Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23n.html](https://proceedings.mlr.press/v202/liu23n.html)

**Abstract**:

Vision Transformer (ViT) is an attention-based model architecture that has demonstrated superior performance on many computer vision tasks. However, its security properties, in particular, the robustness against adversarial attacks, are yet to be thoroughly studied. Recent works have shown that ViT is vulnerable to attention-based adversarial patch attacks, which cover 1-3% area of the input image using adversarial patches and degrades the model accuracy to 0%. This work provides a generic study targeting the attention-based patch attack. First, we experimentally observe that adversarial patches only activate in a few layers and become lazy during attention updating. According to experiments, we study the theory of how a small adversarial patch perturbates the whole model. Based on understanding adversarial patch attacks, we propose a simple but efficient defense that correctly detects more than 95% of adversarial patches.

----

## [894] NUNO: A General Framework for Learning Parametric PDEs with Non-Uniform Data

**Authors**: *Songming Liu, Zhongkai Hao, Chengyang Ying, Hang Su, Ze Cheng, Jun Zhu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23o.html](https://proceedings.mlr.press/v202/liu23o.html)

**Abstract**:

The neural operator has emerged as a powerful tool in learning mappings between function spaces in PDEs. However, when faced with real-world physical data, which are often highly non-uniformly distributed, it is challenging to use mesh-based techniques such as the FFT. To address this, we introduce the Non-Uniform Neural Operator (NUNO), a comprehensive framework designed for efficient operator learning with non-uniform data. Leveraging a K-D tree-based domain decomposition, we transform non-uniform data into uniform grids while effectively controlling interpolation error, thereby paralleling the speed and accuracy of learning from non-uniform data. We conduct extensive experiments on 2D elasticity, (2+1)D channel flow, and a 3D multi-physics heatsink, which, to our knowledge, marks a novel exploration into 3D PDE problems with complex geometries. Our framework has reduced error rates by up to 60% and enhanced training speeds by 2x to 30x. The code is now available at https://github.com/thu-ml/NUNO .

----

## [895] Hierarchical Programmatic Reinforcement Learning via Learning to Compose Programs

**Authors**: *Guan-Ting Liu, En-Pei Hu, Pu-Jen Cheng, Hung-Yi Lee, Shao-Hua Sun*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23p.html](https://proceedings.mlr.press/v202/liu23p.html)

**Abstract**:

Aiming to produce reinforcement learning (RL) policies that are human-interpretable and can generalize better to novel scenarios, Trivedi et al. (2021) present a method (LEAPS) that first learns a program embedding space to continuously parameterize diverse programs from a pre-generated program dataset, and then searches for a task-solving program in the learned program embedding space when given a task. Despite the encouraging results, the program policies that LEAPS can produce are limited by the distribution of the program dataset. Furthermore, during searching, LEAPS evaluates each candidate program solely based on its return, failing to precisely reward correct parts of programs and penalize incorrect parts. To address these issues, we propose to learn a meta-policy that composes a series of programs sampled from the learned program embedding space. By learning to compose programs, our proposed hierarchical programmatic reinforcement learning (HPRL) framework can produce program policies that describe out-of-distributionally complex behaviors and directly assign credits to programs that induce desired behaviors. The experimental results in the Karel domain show that our proposed framework outperforms baselines. The ablation studies confirm the limitations of LEAPS and justify our design choices.

----

## [896] Online Local Differential Private Quantile Inference via Self-normalization

**Authors**: *Yi Liu, Qirui Hu, Lei Ding, Linglong Kong*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23q.html](https://proceedings.mlr.press/v202/liu23q.html)

**Abstract**:

Based on binary inquiries, we developed an algorithm to estimate population quantiles under Local Differential Privacy (LDP). By self-normalizing, our algorithm provides asymptotically normal estimation with valid inference, resulting in tight confidence intervals without the need for nuisance parameters to be estimated. Our proposed method can be conducted fully online, leading to high computational efficiency and minimal storage requirements with $\mathcal{O}(1)$ space. We also proved an optimality result by an elegant application of one central limit theorem of Gaussian Differential Privacy (GDP) when targeting the frequently encountered median estimation problem. With mathematical proof and extensive numerical testing, we demonstrate the validity of our algorithm both theoretically and experimentally.

----

## [897] GFlowOut: Dropout with Generative Flow Networks

**Authors**: *Dianbo Liu, Moksh Jain, Bonaventure F. P. Dossou, Qianli Shen, Salem Lahlou, Anirudh Goyal, Nikolay Malkin, Chris Chinenye Emezue, Dinghuai Zhang, Nadhir Hassen, Xu Ji, Kenji Kawaguchi, Yoshua Bengio*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23r.html](https://proceedings.mlr.press/v202/liu23r.html)

**Abstract**:

Bayesian inference offers principled tools to tackle many critical problems with modern neural networks such as poor calibration and generalization, and data inefficiency. However, scaling Bayesian inference to large architectures is challenging and requires restrictive approximations. Monte Carlo Dropout has been widely used as a relatively cheap way to approximate inference and estimate uncertainty with deep neural networks. Traditionally, the dropout mask is sampled independently from a fixed distribution. Recent research shows that the dropout mask can be seen as a latent variable, which can be inferred with variational inference. These methods face two important challenges: (a) the posterior distribution over masks can be highly multi-modal which can be difficult to approximate with standard variational inference and (b) it is not trivial to fully utilize sample-dependent information and correlation among dropout masks to improve posterior estimation. In this work, we propose GFlowOut to address these issues. GFlowOut leverages the recently proposed probabilistic framework of Generative Flow Networks (GFlowNets) to learn the posterior distribution over dropout masks. We empirically demonstrate that GFlowOut results in predictive distributions that generalize better to out-of-distribution data and provide uncertainty estimates which lead to better performance in downstream tasks.

----

## [898] 2D-Shapley: A Framework for Fragmented Data Valuation

**Authors**: *Zhihong Liu, Hoang Anh Just, Xiangyu Chang, Xi Chen, Ruoxi Jia*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23s.html](https://proceedings.mlr.press/v202/liu23s.html)

**Abstract**:

Data valuation—quantifying the contribution of individual data sources to certain predictive behaviors of a model—is of great importance to enhancing the transparency of machine learning and designing incentive systems for data sharing. Existing work has focused on evaluating data sources with the shared feature or sample space. How to valuate fragmented data sources of which each only contains partial features and samples remains an open question. We start by presenting a method to calculate the counterfactual of removing a fragment from the aggregated data matrix. Based on the counterfactual calculation, we further propose 2D-Shapley, a theoretical framework for fragmented data valuation that uniquely satisfies some appealing axioms in the fragmented data context. 2D-Shapley empowers a range of new use cases, such as selecting useful data fragments, providing interpretation for sample-wise data values, and fine-grained data issue diagnosis.

----

## [899] Causal Structure Learning for Latent Intervened Non-stationary Data

**Authors**: *Chenxi Liu, Kun Kuang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23t.html](https://proceedings.mlr.press/v202/liu23t.html)

**Abstract**:

Causal structure learning can reveal the causal mechanism behind natural systems. It is well studied that the multiple domain data consisting of observational and interventional samples benefit causal identifiability. However, for non-stationary time series data, domain indexes are often unavailable, making it difficult to distinguish observational samples from interventional samples. To address these issues, we propose a novel Latent Intervened Non-stationary learning (LIN) method to make the domain indexes recovery process and the causal structure learning process mutually promote each other. We characterize and justify a possible faithfulness condition to guarantee the identifiability of the proposed LIN method. Extensive experiments on both synthetic and real-world datasets demonstrate that our method outperforms the baselines on causal structure learning for latent intervened non-stationary data.

----

## [900] Structural Re-weighting Improves Graph Domain Adaptation

**Authors**: *Shikun Liu, Tianchun Li, Yongbin Feng, Nhan Tran, Han Zhao, Qiang Qiu, Pan Li*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23u.html](https://proceedings.mlr.press/v202/liu23u.html)

**Abstract**:

In many real-world applications, graph-structured data used for training and testing have differences in distribution, such as in high energy physics (HEP) where simulation data used for training may not match real experiments. Graph domain adaptation (GDA) is a method used to address these differences. However, current GDA primarily works by aligning the distributions of node representations output by a single graph neural network encoder shared across the training and testing domains, which may often yield sub-optimal solutions. This work examines different impacts of distribution shifts caused by either graph structure or node attributes and identifies a new type of shift, named conditional structure shift (CSS), which current GDA approaches are provably sub-optimal to deal with. A novel approach, called structural reweighting (StruRW), is proposed to address this issue and is tested on synthetic graphs, four benchmark datasets, and a new application in HEP. StruRW has shown significant performance improvement over the baselines in the settings with large graph structure shifts, and reasonable performance improvement when node attribute shift dominates.

----

## [901] Dink-Net: Neural Clustering on Large Graphs

**Authors**: *Yue Liu, Ke Liang, Jun Xia, Sihang Zhou, Xihong Yang, Xinwang Liu, Stan Z. Li*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23v.html](https://proceedings.mlr.press/v202/liu23v.html)

**Abstract**:

Deep graph clustering, which aims to group the nodes of a graph into disjoint clusters with deep neural networks, has achieved promising progress in recent years. However, the existing methods fail to scale to the large graph with million nodes. To solve this problem, a scalable deep graph clustering method (Dink-Net) is proposed with the idea of dilation and shrink. Firstly, by discriminating nodes, whether being corrupted by augmentations, representations are learned in a self-supervised manner. Meanwhile, the cluster centers are initialized as learnable neural parameters. Subsequently, the clustering distribution is optimized by minimizing the proposed cluster dilation loss and cluster shrink loss in an adversarial manner. By these settings, we unify the two-step clustering, i.e., representation learning and clustering optimization, into an end-to-end framework, guiding the network to learn clustering-friendly features. Besides, Dink-Net scales well to large graphs since the designed loss functions adopt the mini-batch data to optimize the clustering distribution even without performance drops. Both experimental results and theoretical analyses demonstrate the superiority of our method. Compared to the runner-up, Dink-Net achieves $9.62%$ NMI improvement on the ogbn-papers100M dataset with 111 million nodes and 1.6 billion edges. The source code is released: https://github.com/yueliu1999/Dink-Net. Besides, a collection (papers, codes, and datasets) of deep graph clustering is shared on GitHub https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering.

----

## [902] Oscillation-free Quantization for Low-bit Vision Transformers

**Authors**: *Shih-Yang Liu, Zechun Liu, Kwang-Ting Cheng*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23w.html](https://proceedings.mlr.press/v202/liu23w.html)

**Abstract**:

Weight oscillation is a by-product of quantization-aware training, in which quantized weights frequently jump between two quantized levels, resulting in training instability and a sub-optimal final model. We discover that the learnable scaling factor, a widely-used $\textit{de facto}$ setting in quantization aggravates weight oscillation. In this work, we investigate the connection between learnable scaling factor and quantized weight oscillation using ViT, and we additionally find that the interdependence between quantized weights in $\textit{query}$ and $\textit{key}$ of a self-attention layer also makes ViT vulnerable to oscillation. We propose three techniques correspondingly: statistical weight quantization ($\rm StatsQ$) to improve quantization robustness compared to the prevalent learnable-scale-based method; confidence-guided annealing ($\rm CGA$) that freezes the weights with $\textit{high confidence}$ and calms the oscillating weights; and $\textit{query}$-$\textit{key}$ reparameterization ($\rm QKR$) to resolve the query-key intertwined oscillation and mitigate the resulting gradient misestimation. Extensive experiments demonstrate that our algorithms successfully abate weight oscillation and consistently achieve substantial accuracy improvement on ImageNet. Specifically, our 2-bit DeiT-T/DeiT-S surpass the previous state-of-the-art by 9.8% and 7.7%, respectively. The code is included in the supplementary material and will be released.

----

## [903] Understanding the Distillation Process from Deep Generative Models to Tractable Probabilistic Circuits

**Authors**: *Xuejie Liu, Anji Liu, Guy Van den Broeck, Yitao Liang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23x.html](https://proceedings.mlr.press/v202/liu23x.html)

**Abstract**:

Probabilistic Circuits (PCs) are a general and unified computational framework for tractable probabilistic models that support efficient computation of various inference tasks (e.g., computing marginal probabilities). Towards enabling such reasoning capabilities in complex real-world tasks, Liu et al. (2022) propose to distill knowledge (through latent variable assignments) from less tractable but more expressive deep generative models. However, it is still unclear what factors make this distillation work well. In this paper, we theoretically and empirically discover that the performance of a PC can exceed that of its teacher model. Therefore, instead of performing distillation from the most expressive deep generative model, we study what properties the teacher model and the PC should have in order to achieve good distillation performance. This leads to a generic algorithmic improvement as well as other data-type-specific ones over the existing latent variable distillation pipeline. Empirically, we outperform SoTA TPMs by a large margin on challenging image modeling benchmarks. In particular, on ImageNet32, PCs achieve 4.06 bits-per-dimension, which is only 0.34 behind variational diffusion models (Kingma et al., 2021).

----

## [904] Averaged Method of Multipliers for Bi-Level Optimization without Lower-Level Strong Convexity

**Authors**: *Risheng Liu, Yaohua Liu, Wei Yao, Shangzhi Zeng, Jin Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23y.html](https://proceedings.mlr.press/v202/liu23y.html)

**Abstract**:

Gradient methods have become mainstream techniques for Bi-Level Optimization (BLO) in learning fields. The validity of existing works heavily rely on either a restrictive Lower- Level Strong Convexity (LLSC) condition or on solving a series of approximation subproblems with high accuracy or both. In this work, by averaging the upper and lower level objectives, we propose a single loop Bi-level Averaged Method of Multipliers (sl-BAMM) for BLO that is simple yet efficient for large-scale BLO and gets rid of the limited LLSC restriction. We further provide non-asymptotic convergence analysis of sl-BAMM towards KKT stationary points, and the comparative advantage of our analysis lies in the absence of strong gradient boundedness assumption, which is always required by others. Thus our theory safely captures a wider variety of applications in deep learning, especially where the upper-level objective is quadratic w.r.t. the lower-level variable. Experimental results demonstrate the superiority of our method.

----

## [905] Graph Switching Dynamical Systems

**Authors**: *Yongtuo Liu, Sara Magliacane, Miltiadis Kofinas, Efstratios Gavves*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23z.html](https://proceedings.mlr.press/v202/liu23z.html)

**Abstract**:

Dynamical systems with complex behaviours, e.g. immune system cells interacting with a pathogen, are commonly modelled by splitting the behaviour in different regimes, or modes, each with simpler dynamics, and then learn the switching behaviour from one mode to another. To achieve this, Switching Dynamical Systems (SDS) are a powerful tool that automatically discovers these modes and mode-switching behaviour from time series data. While effective, these methods focus on independent objects, where the modes of one object are independent of the modes of the other objects. In this paper, we focus on the more general interacting object setting for switching dynamical systems, where the per-object dynamics also depend on an unknown and dynamically changing subset of other objects and their modes. To this end, we propose a novel graph-based approach for switching dynamical systems, GRAph Switching dynamical Systems (GRASS), in which we use a dynamic graph to characterize interactions between objects and learn both intra-object and inter-object mode-switching behaviour. For benchmarking, we create two new datasets, a synthesized ODE-driven particles dataset and a real-world Salsa-couple dancing dataset. Experiments show that GRASS can consistently outperforms previous state-of-the-art methods. We will release code and data after acceptance.

----

## [906] High Probability Convergence of Stochastic Gradient Methods

**Authors**: *Zijian Liu, Ta Duy Nguyen, Thien Hang Nguyen, Alina Ene, Huy Nguyen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23aa.html](https://proceedings.mlr.press/v202/liu23aa.html)

**Abstract**:

In this work, we describe a generic approach to show convergence with high probability for both stochastic convex and non-convex optimization with sub-Gaussian noise. In previous works for convex optimization, either the convergence is only in expectation or the bound depends on the diameter of the domain. Instead, we show high probability convergence with bounds depending on the initial distance to the optimal solution. The algorithms use step sizes analogous to the standard settings and are universal to Lipschitz functions, smooth functions, and their linear combinations. The method can be applied to the non-convex case. We demonstrate an $O((1+\sigma^{2}\log(1/\delta))/T+\sigma/\sqrt{T})$ convergence rate when the number of iterations $T$ is known and an $O((1+\sigma^{2}\log(T/\delta))/\sqrt{T})$ convergence rate when $T$ is unknown for SGD, where $1-\delta$ is the desired success probability. These bounds improve over existing bounds in the literature. We also revisit AdaGrad-Norm (Ward et al., 2019) and show a new analysis to obtain a high probability bound that does not require the bounded gradient assumption made in previous works. The full version of our paper contains results for the standard per-coordinate AdaGrad.

----

## [907] OMS-DPM: Optimizing the Model Schedule for Diffusion Probabilistic Models

**Authors**: *Enshu Liu, Xuefei Ning, Zinan Lin, Huazhong Yang, Yu Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ab.html](https://proceedings.mlr.press/v202/liu23ab.html)

**Abstract**:

Diffusion probabilistic models (DPMs) are a new class of generative models that have achieved state-of-the-art generation quality in various domains. Despite the promise, one major drawback of DPMs is the slow generation speed due to the large number of neural network evaluations required in the generation process. In this paper, we reveal an overlooked dimension—model schedule—for optimizing the trade-off between generation quality and speed. More specifically, we observe that small models, though having worse generation quality when used alone, could outperform large models in certain generation steps. Therefore, unlike the traditional way of using a single model, using different models in different generation steps in a carefully designed model schedule could potentially improve generation quality and speed simultaneously. We design OMS-DPM, a predictor-based search algorithm, to determine the optimal model schedule given an arbitrary generation time budget and a set of pre-trained models. We demonstrate that OMS-DPM can find model schedules that improve generation quality and speed than prior state-of-the-art methods across CIFAR-10, CelebA, ImageNet, and LSUN datasets. When applied to the public checkpoints of the Stable Diffusion model, we are able to accelerate the sampling by 2x while maintaining the generation quality.

----

## [908] Lazy Agents: A New Perspective on Solving Sparse Reward Problem in Multi-agent Reinforcement Learning

**Authors**: *Boyin Liu, Zhiqiang Pu, Yi Pan, Jianqiang Yi, Yanyan Liang, Du Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ac.html](https://proceedings.mlr.press/v202/liu23ac.html)

**Abstract**:

Sparse reward remains a valuable and challenging problem in multi-agent reinforcement learning (MARL). This paper addresses this issue from a new perspective, i.e., lazy agents. We empirically illustrate how lazy agents damage learning from both exploration and exploitation. Then, we propose a novel MARL framework called Lazy Agents Avoidance through Influencing External States (LAIES). Firstly, we examine the causes and types of lazy agents in MARL using a causal graph of the interaction between agents and their environment. Then, we mathematically define the concept of fully lazy agents and teams by calculating the causal effect of their actions on external states using the do-calculus process. Based on definitions, we provide two intrinsic rewards to motivate agents, i.e., individual diligence intrinsic motivation (IDI) and collaborative diligence intrinsic motivation (CDI). IDI and CDI employ counterfactual reasoning based on the external states transition model (ESTM) we developed. Empirical results demonstrate that our proposed method achieves state-of-the-art performance on various tasks, including the sparse-reward version of StarCraft multi-agent challenge (SMAC) and Google Research Football (GRF). Our code is open-source and available at https://github.com/liuboyin/LAIES.

----

## [909] RSC: Accelerate Graph Neural Networks Training via Randomized Sparse Computations

**Authors**: *Zirui Liu, Shengyuan Chen, Kaixiong Zhou, Daochen Zha, Xiao Huang, Xia Hu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ad.html](https://proceedings.mlr.press/v202/liu23ad.html)

**Abstract**:

Training graph neural networks (GNNs) is extremely time consuming because sparse graph-based operations are hard to be accelerated by community hardware. Prior art successfully reduces the computation cost of dense matrix based operations (e.g., convolution and linear) via sampling-based approximation. However, unlike dense matrices, sparse matrices are stored in the irregular data format such that each row/column may have different number of non-zero entries. Thus, compared to the dense counterpart, approximating sparse operations has two unique challenges (1) we cannot directly control the efficiency of approximated sparse operation since the computation is only executed on non-zero entries; (2) sampling sparse matrices is much more inefficient due to the irregular data format. To address the issues, our key idea is to control the accuracy-efficiency trade off by optimizing computation resource allocation layer-wisely and epoch-wisely. For the first challenge, we customize the computation resource to different sparse operations, while limit the total used resource below a certain budget. For the second challenge, we cache previous sampled sparse matrices to reduce the epoch-wise sampling overhead. Finally, we propose a switching mechanisms to improve the generalization of GNNs trained with approximated operations. To this end, we propose Randomized Sparse Computation. In practice, rsc can achieve up to 11.6X speedup for a single sparse operation and 1.6X end-to-end wall-clock time speedup with almost no accuracy drop.

----

## [910] Algorithms for bounding contribution for histogram estimation under user-level privacy

**Authors**: *Yuhan Liu, Ananda Theertha Suresh, Wennan Zhu, Peter Kairouz, Marco Gruteser*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ae.html](https://proceedings.mlr.press/v202/liu23ae.html)

**Abstract**:

We study the problem of histogram estimation under user-level differential privacy, where the goal is to preserve the privacy of all entries of any single user. We consider the heterogeneous scenario where the quantity of data can be different for each user. In this scenario, the amount of noise injected into the histogram to obtain differential privacy is proportional to the maximum user contribution, which can be amplified by few outliers. One approach to circumvent this would be to bound (or limit) the contribution of each user to the histogram. However, if users are limited to small contributions, a significant amount of data will be discarded. In this work, we propose algorithms to choose the best user contribution bound for histogram estimation under both bounded and unbounded domain settings. When the size of the domain is bounded, we propose a user contribution bounding strategy that almost achieves a two-approximation with respect to the best contribution bound in hindsight. For unbounded domain histogram estimation, we propose an algorithm that is logarithmic-approximation with respect to the best contribution bound in hindsight. This result holds without any distribution assumptions on the data. Experiments on both real and synthetic datasets verify our theoretical findings and demonstrate the effectiveness of our algorithms. We also show that clipping bias introduced by bounding user contribution may be reduced under mild distribution assumptions, which can be of independent interest.

----

## [911] Simple Embodied Language Learning as a Byproduct of Meta-Reinforcement Learning

**Authors**: *Evan Zheran Liu, Sahaana Suri, Tong Mu, Allan Zhou, Chelsea Finn*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23af.html](https://proceedings.mlr.press/v202/liu23af.html)

**Abstract**:

Whereas machine learning models typically learn language by directly training on language tasks (e.g., next-word prediction), language emerges in human children as a byproduct of solving non-language tasks (e.g., acquiring food). Motivated by this observation, we ask: can embodied reinforcement learning (RL) agents also indirectly learn language from non-language tasks? Learning to associate language with its meaning requires a dynamic environment with varied language. Therefore, we investigate this question in a multi-task environment with language that varies across the different tasks. Specifically, we design an office navigation environment, where the agent’s goal is to find a particular office, and office locations differ in different buildings (i.e., tasks). Each building includes a floor plan with a simple language description of the goal office’s location, which can be visually read as an RGB image when visited. We find RL agents indeed are able to indirectly learn language. Agents trained with current meta-RL algorithms successfully generalize to reading floor plans with held-out layouts and language phrases, and quickly navigate to the correct office, despite receiving no direct language supervision.

----

## [912] Generating Private Synthetic Data with Genetic Algorithms

**Authors**: *Terrance Liu, Jingwu Tang, Giuseppe Vietri, Steven Wu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ag.html](https://proceedings.mlr.press/v202/liu23ag.html)

**Abstract**:

We study the problem of efficiently generating differentially private synthetic data that approximate the statistical properties of an underlying sensitive dataset. In recent years, there has been a growing line of work that approaches this problem using first-order optimization techniques. However, such techniques are restricted to optimizing differentiable objectives only, severely limiting the types of analyses that can be conducted. For example, first-order mechanisms have been primarily successful in approximating statistical queries only in the form of marginals for discrete data domains. In some cases, one can circumvent such issues by relaxing the task’s objective to maintain differentiability. However, even when possible, these approaches impose a fundamental limitation in which modifications to the minimization problem become additional sources of error. Therefore, we propose Private-GSD, a private genetic algorithm based on zeroth-order optimization heuristics that do not require modifying the original objective; thus, it avoids the aforementioned limitations of first-order optimization. We demonstrate empirically that on data with both discrete and real-valued attributes, Private-GSD outperforms the state-of-the-art methods on non-differential queries while matching accuracy in approximating differentiable ones.

----

## [913] FusionRetro: Molecule Representation Fusion via In-Context Learning for Retrosynthetic Planning

**Authors**: *Songtao Liu, Zhengkai Tu, Minkai Xu, Zuobai Zhang, Lu Lin, Rex Ying, Jian Tang, Peilin Zhao, Dinghao Wu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ah.html](https://proceedings.mlr.press/v202/liu23ah.html)

**Abstract**:

Retrosynthetic planning aims to devise a complete multi-step synthetic route from starting materials to a target molecule. Current strategies use a decoupled approach of single-step retrosynthesis models and search algorithms, taking only the product as the input to predict the reactants for each planning step and ignoring valuable context information along the synthetic route. In this work, we propose a novel framework that utilizes context information for improved retrosynthetic planning. We view synthetic routes as reaction graphs and propose to incorporate context through three principled steps: encode molecules into embeddings, aggregate information over routes, and readout to predict reactants. Our approach is the first attempt to utilize in-context learning for retrosynthesis prediction in retrosynthetic planning. The entire framework can be efficiently optimized in an end-to-end fashion and produce more practical and accurate predictions. Comprehensive experiments demonstrate that by fusing in the context information over routes, our model significantly improves the performance of retrosynthetic planning over baselines that are not context-aware, especially for long synthetic routes. Code is available at https://github.com/SongtaoLiu0823/FusionRetro.

----

## [914] I2SB: Image-to-Image Schrödinger Bridge

**Authors**: *Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A. Theodorou, Weili Nie, Anima Anandkumar*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ai.html](https://proceedings.mlr.press/v202/liu23ai.html)

**Abstract**:

We propose Image-to-Image Schrödinger Bridge (I$^2$SB), a new class of conditional diffusion models that directly learn the nonlinear diffusion processes between two given distributions. These diffusion bridges are particularly useful for image restoration, as the degraded images are structurally informative priors for reconstructing the clean images. I$^2$SB belongs to a tractable class of Schrödinger bridge, the nonlinear extension to score-based models, whose marginal distributions can be computed analytically given boundary pairs. This results in a simulation-free framework for nonlinear diffusions, where the I$^2$SB training becomes scalable by adopting practical techniques used in standard diffusion models. We validate I$^2$SB in solving various image restoration tasks, including inpainting, super-resolution, deblurring, and JPEG restoration on ImageNet 256$\times$256 and show that I$^2$SB surpasses standard conditional diffusion models with more interpretable generative processes. Moreover, I$^2$SB matches the performance of inverse methods that additionally require the knowledge of the corruption operators. Our work opens up new algorithmic opportunities for developing efficient nonlinear diffusion models on a large scale. Project page and codes: https://i2sb.github.io/

----

## [915] What can online reinforcement learning with function approximation benefit from general coverage conditions?

**Authors**: *Fanghui Liu, Luca Viano, Volkan Cevher*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23aj.html](https://proceedings.mlr.press/v202/liu23aj.html)

**Abstract**:

In online reinforcement learning (RL), instead of employing standard structural assumptions on Markov decision processes (MDPs), using a certain coverage condition (original from offline RL) is enough to ensure sample-efficient guarantees (Xie et al. 2023). In this work, we focus on this new direction by digging more possible and general coverage conditions, and study the potential and the utility of them in efficient online RL. We identify more concepts, including the $L^p$ variant of concentrability, the density ratio realizability, and trade-off on the partial/rest coverage condition, that can be also beneficial to sample-efficient online RL, achieving improved regret bound. Furthermore, if exploratory offline data are used, under our coverage conditions, both statistically and computationally efficient guarantees can be achieved for online RL. Besides, even though the MDP structure is given, e.g., linear MDP, we elucidate that, good coverage conditions are still beneficial to obtain faster regret bound beyond $\widetilde{\mathcal{O}}(\sqrt{T})$ and even a logarithmic order regret. These results provide a good justification for the usage of general coverage conditions in efficient online RL.

----

## [916] TR0N: Translator Networks for 0-Shot Plug-and-Play Conditional Generation

**Authors**: *Zhaoyan Liu, Noël Vouitsis, Satya Krishna Gorti, Jimmy Ba, Gabriel Loaiza-Ganem*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ak.html](https://proceedings.mlr.press/v202/liu23ak.html)

**Abstract**:

We propose TR0N, a highly general framework to turn pre-trained unconditional generative models, such as GANs and VAEs, into conditional models. The conditioning can be highly arbitrary, and requires only a pre-trained auxiliary model. For example, we show how to turn unconditional models into class-conditional ones with the help of a classifier, and also into text-to-image models by leveraging CLIP. TR0N learns a lightweight stochastic mapping which "translates’" between the space of conditions and the latent space of the generative model, in such a way that the generated latent corresponds to a data sample satisfying the desired condition. The translated latent samples are then further improved upon through Langevin dynamics, enabling us to obtain higher-quality data samples. TR0N requires no training data nor fine-tuning, yet can achieve a zero-shot FID of 10.9 on MS-COCO, outperforming competing alternatives not only on this metric, but also in sampling speed – all while retaining a much higher level of generality. Our code is available at https://github.com/layer6ai-labs/tr0n.

----

## [917] Global Optimization with Parametric Function Approximation

**Authors**: *Chong Liu, Yu-Xiang Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23al.html](https://proceedings.mlr.press/v202/liu23al.html)

**Abstract**:

We consider the problem of global optimization with noisy zeroth order oracles — a well-motivated problem useful for various applications ranging from hyper-parameter tuning for deep learning to new material design. Existing work relies on Gaussian processes or other non-parametric family, which suffers from the curse of dimensionality. In this paper, we propose a new algorithm GO-UCB that leverages a parametric family of functions (e.g., neural networks) instead. Under a realizable assumption and a few other mild geometric conditions, we show that GO-UCB achieves a cumulative regret of $\tilde{O}(\sqrt{T})$ where $T$ is the time horizon. At the core of GO-UCB is a carefully designed uncertainty set over parameters based on gradients that allows optimistic exploration. Synthetic and real-world experiments illustrate GO-UCB works better than popular Bayesian optimization approaches, even if the model is misspecified.

----

## [918] Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time

**Authors**: *Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Ré, Beidi Chen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23am.html](https://proceedings.mlr.press/v202/liu23am.html)

**Abstract**:

Large language models (LLMs) with hundreds of billions of parameters have sparked a new wave of exciting AI applications. However, they are computationally expensive at inference time. Sparsity is a natural approach to reduce this cost, but existing methods either require costly retraining, have to forgo LLM’s in-context learning ability, or do not yield wall-clock time speedup on modern hardware. We hypothesize that contextual sparsity, which are small, input-dependent sets of attention heads and MLP parameters that yield approximately the same output as the dense model for a given input, can address these issues. We show that contextual sparsity exists, that it can be accurately predicted, and that we can exploit it to speed up LLM inference in wall-clock time without compromising LLM’s quality or in-context learning ability. Based on these insights, we propose DejaVu, a system that uses a low-cost algorithm to predict contextual sparsity on the fly given inputs to each layer, along with an asynchronous and hardware-aware implementation that speeds up LLM inference. We validate that DejaVu can reduce the inference latency of OPT-175B by over 2$\times$ compared to the state-of-the-art FasterTransformer, and over 6$\times$ compared to the widely used Hugging Face implementation, without compromising model quality. The code is available at https://github.com/FMInference/DejaVu.

----

## [919] Trapdoor Normalization with Irreversible Ownership Verification

**Authors**: *Hanwen Liu, Zhenyu Weng, Yuesheng Zhu, Yadong Mu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23an.html](https://proceedings.mlr.press/v202/liu23an.html)

**Abstract**:

This paper introduces a deep model watermark with an irreversible ownership verification scheme: Trapdoor Normalization (TdN), inspired by the trapdoor function in traditional cryptography. To protect intellectual property within deep models, the proposed method is able to embed ownership information into normalization layers during training. We argue and empirically validate that relevant methods are vulnerable to ambiguity attacks, where the forged watermarks can cast ambiguity over the ownership verification. The primary trait that distinguishes this work from previous ones, is its design of a bidirectional connection between watermarks and deep models. Thereby, TdN enables an irreversible ownership verification scheme that is difficult for the adversary to compromise. In this way, the proposed TdN can effectively defeat ambiguity attacks. Extensive experiments demonstrate that the proposed method is not only superior to previous state-of-the-art methods in robustness, but also has better efficiency.

----

## [920] Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models

**Authors**: *Hong Liu, Sang Michael Xie, Zhiyuan Li, Tengyu Ma*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ao.html](https://proceedings.mlr.press/v202/liu23ao.html)

**Abstract**:

Language modeling on large-scale datasets improves performance of various downstream tasks. The validation pre-training loss is often used as the evaluation metric for language models since the pre-training loss tends to be well-correlated with downstream performance (which is itself hard to evaluate comprehensively). Contrary to the conventional wisdom, this paper shows that 1) pre-training loss cannot fully explain downstream performance and 2) flatness of the model is well-correlated with downstream performance where pre-training loss is not. We identify three ways to produce models with the same pre-training loss but different downstream performance: continue pre-training after convergence, increasing the model size, and changing the pre-training algorithms. These experiments demonstrate the existence of implicit bias of pre-training algorithms—among models with the same minimal pre-training loss, they implicitly prefer more transferable ones. Toward understanding this implicit bias, we prove that SGD with standard mini-batch noise implicitly prefers flatter minima of pre-training loss in language models, and empirically observe a strong correlation between flatness (measured by the trace of Hessian) and downstream performance among models with the same pre-training loss. We also prove in a synthetic language setting that among models with the minimal pre-training loss, the flattest model transfers to downstream tasks.

----

## [921] Taxonomy-Structured Domain Adaptation

**Authors**: *Tianyi Liu, Zihao Xu, Hao He, Guang-Yuan Hao, Guang-He Lee, Hao Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ap.html](https://proceedings.mlr.press/v202/liu23ap.html)

**Abstract**:

Domain adaptation aims to mitigate distribution shifts among different domains. However, traditional formulations are mostly limited to categorical domains, greatly simplifying nuanced domain relationships in the real world. In this work, we tackle a generalization with taxonomy-structured domains, which formalizes domains with nested, hierarchical similarity structures such as animal species and product catalogs. We build on the classic adversarial framework and introduce a novel taxonomist, which competes with the adversarial discriminator to preserve the taxonomy information. The equilibrium recovers the classic adversarial domain adaptation’s solution if given a non-informative domain taxonomy (e.g., a flat taxonomy where all leaf nodes connect to the root node) while yielding non-trivial results with other taxonomies. Empirically, our method achieves state-of-the-art performance on both synthetic and real-world datasets with successful adaptation.

----

## [922] Dropout Reduces Underfitting

**Authors**: *Zhuang Liu, Zhiqiu Xu, Joseph Jin, Zhiqiang Shen, Trevor Darrell*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23aq.html](https://proceedings.mlr.press/v202/liu23aq.html)

**Abstract**:

Introduced by Hinton et al. in 2012, dropout has stood the test of time as a regularizer for preventing overfitting in neural networks. In this study, we demonstrate that dropout can also mitigate underfitting when used at the start of training. During the early phase, we find dropout reduces the directional variance of gradients across mini-batches and helps align the mini-batch gradients with the entire dataset’s gradient. This helps counteract the stochasticity of SGD and limit the influence of individual batches on model training. Our findings lead us to a solution for improving performance in underfitting models - early dropout: dropout is applied only during the initial phases of training, and turned off afterwards. Models equipped with early dropout achieve lower final training loss compared to their counterparts without dropout. Additionally, we explore a symmetric technique for regularizing overfitting models - late dropout, where dropout is not used in the early iterations and is only activated later in training. Experiments on ImageNet and various vision tasks demonstrate that our methods consistently improve generalization accuracy. Our results encourage more research on understanding regularization in deep learning and our methods can be useful tools for future neural network training, especially in the era of large data. Code is available at https://github.com/facebookresearch/dropout.

----

## [923] Revisiting Pseudo-Label for Single-Positive Multi-Label Learning

**Authors**: *Biao Liu, Ning Xu, Jiaqi Lv, Xin Geng*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ar.html](https://proceedings.mlr.press/v202/liu23ar.html)

**Abstract**:

To deal with the challenge of high cost of annotating all relevant labels for each example in multi-label learning, single-positive multi-label learning (SPMLL) has been studied in recent years, where each example is annotated with only one positive label. By adopting pseudo-label generation, i.e., assigning pseudo-label to each example by various strategies, existing methods have empirically validated that SPMLL would significantly reduce the amount of supervision with a tolerable damage in classification performance. However, there is no existing method that can provide a theoretical guarantee for learning from pseudo-label on SPMLL. In this paper, the conditions of the effectiveness of learning from pseudo-label for SPMLL are shown and the learnability of pseudo-label-based methods is proven. Furthermore, based on the theoretical guarantee of pseudo-label for SPMLL, we propose a novel SPMLL method named MIME, i.e., Mutual label enhancement for sIngle-positive Multi-label lEarning and prove that the generated pseudo-label by MIME approximately converges to the fully-supervised case. Experiments on four image datasets and five MLL datasets show the effectiveness of our methods over several existing SPMLL approaches.

----

## [924] Retrosynthetic Planning with Dual Value Networks

**Authors**: *Guoqing Liu, Di Xue, Shufang Xie, Yingce Xia, Austin Tripp, Krzysztof Maziarz, Marwin H. S. Segler, Tao Qin, Zongzhang Zhang, Tie-Yan Liu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23as.html](https://proceedings.mlr.press/v202/liu23as.html)

**Abstract**:

Retrosynthesis, which aims to find a route to synthesize a target molecule from commercially available starting materials, is a critical task in drug discovery and materials design. Recently, the combination of ML-based single-step reaction predictors with multi-step planners has led to promising results. However, the single-step predictors are mostly trained offline to optimize the single-step accuracy, without considering complete routes. Here, we leverage reinforcement learning (RL) to improve the single-step predictor, by using a tree-shaped MDP to optimize complete routes. Specifically, we propose a novel online training algorithm, called Planning with Dual Value Networks (PDVN), which alternates between the planning phase and updating phase. In PDVN, we construct two separate value networks to predict the synthesizability and cost of molecules, respectively. To maintain the single-step accuracy, we design a two-branch network structure for the single-step predictor. On the widely-used USPTO dataset, our PDVN algorithm improves the search success rate of existing multi-step planners (e.g., increasing the success rate from 85.79% to 98.95% for Retro$^{\ast}$, and reducing the number of model calls by half while solving 99.47% molecules for RetroGraph). Additionally, PDVN helps find shorter synthesis routes (e.g., reducing the average route length from 5.76 to 4.83 for Retro$^{\ast}$, and from 5.63 to 4.78 for RetroGraph).

----

## [925] Online Nonstochastic Control with Adversarial and Static Constraints

**Authors**: *Xin Liu, Zixian Yang, Lei Ying*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23at.html](https://proceedings.mlr.press/v202/liu23at.html)

**Abstract**:

This paper studies online nonstochastic control problems with adversarial and static constraints. We propose online nonstochastic control algorithms that achieve both sublinear regret and sublinear adversarial constraint violation while keeping static constraint violation minimal against the optimal constrained linear control policy in hindsight. To establish the results, we introduce an online convex optimization with memory framework under adversarial and static constraints, which serves as a subroutine for the constrained online nonstochastic control algorithms. This subroutine also achieves the state-of-the-art regret and constraint violation bounds for constrained online convex optimization problems, which is of independent interest. Our experiments demonstrate the proposed control algorithms are adaptive to adversarial constraints and achieve smaller cumulative costs and violations. Moreover, our algorithms are less conservative and achieve significantly smaller cumulative costs than the state-of-the-art algorithm.

----

## [926] Optimization for Amortized Inverse Problems

**Authors**: *Tianci Liu, Tong Yang, Quan Zhang, Qi Lei*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23au.html](https://proceedings.mlr.press/v202/liu23au.html)

**Abstract**:

Incorporating a deep generative model as the prior distribution in inverse problems has established substantial success in reconstructing images from corrupted observations. Notwithstanding, the existing optimization approaches use gradient descent largely without adapting to the non-convex nature of the problem and can be sensitive to initial values, impeding further performance improvement. In this paper, we propose an efficient amortized optimization scheme for inverse problems with a deep generative prior. Specifically, the optimization task with high degrees of difficulty is decomposed into optimizing a sequence of much easier ones. We provide a theoretical guarantee of the proposed algorithm and empirically validate it on different inverse problems. As a result, our approach outperforms baseline methods qualitatively and quantitatively by a large margin.

----

## [927] Active Policy Improvement from Multiple Black-box Oracles

**Authors**: *Xuefeng Liu, Takuma Yoneda, Chaoqi Wang, Matthew R. Walter, Yuxin Chen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23av.html](https://proceedings.mlr.press/v202/liu23av.html)

**Abstract**:

Reinforcement learning (RL) has made significant strides in various complex domains. However, identifying an effective policy via RL often necessitates extensive exploration. Imitation learning aims to mitigate this issue by using expert demonstrations to guide exploration. In real-world scenarios, one often has access to multiple suboptimal black-box experts, rather than a single optimal oracle. These experts do not universally outperform each other across all states, presenting a challenge in actively deciding which oracle to use and in which state. We introduce MAPS and MAPS-SE, a class of policy improvement algorithms that perform imitation learning from multiple suboptimal oracles. In particular, MAPS actively selects which of the oracles to imitate and improve their value function estimates, and MAPS-SE additionally leverages an active state exploration criterion to determine which states one should explore. We provide a comprehensive theoretical analysis and demonstrate that MAPS and MAPS-SE enjoy sample efficiency advantage over the state-of-the-art policy improvement algorithms. Empirical results show that MAPS-SE significantly accelerates policy optimization via state-wise imitation learning from multiple oracles across a broad spectrum of control tasks in the DeepMind Control Suite.

----

## [928] Gradient-based Wang-Landau Algorithm: A Novel Sampler for Output Distribution of Neural Networks over the Input Space

**Authors**: *Weitang Liu, Yi-Zhuang You, Ying-Wai Li, Jingbo Shang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23aw.html](https://proceedings.mlr.press/v202/liu23aw.html)

**Abstract**:

The output distribution of a neural network (NN) over the entire input space captures the complete input-output mapping relationship, offering in- sights toward a more comprehensive NN under- standing. Exhaustive enumeration or traditional Monte Carlo methods for the entire input space can exhibit impractical sampling time, especially for high-dimensional inputs. To make such difficult sampling computationally feasible, in this paper, we propose a novel Gradient-based Wang-Landau (GWL) sampler. We first draw the connection between the output distribution of a NN and the density of states (DOS) of a physical system. Then, we renovate the classic sampler for the DOS problem, Wang-Landau algorithm, by re-placing its random proposals with gradient-based Monte Carlo proposals. This way, our GWL sampler investigates the under-explored subsets of the input space much more efficiently. Extensive experiments have verified the accuracy of the output distribution generated by GWL and also showcased several interesting findings - for example, in a binary image classification task, both CNN and ResNet mapped the majority of human unrecognizable images to very negative logit values.

----

## [929] VectorMapNet: End-to-end Vectorized HD Map Learning

**Authors**: *Yicheng Liu, Tianyuan Yuan, Yue Wang, Yilun Wang, Hang Zhao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ax.html](https://proceedings.mlr.press/v202/liu23ax.html)

**Abstract**:

Autonomous driving systems require High-Definition (HD) semantic maps to navigate around urban roads. Existing solutions approach the semantic mapping problem by offline manual annotation, which suffers from serious scalability issues. Recent learning-based methods produce dense rasterized segmentation predictions to construct maps. However, these predictions do not include instance information of individual map elements and require heuristic post-processing to obtain vectorized maps. To tackle these challenges, we introduce an end-to-end vectorized HD map learning pipeline, termed VectorMapNet. VectorMapNet takes onboard sensor observations and predicts a sparse set of polylines in the bird’s-eye view. This pipeline can explicitly model the spatial relation between map elements and generate vectorized maps that are friendly to downstream autonomous driving tasks. Extensive experiments show that VectorMapNet achieve strong map learning performance on both nuScenes and Argoverse2 dataset, surpassing previous state-of-the-art methods by 14.2 mAP and 14.6mAP. Qualitatively, VectorMapNet is capable of generating comprehensive maps and capturing fine-grained details of road geometry. To the best of our knowledge, VectorMapNet is the first work designed towards end-to-end vectorized map learning from onboard observations.

----

## [930] Partially Observable Multi-agent RL with (Quasi-)Efficiency: The Blessing of Information Sharing

**Authors**: *Xiangyu Liu, Kaiqing Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ay.html](https://proceedings.mlr.press/v202/liu23ay.html)

**Abstract**:

We study provable multi-agent reinforcement learning (MARL) in the general framework of partially observable stochastic games (POSGs). To circumvent the known hardness results and the use of computationally intractable oracles, we propose to leverage the potential information-sharing among agents, a standard practice in empirical MARL and a common model for multi-agent control systems with communications. We first establish several computation complexity results to justify the necessity of information-sharing, as well as the observability assumption that has enabled quasi-efficient single-agent RL with partial observations, for computational efficiency in solving POSGs. We then propose to further approximate the shared common information to construct an approximate model of the POSG, in which planning an approximate equilibrium (in terms of solving the original POSG) can be quasi-efficient, i.e., of quasi-polynomial-time, under the aforementioned assumptions. Furthermore, we develop a partially observable MARL algorithm that is both statistically and computationally quasi-efficient. We hope our study can open up the possibilities of leveraging and even designing different information structures, for developing both sample- and computation-efficient partially observable MARL.

----

## [931] Prometheus: Taming Sample and Communication Complexities in Constrained Decentralized Stochastic Bilevel Learning

**Authors**: *Zhuqing Liu, Xin Zhang, Prashant Khanduri, Songtao Lu, Jia Liu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23az.html](https://proceedings.mlr.press/v202/liu23az.html)

**Abstract**:

In recent years, decentralized bilevel optimization has gained significant attention thanks to its versatility in modeling a wide range of multi-agent learning problems, such as multi-agent reinforcement learning and multi-agent meta-learning. However, one unexplored and fundamental problem in this area is how to solve decentralized stochastic bilevel optimization problems with domain constraints while achieving low sample and communication complexities. This problem often arises from multi-agent learning problems with safety constraints. As shown in this paper, constrained decentralized bilevel optimization is far more challenging than its unconstrained counterpart due to the complex coupling structure, which necessitates new algorithm design and analysis techniques. Toward this end, we investigate a class of constrained decentralized bilevel optimization problems, where multiple agents collectively solve a nonconvex-strongly-convex bilevel problem with constraints in the upper-level variables. We propose an algorithm called Prometheus (proximal tracked stochastic recursive estimator) that achieves the first $\mathcal{O}(\epsilon^{-1})$ results in both sample and communication complexities for constrained decentralized bilevel optimization, where $\epsilon>0$ is a desired stationarity error. Collectively, the results in this work contribute to a theoretical foundation for low sample- and communication-complexity constrained decentralized bilevel learning.

----

## [932] D2Match: Leveraging Deep Learning and Degeneracy for Subgraph Matching

**Authors**: *Xuanzhou Liu, Lin Zhang, Jiaqi Sun, Yujiu Yang, Haiqin Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23ba.html](https://proceedings.mlr.press/v202/liu23ba.html)

**Abstract**:

Subgraph matching is a fundamental building block for graph-based applications and is challenging due to its high-order combinatorial nature. Existing studies usually tackle it by combinatorial optimization or learning-based methods. However, they suffer from exponential computational costs or searching the matching without theoretical guarantees. In this paper, we develop $D^2$Match by leveraging the efficiency of Deep learning and Degeneracy for subgraph matching. More specifically, we first prove that subgraph matching can degenerate to subtree matching, and subsequently is equivalent to finding a perfect matching on a bipartite graph. We can then yield an implementation of linear time complexity by the built-in tree-structured aggregation mechanism on graph neural networks. Moreover, circle structures and node attributes can be easily incorporated in $D^2$Match to boost the matching performance. Finally, we conduct extensive experiments to show the superior performance of our $D^2$Match and confirm that our $D^2$Match indeed exploits the subtrees and differs from existing GNNs-based subgraph matching methods that depend on memorizing the data distribution divergence.

----

## [933] Image Shortcut Squeezing: Countering Perturbative Availability Poisons with Compression

**Authors**: *Zhuoran Liu, Zhengyu Zhao, Martha A. Larson*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23bb.html](https://proceedings.mlr.press/v202/liu23bb.html)

**Abstract**:

Perturbative availability poisoning (PAP) adds small changes to images to prevent their use for model training. Current research adopts the belief that practical and effective approaches to countering such poisons do not exist. In this paper, we argue that it is time to abandon this belief. We present extensive experiments showing that 12 state-of-the-art PAP methods are vulnerable to Image Shortcut Squeezing (ISS), which is based on simple compression. For example, on average, ISS restores the CIFAR-10 model accuracy to 81.73%, surpassing the previous best preprocessing-based countermeasures by 37.97% absolute. ISS also (slightly) outperforms adversarial training and has higher generalizability to unseen perturbation norms and also higher efficiency. Our investigation reveals that the property of PAP perturbations depends on the type of surrogate model used for poison generation, and it explains why a specific ISS compression yields the best performance for a specific type of PAP perturbation. We further test stronger, adaptive poisoning, and show it falls short of being an ideal defense against ISS. Overall, our results demonstrate the importance of considering various (simple) countermeasures to ensure the meaningfulness of analysis carried out during the development of availability poisons.

----

## [934] Which Invariance Should We Transfer? A Causal Minimax Learning Approach

**Authors**: *Mingzhou Liu, Xiangyu Zheng, Xinwei Sun, Fang Fang, Yizhou Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23bc.html](https://proceedings.mlr.press/v202/liu23bc.html)

**Abstract**:

A major barrier to deploying current machine learning models lies in their non-reliability to dataset shifts. To resolve this problem, most existing studies attempted to transfer stable information to unseen environments. Particularly, independent causal mechanisms-based methods proposed to remove mutable causal mechanisms via the do-operator. Compared to previous methods, the obtained stable predictors are more effective in identifying stable information. However, a key question remains: which subset of this whole stable information should the model transfer, in order to achieve optimal generalization ability? To answer this question, we present a comprehensive minimax analysis from a causal perspective. Specifically, we first provide a graphical condition for the whole stable set to be optimal. When this condition fails, we surprisingly find with an example that this whole stable set, although can fully exploit stable information, is not the optimal one to transfer. To identify the optimal subset under this case, we propose to estimate the worst-case risk with a novel optimization scheme over the intervention functions on mutable causal mechanisms. We then propose an efficient algorithm to search for the subset with minimal worst-case risk, based on a newly defined equivalence relation between stable subsets. Compared to the exponential cost of exhaustively searching over all subsets, our searching strategy enjoys a polynomial complexity. The effectiveness and efficiency of our methods are demonstrated on synthetic data and the diagnosis of Alzheimer’s disease.

----

## [935] Unsupervised Out-of-Distribution Detection with Diffusion Inpainting

**Authors**: *Zhenzhen Liu, Jin Peng Zhou, Yufan Wang, Kilian Q. Weinberger*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23bd.html](https://proceedings.mlr.press/v202/liu23bd.html)

**Abstract**:

Unsupervised out-of-distribution detection (OOD) seeks to identify out-of-domain data by learning only from unlabeled in-domain data. We present a novel approach for this task – Lift, Map, Detect (LMD) – that leverages recent advancement in diffusion models. Diffusion models are one type of generative models. At their core, they learn an iterative denoising process that gradually maps a noisy image closer to their training manifolds. LMD leverages this intuition for OOD detection. Specifically, LMD lifts an image off its original manifold by corrupting it, and maps it towards the in-domain manifold with a diffusion model. For an OOD image, the mapped image would have a large distance away from its original manifold, and LMD would identify it as OOD accordingly. We show through extensive experiments that LMD achieves competitive performance across a broad variety of datasets. Code can be found at https://github.com/zhenzhel/lift_map_detect.

----

## [936] NA2Q: Neural Attention Additive Model for Interpretable Multi-Agent Q-Learning

**Authors**: *Zichuan Liu, Yuanyang Zhu, Chunlin Chen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23be.html](https://proceedings.mlr.press/v202/liu23be.html)

**Abstract**:

Value decomposition is widely used in cooperative multi-agent reinforcement learning, however, its implicit credit assignment mechanism is not yet fully understood due to black-box networks. In this work, we study an interpretable value decomposition framework via the family of generalized additive models. We present a novel method, named Neural Attention Additive Q-learning (N$\text{A}^\text{2}$Q), providing inherent intelligibility of collaboration behavior. N$\text{A}^\text{2}$Q can explicitly factorize the optimal joint policy induced by enriching shape functions to model all possible coalition of agents into individual policies. Moreover, we construct the identity semantics to promote estimating credits together with the global state and individual value functions, where local semantic masks help us diagnose whether each agent captures the relevant-task information. Extensive experiments show that N$\text{A}^\text{2}$Q consistently achieves superior performance compared to different state-of-the-art methods on all challenging tasks, while yielding human-like interpretability.

----

## [937] Contextual Combinatorial Bandits with Probabilistically Triggered Arms

**Authors**: *Xutong Liu, Jinhang Zuo, Siwei Wang, John C. S. Lui, Mohammad Hajiesmaili, Adam Wierman, Wei Chen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/liu23bf.html](https://proceedings.mlr.press/v202/liu23bf.html)

**Abstract**:

We study contextual combinatorial bandits with probabilistically triggered arms (C$^2$MAB-T) under a variety of smoothness conditions that capture a wide range of applications, such as contextual cascading bandits and contextual influence maximization bandits. Under the triggering probability modulated (TPM) condition, we devise the C$^2$-UCB-T algorithm and propose a novel analysis that achieves an $\tilde{O}(d\sqrt{KT})$ regret bound, removing a potentially exponentially large factor $O(1/p_{\min})$, where $d$ is the dimension of contexts, $p_{\min}$ is the minimum positive probability that any arm can be triggered, and batch-size $K$ is the maximum number of arms that can be triggered per round. Under the variance modulated (VM) or triggering probability and variance modulated (TPVM) conditions, we propose a new variance-adaptive algorithm VAC$^2$-UCB and derive a regret bound $\tilde{O}(d\sqrt{T})$, which is independent of the batch-size $K$. As a valuable by-product, our analysis technique and variance-adaptive algorithm can be applied to the CMAB-T and C$^2$MAB setting, improving existing results there as well. We also include experiments that demonstrate the improved performance of our algorithms compared with benchmark algorithms on synthetic and real-world datasets.

----

## [938] Flipping Coins to Estimate Pseudocounts for Exploration in Reinforcement Learning

**Authors**: *Sam Lobel, Akhil Bagaria, George Konidaris*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lobel23a.html](https://proceedings.mlr.press/v202/lobel23a.html)

**Abstract**:

We propose a new method for count-based exploration in high-dimensional state spaces. Unlike previous work which relies on density models, we show that counts can be derived by averaging samples from the Rademacher distribution (or coin flips). This insight is used to set up a simple supervised learning objective which, when optimized, yields a state’s visitation count. We show that our method is significantly more effective at deducing ground-truth visitation counts than previous work; when used as an exploration bonus for a model-free reinforcement learning algorithm, it outperforms existing approaches on most of 9 challenging exploration tasks, including the Atari game Montezuma’s Revenge.

----

## [939] Multi-Symmetry Ensembles: Improving Diversity and Generalization via Opposing Symmetries

**Authors**: *Charlotte Loh, Seungwook Han, Shivchander Sudalairaj, Rumen Dangovski, Kai Xu, Florian Wenzel, Marin Soljacic, Akash Srivastava*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/loh23a.html](https://proceedings.mlr.press/v202/loh23a.html)

**Abstract**:

Deep ensembles (DE) have been successful in improving model performance by learning diverse members via the stochasticity of random initialization. While recent works have attempted to promote further diversity in DE via hyperparameters or regularizing loss functions, these methods primarily still rely on a stochastic approach to explore the hypothesis space. In this work, we present Multi-Symmetry Ensembles (MSE), a framework for constructing diverse ensembles by capturing the multiplicity of hypotheses along symmetry axes, which explore the hypothesis space beyond stochastic perturbations of model weights and hyperparameters. We leverage recent advances in contrastive representation learning to create models that separately capture opposing hypotheses of invariant and equivariant functional classes and present a simple ensembling approach to efficiently combine appropriate hypotheses for a given task. We show that MSE effectively captures the multiplicity of conflicting hypotheses that is often required in large, diverse datasets like ImageNet. As a result of their inherent diversity, MSE improves classification performance, uncertainty quantification, and generalization across a series of transfer tasks. Our code is available at https://github.com/clott3/multi-sym-ensem

----

## [940] The Flan Collection: Designing Data and Methods for Effective Instruction Tuning

**Authors**: *Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V. Le, Barret Zoph, Jason Wei, Adam Roberts*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/longpre23a.html](https://proceedings.mlr.press/v202/longpre23a.html)

**Abstract**:

We study the design decision of publicly available instruction tuning methods, by reproducing and breaking down the development of Flan 2022 (Chung et al., 2022). Through careful ablation studies on the Flan Collection of tasks and methods, we tease apart the effect of design decisions which enable Flan-T5 to outperform prior work by 3-17% across evaluation settings. We find task balancing and enrichment techniques are overlooked but critical to effective instruction tuning, and in particular, training with mixed prompt settings (zero-shot, few-shot, chain-of-thought) actually yields equivalent or stronger (2%) performance in all settings. In further experiments we show Flan-T5 requires less finetuning to converge higher and faster than T5 on single downstream tasks – motivating instruction-tuned models as more computationally-efficient starting checkpoints for new tasks. Finally, to accelerate research on instruction tuning, we make the Flan 2022 collection of datasets, templates, and methods publicly available.

----

## [941] Dataset Distillation with Convexified Implicit Gradients

**Authors**: *Noel Loo, Ramin M. Hasani, Mathias Lechner, Daniela Rus*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/loo23a.html](https://proceedings.mlr.press/v202/loo23a.html)

**Abstract**:

We propose a new dataset distillation algorithm using reparameterization and convexification of implicit gradients (RCIG), that substantially improves the state-of-the-art. To this end, we first formulate dataset distillation as a bi-level optimization problem. Then, we show how implicit gradients can be effectively used to compute meta-gradient updates. We further equip the algorithm with a convexified approximation that corresponds to learning on top of a frozen finite-width neural tangent kernel. Finally, we improve bias in implicit gradients by parameterizing the neural network to enable analytical computation of final-layer parameters given the body parameters. RCIG establishes the new state-of-the-art on a diverse series of dataset distillation tasks. Notably, with one image per class, on resized ImageNet, RCIG sees on average a 108% improvement over the previous state-of-the-art distillation algorithm. Similarly, we observed a 66% gain over SOTA on Tiny-ImageNet and 37% on CIFAR-100.

----

## [942] Reflected Diffusion Models

**Authors**: *Aaron Lou, Stefano Ermon*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lou23a.html](https://proceedings.mlr.press/v202/lou23a.html)

**Abstract**:

Score-based diffusion models learn to reverse a stochastic differential equation that maps data to noise. However, for complex tasks, numerical error can compound and result in highly unnatural samples. Previous work mitigates this drift with thresholding, which projects to the natural data domain (such as pixel space for images) after each diffusion step, but this leads to a mismatch between the training and generative processes. To incorporate data constraints in a principled manner, we present Reflected Diffusion Models, which instead reverse a reflected stochastic differential equation evolving on the support of the data. Our approach learns the perturbed score function through a generalized score matching loss and extends key components of standard diffusion models including diffusion guidance, likelihood-based training, and ODE sampling. We also bridge the theoretical gap with thresholding: such schemes are just discretizations of reflected SDEs. On standard image benchmarks, our method is competitive with or surpasses the state of the art without architectural modifications and, for classifier-free guidance, our approach enables fast exact sampling with ODEs and produces more faithful samples under high guidance weight.

----

## [943] Never mind the metrics - what about the uncertainty? Visualising binary confusion matrix metric distributions to put performance in perspective

**Authors**: *David R. Lovell, Dimity Miller, Jaiden Capra, Andrew P. Bradley*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lovell23a.html](https://proceedings.mlr.press/v202/lovell23a.html)

**Abstract**:

There are strong incentives to build classification systems that show outstanding performance on various datasets and benchmarks. This can encourage a narrow focus on models and the performance metrics used to evaluate and compare them—resulting in a growing body of literature to evaluate and compare metrics. This paper strives for a more balanced perspective on binary classifier performance metrics by showing how uncertainty in these metrics can easily eclipse differences in empirical performance. We emphasise the discrete nature of confusion matrices and show how they can be well represented in a 3D lattice whose cross-sections form the space of receiver operating characteristic (ROC) curves. We develop novel interactive visualisations of performance metric contours within (and beyond) ROC space, showing the discrete probability mass functions of true and false positive rates and how these relate to performance metric distributions. We aim to raise awareness of the substantial uncertainty in performance metric estimates that can arise when classifiers are evaluated on empirical datasets and benchmarks, and that performance claims should be tempered by this understanding.

----

## [944] Bilevel Optimization with Coupled Decision-Dependent Distributions

**Authors**: *Songtao Lu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lu23a.html](https://proceedings.mlr.press/v202/lu23a.html)

**Abstract**:

Bilevel optimization has gained significant popularity in recent years due to its ability to formulate various machine learning problems. For instance, in meta-learning, the upper-level (UL) problem offers a good initialization for the lower-level (LL) model to facilitate adaptation. However, the decision variables can impact data features and outcomes, leading to the phenomenon known as performativity. In this work, we investigate the inclusion of decision-dependent distributions in bilevel optimization. Specifically, we consider the scenarios where the UL data distribution depends on the LL optimization variable, and the LL data distribution also depends on the UL decision variable. We first establish sufficient conditions for the existence of performatively stable (PS) solutions in this class of bilevel problems. Also, we propose efficient stochastic algorithms to find the PS point with theoretical convergence rate analysis and discuss the theoretical optimality of the obtained solution. Our theoretical analysis is corroborated through a series of numerical experiments, wherein we evaluate the performance of the bilevel performative prediction algorithms alongside non-performative counterparts in the context of meta strategic learning problems.

----

## [945] Two-Scale Gradient Descent Ascent Dynamics Finds Mixed Nash Equilibria of Continuous Games: A Mean-Field Perspective

**Authors**: *Yulong Lu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lu23b.html](https://proceedings.mlr.press/v202/lu23b.html)

**Abstract**:

Finding the mixed Nash equilibria (MNE) of a two-player zero sum continuous game is an important and challenging problem in machine learning. A canonical algorithm to finding the MNE is the noisy gradient descent ascent method which in the infinite particle limit gives rise to the Mean-Field Gradient Descent Ascent (GDA) dynamics on the space of probability measures. In this paper, we first study the convergence of a two-scale Mean-Field GDA dynamics for finding the MNE of the entropy-regularized objective. More precisely we show that for each finite temperature (or regularization parameter), the two-scale Mean-Field GDA with a suitable finite scale ratio converges exponentially to the unique MNE without assuming the convexity or concavity of the interaction potential. The key ingredient of our proof lies in the construction of new Lyapunov functions that dissipate exponentially along the Mean-Field GDA. We further study the simulated annealing of the Mean-Field GDA dynamics. We show that with a temperature schedule that decays logarithmically in time the annealed Mean-Field GDA converges to the MNE of the original unregularized objective.

----

## [946] STEP: Learning N: M Structured Sparsity Masks from Scratch with Precondition

**Authors**: *Yucheng Lu, Shivani Agrawal, Suvinay Subramanian, Oleg Rybakov, Christopher De Sa, Amir Yazdanbakhsh*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lu23c.html](https://proceedings.mlr.press/v202/lu23c.html)

**Abstract**:

Recent innovations on hardware (e.g. Nvidia A100) have motivated learning N:M structured sparsity masks from scratch for fast model inference. However, state-of-the-art learning recipes in this regime (e.g. SR-STE) are proposed for non-adaptive optimizers like momentum SGD, while incurring non-trivial accuracy drop for Adam-trained models like attention-based LLMs. In this paper, we first demonstrate such gap origins from poorly estimated second moment (i.e. variance) in Adam states given by the masked weights. We conjecture that learning N:M masks with Adam should take the critical regime of variance estimation into account. In light of this, we propose STEP, an Adam-aware recipe that learns N:M masks with two phases: first, STEP calculates a reliable variance estimate (precondition phase) and subsequently, the variance remains fixed and is used as a precondition to learn N:M masks (mask-learning phase). STEP automatically identifies the switching point of two phases by dynamically sampling variance changes over the training trajectory and testing the sample concentration. Empirically, we evaluate STEP and other baselines such as ASP and SR-STE on multiple tasks including CIFAR classification, machine translation and LLM fine-tuning (BERT-Base, GPT-2). We show STEP mitigates the accuracy drop of baseline recipes and is robust to aggressive structured sparsity ratios.

----

## [947] Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning

**Authors**: *Cheng Lu, Huayu Chen, Jianfei Chen, Hang Su, Chongxuan Li, Jun Zhu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lu23d.html](https://proceedings.mlr.press/v202/lu23d.html)

**Abstract**:

Guided sampling is a vital approach for applying diffusion models in real-world tasks that embeds human-defined guidance during the sampling procedure. This paper considers a general setting where the guidance is defined by an (unnormalized) energy function. The main challenge for this setting is that the intermediate guidance during the diffusion sampling procedure, which is jointly defined by the sampling distribution and the energy function, is unknown and is hard to estimate. To address this challenge, we propose an exact formulation of the intermediate guidance as well as a novel training objective named contrastive energy prediction (CEP) to learn the exact guidance. Our method is guaranteed to converge to the exact guidance under unlimited model capacity and data samples, while previous methods can not. We demonstrate the effectiveness of our method by applying it to offline reinforcement learning (RL). Extensive experiments on D4RL benchmarks demonstrate that our method outperforms existing state-of-the-art algorithms. We also provide some examples of applying CEP for image synthesis to demonstrate the scalability of CEP on high-dimensional data.

----

## [948] Exploring the Limits of Model-Targeted Indiscriminate Data Poisoning Attacks

**Authors**: *Yiwei Lu, Gautam Kamath, Yaoliang Yu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lu23e.html](https://proceedings.mlr.press/v202/lu23e.html)

**Abstract**:

Indiscriminate data poisoning attacks aim to decrease a model’s test accuracy by injecting a small amount of corrupted training data. Despite significant interest, existing attacks remain relatively ineffective against modern machine learning (ML) architectures. In this work, we introduce the notion of model poisoning reachability as a technical tool to explore the intrinsic limits of data poisoning attacks towards target parameters (i.e., model-targeted attacks). We derive an easily computable threshold to establish and quantify a surprising phase transition phenomenon among popular ML models: data poisoning attacks can achieve certain target parameters only when the poisoning ratio exceeds our threshold. Building on existing parameter corruption attacks and refining the Gradient Canceling attack, we perform extensive experiments to confirm our theoretical findings, test the predictability of our transition threshold, and significantly improve existing indiscriminate data poisoning baselines over a range of datasets and models. Our work highlights the critical role played by the poisoning ratio, and sheds new insights on existing empirical results, attacks and mitigation strategies in data poisoning.

----

## [949] QAS-Bench: Rethinking Quantum Architecture Search and A Benchmark

**Authors**: *Xudong Lu, Kaisen Pan, Ge Yan, Jiaming Shan, Wenjie Wu, Junchi Yan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lu23f.html](https://proceedings.mlr.press/v202/lu23f.html)

**Abstract**:

Automatic quantum architecture search (QAS) has been widely studied across disciplines with different implications. In this paper, beyond a particular domain, we formulate the QAS problem into two basic (and relatively even ideal) tasks: i) arbitrary quantum circuit (QC) regeneration given a target QC; ii) approximating an arbitrary unitary (oracle). The latter can be connected to the setting of various quantum machine learning tasks and other QAS applications. Based on these two tasks, we generate a public QAS benchmark including 900 random QCs and 400 random unitary matrices which is still missing in the literature. We evaluate six baseline algorithms including brute force search, simulated annealing, genetic algorithm, reinforcement learning, hybrid algorithm, and differentiable algorithm as part of our benchmark. One characteristic of our proposed evaluation protocol on the basic tasks is that it deprives the domain-specific designs and techniques as used in existing QAS literature, making a unified evaluation possible and focusing on the vanilla search methods themselves without coupling with domain prior. In fact, the unitary approximation task could be algorithmically more difficult than the specific problems as it needs to explore the whole matrix space to fit the unitary. While specific tasks often only need to fit a partial observation of the unitary as the objective for search. Data and code are available at https://github.com/Lucky-Lance/QAS-Bench.

----

## [950] Learning Dense Correspondences between Photos and Sketches

**Authors**: *Xuanchen Lu, Xiaolong Wang, Judith E. Fan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lu23g.html](https://proceedings.mlr.press/v202/lu23g.html)

**Abstract**:

Humans effortlessly grasp the connection between sketches and real-world objects, even when these sketches are far from realistic. Moreover, human sketch understanding goes beyond categorization – critically, it also entails understanding how individual elements within a sketch correspond to parts of the physical world it represents. What are the computational ingredients needed to support this ability? Towards answering this question, we make two contributions: first, we introduce a new sketch-photo correspondence benchmark, PSC6k, containing 150K annotations of 6250 sketch-photo pairs across 125 object categories, augmenting the existing Sketchy dataset with fine-grained correspondence metadata. Second, we propose a self-supervised method for learning dense correspondences between sketch-photo pairs, building upon recent advances in correspondence learning for pairs of photos. Our model uses a spatial transformer network to estimate the warp flow between latent representations of a sketch and photo extracted by a contrastive learning-based ConvNet backbone. We found that this approach outperformed several strong baselines and produced predictions that were quantitatively consistent with other warp-based methods. However, our benchmark also revealed systematic differences between predictions of the suite of models we tested and those of humans. Taken together, our work suggests a promising path towards developing artificial systems that achieve more human-like understanding of visual images at different levels of abstraction. Project page: https://photo-sketch-correspondence.github.io

----

## [951] Adversarial Cheap Talk

**Authors**: *Chris Lu, Timon Willi, Alistair Letcher, Jakob Nicolaus Foerster*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lu23h.html](https://proceedings.mlr.press/v202/lu23h.html)

**Abstract**:

Adversarial attacks in reinforcement learning (RL) often assume highly-privileged access to the victim’s parameters, environment, or data. Instead, this paper proposes a novel adversarial setting called a Cheap Talk MDP in which an Adversary can merely append deterministic messages to the Victim’s observation, resulting in a minimal range of influence. The Adversary cannot occlude ground truth, influence underlying environment dynamics or reward signals, introduce non-stationarity, add stochasticity, see the Victim’s actions, or access their parameters. Additionally, we present a simple meta-learning algorithm called Adversarial Cheap Talk (ACT) to train Adversaries in this setting. We demonstrate that an Adversary trained with ACT can still significantly influence the Victim’s training and testing performance, despite the highly constrained setting. Affecting train-time performance reveals a new attack vector and provides insight into the success and failure modes of existing RL algorithms. More specifically, we show that an ACT Adversary is capable of harming performance by interfering with the learner’s function approximation, or instead helping the Victim’s performance by outputting useful features. Finally, we show that an ACT Adversary can manipulate messages during train-time to directly and arbitrarily control the Victim at test-time.

----

## [952] Federated Conformal Predictors for Distributed Uncertainty Quantification

**Authors**: *Charles Lu, Yaodong Yu, Sai Praneeth Karimireddy, Michael I. Jordan, Ramesh Raskar*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lu23i.html](https://proceedings.mlr.press/v202/lu23i.html)

**Abstract**:

Conformal prediction is emerging as a popular paradigm for providing rigorous uncertainty quantification in machine learning since it can be easily applied as a post-processing step to already trained models. In this paper, we extend conformal prediction to the federated learning setting. The main challenge we face is data heterogeneity across the clients — this violates the fundamental tenet of exchangeability required for conformal prediction. We propose a weaker notion of partial exchangeability, better suited to the FL setting, and use it to develop the Federated Conformal Prediction (FCP) framework. We show FCP enjoys rigorous theoretical guarantees and excellent empirical performance on several computer vision and medical imaging datasets. Our results demonstrate a practical approach to incorporating meaningful uncertainty quantification in distributed and heterogeneous environments. We provide code used in our experiments https://github.com/clu5/federated-conformal.

----

## [953] Mechanistic Mode Connectivity

**Authors**: *Ekdeep Singh Lubana, Eric J. Bigelow, Robert P. Dick, David Scott Krueger, Hidenori Tanaka*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lubana23a.html](https://proceedings.mlr.press/v202/lubana23a.html)

**Abstract**:

We study neural network loss landscapes through the lens of mode connectivity, the observation that minimizers of neural networks retrieved via training on a dataset are connected via simple paths of low loss. Specifically, we ask the following question: are minimizers that rely on different mechanisms for making their predictions connected via simple paths of low loss? We provide a definition of mechanistic similarity as shared invariances to input transformations and demonstrate that lack of linear connectivity between two models implies they use dissimilar mechanisms for making their predictions. Relevant to practice, this result helps us demonstrate that naive fine-tuning on a downstream dataset can fail to alter a model’s mechanisms, e.g., fine-tuning can fail to eliminate a model’s reliance on spurious attributes. Our analysis also motivates a method for targeted alteration of a model’s mechanisms, named connectivity-based fine-tuning (CBFT), which we analyze using several synthetic datasets for the task of reducing a model’s reliance on spurious attributes.

----

## [954] A Unifying Framework to the Analysis of Interaction Methods using Synergy Functions

**Authors**: *Daniel Lundström, Meisam Razaviyayn*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lundstrom23a.html](https://proceedings.mlr.press/v202/lundstrom23a.html)

**Abstract**:

Deep learning has revolutionized many areas of machine learning, from computer vision to natural language processing, but these high-performance models are generally “black box." Explaining such models would improve transparency and trust in AI-powered decision making and is necessary for understanding other practical needs such as robustness and fairness. A popular means of enhancing model transparency is to quantify how individual inputs contribute to model outputs (called attributions) and the magnitude of interactions between groups of inputs. A growing number of these methods import concepts and results from game theory to produce attributions and interactions. This work presents a unifying framework for game-theory-inspired attribution and $k^\text{th}$-order interaction methods. We show that, given modest assumptions, a unique full account of interactions between features, called synergies, is possible in the continuous input setting. We identify how various methods are characterized by their policy of distributing synergies. We establish that gradient-based methods are characterized by their actions on monomials, a type of synergy function, and introduce unique gradient-based methods. We show that the combination of various criteria uniquely defines the attribution/interaction methods. Thus, the community needs to identify goals and contexts when developing and employing attribution and interaction methods.

----

## [955] SegCLIP: Patch Aggregation with Learnable Centers for Open-Vocabulary Semantic Segmentation

**Authors**: *Huaishao Luo, Junwei Bao, Youzheng Wu, Xiaodong He, Tianrui Li*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/luo23a.html](https://proceedings.mlr.press/v202/luo23a.html)

**Abstract**:

Recently, the contrastive language-image pre-training, e.g., CLIP, has demonstrated promising results on various downstream tasks. The pre-trained model can capture enriched visual concepts for images by learning from a large scale of text-image data. However, transferring the learned visual knowledge to open-vocabulary semantic segmentation is still under-explored. In this paper, we propose a CLIP-based model named SegCLIP for the topic of open-vocabulary segmentation in an annotation-free manner. The SegCLIP achieves segmentation based on ViT and the main idea is to gather patches with learnable centers to semantic regions through training on text-image pairs. The gathering operation can dynamically capture the semantic groups, which can be used to generate the final segmentation results. We further propose a reconstruction loss on masked patches and a superpixel-based KL loss with pseudo-labels to enhance the visual representation. Experimental results show that our model achieves comparable or superior segmentation accuracy on the PASCAL VOC 2012 (+0.3% mIoU), PASCAL Context (+2.3% mIoU), and COCO (+2.2% mIoU) compared with baselines. We release the code at https://github.com/ArrowLuo/SegCLIP.

----

## [956] Image Restoration with Mean-Reverting Stochastic Differential Equations

**Authors**: *Ziwei Luo, Fredrik K. Gustafsson, Zheng Zhao, Jens Sjölund, Thomas B. Schön*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/luo23b.html](https://proceedings.mlr.press/v202/luo23b.html)

**Abstract**:

This paper presents a stochastic differential equation (SDE) approach for general-purpose image restoration. The key construction consists in a mean-reverting SDE that transforms a high-quality image into a degraded counterpart as a mean state with fixed Gaussian noise. Then, by simulating the corresponding reverse-time SDE, we are able to restore the origin of the low-quality image without relying on any task-specific prior knowledge. Crucially, the proposed mean-reverting SDE has a closed-form solution, allowing us to compute the ground truth time-dependent score and learn it with a neural network. Moreover, we propose a maximum likelihood objective to learn an optimal reverse trajectory that stabilizes the training and improves the restoration results. The experiments show that our proposed method achieves highly competitive performance in quantitative comparisons on image deraining, deblurring, and denoising, setting a new state-of-the-art on two deraining datasets. Finally, the general applicability of our approach is further demonstrated via qualitative results on image super-resolution, inpainting, and dehazing. Code is available at https://github.com/Algolzw/image-restoration-sde.

----

## [957] Dimensionality Reduction for General KDE Mode Finding

**Authors**: *Xinyu Luo, Christopher Musco, Cas Widdershoven*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/luo23c.html](https://proceedings.mlr.press/v202/luo23c.html)

**Abstract**:

Finding the mode of a high dimensional probability distribution $\mathcal{D}$ is a fundamental algorithmic problem in statistics and data analysis. There has been particular interest in efficient methods for solving the problem when $\mathcal{D}$ is represented as a mixture model or kernel density estimate, although few algorithmic results with worst-case approximation and runtime guarantees are known. In this work, we significantly generalize a result of (LeeLiMusco:2021) on mode approximation for Gaussian mixture models. We develop randomized dimensionality reduction methods for mixtures involving a broader class of kernels, including the popular logistic, sigmoid, and generalized Gaussian kernels. As in Lee et al.’s work, our dimensionality reduction results yield quasi-polynomial algorithms for mode finding with multiplicative accuracy $(1-\epsilon)$ for any $\epsilon > 0$. Moreover, when combined with gradient descent, they yield efficient practical heuristics for the problem. In addition to our positive results, we prove a hardness result for box kernels, showing that there is no polynomial time algorithm for finding the mode of a kernel density estimate, unless $\mathit{P} = \mathit{NP}$. Obtaining similar hardness results for kernels used in practice (like Gaussian or logistic kernels) is an interesting future direction.

----

## [958] Iterative Approximate Cross-Validation

**Authors**: *Yuetian Luo, Zhimei Ren, Rina Barber*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/luo23d.html](https://proceedings.mlr.press/v202/luo23d.html)

**Abstract**:

Cross-validation (CV) is one of the most popular tools for assessing and selecting predictive models. However, standard CV suffers from high computational cost when the number of folds is large. Recently, under the empirical risk minimization (ERM) framework, a line of works proposed efficient methods to approximate CV based on the solution of the ERM problem trained on the full dataset. However, in large-scale problems, it can be hard to obtain the exact solution of the ERM problem, either due to limited computational resources or due to early stopping as a way of preventing overfitting. In this paper, we propose a new paradigm to efficiently approximate CV when the ERM problem is solved via an iterative first-order algorithm, without running until convergence. Our new method extends existing guarantees for CV approximation to hold along the whole trajectory of the algorithm, including at convergence, thus generalizing existing CV approximation methods. Finally, we illustrate the accuracy and computational efficiency of our method through a range of empirical studies.

----

## [959] A Closer Look at Few-shot Classification Again

**Authors**: *Xu Luo, Hao Wu, Ji Zhang, Lianli Gao, Jing Xu, Jingkuan Song*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/luo23e.html](https://proceedings.mlr.press/v202/luo23e.html)

**Abstract**:

Few-shot classification consists of a training phase where a model is learned on a relatively large dataset and an adaptation phase where the learned model is adapted to previously-unseen tasks with limited labeled samples. In this paper, we empirically prove that the training algorithm and the adaptation algorithm can be completely disentangled, which allows algorithm analysis and design to be done individually for each phase. Our meta-analysis for each phase reveals several interesting insights that may help better understand key aspects of few-shot classification and connections with other fields such as visual representation learning and transfer learning. We hope the insights and research challenges revealed in this paper can inspire future work in related directions. Code and pre-trained models (in PyTorch) are available at https://github.com/Frankluox/CloserLookAgainFewShot.

----

## [960] HOPE: High-order Graph ODE For Modeling Interacting Dynamics

**Authors**: *Xiao Luo, Jingyang Yuan, Zijie Huang, Huiyu Jiang, Yifang Qin, Wei Ju, Ming Zhang, Yizhou Sun*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/luo23f.html](https://proceedings.mlr.press/v202/luo23f.html)

**Abstract**:

Leading graph ordinary differential equation (ODE) models have offered generalized strategies to model interacting multi-agent dynamical systems in a data-driven approach. They typically consist of a temporal graph encoder to get the initial states and a neural ODE-based generative model to model the evolution of dynamical systems. However, existing methods have severe deficiencies in capacity and efficiency due to the failure to model high-order correlations in long-term temporal trends. To tackle this, in this paper, we propose a novel model named High-order graph ODE (HOPE) for learning from dynamic interaction data, which can be naturally represented as a graph. It first adopts a twin graph encoder to initialize the latent state representations of nodes and edges, which consists of two branches to capture spatio-temporal correlations in complementary manners. More importantly, our HOPE utilizes a second-order graph ODE function which models the dynamics for both nodes and edges in the latent space respectively, which enables efficient learning of long-term dependencies from complex dynamical systems. Experiment results on a variety of datasets demonstrate both the effectiveness and efficiency of our proposed method.

----

## [961] Stabilizing GANs' Training with Brownian Motion Controller

**Authors**: *Tianjiao Luo, Ziyu Zhu, Jianfei Chen, Jun Zhu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/luo23g.html](https://proceedings.mlr.press/v202/luo23g.html)

**Abstract**:

The training process of generative adversarial networks (GANs) is unstable and does not converge globally. In this paper, we examine the stability of GANs from the perspective of control theory and propose a universal higher-order noise-based controller called Brownian Motion Controller (BMC). Starting with the prototypical case of Dirac-GANs, we design a BMC to retrieve precisely the same but reachable optimal equilibrium. We theoretically prove that the training process of DiracGANs-BMC is globally exponential stable and derive bounds on the rate of convergence. Then we extend our BMC to normal GANs and provide implementation instructions on GANs-BMC. Our experiments show that our GANs-BMC effectively stabilizes GANs’ training under StyleGANv2-ada frameworks with a faster rate of convergence, a smaller range of oscillation, and better performance in terms of FID score.

----

## [962] OCD: Learning to Overfit with Conditional Diffusion Models

**Authors**: *Shahar Lutati, Lior Wolf*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lutati23a.html](https://proceedings.mlr.press/v202/lutati23a.html)

**Abstract**:

We present a dynamic model in which the weights are conditioned on an input sample x and are learned to match those that would be obtained by finetuning a base model on x and its label y. This mapping between an input sample and network weights is approximated by a denoising diffusion model. The diffusion model we employ focuses on modifying a single layer of the base model and is conditioned on the input, activations, and output of this layer. Since the diffusion model is stochastic in nature, multiple initializations generate different networks, forming an ensemble, which leads to further improvements. Our experiments demonstrate the wide applicability of the method for image classification, 3D reconstruction, tabular data, speech separation, and natural language processing.

----

## [963] DiscoBAX: Discovery of optimal intervention sets in genomic experiment design

**Authors**: *Clare Lyle, Arash Mehrjou, Pascal Notin, Andrew Jesson, Stefan Bauer, Yarin Gal, Patrick Schwab*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lyle23a.html](https://proceedings.mlr.press/v202/lyle23a.html)

**Abstract**:

The discovery of therapeutics to treat genetically-driven pathologies relies on identifying genes involved in the underlying disease mechanism. Existing approaches search over the billions of potential interventions to maximize the expected influence on the target phenotype. However, to reduce the risk of failure in future stages of trials, practical experiment design aims to find a set of interventions that maximally change a target phenotype via diverse mechanisms. We propose DiscoBAX - a sample-efficient method for maximizing the rate of significant discoveries per experiment while simultaneously probing for a wide range of diverse mechanisms during a genomic experiment campaign. We provide theoretical guarantees of optimality under standard assumptions, and conduct a comprehensive experimental evaluation covering both synthetic as well as real-world experimental design tasks. DiscoBAX outperforms existing state-of-the-art methods for experimental design, selecting effective and diverse perturbations in biological systems.

----

## [964] Understanding Plasticity in Neural Networks

**Authors**: *Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Ávila Pires, Razvan Pascanu, Will Dabney*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lyle23b.html](https://proceedings.mlr.press/v202/lyle23b.html)

**Abstract**:

Plasticity, the ability of a neural network to quickly change its predictions in response to new information, is essential for the adaptability and robustness of deep reinforcement learning systems. Deep neural networks are known to lose plasticity over the course of training even in relatively simple learning problems, but the mechanisms driving this phenomenon are still poorly understood. This paper conducts a systematic empirical analysis into plasticity loss, with the goal of understanding the phenomenon mechanistically in order to guide the future development of targeted solutions. We find that loss of plasticity is deeply connected to changes in the curvature of the loss landscape, but that it often occurs in the absence of saturated units. Based on this insight, we identify a number of parameterization and optimization design choices which enable networks to better preserve plasticity over the course of training. We validate the utility of these findings on larger-scale RL benchmarks in the Arcade Learning Environment.

----

## [965] Bandits with Knapsacks: Advice on Time-Varying Demands

**Authors**: *Lixing Lyu, Wang Chi Cheung*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lyu23a.html](https://proceedings.mlr.press/v202/lyu23a.html)

**Abstract**:

We consider a non-stationary Bandits with Knapsack problem. The outcome distribution at each time is scaled by a non-stationary quantity that signifies changing demand volumes. Instead of studying settings with limited non-stationarity, we investigate how online predictions on the total demand volume $Q$ allows us to improve our performance guarantees. We show that, without any prediction, any online algorithm incurs a linear-in-$T$ regret. In contrast, with online predictions on $Q$, we propose an online algorithm that judiciously incorporates the predictions, and achieve regret bounds that depends on the accuracy of the predictions. These bounds are shown to be tight in settings when prediction accuracy improves across time. Our theoretical results are corroborated by our numerical findings.

----

## [966] Pairwise Ranking Losses of Click-Through Rates Prediction for Welfare Maximization in Ad Auctions

**Authors**: *Boxiang Lyu, Zhe Feng, Zachary Robertson, Sanmi Koyejo*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lyu23b.html](https://proceedings.mlr.press/v202/lyu23b.html)

**Abstract**:

We study the design of loss functions for click-through rates (CTR) to optimize (social) welfare in advertising auctions. Existing works either only focus on CTR predictions without consideration of business objectives (e.g., welfare) in auctions or assume that the distribution over the participants’ expected cost-per-impression (eCPM) is known a priori, then use various additional assumptions on the parametric form of the distribution to derive loss functions for predicting CTRs. In this work, we bring back the welfare objectives of ad auctions into CTR predictions and propose a novel weighted rankloss to train the CTR model. Compared to existing literature, our approach provides a provable guarantee on welfare but without assumptions on the eCPMs’ distribution while also avoiding the intractability of naively applying existing learning-to-rank methods. Further, we propose a theoretically justifiable technique for calibrating the losses using labels generated from a teacher network, only assuming that the teacher network has bounded $\ell_2$ generalization error. Finally, we demonstrate the advantages of the proposed loss on synthetic and real-world data.

----

## [967] Which Tricks are Important for Learning to Rank?

**Authors**: *Ivan Lyzhin, Aleksei Ustimenko, Andrey Gulin, Liudmila Prokhorenkova*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/lyzhin23a.html](https://proceedings.mlr.press/v202/lyzhin23a.html)

**Abstract**:

Nowadays, state-of-the-art learning-to-rank methods are based on gradient-boosted decision trees (GBDT). The most well-known algorithm is LambdaMART which was proposed more than a decade ago. Recently, several other GBDT-based ranking algorithms were proposed. In this paper, we thoroughly analyze these methods in a unified setup. In particular, we address the following questions. Is direct optimization of a smoothed ranking loss preferable over optimizing a convex surrogate? How to properly construct and smooth surrogate ranking losses? To address these questions, we compare LambdaMART with YetiRank and StochasticRank methods and their modifications. We also propose a simple improvement of the YetiRank approach that allows for optimizing specific ranking loss functions. As a result, we gain insights into learning-to-rank techniques and obtain a new state-of-the-art algorithm.

----

## [968] Learning Neural Constitutive Laws from Motion Observations for Generalizable PDE Dynamics

**Authors**: *Pingchuan Ma, Peter Yichen Chen, Bolei Deng, Joshua B. Tenenbaum, Tao Du, Chuang Gan, Wojciech Matusik*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ma23a.html](https://proceedings.mlr.press/v202/ma23a.html)

**Abstract**:

We propose a hybrid neural network (NN) and PDE approach for learning generalizable PDE dynamics from motion observations. Many NN approaches learn an end-to-end model that implicitly models both the governing PDE and constitutive models (or material models). Without explicit PDE knowledge, these approaches cannot guarantee physical correctness and have limited generalizability. We argue that the governing PDEs are often well-known and should be explicitly enforced rather than learned. Instead, constitutive models are particularly suitable for learning due to their data-fitting nature. To this end, we introduce a new framework termed "Neural Constitutive Laws" (NCLaw), which utilizes a network architecture that strictly guarantees standard constitutive priors, including rotation equivariance and undeformed state equilibrium. We embed this network inside a differentiable simulation and train the model by minimizing a loss function based on the difference between the simulation and the motion observation. We validate NCLaw on various large-deformation dynamical systems, ranging from solids to fluids. After training on a single motion trajectory, our method generalizes to new geometries, initial/boundary conditions, temporal ranges, and even multi-physics systems. On these extremely out-of-distribution generalization tasks, NCLaw is orders-of-magnitude more accurate than previous NN approaches. Real-world experiments demonstrate our method’s ability to learn constitutive laws from videos.

----

## [969] LIV: Language-Image Representations and Rewards for Robotic Control

**Authors**: *Yecheng Jason Ma, Vikash Kumar, Amy Zhang, Osbert Bastani, Dinesh Jayaraman*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ma23b.html](https://proceedings.mlr.press/v202/ma23b.html)

**Abstract**:

We present Language-Image Value learning (LIV), a unified objective for vision-language representation and reward learning from action-free videos with text annotations. Exploiting a novel connection between dual reinforcement learning and mutual information contrastive learning, the LIV objective trains a multi-modal representation that implicitly encodes a universal value function for tasks specified as language or image goals. We use LIV to pre-train the first control-centric vision-language representation from large human video datasets such as EpicKitchen. Given only a language or image goal, the pre-trained LIV model can assign dense rewards to each frame in videos of unseen robots or humans attempting that task in unseen environments. Further, when some target domain-specific data is available, the same objective can be used to fine-tune and improve LIV and even other pre-trained representations for robotic control and reward specification in that domain. In our experiments on several simulated and real-world robot environments, LIV models consistently outperform the best prior input state representations for imitation learning, as well as reward specification methods for policy synthesis. Our results validate the advantages of joint vision-language representation and reward learning within the unified, compact LIV framework.

----

## [970] Graph Inductive Biases in Transformers without Message Passing

**Authors**: *Liheng Ma, Chen Lin, Derek Lim, Adriana Romero-Soriano, Puneet K. Dokania, Mark Coates, Philip H. S. Torr, Ser-Nam Lim*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ma23c.html](https://proceedings.mlr.press/v202/ma23c.html)

**Abstract**:

Transformers for graph data are increasingly widely studied and successful in numerous learning tasks. Graph inductive biases are crucial for Graph Transformers, and previous works incorporate them using message-passing modules and/or positional encodings. However, Graph Transformers that use message-passing inherit known issues of message-passing, and differ significantly from Transformers used in other domains, thus making transfer of research advances more difficult. On the other hand, Graph Transformers without message-passing often perform poorly on smaller datasets, where inductive biases are more crucial. To bridge this gap, we propose the Graph Inductive bias Transformer (GRIT) — a new Graph Transformer that incorporates graph inductive biases without using message passing. GRIT is based on several architectural changes that are each theoretically and empirically justified, including: learned relative positional encodings initialized with random walk probabilities, a flexible attention mechanism that updates node and node-pair representations, and injection of degree information in each layer. We prove that GRIT is expressive — it can express shortest path distances and various graph propagation matrices. GRIT achieves state-of-the-art empirical performance across a variety of graph datasets, thus showing the power that Graph Transformers without message-passing can deliver.

----

## [971] Learning Signed Distance Functions from Noisy 3D Point Clouds via Noise to Noise Mapping

**Authors**: *Baorui Ma, Yu-Shen Liu, Zhizhong Han*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ma23d.html](https://proceedings.mlr.press/v202/ma23d.html)

**Abstract**:

Learning signed distance functions (SDFs) from 3D point clouds is an important task in 3D computer vision. However, without ground truth signed distances, point normals or clean point clouds, current methods still struggle from learning SDFs from noisy point clouds. To overcome this challenge, we propose to learn SDFs via a noise to noise mapping, which does not require any clean point cloud or ground truth supervision for training. Our novelty lies in the noise to noise mapping which can infer a highly accurate SDF of a single object or scene from its multiple or even single noisy point cloud observations. Our novel learning manner is supported by modern Lidar systems which capture multiple noisy observations per second. We achieve this by a novel loss which enables statistical reasoning on point clouds and maintains geometric consistency although point clouds are irregular, unordered and have no point correspondence among noisy observations. Our evaluation under the widely used benchmarks demonstrates our superiority over the state-of-the-art methods in surface reconstruction, point cloud denoising and upsampling. Our code, data, and pre-trained models are available at https://github.com/mabaorui/Noise2NoiseMapping/ .

----

## [972] Learning Intuitive Policies Using Action Features

**Authors**: *Mingwei Ma, Jizhou Liu, Samuel Sokota, Max Kleiman-Weiner, Jakob Nicolaus Foerster*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ma23e.html](https://proceedings.mlr.press/v202/ma23e.html)

**Abstract**:

An unaddressed challenge in multi-agent coordination is to enable AI agents to exploit the semantic relationships between the features of actions and the features of observations. Humans take advantage of these relationships in highly intuitive ways. For instance, in the absence of a shared language, we might point to the object we desire or hold up our fingers to indicate how many objects we want. To address this challenge, we investigate the effect of network architecture on the propensity of learning algorithms to exploit these semantic relationships. Across a procedurally generated coordination task, we find that attention-based architectures that jointly process a featurized representation of observations and actions have a better inductive bias for learning intuitive policies. Through fine-grained evaluation and scenario analysis, we show that the resulting policies are human-interpretable. Moreover, such agents coordinate with people without training on any human data.

----

## [973] Over-parametrization via Lifting for Low-rank Matrix Sensing: Conversion of Spurious Solutions to Strict Saddle Points

**Authors**: *Ziye Ma, Igor Molybog, Javad Lavaei, Somayeh Sojoudi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ma23f.html](https://proceedings.mlr.press/v202/ma23f.html)

**Abstract**:

This paper studies the role of over-parametrization in solving non-convex optimization problems. The focus is on the important class of low-rank matrix sensing, where we propose an infinite hierarchy of non-convex problems via the lifting technique and the Burer-Monteiro factorization. This contrasts with the existing over-parametrization technique where the search rank is limited by the dimension of the matrix and it does not allow a rich over-parametrization of an arbitrary degree. We show that although the spurious solutions of the problem remain stationary points through the hierarchy, they will be transformed into strict saddle points (under some technical conditions) and can be escaped via local search methods. This is the first result in the literature showing that over-parametrization creates a negative curvature for escaping spurious solutions. We also derive a bound on how much over-parametrization is requited to enable the elimination of spurious solutions.

----

## [974] Buying Information for Stochastic Optimization

**Authors**: *Mingchen Ma, Christos Tzamos*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ma23g.html](https://proceedings.mlr.press/v202/ma23g.html)

**Abstract**:

Stochastic optimization is one of the central problems in Machine Learning and Theoretical Computer Science. In the standard model, the algorithm is given a fixed distribution known in advance. In practice though, one may acquire at a cost extra information to make better decisions. In this paper, we study how to buy information for stochastic optimization and formulate this question as an online learning problem. Assuming the learner has an oracle for the original optimization problem, we design a $2$-competitive deterministic algorithm and a $e/(e-1)$-competitive randomized algorithm for buying information. We show that this ratio is tight as the problem is equivalent to a robust generalization of the ski-rental problem, which we call super-martingale stopping. We also consider an adaptive setting where the learner can choose to buy information after taking some actions for the underlying optimization problem. We focus on the classic optimization problem, Min-Sum Set Cover, where the goal is to quickly find an action that covers a given request drawn from a known distribution. We provide an $8$-competitive algorithm running in polynomial time that chooses actions and decides when to buy information about the underlying request.

----

## [975] Generated Graph Detection

**Authors**: *Yihan Ma, Zhikun Zhang, Ning Yu, Xinlei He, Michael Backes, Yun Shen, Yang Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ma23h.html](https://proceedings.mlr.press/v202/ma23h.html)

**Abstract**:

Graph generative models become increasingly effective for data distribution approximation and data augmentation. While they have aroused public concerns about their malicious misuses or misinformation broadcasts, just as what Deepfake visual and auditory media has been delivering to society. Hence it is essential to regulate the prevalence of generated graphs. To tackle this problem, we pioneer the formulation of the generated graph detection problem to distinguish generated graphs from real ones. We propose the first framework to systematically investigate a set of sophisticated models and their performance in four classification scenarios. Each scenario switches between seen and unseen datasets/generators during testing to get closer to real-world settings and progressively challenge the classifiers. Extensive experiments evidence that all the models are qualified for generated graph detection, with specific models having advantages in specific scenarios. Resulting from the validated generality and oblivion of the classifiers to unseen datasets/generators, we draw a safe conclusion that our solution can sustain for a decent while to curb generated graph misuses.

----

## [976] Calibrating Multimodal Learning

**Authors**: *Huan Ma, Qingyang Zhang, Changqing Zhang, Bingzhe Wu, Huazhu Fu, Joey Tianyi Zhou, Qinghua Hu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ma23i.html](https://proceedings.mlr.press/v202/ma23i.html)

**Abstract**:

Multimodal machine learning has achieved remarkable progress in a wide range of scenarios. However, the reliability of multimodal learning remains largely unexplored. In this paper, through extensive empirical studies, we identify current multimodal classification methods suffer from unreliable predictive confidence that tend to rely on partial modalities when estimating confidence. Specifically, we find that the confidence estimated by current models could even increase when some modalities are corrupted. To address the issue, we introduce an intuitive principle for multimodal learning, i.e., the confidence should not increase when one modality is removed. Accordingly, we propose a novel regularization technique, i.e., Calibrating Multimodal Learning (CML) regularization, to calibrate the predictive confidence of previous methods. This technique could be flexibly equipped by existing models and improve the performance in terms of confidence calibration, classification accuracy, and model robustness.

----

## [977] AutoCoreset: An Automatic Practical Coreset Construction Framework

**Authors**: *Alaa Maalouf, Murad Tukan, Vladimir Braverman, Daniela Rus*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/maalouf23a.html](https://proceedings.mlr.press/v202/maalouf23a.html)

**Abstract**:

A coreset is a small weighted subset of an input set that approximates its loss function, for a given set of queries. Coresets became prevalent in machine learning as they have shown to be advantageous for many applications. Unfortunately, coresets are constructed in a problem-dependent manner, where for each problem, a new coreset construction algorithm is suggested, taking years to prove its correctness. Even the generic frameworks require additional (problem-dependent) computations or proofs to be done by the user. Besides, many problems do not have (provable) small coresets, limiting their applicability. To this end, we suggest an automatic practical framework for constructing coresets, which requires (only) the input data and the desired cost function from the user, without the need for any other task-related computation to be done by the user. To do so, we reduce the problem of approximating a loss function to an instance of vector summation approximation, where the vectors we aim to sum are loss vectors of a specific subset of the queries, such that we aim to approximate the image of the function on this subset. We show that while this set is limited, the coreset is quite general. An extensive experimental study on various machine learning applications is also conducted. Finally, we provide a “plug and play" style implementation, proposing a user-friendly system that can be easily used to apply coresets for many problems. We believe that these contributions enable future research and easier use and applications of coresets.

----

## [978] Learning GFlowNets From Partial Episodes For Improved Convergence And Stability

**Authors**: *Kanika Madan, Jarrid Rector-Brooks, Maksym Korablyov, Emmanuel Bengio, Moksh Jain, Andrei Cristian Nica, Tom Bosc, Yoshua Bengio, Nikolay Malkin*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/madan23a.html](https://proceedings.mlr.press/v202/madan23a.html)

**Abstract**:

Generative flow networks (GFlowNets) are a family of algorithms for training a sequential sampler of discrete objects under an unnormalized target density and have been successfully used for various probabilistic modeling tasks. Existing training objectives for GFlowNets are either local to states or transitions, or propagate a reward signal over an entire sampling trajectory. We argue that these alternatives represent opposite ends of a gradient bias-variance tradeoff and propose a way to exploit this tradeoff to mitigate its harmful effects. Inspired by the TD($\lambda$) algorithm in reinforcement learning, we introduce subtrajectory balance or SubTB($\lambda$), a GFlowNet training objective that can learn from partial action subsequences of varying lengths. We show that SubTB($\lambda$) accelerates sampler convergence in previously studied and new environments and enables training GFlowNets in environments with longer action sequences and sparser reward landscapes than what was possible before. We also perform a comparative analysis of stochastic gradient dynamics, shedding light on the bias-variance tradeoff in GFlowNet training and the advantages of subtrajectory balance.

----

## [979] Applied Online Algorithms with Heterogeneous Predictors

**Authors**: *Jessica Maghakian, Russell Lee, Mohammad Hajiesmaili, Jian Li, Ramesh K. Sitaraman, Zhenhua Liu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/maghakian23a.html](https://proceedings.mlr.press/v202/maghakian23a.html)

**Abstract**:

For many application domains, the integration of machine learning (ML) models into decision making is hindered by the poor explainability and theoretical guarantees of black box models. Although the emerging area of algorithms with predictions offers a way to leverage ML while enjoying worst-case guarantees, existing work usually assumes access to only one predictor. We demonstrate how to more effectively utilize historical datasets and application domain knowledge by intentionally using predictors of different quantities. By leveraging the heterogeneity in our predictors, we are able to achieve improved performance, explainability and computational efficiency over predictor-agnostic methods. Theoretical results are supplemented by large-scale empirical evaluations with production data demonstrating the success of our methods on optimization problems occurring in large distributed computing systems.

----

## [980] CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations

**Authors**: *Gengchen Mai, Ni Lao, Yutong He, Jiaming Song, Stefano Ermon*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mai23a.html](https://proceedings.mlr.press/v202/mai23a.html)

**Abstract**:

Geo-tagged images are publicly available in large quantities, whereas labels such as object classes are rather scarce and expensive to collect. Meanwhile, contrastive learning has achieved tremendous success in various natural image and language tasks with limited labeled data. However, existing methods fail to fully leverage geospatial information, which can be paramount to distinguishing objects that are visually similar. To directly leverage the abundant geospatial information associated with images in pre-training, fine-tuning, and inference stages, we present Contrastive Spatial Pre-Training (CSP), a self-supervised learning framework for geo-tagged images. We use a dual-encoder to separately encode the images and their corresponding geo-locations, and use contrastive objectives to learn effective location representations from images, which can be transferred to downstream supervised tasks such as image classification. Experiments show that CSP can improve model performance on both iNat2018 and fMoW datasets. Especially, on iNat2018, CSP significantly boosts the model performance with 10-34% relative improvement with various labeled training data sampling ratios.

----

## [981] Vertical Federated Graph Neural Network for Recommender System

**Authors**: *Peihua Mai, Yan Pang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mai23b.html](https://proceedings.mlr.press/v202/mai23b.html)

**Abstract**:

Conventional recommender systems are required to train the recommendation model using a centralized database. However, due to data privacy concerns, this is often impractical when multi-parties are involved in recommender system training. Federated learning appears as an excellent solution to the data isolation and privacy problem. Recently, Graph neural network (GNN) is becoming a promising approach for federated recommender systems. However, a key challenge is to conduct embedding propagation while preserving the privacy of the graph structure. Few studies have been conducted on the federated GNN-based recommender system. Our study proposes the first vertical federated GNN-based recommender system, called VerFedGNN. We design a framework to transmit: (i) the summation of neighbor embeddings using random projection, and (ii) gradients of public parameter perturbed by ternary quantization mechanism. Empirical studies show that VerFedGNN has competitive prediction accuracy with existing privacy preserving GNN frameworks while enhanced privacy protection for users’ interaction information.

----

## [982] Can Neural Network Memorization Be Localized?

**Authors**: *Pratyush Maini, Michael Curtis Mozer, Hanie Sedghi, Zachary Chase Lipton, J. Zico Kolter, Chiyuan Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/maini23a.html](https://proceedings.mlr.press/v202/maini23a.html)

**Abstract**:

Recent efforts at explaining the interplay of memorization and generalization in deep overparametrized networks have posited that neural networks memorize “hard” examples in the final few layers of the model. Memorization refers to the ability to correctly predict on atypical examples of the training set. In this work, we show that rather than being confined to individual layers, memorization is a phenomenon confined to a small set of neurons in various layers of the model. First, via three experimental sources of converging evidence, we find that most layers are redundant for the memorization of examples and the layers that contribute to example memorization are, in general, not the final layers. The three sources are gradient accounting (measuring the contribution to the gradient norms from memorized and clean examples), layer rewinding (replacing specific model weights of a converged model with previous training checkpoints), and retraining (training rewound layers only on clean examples). Second, we ask a more generic question: can memorization be localized anywhere in a model? We discover that memorization is often confined to a small number of neurons or channels (around 5) of the model. Based on these insights we propose a new form of dropout—example-tied dropout that enables us to direct the memorization of examples to an aprior determined set of neurons. By dropping out these neurons, we are able to reduce the accuracy on memorized examples from 100% to 3%, while also reducing the generalization gap.

----

## [983] Fundamental Tradeoffs in Learning with Prior Information

**Authors**: *Anirudha Majumdar*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/majumdar23a.html](https://proceedings.mlr.press/v202/majumdar23a.html)

**Abstract**:

We seek to understand fundamental tradeoffs between the accuracy of prior information that a learner has on a given problem and its learning performance. We introduce the notion of prioritized risk, which differs from traditional notions of minimax and Bayes risk by allowing us to study such fundamental tradeoffs in settings where reality does not necessarily conform to the learner’s prior. We present a general reduction-based approach for extending classical minimax lower-bound techniques in order to lower bound the prioritized risk for statistical estimation problems. We also introduce a novel generalization of Fano’s inequality (which may be of independent interest) for lower bounding the prioritized risk in more general settings involving unbounded losses. We illustrate the ability of our framework to provide insights into tradeoffs between prior information and learning performance for problems in estimation, regression, and reinforcement learning.

----

## [984] Additive Causal Bandits with Unknown Graph

**Authors**: *Alan Malek, Virginia Aglietti, Silvia Chiappa*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/malek23a.html](https://proceedings.mlr.press/v202/malek23a.html)

**Abstract**:

We explore algorithms to select actions in the causal bandit setting where the learner can choose to intervene on a set of random variables related by a causal graph, and the learner sequentially chooses interventions and observes a sample from the interventional distribution. The learner’s goal is to quickly find the intervention, among all interventions on observable variables, that maximizes the expectation of an outcome variable. We depart from previous literature by assuming no knowledge of the causal graph except that latent confounders between the outcome and its ancestors are not present. We first show that the unknown graph problem can be exponentially hard in the parents of the outcome. To remedy this, we adopt an additional additive assumption on the outcome which allows us to solve the problem by casting it as an additive combinatorial linear bandit problem with full-bandit feedback. We propose a novel action-elimination algorithm for this setting, show how to apply this algorithm to the causal bandit problem, provide sample complexity bounds, and empirically validate our findings on a suite of randomly generated causal models, effectively showing that one does not need to explicitly learn the parents of the outcome to identify the best intervention.

----

## [985] Weighted Tallying Bandits: Overcoming Intractability via Repeated Exposure Optimality

**Authors**: *Dhruv Malik, Conor Igoe, Yuanzhi Li, Aarti Singh*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/malik23a.html](https://proceedings.mlr.press/v202/malik23a.html)

**Abstract**:

In human-interactive applications of online learning, a human’s preferences or abilities are often a function of the algorithm’s recent actions. Motivated by this, a significant line of work has formalized settings where an action’s loss is a function of the number of times it was played in the prior $m$ timesteps, where $m$ corresponds to a bound on human memory capacity. To more faithfully capture decay of human memory with time, we introduce the Weighted Tallying Bandit (WTB), which generalizes this setting by requiring that an action’s loss is a function of a weighted summation of the number of times it was played in the last $m$ timesteps. WTB is intractable without further assumption. So we study it under Repeated Exposure Optimality (REO), a condition requiring the existence of an action that when repetitively played will eventually yield smaller loss than any other action sequence. We study the minimization of complete policy regret (CPR), which is the strongest notion of regret, in WTB under REO. Since $m$ is often unknown, we only assume access to an upper bound $M$ on $m$. We show that for problems with $K$ actions and horizon $T$, a simple modification of the successive elimination algorithm has $\mathcal{O} \left( \sqrt{KT} + (m+M)K \right)$ CPR. Upto an additive (in lieu of mutliplicative) factor in $(m+M)K$, this recovers the classical guarantee for the far simpler stochastic multi-armed bandit with traditional regret. We additionally show that in our setting, any algorithm will suffer additive CPR of $\Omega \left( mK + M \right)$, demonstrating our result is near optimal. Our method is computationally efficient, and we experimentally demonstrate its practicality and superiority over various baselines.

----

## [986] A Kernel-Based View of Language Model Fine-Tuning

**Authors**: *Sadhika Malladi, Alexander Wettig, Dingli Yu, Danqi Chen, Sanjeev Arora*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/malladi23a.html](https://proceedings.mlr.press/v202/malladi23a.html)

**Abstract**:

It has become standard to solve NLP tasks by fine-tuning pre-trained language models (LMs), especially in low-data settings. There is minimal theoretical understanding of empirical success, e.g., why fine-tuning a model with $10^8$ or more parameters on a couple dozen training points does not result in overfitting. We investigate whether the Neural Tangent Kernel (NTK)—which originated as a model to study the gradient descent dynamics of infinitely wide networks with suitable random initialization—describes fine-tuning of pre-trained LMs. This study was inspired by the decent performance of NTK for computer vision tasks (Wei et al., 2022). We extend the NTK formalism to Adam and use Tensor Programs (Yang, 2020) to characterize conditions under which the NTK lens may describe fine-tuning updates to pre-trained language models. Extensive experiments on 14 NLP tasks validate our theory and show that formulating the downstream task as a masked word prediction problem through prompting often induces kernel-based dynamics during fine-tuning. Finally, we use this kernel view to propose an explanation for the success of parameter-efficient subspace-based fine-tuning methods.

----

## [987] Performative Reinforcement Learning

**Authors**: *Debmalya Mandal, Stelios Triantafyllou, Goran Radanovic*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mandal23a.html](https://proceedings.mlr.press/v202/mandal23a.html)

**Abstract**:

We introduce the framework of performative reinforcement learning where the policy chosen by the learner affects the underlying reward and transition dynamics of the environment. Following the recent literature on performative prediction (Perdomo et al., 2020), we introduce the concept of performatively stable policy. We then consider a regularized version of the reinforcement learning problem and show that repeatedly optimizing this objective converges to a performatively stable policy under reasonable assumptions on the transition dynamics. Our proof utilizes the dual perspective of the reinforcement learning problem and may be of independent interest in analyzing the convergence of other algorithms with decision-dependent environments. We then extend our results for the setting where the learner just performs gradient ascent steps instead of fully optimizing the objective, and for the setting where the learner has access to a finite number of trajectories from the changed environment. For both the settings, we leverage the dual formulation of performative reinforcement learning, and establish convergence to a stable solution. Finally, through extensive experiments on a grid-world environment, we demonstrate the dependence of convergence on various parameters e.g. regularization, smoothness, and the number of samples.

----

## [988] Differential Privacy has Bounded Impact on Fairness in Classification

**Authors**: *Paul Mangold, Michaël Perrot, Aurélien Bellet, Marc Tommasi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mangold23a.html](https://proceedings.mlr.press/v202/mangold23a.html)

**Abstract**:

We theoretically study the impact of differential privacy on fairness in classification. We prove that, given a class of models, popular group fairness measures are pointwise Lipschitz-continuous with respect to the parameters of the model. This result is a consequence of a more general statement on accuracy conditioned on an arbitrary event (such as membership to a sensitive group), which may be of independent interest. We use this Lipschitz property to prove a non-asymptotic bound showing that, as the number of samples increases, the fairness level of private models gets closer to the one of their non-private counterparts. This bound also highlights the importance of the confidence margin of a model on the disparate impact of differential privacy.

----

## [989] Random Classification Noise does not defeat All Convex Potential Boosters Irrespective of Model Choice

**Authors**: *Yishay Mansour, Richard Nock, Robert C. Williamson*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mansour23a.html](https://proceedings.mlr.press/v202/mansour23a.html)

**Abstract**:

A landmark negative result of Long and Servedio has had a considerable impact on research and development in boosting algorithms, around the now famous tagline that "noise defeats all convex boosters". In this paper, we appeal to the half-century+ founding theory of losses for class probability estimation, an extension of Long and Servedio’s results and a new general convex booster to demonstrate that the source of their negative result is in fact the model class, linear separators. Losses or algorithms are neither to blame. This leads us to a discussion on an otherwise praised aspect of ML, parameterisation.

----

## [990] H-Consistency Bounds for Pairwise Misranking Loss Surrogates

**Authors**: *Anqi Mao, Mehryar Mohri, Yutao Zhong*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mao23a.html](https://proceedings.mlr.press/v202/mao23a.html)

**Abstract**:

We present a detailed study of $H$-consistency bounds for score-based ranking. These are upper bounds on the target loss estimation error of a predictor in a hypothesis set $H$, expressed in terms of the surrogate loss estimation error of that predictor. We will show that both in the general pairwise ranking scenario and in the bipartite ranking scenario, there are no meaningful $H$-consistency bounds for most hypothesis sets used in practice including the family of linear models and that of the neural networks, which satisfy the equicontinuous property with respect to the input. To come up with ranking surrogate losses with theoretical guarantees, we show that a natural solution consists of resorting to a pairwise abstention loss in the general pairwise ranking scenario, and similarly, a bipartite abstention loss in the bipartite ranking scenario, to abstain from making predictions at some limited cost $c$. For surrogate losses of these abstention loss functions, we give a series of $H$-consistency bounds for both the family of linear functions and that of neural networks with one hidden-layer. Our experimental results illustrate the effectiveness of ranking with abstention.

----

## [991] Cross-Entropy Loss Functions: Theoretical Analysis and Applications

**Authors**: *Anqi Mao, Mehryar Mohri, Yutao Zhong*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mao23b.html](https://proceedings.mlr.press/v202/mao23b.html)

**Abstract**:

Cross-entropy is a widely used loss function in applications. It coincides with the logistic loss applied to the outputs of a neural network, when the softmax is used. But, what guarantees can we rely on when using cross-entropy as a surrogate loss? We present a theoretical analysis of a broad family of loss functions, comp-sum losses, that includes cross-entropy (or logistic loss), generalized cross-entropy, the mean absolute error and other cross-entropy-like loss functions. We give the first $H$-consistency bounds for these loss functions. These are non-asymptotic guarantees that upper bound the zero-one loss estimation error in terms of the estimation error of a surrogate loss, for the specific hypothesis set $H$ used. We further show that our bounds are tight. These bounds depend on quantities called minimizability gaps. To make them more explicit, we give a specific analysis of these gaps for comp-sum losses. We also introduce a new family of loss functions, smooth adversarial comp-sum losses, that are derived from their comp-sum counterparts by adding in a related smooth term. We show that these loss functions are beneficial in the adversarial setting by proving that they admit $H$-consistency bounds. This leads to new adversarial robustness algorithms that consist of minimizing a regularized smooth adversarial comp-sum loss. While our main purpose is a theoretical analysis, we also present an extensive empirical analysis comparing comp-sum losses. We further report the results of a series of experiments demonstrating that our adversarial robustness algorithms outperform the current state-of-the-art, while also achieving a superior non-adversarial accuracy.

----

## [992] Supported Trust Region Optimization for Offline Reinforcement Learning

**Authors**: *Yixiu Mao, Hongchang Zhang, Chen Chen, Yi Xu, Xiangyang Ji*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mao23c.html](https://proceedings.mlr.press/v202/mao23c.html)

**Abstract**:

Offline reinforcement learning suffers from the out-of-distribution issue and extrapolation error. Most policy constraint methods regularize the density of the trained policy towards the behavior policy, which is too restrictive in most cases. We propose Supported Trust Region optimization (STR) which performs trust region policy optimization with the policy constrained within the support of the behavior policy, enjoying the less restrictive support constraint. We show that, when assuming no approximation and sampling error, STR guarantees strict policy improvement until convergence to the optimal support-constrained policy in the dataset. Further with both errors incorporated, STR still guarantees safe policy improvement for each step. Empirical results validate the theory of STR and demonstrate its state-of-the-art performance on MuJoCo locomotion domains and much more challenging AntMaze domains.

----

## [993] Robust Perception through Equivariance

**Authors**: *Chengzhi Mao, Lingyu Zhang, Abhishek Vaibhav Joshi, Junfeng Yang, Hao Wang, Carl Vondrick*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mao23d.html](https://proceedings.mlr.press/v202/mao23d.html)

**Abstract**:

Deep networks for computer vision are not reliable when they encounter adversarial examples. In this paper, we introduce a framework that uses the dense intrinsic constraints in natural images to robustify inference. By introducing constraints at inference time, we can shift the burden of robustness from training to testing, thereby allowing the model to dynamically adjust to each individual image’s unique and potentially novel characteristics at inference time. Our theoretical results show the importance of having dense constraints at inference time. In contrast to existing single-constraint methods, we propose to use equivariance, which naturally allows dense constraints at a fine-grained level in the feature space. Our empirical experiments show that restoring feature equivariance at inference time defends against worst-case adversarial perturbations. The method obtains improved adversarial robustness on four datasets (ImageNet, Cityscapes, PASCAL VOC, and MS-COCO) on image recognition, semantic segmentation, and instance segmentation tasks.

----

## [994] Reliable Measures of Spread in High Dimensional Latent Spaces

**Authors**: *Anna C. Marbut, Katy McKinney-Bock, Travis J. Wheeler*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/marbut23a.html](https://proceedings.mlr.press/v202/marbut23a.html)

**Abstract**:

Understanding geometric properties of the latent spaces of natural language processing models allows the manipulation of these properties for improved performance on downstream tasks. One such property is the amount of data spread in a model’s latent space, or how fully the available latent space is being used. We demonstrate that the commonly used measures of data spread, average cosine similarity and a partition function min/max ratio I(V), do not provide reliable metrics to compare the use of latent space across data distributions. We propose and examine six alternative measures of data spread, all of which improve over these current metrics when applied to seven synthetic data distributions. Of our proposed measures, we recommend one principal component-based measure and one entropy-based measure that provide reliable, relative measures of spread and can be used to compare models of different sizes and dimensionalities.

----

## [995] SRATTA: Sample Re-ATTribution Attack of Secure Aggregation in Federated Learning

**Authors**: *Tanguy Marchand, Regis Loeb, Ulysse Marteau-Ferey, Jean Ogier du Terrail, Arthur Pignet*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/marchand23a.html](https://proceedings.mlr.press/v202/marchand23a.html)

**Abstract**:

We consider a federated learning (FL) setting where a machine learning model with a fully connected first layer is trained between different clients and a central server using FedAvg, and where the aggregation step can be performed with secure aggregation (SA). We present SRATTA an attack relying only on aggregated models which, under realistic assumptions, (i) recovers data samples from the different clients, and (ii) groups data samples coming from the same client together. While sample recovery has already been explored in an FL setting, the ability to group samples per client, despite the use of SA, is novel. This poses a significant unforeseen security threat to FL and effectively breaks SA. We show that SRATTA is both theoretically grounded and can be used in practice on realistic models and datasets. We also propose counter-measures, and claim that clients should play an active role to guarantee their privacy during training.

----

## [996] Neuro-Symbolic Continual Learning: Knowledge, Reasoning Shortcuts and Concept Rehearsal

**Authors**: *Emanuele Marconato, Gianpaolo Bontempo, Elisa Ficarra, Simone Calderara, Andrea Passerini, Stefano Teso*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/marconato23a.html](https://proceedings.mlr.press/v202/marconato23a.html)

**Abstract**:

We introduce Neuro-Symbolic Continual Learning, where a model has to solve a sequence of neuro-symbolic tasks, that is, it has to map sub-symbolic inputs to high-level concepts and compute predictions by reasoning consistently with prior knowledge. Our key observation is that neuro-symbolic tasks, although different, often share concepts whose semantics remains stable over time. Traditional approaches fall short: existing continual strategies ignore knowledge altogether, while stock neuro-symbolic architectures suffer from catastrophic forgetting. We show that leveraging prior knowledge by combining neuro-symbolic architectures with continual strategies does help avoid catastrophic forgetting, but also that doing so can yield models affected by reasoning shortcuts. These undermine the semantics of the acquired concepts, even when detailed prior knowledge is provided upfront and inference is exact, and in turn continual performance. To overcome these issues, we introduce COOL, a COncept-level cOntinual Learning strategy tailored for neuro-symbolic continual problems that acquires high-quality concepts and remembers them over time. Our experiments on three novel benchmarks highlights how COOL attains sustained high performance on neuro-symbolic continual learning tasks in which other strategies fail.

----

## [997] Evaluating Unsupervised Denoising Requires Unsupervised Metrics

**Authors**: *Adria Marcos-Morales, Matan Leibovich, Sreyas Mohan, Joshua Lawrence Vincent, Piyush Haluai, Mai Tan, Peter A. Crozier, Carlos Fernandez-Granda*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/marcos-morales23a.html](https://proceedings.mlr.press/v202/marcos-morales23a.html)

**Abstract**:

Unsupervised denoising is a crucial challenge in real-world imaging applications. Unsupervised deep-learning methods have demonstrated impressive performance on benchmarks based on synthetic noise. However, no metrics exist to evaluate these methods in an unsupervised fashion. This is highly problematic for the many practical applications where ground-truth clean images are not available. In this work, we propose two novel metrics: the unsupervised mean squared error (MSE) and the unsupervised peak signal-to-noise ratio (PSNR), which are computed using only noisy data. We provide a theoretical analysis of these metrics, showing that they are asymptotically consistent estimators of the supervised MSE and PSNR. Controlled numerical experiments with synthetic noise confirm that they provide accurate approximations in practice. We validate our approach on real-world data from two imaging modalities: videos in raw format and transmission electron microscopy. Our results demonstrate that the proposed metrics enable unsupervised evaluation of denoising methods based exclusively on noisy data.

----

## [998] Regions of Reliability in the Evaluation of Multivariate Probabilistic Forecasts

**Authors**: *Étienne Marcotte, Valentina Zantedeschi, Alexandre Drouin, Nicolas Chapados*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/marcotte23a.html](https://proceedings.mlr.press/v202/marcotte23a.html)

**Abstract**:

Multivariate probabilistic time series forecasts are commonly evaluated via proper scoring rules, i.e., functions that are minimal in expectation for the ground-truth distribution. However, this property is not sufficient to guarantee good discrimination in the non-asymptotic regime. In this paper, we provide the first systematic finite-sample study of proper scoring rules for time series forecasting evaluation. Through a power analysis, we identify the “region of reliability” of a scoring rule, i.e., the set of practical conditions where it can be relied on to identify forecasting errors. We carry out our analysis on a comprehensive synthetic benchmark, specifically designed to test several key discrepancies between ground-truth and forecast distributions, and we gauge the generalizability of our findings to real-world tasks with an application to an electricity production problem. Our results reveal critical shortcomings in the evaluation of multivariate probabilistic forecasts as commonly performed in the literature.

----

## [999] Analyzing Diffusion as Serial Reproduction

**Authors**: *Raja Marjieh, Ilia Sucholutsky, Thomas A. Langlois, Nori Jacoby, Thomas L. Griffiths*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/marjieh23a.html](https://proceedings.mlr.press/v202/marjieh23a.html)

**Abstract**:

Diffusion models are a class of generative models that learn to synthesize samples by inverting a diffusion process that gradually maps data into noise. While these models have enjoyed great success recently, a full theoretical understanding of their observed properties is still lacking, in particular, their weak sensitivity to the choice of noise family and the role of adequate scheduling of noise levels for good synthesis. By identifying a correspondence between diffusion models and a well-known paradigm in cognitive science known as serial reproduction, whereby human agents iteratively observe and reproduce stimuli from memory, we show how the aforementioned properties of diffusion models can be explained as a natural consequence of this correspondence. We then complement our theoretical analysis with simulations that exhibit these key features. Our work highlights how classic paradigms in cognitive science can shed light on state-of-the-art machine learning problems.

----



[Go to the previous page](ICML-2023-list04.md)

[Go to the next page](ICML-2023-list06.md)

[Go to the catalog section](README.md)