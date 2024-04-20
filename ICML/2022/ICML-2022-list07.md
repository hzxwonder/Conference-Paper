## [1200] Online Decision Transformer

**Authors**: *Qinqing Zheng, Amy Zhang, Aditya Grover*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zheng22c.html](https://proceedings.mlr.press/v162/zheng22c.html)

**Abstract**:

Recent work has shown that offline reinforcement learning (RL) can be formulated as a sequence modeling problem (Chen et al., 2021; Janner et al., 2021) and solved via approaches similar to large-scale language modeling. However, any practical instantiation of RL also involves an online component, where policies pretrained on passive offline datasets are finetuned via task-specific interactions with the environment. We propose Online Decision Transformers (ODT), an RL algorithm based on sequence modeling that blends offline pretraining with online finetuning in a unified framework. Our framework uses sequence-level entropy regularizers in conjunction with autoregressive modeling objectives for sample-efficient exploration and finetuning. Empirically, we show that ODT is competitive with the state-of-the-art in absolute performance on the D4RL benchmark but shows much more significant gains during the finetuning procedure.

----

## [1201] Learning Efficient and Robust Ordinary Differential Equations via Invertible Neural Networks

**Authors**: *Weiming Zhi, Tin Lai, Lionel Ott, Edwin V. Bonilla, Fabio Ramos*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhi22a.html](https://proceedings.mlr.press/v162/zhi22a.html)

**Abstract**:

Advances in differentiable numerical integrators have enabled the use of gradient descent techniques to learn ordinary differential equations (ODEs), where a flexible function approximator (often a neural network) is used to estimate the system dynamics, given as a time derivative. However, these integrators can be unsatisfactorily slow and unstable when learning systems of ODEs from long sequences. We propose to learn an ODE of interest from data by viewing its dynamics as a vector field related to another base vector field via a diffeomorphism (i.e., a differentiable bijection), represented by an invertible neural network (INN). By learning both the INN and the dynamics of the base ODE, we provide an avenue to offload some of the complexity in modelling the dynamics directly on to the INN. Consequently, by restricting the base ODE to be amenable to integration, we can speed up and improve the robustness of integrating trajectories from the learned system. We demonstrate the efficacy of our method in training and evaluating benchmark ODE systems, as well as within continuous-depth neural networks models. We show that our approach attains speed-ups of up to two orders of magnitude when integrating learned ODEs.

----

## [1202] HyperTransformer: Model Generation for Supervised and Semi-Supervised Few-Shot Learning

**Authors**: *Andrey Zhmoginov, Mark Sandler, Maksym Vladymyrov*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhmoginov22a.html](https://proceedings.mlr.press/v162/zhmoginov22a.html)

**Abstract**:

In this work we propose a HyperTransformer, a Transformer-based model for supervised and semi-supervised few-shot learning that generates weights of a convolutional neural network (CNN) directly from support samples. Since the dependence of a small generated CNN model on a specific task is encoded by a high-capacity Transformer model, we effectively decouple the complexity of the large task space from the complexity of individual tasks. Our method is particularly effective for small target CNN architectures where learning a fixed universal task-independent embedding is not optimal and better performance is attained when the information about the task can modulate all model parameters. For larger models we discover that generating the last layer alone allows us to produce competitive or better results than those obtained with state-of-the-art methods while being end-to-end differentiable.

----

## [1203] Describing Differences between Text Distributions with Natural Language

**Authors**: *Ruiqi Zhong, Charlie Snell, Dan Klein, Jacob Steinhardt*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhong22a.html](https://proceedings.mlr.press/v162/zhong22a.html)

**Abstract**:

How do two distributions of text differ? Humans are slow at answering this, since discovering patterns might require tediously reading through hundreds of samples. We propose to automatically summarize the differences by “learning a natural language hypothesis": given two distributions $D_{0}$ and $D_{1}$, we search for a description that is more often true for $D_{1}$, e.g., “is military-related." To tackle this problem, we fine-tune GPT-3 to propose descriptions with the prompt: “[samples of $D_{0}$] + [samples of $D_{1}$] + the difference between them is \underline{\space\space\space\space}". We then re-rank the descriptions by checking how often they hold on a larger set of samples with a learned verifier. On a benchmark of 54 real-world binary classification tasks, while GPT-3 Curie (13B) only generates a description similar to human annotation 7% of the time, the performance reaches 61% with fine-tuning and re-ranking, and our best system using GPT-3 Davinci (175B) reaches 76%. We apply our system to describe distribution shifts, debug dataset shortcuts, summarize unknown tasks, and label text clusters, and present analyses based on automatically generated descriptions.

----

## [1204] Pessimistic Minimax Value Iteration: Provably Efficient Equilibrium Learning from Offline Datasets

**Authors**: *Han Zhong, Wei Xiong, Jiyuan Tan, Liwei Wang, Tong Zhang, Zhaoran Wang, Zhuoran Yang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhong22b.html](https://proceedings.mlr.press/v162/zhong22b.html)

**Abstract**:

We study episodic two-player zero-sum Markov games (MGs) in the offline setting, where the goal is to find an approximate Nash equilibrium (NE) policy pair based on a dataset collected a priori. When the dataset does not have uniform coverage over all policy pairs, finding an approximate NE involves challenges in three aspects: (i) distributional shift between the behavior policy and the optimal policy, (ii) function approximation to handle large state space, and (iii) minimax optimization for equilibrium solving. We propose a pessimism-based algorithm, dubbed as pessimistic minimax value iteration (PMVI), which overcomes the distributional shift by constructing pessimistic estimates of the value functions for both players and outputs a policy pair by solving a correlated coarse equilibrium based on the two value functions. Furthermore, we establish a data-dependent upper bound on the suboptimality which recovers a sublinear rate without the assumption on uniform coverage of the dataset. We also prove an information-theoretical lower bound, which shows our upper bound is nearly minimax optimal, which suggests that the data-dependent term is intrinsic. Our theoretical results also highlight a notion of “relative uncertainty”, which characterizes the necessary and sufficient condition for achieving sample efficiency in offline MGs. To the best of our knowledge, we provide the first nearly minimax optimal result for offline MGs with function approximation.

----

## [1205] Dimension-free Complexity Bounds for High-order Nonconvex Finite-sum Optimization

**Authors**: *Dongruo Zhou, Quanquan Gu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22a.html](https://proceedings.mlr.press/v162/zhou22a.html)

**Abstract**:

Stochastic high-order methods for finding first-order stationary points in nonconvex finite-sum optimization have witnessed increasing interest in recent years, and various upper and lower bounds of the oracle complexity have been proved. However, under standard regularity assumptions, existing complexity bounds are all dimension-dependent (e.g., polylogarithmic dependence), which contrasts with the dimension-free complexity bounds for stochastic first-order methods and deterministic high-order methods. In this paper, we show that the polylogarithmic dimension dependence gap is not essential and can be closed. More specifically, we propose stochastic high-order algorithms with novel first-order and high-order derivative estimators, which can achieve dimension-free complexity bounds. With the access to $p$-th order derivatives of the objective function, we prove that our algorithm finds $\epsilon$-stationary points with $O(n^{(2p-1)/(2p)}/\epsilon^{(p+1)/p})$ high-order oracle complexities, where $n$ is the number of individual functions. Our result strictly improves the complexity bounds of existing high-order deterministic methods with respect to the dependence on $n$, and it is dimension-free compared with existing stochastic high-order methods.

----

## [1206] A Hierarchical Bayesian Approach to Inverse Reinforcement Learning with Symbolic Reward Machines

**Authors**: *Weichao Zhou, Wenchao Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22b.html](https://proceedings.mlr.press/v162/zhou22b.html)

**Abstract**:

A misspecified reward can degrade sample efficiency and induce undesired behaviors in reinforcement learning (RL) problems. We propose symbolic reward machines for incorporating high-level task knowledge when specifying the reward signals. Symbolic reward machines augment existing reward machine formalism by allowing transitions to carry predicates and symbolic reward outputs. This formalism lends itself well to inverse reinforcement learning, whereby the key challenge is determining appropriate assignments to the symbolic values from a few expert demonstrations. We propose a hierarchical Bayesian approach for inferring the most likely assignments such that the concretized reward machine can discriminate expert demonstrated trajectories from other trajectories with high accuracy. Experimental results show that learned reward machines can significantly improve training efficiency for complex RL tasks and generalize well across different task environment configurations.

----

## [1207] On the Optimization Landscape of Neural Collapse under MSE Loss: Global Optimality with Unconstrained Features

**Authors**: *Jinxin Zhou, Xiao Li, Tianyu Ding, Chong You, Qing Qu, Zhihui Zhu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22c.html](https://proceedings.mlr.press/v162/zhou22c.html)

**Abstract**:

When training deep neural networks for classification tasks, an intriguing empirical phenomenon has been widely observed in the last-layer classifiers and features, where (i) the class means and the last-layer classifiers all collapse to the vertices of a Simplex Equiangular Tight Frame (ETF) up to scaling, and (ii) cross-example within-class variability of last-layer activations collapses to zero. This phenomenon is called Neural Collapse (NC), which seems to take place regardless of the choice of loss functions. In this work, we justify NC under the mean squared error (MSE) loss, where recent empirical evidence shows that it performs comparably or even better than the de-facto cross-entropy loss. Under a simplified unconstrained feature model, we provide the first global landscape analysis for vanilla nonconvex MSE loss and show that the (only!) global minimizers are neural collapse solutions, while all other critical points are strict saddles whose Hessian exhibit negative curvature directions. Furthermore, we justify the usage of rescaled MSE loss by probing the optimization landscape around the NC solutions, showing that the landscape can be improved by tuning the rescaling hyperparameters. Finally, our theoretical findings are experimentally verified on practical network architectures.

----

## [1208] Model Agnostic Sample Reweighting for Out-of-Distribution Learning

**Authors**: *Xiao Zhou, Yong Lin, Renjie Pi, Weizhong Zhang, Renzhe Xu, Peng Cui, Tong Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22d.html](https://proceedings.mlr.press/v162/zhou22d.html)

**Abstract**:

Distributionally robust optimization (DRO) and invariant risk minimization (IRM) are two popular methods proposed to improve out-of-distribution (OOD) generalization performance of machine learning models. While effective for small models, it has been observed that these methods can be vulnerable to overfitting with large overparameterized models. This work proposes a principled method, Model Agnostic samPLe rEweighting (MAPLE), to effectively address OOD problem, especially in overparameterized scenarios. Our key idea is to find an effective reweighting of the training samples so that the standard empirical risk minimization training of a large model on the weighted training data leads to superior OOD generalization performance. The overfitting issue is addressed by considering a bilevel formulation to search for the sample reweighting, in which the generalization complexity depends on the search space of sample weights instead of the model size. We present theoretical analysis in linear case to prove the insensitivity of MAPLE to model size, and empirically verify its superiority in surpassing state-of-the-art methods by a large margin.

----

## [1209] Sparse Invariant Risk Minimization

**Authors**: *Xiao Zhou, Yong Lin, Weizhong Zhang, Tong Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22e.html](https://proceedings.mlr.press/v162/zhou22e.html)

**Abstract**:

Invariant Risk Minimization (IRM) is an emerging invariant feature extracting technique to help generalization with distributional shift. However, we find that there exists a basic and intractable contradiction between the model trainability and generalization ability in IRM. On one hand, recent studies on deep learning theory indicate the importance of large-sized or even overparameterized neural networks to make the model easy to train. On the other hand, unlike empirical risk minimization that can be benefited from overparameterization, our empirical and theoretical analyses show that the generalization ability of IRM is much easier to be demolished by overfitting caused by overparameterization. In this paper, we propose a simple yet effective paradigm named Sparse Invariant Risk Minimization (SparseIRM) to address this contradiction. Our key idea is to employ a global sparsity constraint as a defense to prevent spurious features from leaking in during the whole IRM process. Compared with sparisfy-after-training prototype by prior work which can discard invariant features, the global sparsity constraint limits the budget for feature selection and enforces SparseIRM to select the invariant features. We illustrate the benefit of SparseIRM through a theoretical analysis on a simple linear case. Empirically we demonstrate the power of SparseIRM through various datasets and models and surpass state-of-the-art methods with a gap up to 29%.

----

## [1210] Prototype-Anchored Learning for Learning with Imperfect Annotations

**Authors**: *Xiong Zhou, Xianming Liu, Deming Zhai, Junjun Jiang, Xin Gao, Xiangyang Ji*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22f.html](https://proceedings.mlr.press/v162/zhou22f.html)

**Abstract**:

The success of deep neural networks greatly relies on the availability of large amounts of high-quality annotated data, which however are difficult or expensive to obtain. The resulting labels may be class imbalanced, noisy or human biased. It is challenging to learn unbiased classification models from imperfectly annotated datasets, on which we usually suffer from overfitting or underfitting. In this work, we thoroughly investigate the popular softmax loss and margin-based loss, and offer a feasible approach to tighten the generalization error bound by maximizing the minimal sample margin. We further derive the optimality condition for this purpose, which indicates how the class prototypes should be anchored. Motivated by theoretical analysis, we propose a simple yet effective method, namely prototype-anchored learning (PAL), which can be easily incorporated into various learning-based classification schemes to handle imperfect annotation. We verify the effectiveness of PAL on class-imbalanced learning and noise-tolerant learning by extensive experiments on synthetic and real-world datasets.

----

## [1211] FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting

**Authors**: *Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, Rong Jin*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22g.html](https://proceedings.mlr.press/v162/zhou22g.html)

**Abstract**:

Long-term time series forecasting is challenging since prediction accuracy tends to decrease dramatically with the increasing horizon. Although Transformer-based methods have significantly improved state-of-the-art results for long-term forecasting, they are not only computationally expensive but more importantly, are unable to capture the global view of time series (e.g. overall trend). To address these problems, we propose to combine Transformer with the seasonal-trend decomposition method, in which the decomposition method captures the global profile of time series while Transformers capture more detailed structures. To further enhance the performance of Transformer for long-term prediction, we exploit the fact that most time series tend to have a sparse representation in a well-known basis such as Fourier transform, and develop a frequency enhanced Transformer. Besides being more effective, the proposed method, termed as Frequency Enhanced Decomposed Transformer (FEDformer), is more efficient than standard Transformer with a linear complexity to the sequence length. Our empirical studies with six benchmark datasets show that compared with state-of-the-art methods, Fedformer can reduce prediction error by 14.8% and 22.6% for multivariate and univariate time series, respectively. Code is publicly available at https://github.com/MAZiqing/FEDformer.

----

## [1212] Probabilistic Bilevel Coreset Selection

**Authors**: *Xiao Zhou, Renjie Pi, Weizhong Zhang, Yong Lin, Zonghao Chen, Tong Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22h.html](https://proceedings.mlr.press/v162/zhou22h.html)

**Abstract**:

The goal of coreset selection in supervised learning is to produce a weighted subset of data, so that training only on the subset achieves similar performance as training on the entire dataset. Existing methods achieved promising results in resource-constrained scenarios such as continual learning and streaming. However, most of the existing algorithms are limited to traditional machine learning models. A few algorithms that can handle large models adopt greedy search approaches due to the difficulty in solving the discrete subset selection problem, which is computationally costly when coreset becomes larger and often produces suboptimal results. In this work, for the first time we propose a continuous probabilistic bilevel formulation of coreset selection by learning a probablistic weight for each training sample. The overall objective is posed as a bilevel optimization problem, where 1) the inner loop samples coresets and train the model to convergence and 2) the outer loop updates the sample probability progressively according to the model’s performance. Importantly, we develop an efficient solver to the bilevel optimization problem via unbiased policy gradient without trouble of implicit differentiation. We theoretically prove the convergence of this training procedure and demonstrate the superiority of our algorithm against various coreset selection methods in various tasks, especially in more challenging label-noise and class-imbalance scenarios.

----

## [1213] Approximate Frank-Wolfe Algorithms over Graph-structured Support Sets

**Authors**: *Baojian Zhou, Yifan Sun*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22i.html](https://proceedings.mlr.press/v162/zhou22i.html)

**Abstract**:

In this paper, we consider approximate Frank-Wolfe (FW) algorithms to solve convex optimization problems over graph-structured support sets where the linear minimization oracle (LMO) cannot be efficiently obtained in general. We first demonstrate that two popular approximation assumptions (additive and multiplicative gap errors) are not applicable in that no cheap gap-approximate LMO oracle exists. Thus, approximate dual maximization oracles (DMO) are proposed, which approximate the inner product rather than the gap. We prove that the standard FW method using a $\delta$-approximate DMO converges as $O((1-\delta) \sqrt{s}/\delta)$ in the worst case, and as $O(L/(\delta^2 t))$ over a $\delta$-relaxation of the constraint set. Furthermore, when the solution is on the boundary, a variant of FW converges as $O(1/t^2)$ under the quadratic growth assumption. Our empirical results suggest that even these improved bounds are pessimistic, showing fast convergence in recovering real-world images with graph-structured sparsity.

----

## [1214] Improving Adversarial Robustness via Mutual Information Estimation

**Authors**: *Dawei Zhou, Nannan Wang, Xinbo Gao, Bo Han, Xiaoyu Wang, Yibing Zhan, Tongliang Liu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22j.html](https://proceedings.mlr.press/v162/zhou22j.html)

**Abstract**:

Deep neural networks (DNNs) are found to be vulnerable to adversarial noise. They are typically misled by adversarial samples to make wrong predictions. To alleviate this negative effect, in this paper, we investigate the dependence between outputs of the target model and input adversarial samples from the perspective of information theory, and propose an adversarial defense method. Specifically, we first measure the dependence by estimating the mutual information (MI) between outputs and the natural patterns of inputs (called natural MI) and MI between outputs and the adversarial patterns of inputs (called adversarial MI), respectively. We find that adversarial samples usually have larger adversarial MI and smaller natural MI compared with those w.r.t. natural samples. Motivated by this observation, we propose to enhance the adversarial robustness by maximizing the natural MI and minimizing the adversarial MI during the training process. In this way, the target model is expected to pay more attention to the natural pattern that contains objective semantics. Empirical evaluations demonstrate that our method could effectively improve the adversarial accuracy against multiple attacks.

----

## [1215] Modeling Adversarial Noise for Adversarial Training

**Authors**: *Dawei Zhou, Nannan Wang, Bo Han, Tongliang Liu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22k.html](https://proceedings.mlr.press/v162/zhou22k.html)

**Abstract**:

Deep neural networks have been demonstrated to be vulnerable to adversarial noise, promoting the development of defense against adversarial attacks. Motivated by the fact that adversarial noise contains well-generalizing features and that the relationship between adversarial data and natural data can help infer natural data and make reliable predictions, in this paper, we study to model adversarial noise by learning the transition relationship between adversarial labels (i.e. the flipped labels used to generate adversarial data) and natural labels (i.e. the ground truth labels of the natural data). Specifically, we introduce an instance-dependent transition matrix to relate adversarial labels and natural labels, which can be seamlessly embedded with the target model (enabling us to model stronger adaptive adversarial noise). Empirical evaluations demonstrate that our method could effectively improve adversarial accuracy.

----

## [1216] Contrastive Learning with Boosted Memorization

**Authors**: *Zhihan Zhou, Jiangchao Yao, Yan-Feng Wang, Bo Han, Ya Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22l.html](https://proceedings.mlr.press/v162/zhou22l.html)

**Abstract**:

Self-supervised learning has achieved a great success in the representation learning of visual and textual data. However, the current methods are mainly validated on the well-curated datasets, which do not exhibit the real-world long-tailed distribution. Recent attempts to consider self-supervised long-tailed learning are made by rebalancing in the loss perspective or the model perspective, resembling the paradigms in the supervised long-tailed learning. Nevertheless, without the aid of labels, these explorations have not shown the expected significant promise due to the limitation in tail sample discovery or the heuristic structure design. Different from previous works, we explore this direction from an alternative perspective, i.e., the data perspective, and propose a novel Boosted Contrastive Learning (BCL) method. Specifically, BCL leverages the memorization effect of deep neural networks to automatically drive the information discrepancy of the sample views in contrastive learning, which is more efficient to enhance the long-tailed learning in the label-unaware context. Extensive experiments on a range of benchmark datasets demonstrate the effectiveness of BCL over several state-of-the-art methods. Our code is available at https://github.com/MediaBrain-SJTU/BCL.

----

## [1217] Understanding The Robustness in Vision Transformers

**Authors**: *Daquan Zhou, Zhiding Yu, Enze Xie, Chaowei Xiao, Animashree Anandkumar, Jiashi Feng, José M. Álvarez*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22m.html](https://proceedings.mlr.press/v162/zhou22m.html)

**Abstract**:

Recent studies show that Vision Transformers (ViTs) exhibit strong robustness against various corruptions. Although this property is partly attributed to the self-attention mechanism, there is still a lack of an explanatory framework towards a more systematic understanding. In this paper, we examine the role of self-attention in learning robust representations. Our study is motivated by the intriguing properties of self-attention in visual grouping which indicate that self-attention could promote improved mid-level representation and robustness. We thus propose a family of fully attentional networks (FANs) that incorporate self-attention in both token mixing and channel processing. We validate the design comprehensively on various hierarchical backbones. Our model with a DeiT architecture achieves a state-of-the-art 47.6% mCE on ImageNet-C with 29M parameters. We also demonstrate significantly improved robustness in two downstream tasks: semantic segmentation and object detection

----

## [1218] VLUE: A Multi-Task Multi-Dimension Benchmark for Evaluating Vision-Language Pre-training

**Authors**: *Wangchunshu Zhou, Yan Zeng, Shizhe Diao, Xinsong Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhou22n.html](https://proceedings.mlr.press/v162/zhou22n.html)

**Abstract**:

Recent advances in vision-language pre-training (VLP) have demonstrated impressive performance in a range of vision-language (VL) tasks. However, there exist several challenges for measuring the community’s progress in building general multi-modal intelligence. First, most of the downstream VL datasets are annotated using raw images that are already seen during pre-training, which may result in an overestimation of current VLP models’ generalization ability. Second, recent VLP work mainly focuses on absolute performance but overlooks the efficiency-performance trade-off, which is also an important indicator for measuring progress. To this end, we introduce the Vision-Language Understanding Evaluation (VLUE) benchmark, a multi-task multi-dimension benchmark for evaluating the generalization capabilities and the efficiency-performance trade-off (“Pareto SOTA”) of VLP models. We demonstrate that there is a sizable generalization gap for all VLP models when testing on out-of-distribution test sets annotated on images from a more diverse distribution that spreads across cultures. Moreover, we find that measuring the efficiency-performance trade-off of VLP models leads to complementary insights for several design choices of VLP. We release the VLUE benchmark to promote research on building vision-language models that generalize well to images unseen during pre-training and are practical in terms of efficiency-performance trade-off.

----

## [1219] Detecting Corrupted Labels Without Training a Model to Predict

**Authors**: *Zhaowei Zhu, Zihao Dong, Yang Liu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhu22a.html](https://proceedings.mlr.press/v162/zhu22a.html)

**Abstract**:

Label noise in real-world datasets encodes wrong correlation patterns and impairs the generalization of deep neural networks (DNNs). It is critical to find efficient ways to detect corrupted patterns. Current methods primarily focus on designing robust training techniques to prevent DNNs from memorizing corrupted patterns. These approaches often require customized training processes and may overfit corrupted patterns, leading to a performance drop in detection. In this paper, from a more data-centric perspective, we propose a training-free solution to detect corrupted labels. Intuitively, “closer” instances are more likely to share the same clean label. Based on the neighborhood information, we propose two methods: the first one uses “local voting" via checking the noisy label consensuses of nearby features. The second one is a ranking-based approach that scores each instance and filters out a guaranteed number of instances that are likely to be corrupted. We theoretically analyze how the quality of features affects the local voting and provide guidelines for tuning neighborhood size. We also prove the worst-case error bound for the ranking-based method. Experiments with both synthetic and real-world label noise demonstrate our training-free solutions consistently and significantly improve most of the training-based baselines. Code is available at github.com/UCSC-REAL/SimiFeat.

----

## [1220] Contextual Bandits with Large Action Spaces: Made Practical

**Authors**: *Yinglun Zhu, Dylan J. Foster, John Langford, Paul Mineiro*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhu22b.html](https://proceedings.mlr.press/v162/zhu22b.html)

**Abstract**:

A central problem in sequential decision making is to develop algorithms that are practical and computationally efficient, yet support the use of flexible, general-purpose models. Focusing on the contextual bandit problem, recent progress provides provably efficient algorithms with strong empirical performance when the number of possible alternatives (“actions”) is small, but guarantees for decision making in large, continuous action spaces have remained elusive, leading to a significant gap between theory and practice. We present the first efficient, general-purpose algorithm for contextual bandits with continuous, linearly structured action spaces. Our algorithm makes use of computational oracles for (i) supervised learning, and (ii) optimization over the action space, and achieves sample complexity, runtime, and memory independent of the size of the action space. In addition, it is simple and practical. We perform a large-scale empirical evaluation, and show that our approach typically enjoys superior performance and efficiency compared to standard baselines.

----

## [1221] Neural-Symbolic Models for Logical Queries on Knowledge Graphs

**Authors**: *Zhaocheng Zhu, Mikhail Galkin, Zuobai Zhang, Jian Tang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhu22c.html](https://proceedings.mlr.press/v162/zhu22c.html)

**Abstract**:

Answering complex first-order logic (FOL) queries on knowledge graphs is a fundamental task for multi-hop reasoning. Traditional symbolic methods traverse a complete knowledge graph to extract the answers, which provides good interpretation for each step. Recent neural methods learn geometric embeddings for complex queries. These methods can generalize to incomplete knowledge graphs, but their reasoning process is hard to interpret. In this paper, we propose Graph Neural Network Query Executor (GNN-QE), a neural-symbolic model that enjoys the advantages of both worlds. GNN-QE decomposes a complex FOL query into relation projections and logical operations over fuzzy sets, which provides interpretability for intermediate variables. To reason about the missing links, GNN-QE adapts a graph neural network from knowledge graph completion to execute the relation projections, and models the logical operations with product fuzzy logic. Experiments on 3 datasets show that GNN-QE significantly improves over previous state-of-the-art models in answering FOL queries. Meanwhile, GNN-QE can predict the number of answers without explicit supervision, and provide visualizations for intermediate variables.

----

## [1222] Topology-aware Generalization of Decentralized SGD

**Authors**: *Tongtian Zhu, Fengxiang He, Lan Zhang, Zhengyang Niu, Mingli Song, Dacheng Tao*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhu22d.html](https://proceedings.mlr.press/v162/zhu22d.html)

**Abstract**:

This paper studies the algorithmic stability and generalizability of decentralized stochastic gradient descent (D-SGD). We prove that the consensus model learned by D-SGD is $\mathcal{O}{(m/N\unaryplus1/m\unaryplus\lambda^2)}$-stable in expectation in the non-convex non-smooth setting, where $N$ is the total sample size of the whole system, $m$ is the worker number, and $1\unaryminus\lambda$ is the spectral gap that measures the connectivity of the communication topology. These results then deliver an $\mathcal{O}{(1/N\unaryplus{({(m^{-1}\lambda^2)}^{\frac{\alpha}{2}}\unaryplus m^{\unaryminus\alpha})}/{N^{1\unaryminus\frac{\alpha}{2}}})}$ in-average generalization bound, which is non-vacuous even when $\lambda$ is closed to $1$, in contrast to vacuous as suggested by existing literature on the projected version of D-SGD. Our theory indicates that the generalizability of D-SGD has a positive correlation with the spectral gap, and can explain why consensus control in initial training phase can ensure better generalization. Experiments of VGG-11 and ResNet-18 on CIFAR-10, CIFAR-100 and Tiny-ImageNet justify our theory. To our best knowledge, this is the first work on the topology-aware generalization of vanilla D-SGD. Code is available at \url{https://github.com/Raiden-Zhu/Generalization-of-DSGD}.

----

## [1223] Resilient and Communication Efficient Learning for Heterogeneous Federated Systems

**Authors**: *Zhuangdi Zhu, Junyuan Hong, Steve Drew, Jiayu Zhou*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhu22e.html](https://proceedings.mlr.press/v162/zhu22e.html)

**Abstract**:

The rise of Federated Learning (FL) is bringing machine learning to edge computing by utilizing data scattered across edge devices. However, the heterogeneity of edge network topologies and the uncertainty of wireless transmission are two major obstructions of FL’s wide application in edge computing, leading to prohibitive convergence time and high communication cost. In this work, we propose an FL scheme to address both challenges simultaneously. Specifically, we enable edge devices to learn self-distilled neural networks that are readily prunable to arbitrary sizes, which capture the knowledge of the learning domain in a nested and progressive manner. Not only does our approach tackle system heterogeneity by serving edge devices with varying model architectures, but it also alleviates the issue of connection uncertainty by allowing transmitting part of the model parameters under faulty network connections, without wasting the contributing knowledge of the transmitted parameters. Extensive empirical studies show that under system heterogeneity and network instability, our approach demonstrates significant resilience and higher communication efficiency compared to the state-of-the-art.

----

## [1224] On Numerical Integration in Neural Ordinary Differential Equations

**Authors**: *Aiqing Zhu, Pengzhan Jin, Beibei Zhu, Yifa Tang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhu22f.html](https://proceedings.mlr.press/v162/zhu22f.html)

**Abstract**:

The combination of ordinary differential equations and neural networks, i.e., neural ordinary differential equations (Neural ODE), has been widely studied from various angles. However, deciphering the numerical integration in Neural ODE is still an open challenge, as many researches demonstrated that numerical integration significantly affects the performance of the model. In this paper, we propose the inverse modified differential equations (IMDE) to clarify the influence of numerical integration on training Neural ODE models. IMDE is determined by the learning task and the employed ODE solver. It is shown that training a Neural ODE model actually returns a close approximation of the IMDE, rather than the true ODE. With the help of IMDE, we deduce that (i) the discrepancy between the learned model and the true ODE is bounded by the sum of discretization error and learning loss; (ii) Neural ODE using non-symplectic numerical integration fail to learn conservation laws theoretically. Several experiments are performed to numerically verify our theoretical analysis.

----

## [1225] When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee

**Authors**: *Dixian Zhu, Gang Li, Bokun Wang, Xiaodong Wu, Tianbao Yang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhu22g.html](https://proceedings.mlr.press/v162/zhu22g.html)

**Abstract**:

In this paper, we propose systematic and efficient gradient-based methods for both one-way and two-way partial AUC (pAUC) maximization that are applicable to deep learning. We propose new formulations of pAUC surrogate objectives by using the distributionally robust optimization (DRO) to define the loss for each individual positive data. We consider two formulations of DRO, one of which is based on conditional-value-at-risk (CVaR) that yields a non-smooth but exact estimator for pAUC, and another one is based on a KL divergence regularized DRO that yields an inexact but smooth (soft) estimator for pAUC. For both one-way and two-way pAUC maximization, we propose two algorithms and prove their convergence for optimizing their two formulations, respectively. Experiments demonstrate the effectiveness of the proposed algorithms for pAUC maximization for deep learning on various datasets.

----

## [1226] Contextual Bandits with Smooth Regret: Efficient Learning in Continuous Action Spaces

**Authors**: *Yinglun Zhu, Paul Mineiro*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhu22h.html](https://proceedings.mlr.press/v162/zhu22h.html)

**Abstract**:

Designing efficient general-purpose contextual bandit algorithms that work with large—or even infinite—action spaces would facilitate application to important scenarios such as information retrieval, recommendation systems, and continuous control. While obtaining standard regret guarantees can be hopeless, alternative regret notions have been proposed to tackle the large action setting. We propose a smooth regret notion for contextual bandits, which dominates previously proposed alternatives. We design a statistically and computationally efficient algorithm—for the proposed smooth regret—that works with general function approximation under standard supervised oracles. We also present an adaptive algorithm that automatically adapts to any smoothness level. Our algorithms can be used to recover the previous minimax/Pareto optimal guarantees under the standard regret, e.g., in bandit problems with multiple best arms and Lipschitz/H{ö}lder bandits. We conduct large-scale empirical evaluations demonstrating the efficacy of our proposed algorithms.

----

## [1227] Residual-Based Sampling for Online Outlier-Robust PCA

**Authors**: *Tianhao Zhu, Jie Shen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhu22i.html](https://proceedings.mlr.press/v162/zhu22i.html)

**Abstract**:

Outlier-robust principal component analysis (ORPCA) has been broadly applied in scientific discovery in the last decades. In this paper, we study online ORPCA, an important variant that addresses the practical challenge that the data points arrive in a sequential manner and the goal is to recover the underlying subspace of the clean data with one pass of the data. Our main contribution is the first provable algorithm that enjoys comparable recovery guarantee to the best known batch algorithm, while significantly improving upon the state-of-the-art online ORPCA algorithms. The core technique is a robust version of the residual norm which, informally speaking, leverages not only the importance of a data point, but also how likely it behaves as an outlier.

----

## [1228] Region-Based Semantic Factorization in GANs

**Authors**: *Jiapeng Zhu, Yujun Shen, Yinghao Xu, Deli Zhao, Qifeng Chen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhu22j.html](https://proceedings.mlr.press/v162/zhu22j.html)

**Abstract**:

Despite the rapid advancement of semantic discovery in the latent space of Generative Adversarial Networks (GANs), existing approaches either are limited to finding global attributes or rely on a number of segmentation masks to identify local attributes. In this work, we present a highly efficient algorithm to factorize the latent semantics learned by GANs concerning an arbitrary image region. Concretely, we revisit the task of local manipulation with pre-trained GANs and formulate region-based semantic discovery as a dual optimization problem. Through an appropriately defined generalized Rayleigh quotient, we manage to solve such a problem without any annotations or training. Experimental results on various state-of-the-art GAN models demonstrate the effectiveness of our approach, as well as its superiority over prior arts regarding precise control, region robustness, speed of implementation, and simplicity of use.

----

## [1229] Beyond Images: Label Noise Transition Matrix Estimation for Tasks with Lower-Quality Features

**Authors**: *Zhaowei Zhu, Jialu Wang, Yang Liu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zhu22k.html](https://proceedings.mlr.press/v162/zhu22k.html)

**Abstract**:

The label noise transition matrix, denoting the transition probabilities from clean labels to noisy labels, is crucial for designing statistically robust solutions. Existing estimators for noise transition matrices, e.g., using either anchor points or clusterability, focus on computer vision tasks that are relatively easier to obtain high-quality representations. We observe that tasks with lower-quality features fail to meet the anchor-point or clusterability condition, due to the coexistence of both uninformative and informative representations. To handle this issue, we propose a generic and practical information-theoretic approach to down-weight the less informative parts of the lower-quality features. This improvement is crucial to identifying and estimating the label noise transition matrix. The salient technical challenge is to compute the relevant information-theoretical metrics using only noisy labels instead of clean ones. We prove that the celebrated $f$-mutual information measure can often preserve the order when calculated using noisy labels. We then build our transition matrix estimator using this distilled version of features. The necessity and effectiveness of the proposed method are also demonstrated by evaluating the estimation error on a varied set of tabular data and text classification tasks with lower-quality features. Code is available at github.com/UCSC-REAL/BeyondImages.

----

## [1230] Towards Uniformly Superhuman Autonomy via Subdominance Minimization

**Authors**: *Brian D. Ziebart, Sanjiban Choudhury, Xinyan Yan, Paul Vernaza*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ziebart22a.html](https://proceedings.mlr.press/v162/ziebart22a.html)

**Abstract**:

Prevalent imitation learning methods seek to produce behavior that matches or exceeds average human performance. This often prevents achieving expert-level or superhuman performance when identifying the better demonstrations to imitate is difficult. We instead assume demonstrations are of varying quality and seek to induce behavior that is unambiguously better (i.e., Pareto dominant or minimally subdominant) than all human demonstrations. Our minimum subdominance inverse optimal control training objective is primarily defined by high quality demonstrations; lower quality demonstrations, which are more easily dominated, are effectively ignored instead of degrading imitation. With increasing probability, our approach produces superhuman behavior incurring lower cost than demonstrations on the demonstrator’s unknown cost function{—}even if that cost function differs for each demonstration. We apply our approach on a computer cursor pointing task, producing behavior that is 78% superhuman, while minimizing demonstration suboptimality provides 50% superhuman behavior{—}and only 72% even after selective data cleaning.

----

## [1231] Inductive Matrix Completion: No Bad Local Minima and a Fast Algorithm

**Authors**: *Pini Zilber, Boaz Nadler*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zilber22a.html](https://proceedings.mlr.press/v162/zilber22a.html)

**Abstract**:

The inductive matrix completion (IMC) problem is to recover a low rank matrix from few observed entries while incorporating prior knowledge about its row and column subspaces. In this work, we make three contributions to the IMC problem: (i) we prove that under suitable conditions, the IMC optimization landscape has no bad local minima; (ii) we derive a simple scheme with theoretical guarantees to estimate the rank of the unknown matrix; and (iii) we propose GNIMC, a simple Gauss-Newton based method to solve the IMC problem, analyze its runtime and derive for it strong recovery guarantees. The guarantees for GNIMC are sharper in several aspects than those available for other methods, including a quadratic convergence rate, fewer required observed entries and stability to errors or deviations from low-rank. Empirically, given entries observed uniformly at random, GNIMC recovers the underlying matrix substantially faster than several competing methods.

----

## [1232] Counterfactual Prediction for Outcome-Oriented Treatments

**Authors**: *Hao Zou, Bo Li, Jiangang Han, Shuiping Chen, Xuetao Ding, Peng Cui*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zou22a.html](https://proceedings.mlr.press/v162/zou22a.html)

**Abstract**:

Large amounts of efforts have been devoted into learning counterfactual treatment outcome under various settings, including binary/continuous/multiple treatments. Most of these literature aims to minimize the estimation error of counterfactual outcome for the whole treatment space. However, in most scenarios when the counterfactual prediction model is utilized to assist decision-making, people are only concerned with the small fraction of treatments that can potentially induce superior outcome (i.e. outcome-oriented treatments). This gap of objective is even more severe when the number of possible treatments is large, for example under the continuous treatment setting. To overcome it, we establish a new objective of optimizing counterfactual prediction on outcome-oriented treatments, propose a novel Outcome-Oriented Sample Re-weighting (OOSR) method to make the predictive model concentrate more on outcome-oriented treatments, and theoretically analyze that our method can improve treatment selection towards the optimal one. Extensive experimental results on both synthetic datasets and semi-synthetic datasets demonstrate the effectiveness of our method.

----

## [1233] SpaceMAP: Visualizing High-Dimensional Data by Space Expansion

**Authors**: *Xinrui Zu, Qian Tao*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/zu22a.html](https://proceedings.mlr.press/v162/zu22a.html)

**Abstract**:

Dimensionality reduction (DR) of high-dimensional data is of theoretical and practical interest in machine learning. However, there exist intriguing, non-intuitive discrepancies between the geometry of high- and low-dimensional space. We look into such discrepancies and propose a novel visualization method called Space-based Manifold Approximation and Projection (SpaceMAP). Our method establishes an analytical transformation on distance metrics between spaces to address the “crowding problem" in DR. With the proposed equivalent extended distance (EED), we are able to match the capacity of high- and low-dimensional space in a principled manner. To handle complex data with different manifold properties, we propose hierarchical manifold approximation to model the similarity function in a data-specific manner. We evaluated SpaceMAP on a range of synthetic and real datasets with varying manifold properties, and demonstrated its excellent performance in comparison with classical and state-of-the-art DR methods. In particular, the concept of space expansion provides a generic framework for understanding nonlinear DR methods including the popular t-distributed Stochastic Neighbor Embedding (t-SNE) and Uniform Manifold Approximation and Projection

----



[Go to the previous page](ICML-2022-list06.md)

[Go to the catalog section](README.md)