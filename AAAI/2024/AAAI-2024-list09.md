## [1600] HOP to the Next Tasks and Domains for Continual Learning in NLP

**Authors**: *Umberto Michieli, Mete Ozay*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29349](https://doi.org/10.1609/aaai.v38i13.29349)

**Abstract**:

Continual Learning (CL) aims to learn a sequence of problems (i.e., tasks and domains) by transferring knowledge acquired on previous problems, whilst avoiding forgetting of past ones. Different from previous approaches which focused on CL for one NLP task or domain in a specific use-case, in this paper, we address a more general CL setting to learn from a sequence of problems in a unique framework. Our method, HOP, permits to hop across tasks and domains by addressing the CL problem along three directions: (i) we employ a set of adapters to generalize a large pre-trained model to unseen problems, (ii) we compute high-order moments over the distribution of embedded representations to distinguish independent and correlated statistics across different tasks and domains, (iii) we process this enriched information with auxiliary heads specialized for each end problem. Extensive experimental campaign on 4 NLP applications, 5 benchmarks and 2 CL setups demonstrates the effectiveness of our HOP.

----

## [1601] Leveraging Local Variance for Pseudo-Label Selection in Semi-supervised Learning

**Authors**: *Zeping Min, Jinfeng Bai, Chengfei Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29350](https://doi.org/10.1609/aaai.v38i13.29350)

**Abstract**:

Semi-supervised learning algorithms that use pseudo-labeling have become increasingly popular for improving model performance by utilizing both labeled and unlabeled data. 
In this paper, we offer a fresh perspective on the selection of pseudo-labels, inspired by theoretical insights. We suggest that pseudo-labels with a high degree of local variance are more prone to inaccuracies. Based on this premise, we introduce the Local Variance Match (LVM) method, which aims to optimize the selection of pseudo-labels in semi-supervised learning (SSL) tasks. Our methodology is validated through a series of experiments on widely-used image classification datasets, such as CIFAR-10, CIFAR-100, and SVHN, spanning various labeled data quantity scenarios. The empirical findings show that the LVM method substantially outpaces current SSL techniques, achieving state-of-the-art results in many of these scenarios. For instance, we observed an error rate of 5.41% on CIFAR-10 with a single label for each class, 35.87% on CIFAR-100 when using four labels per class, and 1.94% on SVHN with four labels for each class. Notably, the standout error rate of 5.41% is less than 1% shy of the performance in a fully-supervised learning environment. In experiments on ImageNet with 100k labeled data, the LVM also reached state-of-the-art outcomes. Additionally, the efficacy of the LVM method is further validated by its stellar performance in speech recognition experiments.

----

## [1602] Input Margins Can Predict Generalization Too

**Authors**: *Coenraad Mouton, Marthinus Wilhelmus Theunissen, Marelie H. Davel*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29351](https://doi.org/10.1609/aaai.v38i13.29351)

**Abstract**:

Understanding generalization in deep neural networks is an active area of research. A promising avenue of exploration has been that of margin measurements: the shortest distance to the decision boundary for a given sample or its representation internal to the network. While margins have been shown to be correlated with the generalization ability of a model when measured at its hidden representations (hidden margins), no such link between large margins and generalization has been established for input margins. We show that while input margins are not generally predictive of generalization, they can be if the search space is appropriately constrained.
We develop such a measure based on input margins, which we refer to as 'constrained margins'. The predictive power of this new measure is demonstrated on the 'Predicting Generalization in Deep Learning' (PGDL) dataset and contrasted with hidden representation margins. We find that constrained margins achieve highly competitive scores and outperform other margin measurements in general. This provides a novel insight on the relationship between generalization and classification margins, and highlights the importance of considering the data manifold for investigations of generalization in DNNs.

----

## [1603] Beyond TreeSHAP: Efficient Computation of Any-Order Shapley Interactions for Tree Ensembles

**Authors**: *Maximilian Muschalik, Fabian Fumagalli, Barbara Hammer, Eyke Hüllermeier*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29352](https://doi.org/10.1609/aaai.v38i13.29352)

**Abstract**:

While shallow decision trees may be interpretable, larger ensemble models like gradient-boosted trees, which often set the state of the art in machine learning problems involving tabular data, still remain black box models. As a remedy, the Shapley value (SV) is a well-known concept in explainable artificial intelligence (XAI) research for quantifying additive feature attributions of predictions. The model-specific TreeSHAP methodology solves the exponential complexity for retrieving exact SVs from tree-based models. Expanding beyond individual feature attribution, Shapley interactions reveal the impact of intricate feature interactions of any order. In this work, we present TreeSHAP-IQ, an efficient method to compute any-order additive Shapley interactions for predictions of tree-based models. TreeSHAP-IQ is supported by a mathematical framework that exploits polynomial arithmetic to compute the interaction scores in a single recursive traversal of the tree, akin to Linear TreeSHAP. We apply TreeSHAP-IQ on state-of-the-art tree ensembles and explore interactions on well-established benchmark datasets.

----

## [1604] Continuous Treatment Effect Estimation Using Gradient Interpolation and Kernel Smoothing

**Authors**: *Lokesh Nagalapatti, Akshay Iyer, Abir De, Sunita Sarawagi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29353](https://doi.org/10.1609/aaai.v38i13.29353)

**Abstract**:

We address the Individualized continuous treatment effect
(ICTE) estimation problem where we predict the effect of
any continuous valued treatment on an individual using ob-
servational data. The main challenge in this estimation task
is the potential confounding of treatment assignment with in-
dividual’s covariates in the training data, whereas during in-
ference ICTE requires prediction on independently sampled
treatments. In contrast to prior work that relied on regularizers
or unstable GAN training, we advocate the direct approach
of augmenting training individuals with independently sam-
pled treatments and inferred counterfactual outcomes. We in-
fer counterfactual outcomes using a two-pronged strategy: a
Gradient Interpolation for close-to-observed treatments, and
a Gaussian Process based Kernel Smoothing which allows
us to down weigh high variance inferences. We evaluate our
method on five benchmarks and show that our method out-
performs six state-of-the-art methods on the counterfactual
estimation error. We analyze the superior performance of our
method by showing that (1) our inferred counterfactual re-
sponses are more accurate, and (2) adding them to the train-
ing data reduces the distributional distance between the con-
founded training distribution and test distribution where treat-
ment is independent of covariates. Our proposed method is
model-agnostic and we show that it improves ICTE accuracy
of several existing models.

----

## [1605] Revisiting Disentanglement in Downstream Tasks: A Study on Its Necessity for Abstract Visual Reasoning

**Authors**: *Ruiqian Nai, Zixin Wen, Ji Li, Yuanzhi Li, Yang Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29354](https://doi.org/10.1609/aaai.v38i13.29354)

**Abstract**:

In representation learning, a disentangled representation is highly desirable as it encodes generative factors of data in a separable and compact pattern. Researchers have advocated leveraging disentangled representations to complete downstream tasks with encouraging empirical evidence. This paper further investigates the necessity of disentangled representation in downstream applications. Specifically, we show that dimension-wise disentangled representations are unnecessary on a fundamental downstream task, abstract visual reasoning. We provide extensive empirical evidence against the necessity of disentanglement, covering multiple datasets, representation learning methods, and downstream network architectures. Furthermore, our findings suggest that the informativeness of representations is a better indicator of downstream performance than disentanglement. Finally, the positive correlation between informativeness and disentanglement explains the claimed usefulness of disentangled representations in previous works.  The source code is available at https://github.com/Richard-coder-Nai/disentanglement-lib-necessity.git

----

## [1606] Thompson Sampling for Real-Valued Combinatorial Pure Exploration of Multi-Armed Bandit

**Authors**: *Shintaro Nakamura, Masashi Sugiyama*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29355](https://doi.org/10.1609/aaai.v38i13.29355)

**Abstract**:

We study the real-valued combinatorial pure exploration of the multi-armed bandit (R-CPE-MAB) problem. In R-CPE-MAB, a player is given stochastic arms, and the reward of each arm follows an unknown distribution. In each time step, a player pulls a single arm and observes its reward. The player's goal is to identify the optimal action from a finite-sized real-valued action set with as few arm pulls as possible. Previous methods in the R-CPE-MAB require enumerating all of the feasible actions of the combinatorial optimization problem one is considering. In general, since the size of the action set grows exponentially large with respect to the number of arms, this is almost practically impossible when the number of arms is large. We introduce an algorithm named the Generalized Thompson Sampling Explore (GenTS-Explore) algorithm, which is the first algorithm that can work even when the size of the action set is exponentially large with respect to the number of arms. We also introduce a novel problem-dependent sample complexity lower bound of the R-CPE-MAB problem, and show that the GenTS-Explore algorithm achieves the optimal sample complexity up to a problem-dependent constant factor.

----

## [1607] Efficient Learning of PDEs via Taylor Expansion and Sparse Decomposition into Value and Fourier Domains

**Authors**: *Md. Nasim, Yexiang Xue*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29356](https://doi.org/10.1609/aaai.v38i13.29356)

**Abstract**:

Accelerating the learning of Partial Differential Equations (PDEs) from experimental data will speed up the pace of scientific discovery. Previous randomized algorithms exploit sparsity in PDE updates for acceleration. However such methods are applicable to a limited class of decomposable PDEs, which have sparse features in the value domain. We propose Reel, which accelerates the learning of PDEs via random projection and has much broader applicability. Reel exploits the sparsity by decomposing dense updates into sparse ones in both the value and frequency domains. This decomposition enables efficient learning when the source of the updates consists of gradually changing terms across large areas (sparse in the frequency domain) in addition to a few rapid updates concentrated in a small set of “interfacial” regions (sparse in the value domain). Random projection is then applied to compress the sparse signals for learning. To expand the model applicability, Taylor series expansion is used in Reel to approximate the nonlinear PDE updates with polynomials in the decomposable form. Theoretically, we derive a constant factor approximation between the projected loss function and the original one with poly-logarithmic number of projected dimensions. Experimentally, we provide empirical evidence that our proposed Reel can lead to faster learning of PDE models (70-98% reduction in training time when the data is compressed to 1% of its original size) with comparable quality as the non-compressed models.

----

## [1608] Secure Distributed Sparse Gaussian Process Models Using Multi-Key Homomorphic Encryption

**Authors**: *Adil Nawaz, Guopeng Chen, Muhammad Umair Raza, Zahid Iqbal, Jianqiang Li, Victor C. M. Leung, Jie Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29357](https://doi.org/10.1609/aaai.v38i13.29357)

**Abstract**:

Distributed sparse Gaussian process (dGP) models provide an ability to achieve accurate predictive performance using data from multiple devices in a time efficient and scalable manner. The distributed computation of model, however, risks exposure of privately owned data to public manipulation. In this paper we propose a secure solution for dGP regression models using multi-key homomorphic encryption. Experimental results show that with a little sacrifice in terms of time complexity, we achieve a secure dGP model without deteriorating the predictive performance compared to traditional non-secure dGP models. We also present a practical implementation of the proposed model using several Nvidia Jetson Nano Developer Kit modules to simulate a real-world scenario. Thus, secure dGP model plugs the data security issues of dGP and provide a secure and trustworthy solution for multiple devices to use privately owned data for model computation in a distributed environment availing speed, scalability and robustness of dGP.

----

## [1609] Multiple Hypothesis Dropout: Estimating the Parameters of Multi-Modal Output Distributions

**Authors**: *David D. Nguyen, David Liebowitz, Salil S. Kanhere, Surya Nepal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29358](https://doi.org/10.1609/aaai.v38i13.29358)

**Abstract**:

In many real-world applications, from robotics to pedestrian trajectory prediction, there is a need to predict multiple real-valued outputs to represent several potential scenarios. Current deep learning techniques to address multiple-output problems are based on two main methodologies: (1) mixture density networks, which suffer from poor stability at high dimensions, or (2) multiple choice learning (MCL), an approach that uses M single-output functions, each only producing a point estimate hypothesis. This paper presents a Mixture of Multiple-Output functions (MoM) approach using a novel variant of dropout, Multiple Hypothesis Dropout. Unlike traditional MCL-based approaches, each multiple-output function not only estimates the mean but also the variance for its hypothesis. This is achieved through a novel stochastic winner-take-all loss which allows each multiple-output function to estimate variance through the spread of its subnetwork predictions.
Experiments on supervised learning problems illustrate that our approach outperforms existing solutions for reconstructing multimodal output distributions.
Additional studies on unsupervised learning problems show that estimating the parameters of latent posterior distributions within a discrete autoencoder significantly improves codebook efficiency, sample quality, precision and recall.

----

## [1610] On Inference Stability for Diffusion Models

**Authors**: *Viet Nguyen, Giang Vu, Tung Nguyen Thanh, Khoat Than, Toan Tran*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29359](https://doi.org/10.1609/aaai.v38i13.29359)

**Abstract**:

Denoising Probabilistic Models (DPMs) represent an emerging domain of generative models that excel in generating diverse and high-quality images. However, most current training methods for DPMs often neglect the correlation between timesteps, limiting the model's performance in generating images effectively. Notably, we theoretically point out that this issue can be caused by the cumulative estimation gap between the predicted and the actual trajectory. To minimize that gap, we propose a novel sequence-aware loss that aims to reduce the estimation gap to enhance the sampling quality. Furthermore, we theoretically show that our proposed loss function is a tighter upper bound of the estimation loss in comparison with the conventional loss in DPMs. Experimental results on several benchmark datasets including CIFAR10, CelebA, and CelebA-HQ consistently show a remarkable improvement of our proposed method regarding the image generalization quality measured by FID and Inception Score compared to several DPM baselines. Our code and pre-trained checkpoints are available at https://github.com/VinAIResearch/SA-DPM.

----

## [1611] Improve Robustness of Reinforcement Learning against Observation Perturbations via l∞ Lipschitz Policy Networks

**Authors**: *Buqing Nie, Jingtian Ji, Yangqing Fu, Yue Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29360](https://doi.org/10.1609/aaai.v38i13.29360)

**Abstract**:

Deep Reinforcement Learning (DRL) has achieved remarkable advances in sequential decision tasks. However, recent works have revealed that DRL agents are susceptible to slight perturbations in observations. This vulnerability raises concerns regarding the effectiveness and robustness of deploying such agents in real-world applications. In this work, we propose a novel robust reinforcement learning method called SortRL, which improves the robustness of DRL policies against observation perturbations from the perspective of the network architecture. We employ a novel architecture for the policy network that incorporates global $l_\infty$ Lipschitz continuity and provide a convenient method to enhance policy robustness based on the output margin. Besides, a training framework is designed for SortRL, which solves given tasks while maintaining robustness against $l_\infty$ bounded perturbations on the observations. Several experiments are conducted to evaluate the effectiveness of our method, including classic control tasks and video games. The results demonstrate that SortRL achieves state-of-the-art robustness performance against different perturbation strength.

----

## [1612] Multi-Class Support Vector Machine with Maximizing Minimum Margin

**Authors**: *Feiping Nie, Zhezheng Hao, Rong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29361](https://doi.org/10.1609/aaai.v38i13.29361)

**Abstract**:

Support Vector Machine (SVM) stands out as a prominent machine learning technique widely applied in practical pattern recognition tasks. It achieves binary classification by maximizing the "margin", which represents the minimum distance between instances and the decision boundary. Although many efforts have been dedicated to expanding SVM for multi-class case through strategies such as one versus one and one versus the rest, satisfactory solutions remain to be developed. In this paper, we propose a novel method for multi-class SVM that incorporates pairwise class loss considerations and maximizes the minimum margin. Adhering to this concept, we embrace a new formulation that imparts heightened flexibility to multi-class SVM.
Furthermore, the correlations between the proposed method and multiple forms of multi-class SVM are analyzed. The proposed regularizer, akin to the concept of "margin", can serve as a seamless enhancement over the softmax in deep learning, providing guidance for network parameter learning. Empirical evaluations demonstrate the effectiveness and superiority of our proposed 
method over existing multi-classification methods. Complete version is available at https://arxiv.org/pdf/2312.06578.pdf. Code is available at https://github.com/zz-haooo/M3SVM.

----

## [1613] Symmetric Q-learning: Reducing Skewness of Bellman Error in Online Reinforcement Learning

**Authors**: *Motoki Omura, Takayuki Osa, Yusuke Mukuta, Tatsuya Harada*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29362](https://doi.org/10.1609/aaai.v38i13.29362)

**Abstract**:

In deep reinforcement learning, estimating the value function to evaluate the quality of states and actions is essential. The value function is often trained using the least squares method, which implicitly assumes a Gaussian error distribution. However, a recent study suggested that the error distribution for training the value function is often skewed because of the properties of the Bellman operator, and violates the implicit assumption of normal error distribution in the least squares method. To address this, we proposed a method called Symmetric Q-learning, in which the synthetic noise generated from a zero-mean distribution is added to the target values to generate a Gaussian error distribution. We evaluated the proposed method on continuous control benchmark tasks in MuJoCo. It improved the sample efficiency of a state-of-the-art reinforcement learning method by reducing the skewness of the error distribution.

----

## [1614] A Primal-Dual Algorithm for Hybrid Federated Learning

**Authors**: *Tom Overman, Garrett Blum, Diego Klabjan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29363](https://doi.org/10.1609/aaai.v38i13.29363)

**Abstract**:

Very few methods for hybrid federated learning, where clients only hold subsets of both features and samples, exist. Yet, this scenario is very important in practical settings. We provide a fast, robust algorithm for hybrid federated learning that hinges on Fenchel Duality. We prove the convergence of the algorithm to the same solution as if the model was trained centrally in a variety of practical regimes. Furthermore, we provide experimental results that demonstrate the performance improvements of the algorithm over a commonly used method in federated learning, FedAvg, and an existing hybrid FL algorithm, HyFEM. We also provide privacy considerations and necessary steps to protect client data.

----

## [1615] Multi-Objective Bayesian Optimization with Active Preference Learning

**Authors**: *Ryota Ozaki, Kazuki Ishikawa, Youhei Kanzaki, Shion Takeno, Ichiro Takeuchi, Masayuki Karasuyama*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29364](https://doi.org/10.1609/aaai.v38i13.29364)

**Abstract**:

There are a lot of real-world black-box optimization problems that need to optimize multiple criteria simultaneously. However, in a multi-objective optimization (MOO) problem, identifying the whole Pareto front requires the prohibitive search cost, while in many practical scenarios, the decision maker (DM) only needs a specific solution among the set of the Pareto optimal solutions. We propose a Bayesian optimization (BO) approach to identifying the most preferred solution in the MOO with expensive objective functions, in which a Bayesian preference model of the DM is adaptively estimated by an interactive manner based on the two types of supervisions called the pairwise preference and improvement request. To explore the most preferred solution, we define an acquisition function in which the uncertainty both in the objective function and the DM preference is incorporated. Further, to minimize the interaction cost with the DM, we also propose an active learning strategy for the preference estimation. We empirically demonstrate the effectiveness of our proposed method through the benchmark function optimization and the hyper-parameter optimization problems for machine learning models.

----

## [1616] Towards Fair Graph Federated Learning via Incentive Mechanisms

**Authors**: *Chenglu Pan, Jiarong Xu, Yue Yu, Ziqi Yang, Qingbiao Wu, Chunping Wang, Lei Chen, Yang Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29365](https://doi.org/10.1609/aaai.v38i13.29365)

**Abstract**:

Graph federated learning (FL) has emerged as a pivotal paradigm enabling  multiple agents to collaboratively train a graph model while preserving local data privacy. Yet, current efforts overlook a key issue: agents are self-interested and would hesitant to share data without fair and satisfactory incentives. This paper is the first endeavor to address this issue by studying the incentive mechanism for graph federated learning. We identify a unique phenomenon in graph federated learning: the presence of agents posing potential harm to the federation and agents contributing with delays. This stands in contrast to previous FL incentive mechanisms that assume all agents contribute positively and in a timely manner. 
In view of this, this paper presents a novel incentive mechanism  tailored for fair graph federated learning, integrating incentives derived from both model gradient and payoff. To achieve this, we first introduce an agent valuation function aimed at quantifying agent contributions through the introduction of two criteria: gradient alignment and graph diversity. Moreover, due to the high heterogeneity in graph federated learning, striking a balance between accuracy and fairness becomes particularly crucial. We introduce motif prototypes to enhance accuracy, communicated between the server and agents, enhancing global model aggregation and aiding agents in local model optimization. Extensive experiments show that our model achieves the best trade-off between accuracy and the fairness of model gradient, as well as superior payoff fairness.

----

## [1617] A Graph Dynamics Prior for Relational Inference

**Authors**: *Liming Pan, Cheng Shi, Ivan Dokmanic*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29366](https://doi.org/10.1609/aaai.v38i13.29366)

**Abstract**:

Relational inference aims to identify interactions between parts of a dynamical system from the observed dynamics. Current state-of-the-art methods fit the dynamics with a graph neural network (GNN) on a learnable graph. They use one-step message-passing GNNs---intuitively the right choice since non-locality of multi-step or spectral GNNs may confuse direct and indirect interactions. But the effective interaction graph depends on the sampling rate and it is rarely localized to direct neighbors, leading to poor local optima for the one-step model. In this work, we propose a graph dynamics prior (GDP) for relational inference. GDP constructively uses error amplification in non-local polynomial filters to steer the solution to the ground-truth graph. To deal with non-uniqueness, GDP simultaneously fits a ``shallow'' one-step model and a polynomial multi-step model with shared graph topology. Experiments show that GDP reconstructs graphs far more accurately than earlier methods, with remarkable robustness to under-sampling. Since appropriate sampling rates for unknown dynamical systems are not known a priori, this robustness makes GDP suitable for real applications in scientific machine learning. Reproducible code is available at https://github.com/DaDaCheng/GDP.

----

## [1618] Learning Reduced Fluid Dynamics

**Authors**: *Zherong Pan, Xifeng Gao, Kui Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29367](https://doi.org/10.1609/aaai.v38i13.29367)

**Abstract**:

Predicting the state evolution of ultra high-dimensional, time-reversible fluid dynamic systems is a crucial but computationally expensive task. Existing physics-informed neural networks either incur high inference cost or cannot preserve the time-reversible nature of the underlying dynamics system. We propose a model-based approach to identify low-dimensional, time reversible, nonlinear fluid dynamic systems. Our method utilizes the symplectic structure of reduced Eulerian fluid and use stochastic Riemann optimization to obtain a low-dimensional bases that minimize the expected trajectory-wise dimension-reduction error over a given distribution of initial conditions. We show that such minimization is well-defined since the reduced trajectories are differentiable with respect to the subspace bases over the entire Grassmannian manifold, under proper choices of timestep sizes and numerical integrators. Finally, we propose a loss function measuring the trajectory-wise discrepancy between the original and reduced models. By tensor precomputation, we show that gradient information of such loss function can be evaluated efficiently over a long trajectory without time-integrating the high-dimensional dynamic system. Through evaluations on a row of simulation benchmarks, we show that our method reduces the discrepancy by 50-90 percent over conventional reduced models and we outperform PINNs by exactly preserving the time reversibility.

----

## [1619] FedLF: Layer-Wise Fair Federated Learning

**Authors**: *Zibin Pan, Chi Li, Fangchen Yu, Shuyi Wang, Haijin Wang, Xiaoying Tang, Junhua Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29368](https://doi.org/10.1609/aaai.v38i13.29368)

**Abstract**:

Fairness has become an important concern in Federated Learning (FL). An unfair model that performs well for some clients while performing poorly for others can reduce the willingness of clients to participate. In this work, we identify a direct cause of unfairness in FL - the use of an unfair direction to update the global model, which favors some clients while conflicting with other clients’ gradients at the model and layer levels. To address these issues, we propose a layer-wise fair Federated Learning algorithm (FedLF). Firstly, we formulate a multi-objective optimization problem with an effective fair-driven objective for FL. A layer-wise fair direction is then calculated to mitigate the model and layer-level gradient conflicts and reduce the improvement bias. We further provide the theoretical analysis on how FedLF can improve fairness and guarantee convergence. Extensive experiments on different learning tasks and models demonstrate that FedLF outperforms the SOTA FL algorithms in terms of accuracy and fairness. The source code is available at https://github.com/zibinpan/FedLF.

----

## [1620] Convolutional Channel-Wise Competitive Learning for the Forward-Forward Algorithm

**Authors**: *Andreas Papachristodoulou, Christos Kyrkou, Stelios Timotheou, Theocharis Theocharides*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29369](https://doi.org/10.1609/aaai.v38i13.29369)

**Abstract**:

The Forward-Forward (FF) Algorithm has been recently proposed to alleviate the issues of backpropagation (BP) commonly used to train deep neural networks. However, its current formulation exhibits limitations such as the generation of negative data, slower convergence, and inadequate performance on complex tasks. In this paper we take the main ideas of FF and improve them by leveraging channel-wise competitive learning in the context of convolutional neural networks for image classification tasks. A layer-wise loss function is introduced that promotes competitive learning and eliminates the need for negative data construction. To enhance both the learning of compositional features and feature space partitioning, a channel-wise feature separator and extractor block is proposed that complements the competitive learning process. Our method outperforms recent FF-based models on image classification tasks, achieving testing errors of 0.58%, 7.69%, 21.89%, and 48.77% on MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 respectively. Our approach bridges the performance gap between FF learning and BP methods, indicating the potential of our proposed approach to learn useful representations in a layer-wise modular fashion, enabling more efficient and flexible learning. Our source code and supplementary material are available at https://github.com/andreaspapac/CwComp.

----

## [1621] REPrune: Channel Pruning via Kernel Representative Selection

**Authors**: *Mincheol Park, Dongjin Kim, Cheonjun Park, Yuna Park, Gyeong Eun Gong, Won Woo Ro, Suhyun Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29370](https://doi.org/10.1609/aaai.v38i13.29370)

**Abstract**:

Channel pruning is widely accepted to accelerate modern convolutional neural networks (CNNs). The resulting pruned model benefits from its immediate deployment on general-purpose software and hardware resources. However, its large pruning granularity, specifically at the unit of a convolution filter, often leads to undesirable accuracy drops due to the inflexibility of deciding how and where to introduce sparsity to the CNNs. In this paper, we propose REPrune, a novel channel pruning technique that emulates kernel pruning, fully exploiting the finer but structured granularity. REPrune identifies similar kernels within each channel using agglomerative clustering. Then, it selects filters that maximize the incorporation of kernel representatives while optimizing the maximum cluster coverage problem. By integrating with a simultaneous training-pruning paradigm, REPrune promotes efficient, progressive pruning throughout training CNNs, avoiding the conventional train-prune-finetune sequence. Experimental results highlight that REPrune performs better in computer vision tasks than existing methods, effectively achieving a balance between acceleration ratio and performance retention.

----

## [1622] ConceptBed: Evaluating Concept Learning Abilities of Text-to-Image Diffusion Models

**Authors**: *Maitreya Patel, Tejas Gokhale, Chitta Baral, Yezhou Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29371](https://doi.org/10.1609/aaai.v38i13.29371)

**Abstract**:

The ability to understand visual concepts and replicate and compose these concepts from images is a central goal for computer vision. Recent advances in text-to-image (T2I) models have lead to high definition and realistic image quality generation by learning from large databases of images and their descriptions. However, the evaluation of T2I models has focused on photorealism and limited qualitative measures of visual understanding. To quantify the ability of T2I models in learning and synthesizing novel visual concepts (a.k.a. personalized T2I), we introduce ConceptBed, a large-scale dataset that consists of 284 unique visual concepts, and 33K composite text prompts. Along with the dataset, we propose an evaluation metric, Concept Confidence Deviation (CCD), that uses the confidence of oracle concept classifiers to measure the alignment between concepts generated by T2I generators and concepts contained in target images. We evaluate visual concepts that are either objects, attributes, or styles, and also evaluate four dimensions of compositionality: counting, attributes, relations, and actions. Our human study shows that CCD is highly correlated with human understanding of concepts. Our results point to a trade-off between learning the concepts and preserving the compositionality which existing approaches struggle to overcome. The data, code, and interactive demo is available at: https://conceptbed.github.io/

----

## [1623] CrystalBox: Future-Based Explanations for Input-Driven Deep RL Systems

**Authors**: *Sagar Patel, Sangeetha Abdu Jyothi, Nina Narodytska*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29372](https://doi.org/10.1609/aaai.v38i13.29372)

**Abstract**:

We present CrystalBox, a novel, model-agnostic, posthoc explainability framework for Deep Reinforcement Learning (DRL) controllers in the large family of input-driven environments which includes computer systems. We combine the natural decomposability of reward functions in input-driven environments with the explanatory power of decomposed returns. We propose an efficient algorithm to generate future-based explanations across both discrete and continuous control environments. Using applications such as adaptive bitrate streaming and congestion control, we demonstrate CrystalBox's capability to generate high-fidelity explanations. We further illustrate its higher utility across three practical use cases: contrastive explanations, network observability, and guided reward design, as opposed to prior explainability techniques that identify salient features.

----

## [1624] HAGO-Net: Hierarchical Geometric Massage Passing for Molecular Representation Learning

**Authors**: *Hongbin Pei, Taile Chen, Chen A, Huiqi Deng, Jing Tao, Pinghui Wang, Xiaohong Guan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29373](https://doi.org/10.1609/aaai.v38i13.29373)

**Abstract**:

Molecular representation learning has emerged as a game-changer at the intersection of AI and chemistry, with great potential in applications such as  drug design and materials discovery. A substantial obstacle in successfully applying molecular representation learning is the difficulty of effectively and completely characterizing and learning molecular geometry, which has not been well addressed to date.  To overcome this challenge, we propose a novel framework that features a novel geometric graph, termed HAGO-Graph, and a specifically designed geometric graph learning model, HAGO-Net. In the framework, the foundation is HAGO-Graph, which enables a complete characterization of molecular geometry in a hierarchical manner. Specifically, we leverage the concept of n-body in physics to characterize geometric patterns at multiple spatial scales. We then specifically design a message passing scheme, HAGO-MPS, and implement the scheme as a geometric graph neural network, HAGO-Net, to effectively learn the representation of HAGO-Graph by horizontal and vertical aggregation. We further prove DHAGO-Net, the derivative function of HAGO-Net, is an equivariant model. The proposed models are validated by extensive comparisons on four challenging benchmarks. Notably, the models exhibited state-of-the-art performance in molecular chirality identification and  property prediction, achieving state-of-the-art performance on five properties of QM9 dataset.  The models also achieved competitive results on molecular dynamics prediction task.

----

## [1625] CARAT: Contrastive Feature Reconstruction and Aggregation for Multi-Modal Multi-Label Emotion Recognition

**Authors**: *Cheng Peng, Ke Chen, Lidan Shou, Gang Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29374](https://doi.org/10.1609/aaai.v38i13.29374)

**Abstract**:

Multi-modal multi-label emotion recognition (MMER) aims to identify relevant emotions from multiple modalities. The challenge of MMER is how to effectively capture discriminative features for multiple labels from heterogeneous data. Recent studies are mainly devoted to exploring various fusion strategies to integrate multi-modal information into a unified representation for all labels. However, such a learning scheme not only overlooks the specificity of each modality but also fails to capture individual discriminative features for different labels. Moreover, dependencies of labels and modalities cannot be effectively modeled. To address these issues, this paper presents ContrAstive feature Reconstruction and AggregaTion (CARAT) for the MMER task. Specifically, we devise a reconstruction-based fusion mechanism to better model fine-grained modality-to-label dependencies by contrastively learning modal-separated and label-specific features. To further exploit the modality complementarity, we introduce a shuffle-based aggregation strategy to enrich co-occurrence collaboration among labels. Experiments on two benchmark datasets CMU-MOSEI and M3ED demonstrate the effectiveness of CARAT over state-of-the-art methods. Code is available at https://github.com/chengzju/CARAT.

----

## [1626] Variational Hybrid-Attention Framework for Multi-Label Few-Shot Aspect Category Detection

**Authors**: *Cheng Peng, Ke Chen, Lidan Shou, Gang Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29375](https://doi.org/10.1609/aaai.v38i13.29375)

**Abstract**:

Multi-label few-shot aspect category detection (FS-ACD) is a challenging sentiment analysis task, which aims to learn a multi-label learning paradigm with limited training data. The difficulty of this task is how to use limited data to generalize effective discriminative representations for different categories. Nowadays, all advanced FS-ACD works utilize the prototypical network to learn label prototypes to represent different aspects. However, such point-based estimation methods are inherently noise-susceptible and bias-vulnerable. To this end, this paper proposes a novel Variational Hybrid-Attention Framework (VHAF) for the FS-ACD task. Specifically, to alleviate the data noise, we adopt a hybrid-attention mechanism to generate more discriminative aspect-specific embeddings. Then, based on these embeddings, we introduce the variational distribution inference to obtain the aspect-specific distribution as a more robust aspect representation, which can eliminate the scarce data bias for better inference. Moreover, we further leverage an adaptive threshold estimation to help VHAF better identify multiple relevant aspects. Extensive experiments on three datasets demonstrate the effectiveness of our VHAF over other state-of-the-art methods. Code is available at https://github.com/chengzju/VHAF.

----

## [1627] Hypothesis, Verification, and Induction: Grounding Large Language Models with Self-Driven Skill Learning

**Authors**: *Shaohui Peng, Xing Hu, Qi Yi, Rui Zhang, Jiaming Guo, Di Huang, Zikang Tian, Ruizhi Chen, Zidong Du, Qi Guo, Yunji Chen, Ling Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29376](https://doi.org/10.1609/aaai.v38i13.29376)

**Abstract**:

Large language models (LLMs) show their powerful automatic reasoning and planning capability with a wealth of semantic knowledge about the human world. However, the grounding problem still hinders the applications of LLMs in the real-world environment. Existing studies try to fine-tune the LLM or utilize pre-defined behavior APIs to bridge the LLMs and the environment, which not only costs huge human efforts to customize for every single task but also weakens the generality strengths of LLMs. To autonomously ground the LLM onto the environment, we proposed the Hypothesis, Verification, and Induction (HYVIN) framework to automatically and progressively ground the LLM with self-driven skill learning. HYVIN first employs the LLM to propose the hypothesis of sub-goals to achieve tasks and then verify the feasibility of the hypothesis via interacting with the underlying environment. Once verified, HYVIN can then learn generalized skills with the guidance of these successfully grounded subgoals. These skills can be further utilized to accomplish more complex tasks that fail to pass the verification phase. Verified in the famous instruction following task set, BabyAI, HYVIN achieves comparable performance in the most challenging tasks compared with imitation learning methods that cost millions of demonstrations, proving the effectiveness of learned skills and showing the feasibility and efficiency of our framework.

----

## [1628] Recurrent Graph Neural Networks and Their Connections to Bisimulation and Logic

**Authors**: *Maximilian Pflueger, David Tena Cucala, Egor V. Kostylev*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29377](https://doi.org/10.1609/aaai.v38i13.29377)

**Abstract**:

The success of Graph Neural Networks (GNNs) in practice has motivated extensive research on their theoretical properties. This includes recent results that characterise node classifiers expressible by GNNs in terms of first order logic. Most of the analysis, however, has been focused on GNNs with fixed number of message-passing iterations (i.e., layers), which cannot realise many simple classifiers such as reachability of a node with a given label. In this paper, we start to fill this gap and study the foundations of GNNs that can perform more than a fixed number of message-passing iterations. We first formalise two generalisations of the basic GNNs: recurrent GNNs (RecGNNs), which repeatedly apply message-passing iterations until the node classifications become stable, and graph-size GNNs (GSGNNs), which exploit a built-in function of the input graph size to decide the number of message-passings. We then formally prove that GNN classifiers are strictly less expressive than RecGNN ones, and RecGNN classifiers are strictly less expressive than GSGNN ones. To get this result, we identify novel semantic characterisations of the three formalisms in terms of suitable variants of bisimulation, which we believe have their own value for our understanding of GNNs. Finally, we prove syntactic logical characterisations of RecGNNs and GSGNNs analogous to the logical characterisation of plain GNNs, where we connect the two formalisms to monadic monotone fixpoint logic---a generalisation of first-order logic that supports recursion.

----

## [1629] Learning Performance Maximizing Ensembles with Explainability Guarantees

**Authors**: *Vincent Pisztora, Jia Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29378](https://doi.org/10.1609/aaai.v38i13.29378)

**Abstract**:

In this paper we propose a method for the optimal allocation of observations between an intrinsically explainable glass box model and a black box model. An optimal allocation being defined as one which, for any given explainability level (i.e. the proportion of observations for which the explainable model is the prediction function), maximizes the performance of the ensemble on the underlying task, and maximizes performance of the explainable model on the observations allocated to it, subject to the maximal ensemble performance condition. The proposed method is shown to produce such explainability optimal allocations on a benchmark suite of tabular datasets across a variety of explainable and black box model types. These learned allocations are found to consistently maintain ensemble performance at very high explainability levels (explaining 74% of observations on average), and in some cases even outperform both the component explainable and black box models while improving explainability.

----

## [1630] Reconciling Predictive and Statistical Parity: A Causal Approach

**Authors**: *Drago Plecko, Elias Bareinboim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29379](https://doi.org/10.1609/aaai.v38i13.29379)

**Abstract**:

Since the rise of fair machine learning as a critical field of inquiry, many different notions on how to quantify and measure discrimination have been proposed in the literature. Some of these notions, however, were shown to be mutually incompatible. Such findings make it appear that numerous different kinds of fairness exist, thereby making a consensus on the appropriate measure of fairness harder to reach, hindering the applications of these tools in practice. In this paper, we investigate one of these key impossibility results that relates the notions of statistical and predictive parity. Specifically, we derive a new causal decomposition formula for the fairness measures associated with predictive parity, and obtain a novel insight into how this criterion is related to statistical parity through the legal doctrines of disparate treatment, disparate impact, and the notion of business necessity. Our results show that through a more careful causal analysis, the notions of statistical and predictive parity are not really mutually exclusive, but complementary and spanning a spectrum of fairness notions through the concept of business necessity. Finally, we demonstrate the importance of our findings on a real-world example.

----

## [1631] Adaptive Feature Imputation with Latent Graph for Deep Incomplete Multi-View Clustering

**Authors**: *Jingyu Pu, Chenhang Cui, Xinyue Chen, Yazhou Ren, Xiaorong Pu, Zhifeng Hao, Philip S. Yu, Lifang He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29380](https://doi.org/10.1609/aaai.v38i13.29380)

**Abstract**:

In recent years, incomplete multi-view clustering (IMVC), which studies the challenging multi-view clustering problem on missing views, has received growing research interests. Previous IMVC methods suffer from the following issues: (1) the inaccurate imputation for missing data, which leads to suboptimal clustering performance, and (2) most existing IMVC models merely consider the explicit presence of graph structure in data, ignoring the fact that latent graphs of different views also provide valuable information for the clustering task. To overcome such challenges, we present a novel method, termed Adaptive feature imputation with latent graph for incomplete multi-view clustering (AGDIMC). Specifically, it captures the embbedded features of each view by incorporating the view-specific deep encoders. Then, we construct partial latent graphs on complete data, which can consolidate the intrinsic relationships within each view while preserving the topological information. With the aim of estimating the missing sample based on the available information, we utilize an adaptive imputation layer to impute the embedded feature of missing data by using cross-view soft cluster assignments and global cluster centroids. As the imputation progresses, the portion of complete data increases, contributing to enhancing the discriminative information contained in global pseudo-labels. Meanwhile, to alleviate the negative impact caused by inferior impute samples and the discrepancy of cluster structures, we further design an adaptive imputation strategy based on the global pseudo-label and the local cluster assignment. Experimental results on multiple real-world datasets demonstrate the effectiveness of our method over existing approaches.

----

## [1632] MDGNN: Multi-Relational Dynamic Graph Neural Network for Comprehensive and Dynamic Stock Investment Prediction

**Authors**: *Hao Qian, Hongting Zhou, Qian Zhao, Hao Chen, Hongxiang Yao, Jingwei Wang, Ziqi Liu, Fei Yu, Zhiqiang Zhang, Jun Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29381](https://doi.org/10.1609/aaai.v38i13.29381)

**Abstract**:

The stock market is a crucial component of the financial system, but predicting the movement of stock prices is challenging due to the dynamic and intricate relations arising from various aspects such as economic indicators, financial reports, global news, and investor sentiment. Traditional sequential methods and graph-based models have been applied in stock movement prediction, but they have limitations in capturing the multifaceted and temporal influences in stock price movements. To address these challenges, the Multi-relational Dynamic Graph Neural Network (MDGNN) framework is proposed, which utilizes a discrete dynamic graph to comprehensively capture multifaceted relations among stocks and their evolution over time. The representation generated from the graph offers a complete perspective on the interrelationships among stocks and associated entities. Additionally, the power of the Transformer structure is leveraged to encode the temporal evolution of multiplex relations, providing a dynamic and effective approach to predicting stock investment. Further, our proposed MDGNN framework achieves the best performance in public datasets compared with the state-of-the-art stock investment methods.

----

## [1633] Towards Modeling Uncertainties of Self-Explaining Neural Networks via Conformal Prediction

**Authors**: *Wei Qian, Chenxu Zhao, Yangyi Li, Fenglong Ma, Chao Zhang, Mengdi Huai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29382](https://doi.org/10.1609/aaai.v38i13.29382)

**Abstract**:

Despite the recent progress in deep neural networks (DNNs), it remains challenging to explain the predictions made by DNNs. Existing explanation methods for DNNs mainly focus on post-hoc explanations where another explanatory model is employed to provide explanations. The fact that post-hoc methods can fail to reveal the actual original reasoning process of DNNs raises the need to build DNNs with built-in interpretability. Motivated by this, many self-explaining neural networks have been proposed to generate not only accurate predictions but also clear and intuitive insights into why a particular decision was made. However, existing self-explaining networks are limited in providing distribution-free uncertainty quantification for the two simultaneously generated prediction outcomes (i.e., a sample's final prediction and its corresponding explanations for interpreting that prediction). Importantly, they also fail to establish a connection between the confidence values assigned to the generated explanations in the interpretation layer and those allocated to the final predictions in the ultimate prediction layer. To tackle the aforementioned challenges, in this paper, we design a novel uncertainty modeling framework for self-explaining networks, which not only demonstrates strong distribution-free uncertainty modeling performance for the generated explanations in the interpretation layer but also excels in producing efficient and effective prediction sets for the final predictions based on the informative high-level basis explanations. We perform the theoretical analysis for the proposed framework. Extensive experimental evaluation demonstrates the effectiveness of the proposed uncertainty framework.

----

## [1634] Upper Bounding Barlow Twins: A Novel Filter for Multi-Relational Clustering

**Authors**: *Xiaowei Qian, Bingheng Li, Zhao Kang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29383](https://doi.org/10.1609/aaai.v38i13.29383)

**Abstract**:

Multi-relational clustering is a challenging task due to the fact that diverse semantic information conveyed in multi-layer graphs is difficult to extract and fuse. Recent methods integrate topology structure and node attribute information through graph filtering. However, they often use a low-pass filter without fully considering the correlation among multiple graphs. To overcome this drawback, we propose to learn a graph filter motivated by the theoretical analysis of Barlow Twins. We find that input with a negative semi-definite inner product provides a lower bound for Barlow Twins loss, which prevents it from reaching a better solution. We thus learn a filter that yields an upper bound for Barlow Twins. Afterward, we design a simple clustering architecture and demonstrate its state-of-the-art performance on four benchmark datasets. The source code is available at https://github.com/XweiQ/BTGF.

----

## [1635] EarnHFT: Efficient Hierarchical Reinforcement Learning for High Frequency Trading

**Authors**: *Molei Qin, Shuo Sun, Wentao Zhang, Haochong Xia, Xinrun Wang, Bo An*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29384](https://doi.org/10.1609/aaai.v38i13.29384)

**Abstract**:

High-frequency trading (HFT) is using computer algorithms to make trading decisions in short time scales (e.g., second-level), which is widely used in the Cryptocurrency (Crypto) market, (e.g., Bitcoin). Reinforcement learning (RL) in financial research has shown stellar performance on many quantitative trading tasks. However, most methods focus on low-frequency trading, e.g., day-level, which cannot be directly applied to HFT because of two challenges. First, RL for HFT involves dealing with extremely long trajectories (e.g., 2.4 million steps per month), which is hard to optimize and evaluate. Second, the dramatic price fluctuations and market trend changes of Crypto make existing algorithms fail to maintain satisfactory performances. To tackle these challenges, we propose an Efficient hieArchical Reinforcement learNing method for High Frequency Trading (EarnHFT), a novel three-stage hierarchical RL framework for HFT. In stage I, we compute a Q-teacher, i.e., the optimal action value based on dynamic programming, for enhancing the performance and training efficiency of second level RL agents. In stage II, we construct a pool of diverse RL agents for different market trends, distinguished by return rates, where hundreds of RL agents are trained with different preferences of return rates and only a tiny fraction of them will be selected into the pool based on their profitability. In stage III, we train a minute-level router which dynamically picks a second-level agent from the pool to achieve stable performance across different markets. Through extensive experiments in various market trends on Crypto markets in a high-fidelity simulation trading environment, we demonstrate that EarnHFT significantly outperforms 6 state-of-art baselines in 6 popular financial criteria, exceeding the runner-up by 30% in profitability.

----

## [1636] Resisting Backdoor Attacks in Federated Learning via Bidirectional Elections and Individual Perspective

**Authors**: *Zhen Qin, Feiyi Chen, Chen Zhi, Xueqiang Yan, Shuiguang Deng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29385](https://doi.org/10.1609/aaai.v38i13.29385)

**Abstract**:

Existing approaches defend against backdoor attacks in federated learning (FL) mainly through a) mitigating the impact of infected models, or b) excluding infected models. The former negatively impacts model accuracy, while the latter usually relies on globally clear boundaries between benign and infected model updates. However, in reality, model updates can easily become mixed and scattered throughout due to the diverse distributions of local data. This work focuses on excluding infected models in FL. Unlike previous perspectives from a global view, we propose Snowball, a novel anti-backdoor FL framework through bidirectional elections from an individual perspective inspired by one principle deduced by us and two principles in FL and deep learning. It is characterized by a) bottom-up election, where each candidate model update votes to several peer ones such that a few model updates are elected as selectees for aggregation; and b) top-down election, where selectees progressively enlarge themselves through picking up from the candidates. We compare Snowball with state-of-the-art defenses to backdoor attacks in FL on five real-world datasets, demonstrating its superior resistance to backdoor attacks and slight impact on the accuracy of the global model.

----

## [1637] Tree Search-Based Evolutionary Bandits for Protein Sequence Optimization

**Authors**: *Jiahao Qiu, Hui Yuan, Jinghong Zhang, Wentao Chen, Huazheng Wang, Mengdi Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29386](https://doi.org/10.1609/aaai.v38i13.29386)

**Abstract**:

While modern biotechnologies allow synthesizing new proteins and function measurements at scale, efficiently exploring a protein sequence space and engineering it remains a daunting task due to the vast sequence space of any given protein. Protein engineering is typically conducted through an iterative process of adding mutations to the wild-type or lead sequences, recombination of mutations, and running new rounds of screening. To enhance the efficiency of such a process, we propose a tree search-based bandit learning method, which expands a tree starting from the initial sequence with the guidance of a bandit machine learning model. Under simplified assumptions and a Gaussian Process prior, we provide theoretical analysis and a Bayesian regret bound, demonstrating that the method can efficiently discover a near-optimal design. The full algorithm is compatible with a suite of randomized tree search heuristics, machine learning models, pre-trained embeddings, and bandit techniques. We test various instances of the algorithm across benchmark protein datasets using simulated screens. Experiment results demonstrate that the algorithm is both sample-efficient, diversity-promoting, and able to find top designs using reasonably small mutation counts.

----

## [1638] Multi-Level Cross-Modal Alignment for Image Clustering

**Authors**: *Liping Qiu, Qin Zhang, Xiaojun Chen, Shaotian Cai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29387](https://doi.org/10.1609/aaai.v38i13.29387)

**Abstract**:

Recently, the cross-modal pretraining model has been employed to produce meaningful pseudo-labels to supervise the training of an image clustering model. However, numerous erroneous alignments in a cross-modal pretraining model could produce poor-quality pseudo labels and degrade clustering performance. To solve the aforementioned issue, we propose a novel Multi-level Cross-modal Alignment method to improve the alignments in a cross-modal pretraining model for downstream tasks, by building a smaller but better semantic space and aligning the images and texts in three levels, i.e., instance-level, prototype-level, and semantic-level. Theoretical results show that our proposed method converges, and suggests effective means to reduce the expected clustering risk of our method. Experimental results on five benchmark datasets clearly show the superiority of our new method.

----

## [1639] Integer Is Enough: When Vertical Federated Learning Meets Rounding

**Authors**: *Pengyu Qiu, Yuwen Pu, Yongchao Liu, Wenyan Liu, Yun Yue, Xiaowei Zhu, Lichun Li, Jinbao Li, Shouling Ji*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29388](https://doi.org/10.1609/aaai.v38i13.29388)

**Abstract**:

Vertical Federated Learning (VFL) is a solution increasingly used by companies with the same user group but differing features, enabling them to collaboratively train a machine learning model. 
VFL ensures that clients exchange intermediate results extracted by their local models, without sharing raw data. 
However, in practice, VFL encounters several challenges, such as computational and communication overhead, privacy leakage risk, and adversarial attack. 
Our study reveals that the usage of floating-point (FP) numbers is a common factor causing these issues, as they can be redundant and contain too much information. 
To address this, we propose a new architecture called rounding layer, which converts intermediate results to integers. 
Our theoretical analysis and empirical results demonstrate the benefits of the rounding layer in reducing computation and memory overhead, providing privacy protection, preserving model performance, and mitigating adversarial attacks. 
We hope this paper inspires further research into novel architectures to address practical issues in VFL.

----

## [1640] Towards Multi-Mode Outlier Robust Tensor Ring Decomposition

**Authors**: *Yuning Qiu, Guoxu Zhou, Andong Wang, Zhenhao Huang, Qibin Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29389](https://doi.org/10.1609/aaai.v38i13.29389)

**Abstract**:

Conventional Outlier Robust Tensor Decomposition (ORTD) approaches generally represent sparse outlier corruption within a specific mode. However, such an assumption, which may hold for matrices, proves inadequate when applied to high-order tensors. In the tensor domain, the outliers are prone to be corrupted in multiple modes simultaneously. Addressing this limitation, this study proposes a novel ORTD approach by recovering low-rank tensors contaminated by outliers spanning multiple modes. In particular, we conceptualize outliers within high-order tensors as latent tensor group sparsity by decomposing the corrupted tensor into a sum of multiple latent components, where each latent component is exclusive to outliers within a particular direction. Thus, it can effectively mitigate the outlier corruptions prevalent in high-order tensors across multiple modes. To theoretically guarantee recovery performance, we rigorously analyze a non-asymptotic upper bound of the estimation error for the proposed ORTD approach. In the optimization process, we develop an efficient alternate direction method of multipliers (ADMM) algorithm. Empirical validation of the approach's efficacy is undertaken through comprehensive experimentation.

----

## [1641] Learning the Topology and Behavior of Discrete Dynamical Systems

**Authors**: *Zirou Qiu, Abhijin Adiga, Madhav V. Marathe, S. S. Ravi, Daniel J. Rosenkrantz, Richard Edwin Stearns, Anil Vullikanti*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29390](https://doi.org/10.1609/aaai.v38i13.29390)

**Abstract**:

Discrete dynamical systems are commonly used to model the spread of contagions on real-world networks. Under the PAC framework, existing research has studied the problem of learning the behavior of a system, assuming that the underlying network is known. In this work, we focus on a more challenging setting: to learn both the behavior and the underlying topology of a black-box system. We show that, in general, this learning problem is computationally intractable. On the positive side, we present efficient learning methods under the PAC model when the underlying graph of the dynamical system belongs to certain classes. Further, we examine a relaxed setting where the topology of an unknown system is partially observed. For this case, we develop an efficient PAC learner to infer the system and establish the sample complexity. Lastly, we present a formal analysis of the expressive power of the hypothesis class of dynamical systems where both the topology and behavior are unknown, using the well-known Natarajan dimension formalism. Our results provide a theoretical foundation for learning both the topology and behavior of discrete dynamical systems.

----

## [1642] LDS2AE: Local Diffusion Shared-Specific Autoencoder for Multimodal Remote Sensing Image Classification with Arbitrary Missing Modalities

**Authors**: *Jiahui Qu, Yuanbo Yang, Wenqian Dong, Yufei Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29391](https://doi.org/10.1609/aaai.v38i13.29391)

**Abstract**:

Recent research on the joint classification of multimodal remote sensing data has achieved great success. However, due to the limitations imposed by imaging conditions, the case of missing modalities often occurs in practice. Most previous researchers regard the classification in case of different missing modalities as independent tasks. They train a specific classification model for each fixed missing modality by extracting multimodal joint representation, which cannot handle the classification of arbitrary (including multiple and random) missing modalities. In this work, we propose a local diffusion shared-specific autoencoder (LDS2AE), which solves the classification of arbitrary missing modalities with a single model. The LDS2AE captures the data distribution of different modalities to learn multimodal shared feature for classification by designing a novel local diffusion autoencoder which consists of a modality-shared encoder and several modality-specific decoders. The modality-shared encoder is designed to extract multimodal shared feature by employing the same parameters to map multimodal data into a shared subspace. The modality-specific decoders put the multimodal shared feature to reconstruct the image of each modality, which facilitates the shared feature to learn unique information of different modalities. In addition, we incorporate masked training to the diffusion autoencoder to achieve local diffusion, which significantly reduces the training cost of model. The approach is tested on widely-used multimodal remote sensing datasets, demonstrating the effectiveness of the proposed LDS2AE in addressing the classification of arbitrary missing modalities. The code is available at https://github.com/Jiahuiqu/LDS2AE.

----

## [1643] Dual-Level Curriculum Meta-Learning for Noisy Few-Shot Learning Tasks

**Authors**: *Xiaofan Que, Qi Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29392](https://doi.org/10.1609/aaai.v38i13.29392)

**Abstract**:

Few-shot learning (FSL) is essential in many practical applications. However, the limited training examples make the models more vulnerable to label noise, which can lead to poor generalization capability. To address this critical challenge, we propose a curriculum meta-learning model that employs a novel dual-level class-example sampling strategy to create a robust curriculum for adaptive task distribution formulation and robust model training. The dual-level framework proposes a heuristic class sampling criterion that measures pairwise class boundary complexity to form a class curriculum; it uses effective example sampling through an under-trained proxy model to form an example curriculum. By utilizing both class-level and example-level information, our approach is more robust to handle limited training data and noisy labels that commonly occur in few-shot learning tasks.
The model has efficient convergence behavior, which is verified through rigorous convergence analysis. Additionally, we establish a novel error bound through a hierarchical PAC-Bayesian analysis for curriculum meta-learning under noise. We conduct extensive experiments that demonstrate the effectiveness of our framework in outperforming existing noisy few-shot learning methods under various few-shot classification benchmarks. Our code is available at https://github.com/ritmininglab/DCML.

----

## [1644] DSD²: Can We Dodge Sparse Double Descent and Compress the Neural Network Worry-Free?

**Authors**: *Victor Quétu, Enzo Tartaglione*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29393](https://doi.org/10.1609/aaai.v38i13.29393)

**Abstract**:

Neoteric works have shown that modern deep learning models can exhibit a sparse double descent phenomenon. Indeed, as the sparsity of the model increases, the test performance first worsens since the model is overfitting the training data; then, the overfitting reduces, leading to an improvement in performance, and finally, the model begins to forget critical information, resulting in underfitting. Such a behavior prevents using traditional early stop criteria.

In this work, we have three key contributions. First, we propose a learning framework that avoids such a phenomenon and improves generalization. Second, we introduce an entropy measure providing more insights into the insurgence of this phenomenon and enabling the use of traditional stop criteria. Third, we provide a comprehensive quantitative analysis of contingent factors such as re-initialization methods, model width and depth, and dataset noise. The contributions are supported by empirical evidence in typical setups. Our code is available at https://github.com/VGCQ/DSD2.

----

## [1645] Fair Participation via Sequential Policies

**Authors**: *Reilly Raab, Ross Boczar, Maryam Fazel, Yang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29394](https://doi.org/10.1609/aaai.v38i13.29394)

**Abstract**:

Leading approaches to algorithmic fairness and policy-induced distribution shift are often misaligned with long-term objectives in sequential settings. We aim to correct these shortcomings by ensuring that both the objective and fairness constraints account for policy-induced distribution shift. First, we motivate this problem using an example in which individuals subject to algorithmic predictions modulate their willingness to participate with the policy maker. Fairness in this example is measured by the variance of group participation rates. Next, we develop a method for solving the resulting constrained, non-linear optimization problem and prove that this method converges to a fair, locally optimal policy given first-order information. Finally, we experimentally validate our claims in a semi-synthetic setting.

----

## [1646] Understanding the Generalization of Pretrained Diffusion Models on Out-of-Distribution Data

**Authors**: *Sai Niranjan Ramachandran, Rudrabha Mukhopadhyay, Madhav Agarwal, C. V. Jawahar, Vinay P. Namboodiri*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29395](https://doi.org/10.1609/aaai.v38i13.29395)

**Abstract**:

This work tackles the important task of understanding out-of-distribution behavior in two prominent types of generative models, i.e., GANs and Diffusion models. Understanding this behavior is crucial in understanding their broader utility and risks as these systems are increasingly deployed in our daily lives. Our first contribution is demonstrating that diffusion spaces outperform GANs' latent spaces in inverting high-quality OOD images. We also provide a theoretical analysis attributing this to the lack of prior holes in diffusion spaces. Our second significant contribution is to provide a theoretical hypothesis that diffusion spaces can be projected onto a bounded hypersphere, enabling image manipulation through geodesic traversal between inverted images. Our analysis shows that different geodesics share common attributes for the same manipulation, which we leverage to perform various image manipulations. We conduct thorough empirical evaluations to support and validate our claims. Finally, our third and final contribution introduces a novel approach to the few-shot sampling for out-of-distribution data by inverting a few images to sample from the cluster formed by the inverted latents. The proposed technique achieves state-of-the-art results for the few-shot generation task in terms of image quality. Our research underscores the promise of diffusion spaces in out-of-distribution imaging and offers avenues for further exploration. Please find more details about our project at \url{http://cvit.iiit.ac.in/research/projects/cvit-projects/diffusionOOD}

----

## [1647] GINN-LP: A Growing Interpretable Neural Network for Discovering Multivariate Laurent Polynomial Equations

**Authors**: *Nisal Ranasinghe, Damith A. Senanayake, Sachith Seneviratne, Malin Premaratne, Saman K. Halgamuge*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29396](https://doi.org/10.1609/aaai.v38i13.29396)

**Abstract**:

Traditional machine learning is generally treated as a black-box optimization problem and does not typically produce interpretable functions that connect inputs and outputs. However, the ability to discover such interpretable functions is desirable. In this work, we propose GINN-LP, an interpretable neural network to discover the form and coefficients of the underlying equation of a dataset, when the equation is assumed to take the form of a multivariate Laurent Polynomial. This is facilitated by a new type of interpretable neural network block, named the “power-term approximator block”, consisting of logarithmic and exponential activation functions. GINN-LP is end-to-end differentiable, making it possible to use backpropagation for training. We propose a neural network growth strategy that will enable finding the suitable number of terms in the Laurent polynomial that represents the data, along with sparsity regularization to promote the discovery of concise equations. To the best of our knowledge, this is the first model that can discover arbitrary multivariate Laurent polynomial terms without any prior information on the order. Our approach is first evaluated on a subset of data used in SRBench, a benchmark for symbolic regression. We first show that GINN-LP outperforms the state-of-the-art symbolic regression methods on datasets generated using 48 real-world equations in the form of multivariate Laurent polynomials. Next, we propose an ensemble method that combines our method with a high-performing symbolic regression method, enabling us to discover non-Laurent polynomial equations. We achieve state-of-the-art results in equation discovery, showing an absolute improvement of 7.1% over the best contender, by applying this ensemble method to 113 datasets within SRBench with known ground-truth equations.

----

## [1648] Using Stratified Sampling to Improve LIME Image Explanations

**Authors**: *Muhammad Rashid, Elvio G. Amparore, Enrico Ferrari, Damiano Verda*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29397](https://doi.org/10.1609/aaai.v38i13.29397)

**Abstract**:

We investigate the use of a stratified sampling approach for LIME Image, a popular model-agnostic explainable AI method for computer vision tasks, in order to reduce the artifacts generated by typical Monte Carlo sampling.
Such artifacts are due to the undersampling of the dependent variable in the synthetic neighborhood around the image being explained, which may result in inadequate explanations due to the impossibility of fitting a linear regressor on the sampled data.
We then highlight a connection with the Shapley theory, where similar arguments about undersampling and sample relevance were suggested in the past.
We derive all the formulas and adjustment factors required for an unbiased stratified sampling estimator. 
Experiments show the efficacy of the proposed approach.

----

## [1649] NESTER: An Adaptive Neurosymbolic Method for Causal Effect Estimation

**Authors**: *Abbavaram Gowtham Reddy, Vineeth N. Balasubramanian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29398](https://doi.org/10.1609/aaai.v38i13.29398)

**Abstract**:

Causal effect estimation from observational data is a central problem in causal inference. Methods based on potential outcomes framework solve this problem by exploiting inductive biases and heuristics from causal inference. Each of these methods addresses a specific aspect of causal effect estimation, such as controlling propensity score, enforcing randomization, etc., by designing neural network (NN) architectures and regularizers. In this paper, we propose an adaptive method called Neurosymbolic Causal Effect Estimator (NESTER), a generalized method for causal effect estimation. NESTER integrates the ideas used in existing methods based on multi-head NNs for causal effect estimation into one framework. We design a Domain Specific Language (DSL) tailored for causal effect estimation based on causal inductive biases used in literature. We conduct a theoretical analysis to investigate NESTER's efficacy in estimating causal effects. Our comprehensive empirical results show that NESTER performs better than state-of-the-art methods on benchmark datasets.

----

## [1650] Towards Learning and Explaining Indirect Causal Effects in Neural Networks

**Authors**: *Abbavaram Gowtham Reddy, Saketh Bachu, Harsharaj Pathak, Benin Godfrey L, Varshaneya V, Vineeth N. Balasubramanian, Satyanarayan Kar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29399](https://doi.org/10.1609/aaai.v38i13.29399)

**Abstract**:

Recently, there has been a growing interest in learning and explaining causal effects within Neural Network (NN) models. By virtue of NN architectures, previous approaches consider only direct and total causal effects assuming independence among input variables. We view an NN as a structural causal model (SCM) and extend our focus to include indirect causal effects by introducing feedforward connections among input neurons. We propose an ante-hoc method that captures and maintains direct, indirect, and total causal effects during NN model training. We also propose an algorithm for quantifying learned causal effects in an NN model and efficient approximation strategies for quantifying causal effects in high-dimensional data. Extensive experiments conducted on synthetic and real-world datasets demonstrate that the causal effects learned by our ante-hoc method better approximate the ground truth effects compared to existing methods.

----

## [1651] Towards Improved Proxy-Based Deep Metric Learning via Data-Augmented Domain Adaptation

**Authors**: *Li Ren, Chen Chen, Liqiang Wang, Kien A. Hua*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29400](https://doi.org/10.1609/aaai.v38i13.29400)

**Abstract**:

Deep Metric Learning (DML) plays an important role in modern computer vision research, where we learn a distance metric for a set of image representations. Recent DML techniques utilize the proxy to interact with the corresponding image samples in the embedding space. However, existing proxy-based DML methods focus on learning individual proxy-to-sample distance, while the overall distribution of samples and proxies lacks attention. In this paper, we present a novel proxy-based DML framework that focuses on aligning the sample and proxy distributions to improve the efficiency of proxy-based DML losses. Specifically, we propose the Data-Augmented Domain Adaptation (DADA) method to adapt the domain gap between the group of samples and proxies. To the best of our knowledge, we are the first to leverage domain adaptation to boost the performance of proxy-based DML. We show that our method can be easily plugged into existing proxy-based DML losses. Our experiments on benchmarks, including the popular CUB-200-2011, CARS196, Stanford Online Products, and In-Shop Clothes Retrieval, show that our learning algorithm significantly improves the existing proxy losses and achieves superior results compared to the existing methods. The code and Appendix are available at: https://github.com/Noahsark/DADA

----

## [1652] Statistical Spatially Inhomogeneous Diffusion Inference

**Authors**: *Yinuo Ren, Yiping Lu, Lexing Ying, Grant M. Rotskoff*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29401](https://doi.org/10.1609/aaai.v38i13.29401)

**Abstract**:

Inferring a diffusion equation from discretely observed measurements is a statistical challenge of significant importance in a variety of fields, from single-molecule tracking in biophysical systems to modeling financial instruments.
Assuming that the underlying dynamical process obeys a d-dimensional stochastic differential equation of the form dx_t = b(x_t)dt + \Sigma(x_t)dw_t, we propose neural network-based estimators of both the drift b and the spatially-inhomogeneous diffusion tensor D = \Sigma\Sigma^T/2 and provide statistical convergence guarantees when b and D are s-Hölder continuous. 
Notably, our bound aligns with the minimax optimal rate N^{-\frac{2s}{2s+d}} for nonparametric function estimation even in the presence of correlation within observational data, which necessitates careful handling when establishing fast-rate generalization bounds.
Our theoretical results are bolstered by numerical experiments demonstrating accurate inference of spatially-inhomogeneous diffusion tensors.

----

## [1653] Protect Your Score: Contact-Tracing with Differential Privacy Guarantees

**Authors**: *Rob Romijnders, Christos Louizos, Yuki M. Asano, Max Welling*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29402](https://doi.org/10.1609/aaai.v38i13.29402)

**Abstract**:

The pandemic in 2020 and 2021 had enormous economic and societal consequences, and studies show that contact tracing algorithms can be key in the early containment of the virus. While large strides have been made towards more effective contact tracing algorithms, we argue that privacy concerns currently hold deployment back. The essence of a contact tracing algorithm constitutes the communication of a risk score. Yet, it is precisely the communication and release of this score to a user that an adversary can leverage to gauge the private health status of an individual. We pinpoint a realistic attack scenario and propose a contact tracing algorithm with differential privacy guarantees against this attack. The algorithm is tested on the two most widely used agent-based COVID19 simulators and demonstrates superior performance in a wide range of settings. Especially for realistic test scenarios and while releasing each risk score with epsilon=1 differential privacy, we achieve a two to ten-fold reduction in the infection rate of the virus. To the best of our knowledge, this presents the first contact tracing algorithm with differential privacy guarantees when revealing risk scores for COVID19.

----

## [1654] Limitations of Face Image Generation

**Authors**: *Harrison Rosenberg, Shimaa Ahmed, Guruprasad V. Ramesh, Kassem Fawaz, Ramya Korlakai Vinayak*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29403](https://doi.org/10.1609/aaai.v38i13.29403)

**Abstract**:

Text-to-image diffusion models have achieved widespread popularity due to their unprecedented image generation capability. In particular, their ability to synthesize and modify human faces has spurred research into using generated face images in both training data augmentation and model performance assessments. In this paper, we study the efficacy and shortcomings of generative models in the context of face generation. Utilizing a combination of qualitative and quantitative measures, including embedding-based metrics and user studies, we present a framework to audit the characteristics of generated faces conditioned on a set of social attributes. We applied our framework on faces generated through state-of-the-art text-to-image diffusion models. We identify several limitations of face image generation that include faithfulness to the text prompt, demographic disparities, and distributional shifts. Furthermore, we present an analytical model that provides insights into how training data selection contributes to the performance of generative models. Our survey data and analytics code can be found online at https://github.com/wi-pi/Limitations_of_Face_Generation

----

## [1655] Scaling Up Semi-supervised Learning with Unconstrained Unlabelled Data

**Authors**: *Shuvendu Roy, Ali Etemad*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29404](https://doi.org/10.1609/aaai.v38i13.29404)

**Abstract**:

We propose UnMixMatch, a semi-supervised learning framework which can learn effective representations from unconstrained unlabelled data in order to scale up performance. Most existing semi-supervised methods rely on the assumption that labelled and unlabelled samples are drawn from the same distribution, which limits the potential for improvement through the use of free-living unlabeled data. Consequently, the generalizability and scalability of semi-supervised learning are often hindered by this assumption. Our method aims to overcome these constraints and effectively utilize unconstrained unlabelled data in semi-supervised learning. UnMixMatch consists of three main components: a supervised learner with hard augmentations that provides strong regularization, a contrastive consistency regularizer to learn underlying representations from the unlabelled data, and a self-supervised loss to enhance the representations that are learnt from the unlabelled data. We perform extensive experiments on 4 commonly used datasets and demonstrate superior performance over existing semi-supervised methods with a performance boost of 4.79%. Extensive ablation and sensitivity studies show the effectiveness and impact of each of the proposed components of our method. The code for our work is publicly available.

----

## [1656] SimPSI: A Simple Strategy to Preserve Spectral Information in Time Series Data Augmentation

**Authors**: *Hyun Ryu, Sunjae Yoon, Hee Suk Yoon, Eunseop Yoon, Chang D. Yoo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29405](https://doi.org/10.1609/aaai.v38i13.29405)

**Abstract**:

Data augmentation is a crucial component in training neural networks to overcome the limitation imposed by data size, and several techniques have been studied for time series. Although these techniques are effective in certain tasks, they have yet to be generalized to time series benchmarks. We find that current data augmentation techniques ruin the core information contained within the frequency domain. To address this issue, we propose a simple strategy to preserve spectral information (SimPSI) in time series data augmentation. SimPSI preserves the spectral information by mixing the original and augmented input spectrum weighted by a preservation map, which indicates the importance score of each frequency. Specifically, our experimental contributions are to build three distinct preservation maps: magnitude spectrum, saliency map, and spectrum-preservative map. We apply SimPSI to various time series data augmentations and evaluate its effectiveness across a wide range of time series benchmarks. Our experimental results support that SimPSI considerably enhances the performance of time series data augmentations by preserving core spectral information. The source code used in the paper is available at https://github.com/Hyun-Ryu/simpsi.

----

## [1657] Learning the Causal Structure of Networked Dynamical Systems under Latent Nodes and Structured Noise

**Authors**: *Augusto Santos, Diogo Rente, Rui Seabra, José M. F. Moura*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29406](https://doi.org/10.1609/aaai.v38i13.29406)

**Abstract**:

This paper considers learning the hidden causal network of a linear networked dynamical system (NDS) from the time series data at some of its nodes -- partial observability. The dynamics of the NDS are driven by colored noise that generates spurious associations across pairs of nodes, rendering the problem much harder. To address the challenge of noise correlation and partial observability, we assign to each pair of nodes a feature vector computed from the time series data of observed nodes. The feature embedding is engineered to yield structural consistency: there exists an affine hyperplane that consistently partitions the set of features, separating the feature vectors corresponding to connected pairs of nodes from those corresponding to disconnected pairs. The causal inference problem is thus addressed via clustering the designed features. We demonstrate with simple baseline supervised methods the competitive performance of the proposed causal inference mechanism under broad connectivity regimes and noise correlation levels, including a real world network.  Further, we devise novel technical guarantees of structural consistency for linear NDS under the considered regime.

----

## [1658] XKD: Cross-Modal Knowledge Distillation with Domain Alignment for Video Representation Learning

**Authors**: *Pritam Sarkar, Ali Etemad*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29407](https://doi.org/10.1609/aaai.v38i13.29407)

**Abstract**:

We present XKD, a novel self-supervised framework to learn meaningful representations from unlabelled videos. XKD is trained with two pseudo objectives. First, masked data reconstruction is performed to learn modality-specific representations from audio and visual streams. Next, self-supervised cross-modal knowledge distillation is performed between the two modalities through a teacher-student setup to learn complementary information. We introduce a novel domain alignment strategy to tackle domain discrepancy between audio and visual modalities enabling effective cross-modal knowledge distillation.
Additionally, to develop a general-purpose network capable of handling both audio and visual streams, modality-agnostic variants of XKD are introduced, which use the same pretrained backbone for different audio and visual tasks. Our proposed cross-modal knowledge distillation improves video action classification by 8% to 14% on UCF101, HMDB51, and Kinetics400. Additionally, XKD improves multimodal action classification by 5.5% on Kinetics-Sound. XKD shows state-of-the-art performance in sound classification on ESC50, achieving top-1 accuracy of 96.5%.

----

## [1659] Understanding and Leveraging the Learning Phases of Neural Networks

**Authors**: *Johannes Schneider, Mohit Prabhushankar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29408](https://doi.org/10.1609/aaai.v38i13.29408)

**Abstract**:

The learning dynamics of deep neural networks are not well understood. The information bottleneck (IB) theory proclaimed separate fitting and compression phases. But they have since been heavily debated. We comprehensively analyze the learning dynamics by investigating a layer's reconstruction ability of the input and prediction performance based on the evolution of parameters during training. We empirically show the existence of three phases using common datasets and architectures such as ResNet and VGG: (i) near constant reconstruction loss, (ii) decrease, and (iii) increase. We also derive an empirically grounded data model and prove the existence of phases for single-layer networks. Technically, our approach leverages classical complexity analysis. It differs from IB by relying on measuring reconstruction loss rather than information theoretic measures to relate information of intermediate layers and inputs. Our work implies a new best practice for transfer learning: We show empirically that the pre-training of a classifier should stop well before its performance is optimal.

----

## [1660] What Do Hebbian Learners Learn? Reduction Axioms for Iterated Hebbian Learning

**Authors**: *Caleb Schultz Kisby, Saúl A. Blanco, Lawrence S. Moss*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29409](https://doi.org/10.1609/aaai.v38i13.29409)

**Abstract**:

This paper is a contribution to neural network semantics, a foundational framework for neuro-symbolic AI. The key insight of this theory is that logical operators can be mapped to operators on neural network states. In this paper, we do this for a neural network learning operator. We map a dynamic operator [φ] to iterated Hebbian learning, a simple learning policy that updates a neural network by repeatedly applying Hebb's learning rule until the net reaches a fixed-point. Our main result is that we can "translate away" [φ]-formulas via reduction axioms. This means that completeness for the logic of iterated Hebbian learning follows from completeness of the base logic. These reduction axioms also provide (1) a human-interpretable description of iterated Hebbian learning as a kind of plausibility upgrade, and (2) an approach to building neural networks with guarantees on what they can learn.

----

## [1661] Leaving the Nest: Going beyond Local Loss Functions for Predict-Then-Optimize

**Authors**: *Sanket Shah, Bryan Wilder, Andrew Perrault, Milind Tambe*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29410](https://doi.org/10.1609/aaai.v38i13.29410)

**Abstract**:

Predict-then-Optimize is a framework for using machine learning to perform decision-making under uncertainty. The central research question it asks is, "How can we use the structure of a decision-making task to tailor ML models for that specific task?" To this end, recent work has proposed learning task-specific loss functions that capture this underlying structure. However, current approaches make restrictive assumptions about the form of these losses and their impact on ML model behavior. These assumptions both lead to approaches with high computational cost, and when they are violated in practice, poor performance. In this paper, we propose solutions to these issues, avoiding the aforementioned assumptions and utilizing the ML model's features to increase the sample efficiency of learning loss functions. We empirically show that our method achieves state-of-the-art results in four domains from the literature, often requiring an order of magnitude fewer samples than comparable methods from past work. Moreover, our approach outperforms the best existing method by nearly 200% when the localness assumption is broken.

----

## [1662] Phoneme Hallucinator: One-Shot Voice Conversion via Set Expansion

**Authors**: *Siyuan Shan, Yang Li, Amartya Banerjee, Junier B. Oliva*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29411](https://doi.org/10.1609/aaai.v38i13.29411)

**Abstract**:

Voice conversion (VC) aims at altering a person's voice to make it sound similar to the voice of another person while preserving linguistic content. Existing methods suffer from a dilemma between content intelligibility and speaker similarity; i.e., methods with higher intelligibility usually have a lower speaker similarity, while methods with higher speaker similarity usually require plenty of target speaker voice data to achieve high intelligibility. In this work, we propose a novel method Phoneme Hallucinator that achieves the best of both worlds. Phoneme Hallucinator is a one-shot VC model; it adopts a novel model to hallucinate diversified and high-fidelity target speaker phonemes based just on a short target speaker voice (e.g. 3 seconds). The hallucinated phonemes are then exploited to perform neighbor-based voice conversion. Our model is a text-free, any-to-any VC model that requires no text annotations and supports conversion to any unseen speaker. Quantitative and qualitative evaluations show that Phoneme Hallucinator outperforms existing VC methods for both intelligibility and speaker similarity.

----

## [1663] No Internal Regret with Non-convex Loss Functions

**Authors**: *Dravyansh Sharma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29412](https://doi.org/10.1609/aaai.v38i13.29412)

**Abstract**:

Internal regret is a measure of performance of an online learning algorithm, which measures the change in performance by substituting every occurrence of a given action i by an alternative action j. Algorithms for minimizing internal regret are known for the finite experts setting, including a general reduction to the problem of minimizing external regret for this case. The reduction however crucially depends on the finiteness of the action space. In this work we approach the problem of minimizing internal regret for a continuous action space. For the full information setting, we show how to obtain O(sqrt(T)) internal regret for the class of Lipschitz functions, as well as non-Lipschitz dispersed functions, i.e. the non-Lipschitzness may not concentrate in a small region of the action space. We also consider extensions to partial feedback settings, and again obtain sublinear internal regret. Finally we discuss applications of internal regret minimization over continuous spaces to correlated equilibria in pricing problems and auction design, as well as to data-driven hyperparameter tuning.

----

## [1664] Symbolic Cognitive Diagnosis via Hybrid Optimization for Intelligent Education Systems

**Authors**: *Junhao Shen, Hong Qian, Wei Zhang, Aimin Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29413](https://doi.org/10.1609/aaai.v38i13.29413)

**Abstract**:

Cognitive diagnosis assessment is a fundamental and crucial task for student learning. It models the student-exercise interaction, and discovers the students' proficiency levels on each knowledge attribute. In real-world intelligent education systems, generalization and interpretability of cognitive diagnosis methods are of equal importance. However, most existing methods can hardly make the best of both worlds due to the complicated student-exercise interaction. To this end, this paper proposes a symbolic cognitive diagnosis~(SCD) framework to simultaneously enhance generalization and interpretability. The SCD framework incorporates the symbolic tree to explicably represent the complicated student-exercise interaction function, and utilizes gradient-based optimization methods to effectively learn the student and exercise parameters. Meanwhile, the accompanying challenge is that we need to tunnel the discrete symbolic representation and continuous parameter optimization. To address this challenge, we propose to hybridly optimize the representation and parameters in an alternating manner. To fulfill SCD, it alternately learns the symbolic tree by derivative-free genetic programming and learns the student and exercise parameters via gradient-based Adam. The extensive experimental results on various real-world datasets show the superiority of SCD on both generalization and interpretability. The ablation study verifies the efficacy of each ingredient in SCD, and the case study explicitly showcases how the interpretable ability of SCD works.

----

## [1665] BBScore: A Brownian Bridge Based Metric for Assessing Text Coherence

**Authors**: *Zhecheng Sheng, Tianhao Zhang, Chen Jiang, Dongyeop Kang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29414](https://doi.org/10.1609/aaai.v38i13.29414)

**Abstract**:

Measuring the coherence of text is a vital aspect of evaluating the quality of written content. Recent advancements in neural coherence modeling have demonstrated their efficacy in capturing entity coreference and discourse relations, thereby enhancing coherence evaluation. However, many existing methods heavily depend on static embeddings or focus narrowly on nearby context, constraining their capacity to measure the overarching coherence of long texts.
In this paper, we posit that coherent texts inherently manifest a sequential and cohesive interplay among sentences, effectively conveying the central theme, purpose, or standpoint. To explore this abstract relationship, we introduce the "BB Score," a novel reference-free metric grounded in Brownian bridge theory for assessing text coherence. Our findings showcase that when synergized with a simple additional classification component, this metric attains a performance level comparable to state-of-the-art techniques on standard artificial discrimination tasks.
We also establish in downstream tasks that this metric effectively differentiates between human-written documents and text generated by large language models within specific domains. Furthermore, we illustrate the efficacy of this approach in detecting written styles attributed to various large language models, underscoring its potential for generalizability. In summary, we present a novel Brownian bridge coherence metric capable of measuring both local and global text coherence, while circumventing the need for end-to-end model training. This flexibility allows for its application in various downstream tasks.

----

## [1666] Building Variable-Sized Models via Learngene Pool

**Authors**: *Boyu Shi, Shiyu Xia, Xu Yang, Haokun Chen, Zhiqiang Kou, Xin Geng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29415](https://doi.org/10.1609/aaai.v38i13.29415)

**Abstract**:

Recently, Stitchable Neural Networks (SN-Net) is proposed to stitch some pre-trained networks for quickly building numerous networks with different complexity and performance trade-offs. In this way, the burdens of designing or training the variable-sized networks, which can be used in application scenarios with diverse resource constraints, are alleviated. However, SN-Net still faces a few challenges. 1) Stitching from multiple independently pre-trained anchors introduces high storage resource consumption. 2) SN-Net faces challenges to build smaller models for low resource constraints. 3). SN-Net uses an unlearned initialization method for stitch layers, limiting the final performance.
To overcome these challenges, motivated by the recently proposed Learngene framework, we propose a novel method called Learngene Pool. Briefly, Learngene distills the critical knowledge from a large pre-trained model into a small part (termed as learngene) and then expands this small part into a few variable-sized models. In our proposed method, we distill one pre-trained large model into multiple small models whose network blocks are used as learngene instances to construct the learngene pool. Since only one large model is used, we do not need to store more large models as SN-Net and after distilling, smaller learngene instances can be created to build small models to satisfy low resource constraints. We also insert learnable transformation matrices between the instances to stitch them into variable-sized models to improve the performance of these models. Exhaustive experiments have been implemented and the results validate the effectiveness of the proposed Learngene Pool compared with SN-Net.

----

## [1667] CLIP-Guided Federated Learning on Heterogeneity and Long-Tailed Data

**Authors**: *Jiangming Shi, Shanshan Zheng, Xiangbo Yin, Yang Lu, Yuan Xie, Yanyun Qu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29416](https://doi.org/10.1609/aaai.v38i13.29416)

**Abstract**:

Federated learning (FL) provides a decentralized machine learning paradigm where a server collaborates with a group of clients to learn a global model without accessing the clients' data. User heterogeneity is a significant challenge for FL, which together with the class-distribution imbalance further enhances the difficulty of FL. Great progress has been made in large vision-language models, such as Contrastive Language-Image Pre-training (CLIP), which paves a new way for image classification and object recognition. Inspired by the success of CLIP on few-shot and zero-shot learning, we use CLIP to optimize the federated learning between server and client models under its vision-language supervision. It is promising to mitigate the user heterogeneity and class-distribution balance due to the powerful cross-modality representation and rich open-vocabulary prior knowledge. In this paper, we propose the CLIP-guided FL (CLIP2FL) method on heterogeneous and long-tailed data. In CLIP2FL, the knowledge of the off-the-shelf CLIP model is transferred to the client-server models, and a bridge is built between the client and server. Specifically, for client-side learning, knowledge distillation is conducted between client models and CLIP to improve the ability of client-side feature representation. For server-side learning, in order to mitigate the heterogeneity and class-distribution imbalance, we generate federated features to retrain the server model. A prototype contrastive learning with the supervision of the text encoder of CLIP is introduced to generate federated features depending on the client-side gradients, and they are used to retrain a balanced server classifier. Extensive experimental results on several benchmarks demonstrate that CLIP2FL achieves impressive performance and effectively deals with data heterogeneity and long-tail distribution. The code is available at https://github.com/shijiangming1/CLIP2FL.

----

## [1668] Structural Information Enhanced Graph Representation for Link Prediction

**Authors**: *Lei Shi, Bin Hu, Deng Zhao, Jianshan He, Zhiqiang Zhang, Jun Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29417](https://doi.org/10.1609/aaai.v38i13.29417)

**Abstract**:

Link prediction is a fundamental task of graph machine learning, and Graph Neural Network (GNN) based methods have become the mainstream approach due to their good performance. However, the typical practice learns node representations through neighborhood aggregation, lacking awareness of the structural relationships between target nodes. Recently, some methods have attempted to address this issue by node labeling tricks. However, they still rely on the node-centric neighborhood message passing of GNNs, which we believe involves two limitations in terms of information perception and transmission for link prediction. First, it cannot perceive long-range structural information due to the restricted receptive fields. Second, there may be information loss of node-centric model on link-centric task. In addition, we empirically find that the neighbor node features could introduce noise for link prediction. To address these issues, we propose a structural information enhanced link prediction framework, which involves removing the neighbor node features while fitting neighborhood graph structures more focused through GNN. Furthermore, we introduce Binary Structural Transformer (BST) to encode the structural relationships between target nodes, complementing the deficiency of GNN. Our approach achieves remarkable results on multiple popular benchmarks, including ranking first on ogbl-ppa, ogbl-citation2 and Pubmed.

----

## [1669] A Closer Look at Curriculum Adversarial Training: From an Online Perspective

**Authors**: *Lianghe Shi, Weiwei Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29418](https://doi.org/10.1609/aaai.v38i13.29418)

**Abstract**:

Curriculum adversarial training empirically finds that gradually increasing the hardness of adversarial examples can further improve the adversarial robustness of the trained model compared to conventional adversarial training. However, theoretical understanding of this strategy remains limited. In an attempt to bridge this gap, we analyze the adversarial training process from an online perspective. Specifically, we treat adversarial examples in different iterations as samples from different adversarial distributions. We then introduce the time series prediction framework and deduce novel generalization error bounds. Our theoretical results not only demonstrate the effectiveness of the conventional adversarial training algorithm but also explain why curriculum adversarial training methods can further improve adversarial generalization. We conduct comprehensive experiments to support our theory.

----

## [1670] Double-Bounded Optimal Transport for Advanced Clustering and Classification

**Authors**: *Liangliang Shi, Zhaoqi Shen, Junchi Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29419](https://doi.org/10.1609/aaai.v38i13.29419)

**Abstract**:

Optimal transport (OT) is attracting increasing attention in machine learning. It aims to transport a source distribution to a target one at minimal cost. In its vanilla form, the source and target distributions are predetermined, which contracts to the real-world case involving undetermined targets. In this paper, we propose Doubly Bounded Optimal Transport (DB-OT), which assumes that the target distribution is restricted within two boundaries instead of a fixed one, thus giving more freedom for the transport to find solutions. Based on the entropic regularization of DB-OT, three scaling-based algorithms are devised for calculating the optimal solution. We also show that our DB-OT is helpful for barycenter-based clustering, which can avoid the excessive concentration of samples in a single cluster. Then we further develop DB-OT techniques for long-tailed classification which is an emerging and open problem. We first propose a connection between OT and classification, that is, in the classification task, training involves optimizing the Inverse OT to learn the representations, while testing involves optimizing the OT for predictions. with this OT perspective, we first apply DB-OT to improve the loss, and the Balanced Softmax is shown as a special case. Then we apply DB-OT for inference in the testing process. Even with vanilla Softmax trained features, our experiments show that our method can achieve good results with our improved inference scheme in the testing stage.

----

## [1671] Teacher as a Lenient Expert: Teacher-Agnostic Data-Free Knowledge Distillation

**Authors**: *Hyunjune Shin, Dong-Wan Choi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29420](https://doi.org/10.1609/aaai.v38i13.29420)

**Abstract**:

Data-free knowledge distillation (DFKD) aims to distill pretrained knowledge to a student model with the help of a generator without using original data. In such data-free scenarios, achieving stable performance of DFKD is essential due to the unavailability of validation data. Unfortunately, this paper has discovered that existing DFKD methods are quite sensitive to different teacher models, occasionally showing catastrophic failures of distillation, even when using well-trained teacher models. Our observation is that the generator in DFKD is not always guaranteed to produce precise yet diverse samples using the existing representative strategy of minimizing both class-prior and adversarial losses. Through our empirical study, we focus on the fact that class-prior not only decreases the diversity of generated samples, but also cannot completely address the problem of generating unexpectedly low-quality samples depending on teacher models. In this paper, we propose the teacher-agnostic data-free knowledge distillation (TA-DFKD) method, with the goal of more robust and stable performance regardless of teacher models. Our basic idea is to assign the teacher model a lenient expert role for evaluating samples, rather than a strict supervisor that enforces its class-prior on the generator. Specifically, we design a sample selection approach that takes only clean samples verified by the teacher model without imposing restrictions on the power of generating diverse samples. Through extensive experiments, we show that our method successfully achieves both robustness and training stability across various teacher models, while outperforming the existing DFKD methods.

----

## [1672] SemTra: A Semantic Skill Translator for Cross-Domain Zero-Shot Policy Adaptation

**Authors**: *Sangwoo Shin, Minjong Yoo, Jeongwoo Lee, Honguk Woo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29421](https://doi.org/10.1609/aaai.v38i13.29421)

**Abstract**:

This work explores the zero-shot adaptation capability of semantic skills, semantically interpretable experts' behavior patterns, in cross-domain settings, where a user input in interleaved multi-modal snippets can prompt a new long-horizon task for different domains. In these cross-domain settings, we present a semantic skill translator framework SemTra which utilizes a set of multi-modal models to extract skills from the snippets, and leverages the reasoning capabilities of a pretrained language model to adapt these extracted skills to the target domain. The framework employs a two-level hierarchy for adaptation: task adaptation and skill adaptation. During task adaptation, seq-to-seq translation by the language model transforms the extracted skills into a semantic skill sequence, which is tailored to fit the cross-domain contexts. Skill adaptation focuses on optimizing each semantic skill for the target domain context, through parametric instantiations that are facilitated by language prompting and contrastive learning-based context inferences. This hierarchical adaptation empowers the framework to not only infer a complex task specification in one-shot from the interleaved multi-modal snippets, but also adapt it to new domains with zero-shot learning abilities. We evaluate our framework with Meta-World, Franka Kitchen, RLBench, and CARLA environments. The results clarify the framework's superiority in performing long-horizon tasks and adapting to different domains, showing its broad applicability in practical use cases, such as cognitive robots interpreting abstract instructions and autonomous vehicles operating under varied configurations.

----

## [1673] Region-Disentangled Diffusion Model for High-Fidelity PPG-to-ECG Translation

**Authors**: *Debaditya Shome, Pritam Sarkar, Ali Etemad*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29422](https://doi.org/10.1609/aaai.v38i13.29422)

**Abstract**:

The high prevalence of cardiovascular diseases (CVDs) calls for accessible and cost-effective continuous cardiac monitoring tools. Despite Electrocardiography (ECG) being the gold standard, continuous monitoring remains a challenge, leading to the exploration of Photoplethysmography (PPG), a promising but more basic alternative available in consumer wearables. This notion has recently spurred interest in translating PPG to ECG signals. In this work, we introduce Region-Disentangled Diffusion Model (RDDM), a novel diffusion model designed to capture the complex temporal dynamics of ECG. Traditional Diffusion models like Denoising Diffusion Probabilistic Models (DDPM) face challenges in capturing such nuances due to the indiscriminate noise addition process across the entire signal. Our proposed RDDM overcomes such limitations by incorporating a novel forward process that selectively adds noise to specific regions of interest (ROI) such as QRS complex in ECG signals, and a reverse process that disentangles the denoising of ROI and non-ROI regions. Quantitative experiments demonstrate that RDDM can generate high-fidelity ECG from PPG in as few as 10 diffusion steps, making it highly effective and computationally efficient. Additionally, to rigorously validate the usefulness of the generated ECG signals, we introduce CardioBench, a comprehensive evaluation benchmark for a variety of cardiac-related tasks including heart rate and blood pressure estimation, stress classification,  and the detection of atrial fibrillation and diabetes. Our thorough experiments show that RDDM achieves state-of-the-art performance on CardioBench. To the best of our knowledge, RDDM is the first diffusion model for cross-modal signal-to-signal translation in the bio-signal domain.

----

## [1674] Fusing Conditional Submodular GAN and Programmatic Weak Supervision

**Authors**: *Kumar Shubham, Pranav Sastry, Prathosh AP*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29423](https://doi.org/10.1609/aaai.v38i13.29423)

**Abstract**:

Programmatic Weak Supervision (PWS) and generative models serve as crucial tools that enable researchers to maximize the utility of existing datasets without resorting to laborious data gathering and manual annotation processes. PWS uses various weak supervision techniques to estimate the underlying class labels of data, while generative models primarily concentrate on sampling from the underlying distribution of the given dataset. Although these methods have the potential to complement each other, they have mostly been studied independently.
    Recently, WSGAN proposed a mechanism to fuse these two models. Their approach utilizes the discrete latent factors of InfoGAN for the training of the label models and leverages the class-dependent information of the label models to generate images of specific classes. However, the disentangled latent factor learned by the InfoGAN may not necessarily be class specific and hence could potentially affect the label model's accuracy. Moreover, the prediction of the label model is often noisy in nature and can have a detrimental impact on the quality of images generated by GAN. In our work, we address these challenges by (i) implementing a noise-aware classifier using the pseudo labels generated by the label model, (ii) utilizing the prediction of the noise-aware classifier for training the label model as well as generation of class-conditioned images. Additionally, We also investigate the effect of training the classifier with a subset of the dataset within a defined uncertainty budget on pseudo labels. We accomplish this by formalizing the subset selection problem as submodular maximization with a knapsack constraint on the entropy of pseudo labels. We conduct experiments on multiple datasets and demonstrate the efficacy of our methods on several tasks vis-a-vis the current state-of-the-art methods. Our implementation is
available at https://github.com/kyrs/subpws-gan

----

## [1675] Partial Label Learning with a Partner

**Authors**: *Chongjie Si, Zekun Jiang, Xuehui Wang, Yan Wang, Xiaokang Yang, Wei Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29424](https://doi.org/10.1609/aaai.v38i13.29424)

**Abstract**:

In partial label learning (PLL), each instance is associated with a set of candidate labels among which only one is ground-truth. The majority of the existing works focuses on constructing robust classifiers to estimate the labeling confidence of candidate labels in order to identify the correct one. However, these methods usually struggle to rectify mislabeled samples. To help existing PLL methods identify and rectify mislabeled samples, in this paper, we introduce a novel partner classifier and propose a novel ``mutual supervision'' paradigm. Specifically, we instantiate the partner classifier predicated on the implicit fact that non-candidate labels of a sample should not be assigned to it, which is inherently accurate and has not been fully investigated in PLL. Furthermore, a novel collaborative term is formulated to link the base classifier and the partner one. During each stage of mutual supervision, both classifiers will blur each other's predictions through a blurring mechanism to prevent overconfidence in a specific label. Extensive experiments demonstrate that the performance and disambiguation ability of several well-established stand-alone and deep-learning based PLL approaches can be significantly improved by coupling with this learning paradigm.

----

## [1676] Online Submodular Maximization via Online Convex Optimization

**Authors**: *Tareq Si Salem, Gözde Özcan, Iasonas Nikolaou, Evimaria Terzi, Stratis Ioannidis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29425](https://doi.org/10.1609/aaai.v38i13.29425)

**Abstract**:

We study monotone submodular maximization under general matroid constraints in the online setting. We prove that online optimization of a large class of submodular functions, namely, threshold potential functions, reduces to online convex optimization (OCO). This is precisely because functions in this class admit a concave relaxation; as a result, OCO policies, coupled with an appropriate rounding scheme, can be used to achieve sublinear regret in the combinatorial setting. We also show that our reduction extends to many different versions of the online learning problem, including the dynamic regret, bandit, and optimistic-learning settings.

----

## [1677] Robustly Train Normalizing Flows via KL Divergence Regularization

**Authors**: *Kun Song, Ruben Solozabal, Hao Li, Martin Takác, Lu Ren, Fakhri Karray*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29426](https://doi.org/10.1609/aaai.v38i13.29426)

**Abstract**:

In this paper, we find that the training of Normalizing Flows (NFs) are easily affected by the outliers and a small number (or high dimensionality) of training samples. To solve this problem, we propose a Kullback–Leibler (KL) divergence regularization on the Jacobian matrix of NFs. We prove that such regularization is equivalent to adding a set of samples whose covariance matrix is the identity matrix to the training set. Thus, it reduces the negative influence of the outliers and the small sample number on the estimation of the covariance matrix, simultaneously. Therefore, our regularization makes the training of NFs robust. Ultimately, we evaluate the performance of NFs on out-of-distribution (OoD) detection tasks. The excellent results obtained demonstrate the effectiveness of the proposed regularization term. For example, with the help of the proposed regularization, the OoD detection score increases at most 30% compared with the one without the regularization.

----

## [1678] Non-exemplar Domain Incremental Object Detection via Learning Domain Bias

**Authors**: *Xiang Song, Yuhang He, Songlin Dong, Yihong Gong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29427](https://doi.org/10.1609/aaai.v38i13.29427)

**Abstract**:

Domain incremental object detection (DIOD) aims to gradually learn a unified object detection model from a dataset stream composed of different domains, achieving good performance in all encountered domains. The most critical obstacle to this goal is the catastrophic forgetting problem, where the performance of the model improves rapidly in new domains but deteriorates sharply in old ones after a few sessions. To address this problem, we propose a non-exemplar DIOD method named learning domain bias (LDB), which learns domain bias independently at each new session, avoiding saving examples from old domains. Concretely, a base model is first obtained through training during session 1. Then, LDB freezes the weights of the base model and trains individual domain bias for each new incoming domain, adapting the base model to the distribution of new domains. At test time, since the domain ID is unknown, we propose a domain selector based on nearest mean classifier (NMC), which selects the most appropriate domain bias for a test image. Extensive experimental evaluations on two series of datasets demonstrate the effectiveness of the proposed LDB method in achieving high accuracy on new and old domain datasets. The code is available at https://github.com/SONGX1997/LDB.

----

## [1679] Reinforcement Learning as a Parsimonious Alternative to Prediction Cascades: A Case Study on Image Segmentation

**Authors**: *Bharat Srikishan, Anika Tabassum, Srikanth Allu, Ramakrishnan Kannan, Nikhil Muralidhar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29428](https://doi.org/10.1609/aaai.v38i13.29428)

**Abstract**:

Deep learning architectures have achieved state-of-the-art (SOTA) performance on computer vision tasks such as object detection and image segmentation. This may be attributed to the use of over-parameterized, monolithic deep learning architectures executed on large datasets. Although such large architectures lead to increased accuracy, this is usually accompanied by a larger increase in computation and memory requirements during inference. While this is a non-issue in traditional machine learning (ML) pipelines, the recent confluence of machine learning and fields like the Internet of Things (IoT) has rendered such large architectures infeasible for execution in low-resource settings. For some datasets, large monolithic pipelines may be overkill for simpler inputs. To address this problem, previous efforts have proposed decision cascades where inputs are passed through models of increasing complexity until desired performance is achieved. However, we argue that cascaded prediction leads to sub-optimal throughput and increased computational cost due to wasteful intermediate computations. To address this, we propose PaSeR (Parsimonious Segmentation with Reinforcement Learning) a non-cascading, cost-aware learning pipeline as an efficient alternative to cascaded decision architectures. Through experimental evaluation on both real-world and standard datasets, we demonstrate that PaSeR achieves better accuracy while minimizing computational cost relative to cascaded models. Further, we introduce a new metric IoU/GigaFlop to evaluate the balance between cost and performance. On the real-world task of battery material phase segmentation, PaSeR yields a minimum performance improvement of 174% on the IoU/GigaFlop metric with respect to baselines. We also demonstrate PaSeR's adaptability to complementary models trained on a noisy MNIST dataset, where it achieved a minimum performance improvement on IoU/GigaFlop of 13.4% over SOTA models. Code and data are available at github.com/scailab/paser.

----

## [1680] United We Stand: Using Epoch-Wise Agreement of Ensembles to Combat Overfit

**Authors**: *Uri Stern, Daniel Shwartz, Daphna Weinshall*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29429](https://doi.org/10.1609/aaai.v38i13.29429)

**Abstract**:

Deep neural networks have become the method of choice for solving many classification tasks, largely because they can fit very complex functions defined over raw data. The downside of such powerful learners is the danger of overfit. In this paper, we introduce a novel ensemble classifier for deep networks that effectively overcomes overfitting by combining models generated at specific intermediate epochs during training. Our method allows for the incorporation of useful knowledge obtained by the models during the overfitting phase without deterioration of the general performance, which is usually missed when early stopping is used.

To motivate this approach, we begin with the theoretical analysis of a regression model, whose prediction - that the variance among classifiers increases when overfit occurs - is demonstrated empirically in deep networks in common use. Guided by these results, we construct a new ensemble-based prediction method, where the prediction is determined by the class that attains the most consensual prediction throughout the training epochs. Using multiple image and text classification datasets, we show that when regular ensembles suffer from overfit, our method eliminates the harmful reduction in generalization due to overfit, and often even surpasses the performance obtained by early stopping. Our method is easy to implement and can be integrated with any training scheme and architecture, without additional prior knowledge beyond the training set. It is thus a practical and useful tool to overcome overfit.

----

## [1681] Multi-Dimensional Fair Federated Learning

**Authors**: *Cong Su, Guoxian Yu, Jun Wang, Hui Li, Qingzhong Li, Han Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29430](https://doi.org/10.1609/aaai.v38i13.29430)

**Abstract**:

Federated learning (FL) has emerged as a promising collaborative and secure paradigm for training a model from decentralized data without compromising privacy. Group fairness and client fairness are two dimensions of fairness that are important for FL. Standard FL can result in disproportionate disadvantages for certain clients, and it still faces the challenge of treating different groups equitably in a population. The problem of privately training fair FL models without compromising the generalization capability of disadvantaged clients remains open. In this paper, we propose a method, called mFairFL, to address this problem and achieve group fairness and client fairness simultaneously. mFairFL leverages differential multipliers to construct an optimization objective for empirical risk minimization with fairness constraints. Before aggregating locally trained models, it first detects conflicts among their gradients, and then iteratively curates the direction and magnitude of gradients to mitigate these conflicts. Theoretical analysis proves mFairFL facilitates the fairness in model development. The experimental evaluations based on three benchmark datasets show significant advantages of mFairFL compared to seven state-of-the-art baselines.

----

## [1682] Sharpness-Aware Model-Agnostic Long-Tailed Domain Generalization

**Authors**: *Houcheng Su, Weihao Luo, Daixian Liu, Mengzhu Wang, Jing Tang, Junyang Chen, Cong Wang, Zhenghan Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29431](https://doi.org/10.1609/aaai.v38i13.29431)

**Abstract**:

Domain Generalization (DG) aims to improve the generalization ability of models trained on a specific group of source domains, enabling them to perform well on new, unseen target domains. Recent studies have shown that methods that converge to smooth optima can enhance the generalization performance of supervised learning tasks such as classification. In this study, we examine the impact of smoothness-enhancing formulations on domain adversarial training, which combines task loss and adversarial loss objectives. Our approach leverages the fact that converging to a smooth minimum with respect to task loss can stabilize the task loss and lead to better performance on unseen domains. Furthermore, we recognize that the distribution of objects in the real world often follows a long-tailed class distribution, resulting in a mismatch between machine learning models and our expectations of their performance on all classes of datasets with long-tailed class distributions. To address this issue, we consider the domain generalization problem from the perspective of the long-tail distribution and propose using the maximum square loss to balance different classes which can improve model generalizability. Our method's effectiveness is demonstrated through comparisons with state-of-the-art methods on various domain generalization datasets. Code: https://github.com/bamboosir920/SAMALTDG.

----

## [1683] Multiscale Attention Wavelet Neural Operator for Capturing Steep Trajectories in Biochemical Systems

**Authors**: *Jiayang Su, Junbo Ma, Songyang Tong, Enze Xu, Minghan Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29432](https://doi.org/10.1609/aaai.v38i13.29432)

**Abstract**:

In biochemical modeling, some foundational systems can exhibit sudden and profound behavioral shifts, such as the cellular signaling pathway models, in which the physiological responses promptly react to environmental changes, resulting in steep changes in their dynamic model trajectories. These steep changes are one of the major challenges in biochemical modeling governed by nonlinear differential equations. One promising way to tackle this challenge is converting the input data from the time domain to the frequency domain through Fourier Neural Operators, which enhances the ability to analyze data periodicity and regularity. However, the effectiveness of these Fourier based methods diminishes in scenarios with complex abrupt switches. To address this limitation, an innovative Multiscale Attention Wavelet Neural Operator (MAWNO) method is proposed in this paper, which comprehensively combines the attention mechanism with the versatile wavelet transforms to effectively capture these abrupt switches. Specifically, the wavelet transform scrutinizes data across multiple scales to extract the characteristics of abrupt signals into wavelet coefficients, while the self-attention mechanism is adeptly introduced to enhance the wavelet coefficients in high-frequency signals that can better characterize the abrupt switches. Experimental results substantiate MAWNO’s supremacy in terms of accuracy on three classical biochemical models featuring periodic and steep trajectories. https://github.com/SUDERS/MAWNO.

----

## [1684] GSENet: Global Semantic Enhancement Network for Lane Detection

**Authors**: *Junhao Su, Zhenghan Chen, Chenghao He, Dongzhi Guan, Changpeng Cai, Tongxi Zhou, Jiashen Wei, Wenhua Tian, Zhihuai Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29433](https://doi.org/10.1609/aaai.v38i13.29433)

**Abstract**:

Lane detection is the cornerstone of autonomous driving. Although existing methods have achieved promising results, there are still limitations in addressing challenging scenarios such as abnormal weather, occlusion, and curves. These scenarios with low visibility usually require to rely on the broad information of the entire scene provided by global semantics and local texture information to predict the precise position and shape of the lane lines. In this paper, we propose a Global Semantic Enhancement Network for lane detection, which involves a complete set of systems for feature extraction and global features transmission. Traditional methods for global feature extraction usually require deep convolution layer stacks. However, this approach of obtaining global features solely through a larger receptive field not only fails to capture precise global features but also leads to an overly deep model, which results in slow inference speed. To address these challenges, we propose a novel operation called the Global feature Extraction Module (GEM). Additionally, we introduce the Top Layer Auxiliary Module (TLAM) as a channel for feature distillation, which facilitates a bottom-up transmission of global features. Furthermore, we introduce two novel loss functions: the Angle Loss, which account for the angle between predicted and ground truth lanes, and the Generalized Line IoU Loss function that considers the scenarios where significant deviations occur between the prediction of lanes and ground truth in some harsh conditions. The experimental results reveal that the proposed method exhibits remarkable superiority over the current state-of-the-art techniques for lane detection.Our codes are available at:https://github.com/crystal250/GSENet.

----

## [1685] Federated Adaptive Prompt Tuning for Multi-Domain Collaborative Learning

**Authors**: *Shangchao Su, Mingzhao Yang, Bin Li, Xiangyang Xue*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29434](https://doi.org/10.1609/aaai.v38i13.29434)

**Abstract**:

Federated learning (FL) enables multiple clients to collaboratively train a global model without disclosing their data. Previous researches often require training the complete model parameters. However, the emergence of powerful pre-trained models makes it possible to achieve higher performance with fewer learnable parameters in FL. In this paper, we propose a federated adaptive prompt tuning algorithm, FedAPT, for multi-domain collaborative image classification with powerful foundation models, like CLIP. Compared with direct federated prompt tuning, our core idea is to adaptively unlock specific domain knowledge for each test sample in order to provide them with personalized prompts. To implement this idea, we design an adaptive prompt tuning module, which consists of a meta prompt, an adaptive network, and some keys. The server randomly generates a set of keys and assigns a unique key to each client. Then all clients cooperatively train the global adaptive network and meta prompt with the local datasets and the frozen keys. Ultimately, the global aggregation model can assign a personalized prompt to CLIP based on the domain features of each test sample. We perform extensive experiments on two multi-domain image classification datasets across two different settings -- supervised and unsupervised. The results show that FedAPT can achieve better performance with less than 10% of the number of parameters of the fully trained model, and the global model can perform well in diverse client domains simultaneously.

----

## [1686] Towards Real-World Test-Time Adaptation: Tri-net Self-Training with Balanced Normalization

**Authors**: *Yongyi Su, Xun Xu, Kui Jia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29435](https://doi.org/10.1609/aaai.v38i13.29435)

**Abstract**:

Test-Time Adaptation aims to adapt source domain model to testing data at inference stage with success demonstrated in adapting to unseen corruptions. However, these attempts may fail under more challenging real-world scenarios. Existing works mainly consider real-world test-time adaptation under non-i.i.d. data stream and continual domain shift. In this work, we first complement the existing real-world TTA protocol with a globally class imbalanced testing set. We demonstrate that combining all settings together poses new challenges to existing methods. We argue the failure of state-of-the-art methods is first caused by indiscriminately adapting normalization layers to imbalanced testing data. To remedy this shortcoming, we propose a balanced batchnorm layer to swap out the regular batchnorm at inference stage. The new batchnorm layer is capable of adapting without biasing towards majority classes. We are further inspired by the success of self-training (ST) in learning from unlabeled data and adapt ST for test-time adaptation. However, ST alone is prone to over adaption which is responsible for the poor performance under continual domain shift. Hence, we propose to improve self-training under continual domain shift by regularizing model updates with an anchored loss. The final TTA model, termed as TRIBE, is built upon a tri-net architecture with balanced batchnorm layers. We evaluate TRIBE on four datasets representing real-world TTA settings. TRIBE consistently achieves the state-of-the-art performance across multiple evaluation protocols. 
The code is available at https://github.com/Gorilla-Lab-SCUT/TRIBE.

----

## [1687] Unraveling Batch Normalization for Realistic Test-Time Adaptation

**Authors**: *Zixian Su, Jingwei Guo, Kai Yao, Xi Yang, Qiufeng Wang, Kaizhu Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29436](https://doi.org/10.1609/aaai.v38i13.29436)

**Abstract**:

While recent test-time adaptations exhibit efficacy by adjusting batch normalization to narrow domain disparities, their effectiveness diminishes with realistic mini-batches due to inaccurate target estimation. As previous attempts merely introduce source statistics to mitigate this issue, the fundamental problem of inaccurate target estimation still persists, leaving the intrinsic test-time domain shifts unresolved. This paper delves into the problem of mini-batch degradation. By unraveling batch normalization, we discover that the inexact target statistics largely stem from the substantially reduced class diversity in batch. Drawing upon this insight, we introduce a straightforward tool, Test-time Exponential Moving Average (TEMA), to bridge the class diversity gap between training and testing batches. Importantly, our TEMA adaptively extends the scope of typical methods beyond the current batch to incorporate a diverse set of class information, which in turn boosts an accurate target estimation. Built upon this foundation, we further design a novel layer-wise rectification strategy to consistently promote test-time performance. Our proposed method enjoys a unique advantage as it requires neither training nor tuning parameters, offering a truly hassle-free solution. It significantly enhances model robustness against shifted domains and maintains resilience in diverse real-world scenarios with various  batch sizes, achieving state-of-the-art performance on several major benchmarks.  Code is available at https://github.com/kiwi12138/RealisticTTA.

----

## [1688] CUDC: A Curiosity-Driven Unsupervised Data Collection Method with Adaptive Temporal Distances for Offline Reinforcement Learning

**Authors**: *Chenyu Sun, Hangwei Qian, Chunyan Miao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29437](https://doi.org/10.1609/aaai.v38i13.29437)

**Abstract**:

Offline reinforcement learning (RL) aims to learn an effective policy from a pre-collected dataset. Most existing works are to develop sophisticated learning algorithms, with less emphasis on improving the data collection process. Moreover, it is even challenging to extend the single-task setting and collect a task-agnostic dataset that allows an agent to perform multiple downstream tasks. In this paper, we propose a Curiosity-driven Unsupervised Data Collection (CUDC) method to expand feature space using adaptive temporal distances for task-agnostic data collection and ultimately improve learning efficiency and capabilities for multi-task offline RL. To achieve this, CUDC estimates the probability of the k-step future states being reachable from the current states, and adapts how many steps into the future that the dynamics model should predict. With this adaptive reachability mechanism in place, the feature representation can be diversified, and the agent can navigate itself to collect higher-quality data with curiosity. Empirically, CUDC surpasses existing unsupervised methods in efficiency and learning performance in various downstream offline RL tasks of the DeepMind control suite.

----

## [1689] T2MAC: Targeted and Trusted Multi-Agent Communication through Selective Engagement and Evidence-Driven Integration

**Authors**: *Chuxiong Sun, Zehua Zang, Jiabao Li, Jiangmeng Li, Xiao Xu, Rui Wang, Changwen Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29438](https://doi.org/10.1609/aaai.v38i13.29438)

**Abstract**:

Communication stands as a potent mechanism to harmonize the behaviors of multiple agents. However, existing work primarily concentrates on broadcast communication, which not only lacks practicality, but also leads to information redundancy. This surplus, one-fits-all information could adversely impact the communication efficiency. Furthermore, existing works often resort to basic mechanisms to integrate observed and received information, impairing the learning process. To tackle these difficulties, we propose Targeted and Trusted Multi-Agent Communication (T2MAC), a straightforward yet effective method that enables agents to learn selective engagement and evidence-driven integration. With T2MAC, agents have the capability to craft individualized messages, pinpoint ideal communication windows, and engage with reliable partners, thereby refining communication efficiency. Following the reception of messages, the agents integrate information observed and received from different sources at an evidence level. This process enables agents to collectively use evidence garnered from multiple perspectives, fostering trusted and cooperative behaviors. We evaluate our method on a diverse set of cooperative multi-agent tasks, with varying difficulties, involving different scales and ranging from Hallway, MPE to SMAC. The experiments indicate that the proposed model not only surpasses the state-of-the-art methods in terms of cooperative performance and communication efficiency, but also exhibits impressive generalization.

----

## [1690] On the Role of Server Momentum in Federated Learning

**Authors**: *Jianhui Sun, Xidong Wu, Heng Huang, Aidong Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29439](https://doi.org/10.1609/aaai.v38i13.29439)

**Abstract**:

Federated Averaging (FedAvg) is known to experience convergence issues when encountering significant clients system heterogeneity and data heterogeneity. Server momentum has been proposed as an effective mitigation. However, existing server momentum works are restrictive in the momentum formulation, do not properly schedule hyperparameters and focus only on system homogeneous settings, which leaves the role of server momentum still an under-explored problem. In this paper, we propose a general framework for server momentum, that (a) covers a large class of momentum schemes that are unexplored in federated learning (FL), (b) enables a popular stagewise hyperparameter scheduler, (c) allows heterogeneous and asynchronous local computing. We provide rigorous convergence analysis for the proposed framework. To our best knowledge, this is the first work that thoroughly analyzes the performances of server momentum with a hyperparameter scheduler and system heterogeneity. Extensive experiments validate the effectiveness of our proposed framework. Due to page limit, we leave all proofs to the full version https://arxiv.org/abs/2312.12670.

----

## [1691] RedCore: Relative Advantage Aware Cross-Modal Representation Learning for Missing Modalities with Imbalanced Missing Rates

**Authors**: *Jun Sun, Xinxin Zhang, Shoukang Han, Yu-Ping Ruan, Taihao Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29440](https://doi.org/10.1609/aaai.v38i13.29440)

**Abstract**:

Multimodal learning is susceptible to modality missing, which poses a major obstacle for its practical applications and, thus, invigorates increasing research interest. In this paper, we investigate two challenging problems: 1) when modality missing exists in the training data, how to exploit the incomplete samples while guaranteeing that they are properly supervised? 2) when the missing rates of different modalities vary, causing or exacerbating the imbalance among modalities, how to address the imbalance and ensure all modalities are well-trained. To tackle these two challenges, we first introduce the variational information bottleneck (VIB) method for the cross-modal representation learning of missing modalities, which capitalizes on the available modalities and the labels as supervision. Then, accounting for the imbalanced missing rates, we define relative advantage to quantify the advantage of each modality over others. Accordingly, a bi-level optimization problem is formulated to adaptively regulate the supervision of all modalities during training. As a whole, the proposed approach features Relative advantage aware Cross-modal representation learning (abbreviated as RedCore) for missing modalities with imbalanced missing rates. Extensive empirical results demonstrate that RedCore outperforms competing models in that it exhibits superior robustness against either large or imbalanced missing rates. The code is available at: https://github.com/sunjunaimer/RedCore.

----

## [1692] Dual Self-Paced Cross-Modal Hashing

**Authors**: *Yuan Sun, Jian Dai, Zhenwen Ren, Yingke Chen, Dezhong Peng, Peng Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29441](https://doi.org/10.1609/aaai.v38i14.29441)

**Abstract**:

Cross-modal hashing~(CMH) is an efficient technique to retrieve relevant data across different modalities, such as images, texts, and videos, which has attracted more and more attention due to its low storage cost and fast query speed. Although existing CMH methods achieve remarkable processes, almost all of them treat all samples of varying difficulty levels without discrimination, thus leaving them vulnerable to noise or outliers. Based on this observation, we reveal and study dual difficulty levels implied in cross-modal hashing learning, \ie instance-level and feature-level difficulty. To address this problem, we propose a novel Dual Self-Paced Cross-Modal Hashing (DSCMH) that mimics human cognitive learning to learn hashing from ``easy'' to ``hard'' in both instance and feature levels, thereby embracing robustness against noise/outliers. Specifically, our DSCMH assigns weights to each instance and feature to measure their difficulty or reliability, and then uses these weights to automatically filter out the noisy and irrelevant data points in the original space. By gradually increasing the weights during training, our method can focus on more instances and features from ``easy'' to ``hard'' in training, thus mitigating the adverse effects of noise or outliers. Extensive experiments are conducted on three widely-used benchmark datasets to demonstrate the effectiveness and robustness of the proposed DSCMH over 12 state-of-the-art CMH methods.

----

## [1693] ACAMDA: Improving Data Efficiency in Reinforcement Learning through Guided Counterfactual Data Augmentation

**Authors**: *Yuewen Sun, Erli Wang, Biwei Huang, Chaochao Lu, Lu Feng, Changyin Sun, Kun Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29442](https://doi.org/10.1609/aaai.v38i14.29442)

**Abstract**:

Data augmentation plays a crucial role in improving the data efficiency of reinforcement learning (RL). However, the generation of high-quality augmented data remains a significant challenge. To overcome this, we introduce ACAMDA (Adversarial Causal Modeling for Data Augmentation), a novel framework that integrates two causality-based tasks: causal structure recovery and counterfactual estimation. The unique aspect of ACAMDA lies in its ability to recover temporal causal relationships from limited non-expert datasets. The identification of the sequential cause-and-effect allows the creation of realistic yet unobserved scenarios. We utilize this characteristic to generate guided counterfactual datasets, which, in turn, substantially reduces the need for extensive data collection. By simulating various state-action pairs under hypothetical actions, ACAMDA enriches the training dataset for diverse and heterogeneous conditions. Our experimental evaluation shows that ACAMDA outperforms existing methods, particularly when applied to novel and unseen domains.

----

## [1694] Learning Not to Regret

**Authors**: *David Sychrovsky, Michal Sustr, Elnaz Davoodi, Michael Bowling, Marc Lanctot, Martin Schmid*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29443](https://doi.org/10.1609/aaai.v38i14.29443)

**Abstract**:

The literature on game-theoretic equilibrium finding predominantly focuses on single games or their repeated play.
Nevertheless, numerous real-world scenarios feature playing a game sampled from a distribution of similar, but not identical games, such as playing poker with different public cards or trading correlated assets on the stock market. 
As these similar games feature similar equilibra, we investigate a way to accelerate equilibrium finding on such a distribution.
We present a novel ``learning not to regret'' framework, enabling us to meta-learn a regret minimizer tailored to a specific distribution. 
Our key contribution, Neural Predictive Regret Matching, is uniquely meta-learned to converge rapidly for the chosen distribution of games, while having regret minimization guarantees on any game.
We validated our algorithms' faster convergence on a distribution of river poker games. 
Our experiments show that the meta-learned algorithms outpace their non-meta-learned counterparts, achieving more than tenfold improvements.

----

## [1695] Optimal Transport with Cyclic Symmetry

**Authors**: *Shoichiro Takeda, Yasunori Akagi, Naoki Marumo, Kenta Niwa*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29444](https://doi.org/10.1609/aaai.v38i14.29444)

**Abstract**:

We propose novel fast algorithms for optimal transport (OT) utilizing a cyclic symmetry structure of input data. Such OT with cyclic symmetry appears universally in various real-world examples: image processing, urban planning, and graph processing. Our main idea is to reduce OT to a small optimization problem that has significantly fewer variables by utilizing cyclic symmetry and various optimization techniques. On the basis of this reduction, our algorithms solve the small optimization problem instead of the original OT. As a result, our algorithms obtain the optimal solution and the objective function value of the original OT faster than solving the original OT directly. In this paper, our focus is on two crucial OT formulations: the linear programming OT (LOT) and the strongly convex-regularized OT, which includes the well-known entropy-regularized OT (EROT). Experiments show the effectiveness of our algorithms for LOT and EROT in synthetic/real-world data that has a strict/approximate cyclic symmetry structure. Through theoretical and experimental results, this paper successfully introduces the concept of symmetry into the OT research field for the first time.

----

## [1696] Cross-Gate MLP with Protein Complex Invariant Embedding Is a One-Shot Antibody Designer

**Authors**: *Cheng Tan, Zhangyang Gao, Lirong Wu, Jun Xia, Jiangbin Zheng, Xihong Yang, Yue Liu, Bozhen Hu, Stan Z. Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29445](https://doi.org/10.1609/aaai.v38i14.29445)

**Abstract**:

Antibodies are crucial proteins produced by the immune system in response to foreign substances or antigens. The specificity of an antibody is determined by its complementarity-determining regions (CDRs), which are located in the variable domains of the antibody chains and form the antigen-binding site. Previous studies have utilized complex techniques to generate CDRs, but they suffer from inadequate geometric modeling. Moreover, the common iterative refinement strategies lead to an inefficient inference. In this paper, we propose a simple yet effective model that can co-design 1D sequences and 3D structures of CDRs in a one-shot manner. To achieve this, we decouple the antibody CDR design problem into two stages: (i) geometric modeling of protein complex structures and (ii) sequence-structure co-learning. We develop a novel macromolecular structure invariant embedding, typically for protein complexes, that captures both intra- and inter-component interactions among the backbone atoms, including Calpha, N, C, and O atoms, to achieve comprehensive geometric modeling. Then, we introduce a simple cross-gate MLP for sequence-structure co-learning, allowing sequence and structure representations to implicitly refine each other. This enables our model to design desired sequences and structures in a one-shot manner. Extensive experiments are conducted to evaluate our results at both the sequence and structure level, which demonstrate that our model achieves superior performance compared to the state-of-the-art antibody CDR design methods.

----

## [1697] FedCompetitors: Harmonious Collaboration in Federated Learning with Competing Participants

**Authors**: *Shanli Tan, Hao Cheng, Xiaohu Wu, Han Yu, Tiantian He, Yew Soon Ong, Chongjun Wang, Xiaofeng Tao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29446](https://doi.org/10.1609/aaai.v38i14.29446)

**Abstract**:

Federated learning (FL) provides a privacy-preserving approach for collaborative training of machine learning models. Given the potential data heterogeneity, it is crucial to select appropriate collaborators for each FL participant (FL-PT) based on data complementarity. Recent studies have addressed this challenge. Similarly, it is imperative to consider the inter-individual relationships among FL-PTs where some FL-PTs engage in competition. Although FL literature has acknowledged the significance of this scenario, practical methods for establishing FL ecosystems remain largely unexplored. In this paper, we extend a principle from the balance theory, namely “the friend of my enemy is my enemy”, to ensure the absence of conflicting interests within an FL ecosystem. The extended principle and the resulting problem are formulated via graph theory and integer linear programming. A polynomial-time algorithm is proposed to determine the collaborators of each FL-PT. The solution guarantees high scalability, allowing even competing FL-PTs to smoothly join the ecosystem without conflict of interest. The proposed framework jointly considers competition and data heterogeneity. Extensive experiments on real-world and synthetic data demonstrate its efficacy compared to five alternative approaches, and its ability to establish efficient collaboration networks among FL-PTs.

----

## [1698] Harnessing the Power of Beta Scoring in Deep Active Learning for Multi-Label Text Classification

**Authors**: *Wei Tan, Ngoc Dang Nguyen, Lan Du, Wray L. Buntine*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29447](https://doi.org/10.1609/aaai.v38i14.29447)

**Abstract**:

Within the scope of natural language processing, the domain of multi-label text classification is uniquely challenging due to its expansive and uneven label distribution. The complexity deepens due to the demand for an extensive set of annotated data for training an advanced deep learning model, especially in specialized fields where the labeling task can be labor-intensive and often requires domain-specific knowledge. Addressing these challenges, our study introduces a novel deep active learning strategy, capitalizing on the Beta family of proper scoring rules within the Expected Loss Reduction framework. It computes the expected increase in scores using the Beta Scoring Rules,  which are then transformed into sample vector representations. These vector representations guide the diverse selection of informative sample, directly linking this process to the model's expected proper score. Comprehensive evaluations across both synthetic and real datasets reveal our method's capability to often outperform established acquisition techniques in multi-label text classification, presenting encouraging outcomes across various architectural and dataset scenarios.

----

## [1699] A Two-Stage Information Extraction Network for Incomplete Multi-View Multi-Label Classification

**Authors**: *Xin Tan, Ce Zhao, Chengliang Liu, Jie Wen, Zhanyan Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29448](https://doi.org/10.1609/aaai.v38i14.29448)

**Abstract**:

Recently, multi-view multi-label classification (MvMLC) has received a significant amount of research interest and many methods have been proposed based on the assumptions of view completion and label completion. However, in real-world scenarios, multi-view multi-label data tends to be incomplete due to various uncertainties involved in data collection and manual annotation. As a result, the conventional MvMLC methods fail. In this paper, we propose a new two-stage MvMLC network to solve this incomplete MvMLC issue with partial missing views and missing labels. Different from the existing works, our method attempts to leverage the diverse information from the partially missing data based on the information theory. Specifically, our method aims to minimize task-irrelevant information while maximizing task-relevant information through the principles of information bottleneck theory and mutual information extraction. The first stage of our network involves training view-specific classifiers to concentrate the task-relevant information. Subsequently, in the second stage, the hidden states of these classifiers serve as input for an alignment model, an autoencoder-based mutual information extraction framework, and a weighted fusion classifier to make the final prediction. Extensive experiments performed on five datasets validate that our method outperforms other state-of-the-art methods. Code is available at https://github.com/KevinTan10/TSIEN.

----

## [1700] An Effective Augmented Lagrangian Method for Fine-Grained Multi-View Optimization

**Authors**: *Yuze Tan, Hecheng Cai, Shudong Huang, Shuping Wei, Fan Yang, Jiancheng Lv*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29449](https://doi.org/10.1609/aaai.v38i14.29449)

**Abstract**:

The significance of multi-view learning in effectively mitigating the intricate intricacies entrenched within heterogeneous data has garnered substantial attention in recent years. Notwithstanding the favorable achievements showcased by recent strides in this area, a confluence of noteworthy challenges endures. To be specific, a majority of extant methodologies unceremoniously assign weights to data points view-wisely. This ineluctably disregards the intrinsic reality that disparate views confer diverse contributions to each individual sample, consequently neglecting the rich wellspring of sample-level structural insights harbored within the dataset. In this paper, we proposed an effective Augmented Lagrangian MethOd for fiNe-graineD (ALMOND) multi-view optimization. 
This innovative approach scrutinizes the interplay among multiple views at the granularity of individual samples, thereby fostering the enhanced preservation of local structural coherence. The Augmented Lagrangian Method (ALM) is elaborately incorporated into our framework, which enables us to achieve an optimal solution without involving an inexplicable intermediate variable as previous methods do. Empirical experiments on multi-view clustering tasks across heterogeneous datasets serve to incontrovertibly showcase the effectiveness of our proposed methodology, corroborating its preeminence over incumbent state-of-the-art alternatives.

----

## [1701] SimCalib: Graph Neural Network Calibration Based on Similarity between Nodes

**Authors**: *Boshi Tang, Zhiyong Wu, Xixin Wu, Qiaochu Huang, Jun Chen, Shun Lei, Helen Meng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29450](https://doi.org/10.1609/aaai.v38i14.29450)

**Abstract**:

Graph neural networks (GNNs) have exhibited impressive performance in modeling graph data as exemplified in various applications. Recently, the GNN calibration problem has attracted increasing attention, especially in cost-sensitive scenarios. Previous work has gained empirical insights on the issue, and devised effective approaches for it, but theoretical supports still fall short. In this work, we shed light on the relationship between GNN calibration and nodewise similarity via theoretical analysis. A novel calibration framework, named SimCalib, is accordingly proposed to consider similarity between nodes at global and local levels. At the global level, the Mahalanobis distance between the current node and class prototypes is integrated to implicitly consider similarity between the current node and all nodes in the same class. At the local level, the similarity of node representation movement dynamics, quantified by nodewise homophily and relative degree, is considered. Informed about the application of nodewise movement patterns in analyzing nodewise behavior on the over-smoothing problem, we empirically present a possible relationship between over-smoothing and GNN calibration problem. Experimentally, we discover a correlation between nodewise similarity and model calibration improvement, in alignment with our theoretical results. Additionally, we conduct extensive experiments investigating different design factors and demonstrate the effectiveness of our proposed SimCalib framework for GNN calibration by achieving state-of-the-art performance on 14 out of 16 benchmarks.

----

## [1702] DP-AdamBC: Your DP-Adam Is Actually DP-SGD (Unless You Apply Bias Correction)

**Authors**: *Qiaoyue Tang, Frederick Shpilevskiy, Mathias Lécuyer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29451](https://doi.org/10.1609/aaai.v38i14.29451)

**Abstract**:

The Adam optimizer is a popular choice in contemporary deep learning due to its strong empirical performance. However we observe that in privacy sensitive scenarios, the traditional use of Differential Privacy (DP) with the Adam optimizer leads to sub-optimal performance on several tasks. We find that this performance degradation is due to a DP bias in Adam's second moment estimator, introduced by the addition of independent noise in the gradient computation to enforce DP guarantees. This DP bias leads to a different scaling for low variance parameter updates, that is inconsistent with the behavior of non-private Adam, and Adam's sign descent interpretation. We propose the DP-AdamBC optimization algorithm, which corrects for the bias in the second moment estimation and retrieves the expected behaviour of Adam. Empirically, DP-AdamBC significantly improves the optimization performance of DP-Adam by up to 3.5% in final accuracy in image, text, and graph node classification tasks.

----

## [1703] Non-monotone Sequential Submodular Maximization

**Authors**: *Shaojie Tang, Jing Yuan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29452](https://doi.org/10.1609/aaai.v38i14.29452)

**Abstract**:

In this paper, we study a fundamental problem in submodular optimization known as sequential submodular maximization. The primary objective of this problem is to select and rank a sequence of items to optimize a group of submodular functions.
The existing research on this problem has predominantly concentrated on the monotone setting, assuming that the submodular functions are non-decreasing. However, in various real-world scenarios, like diversity-aware recommendation systems, adding items to an existing set might negatively impact the overall utility. In response, we propose to study this problem with non-monotone submodular functions and develop approximation algorithms for both flexible and fixed length constraints, as well as a special case with identical utility functions. The empirical evaluations further validate the effectiveness of our proposed algorithms in the domain of video recommendations.

----

## [1704] Comprehensive View Embedding Learning for Single-Cell Multimodal Integration

**Authors**: *Zhenchao Tang, Jiehui Huang, Guanxing Chen, Calvin Yu-Chian Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29453](https://doi.org/10.1609/aaai.v38i14.29453)

**Abstract**:

Motivation: Advances in single-cell measurement techniques provide rich multimodal data, which helps us to explore the life state of cells more deeply. However, multimodal integration, or, learning joint embeddings from multimodal data remains a current challenge. The difficulty in integrating unpaired single-cell multimodal data is that different modalities have different feature spaces, which easily leads to information loss in joint embedding. And few existing methods have fully exploited and fused the information in single-cell multimodal data. Result: In this study, we propose CoVEL, a deep learning method for unsupervised integration of single-cell multimodal data. CoVEL learns single-cell representations from a comprehensive view, including regulatory relationships between modalities, fine-grained representations of cells, and relationships between different cells. The comprehensive view embedding enables CoVEL to remove the gap between modalities while protecting biological heterogeneity. Experimental results on multiple public datasets show that CoVEL is accurate and robust to single-cell multimodal integration. Data availability: https://github.com/shapsider/scintegration.

----

## [1705] z-SignFedAvg: A Unified Stochastic Sign-Based Compression for Federated Learning

**Authors**: *Zhiwei Tang, Yanmeng Wang, Tsung-Hui Chang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29454](https://doi.org/10.1609/aaai.v38i14.29454)

**Abstract**:

Federated Learning (FL) is a promising privacy-preserving
distributed learning paradigm but suffers from high communi-
cation cost when training large-scale machine learning models.
Sign-based methods, such as SignSGD, have been proposed
as a biased gradient compression technique for reducing the
communication cost. However, sign-based algorithms could
diverge under heterogeneous data, which thus motivated the de-
velopment of advanced techniques, such as the error-feedback
method and stochastic sign-based compression, to fix this
issue. Nevertheless, these methods still suffer from slower
convergence rates, and none of them allows multiple local
SGD updates like FedAvg. In this paper, we propose a novel
noisy perturbation scheme with a general symmetric noise
distribution for sign-based compression, which not only al-
lows one to flexibly control the bias-variance tradeoff for the
compressed gradient, but also provides a unified viewpoint
to existing stochastic sign-based methods. More importantly,
the proposed scheme enables the development of the very first
sign-based FedAvg algorithm (z-SignFedAvg) to accelerate
the convergence. Theoretically, we show that z-SignFedAvg
achieves a faster convergence rate than existing sign-based
methods and, under the uniformly distributed noise, can enjoy
the same convergence rate as its uncompressed counterpart.
Extensive experiments are conducted to demonstrate that the
z-SignFedAvg can achieve competitive empirical performance
on real datasets and outperforms existing schemes.

----

## [1706] Deciphering Raw Data in Neuro-Symbolic Learning with Provable Guarantees

**Authors**: *Lue Tao, Yu-Xuan Huang, Wang-Zhou Dai, Yuan Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29455](https://doi.org/10.1609/aaai.v38i14.29455)

**Abstract**:

Neuro-symbolic hybrid systems are promising for integrating machine learning and symbolic reasoning, where perception models are facilitated with information inferred from a symbolic knowledge base through logical reasoning. Despite empirical evidence showing the ability of hybrid systems to learn accurate perception models, the theoretical understanding of learnability is still lacking. Hence, it remains unclear why a hybrid system succeeds for a specific task and when it may fail given a different knowledge base. In this paper, we introduce a novel way of characterising supervision signals from a knowledge base, and establish a criterion for determining the knowledge’s efficacy in facilitating successful learning. This, for the first time, allows us to address the two questions above by inspecting the knowledge base under investigation. Our analysis suggests that many knowledge bases satisfy the criterion, thus enabling effective learning, while some fail to satisfy it, indicating potential failures. Comprehensive experiments confirm the utility of our criterion on benchmark tasks.

----

## [1707] Efficient Nonparametric Tensor Decomposition for Binary and Count Data

**Authors**: *Zerui Tao, Toshihisa Tanaka, Qibin Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29456](https://doi.org/10.1609/aaai.v38i14.29456)

**Abstract**:

In numerous applications, binary reactions or event counts are observed and stored within high-order tensors. Tensor decompositions (TDs) serve as a powerful tool to handle such high-dimensional and sparse data. However, many traditional TDs are explicitly or implicitly designed based on the Gaussian distribution, which is unsuitable for discrete data. Moreover, most TDs rely on predefined multi-linear structures, such as CP and Tucker formats. Therefore, they may not be effective enough to handle complex real-world datasets. To address these issues, we propose ENTED, an Efficient Nonparametric TEnsor Decomposition for binary and count tensors.  Specifically, we first employ a nonparametric Gaussian process (GP) to replace traditional multi-linear structures. Next, we utilize the Pólya-Gamma augmentation which provides a unified framework to establish conjugate models for binary and count distributions. Finally, to address the computational issue of GPs, we enhance the model by incorporating sparse orthogonal variational inference of inducing points, which offers a more effective covariance approximation within GPs and stochastic natural gradient updates for nonparametric models. We evaluate our model on several real-world tensor completion tasks, considering binary and count datasets. The results manifest both better performance and computational advantages of the proposed model.

----

## [1708] FFT-Based Dynamic Token Mixer for Vision

**Authors**: *Yuki Tatsunami, Masato Taki*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29457](https://doi.org/10.1609/aaai.v38i14.29457)

**Abstract**:

Multi-head-self-attention (MHSA)-equipped models have achieved notable performance in computer vision. Their computational complexity is proportional to quadratic numbers of pixels in input feature maps, resulting in slow processing, especially when dealing with high-resolution images. New types of token-mixer are proposed as an alternative to MHSA to circumvent this problem: an FFT-based token-mixer involves global operations similar to MHSA but with lower computational complexity. However, despite its attractive properties, the FFT-based token-mixer has not been carefully examined in terms of its compatibility with the rapidly evolving MetaFormer architecture. Here, we propose a novel token-mixer called Dynamic Filter and novel image recognition models, DFFormer and CDFFormer, to close the gaps above. The results of image classification and downstream tasks, analysis, and visualization show that our models are helpful. Notably, their throughput and memory efficiency when dealing with high-resolution image recognition is remarkable. Our results indicate that Dynamic Filter is one of the token-mixer options that should be seriously considered. The code is available at https://github.com/okojoalg/dfformer

----

## [1709] An Information-Flow Perspective on Algorithmic Fairness

**Authors**: *Samuel Teuber, Bernhard Beckert*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29458](https://doi.org/10.1609/aaai.v38i14.29458)

**Abstract**:

This work presents insights gained by investigating the relationship between algorithmic fairness and the concept of secure information flow. The problem of enforcing secure information flow is well-studied in the context of information security: If secret information may "flow" through an algorithm or program in such a way that it can influence the program’s output, then that is considered insecure information flow as attackers could potentially observe (parts of) the secret.

There is a strong correspondence between secure information flow and algorithmic fairness: if protected attributes such as race, gender, or age are treated as secret program inputs, then secure information flow means that these "secret" attributes cannot influence the result of a program. While most research in algorithmic fairness evaluation concentrates on studying the impact of algorithms (often treating the algorithm as a black-box), the concepts derived from information flow can be used both for the analysis of disparate treatment as well as disparate impact w.r.t. a structural causal model.

In this paper, we examine the relationship between quantitative as well as qualitative information-flow properties and fairness. Moreover, based on this duality, we derive a new quantitative notion of fairness called fairness spread, which can be easily analyzed using quantitative information flow and which strongly relates to counterfactual fairness. We demonstrate that off-the-shelf tools for information-flow properties can be used in order to formally analyze a program's algorithmic fairness properties, including the new notion of fairness spread as well as established notions such as demographic parity.

----

## [1710] Amalgamating Multi-Task Models with Heterogeneous Architectures

**Authors**: *Jidapa Thadajarassiri, Walter Gerych, Xiangnan Kong, Elke A. Rundensteiner*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29459](https://doi.org/10.1609/aaai.v38i14.29459)

**Abstract**:

Multi-task learning (MTL) is essential for real-world applications that handle multiple tasks simultaneously, such as selfdriving cars. MTL methods improve the performance of all tasks by utilizing information across tasks to learn a robust shared representation. However, acquiring sufficient labeled data tends to be extremely expensive, especially when having to support many tasks. Recently, Knowledge Amalgamation (KA) has emerged as an effective strategy for addressing the lack of labels by instead learning directly from pretrained models (teachers). KA learns one unified multi-task student that masters all tasks across all teachers. Existing KA for MTL works are limited to teachers with identical architectures, and thus propose layer-to-layer based approaches. Unfortunately, in practice, teachers may have heterogeneous architectures; their layers may not be aligned and their dimensionalities or scales may be incompatible. Amalgamating multi-task teachers with heterogeneous architectures remains an open problem. For this, we design Versatile Common Feature Consolidator (VENUS), the first solution to this problem. VENUS fuses knowledge from the shared representations of each teacher into one unified generalized representation for all tasks. Specifically, we design the Feature Consolidator network that leverages an array of teacher-specific trainable adaptors. These adaptors enable the student to learn from multiple teachers, even if they have incompatible learned representations. We demonstrate that VENUS outperforms five alternative methods on numerous benchmark datasets across a broad spectrum of experiments.

----

## [1711] ConSequence: Synthesizing Logically Constrained Sequences for Electronic Health Record Generation

**Authors**: *Brandon Theodorou, Shrusti Jain, Cao Xiao, Jimeng Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29460](https://doi.org/10.1609/aaai.v38i14.29460)

**Abstract**:

Generative models can produce synthetic patient records for analytical tasks when real data is unavailable or limited. However, current methods struggle with adhering to domain-specific knowledge and removing invalid data. We present ConSequence, an effective approach to integrating domain knowledge into sequential generative neural network outputs. Our rule-based formulation includes temporal aggregation and antecedent evaluation modules, ensured by an efficient matrix multiplication formulation, to satisfy hard and soft logical constraints across time steps. Existing constraint methods often fail to guarantee constraint satisfaction, lack the ability to handle temporal constraints, and hinder the learning and computational efficiency of the model. In contrast, our approach efficiently handles all types of constraints with guaranteed logical coherence. We demonstrate ConSequence's effectiveness in generating electronic health records, outperforming competitors in achieving complete temporal and spatial constraint satisfaction without compromising runtime performance or generative quality. Specifically, ConSequence successfully prevents all rule violations while improving the model quality in reducing its test perplexity by 5% and incurring less than a 13% slowdown in generation speed compared to an unconstrained model.

----

## [1712] N-gram Unsupervised Compoundation and Feature Injection for Better Symbolic Music Understanding

**Authors**: *Jinhao Tian, Zuchao Li, Jiajia Li, Ping Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29461](https://doi.org/10.1609/aaai.v38i14.29461)

**Abstract**:

The first step to apply deep learning techniques for symbolic music understanding is to transform musical pieces (mainly in MIDI format) into sequences of predefined tokens like note pitch, note velocity, and chords. Subsequently, the sequences are fed into a neural sequence model to accomplish specific tasks.
Music sequences exhibit strong correlations between adjacent elements, making them prime candidates for N-gram techniques from Natural Language Processing (NLP). Consider classical piano music: specific melodies might recur throughout a piece, with subtle variations each time.
In this paper, we propose a novel method, NG-Midiformer, for understanding symbolic music sequences that leverages the N-gram approach. Our method involves first processing music pieces into word-like sequences with our proposed unsupervised compoundation, followed by using our N-gram Transformer encoder, which can effectively incorporate N-gram information to enhance the primary encoder part for better understanding of music sequences.
The pre-training process on large-scale music datasets enables the model to thoroughly learn the N-gram information contained within music sequences, and subsequently apply this information for making inferences during the fine-tuning stage.
Experiment on various datasets demonstrate the effectiveness of our method and achieved state-of-the-art performance on a series of music understanding downstream tasks. The code and model weights will be released at https://github.com/CinqueOrigin/NG-Midiformer.

----

## [1713] DeRDaVa: Deletion-Robust Data Valuation for Machine Learning

**Authors**: *Xiao Tian, Rachael Hwee Ling Sim, Jue Fan, Bryan Kian Hsiang Low*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29462](https://doi.org/10.1609/aaai.v38i14.29462)

**Abstract**:

Data valuation is concerned with determining a fair valuation of data from data sources to compensate them or to identify training examples that are the most or least useful for predictions. With the rising interest in personal data ownership and data protection regulations, model owners will likely have to fulfil more data deletion requests. This raises issues that have not been addressed by existing works: Are the data valuation scores still fair with deletions? Must the scores be expensively recomputed? The answer is no. To avoid recomputations, we propose using our data valuation framework DeRDaVa upfront for valuing each data source's contribution to preserving robust model performance after anticipated data deletions. DeRDaVa can be efficiently approximated and will assign higher values to data that are more useful or less likely to be deleted. We further generalize DeRDaVa to Risk-DeRDaVa to cater to risk-averse/seeking model owners who are concerned with the worst/best-cases model utility. We also empirically demonstrate the practicality of our solutions.

----

## [1714] Weisfeiler and Lehman Go Paths: Learning Topological Features via Path Complexes

**Authors**: *Quang Truong, Peter Chin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29463](https://doi.org/10.1609/aaai.v38i14.29463)

**Abstract**:

Graph Neural Networks (GNNs), despite achieving remarkable performance across different tasks, are theoretically bounded by the 1-Weisfeiler-Lehman test, resulting in limitations in terms of graph expressivity. Even though prior works on topological higher-order GNNs overcome that boundary, these models often depend on assumptions about sub-structures of graphs. Specifically, topological GNNs leverage the prevalence of cliques, cycles, and rings to enhance the message-passing procedure. Our study presents a novel perspective by focusing on simple paths within graphs during the topological message-passing process, thus liberating the model from restrictive inductive biases. We prove that by lifting graphs to path complexes, our model can generalize the existing works on topology while inheriting several theoretical results on simplicial complexes and regular cell complexes. Without making prior assumptions about graph sub-structures, our method outperforms earlier works in other topological domains and achieves state-of-the-art results on various benchmarks.

----

## [1715] Attribute-Missing Graph Clustering Network

**Authors**: *Wenxuan Tu, Renxiang Guan, Sihang Zhou, Chuan Ma, Xin Peng, Zhiping Cai, Zhe Liu, Jieren Cheng, Xinwang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29464](https://doi.org/10.1609/aaai.v38i14.29464)

**Abstract**:

Deep clustering with attribute-missing graphs, where only a subset of nodes possesses complete attributes while those of others are missing, is an important yet challenging topic in various practical applications. It has become a prevalent learning paradigm in existing studies to perform data imputation first and subsequently conduct clustering using the imputed information. However, these ``two-stage" methods disconnect the clustering and imputation processes, preventing the model from effectively learning clustering-friendly graph embedding. Furthermore, they are not tailored for clustering tasks, leading to inferior clustering results. To solve these issues, we propose a novel Attribute-Missing Graph Clustering (AMGC) method to alternately promote clustering and imputation in a unified framework, where we iteratively produce the clustering-enhanced nearest neighbor information to conduct the data imputation process and utilize the imputed information to implicitly refine the clustering distribution through model optimization. Specifically, in the imputation step, we take the learned clustering information as imputation prompts to help each attribute-missing sample gather highly correlated features within its clusters for data completion, such that the intra-class compactness can be improved. Moreover, to support reliable clustering, we maximize inter-class separability by conducting cost-efficient dual non-contrastive learning over the imputed latent features, which in turn promotes greater graph encoding capability for clustering sub-network. Extensive experiments on five datasets have verified the superiority of AMGC against competitors.

----

## [1716] Parameterized Projected Bellman Operator

**Authors**: *Théo Vincent, Alberto Maria Metelli, Boris Belousov, Jan Peters, Marcello Restelli, Carlo D'Eramo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29465](https://doi.org/10.1609/aaai.v38i14.29465)

**Abstract**:

Approximate value iteration (AVI) is a family of algorithms for reinforcement learning (RL) that aims to obtain an approximation of the optimal value function. Generally, AVI algorithms implement an iterated procedure where each step consists of (i) an application of the Bellman operator and (ii) a projection step into a considered function space. Notoriously, the Bellman operator leverages transition samples, which strongly determine its behavior, as uninformative samples can result in negligible updates or long detours, whose detrimental effects are further exacerbated by the computationally intensive projection step. To address these issues, we propose a novel alternative approach based on learning an approximate version of the Bellman operator rather than estimating it through samples as in AVI approaches. This way, we are able to (i) generalize across transition samples and (ii) avoid the computationally intensive projection step. For this reason, we call our novel operator projected Bellman operator (PBO). We formulate an optimization problem to learn PBO for generic sequential decision-making problems, and we theoretically analyze its properties in two representative classes of RL problems. Furthermore, we theoretically study our approach under the lens of AVI and devise algorithmic implementations to learn PBO in offline and online settings by leveraging neural network parameterizations. Finally, we empirically showcase the benefits of PBO w.r.t. the regular Bellman operator on several RL problems.

----

## [1717] Causal Strategic Learning with Competitive Selection

**Authors**: *Kiet Q. H. Vo, Muneeb Aadil, Siu Lun Chau, Krikamol Muandet*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29466](https://doi.org/10.1609/aaai.v38i14.29466)

**Abstract**:

We study the problem of agent selection in causal strategic learning under multiple decision makers and address two key challenges that come with it. 
Firstly, while much of prior work focuses on studying a fixed pool of agents that remains static regardless of their evaluations, we consider the impact of selection procedure by which agents are not only evaluated, but also selected.
When each decision maker unilaterally selects agents by maximising their own utility, we show that the optimal selection rule is a trade-off between selecting the best agents and providing incentives to maximise the agents' improvement. 
Furthermore, this optimal selection rule relies on incorrect predictions of agents' outcomes. 
Hence, we study the conditions under which a decision maker's optimal selection rule will not lead to deterioration of agents' outcome nor cause unjust reduction in agents' selection chance. 
To that end, we provide an analytical form of the optimal selection rule and a mechanism to retrieve the causal parameters from observational data, under certain assumptions on agents' behaviour. 
Secondly, when there are multiple decision makers, the interference between selection rules introduces another source of biases in estimating the underlying causal parameters. 
To address this problem, we provide a cooperative protocol which all decision makers must collectively adopt to recover the true causal parameters. 
Lastly, we complement our theoretical results with simulation studies.
Our results highlight not only the importance of causal modeling as a strategy to mitigate the effect of gaming, as suggested by previous work, but also the need of a benevolent regulator to enable it.

----

## [1718] Data Disparity and Temporal Unavailability Aware Asynchronous Federated Learning for Predictive Maintenance on Transportation Fleets

**Authors**: *Leonie von Wahl, Niklas Heidenreich, Prasenjit Mitra, Michael Nolting, Nicolas Tempelmeier*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29467](https://doi.org/10.1609/aaai.v38i14.29467)

**Abstract**:

Predictive maintenance has emerged as a critical application in modern transportation, leveraging sensor data to forecast potential damages proactively using machine learning. However, privacy concerns limit data sharing, making Federated learning an appealing approach to preserve data privacy. Nevertheless, challenges arise due to disparities in data distribution and temporal unavailability caused by individual usage patterns in transportation. In this paper, we present a novel asynchronous federated learning approach to address system heterogeneity and facilitate machine learning for predictive maintenance on transportation fleets. The approach introduces a novel data disparity aware aggregation scheme and a federated early stopping method for training. To validate the effectiveness of our approach, we evaluate it on two independent real-world datasets from the transportation domain: 1) oil dilution prediction of car combustion engines and 2) remaining lifetime prediction of plane turbofan engines. Our experiments show that we reliably outperform five state-of-the-art baselines, including federated and classical machine learning models. Moreover, we show that our approach generalises to various prediction model architectures.

----

## [1719] Federated Graph Learning under Domain Shift with Generalizable Prototypes

**Authors**: *Guancheng Wan, Wenke Huang, Mang Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29468](https://doi.org/10.1609/aaai.v38i14.29468)

**Abstract**:

Federated Graph Learning is a privacy-preserving collaborative approach for training a shared model on graph-structured data in the distributed environment. However, in real-world scenarios, the client graph data usually originate from diverse domains, this unavoidably hinders the generalization performance of the final global model. To address this challenge, we start the first attempt to investigate this scenario by learning a well-generalizable model. In order to improve the performance of the global model from different perspectives, we propose a novel framework called Federated Graph Learning with Generalizable Prototypes (FGGP). It decouples the global model into two levels and bridges them via prototypes.  These prototypes, which are semantic centers derived from the feature extractor, can provide valuable classification information. At the classification model level, we innovatively eschew the traditional classifiers, then instead leverage clustered prototypes to capture fruitful domain information and enhance the discriminative capability of the classes, improving the performance of multi-domain predictions. Furthermore, at the feature extractor level, we go beyond traditional approaches by implicitly injecting distinct global knowledge and employing contrastive learning to obtain more powerful prototypes while enhancing the feature extractor generalization ability. Experimental results on various datasets are presented to validate the effectiveness of the proposed method.

----

## [1720] Unlocking the Power of Open Set: A New Perspective for Open-Set Noisy Label Learning

**Authors**: *Wenhai Wan, Xinrui Wang, Ming-Kun Xie, Shao-Yuan Li, Sheng-Jun Huang, Songcan Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29469](https://doi.org/10.1609/aaai.v38i14.29469)

**Abstract**:

Learning from noisy data has attracted much attention, where most methods focus on closed-set label noise. However, a more common scenario in the real world is the presence of both open-set and closed-set noise. Existing methods typically identify and handle these two types of label noise separately by designing a specific strategy for each type. However, in many real-world scenarios, it would be challenging to identify open-set examples, especially when the dataset has been severely corrupted. Unlike the previous works, we explore how models behave when faced with open-set examples, and find that a part of open-set examples gradually get integrated into certain known classes, which is beneficial for the separation among known classes. Motivated by the phenomenon, we propose a novel two-step contrastive learning method CECL (Class Expansion Contrastive Learning) which aims to deal with both types of label noise by exploiting the useful information of open-set examples. Specifically, we incorporate some open-set examples into closed-set classes to enhance performance while treating others as delimiters to improve representative ability. Extensive experiments on synthetic and real-world datasets with diverse label noise demonstrate the effectiveness of CECL.

----

## [1721] DiffAIL: Diffusion Adversarial Imitation Learning

**Authors**: *Bingzheng Wang, Guoqiang Wu, Teng Pang, Yan Zhang, Yilong Yin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29470](https://doi.org/10.1609/aaai.v38i14.29470)

**Abstract**:

Imitation learning aims to solve the problem of defining reward functions in real-world decision-making tasks. The current popular approach is the Adversarial Imitation Learning (AIL) framework, which matches expert state-action occupancy measures to obtain a surrogate reward for forward reinforcement learning. However, the traditional discriminator is a simple binary classifier and doesn't learn an accurate distribution, which may result in failing to identify expert-level state-action pairs induced by the policy interacting with the environment. To address this issue, we propose a method named diffusion adversarial imitation learning (DiffAIL), which introduces the diffusion model into the AIL framework. Specifically, DiffAIL models the state-action pairs as unconditional diffusion models and uses diffusion loss as part of the discriminator's learning objective, which enables the discriminator to capture better expert demonstrations and improve generalization. Experimentally, the results show that our method achieves state-of-the-art performance and significantly surpasses expert demonstration on two benchmark tasks, including the standard state-action setting and state-only settings.

----

## [1722] DR-Label: Label Deconstruction and Reconstruction of GNN Models for Catalysis Systems

**Authors**: *Bowen Wang, Chen Liang, Jiaze Wang, Jiezhong Qiu, Furui Liu, Shaogang Hao, Dong Li, Guangyong Chen, Xiaolong Zou, Pheng-Ann Heng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29471](https://doi.org/10.1609/aaai.v38i14.29471)

**Abstract**:

Attaining the equilibrium geometry of a catalyst-adsorbate system is key to fundamentally assessing its effective properties, such as adsorption energy. While machine learning methods with advanced representation or supervision strategies have been applied to boost and guide the relaxation processes of catalysis systems, existing methods that produce linearly aggregated geometry predictions are susceptible to edge representations ambiguity, and are therefore vulnerable to graph variations. In this paper, we present a novel graph neural network (GNN) supervision and prediction strategy DR-Label. Our approach mitigates the multiplicity of solutions in edge representation and encourages model predictions that are independent of graph structural variations. DR-Label first Deconstructs finer-grained equilibrium state information to the model by projecting the node-level supervision signal to each edge. Reversely, the model Reconstructs a more robust equilibrium state prediction by converting edge-level predictions to node-level via a sphere-fitting algorithm. When applied to three fundamentally different models, DR-Label consistently enhanced performance. Leveraging the graph structure invariance of the DR-Label strategy, we further propose DRFormer, which applied explicit intermediate positional update and achieves a new state-of-the-art performance on the Open Catalyst 2020 (OC20) dataset and the Cu-based single-atom alloys CO adsorption (SAA) dataset. We expect our work to highlight vital principles for advancing geometric GNN models for catalysis systems and beyond. Our code is available at https://github.com/bowenwang77/DR-Label

----

## [1723] GAD-PVI: A General Accelerated Dynamic-Weight Particle-Based Variational Inference Framework

**Authors**: *Fangyikang Wang, Huminhao Zhu, Chao Zhang, Hanbin Zhao, Hui Qian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29472](https://doi.org/10.1609/aaai.v38i14.29472)

**Abstract**:

Particle-based Variational Inference (ParVI) methods approximate the target distribution by iteratively evolving finite weighted particle systems. Recent advances of ParVI methods reveal the benefits of accelerated position update strategies and dynamic weight adjustment approaches. In this paper, we propose the first ParVI framework that possesses both accelerated position update and dynamical weight adjustment simultaneously, named the General Accelerated Dynamic-Weight Particle-based Variational Inference (GAD-PVI) framework. Generally, GAD-PVI simulates the semi-Hamiltonian gradient flow on a novel Information-Fisher-Rao space, which yields an additional decrease on the local functional dissipation. GAD-PVI is compatible with different dissimilarity functionals and associated smoothing approaches under three information metrics. Experiments on both synthetic and real-world data demonstrate the faster convergence and reduced approximation error of GAD-PVI methods over the state-of-the-art.

----

## [1724] Generative Model-Based Feature Knowledge Distillation for Action Recognition

**Authors**: *Guiqin Wang, Peng Zhao, Yanjiang Shi, Cong Zhao, Shusen Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29473](https://doi.org/10.1609/aaai.v38i14.29473)

**Abstract**:

Knowledge distillation (KD), a technique widely employed in computer vision, has emerged as a de facto standard for improving the performance of small neural networks. However, prevailing KD-based approaches in video tasks primarily focus on designing loss functions and fusing cross-modal information. This overlooks the spatial-temporal feature semantics, resulting in limited advancements in model compression. Addressing this gap, our paper introduces an innovative knowledge distillation framework, with the generative model for training a lightweight student model. In particular, the framework is organized into two steps: the initial phase is Feature Representation, wherein a generative model-based attention module is trained to represent feature semantics; Subsequently, the Generative-based Feature Distillation phase encompasses both Generative Distillation and Attention Distillation, with the objective of transferring attention-based feature semantics with the generative model. The efficacy of our approach is demonstrated through comprehensive experiments on diverse popular datasets, proving considerable enhancements in video action recognition task. Moreover, the effectiveness of our proposed framework is validated in the context of more intricate video action detection task. Our code is available at https://github.com/aaai-24/Generative-based-KD.

----

## [1725] Gradient-Guided Modality Decoupling for Missing-Modality Robustness

**Authors**: *Hao Wang, Shengda Luo, Guosheng Hu, Jianguo Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29474](https://doi.org/10.1609/aaai.v38i14.29474)

**Abstract**:

Multimodal learning with incomplete input data (missing modality) is very practical and challenging. In this work, we conduct an in-depth analysis of this challenge and find that modality dominance has a significant negative impact on the model training, greatly degrading the missing modality performance. Motivated by Grad-CAM, we introduce a novel indicator, gradients, to monitor and reduce modality dominance which widely exists in the missing-modality scenario. In aid of this indicator, we present a novel Gradient-guided Modality Decoupling (GMD) method to decouple the dependency on dominating modalities. Specifically, GMD removes the conflicted gradient components from different modalities to achieve this decoupling, significantly improving the performance. In addition, to flexibly handle modal-incomplete data, we design a parameter-efficient Dynamic Sharing (DS) framework which can adaptively switch on/off the network parameters based on whether one modality is available. We conduct extensive experiments on three popular multimodal benchmarks, including BraTS 2018 for medical segmentation, CMU-MOSI, and CMU-MOSEI for sentiment analysis. The results show that our method can significantly outperform the competitors, showing the effectiveness of the proposed solutions. Our code is released here: https://github.com/HaoWang420/Gradient-guided-Modality-Decoupling.

----

## [1726] V2A-Mapper: A Lightweight Solution for Vision-to-Audio Generation by Connecting Foundation Models

**Authors**: *Heng Wang, Jianbo Ma, Santiago Pascual, Richard Cartwright, Weidong Cai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29475](https://doi.org/10.1609/aaai.v38i14.29475)

**Abstract**:

Building artificial intelligence (AI) systems on top of a set of foundation models (FMs) is becoming a new paradigm in AI research. Their representative and generative abilities learnt from vast amounts of data can be easily adapted and transferred to a wide range of downstream tasks without extra training from scratch. However, leveraging FMs in cross-modal generation remains under-researched when audio modality is involved. On the other hand, automatically generating semantically-relevant sound from visual input is an important problem in cross-modal generation studies. To solve this vision-to-audio (V2A) generation problem, existing methods tend to design and build complex systems from scratch using modestly sized datasets. In this paper, we propose a lightweight solution to this problem by leveraging foundation models, specifically CLIP, CLAP, and AudioLDM. We first investigate the domain gap between the latent space of the visual CLIP and the auditory CLAP models. Then we propose a simple yet effective mapper mechanism (V2A-Mapper) to bridge the domain gap by translating the visual input between CLIP and CLAP spaces. Conditioned on the translated CLAP embedding, pretrained audio generative FM AudioLDM is adopted to produce high-fidelity and visually-aligned sound. Compared to previous approaches, our method only requires a quick training of the V2A-Mapper. We further analyze and conduct extensive experiments on the choice of the V2A-Mapper and show that a generative mapper is better at fidelity and variability (FD) while a regression mapper is slightly better at relevance (CS). Both objective and subjective evaluation on two V2A datasets demonstrate the superiority of our proposed method compared to current state-of-the-art approaches - trained with 86% fewer parameters but achieving 53% and 19% improvement in FD and CS, respectively. Supplementary materials such as audio samples are provided at our demo website: https://v2a-mapper.github.io/.

----

## [1727] Practical Privacy-Preserving MLaaS: When Compressive Sensing Meets Generative Networks

**Authors**: *Jia Wang, Wuqiang Su, Zushu Huang, Jie Chen, Chengwen Luo, Jianqiang Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29476](https://doi.org/10.1609/aaai.v38i14.29476)

**Abstract**:

The Machine-Learning-as-a-Service (MLaaS) framework allows one to grab low-hanging fruit of machine learning techniques and data science, without either much expertise for this sophisticated sphere or provision of specific infrastructures. However, the requirement of revealing all training data to the service provider raises new concerns in terms of privacy leakage, storage consumption, efficiency, bandwidth, etc. In this paper, we propose a lightweight privacy-preserving MLaaS framework by combining Compressive Sensing (CS) and Generative Networks. It’s constructed on the favorable facts observed in recent works that general inference tasks could be fulfilled with generative networks and classifier trained on compressed measurements, since the generator could model the data distribution and capture discriminative information which are useful for classification. To improve the performance of the MLaaS framework, the supervised generative models of the server are trained and optimized with prior knowledge provided by the client. In order to prevent the service provider from recovering the original data as well as identifying the queried results, a noise-addition mechanism is designed and adopted into the compressed data domain. Empirical results confirmed its performance superiority in accuracy and resource consumption against the state-of-the-art privacy preserving MLaaS frameworks.

----

## [1728] Towards Stability and Generalization Bounds in Decentralized Minibatch Stochastic Gradient Descent

**Authors**: *Jiahuan Wang, Hong Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29477](https://doi.org/10.1609/aaai.v38i14.29477)

**Abstract**:

Decentralized Stochastic Gradient Descent (D-SGD) represents an efficient communication approach tailored for mastering insights from vast, distributed datasets. Inspired by parallel optimization paradigms, the incorporation of minibatch serves to diminish variance, consequently expediting the optimization process. Nevertheless, as per our current understanding, the existing literature has not thoroughly explored the learning theory foundation of Decentralized Minibatch Stochastic Gradient Descent (DM-SGD). In this paper, we try to address this theoretical gap by investigating the generalization properties of DM-SGD. We establish the sharper generalization bounds for the DM-SGD algorithm with replacement (without replacement) on (non)convex and (non)smooth cases. Moreover, our results consistently recover to the results of Centralized Stochastic Gradient Descent (C-SGD). In addition, we derive generalization analysis for Zero-Order (ZO) version of DM-SGD.

----

## [1729] SURER: Structure-Adaptive Unified Graph Neural Network for Multi-View Clustering

**Authors**: *Jing Wang, Songhe Feng, Gengyu Lyu, Jiazheng Yuan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29478](https://doi.org/10.1609/aaai.v38i14.29478)

**Abstract**:

Deep Multi-view Graph Clustering (DMGC) aims to partition instances into different groups using the graph information extracted from multi-view data. The mainstream framework of DMGC methods applies graph neural networks to embed structure information into the view-specific representations and fuse them for the consensus representation. However, on one hand, we find that the graph learned in advance is not ideal for clustering as it is constructed by original multi-view data and localized connecting. On the other hand, most existing methods learn the consensus representation in a late fusion manner, which fails to propagate the structure relations across multiple views. Inspired by the observations, we propose a Structure-adaptive Unified gRaph nEural network for multi-view clusteRing (SURER), which can jointly learn a heterogeneous multi-view unified graph and robust graph neural networks for multi-view clustering. Specifically, we first design a graph structure learning module to refine the original view-specific attribute graphs, which removes false edges and discovers the potential connection. According to the view-specific refined attribute graphs, we integrate them into a unified heterogeneous graph by linking the representations of the same sample from different views. Furthermore, we use the unified heterogeneous graph as the input of the graph neural network to learn the consensus representation for each instance, effectively integrating complementary information from various views. Extensive experiments on diverse datasets demonstrate the superior effectiveness of our method compared to other state-of-the-art approaches.

----

## [1730] Rethinking Graph Masked Autoencoders through Alignment and Uniformity

**Authors**: *Liang Wang, Xiang Tao, Qiang Liu, Shu Wu, Liang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29479](https://doi.org/10.1609/aaai.v38i14.29479)

**Abstract**:

Self-supervised learning on graphs can be bifurcated into contrastive and generative methods. Contrastive methods, also known as graph contrastive learning (GCL), have dominated graph self-supervised learning in the past few years, but the recent advent of graph masked autoencoder (GraphMAE) rekindles the momentum behind generative methods. Despite the empirical success of GraphMAE, there is still a dearth of theoretical understanding regarding its efficacy. Moreover, while both generative and contrastive methods have been shown to be effective, their connections and differences have yet to be thoroughly investigated. Therefore, we theoretically build a bridge between GraphMAE and GCL, and prove that the node-level reconstruction objective in GraphMAE implicitly performs context-level GCL. Based on our theoretical analysis, we further identify the limitations of the GraphMAE from the perspectives of alignment and uniformity, which have been considered as two key properties of high-quality representations in GCL. We point out that GraphMAE's alignment performance is restricted by the masking strategy, and the uniformity is not strictly guaranteed. To remedy the aforementioned limitations, we propose an Alignment-Uniformity enhanced Graph Masked AutoEncoder, named AUG-MAE. Specifically, we propose an easy-to-hard adversarial masking strategy to provide hard-to-align samples, which improves the alignment performance. Meanwhile, we introduce an explicit uniformity regularizer to ensure the uniformity of the learned representations. Experimental results on benchmark datasets demonstrate the superiority of our model over existing state-of-the-art methods. The code is available at: https://github.com/AzureLeon1/AUG-MAE.

----

## [1731] GOODAT: Towards Test-Time Graph Out-of-Distribution Detection

**Authors**: *Luzhi Wang, Dongxiao He, He Zhang, Yixin Liu, Wenjie Wang, Shirui Pan, Di Jin, Tat-Seng Chua*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29480](https://doi.org/10.1609/aaai.v38i14.29480)

**Abstract**:

Graph neural networks (GNNs) have found widespread application in modeling graph data across diverse domains. While GNNs excel in scenarios where the testing data shares the distribution of their training counterparts (in distribution, ID), they often exhibit incorrect predictions when confronted with samples from an unfamiliar distribution (out-of-distribution, OOD). To identify and reject OOD samples with GNNs, recent studies have explored graph OOD detection, often focusing on training a specific model or modifying the data on top of a well-trained GNN. Despite their effectiveness, these methods come with heavy training resources and costs, as they need to optimize the GNN-based models on training data. Moreover, their reliance on modifying the original GNNs and accessing training data further restricts their universality. To this end, this paper introduces a method to detect Graph Out-of-Distribution At Test-time (namely GOODAT), a data-centric, unsupervised, and plug-and-play solution that operates independently of training data and modifications of GNN architecture. With a lightweight graph masker, GOODAT can learn informative subgraphs from test samples, enabling the capture of distinct graph patterns between OOD and ID samples. To optimize the graph masker, we meticulously design three unsupervised objective functions based on the graph information bottleneck principle, motivating the masker to capture compact yet informative subgraphs for OOD detection. Comprehensive evaluations confirm that our GOODAT method outperforms state-of-the-art benchmarks across a variety of real-world datasets.

----

## [1732] TurboSVM-FL: Boosting Federated Learning through SVM Aggregation for Lazy Clients

**Authors**: *Mengdi Wang, Anna Bodonhelyi, Efe Bozkir, Enkelejda Kasneci*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29481](https://doi.org/10.1609/aaai.v38i14.29481)

**Abstract**:

Federated learning is a distributed collaborative machine learning paradigm that has gained strong momentum in recent years. In federated learning, a central server periodically coordinates models with clients and aggregates the models trained locally by clients without necessitating access to local data. Despite its potential, the implementation of federated learning continues to encounter several challenges, predominantly the slow convergence that is largely due to data heterogeneity. The slow convergence becomes particularly problematic in cross-device federated learning scenarios where clients may be strongly limited by computing power and storage space, and hence counteracting methods that induce additional computation or memory cost on the client side such as auxiliary objective terms and larger training iterations can be impractical. In this paper, we propose a novel federated aggregation strategy, TurboSVM-FL, that poses no additional computation burden on the client side and can significantly accelerate convergence for federated classification task, especially when clients are "lazy" and train their models solely for few epochs for next global aggregation. TurboSVM-FL extensively utilizes support vector machine to conduct selective aggregation and max-margin spread-out regularization on class embeddings. We evaluate TurboSVM-FL on multiple datasets including FEMNIST, CelebA, and Shakespeare using user-independent validation with non-iid data distribution. Our results show that TurboSVM-FL can significantly outperform existing popular algorithms on convergence rate and reduce communication rounds while delivering better test metrics including accuracy, F1 score, and MCC.

----

## [1733] MetaCARD: Meta-Reinforcement Learning with Task Uncertainty Feedback via Decoupled Context-Aware Reward and Dynamics Components

**Authors**: *Min Wang, Xin Li, Leiji Zhang, Mingzhong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29482](https://doi.org/10.1609/aaai.v38i14.29482)

**Abstract**:

Meta-Reinforcement Learning (Meta-RL) aims to reveal shared characteristics in dynamics and reward functions across diverse training tasks. This objective is achieved by meta-learning a policy that is conditioned on task representations with encoded trajectory data or context, thus allowing rapid adaptation to new tasks from a known task distribution. However, since the trajectory data generated by the policy may be biased, the task inference module tends to form spurious correlations between trajectory data and specific tasks, thereby leading to poor adaptation to new tasks. To address this issue, we propose the Meta-RL with task unCertAinty feedback through decoupled context-aware Reward and Dynamics components (MetaCARD). MetaCARD distinctly decouples the dynamics and rewards when inferring tasks and integrates task uncertainty feedback from policy evaluation into the task inference module. This design effectively reduces uncertainty in tasks with changes in dynamics or/and reward functions, thereby enabling accurate task identification and adaptation. The experiment results on both Meta-World and classical MuJoCo benchmarks show that MetaCARD significantly outperforms prevailing Meta-RL baselines, demonstrating its remarkable adaptation ability in sophisticated environments that involve changes in both reward functions and dynamics.

----

## [1734] Considering Nonstationary within Multivariate Time Series with Variational Hierarchical Transformer for Forecasting

**Authors**: *Muyao Wang, Wenchao Chen, Bo Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29483](https://doi.org/10.1609/aaai.v38i14.29483)

**Abstract**:

The forecasting of Multivariate Time Series (MTS) has long been an important but challenging task. Due to the non-stationary problem across long-distance time steps, previous studies primarily adopt stationarization method to attenuate the non-stationary problem of original series for better predictability. However, existed methods always adopt the stationarized series, which ignore the inherent non-stationarity, and have difficulty in modeling MTS with complex distributions due to the lack of stochasticity. To tackle these problems, we first develop a powerful hierarchical probabilistic generative module to consider the non-stationarity and stochastity characteristics within MTS, and then combine it with transformer for a well-defined variational generative dynamic model named Hierarchical Time series Variational Transformer (HTV-Trans), which recovers the intrinsic non-stationary information into temporal dependencies. Being an powerful probabilistic model, HTV-Trans is utilized to learn expressive representations of MTS and applied to the forecasting tasks. Extensive experiments on diverse datasets show the efficiency of HTV-Trans on MTS forecasting tasks.

----

## [1735] Controller-Guided Partial Label Consistency Regularization with Unlabeled Data

**Authors**: *Qian-Wei Wang, Bowen Zhao, Mingyan Zhu, Tianxiang Li, Zimo Liu, Shu-Tao Xia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29484](https://doi.org/10.1609/aaai.v38i14.29484)

**Abstract**:

Partial label learning (PLL) learns from training examples each associated with multiple candidate labels, among which only one is valid. In recent years, benefiting from the strong capability of dealing with ambiguous supervision and the impetus of modern data augmentation methods, consistency regularization-based PLL methods have achieved a series of successes and become mainstream. However, as the partial annotation becomes insufficient, their performances drop significantly. In this paper, we leverage easily accessible unlabeled examples to facilitate the partial label consistency regularization. In addition to a partial supervised loss, our method performs a controller-guided consistency regularization at both the label-level and representation-level with the help of unlabeled data. To minimize the disadvantages of insufficient capabilities of the initial supervised model, we use the controller to estimate the confidence of each current prediction to guide the subsequent consistency regularization. Furthermore, we dynamically adjust the confidence thresholds so that the number of samples of each class participating in consistency regularization remains roughly equal to alleviate the problem of class-imbalance. Experiments show that our method achieves satisfactory performances in more practical situations, and its modules can be applied to existing PLL methods to enhance their capabilities.

----

## [1736] A Bregman Proximal Stochastic Gradient Method with Extrapolation for Nonconvex Nonsmooth Problems

**Authors**: *Qingsong Wang, Zehui Liu, Chunfeng Cui, Deren Han*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29485](https://doi.org/10.1609/aaai.v38i14.29485)

**Abstract**:

In this paper, we explore a specific optimization problem that involves the combination of a differentiable nonconvex function and a nondifferentiable function. The differentiable component lacks a global Lipschitz continuous gradient, posing challenges for optimization.  To address this issue and accelerate the convergence, we propose a Bregman proximal stochastic gradient method with extrapolation (BPSGE), which only requires smooth adaptivity of the differentiable part.  Under variance reduction framework, we not only analyze the subsequential and global convergence of the proposed algorithm under certain conditions,  but also analyze the sublinear convergence rate of the subsequence, and the complexity of the algorithm, revealing that the BPSGE algorithm requires at most  O(epsilon\^\,(-2)) iterations in expectation to attain an epsilon-stationary point. To validate the effectiveness of our proposed algorithm, we conduct numerical experiments on three real-world applications: graph regularized nonnegative matrix factorization (NMF), matrix factorization with weakly-convex regularization, and NMF with nonconvex sparsity constraints. These experiments demonstrate that BPSGE is faster than the baselines without extrapolation. The code is available at: https://github.com/nothing2wang/BPSGE-Algorithm.

----

## [1737] ND-MRM: Neuronal Diversity Inspired Multisensory Recognition Model

**Authors**: *Qixin Wang, Chaoqiong Fan, Tianyuan Jia, Yuyang Han, Xia Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29486](https://doi.org/10.1609/aaai.v38i14.29486)

**Abstract**:

Cross-sensory interaction is a key aspect for multisensory recognition. Without cross-sensory interaction, artificial neural networks show inferior performance in multisensory recognition. On the contrary, the human brain has an inherently remarkable ability in multisensory recognition, which stems from the diverse neurons that exhibit distinct responses to sensory inputs, especially the multisensory neurons with multisensory responses hence enabling cross-sensory interaction. Based on this neuronal diversity, we propose a Neuronal Diversity inspired Multisensory Recognition Model (ND-MRM), which, similar to the brain, comprises unisensory neurons and multisensory neurons. To reflect the different responses characteristics of diverse neurons in the brain, special connection constraints are innovatively designed to regulate the features transmission in the ND-MRM. Leveraging this novel concept of neuronal diversity, our model is biologically plausible, enabling more effective recognition of multisensory information. To validate the performance of the proposed ND-MRM, we employ a multisensory emotion recognition task as a case study. The results demonstrate that our model surpasses state-of-the-art brain-inspired baselines on two datasets, proving the potential of brain-inspired methods for advancing multisensory interaction and recognition.

----

## [1738] AQ-DETR: Low-Bit Quantized Detection Transformer with Auxiliary Queries

**Authors**: *Runqi Wang, Huixin Sun, Linlin Yang, Shaohui Lin, Chuanjian Liu, Yan Gao, Yao Hu, Baochang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29487](https://doi.org/10.1609/aaai.v38i14.29487)

**Abstract**:

DEtection TRansformer (DETR)-based models have achieved remarkable performance. However, they are accompanied by a large computation overhead cost, which significantly prevents their applications on resource-limited devices. Prior arts attempt to reduce the computational burden of DETR using low-bit quantization, while these methods sacrifice a severe significant performance on weight-activation-attention low-bit quantization. We observe that the number of matching queries and positive samples affect much on the representation capacity of queries in DETR, while quantifying queries of DETR further reduces its representational capacity, thus leading to a severe performance drop. We introduce a new quantization strategy based on Auxiliary Queries for DETR (AQ-DETR), aiming to enhance the capacity of quantized queries. In addition, a layer-by-layer distillation is proposed to reduce the quantization error between quantized attention and full-precision counterpart. Through our extensive experiments on large-scale open datasets, the performance of the 4-bit quantization of DETR and Deformable DETR models is comparable to  full-precision counterparts.

----

## [1739] Constrained Bayesian Optimization under Partial Observations: Balanced Improvements and Provable Convergence

**Authors**: *Shengbo Wang, Ke Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29488](https://doi.org/10.1609/aaai.v38i14.29488)

**Abstract**:

The partially observable constrained optimization problems (POCOPs) impede data-driven optimization techniques since an infeasible solution of POCOPs can provide little information about the objective as well as the constraints. We endeavor to design an efficient and provable method for expensive POCOPs under the framework of constrained Bayesian optimization. Our method consists of two key components. Firstly, we present an improved design of the acquisition functions that introduce balanced exploration during optimization. We rigorously study the convergence properties of this design to demonstrate its effectiveness. Secondly, we propose Gaussian processes embedding different likelihoods as the surrogate model for partially observable constraints. This model leads to a more accurate representation of the feasible regions compared to traditional classification-based models. Our proposed method is empirically studied on both synthetic and real-world problems. The results demonstrate the competitiveness of our method for solving POCOPs.

----

## [1740] Online Restless Multi-Armed Bandits with Long-Term Fairness Constraints

**Authors**: *Shufan Wang, Guojun Xiong, Jian Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29489](https://doi.org/10.1609/aaai.v38i14.29489)

**Abstract**:

Restless multi-armed bandits (RMAB) have been widely used to model sequential decision making problems with constraints. The decision maker (DM) aims to maximize the expected total reward over an infinite horizon under an “instantaneous activation constraint” that at most B arms can be activated at any decision epoch, where the state of each arm evolves stochastically according to a Markov decision process (MDP). However, this basic model fails to provide any fairness guarantee among arms. In this paper, we introduce RMAB-F, a new RMAB model with “long-term fairness constraints”, where the objective now is to maximize the longterm reward while a minimum long-term activation fraction for each arm must be satisfied. For the online RMAB-F setting (i.e., the underlying MDPs associated with each arm are unknown to the DM), we develop a novel reinforcement learning (RL) algorithm named Fair-UCRL. We prove that Fair-UCRL ensures probabilistic sublinear bounds on both the reward regret and the fairness violation regret. Compared with off-the-shelf RL methods, our Fair-UCRL is much more computationally efficient since it contains a novel exploitation that leverages a low-complexity index policy for making decisions. Experimental results further demonstrate the effectiveness of our Fair-UCRL.

----

## [1741] Exploring Gradient Explosion in Generative Adversarial Imitation Learning: A Probabilistic Perspective

**Authors**: *Wanying Wang, Yichen Zhu, Yirui Zhou, Chaomin Shen, Jian Tang, Zhiyuan Xu, Yaxin Peng, Yangchun Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29490](https://doi.org/10.1609/aaai.v38i14.29490)

**Abstract**:

Generative Adversarial Imitation Learning (GAIL) stands as a cornerstone approach in imitation learning. This paper investigates the gradient explosion in two types of GAIL: GAIL with deterministic policy (DE-GAIL) and GAIL with stochastic policy (ST-GAIL). We begin with the observation that the training can be highly unstable for DE-GAIL at the beginning of the training phase and end up divergence. Conversely, the ST-GAIL training trajectory remains consistent, reliably converging. To shed light on these disparities, we provide an explanation from a theoretical perspective. By establishing a probabilistic lower bound for GAIL, we demonstrate that gradient explosion is an inevitable outcome for DE-GAIL due to occasionally large expert-imitator policy disparity, whereas ST-GAIL does not have the issue with it. To substantiate our assertion, we illustrate how modifications in the reward function can mitigate the gradient explosion challenge. Finally, we propose CREDO, a simple yet effective strategy that clips the reward function during the training phase, allowing the GAIL to enjoy high data efficiency and stable trainability.

----

## [1742] IGAMT: Privacy-Preserving Electronic Health Record Synthesization with Heterogeneity and Irregularity

**Authors**: *Wenjie Wang, Pengfei Tang, Jian Lou, Yuanming Shao, Lance Waller, Yi-an Ko, Li Xiong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29491](https://doi.org/10.1609/aaai.v38i14.29491)

**Abstract**:

Integrating electronic health records (EHR) into machine learning-driven clinical research and hospital applications is important, as it harnesses extensive and high-quality patient data to enhance outcome predictions and treatment personalization. Nonetheless, due to privacy and security concerns, the secondary purpose of EHR data is consistently governed and regulated, primarily for research intentions, thereby constraining researchers' access to EHR data. Generating synthetic EHR data with deep learning methods is a viable and promising approach to mitigate privacy concerns, offering not only a supplementary resource for downstream applications but also sidestepping the confidentiality risks associated with real patient data. While prior efforts have concentrated on EHR data synthesis, significant challenges persist in the domain of generating synthetic EHR data: balancing the heterogeneity of real EHR including temporal and non-temporal features, addressing the missing values and irregular measures, and ensuring the privacy of the real data used for model training. Existing works in this domain only focused on solving one or two aforementioned challenges. In this work, we propose IGAMT, an innovative framework to generate privacy-preserved synthetic EHR data that not only maintain high quality with heterogeneous features, missing values, and irregular measures but also balances the privacy-utility trade-off. Extensive experiments prove that IGAMT significantly outperforms baseline architectures in terms of visual resemblance and comparable performance in downstream applications. Ablation case studies also prove the effectiveness of the techniques applied in IGAMT.

----

## [1743] Decoupled Training: Return of Frustratingly Easy Multi-Domain Learning

**Authors**: *Ximei Wang, Junwei Pan, Xingzhuo Guo, Dapeng Liu, Jie Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29492](https://doi.org/10.1609/aaai.v38i14.29492)

**Abstract**:

Multi-domain learning (MDL) aims to train a model with minimal average risk across multiple overlapping but non-identical domains. To tackle the challenges of dataset bias and domain domination, numerous MDL approaches have been proposed from the perspectives of seeking commonalities by aligning distributions to reduce domain gap or reserving differences by implementing domain-specific towers, gates, and even experts. MDL models are becoming more and more complex with sophisticated network architectures or loss functions, introducing extra parameters and enlarging computation costs. In this paper, we propose a frustratingly easy and hyperparameter-free multi-domain learning method named Decoupled Training (D-Train). D-Train is a tri-phase general-to-specific training strategy that first pre-trains on all domains to warm up a root model, then post-trains on each domain by splitting into multi-heads, and finally fine-tunes the heads by fixing the backbone, enabling decouple training to achieve domain independence. Despite its extraordinary simplicity and efficiency, D-Train performs remarkably well in extensive evaluations of various datasets from standard benchmarks to applications of satellite imagery and recommender systems.

----

## [1744] Probability-Polarized Optimal Transport for Unsupervised Domain Adaptation

**Authors**: *Yan Wang, Chuan-Xian Ren, Yi-Ming Zhai, You-Wei Luo, Hong Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29493](https://doi.org/10.1609/aaai.v38i14.29493)

**Abstract**:

Optimal transport (OT) is an important methodology to measure distribution discrepancy, which has achieved promising performance in artificial intelligence applications, e.g., unsupervised domain adaptation. However, from the view of transportation, there are still limitations: 1) the local discriminative structures for downstream tasks, e.g., cluster structure for classification, cannot be explicitly admitted by the learned OT plan; 2) the entropy regularization induces a dense OT plan with increasing uncertainty. To tackle these issues, we propose a novel Probability-Polarized OT (PPOT) framework, which can characterize the structure of OT plan explicitly. Specifically, the probability polarization mechanism is proposed to guide the optimization direction of OT plan, which generates a clear margin between similar and dissimilar transport pairs and reduces the uncertainty. Further, a dynamic mechanism for margin is developed by incorporating task-related information into the polarization, which directly captures the intra/inter class correspondence for knowledge transportation. A mathematical understanding for PPOT is provided from the view of gradient, which ensures interpretability. Extensive experiments on several datasets validate the effectiveness and empirical efficiency of PPOT.

----

## [1745] Limited-Supervised Multi-Label Learning with Dependency Noise

**Authors**: *Yejiang Wang, Yuhai Zhao, Zhengkui Wang, Wen Shan, Xingwei Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29494](https://doi.org/10.1609/aaai.v38i14.29494)

**Abstract**:

Limited-supervised multi-label learning (LML) leverages weak or noisy supervision for multi-label classification model training over data with label noise, which contain missing labels and/or redundant labels. Existing studies usually solve LML problems by assuming that label noise is independent of the input features and class labels, while ignoring the fact that noisy labels may depend on the input features (instance-dependent) and the classes (label-dependent) in many real-world applications. In this paper, we propose limited-supervised Multi-label Learning with Dependency Noise (MLDN) to simultaneously identify the instance-dependent and label-dependent label noise by factorizing the noise matrix as the outputs of a mapping from the feature and label representations. Meanwhile, we regularize the problem with the manifold constraint on noise matrix to preserve local relationships and uncover the manifold structure. Theoretically, we bound noise recover error for the resulting problem. We solve the problem by using a first-order scheme based on proximal operator, and the convergence rate of it is at least sub-linear. Extensive experiments conducted on various datasets demonstrate the superiority of our proposed method.

----

## [1746] Non-stationary Projection-Free Online Learning with Dynamic and Adaptive Regret Guarantees

**Authors**: *Yibo Wang, Wenhao Yang, Wei Jiang, Shiyin Lu, Bing Wang, Haihong Tang, Yuanyu Wan, Lijun Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29495](https://doi.org/10.1609/aaai.v38i14.29495)

**Abstract**:

Projection-free online learning has drawn increasing interest due to its efficiency in solving high-dimensional problems with complicated constraints. However, most existing projection-free online methods focus on minimizing the static regret, which unfortunately fails to capture the challenge of changing environments. In this paper, we investigate non-stationary projection-free online learning, and choose dynamic regret and adaptive regret to measure the performance. Specifically, we first provide a novel dynamic regret analysis for an existing projection-free method named BOGD_IP, and establish an O(T^¾ (1+P_T)) dynamic regret bound, where P_T denotes the path-length of the comparator sequence. Then, we improve the upper bound to O(T^¾ (1+P_T)^¼) by running multiple BOGD_IP algorithms with different step sizes in parallel, and tracking the best one on the fly. Our results are the first general-case dynamic regret bounds for projection-free online learning, and can recover the existing O(T^¾) static regret by setting P_T = 0. Furthermore, we propose a projection-free method to attain an O(?^¾) adaptive regret bound for any interval with length ?, which nearly matches the static regret over that interval. The essential idea is to maintain a set of BOGD_IP algorithms dynamically, and combine them by a meta algorithm. Moreover, we demonstrate that it is also equipped with an O(T^¾ (1+P_T)^¼) dynamic regret bound. Finally, empirical studies verify our theoretical findings.

----

## [1747] Wavelet Dynamic Selection Network for Inertial Sensor Signal Enhancement

**Authors**: *Yifeng Wang, Yi Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29496](https://doi.org/10.1609/aaai.v38i14.29496)

**Abstract**:

As attitude and motion sensing components, inertial sensors are widely used in various portable devices, covering consumer electronics, sports health, aerospace, etc. But the severe intrinsic errors of inertial sensors heavily restrain their function implementation, especially the advanced functionality, including motion trajectory recovery and motion semantic recognition, which attracts considerable attention. As a mainstream signal processing method, wavelet is hailed as the mathematical microscope of signal due to the plentiful and diverse wavelet basis functions. However, complicated noise types and application scenarios of inertial sensors make selecting wavelet basis perplexing. To this end, we propose a wavelet dynamic selection network (WDSNet), which intelligently selects the appropriate wavelet basis for variable inertial signals. In addition, existing deep learning architectures excel at extracting features from input data but neglect to learn the characteristics of target categories, which is essential to enhance the category awareness capability, thereby improving the selection of wavelet basis. Therefore, we propose a category representation mechanism (CRM), which enables the network to extract and represent category features without increasing trainable parameters. Furthermore, CRM transforms the common fully connected network into category representations, which provide closer supervision to the feature extractor than the far and trivial one-hot classification labels. We call this process of imposing interpretability on a network and using it to supervise the feature extractor the feature supervision mechanism, and its effectiveness is demonstrated experimentally and theoretically in this paper. The enhanced inertial signal can perform impracticable tasks with regard to the original signal, such as trajectory reconstruction. Both quantitative and visual results show that WDSNet outperforms the existing methods. Remarkably, WDSNet, as a weakly-supervised method, achieves the state-of-the-art performance of all the compared fully-supervised methods.

----

## [1748] Lost Domain Generalization Is a Natural Consequence of Lack of Training Domains

**Authors**: *Yimu Wang, Yihan Wu, Hongyang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29497](https://doi.org/10.1609/aaai.v38i14.29497)

**Abstract**:

We show a hardness result for the number of training domains required to achieve a small population error in the test domain. Although many domain generalization algorithms have been developed under various domain-invariance assumptions, there is significant evidence to indicate that out-of-distribution (o.o.d.) test accuracy of state-of-the-art o.o.d. algorithms is on par with empirical risk minimization and random guess on the domain generalization benchmarks such as DomainBed. In this work, we analyze its cause and attribute the lost domain generalization to the lack of training domains. We show that, in a minimax lower bound fashion, any learning algorithm that outputs a classifier with an ε excess error to the Bayes optimal classifier requires at least poly(1/ε) number of training domains, even though the number of training data sampled from each training domain is large. Experiments on the DomainBed benchmark demonstrate that o.o.d. test accuracy is monotonically increasing as the number of training domains increases. Our result sheds light on the intrinsic hardness of domain generalization and suggests benchmarking o.o.d. algorithms by the datasets with a sufficient number of training domains.

----

## [1749] Semi-supervised Learning of Dynamical Systems with Neural Ordinary Differential Equations: A Teacher-Student Model Approach

**Authors**: *Yu Wang, Yuxuan Yin, Karthik Somayaji N. S., Ján Drgona, Malachi Schram, Mahantesh Halappanavar, Frank Liu, Peng Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29498](https://doi.org/10.1609/aaai.v38i14.29498)

**Abstract**:

Modeling dynamical systems is crucial for a wide range of tasks, but it remains challenging due to complex nonlinear dynamics, limited observations, or lack of prior knowledge. Recently, data-driven approaches such as Neural Ordinary Differential Equations (NODE) have shown promising results by leveraging the expressive power of neural networks to model unknown dynamics. However, these approaches often suffer from limited labeled training data, leading to poor generalization and suboptimal predictions. On the other hand, semi-supervised algorithms can utilize abundant unlabeled data and have demonstrated  good performance in classification and regression tasks.
We propose TS-NODE, the first semi-supervised approach to modeling dynamical systems with NODE. TS-NODE explores cheaply generated synthetic pseudo rollouts to broaden exploration in the state space and to tackle the challenges brought by lack of ground-truth system data under a teacher-student model.  TS-NODE employs an unified optimization framework that corrects the teacher model based on the student's feedback while mitigating the potential false system dynamics present in pseudo rollouts.
TS-NODE demonstrates significant performance improvements over a baseline Neural ODE model on multiple dynamical system modeling tasks.

----

## [1750] Critic-Guided Decision Transformer for Offline Reinforcement Learning

**Authors**: *Yuanfu Wang, Chao Yang, Ying Wen, Yu Liu, Yu Qiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29499](https://doi.org/10.1609/aaai.v38i14.29499)

**Abstract**:

Recent advancements in offline reinforcement learning (RL) have underscored the capabilities of Return-Conditioned Supervised Learning (RCSL), a paradigm that learns the action distribution based on target returns for each state in a supervised manner. However, prevailing RCSL methods largely focus on deterministic trajectory modeling, disregarding stochastic state transitions and the diversity of future trajectory distributions. A fundamental challenge arises from the inconsistency between the sampled returns within individual trajectories and the expected returns across multiple trajectories. Fortunately, value-based methods offer a solution by leveraging a value function to approximate the expected returns, thereby addressing the inconsistency effectively. Building upon these insights, we propose a novel approach, termed the Critic-Guided Decision Transformer (CGDT), which combines the predictability of long-term returns from value-based methods with the trajectory modeling capability of the Decision Transformer. By incorporating a learned value function, known as the critic, CGDT ensures a direct alignment between the specified target returns and the expected returns of actions. This integration bridges the gap between the deterministic nature of RCSL and the probabilistic characteristics of value-based methods. Empirical evaluations on stochastic environments and D4RL benchmark datasets demonstrate the superiority of CGDT over traditional RCSL methods. These results highlight the potential of CGDT to advance the state of the art in offline RL and extend the applicability of RCSL to a wide range of RL tasks.

----

## [1751] Fully-Connected Spatial-Temporal Graph for Multivariate Time-Series Data

**Authors**: *Yucheng Wang, Yuecong Xu, Jianfei Yang, Min Wu, Xiaoli Li, Lihua Xie, Zhenghua Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29500](https://doi.org/10.1609/aaai.v38i14.29500)

**Abstract**:

Multivariate Time-Series (MTS) data is crucial in various application fields. With its sequential and multi-source (multiple sensors) properties, MTS data inherently exhibits Spatial-Temporal (ST) dependencies, involving temporal correlations between timestamps and spatial correlations between sensors in each timestamp. To effectively leverage this information, Graph Neural Network-based methods (GNNs) have been widely adopted. However, existing approaches separately capture spatial dependency and temporal dependency and fail to capture the correlations between Different sEnsors at Different Timestamps (DEDT). Overlooking such correlations hinders the comprehensive modelling of ST dependencies within MTS data, thus restricting existing GNNs from learning effective representations. To address this limitation, we propose a novel method called Fully-Connected Spatial-Temporal Graph Neural Network (FC-STGNN), including two key components namely FC graph construction and FC graph convolution. For graph construction, we design a decay graph to connect sensors across all timestamps based on their temporal distances, enabling us to fully model the ST dependencies by considering the correlations between DEDT. Further, we devise FC graph convolution with a moving-pooling GNN layer to effectively capture the ST dependencies for learning effective representations. Extensive experiments show the effectiveness of FC-STGNN on multiple MTS datasets compared to SOTA methods. The code is available at https://github.com/Frank-Wang-oss/FCSTGNN.

----

## [1752] Graph-Aware Contrasting for Multivariate Time-Series Classification

**Authors**: *Yucheng Wang, Yuecong Xu, Jianfei Yang, Min Wu, Xiaoli Li, Lihua Xie, Zhenghua Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29501](https://doi.org/10.1609/aaai.v38i14.29501)

**Abstract**:

Contrastive learning, as a self-supervised learning paradigm, becomes popular for Multivariate Time-Series (MTS) classification. It ensures the consistency across different views of unlabeled samples and then learns effective representations for these samples. Existing contrastive learning methods mainly focus on achieving temporal consistency with temporal augmentation and contrasting techniques, aiming to preserve temporal patterns against perturbations for MTS data. However, they overlook spatial consistency that requires the stability of individual sensors and their correlations. As MTS data typically originate from multiple sensors, ensuring spatial consistency becomes essential for the overall performance of contrastive learning on MTS data. Thus, we propose Graph-Aware Contrasting for spatial consistency across MTS data. Specifically, we propose graph augmentations including node and edge augmentations to preserve the stability of sensors and their correlations, followed by graph contrasting with both node- and graph-level contrasting to extract robust sensor- and global-level features. We further introduce multi-window temporal contrasting to ensure temporal consistency in the data for each sensor. Extensive experiments demonstrate that our proposed method achieves state-of-the-art performance on various MTS classification tasks. The code is available at https://github.com/Frank-Wang-oss/TS-GAC.

----

## [1753] Superposed Atomic Representation for Robust High-Dimensional Data Recovery of Multiple Low-Dimensional Structures

**Authors**: *Yulong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29502](https://doi.org/10.1609/aaai.v38i14.29502)

**Abstract**:

This paper proposes a unified Superposed Atomic Representation (SAR) framework for high-dimensional data recovery with multiple low-dimensional structures. The data can be in various forms ranging from vectors to tensors. The goal of SAR is to recover different components from their sum, where each component has a low-dimensional structure, such as sparsity, low-rankness or be lying a low-dimensional subspace. Examples of SAR include, but not limited to, Robust Sparse Representation (RSR), Robust Principal Component Analysis (RPCA), Tensor RPCA (TRPCA), and Outlier Pursuit (OP). We establish the theoretical guarantee for SAR. To further improve SAR, we also develop a Weighted SAR (WSAR) framework by paying more attention and penalizing less on significant atoms of each component. An effective optimization algorithm is devised for WSAR and the convergence of the algorithm is rigorously proved. By leveraging WSAR as a general platform, several new methods are proposed for high-dimensional data recovery. The experiments on real data demonstrate the superiority of WSAR for various data recovery problems.

----

## [1754] Consistency-GAN: Training GANs with Consistency Model

**Authors**: *Yunpeng Wang, Meng Pang, Shengbo Chen, Hong Rao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29503](https://doi.org/10.1609/aaai.v38i14.29503)

**Abstract**:

For generative learning tasks, there are three crucial criteria for generating samples from the models: quality,  coverage/diversity, and sampling speed. Among the existing generative models, Generative adversarial networks (GANs) and diffusion models demonstrate outstanding quality performance while suffering from notable limitations. GANs can generate high-quality results and enable fast sampling, their drawbacks, however, lie in the limited diversity of the generated samples. On the other hand, diffusion models excel at generating high-quality results with a commendable diversity. Yet, its iterative generation process necessitates hundreds to thousands of sampling steps, leading to slow speeds that are impractical for real-time scenarios. To address the aforementioned problem, this paper proposes a novel Consistency-GAN model. In particular, to aid in the training of the GAN, we introduce instance noise, which employs consistency models using only a few steps compared to the conventional diffusion process. Our evaluations on various datasets indicate that our approach significantly accelerates sampling speeds compared to traditional diffusion models, while preserving sample quality and diversity. Furthermore, our approach also has better model coverage than traditional adversarial training methods.

----

## [1755] DRF: Improving Certified Robustness via Distributional Robustness Framework

**Authors**: *Zekai Wang, Zhengyu Zhou, Weiwei Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29504](https://doi.org/10.1609/aaai.v38i14.29504)

**Abstract**:

Randomized smoothing (RS) has provided state-of-the-art (SOTA) certified robustness against adversarial perturbations for large neural networks. Among studies in this field, methods based on adversarial training (AT) achieve remarkably robust performance by applying adversarial examples to construct the smoothed classifier. These AT-based RS methods typically seek a pointwise adversary that generates the worst-case adversarial examples by perturbing each input independently. However, there are unexplored benefits to considering such adversarial robustness across the entire data distribution. To this end, we provide a novel framework called DRF, which connects AT-based RS methods with distributional robustness (DR), and show that these methods are special cases of their counterparts in our framework. Due to the advantages conferred by DR, our framework can control the trade-off between the clean accuracy and certified robustness of smoothed classifiers to a significant extent. Our experiments demonstrate that DRF can substantially improve the certified robustness of AT-based RS.

----

## [1756] BVT-IMA: Binary Vision Transformer with Information-Modified Attention

**Authors**: *Zhenyu Wang, Hao Luo, Xuemei Xie, Fan Wang, Guangming Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29505](https://doi.org/10.1609/aaai.v38i14.29505)

**Abstract**:

As a compression method that can significantly reduce the cost of calculations and memories, model binarization has been extensively studied in convolutional neural networks. However, the recently popular vision transformer models pose new challenges to such a technique, in which the binarized models suffer from serious performance drops. In this paper, an attention shifting is observed in the binary multi-head self-attention module, which can influence the information fusion between tokens and thus hurts the model performance. From the perspective of information theory, we find a correlation between attention scores and the information quantity, further indicating that a reason for such a phenomenon may be the loss of the information quantity induced by constant moduli of binarized tokens. Finally, we reveal the information quantity hidden in the attention maps of binary vision transformers and propose a simple approach to modify the attention values with look-up information tables so that improve the model performance. Extensive experiments on CIFAR-100/TinyImageNet/ImageNet-1k demonstrate the effectiveness of the proposed information-modified attention on binary vision transformers.

----

## [1757] Stealthy Adversarial Attacks on Stochastic Multi-Armed Bandits

**Authors**: *Zhiwei Wang, Huazheng Wang, Hongning Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29506](https://doi.org/10.1609/aaai.v38i14.29506)

**Abstract**:

Adversarial attacks against stochastic multi-armed bandit (MAB) algorithms have been extensively studied in the literature. In this work, we focus on reward poisoning attacks and find most existing attacks can be easily detected by our proposed detection method based on the test of homogeneity, due to their aggressive nature in reward manipulations. This motivates us to study the notion of stealthy attack against stochastic MABs and investigate the resulting attackability. Our analysis shows that against two popularly employed MAB algorithms, UCB1 and $\epsilon$-greedy, the success of a stealthy attack depends on the environmental conditions and the realized reward of the arm pulled in the first round. We also analyze the situation for general MAB algorithms equipped with our attack detection method and find that it is possible to have a stealthy attack that almost always succeeds. This brings new insights into the security risks of MAB algorithms.

----

## [1758] Building Minimal and Reusable Causal State Abstractions for Reinforcement Learning

**Authors**: *Zizhao Wang, Caroline Wang, Xuesu Xiao, Yuke Zhu, Peter Stone*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29507](https://doi.org/10.1609/aaai.v38i14.29507)

**Abstract**:

Two desiderata of reinforcement learning (RL) algorithms are the ability to learn from relatively little experience and the ability to learn policies that generalize to a range of problem specifications. 
In factored state spaces, one approach towards achieving both goals is to learn state abstractions, which only keep the necessary variables for learning the tasks at hand. 
This paper introduces Causal Bisimulation Modeling (CBM), a method that learns the causal relationships in the dynamics and reward functions for each task to derive a minimal, task-specific abstraction. 
CBM leverages and improves implicit modeling to train a high-fidelity causal dynamics model that can be reused for all tasks in the same environment. 
Empirical validation on two manipulation environments and four tasks reveals that CBM's learned implicit dynamics models identify the underlying causal relationships and state abstractions more accurately than explicit ones. Furthermore, the derived state abstractions allow a task learner to achieve near-oracle levels of sample efficiency and outperform baselines on all tasks.

----

## [1759] EAT: Towards Long-Tailed Out-of-Distribution Detection

**Authors**: *Tong Wei, Bo-Lin Wang, Min-Ling Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29508](https://doi.org/10.1609/aaai.v38i14.29508)

**Abstract**:

Despite recent advancements in out-of-distribution (OOD) detection, most current studies assume a class-balanced in-distribution training dataset, which is rarely the case in real-world scenarios. This paper addresses the challenging task of long-tailed OOD detection, where the in-distribution data follows a long-tailed class distribution. The main difficulty lies in distinguishing OOD data from samples belonging to the tail classes, as the ability of a classifier to detect OOD instances is not strongly correlated with its accuracy on the in-distribution classes. To overcome this issue, we propose two simple ideas: (1) Expanding the in-distribution class space by introducing multiple abstention classes. This approach allows us to build a detector with clear decision boundaries by training on OOD data using virtual labels. (2) Augmenting the context-limited tail classes by overlaying images onto the context-rich OOD data. This technique encourages the model to pay more attention to the discriminative features of the tail classes. We provide a clue for separating in-distribution and OOD data by analyzing gradient noise. Through extensive experiments, we demonstrate that our method outperforms the current state-of-the-art on various benchmark datasets. Moreover, our method can be used as an add-on for existing long-tail learning approaches, significantly enhancing their OOD detection performance. Code is available at: https://github.com/Stomach-ache/Long-Tailed-OOD-Detection.

----

## [1760] Levenshtein Distance Embedding with Poisson Regression for DNA Storage

**Authors**: *Xiang Wei, Alan J. X. Guo, Sihan Sun, Mengyi Wei, Wei Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29509](https://doi.org/10.1609/aaai.v38i14.29509)

**Abstract**:

Efficient computation or approximation of Levenshtein distance, a widely-used metric for evaluating sequence similarity, has attracted significant attention with the emergence of DNA storage and other biological applications. Sequence embedding, which maps Levenshtein distance to a conventional distance between embedding vectors, has emerged as a promising solution. In this paper, a novel neural network-based sequence embedding technique using Poisson regression is proposed. We first provide a theoretical analysis of the impact of embedding dimension on model performance and present a criterion for selecting an appropriate embedding dimension. Under this embedding dimension, the Poisson regression is introduced by assuming the Levenshtein distance between sequences of fixed length following a Poisson distribution, which naturally aligns with the definition of Levenshtein distance. Moreover, from the perspective of the distribution of embedding distances, Poisson regression approximates the negative log likelihood of the chi-squared distribution and offers advancements in removing the skewness. Through comprehensive experiments on real DNA storage data, we demonstrate the superior performance of the proposed method compared to state-of-the-art approaches.

----

## [1761] Multi-Source Collaborative Gradient Discrepancy Minimization for Federated Domain Generalization

**Authors**: *Yikang Wei, Yahong Han*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29510](https://doi.org/10.1609/aaai.v38i14.29510)

**Abstract**:

Federated Domain Generalization aims to learn a domain-invariant model from multiple decentralized source domains for deployment on unseen target domain. Due to privacy concerns, the data from different source domains are kept isolated, which poses challenges in bridging the domain gap. To address this issue, we propose a Multi-source Collaborative Gradient Discrepancy Minimization (MCGDM) method for federated domain generalization. Specifically, we propose intra-domain gradient matching between the original images and augmented images to avoid overfitting the domain-specific information within isolated domains. Additionally, we propose inter-domain gradient matching with the collaboration of other domains, which can further reduce the domain shift across decentralized domains. Combining intra-domain and inter-domain gradient matching, our method enables the learned model to generalize well on unseen domains. Furthermore, our method can be extended to the federated domain adaptation task by fine-tuning the target model on the pseudo-labeled target domain. The extensive experiments on federated domain generalization and adaptation indicate that our method outperforms the state-of-the-art methods significantly.

----

## [1762] Auto-Prox: Training-Free Vision Transformer Architecture Search via Automatic Proxy Discovery

**Authors**: *Zimian Wei, Peijie Dong, Zheng Hui, Anggeng Li, Lujun Li, Menglong Lu, Hengyue Pan, Dongsheng Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29511](https://doi.org/10.1609/aaai.v38i14.29511)

**Abstract**:

The substantial success of Vision Transformer (ViT) in computer vision tasks is largely attributed to the architecture design. This underscores the necessity of efficient architecture search for designing better ViTs automatically. As training-based architecture search methods are computationally intensive, there’s a growing interest in training-free methods that use zero-cost proxies to score ViTs. However, existing training-free approaches require expert knowledge to manually design specific zero-cost proxies. Moreover, these zero-cost proxies exhibit limitations to generalize across diverse domains. In this paper, we introduce Auto-Prox, an automatic proxy discovery framework, to address the problem. First, we build the ViT-Bench-101, which involves different ViT candidates and their actual performance on multiple datasets. Utilizing ViT-Bench-101, we can evaluate zero-cost proxies based on their score-accuracy correlation. Then, we represent zero-cost proxies with computation graphs and organize the zero-cost proxy search space with ViT statistics and primitive operations. To discover generic zero-cost proxies, we propose a joint correlation metric to evolve and mutate different zero-cost proxy candidates. We introduce an elitism-preserve strategy for search efficiency to achieve a better trade-off between exploitation and exploration. Based on the discovered zero-cost proxy, we conduct a ViT architecture search in a training-free manner. Extensive experiments demonstrate that our method generalizes well to different datasets and achieves state-of-the-art results both in ranking correlation and final accuracy. Codes can be found at https://github.com/lilujunai/Auto-Prox-AAAI24.

----

## [1763] Hyperbolic Graph Diffusion Model

**Authors**: *Lingfeng Wen, Xuan Tang, Mingjie Ouyang, Xiangxiang Shen, Jian Yang, Daxin Zhu, Mingsong Chen, Xian Wei*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29512](https://doi.org/10.1609/aaai.v38i14.29512)

**Abstract**:

Diffusion generative models (DMs) have achieved promising results in image and graph generation. However, real-world graphs, such as social networks, molecular graphs, and traffic graphs, generally share non-Euclidean topologies and hidden hierarchies. For example, the degree distributions of graphs are mostly power-law distributions. The current latent diffusion model embeds the hierarchical data in a Euclidean space, which leads to distortions and interferes with modeling the distribution. Instead, hyperbolic space has been found to be more suitable for capturing complex hierarchical structures due to its exponential growth property. In order to simultaneously utilize the data generation capabilities of diffusion models and the ability of hyperbolic embeddings to extract latent hierarchical distributions, we propose a novel graph generation method called, Hyperbolic Graph Diffusion Model (HGDM), which consists of an auto-encoder to encode nodes into successive hyperbolic embeddings, and a DM that operates in the hyperbolic latent space. HGDM captures the crucial graph structure distributions by constructing a hyperbolic potential node space that incorporates edge information. Extensive experiments show that HGDM achieves better performance in generic graph and molecule generation benchmarks, with a 48% improvement in the quality of graph generation with highly hierarchical structures.

----

## [1764] Communication Efficient Distributed Newton Method over Unreliable Networks

**Authors**: *Ming Wen, Chengchang Liu, Yuedong Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29513](https://doi.org/10.1609/aaai.v38i14.29513)

**Abstract**:

Distributed optimization in resource constrained devices demands both communication efficiency and fast convergence rates. Newton-type methods are getting preferable due to their superior convergence rates compared to the first-order methods. In this paper, we study a new problem in regard to the second-order distributed optimization over unreliable networks. The working devices are power-limited or operate in unfavorable wireless channels, experiencing packet losses during their uplink transmission to the server. Our scenario is very common in real-world and leads to instability of classical distributed optimization methods especially the second-order methods because of their sensitivity to the imprecision of local Hessian matrices. To achieve robustness to high packet loss, communication efficiency and fast convergence rates, we propose a novel distributed second-order method, called RED-New (Packet loss Resilient Distributed Approximate Newton). Each iteration of RED-New comprises two rounds of light-weight and lossy transmissions, in which the server aggregates the local information with a new developed scaling strategy. We prove the linear-quadratic convergence rate of RED-New. Experimental results demonstrate its advantage over first-order and second-order baselines, and its tolerance to packet loss rate ranging from 5% to 40%.

----

## [1765] Homophily-Related: Adaptive Hybrid Graph Filter for Multi-View Graph Clustering

**Authors**: *Zichen Wen, Yawen Ling, Yazhou Ren, Tianyi Wu, Jianpeng Chen, Xiaorong Pu, Zhifeng Hao, Lifang He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29514](https://doi.org/10.1609/aaai.v38i14.29514)

**Abstract**:

Recently there is a growing focus on graph data, and multi-view graph clustering has become a popular area of research interest. Most of the existing methods are only applicable to homophilous graphs, yet the extensive real-world graph data can hardly fulfill the homophily assumption, where the connected nodes tend to belong to the same class. Several studies have pointed out that the poor performance on heterophilous graphs is actually due to the fact that conventional graph neural networks (GNNs), which are essentially low-pass filters, discard information other than the low-frequency information on the graph. Nevertheless, on certain graphs, particularly heterophilous ones, neglecting high-frequency information and focusing solely on low-frequency information impedes the learning of node representations. To break this limitation, our motivation is to perform graph filtering that is closely related to the homophily degree of the given graph, with the aim of fully leveraging both low-frequency and high-frequency signals to learn distinguishable node embedding. In this work, we propose Adaptive Hybrid Graph Filter for Multi-View Graph Clustering (AHGFC). Specifically, a graph joint process and graph joint aggregation matrix are first designed by using the intrinsic node features and adjacency relationship, which makes the low and high-frequency signals on the graph more distinguishable. Then we design an adaptive hybrid graph filter that is related to the homophily degree, which learns the node embedding based on the graph joint aggregation matrix. After that, the node embedding of each view is weighted and fused into a consensus embedding for the downstream task. Experimental results show that our proposed model performs well on six datasets containing homophilous and heterophilous graphs.

----

## [1766] Reproduce, Replicate, Reevaluate The Long but Safe Way to Extend Machine Learning Methods

**Authors**: *Luisa Werner, Nabil Layaïda, Pierre Genevès, Jérôme Euzenat, Damien Graux*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29515](https://doi.org/10.1609/aaai.v38i14.29515)

**Abstract**:

Reproducibility is a desirable property of scientific research. On the one hand, it increases confidence in results. On the other hand, reproducible results can be extended on a solid basis. In rapidly developing fields such as machine learning, the latter is particularly important to ensure the reliability of research. In this paper, we present a systematic approach to reproducing (using the available implementation), replicating (using an alternative implementation) and reevaluating (using different datasets) state-of-the-art experiments. This approach enables the early detection and correction of deficiencies and thus the development of more robust and transparent machine learning methods. We detail the independent reproduction, replication, and reevaluation of the initially published experiments with a method that we want to extend. For each step, we identify issues and draw lessons learned. We further discuss solutions that have proven effective in overcoming the encountered problems. This work can serve as a guide for further reproducibility studies and generally improve reproducibility in machine learning.

----

## [1767] Robust Loss Functions for Training Decision Trees with Noisy Labels

**Authors**: *Jonathan Wilton, Nan Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29516](https://doi.org/10.1609/aaai.v38i14.29516)

**Abstract**:

We consider training decision trees using noisily labeled data, focusing on loss functions that can lead to robust learning algorithms. Our contributions are threefold. First, we offer novel theoretical insights on the robustness of many existing loss functions in the context of decision tree learning. We show that some of the losses belong to a class of what we call conservative losses, and the conservative losses lead to an early stopping behavior during training and noise-tolerant predictions during testing. Second, we introduce a framework for constructing robust loss functions, called distribution losses. These losses apply percentile-based penalties based on an assumed margin distribution, and they naturally allow adapting to different noise rates via a robustness parameter. In particular, we introduce a new loss called the negative exponential loss, which leads to an efficient greedy impurity-reduction learning algorithm. Lastly, our experiments on multiple datasets and noise settings validate our theoretical insight and the effectiveness of our adaptive negative exponential loss.

----

## [1768] Neural Network Approximation for Pessimistic Offline Reinforcement Learning

**Authors**: *Di Wu, Yuling Jiao, Li Shen, Haizhao Yang, Xiliang Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29517](https://doi.org/10.1609/aaai.v38i14.29517)

**Abstract**:

Deep reinforcement learning (RL) has shown remarkable success in specific offline decision-making scenarios, yet its theoretical guarantees are still under development. Existing works on offline RL theory primarily emphasize a few trivial settings, such as linear MDP or general function approximation with strong assumptions and independent data, which lack guidance for practical use. The coupling of deep learning and Bellman residuals makes this problem challenging, in addition to the difficulty of data dependence. In this paper, we establish a non-asymptotic estimation error of pessimistic offline RL using general neural network approximation with C-mixing data regarding the structure of networks, the dimension of datasets, and the concentrability of data coverage, under mild assumptions. Our result shows that the estimation error consists of two parts: the first converges to zero at a desired rate on the sample size with partially controllable concentrability, and the second becomes negligible if the residual constraint is tight. This result demonstrates the explicit efficiency of deep adversarial offline RL frameworks. We utilize the empirical process tool for C-mixing sequences and the neural network approximation theory for the Holder class to achieve this. We also develop methods to bound the Bellman estimation error caused by function approximation with empirical Bellman constraint perturbations. Additionally, we present a result that lessens the curse of dimensionality using data with low intrinsic dimensionality and function classes with low complexity. Our estimation provides valuable insights into the development of deep offline RL and guidance for algorithm model design.

----

## [1769] Solving Spectrum Unmixing as a Multi-Task Bayesian Inverse Problem with Latent Factors for Endmember Variability

**Authors**: *Dong Wu, Mingmin Chi, Xuan Zang, Bo Peng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29518](https://doi.org/10.1609/aaai.v38i14.29518)

**Abstract**:

With the increasing customization of spectrometers, spectral unmixing has become a widely used technique in fields such as remote sensing, textiles, and environmental protection.
However, endmember variability is a common issue for unmixing, where changes in lighting, atmospheric, temporal conditions, or the intrinsic spectral characteristics of materials, can all result in variations in the measured spectrum.
Recent studies have employed deep neural networks to tackle endmember variability. However, these approaches rely on generic networks to implicitly resolve the issue, which struggles with the ill-posed nature and lack of effective convergence constraints for endmember variability. This paper proposes a streamlined multi-task learning model to rectify this problem, incorporating abundance regression and multi-label classification with Unmixing as a Bayesian Inverse Problem, denoted as BIPU. 
To address the issue of the ill-posed nature, the uncertainty of unmixing is quantified and minimized through the Laplace approximation in a Bayesian inverse solver. In addition, to improve convergence under the influence of endmember variability, the paper introduces two types of constraints. The first separates background factors of variants from the initial factors for each endmember, while the second identifies and eliminates the influence of non-existent endmembers via multi-label classification during convergence.
The effectiveness of this model is demonstrated not only on a self-collected near-infrared spectral textile dataset (FENIR), but also on three commonly used remote sensing hyperspectral image datasets, where it achieves state-of-the-art unmixing performance and exhibits strong generalization capabilities.

----

## [1770] Distilling Reliable Knowledge for Instance-Dependent Partial Label Learning

**Authors**: *Dong-Dong Wu, Deng-Bao Wang, Min-Ling Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29519](https://doi.org/10.1609/aaai.v38i14.29519)

**Abstract**:

Partial label learning (PLL) refers to the classification task where each training instance is ambiguously annotated with a set of candidate labels. Despite substantial advancements in tackling this challenge, limited attention has been devoted to a more specific and realistic setting, denoted as instance-dependent partial label learning (IDPLL). Within this contex, the assignment of partial labels depends on the distinct features of individual instances, rather than being random. In this paper, we initiate an exploration into a self-distillation framework for this problem, driven by the proven effectiveness and stability of this framework. Nonetheless, a crucial shortfall is identified: the foundational assumption central to IDPLL, involving what we term as partial label knowledge stipulating that candidate labels should exhibit superior confidence compared to non-candidates, is not fully upheld within the distillation process. To address this challenge, we introduce DIRK, a novel distillation approach that leverages a rectification process to DIstill Reliable Knowledge, while concurrently preserves informative fine-grained label confidence. In addition, to harness the rectified confidence to its fullest potential, we propose a knowledge-based representation refinement module, seamlessly integrated into the DIRK framework. This module effectively transmits the essence of similarity knowledge from the label space to the feature space, thereby amplifying representation learning and subsequently engendering marked improvements in model performance. Experiments and analysis on multiple datasets validate the rationality and superiority of our proposed approach.

----

## [1771] OCEAN-MBRL: Offline Conservative Exploration for Model-Based Offline Reinforcement Learning

**Authors**: *Fan Wu, Rui Zhang, Qi Yi, Yunkai Gao, Jiaming Guo, Shaohui Peng, Siming Lan, Husheng Han, Yansong Pan, Kaizhao Yuan, Pengwei Jin, Ruizhi Chen, Yunji Chen, Ling Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29520](https://doi.org/10.1609/aaai.v38i14.29520)

**Abstract**:

Model-based offline reinforcement learning (RL) algorithms have emerged as a promising paradigm for offline RL.
These algorithms usually learn a dynamics model from a static dataset of transitions, use the model to generate synthetic trajectories, and perform conservative policy optimization within these trajectories. 
However, our observations indicate that policy optimization methods used in these model-based offline RL algorithms are not effective at exploring the learned model and induce biased exploration, which ultimately impairs the performance of the algorithm.
To address this issue, we propose Offline Conservative ExplorAtioN (OCEAN),  a novel rollout approach to model-based offline RL.
In our method, we incorporate additional exploration techniques and introduce three conservative constraints based on uncertainty estimation to mitigate the potential impact of significant dynamic errors resulting from exploratory transitions. 
Our work is a plug-in method and can be combined with classical model-based RL algorithms, such as MOPO, COMBO, and RAMBO.
Experiment results of our method on the D4RL MuJoCo benchmark show that OCEAN significantly improves the performance of existing algorithms.

----

## [1772] Earthfarsser: Versatile Spatio-Temporal Dynamical Systems Modeling in One Model

**Authors**: *Hao Wu, Yuxuan Liang, Wei Xiong, Zhengyang Zhou, Wei Huang, Shilong Wang, Kun Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29521](https://doi.org/10.1609/aaai.v38i14.29521)

**Abstract**:

Efficiently modeling spatio-temporal (ST) physical processes and observations presents a challenging problem for the deep learning community. Many recent studies have concentrated on meticulously reconciling various advantages, leading to designed models that are neither simple nor practical. To address this issue, this paper presents a systematic study on existing shortcomings faced by off-the-shelf models, including lack of local fidelity, poor prediction performance over long time-steps, low scalability, and inefficiency. To systematically address the aforementioned problems, we propose an EarthFarseer, a concise framework that combines parallel local convolutions and global Fourier-based transformer architectures,  enabling  dynamically capture the local-global spatial interactions and dependencies. EarthFarseer also incorporates a multi-scale fully convolutional and Fourier architectures to efficiently and effectively capture the temporal  evolution. Our proposal demonstrates strong adaptability across various tasks and datasets, with fast convergence and better local fidelity in long time-steps predictions. Extensive experiments and visualizations over eight human society physical and natural physical datasets demonstrates the state-of-the-art performance of EarthFarseer. We release our code at https://github.com/easylearningscores/EarthFarseer.

----

## [1773] SafeAR: Safe Algorithmic Recourse by Risk-Aware Policies

**Authors**: *Haochen Wu, Shubham Sharma, Sunandita Patra, Sriram Gopalakrishnan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29522](https://doi.org/10.1609/aaai.v38i14.29522)

**Abstract**:

With the growing use of machine learning (ML) models in critical domains such as finance and healthcare, the need to offer recourse for those adversely affected by the decisions of ML models has become more important; individuals ought to be provided with recommendations on actions to take for improving their situation and thus receiving a favorable decision. Prior work on sequential algorithmic recourse---which recommends a series of changes---focuses on action feasibility and uses the proximity of feature changes to determine action costs. However, the uncertainties of feature changes and the risk of higher than average costs in recourse have not been considered. It is undesirable if a recourse could (with some probability) result in a worse situation from which recovery requires an extremely high cost. It is essential to incorporate risks when computing and evaluating recourse. We call the recourse computed with such risk considerations as Safe Algorithmic Recourse (SafeAR). The objective is to empower people to choose a recourse based on their risk tolerance. In this work, we discuss and show how existing recourse desiderata can fail to capture the risk of higher costs. We present a method to compute recourse policies that consider variability in cost and connect algorithmic recourse literature with risk-sensitive reinforcement learning. We also adopt measures "Value at Risk" and "Conditional Value at Risk" from the financial literature to summarize risk concisely. We apply our method to two real-world datasets and compare policies with different risk-aversion levels using risk measures and recourse desiderata (sparsity and proximity).

----

## [1774] SwitchTab: Switched Autoencoders Are Effective Tabular Learners

**Authors**: *Jing Wu, Suiyao Chen, Qi Zhao, Renat Sergazinov, Chen Li, Shengjie Liu, Chongchao Zhao, Tianpei Xie, Hanqing Guo, Cheng Ji, Daniel Cociorva, Hakan Brunzell*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29523](https://doi.org/10.1609/aaai.v38i14.29523)

**Abstract**:

Self-supervised representation learning methods have achieved significant success in computer vision and natural language processing (NLP), where data samples exhibit explicit spatial or semantic dependencies. However, applying these methods to tabular data is challenging due to the less pronounced dependencies among data samples. In this paper, we address this limitation by introducing SwitchTab, a novel self-supervised method specifically designed to capture latent dependencies in tabular data. SwitchTab leverages an asymmetric encoder-decoder framework to decouple mutual and salient features among data pairs, resulting in more representative embeddings. These embeddings, in turn, contribute to better decision boundaries and lead to improved results in downstream tasks. To validate the effectiveness of SwitchTab, we conduct extensive experiments across various domains involving tabular data. The results showcase superior performance in end-to-end prediction tasks with fine-tuning. Moreover, we demonstrate that pre-trained salient embeddings can be utilized as plug-and-play features to enhance the performance of various traditional classification methods (e.g., Logistic Regression, XGBoost, etc.). Lastly, we highlight the capability of SwitchTab to create explainable representations through visualization of decoupled mutual and salient features in the latent space.

----

## [1775] PORTAL: Automatic Curricula Generation for Multiagent Reinforcement Learning

**Authors**: *Jizhou Wu, Jianye Hao, Tianpei Yang, Xiaotian Hao, Yan Zheng, Weixun Wang, Matthew E. Taylor*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29524](https://doi.org/10.1609/aaai.v38i14.29524)

**Abstract**:

Despite many breakthroughs in recent years, it is still hard for MultiAgent Reinforcement Learning (MARL) algorithms to directly solve complex tasks in MultiAgent Systems (MASs) from scratch. In this work, we study how to use Automatic Curriculum Learning (ACL) to reduce the number of environmental interactions required to learn a good policy. In order to solve a difficult task, ACL methods automatically select a sequence of tasks (i.e., curricula). The idea is to obtain maximum learning progress towards the final task by continuously learning on tasks that match the current capabilities of the learners. The key question is how to measure the learning progress of the learner for better curriculum selection. We propose a novel ACL framework, PrOgRessive mulTiagent Automatic curricuLum (PORTAL), for MASs. PORTAL selects curricula according to two critera: 1) How difficult is a task, relative to the learners’ current abilities? 2) How similar is a task, relative to the final task? By learning a shared feature space between tasks, PORTAL is able to characterize different tasks based on the distribution of features and select those that are similar to the final task. Also, the shared feature space can effectively facilitate the policy transfer between curricula. Experimental results show that PORTAL can train agents to master extremely hard cooperative tasks, which can not be achieved with previous state-of-the-art MARL algorithms.

----

## [1776] FedA3I: Annotation Quality-Aware Aggregation for Federated Medical Image Segmentation against Heterogeneous Annotation Noise

**Authors**: *Nannan Wu, Zhaobin Sun, Zengqiang Yan, Li Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29525](https://doi.org/10.1609/aaai.v38i14.29525)

**Abstract**:

Federated learning (FL) has emerged as a promising paradigm for training segmentation models on decentralized medical data, owing to its privacy-preserving property. However, existing research overlooks the prevalent annotation noise encountered in real-world medical datasets, which limits the performance ceilings of FL. In this paper, we, for the first time, identify and tackle this problem. For problem formulation, we propose a contour evolution for modeling non-independent and identically distributed (Non-IID) noise across pixels within each client and then extend it to the case of multi-source data to form a heterogeneous noise model (i.e., Non-IID annotation noise across clients). For robust learning from annotations with such two-level Non-IID noise, we emphasize the importance of data quality in model aggregation, allowing high-quality clients to have a greater impact on FL. To achieve this, we propose Federated learning with Annotation quAlity-aware AggregatIon, named FedA3I, by introducing a quality factor based on client-wise noise estimation. Specifically, noise estimation at each client is accomplished through the Gaussian mixture model and then incorporated into model aggregation in a layer-wise manner to up-weight high-quality clients. Extensive experiments on two real-world medical image segmentation datasets demonstrate the superior performance of FedA3I against the state-of-the-art approaches in dealing with cross-client annotation noise. The code is available at https://github.com/wnn2000/FedAAAI.

----

## [1777] Low-Rank Kernel Tensor Learning for Incomplete Multi-View Clustering

**Authors**: *Tingting Wu, Songhe Feng, Jiazheng Yuan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29526](https://doi.org/10.1609/aaai.v38i14.29526)

**Abstract**:

Incomplete Multiple Kernel Clustering algorithms, which aim to learn a common latent representation from pre-constructed incomplete multiple kernels from the original data, followed by k-means for clustering. They have attracted intensive attention due to their high computational efficiency. However, our observation reveals that the imputation of these approaches for each kernel ignores the influence of other incomplete kernels. In light of this, we present a novel method called Low-Rank Kernel Tensor Learning for Incomplete Multiple Views Clustering (LRKT-IMVC) to address the above issue. Specifically, LRKT-IMVC first introduces the concept of kernel tensor to explore the inter-view correlations, and then the low-rank kernel tensor constraint is used to further capture the consistency information to impute missing kernel elements, thereby improving the quality of clustering. Moreover, we carefully design an alternative optimization method with promising convergence to solve the resulting optimization problem. The proposed method is compared with recent advances in experiments with different missing ratios on seven well-known datasets, demonstrating its effectiveness and the advantages of the proposed interpolation method.

----

## [1778] Test-Time Domain Adaptation by Learning Domain-Aware Batch Normalization

**Authors**: *Yanan Wu, Zhixiang Chi, Yang Wang, Konstantinos N. Plataniotis, Songhe Feng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29527](https://doi.org/10.1609/aaai.v38i14.29527)

**Abstract**:

Test-time domain adaptation aims to adapt the model trained on source domains to unseen target domains using a few unlabeled images. Emerging research has shown that the label and domain information is separately embedded in the weight matrix and batch normalization (BN) layer. Previous works normally update the whole network naively without explicitly decoupling the knowledge between label and domain. As a result, it leads to knowledge interference and defective distribution adaptation. In this work, we propose to reduce such learning interference and elevate the domain knowledge learning by only manipulating the BN layer. However, the normalization step in BN is intrinsically unstable when the statistics are re-estimated from a few samples. We find that ambiguities can be greatly reduced when only updating the two affine parameters in BN while keeping the source domain statistics. To further enhance the domain knowledge extraction from unlabeled data, we construct an auxiliary branch with label-independent self-supervised learning (SSL) to provide supervision. Moreover, we propose a bi-level optimization based on meta-learning to enforce the alignment of two learning objectives of auxiliary and main branches. The goal is to use the auxiliary branch to adapt the domain and benefit main task for subsequent inference. Our method keeps the same computational cost at inference as the auxiliary branch can be thoroughly discarded after adaptation. Extensive experiments show that our method outperforms the prior works on five WILDS real-world domain shift datasets. Our method can also be integrated with methods with label-dependent optimization to further push the performance boundary. Our code is available at https://github.com/ynanwu/MABN.

----

## [1779] H-ensemble: An Information Theoretic Approach to Reliable Few-Shot Multi-Source-Free Transfer

**Authors**: *Yanru Wu, Jianning Wang, Weida Wang, Yang Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29528](https://doi.org/10.1609/aaai.v38i14.29528)

**Abstract**:

Multi-source transfer learning is an effective solution to data scarcity by utilizing multiple source tasks for the learning of the target task. However, access to source data and model details is limited in the era of commercial models, giving rise to the setting of multi-source-free (MSF) transfer learning that aims to leverage source domain knowledge without such access. As a newly defined problem paradigm, MSF transfer learning remains largely underexplored and not clearly formulated. In this work, we adopt an information theoretic perspective on it and propose a framework named H-ensemble, which dynamically learns the optimal linear combination, or ensemble, of source models for the target task, using a generalization of maximal correlation regression. The ensemble weights are optimized by maximizing an information theoretic metric for transferability. Compared to previous works, H-ensemble is characterized by: 1) its adaptability to a novel and realistic MSF setting for few-shot target tasks, 2) theoretical reliability, 3) a lightweight structure easy to interpret and adapt. Our method is empirically validated by ablation studies, along with extensive comparative analysis with other task ensemble and transfer learning methods. We show that the H-ensemble can successfully learn the optimal task ensemble, as well as outperform prior arts.

----

## [1780] Data Poisoning to Fake a Nash Equilibria for Markov Games

**Authors**: *Young Wu, Jeremy McMahan, Xiaojin Zhu, Qiaomin Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29529](https://doi.org/10.1609/aaai.v38i14.29529)

**Abstract**:

We characterize offline data poisoning attacks on Multi-Agent Reinforcement Learning (MARL), where an attacker may change a data set in an attempt to install a (potentially fictitious) unique Markov-perfect Nash equilibrium for a two-player zero-sum Markov game. We propose the unique Nash set, namely the set of games, specified by their Q functions, with a specific joint policy being the unique Nash equilibrium. The unique Nash set is central to poisoning attacks because the attack is successful if and only if data poisoning pushes all plausible games inside it. The unique Nash set generalizes the reward polytope commonly used in inverse reinforcement learning to MARL. For zero-sum Markov games, both the inverse Nash set and the set of plausible games induced by data are polytopes in the Q function space. We exhibit a linear program to efficiently compute the optimal poisoning attack. Our work sheds light on the structure of data poisoning attacks on offline MARL, a necessary step before one can design more robust MARL algorithms.

----

## [1781] Self-Training Based Few-Shot Node Classification by Knowledge Distillation

**Authors**: *Zongqian Wu, Yujie Mo, Peng Zhou, Shangbo Yuan, Xiaofeng Zhu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29530](https://doi.org/10.1609/aaai.v38i14.29530)

**Abstract**:

Self-training based few-shot node classification (FSNC) methods have shown excellent performance in real applications, but they cannot make the full use of the information in the base set and are easily affected by the quality of pseudo-labels. To address these issues, this paper proposes a new self-training  FSNC method by involving the representation distillation and the pseudo-label distillation. Specifically, the representation distillation includes two knowledge distillation methods (i.e., the local representation distillation and the global representation distillation) to transfer the information in the base set to the novel set. The pseudo-label distillation is designed to conduct knowledge distillation on the pseudo-labels to improve their quality. 
Experimental results showed that our method achieves supreme performance, compared with state-of-the-art methods. Our code and a comprehensive theoretical version are available at https://github.com/zongqianwu/KD-FSNC.

----

## [1782] Market-GAN: Adding Control to Financial Market Data Generation with Semantic Context

**Authors**: *Haochong Xia, Shuo Sun, Xinrun Wang, Bo An*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29531](https://doi.org/10.1609/aaai.v38i14.29531)

**Abstract**:

Financial simulators play an important role in enhancing forecasting accuracy, managing risks, and fostering strategic financial decision-making. Despite the development of financial market simulation methodologies, existing frameworks often struggle with adapting to specialized simulation context. We pinpoint the challenges as i) current financial datasets do not contain context labels; ii) current techniques are not designed to generate financial data with context as control, which demands greater precision compared to other modalities; iii) the inherent difficulties in generating context-aligned, high-fidelity data given the non-stationary, noisy nature of financial data. To address these challenges, our contributions are: i) we proposed the Contextual Market Dataset with market dynamics, stock ticker, and history state as context, leveraging a market dynamics modeling method that combines linear regression and clustering to extract market dynamics; ii) we present Market-GAN, a novel architecture incorporating a Generative Adversarial Networks (GAN) for the controllable generation with context, an autoencoder for learning low-dimension features, and supervisors for knowledge transfer; iii) we introduce a two-stage training scheme to ensure that Market-GAN captures the intrinsic market distribution with multiple objectives. In the pertaining stage, with the use of the autoencoder and supervisors, we prepare the generator with a better initialization for the adversarial training stage. We propose a set of holistic evaluation metrics that consider alignment, fidelity, data usability on downstream tasks, and market facts. We evaluate Market-GAN with the Dow Jones Industrial Average data from 2000 to 2023 and showcase superior performance in comparison to 4 state-of-the-art time-series generative models.

----

## [1783] A Separation and Alignment Framework for Black-Box Domain Adaptation

**Authors**: *Mingxuan Xia, Junbo Zhao, Gengyu Lyu, Zenan Huang, Tianlei Hu, Gang Chen, Haobo Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29532](https://doi.org/10.1609/aaai.v38i14.29532)

**Abstract**:

Black-box domain adaptation (BDA) targets to learn a classifier on an unsupervised target domain while assuming only access to black-box predictors trained from unseen source data. Although a few BDA approaches have demonstrated promise by manipulating the transferred labels, they largely overlook the rich underlying structure in the target domain. To address this problem, we introduce a novel separation and alignment framework for BDA. Firstly, we locate those well-adapted samples via loss ranking and a flexible confidence-thresholding procedure. Then, we introduce a novel graph contrastive learning objective that aligns under-adapted samples to their local neighbors and well-adapted samples. Lastly, the adaptation is finally achieved by a nearest-centroid-augmented objective that exploits the clustering effect in the feature space. Extensive experiments demonstrate that our proposed method outperforms best baselines on benchmark datasets, e.g. improving the averaged per-class accuracy by 4.1% on the VisDA dataset. The source code is available at: https://github.com/MingxuanXia/SEAL.

----

## [1784] Transformer as Linear Expansion of Learngene

**Authors**: *Shiyu Xia, Miaosen Zhang, Xu Yang, Ruiming Chen, Haokun Chen, Xin Geng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29533](https://doi.org/10.1609/aaai.v38i14.29533)

**Abstract**:

We propose expanding the shared Transformer module to produce and initialize Transformers of varying depths, enabling adaptation to diverse resource constraints. Drawing an analogy to genetic expansibility, we term such module as learngene. To identify the expansion mechanism, we delve into the relationship between the layer's position and its corresponding weight value, and find that linear function appropriately approximates this relationship. Building on this insight, we present Transformer as Linear Expansion of learnGene (TLEG), a novel approach for flexibly producing and initializing Transformers of diverse depths. Specifically, to learn learngene, we firstly construct an auxiliary Transformer linearly expanded from learngene, after which we train it through employing soft distillation. Subsequently, we can produce and initialize Transformers of varying depths via linearly expanding the well-trained learngene, thereby supporting diverse downstream scenarios. Extensive experiments on ImageNet-1K demonstrate that TLEG achieves comparable or better performance in contrast to many individual models trained from scratch, while reducing around 2× training cost. When transferring to several downstream classification datasets, TLEG surpasses existing initialization methods by a large margin (e.g., +6.87% on iNat 2019 and +7.66% on CIFAR-100). Under the situation where we need to produce models of varying depths adapting for different resource constraints, TLEG achieves comparable results while reducing around 19× parameters stored to initialize these models and around 5× pre-training costs, in contrast to the pre-training and fine-tuning approach. When transferring a fixed set of parameters to initialize different models, TLEG presents better flexibility and competitive performance while reducing around 2.9× parameters stored to initialize, compared to the pre-training approach.

----

## [1785] IVP-VAE: Modeling EHR Time Series with Initial Value Problem Solvers

**Authors**: *Jingge Xiao, Leonie Basso, Wolfgang Nejdl, Niloy Ganguly, Sandipan Sikdar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29534](https://doi.org/10.1609/aaai.v38i14.29534)

**Abstract**:

Continuous-time models such as Neural ODEs and Neural Flows have shown promising results in analyzing irregularly sampled time series frequently encountered in electronic health records. Based on these models, time series are typically processed with a hybrid of an initial value problem (IVP) solver and a recurrent neural network within the variational autoencoder architecture. Sequentially solving IVPs makes such models computationally less efficient. In this paper, we propose to model time series purely with continuous processes whose state evolution can be approximated directly by IVPs. This eliminates the need for recurrent computation and enables multiple states to evolve in parallel. We further fuse the encoder and decoder with one IVP solver utilizing its invertibility, which leads to fewer parameters and faster convergence. Experiments on three real-world datasets show that the proposed method can systematically outperform its predecessors, achieve state-of-the-art results, and have significant advantages in terms of data efficiency.

----

## [1786] SHoP: A Deep Learning Framework for Solving High-Order Partial Differential Equations

**Authors**: *Tingxiong Xiao, Runzhao Yang, Yuxiao Cheng, Jinli Suo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29535](https://doi.org/10.1609/aaai.v38i14.29535)

**Abstract**:

Solving partial differential equations (PDEs) has been a fundamental problem in computational science and of wide applications for both scientific and engineering research. Due to its universal approximation property, neural network is widely used to approximate the solutions of PDEs. However, existing works are incapable of solving high-order PDEs due to insufficient calculation accuracy of higher-order derivatives, and the final network is a black box without explicit explanation. To address these issues, we propose a deep learning framework to solve high-order PDEs, named SHoP. Specifically, we derive the high-order derivative rule for neural network, to get the derivatives quickly and accurately; moreover, we expand the network into a Taylor series, providing an explicit solution for the PDEs. We conduct experimental validations four high-order PDEs with different dimensions, showing that we can solve high-order PDEs efficiently and accurately. The source code can be found at https://github.com/HarryPotterXTX/SHoP.git.

----

## [1787] Enhancing Evolving Domain Generalization through Dynamic Latent Representations

**Authors**: *Binghui Xie, Yongqiang Chen, Jiaqi Wang, Kaiwen Zhou, Bo Han, Wei Meng, James Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29536](https://doi.org/10.1609/aaai.v38i14.29536)

**Abstract**:

Domain generalization is a critical challenge for machine learning systems. Prior domain generalization methods focus on extracting domain-invariant features across several stationary domains to enable generalization to new domains. However, in non-stationary tasks where new domains evolve in an underlying continuous structure, such as time, merely extracting the invariant features is insufficient for generalization to the evolving new domains. Nevertheless, it is non-trivial to learn both evolving and invariant features within a single model due to their conflicts. To bridge this gap, we build causal models to characterize the distribution shifts concerning the two patterns, and propose to learn both dynamic and invariant features via a new framework called Mutual Information-Based Sequential Autoencoders (MISTS). MISTS adopts information theoretic constraints onto sequential autoencoders to disentangle the dynamic and invariant features, and leverage an adaptive classifier to make predictions based on both evolving and invariant information. Our experimental results on both synthetic and real-world datasets demonstrate that MISTS succeeds in capturing both evolving and invariant information, and present promising results in evolving domain generalization tasks.

----

## [1788] Trust Region Methods for Nonconvex Stochastic Optimization beyond Lipschitz Smoothness

**Authors**: *Chenghan Xie, Chenxi Li, Chuwen Zhang, Qi Deng, Dongdong Ge, Yinyu Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29537](https://doi.org/10.1609/aaai.v38i14.29537)

**Abstract**:

In many important machine learning applications, the standard assumption of having a globally Lipschitz continuous gradient may fail to hold. This paper delves into a more general (L0, L1)-smoothness setting, which gains particular significance within the realms of deep neural networks and distributionally robust optimization (DRO).  We demonstrate the significant advantage of trust region methods for stochastic nonconvex optimization under such generalized smoothness assumption.
    We show that first-order trust region methods can recover the normalized and clipped stochastic gradient as special cases and then provide a unified analysis to show their convergence to first-order stationary conditions.
    Motivated by the important application of DRO, we propose a generalized high-order smoothness condition, under which second-order trust region methods can achieve a complexity of O(epsilon(-3.5)) for convergence to  second-order stationary points. By incorporating variance reduction,  the second-order trust region method obtains an even better complexity of O(epsilon(-3)), matching the optimal bound for standard smooth optimization. To our best knowledge, this is the first work to show convergence beyond the first-order stationary condition for generalized smooth optimization.
    Preliminary experiments show that our proposed algorithms perform favorably compared with existing methods.

----

## [1789] AUC Optimization from Multiple Unlabeled Datasets

**Authors**: *Zheng Xie, Yu Liu, Ming Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29538](https://doi.org/10.1609/aaai.v38i14.29538)

**Abstract**:

Weakly supervised learning aims to make machine learning more powerful when the perfect supervision is unavailable, and has attracted much attention from researchers. Among the various scenarios of weak supervision, one of the most challenging cases is learning from multiple unlabeled (U) datasets with only a little knowledge of the class priors, or U^m learning for short. In this paper, we study the problem of building an AUC (area under ROC curve) optimal model from multiple unlabeled datasets, which maximizes the pairwise ranking ability of the classifier. We propose U^m-AUC, an AUC optimization approach that converts the U^m data into a multi-label AUC optimization problem, and can be trained efficiently. We show that the proposed U^m-AUC is effective theoretically and empirically.

----

## [1790] MoDE: A Mixture-of-Experts Model with Mutual Distillation among the Experts

**Authors**: *Zhitian Xie, Yinger Zhang, Chenyi Zhuang, Qitao Shi, Zhining Liu, Jinjie Gu, Guannan Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29539](https://doi.org/10.1609/aaai.v38i14.29539)

**Abstract**:

The application of mixture-of-experts (MoE) is gaining popularity due to its ability to improve model's performance.  In an MoE structure, the gate layer plays a significant role in distinguishing and routing input features to different experts. This enables each expert to specialize in processing their corresponding sub-tasks. However, the gate's routing mechanism also gives rise to "narrow vision": the individual MoE's expert fails to use more samples in learning the allocated subtask, which in turn limits the MoE to further improve its generalization ability. To effectively address this, we propose a method called Mixture-of-Distilled-Expert (MoDE), which applies moderate mutual distillation among experts to enable each expert to pick up more features learned by other experts and gain more accurate perceptions on their allocated sub-tasks. We conduct plenty experiments including tabular, NLP and CV datasets, which shows MoDE's effectiveness, universality and robustness. Furthermore, we develop a parallel study through innovatively constructing "expert probing", to experimentally prove why MoDE works: moderate distilling knowledge from other experts can improve each individual expert's test performances on their assigned tasks, leading to MoE's overall performance improvement.

----

## [1791] MmAP: Multi-Modal Alignment Prompt for Cross-Domain Multi-Task Learning

**Authors**: *Yi Xin, Junlong Du, Qiang Wang, Ke Yan, Shouhong Ding*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29540](https://doi.org/10.1609/aaai.v38i14.29540)

**Abstract**:

Multi-Task Learning (MTL) is designed to train multiple correlated tasks simultaneously, thereby enhancing the performance of individual tasks. Typically, a multi-task network structure consists of a shared backbone and task-specific decoders. However, the complexity of the decoders increases with the number of tasks. To tackle this challenge, we integrate the decoder-free vision-language model CLIP, which exhibits robust zero-shot generalization capability. Recently, parameter-efficient transfer learning methods have been extensively explored with CLIP for adapting to downstream tasks, where prompt tuning showcases strong potential. Nevertheless, these methods solely fine-tune a single modality (text or visual), disrupting the modality structure of CLIP. In this paper, we first propose Multi-modal Alignment Prompt (MmAP) for CLIP, which aligns text and visual modalities during fine-tuning process. Building upon MmAP, we develop an innovative multi-task prompt learning framework. On the one hand, to maximize the complementarity of tasks with high similarity, we utilize a gradient-driven task grouping method that partitions tasks into several disjoint groups and assign a group-shared MmAP to each group. On the other hand, to preserve the unique characteristics of each task, we assign an task-specific MmAP to each task. Comprehensive experiments on two large multi-task learning datasets demonstrate that our method achieves significant performance improvements compared to full fine-tuning while only utilizing approximately ~ 0.09% of trainable parameters.

----

## [1792] VMT-Adapter: Parameter-Efficient Transfer Learning for Multi-Task Dense Scene Understanding

**Authors**: *Yi Xin, Junlong Du, Qiang Wang, Zhiwen Lin, Ke Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29541](https://doi.org/10.1609/aaai.v38i14.29541)

**Abstract**:

Large-scale pre-trained models have achieved remarkable success in various computer vision tasks. A standard approach to leverage these models is to fine-tune all model parameters for downstream tasks, which poses challenges in terms of computational and storage costs. Recently, inspired by Natural Language Processing (NLP), parameter-efficient transfer learning has been successfully applied to vision tasks. However, most existing techniques primarily focus on single-task adaptation, and despite limited research on multi-task adaptation, these methods often exhibit suboptimal training/inference efficiency. In this paper, we first propose an once-for-all Vision Multi-Task Adapter (VMT-Adapter), which strikes approximately O(1) training and inference efficiency w.r.t task number. Concretely, VMT-Adapter shares the knowledge from multiple tasks to enhance cross-task interaction while preserves task-specific knowledge via independent knowledge extraction modules. Notably, since task-specific modules require few parameters, VMT-Adapter can handle an arbitrary number of tasks with a negligible increase of trainable parameters. We also propose VMT-Adapter-Lite, which further reduces the trainable parameters by learning shared parameters between down- and up-projections. Extensive experiments on four dense scene understanding tasks demonstrate the superiority of VMT-Adapter(-Lite), achieving a 3.96% (1.34%) relative improvement compared to single-task full fine-tuning, while utilizing merely ～1% (0.36%) trainable parameters of the pre-trained model.

----

## [1793] BiPFT: Binary Pre-trained Foundation Transformer with Low-Rank Estimation of Binarization Residual Polynomials

**Authors**: *Xingrun Xing, Li Du, Xinyuan Wang, Xianlin Zeng, Yequan Wang, Zheng Zhang, Jiajun Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29542](https://doi.org/10.1609/aaai.v38i14.29542)

**Abstract**:

Pretrained foundation models offer substantial benefits for a wide range of downstream tasks, which can be one of the most potential techniques to access artificial general intelligence. However, scaling up foundation transformers for maximal task-agnostic knowledge has brought about computational challenges, especially on resource-limited devices such as mobiles. This work proposes the first Binary Pretrained Foundation Transformer (BiPFT) for natural language understanding (NLU) tasks, which remarkably saves 56 times operations and 28 times memory. In contrast to previous task-specific binary transformers, BiPFT exhibits a substantial enhancement in the learning capabilities of binary neural networks (BNNs), promoting BNNs into the era of pre-training. Benefiting from extensive pretraining data, we further propose a data-driven binarization method. Specifically, we first analyze the binarization error in self-attention operations and derive the polynomials of binarization error. To simulate full-precision self-attention, we define binarization error as binarization residual polynomials, and then introduce low-rank estimators to model these polynomials. Extensive experiments validate the effectiveness of BiPFTs, surpassing task-specific baseline by 15.4% average performance on the GLUE benchmark. BiPFT also demonstrates improved robustness to hyperparameter changes, improved optimization efficiency, and reduced reliance on downstream distillation, which consequently generalize on various NLU tasks and simplify the downstream pipeline of BNNs. Our code and pretrained models are publicly available at https://github.com/Xingrun-Xing/BiPFT.

----

## [1794] DePRL: Achieving Linear Convergence Speedup in Personalized Decentralized Learning with Shared Representations

**Authors**: *Guojun Xiong, Gang Yan, Shiqiang Wang, Jian Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29543](https://doi.org/10.1609/aaai.v38i14.29543)

**Abstract**:

Decentralized learning has emerged as an alternative method to the popular parameter-server framework which suffers from high communication burden, single-point failure and scalability issues due to the need of a central server.  However, most existing works focus on a single shared model for all workers regardless of the data heterogeneity problem, rendering the resulting model performing poorly on individual workers.  In this work, we propose a novel personalized decentralized learning algorithm named DePRL via shared representations.  Our algorithm  relies on ideas from representation learning theory to learn a low-dimensional global representation collaboratively among all workers in a fully decentralized manner, as well as a user-specific low-dimensional local head leading to a personalized solution for each worker.  We show that DePRL achieves, for the first time, a provable \textit{linear speedup for convergence} with general non-linear representations (i.e., the convergence rate is improved linearly with respect to the number of workers). Experimental results support our theoretical findings showing the superiority of our method in data heterogeneous environments.

----

## [1795] TEILP: Time Prediction over Knowledge Graphs via Logical Reasoning

**Authors**: *Siheng Xiong, Yuan Yang, Ali Payani, James Clayton Kerce, Faramarz Fekri*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29544](https://doi.org/10.1609/aaai.v38i14.29544)

**Abstract**:

Conventional embedding-based models approach event time prediction in temporal knowledge graphs (TKGs) as a ranking problem. However, they often fall short in capturing essential temporal relationships such as order and distance. In this paper, we propose TEILP, a logical reasoning framework that naturaly integrates such temporal elements into knowledge graph predictions. We first convert TKGs into a temporal event knowledge graph (TEKG) which has a more explicit representation of time in term of nodes of the graph. The TEKG equips us to develop a differentiable random walk approach to time prediction. Finally, we introduce conditional probability density functions, associated with the logical rules involving the query interval, using which we arrive at the time prediction. We compare TEILP with state-of-the-art methods on five benchmark datasets. We show that our model achieves a significant improvement over baselines while providing interpretable explanations. In particular, we consider several scenarios where training samples are limited, event types are imbalanced, and forecasting the time of future events based on only past events is desired. In all these cases, TEILP outperforms state-of-the-art methods in terms of robustness.

----

## [1796] FairWASP: Fast and Optimal Fair Wasserstein Pre-processing

**Authors**: *Zikai Xiong, Niccolò Dalmasso, Alan Mishler, Vamsi K. Potluru, Tucker Balch, Manuela Veloso*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29545](https://doi.org/10.1609/aaai.v38i14.29545)

**Abstract**:

Recent years have seen a surge of machine learning approaches aimed at reducing disparities in model outputs across different subgroups. In many settings, training data may be used in multiple downstream applications by different users, which means it may be most effective to intervene on the training data itself. In this work, we present FairWASP, a novel pre-processing approach designed to reduce disparities in classification datasets without modifying the original data. FairWASP returns sample-level weights such that the reweighted dataset minimizes the Wasserstein distance to the original dataset while satisfying (an empirical version of) demographic parity, a popular fairness criterion. We show theoretically that integer weights are optimal, which means our method can be equivalently understood as duplicating or eliminating samples. FairWASP can therefore be used to construct datasets which can be fed into any classification method, not just methods which accept sample weights. Our work is based on reformulating the pre-processing task as a large-scale mixed-integer program (MIP), for which we propose a highly efficient algorithm based on the cutting plane method. Experiments demonstrate that our proposed optimization algorithm significantly outperforms state-of-the-art commercial solvers in solving both the MIP and its linear program relaxation. Further experiments highlight the competitive performance of FairWASP in reducing disparities while preserving accuracy in downstream classification settings.

----

## [1797] Reliable Conflictive Multi-View Learning

**Authors**: *Cai Xu, Jiajun Si, Ziyu Guan, Wei Zhao, Yue Wu, Xiyue Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29546](https://doi.org/10.1609/aaai.v38i14.29546)

**Abstract**:

Multi-view learning aims to combine multiple features to achieve more comprehensive descriptions of data. Most previous works assume that multiple views are strictly aligned. However, real-world multi-view data may contain low-quality conflictive instances, which show conflictive information in different views. Previous methods for this problem mainly focus on eliminating the conflictive data instances by removing them or replacing conflictive views. Nevertheless, real-world applications usually require making decisions for conflictive instances rather than only eliminating them. To solve this, we point out a new Reliable Conflictive Multi-view Learning (RCML) problem, which requires the model to provide decision results and attached reliabilities for conflictive multi-view data. We develop an Evidential Conflictive Multi-view Learning (ECML) method for this problem. ECML first learns view-specific evidence, which could be termed as the amount of support to each category collected from data. Then, we can construct view-specific opinions consisting of decision results and reliability. In the multi-view fusion stage, we propose a conflictive opinion aggregation strategy and theoretically prove this strategy can exactly model the relation of multi-view common and view-specific reliabilities. Experiments performed on 6 datasets verify the effectiveness of ECML. The code is released at https://github.com/jiajunsi/RCML.

----

## [1798] A Label Disambiguation-Based Multimodal Massive Multiple Instance Learning Approach for Immune Repertoire Classification

**Authors**: *Fan Xu, Yu Zhao, Bingzhe Wu, Yueshan Huang, Qin Ren, Yang Xiao, Bing He, Jie Zheng, Jianhua Yao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29547](https://doi.org/10.1609/aaai.v38i14.29547)

**Abstract**:

One individual human’s immune repertoire consists of a huge set of adaptive immune receptors at a certain time point, representing the individual's adaptive immune state. Immune repertoire classification and associated receptor identification have the potential to make a transformative contribution to the development of novel vaccines and therapies. The vast number of instances and exceedingly low witness rate pose a great challenge to the immune repertoire classification, which can be formulated as a Massive Multiple Instance Learning (MMIL) problem. Traditional MIL methods, at both bag-level and instance-level, confront the issues of substantial computational burden or supervision ambiguity when handling massive instances. To address these issues, we propose a novel label disambiguation-based multimodal massive multiple instance learning approach (LaDM³IL) for immune repertoire classification. LaDM³IL adapts the instance-level MIL paradigm to deal with the issue of high computational cost and employs a specially-designed label disambiguation module for label correction, mitigating the impact of misleading supervision. To achieve a more comprehensive representation of each receptor, LaDM³IL leverages a multimodal fusion module with gating-based attention and tensor-fusion to integrate the information from gene segments and amino acid (AA) sequences of each immune receptor. Extensive experiments on the Cytomegalovirus (CMV) and Cancer datasets demonstrate the superior performance of the proposed LaDM³IL for both immune repertoire classification and associated receptor identification tasks. The code is publicly available at https://github.com/Josie-xufan/LaDM3IL.

----

## [1799] Deep Variational Incomplete Multi-View Clustering: Exploring Shared Clustering Structures

**Authors**: *Gehui Xu, Jie Wen, Chengliang Liu, Bing Hu, Yicheng Liu, Lunke Fei, Wei Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29548](https://doi.org/10.1609/aaai.v38i14.29548)

**Abstract**:

Incomplete multi-view clustering (IMVC) aims to reveal shared clustering structures within multi-view data, where only partial views of the samples are available. Existing IMVC methods primarily suffer from two issues: 1) Imputation-based methods inevitably introduce inaccurate imputations, which in turn degrade clustering performance; 2) Imputation-free methods are susceptible to unbalanced information among views and fail to fully exploit shared information. To address these issues, we propose a novel method based on variational autoencoders. Specifically, we adopt multiple view-specific encoders to extract information from each view and utilize the Product-of-Experts approach to efficiently aggregate information to obtain the common representation. To enhance the shared information in the common representation, we introduce a coherence objective to mitigate the influence of information imbalance. By incorporating the Mixture-of-Gaussians prior information into the latent representation, our proposed method is able to learn the common representation with clustering-friendly structures. Extensive experiments on four datasets show that our method achieves competitive clustering performance compared with state-of-the-art methods.

----



[Go to the previous page](AAAI-2024-list08.md)

[Go to the next page](AAAI-2024-list10.md)

[Go to the catalog section](README.md)