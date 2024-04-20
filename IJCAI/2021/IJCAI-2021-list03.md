## [400] Multi-Agent Reinforcement Learning for Automated Peer-to-Peer Energy Trading in Double-Side Auction Market

**Authors**: *Dawei Qiu, Jianhong Wang, Junkai Wang, Goran Strbac*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/401](https://doi.org/10.24963/ijcai.2021/401)

**Abstract**:

With increasing prosumers employed with distributed energy resources (DER), advanced energy management has become increasingly important. To this end, integrating demand-side DER into electricity market is a trend for future smart grids. The double-side auction (DA) market is viewed as a promising peer-to-peer (P2P) energy trading mechanism that enables interactions among prosumers in a distributed manner. To achieve the maximum profit in a dynamic electricity market, prosumers act as price makers to simultaneously optimize their operations and trading strategies. However, the traditional DA market is difficult to be explicitly modelled due to its complex clearing algorithm and the stochastic bidding behaviors of the participants. For this reason, in this paper we model this task as a multi-agent reinforcement learning (MARL) problem and propose an algorithm called DA-MADDPG that is modified based on MADDPG by abstracting the other agents’ observations and actions through the DA market public information for each agent’s critic. The experiments show that 1) prosumers obtain more economic benefits in P2P energy trading w.r.t. the conventional electricity market independently trading with the utility company; and 2) DA-MADDPG performs better than the traditional Zero Intelligence (ZI) strategy and the other MARL algorithms, e.g., IQL, IDDPG, IPPO and MADDPG.

----

## [401] Source-free Domain Adaptation via Avatar Prototype Generation and Adaptation

**Authors**: *Zhen Qiu, Yifan Zhang, Hongbin Lin, Shuaicheng Niu, Yanxia Liu, Qing Du, Mingkui Tan*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/402](https://doi.org/10.24963/ijcai.2021/402)

**Abstract**:

We study a practical domain adaptation task, called source-free unsupervised domain adaptation (UDA) problem, in which we cannot access source domain data due to data privacy issues but only a pre-trained source model and unlabeled target data are available.  This task, however, is very difficult due to one key challenge: the lack of source data and target domain labels makes model adaptation very challenging. To address this, we propose to mine the hidden knowledge in the source model and exploit it to generate source avatar prototypes (i.e. representative features for each source class) as well as target pseudo labels for domain alignment. To this end, we propose a Contrastive Prototype Generation and Adaptation (CPGA) method. Specifically, CPGA consists of two stages: (1) prototype generation: by exploring the classification boundary information of the source model, we train a prototype generator to generate avatar prototypes via contrastive learning. (2) prototype adaptation: based on the generated source prototypes and target pseudo labels, we develop a new robust contrastive prototype adaptation strategy to align each pseudo-labeled target data to the corresponding source prototypes. Extensive experiments on three UDA benchmark datasets demonstrate the effectiveness and superiority of the proposed method.

----

## [402] Exact Acceleration of K-Means++ and K-Means||

**Authors**: *Edward Raff*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/403](https://doi.org/10.24963/ijcai.2021/403)

**Abstract**:

K-Means++ and its distributed variant K-Means|| have become de facto tools for selecting the initial seeds of K-means. While alternatives have been developed, the effectiveness, ease of implementation,and theoretical grounding of the K-means++ and || methods have made them difficult to "best" from a holistic perspective. We focus on using triangle inequality based pruning methods to accelerate both of these algorithms to yield comparable or better run-time without sacrificing any of the benefits of these approaches. For both algorithms we are able to reduce distance computations by over 500×. For K-means++ this results in up to a 17×speedup in run-time and a551×speedup for K-means||. We achieve this with simple, but carefully chosen, modifications to known techniques which makes it easy to integrate our approach into existing implementations of these algorithms.

----

## [403] Stochastic Shortest Path with Adversarially Changing Costs

**Authors**: *Aviv Rosenberg, Yishay Mansour*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/404](https://doi.org/10.24963/ijcai.2021/404)

**Abstract**:

Stochastic shortest path (SSP) is a well-known problem in planning and control, in which an agent has to reach a goal state in minimum total expected cost. In this paper we present the adversarial SSP model that also accounts for adversarial changes in the costs over time, while the underlying transition function remains unchanged. Formally, an agent interacts with an SSP environment for K episodes, the cost function changes arbitrarily between episodes, and the transitions are unknown to the agent. We develop the first algorithms for adversarial SSPs and prove high probability regret bounds of square-root K assuming all costs are strictly positive, and sub-linear regret in the general case. We are the first to consider this natural setting of adversarial SSP and obtain sub-linear regret for it.

----

## [404] Physics-aware Spatiotemporal Modules with Auxiliary Tasks for Meta-Learning

**Authors**: *Sungyong Seo, Chuizheng Meng, Sirisha Rambhatla, Yan Liu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/405](https://doi.org/10.24963/ijcai.2021/405)

**Abstract**:

Modeling the dynamics of real-world physical systems is critical for spatiotemporal prediction tasks, but challenging when data is limited. The scarcity of real-world data and the difficulty in reproducing the data distribution hinder directly applying meta-learning techniques. Although the knowledge of governing partial differential equations (PDE) of the data can be helpful for the fast adaptation to few observations, it is mostly infeasible to exactly find the equation for observations in real-world physical systems. In this work, we propose a framework, physics-aware meta-learning with auxiliary tasks, whose spatial modules incorporate PDE-independent knowledge and temporal modules utilize the generalized features from the spatial modules to be adapted to the limited data, respectively. The framework is inspired by a local conservation law expressed mathematically as a continuity equation and does not require the exact form of governing equation to model the spatiotemporal observations. The proposed method mitigates the need for a large number of real-world tasks for meta-learning by leveraging spatial information in simulated data to meta-initialize the spatial modules. We apply the proposed framework to both synthetic and real-world spatiotemporal prediction tasks and demonstrate its superior performance with limited observations.

----

## [405] Don't Do What Doesn't Matter: Intrinsic Motivation with Action Usefulness

**Authors**: *Mathieu Seurin, Florian Strub, Philippe Preux, Olivier Pietquin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/406](https://doi.org/10.24963/ijcai.2021/406)

**Abstract**:

Sparse rewards are double-edged training signals in reinforcement learning: 
easy to design but hard to optimize. Intrinsic motivation guidances have thus been developed toward alleviating the resulting exploration problem. They usually incentivize agents to look for new states through novelty signals. Yet, such methods encourage exhaustive exploration of the state space rather than focusing on the environment's salient interaction opportunities. We propose a new exploration method, called Don't Do What Doesn't Matter (DoWhaM), shifting the emphasis from state novelty to state with relevant actions. While most actions consistently change the state when used, e.g. moving the agent, some actions are only effective in specific states, e.g., opening a door, grabbing an object. DoWhaM detects and rewards actions that seldom affect the environment.
We evaluate DoWhaM on the procedurally-generated environment MiniGrid against state-of-the-art methods. Experiments consistently show that DoWhaM greatly reduces sample complexity, installing the new state-of-the-art in MiniGrid.

----

## [406] Towards Robust Model Reuse in the Presence of Latent Domains

**Authors**: *Jie-Jing Shao, Zhanzhan Cheng, Yu-Feng Li, Shiliang Pu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/407](https://doi.org/10.24963/ijcai.2021/407)

**Abstract**:

Model reuse tries to adapt well pre-trained models to a new target task, without access of raw data. It attracts much attention since it reduces the learning resources. Previous model reuse studies typically operate in a single-domain scenario, i.e., the target samples arise from one single domain. However, in practice the target samples often arise from multiple latent or unknown domains, e.g., the images for cars may arise from latent domains such as photo, line drawing, cartoon, etc. The methods based on single-domain may no longer be feasible for multiple latent domains and may sometimes even lead to performance degeneration. To address the above issue, in this paper we propose the MRL (Model Reuse for multiple Latent domains) method. Both domain characteristics and pre-trained models are considered for the exploration of instances in the target task. Theoretically, the overall considerations are packed in a bi-level optimization framework with a reliable generalization. Moreover, through an ensemble of multiple models, the model robustness is improved with a theoretical guarantee. Empirical results on diverse real-world data sets clearly validate the effectiveness of proposed algorithms.

----

## [407] Regularizing Variational Autoencoder with Diversity and Uncertainty Awareness

**Authors**: *Dazhong Shen, Chuan Qin, Chao Wang, Hengshu Zhu, Enhong Chen, Hui Xiong*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/408](https://doi.org/10.24963/ijcai.2021/408)

**Abstract**:

As one of the most popular generative models, Variational Autoencoder (VAE) approximates the posterior of latent variables based on amortized variational inference. However, when the decoder network is sufficiently expressive, VAE may lead to posterior collapse; that is, uninformative latent representations may be learned. To this end, in this paper, we propose an alternative model, DU-VAE, for learning a more Diverse and less Uncertain latent space, and thus the representation can be learned in a meaningful and compact manner. Specifically, we first theoretically demonstrate that it will result in better latent space with high diversity and low uncertainty awareness by controlling the distribution of posteriorâ€™s parameters across the whole data accordingly. Then, without the introduction of new loss terms or modifying training strategies, we propose to exploit Dropout on the variances and Batch-Normalization on the means simultaneously to regularize their distributions implicitly. Furthermore, to evaluate the generalization effect, we also exploit DU-VAE for inverse autoregressive flow based-VAE (VAE-IAF) empirically. Finally, extensive experiments on three benchmark datasets clearly show that our approach can outperform state-of-the-art baselines on both likelihood estimation and underlying classification tasks.

----

## [408] Interpretable Compositional Convolutional Neural Networks

**Authors**: *Wen Shen, Zhihua Wei, Shikun Huang, Binbin Zhang, Jiaqi Fan, Ping Zhao, Quanshi Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/409](https://doi.org/10.24963/ijcai.2021/409)

**Abstract**:

This paper proposes a method to modify a traditional convolutional neural network (CNN) into an interpretable compositional CNN, in order to learn filters that encode meaningful visual patterns in intermediate convolutional layers. In a compositional CNN, each filter is supposed to consistently represent a specific compositional object part or image region with a clear meaning. The compositional CNN learns from image labels for classification without any annotations of parts or regions for supervision. Our method can be broadly applied to different types of CNNs. Experiments have demonstrated the effectiveness of our method. The code will be released when the paper is accepted.

----

## [409] Unsupervised Progressive Learning and the STAM Architecture

**Authors**: *James Seale Smith, Cameron E. Taylor, Seth Baer, Constantine Dovrolis*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/410](https://doi.org/10.24963/ijcai.2021/410)

**Abstract**:

We first pose the Unsupervised Progressive Learning (UPL) problem: an online
representation learning problem in which the learner observes a non-stationary
and unlabeled data stream, learning a growing number of features that persist
over time even though the data is not stored or replayed. To solve the UPL
problem we propose the Self-Taught Associative Memory (STAM) architecture.
Layered hierarchies of STAM modules learn based on a combination of online
clustering, novelty detection, forgetting outliers, and storing only prototypical
features rather than specific examples. We evaluate STAM representations using
clustering and classification tasks. While there are no existing learning scenarios
that are directly comparable to UPL, we compare the STAM architecture with two
recent continual learning models, Memory Aware Synapses (MAS) and Gradient
Episodic Memories (GEM), after adapting them in the UPL setting.

----

## [410] Online Risk-Averse Submodular Maximization

**Authors**: *Tasuku Soma, Yuichi Yoshida*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/411](https://doi.org/10.24963/ijcai.2021/411)

**Abstract**:

We present a polynomial-time online algorithm for maximizing the conditional value at risk (CVaR) of a monotone stochastic submodular function. Given T i.i.d. samples from an underlying distribution arriving online, our algorithm produces a sequence of solutions that converges to a (1−1/e)-approximate solution with a convergence rate of O(T −1/4 ) for monotone continuous DR-submodular functions. Compared with previous offline algorithms, which require Ω(T) space, our online algorithm only requires O( √ T) space. We extend our on- line algorithm to portfolio optimization for mono- tone submodular set functions under a matroid constraint. Experiments conducted on real-world datasets demonstrate that our algorithm can rapidly achieve CVaRs that are comparable to those obtained by existing offline algorithms.

----

## [411] Positive-Unlabeled Learning from Imbalanced Data

**Authors**: *Guangxin Su, Weitong Chen, Miao Xu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/412](https://doi.org/10.24963/ijcai.2021/412)

**Abstract**:

Positive-unlabeled (PU) learning deals with the binary classification problem when only positive (P) and unlabeled (U) data are available, without negative (N) data. Existing PU methods perform well on the balanced dataset. However, in real applications such as financial fraud detection or medical diagnosis, data are always imbalanced. It remains unclear whether existing PU methods can perform well on imbalanced data. In this paper, we explore this problem and propose a general learning objective for PU learning targeting specially at imbalanced data. By this general learning objective, state-of-the-art PU methods based on optimizing a consistent risk can be adapted to conquer the imbalance. We theoretically show that in expectation, optimizing our learning objective is equivalent to learning a classifier on the oversampled balanced data with both P and N data available, and further provide an estimation error bound. Finally, experimental results validate the effectiveness of our proposal compared to state-of-the-art PU methods.

----

## [412] Neural Architecture Search of SPD Manifold Networks

**Authors**: *Rhea Sanjay Sukthanker, Zhiwu Huang, Suryansh Kumar, Erik Goron Endsjo, Yan Wu, Luc Van Gool*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/413](https://doi.org/10.24963/ijcai.2021/413)

**Abstract**:

In this paper, we propose a new neural architecture search (NAS) problem of Symmetric Positive Definite (SPD) manifold networks, aiming to automate the design of SPD neural architectures. To address this problem, we first introduce a geometrically rich and diverse SPD neural architecture search space for an efficient SPD cell design. Further, we model our new NAS problem with a one-shot training process of a single supernet. Based on the supernet modeling, we exploit a differentiable NAS algorithm on our relaxed continuous search space for SPD neural architecture search. Statistical evaluation of our method on drone, action, and emotion recognition tasks mostly provides better results than the state-of-the-art SPD networks and traditional NAS algorithms. Empirical results show that our algorithm excels in discovering better performing SPD network design and provides models that are more than three times lighter than searched by the state-of-the-art NAS algorithms.

----

## [413] TE-ESN: Time Encoding Echo State Network for Prediction Based on Irregularly Sampled Time Series Data

**Authors**: *Chenxi Sun, Shenda Hong, Moxian Song, Yen-hsiu Chou, Yongyue Sun, Derun Cai, Hongyan Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/414](https://doi.org/10.24963/ijcai.2021/414)

**Abstract**:

Prediction based on Irregularly Sampled Time Series (ISTS) is of wide concern in real-world applications. For more accurate prediction, methods had better grasp more data characteristics. Different from ordinary time series, ISTS is characterized by irregular time intervals of intra-series and different sampling rates of inter-series. However, existing methods have suboptimal predictions due to artificially introducing new dependencies in a time series and biasedly learning relations among time series when modeling these two characteristics. In this work, we propose a novel Time Encoding (TE) mechanism. TE can embed the time information as time vectors in the complex domain. It has the properties of absolute distance and relative distance under different sampling rates, which helps to represent two irregularities. Meanwhile, we create a new model named Time Encoding Echo State Network (TE-ESN). It is the first ESNs-based model that can process ISTS data. Besides, TE-ESN incorporates long short-term memories and series fusion to grasp horizontal and vertical relations. Experiments on one chaos system and three real-world datasets show that TE-ESN performs better than all baselines and has better reservoir property.

----

## [414] MFNP: A Meta-optimized Model for Few-shot Next POI Recommendation

**Authors**: *Huimin Sun, Jiajie Xu, Kai Zheng, Pengpeng Zhao, Pingfu Chao, Xiaofang Zhou*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/415](https://doi.org/10.24963/ijcai.2021/415)

**Abstract**:

Next Point-of-Interest (POI) recommendation is of great value for location-based services. Existing solutions mainly rely on extensive observed data and are brittle to users with few interactions. Unfortunately, the problem of few-shot next POI recommendation has not been well studied yet. In this paper, we propose a novel meta-optimized model MFNP, which can rapidly adapt to users with few check-in records. Towards the cold-start problem, it seamlessly integrates carefully designed user-specific and region-specific tasks in meta-learning, such that region-aware user preferences can be captured via a rational fusion of region-independent personal preferences and region-dependent crowd preferences. In modelling region-dependent crowd preferences, a cluster-based adaptive network is adopted to capture shared preferences from similar users for knowledge transfer. Experimental results on two real-world datasets show that our model outperforms the state-of-the-art methods on next POI recommendation for cold-start users.

----

## [415] Towards Reducing Biases in Combining Multiple Experts Online

**Authors**: *Yi Sun, Iván Ramírez Díaz, Alfredo Cuesta-Infante, Kalyan Veeramachaneni*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/416](https://doi.org/10.24963/ijcai.2021/416)

**Abstract**:

In many real life situations, including job and loan applications, gatekeepers must make justified and fair real-time decisions about a person’s fitness for a particular opportunity. In this paper, we aim to accomplish approximate group fairness in an online stochastic decision-making process, where the fairness metric we consider is equalized odds. Our work follows from the classical learning-from-experts scheme, assuming a finite set of classifiers (human experts, rules, options, etc) that cannot be modified. We run separate instances of the algorithm for each label class as well as sensitive groups, where the probability of choosing each instance is optimized for both fairness and regret. Our theoretical results show that approximately equalized odds can be achieved without sacrificing much regret. We also demonstrate the performance of the algorithm on real data sets commonly used by the fairness community.

----

## [416] Predicting Traffic Congestion Evolution: A Deep Meta Learning Approach

**Authors**: *Yidan Sun, Guiyuan Jiang, Siew Kei Lam, Peilan He*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/417](https://doi.org/10.24963/ijcai.2021/417)

**Abstract**:

Many efforts are devoted to predicting congestion evolution using propagation patterns that are mined from historical traffic data.
However, the prediction quality is limited to the intrinsic properties that are present in the mined patterns. In addition, these mined patterns frequently fail to sufficiently capture many realistic characteristics of true congestion evolution (e.g., asymmetric transitivity, local proximity). In this paper, we propose a representation learning framework to characterize and predict congestion evolution between any pair of road segments (connected via single or multiple paths). Specifically, we build dynamic attributed networks (DAN) to incorporate both dynamic and static impact factors while preserving dynamic topological structures. We propose a Deep Meta Learning Model (DMLM) for learning representations of road segments which support accurate prediction of congestion evolution. DMLM relies on matrix factorization techniques and meta-LSTM modules to exploit temporal correlations at multiple scales, and employ meta-Attention modules to merge heterogeneous features while learning the time-varying impacts of both dynamic and static features. Compared to all state-of-the-art methods, our framework achieves significantly better prediction performance on two congestion evolution behaviors (propagation and decay) when evaluated using real-world dataset.

----

## [417] Hyperspectral Band Selection via Spatial-Spectral Weighted Region-wise Multiple Graph Fusion-Based Spectral Clustering

**Authors**: *Chang Tang, Xinwang Liu, En Zhu, Lizhe Wang, Albert Y. Zomaya*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/418](https://doi.org/10.24963/ijcai.2021/418)

**Abstract**:

In this paper, we propose a hyperspectral band selection method via spatial-spectral weighted region-wise multiple graph fusion-based spectral clustering, referred to as RMGF briefly. Considering that different objects have different reflection characteristics, we use a superpixel segmentation algorithm to segment the first principal component of original hyperspectral image cube into homogeneous regions. For each superpixel, we construct a corresponding similarity graph to reflect the similarity between band pairs. Then, a multiple graph diffusion strategy with theoretical convergence guarantee is designed to learn a unified graph for partitioning the whole hyperspectral cube into several subcubes via spectral clustering. During the graph diffusion process, the spatial and spectral information of each superpixel are embedded to make spatial/spectral similar superpixels contribute more to each other. Finally, the band containing minimum noise in each subcube is selected to represent the whole subcube. Extensive experiments are conducted on three public datasets to validate the superiority of the proposed method when compared with other state-of-the-art ones.

----

## [418] Self-supervised Network Evolution for Few-shot Classification

**Authors**: *Xuwen Tang, Zhu Teng, Baopeng Zhang, Jianping Fan*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/419](https://doi.org/10.24963/ijcai.2021/419)

**Abstract**:

Few-shot classification aims to recognize new classes by learning reliable models from very few available samples. It could be very challenging when there is no intersection between the alreadyknown classes (base set) and the novel set (new classes). To alleviate this problem, we propose to evolve the network (for the base set) via label propagation and self-supervision to shrink the distribution difference between the base set and the novel set. Our network evolution approach transfers the latent distribution from the already-known classes to the unknown (novel) classes by: (a) label propagation of the novel/new classes (novel set); and (b) design of dual-task to exploit a discriminative representation to effectively diminish the overfitting on the base set and enhance the generalization ability on the novel set. We conduct comprehensive experiments to examine our network evolution approach against numerous state-of-the-art ones, especially in a higher way setup and cross-dataset scenarios. Notably, our approach outperforms the second best state-of-the-art method by a large margin of 3.25% for one-shot evaluation over miniImageNet.

----

## [419] Dual Active Learning for Both Model and Data Selection

**Authors**: *Ying-Peng Tang, Sheng-Jun Huang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/420](https://doi.org/10.24963/ijcai.2021/420)

**Abstract**:

To learn an effective model with less training examples, existing active learning methods typically assume that there is a given target model, and try to fit it by selecting the most informative examples. However, it is less likely to determine the best target model in prior, and thus may get suboptimal performance even if the data is perfectly selected. To tackle with this practical challenge, this paper proposes a novel framework of dual active learning (DUAL) to simultaneously perform model search and data selection. Specifically, an effective method with truncated importance sampling is proposed for Combined Algorithm Selection and Hyperparameter optimization (CASH), which mitigates the model evaluation bias on the labeled data. Further, we propose an active query strategy to label the most valuable examples. The strategy on one hand favors discriminative data to help CASH search the best model, and on the other hand prefers informative examples to accelerate the convergence of winner models. Extensive experiments are conducted on 12 openML datasets. The results demonstrate the proposed method can effectively learn a superior model with less labeled examples.

----

## [420] Compositional Neural Logic Programming

**Authors**: *Son N. Tran*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/421](https://doi.org/10.24963/ijcai.2021/421)

**Abstract**:

This paper introduces Compositional Neural Logic Programming (CNLP), a framework that integrates neural networks and logic programming for symbolic and sub-symbolic reasoning. We adopt the idea of compositional neural networks to represent first-order logic predicates and rules. A voting backward-forward chaining algorithm is proposed for inference with both symbolic and sub-symbolic variables in an argument-retrieval style. The framework is highly flexible in that it can be constructed incrementally with new knowledge, and it also supports batch reasoning in certain cases. In the experiments, we demonstrate the advantages of CNLP in discriminative tasks and generative tasks.

----

## [421] Sensitivity Direction Learning with Neural Networks Using Domain Knowledge as Soft Shape Constraints

**Authors**: *Kazuyuki Wakasugi*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/422](https://doi.org/10.24963/ijcai.2021/422)

**Abstract**:

If domain knowledge can be integrated as an appropriate constraint, it is highly possible that the generalization performance of a neural network model can be improved. We propose Sensitivity Direction Learning (SDL) for learning about the neural network model with user-specified relationships (e.g., monotonicity, convexity) between each input feature and the output of the model by imposing soft shape constraints which represent domain knowledge. To impose soft shape constraints, SDL uses a novel penalty function, Sensitivity Direction Error (SDE) function, which returns the squared error between coefficients of the approximation curve for each Individual Conditional Expectation plot and coefficient constraints which represent domain knowledge. The effectiveness of our concept was verified by simple experiments. Similar to those such as L2 regularization and dropout, SDL and SDE can be used without changing neural network architecture. We believe our algorithm can be a strong candidate for neural network users who want to incorporate domain knowledge.

----

## [422] Learning from Complementary Labels via Partial-Output Consistency Regularization

**Authors**: *Deng-Bao Wang, Lei Feng, Min-Ling Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/423](https://doi.org/10.24963/ijcai.2021/423)

**Abstract**:

In complementary-label learning (CLL), a multi-class classifier is learned from training instances each associated with complementary labels, which specify the classes that the instance does not belong to. Previous studies focus on unbiased risk estimator or surrogate loss while neglect the importance of regularization in training phase. In this paper, we give the first attempt to leverage regularization techniques for CLL. By decoupling a label vector into complementary labels and partial unknown labels, we simultaneously inhibit the outputs of complementary labels with a complementary loss and penalize the sensitivity of the classifier on the partial outputs of these unknown classes by consistency regularization. Then we unify the complementary loss and consistency loss together by a specially designed dynamic weighting factor. We conduct a series of experiments showing that the proposed method achieves highly competitive performance in CLL.

----

## [423] Probabilistic Sufficient Explanations

**Authors**: *Eric Wang, Pasha Khosravi, Guy Van den Broeck*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/424](https://doi.org/10.24963/ijcai.2021/424)

**Abstract**:

Understanding the behavior of learned classifiers is an important task, and various black-box explanations, logical reasoning approaches, and model-specific methods have been proposed. In this paper, we introduce probabilistic sufficient explanations, which formulate explaining an instance of classification as choosing the "simplest" subset of features such that only observing those features is "sufficient" to explain the classification. That is, sufficient to give us strong probabilistic guarantees that the model will behave similarly when all features are observed under the data distribution. In addition, we leverage tractable probabilistic reasoning tools such as probabilistic circuits and expected predictions to design a scalable algorithm for finding the desired explanations while keeping the guarantees intact. Our experiments demonstrate the effectiveness of our algorithm in finding sufficient explanations, and showcase its advantages compared to Anchors and logical explanations.

----

## [424] Multi-hop Attention Graph Neural Networks

**Authors**: *Guangtao Wang, Rex Ying, Jing Huang, Jure Leskovec*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/425](https://doi.org/10.24963/ijcai.2021/425)

**Abstract**:

Self-attention mechanism in graph neural networks (GNNs) led to state-of-the-art performance on many graph representation learning tasks. Currently, at every layer, attention is computed between connected pairs of nodes and depends solely on the representation of the two nodes. However, such attention mechanism does not account for nodes that are not directly connected but provide important network context. Here we propose Multi-hop Attention Graph Neural Network (MAGNA), a principled way to incorporate multi-hop context information into every layer of attention computation. MAGNA diffuses the attention scores across the network, which increases the receptive field for every layer of the GNN. Unlike previous approaches, MAGNA uses a diffusion prior on attention values, to efficiently account for all paths between the pair of disconnected nodes. We demonstrate in theory and experiments that MAGNA captures large-scale structural information in every layer, and has a low-pass effect that eliminates noisy high-frequency information from graph data. Experimental results on node classification as well as the knowledge graph completion benchmarks show that MAGNA achieves state-of-the-art results: MAGNA achieves up to 5.7% relative error reduction over the previous state-of-the-art on Cora, Citeseer, and Pubmed. MAGNA also obtains the best performance on a large-scale Open Graph Benchmark dataset. On knowledge graph completion MAGNA advances state-of-the-art on WN18RR and FB15k-237 across four different performance metrics.

----

## [425] Learn the Highest Label and Rest Label Description Degrees

**Authors**: *Jing Wang, Xin Geng*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/426](https://doi.org/10.24963/ijcai.2021/426)

**Abstract**:

Although Label Distribution Learning (LDL) has found wide applications in varieties of classification problems, it may face the challenge of objective mismatch -- LDL neglects the optimal label for the sake of learning the whole label distribution, which leads to performance deterioration. To improve classification performance and solve the objective mismatch, we propose  a new LDL algorithm called LDL-HR. LDL-HR provides a new perspective of label distribution, \textit{i.e.}, a combination of the \textbf{highest label} and the \textbf{rest label description degrees}.   
  It works as follows. First, we learn the highest label by fitting the degenerated label distribution and large margin. Second, we learn the rest label description degrees to exploit generalization. Theoretical analysis shows the generalization of LDL-HR. Besides, the experimental results on 18 real-world datasets validate the statistical superiority of our method.

----

## [426] Stability and Generalization for Randomized Coordinate Descent

**Authors**: *Puyu Wang, Liang Wu, Yunwen Lei*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/427](https://doi.org/10.24963/ijcai.2021/427)

**Abstract**:

Randomized coordinate descent (RCD) is a popular optimization algorithm with wide applications in various machine learning problems, which motivates a lot of theoretical analysis on its convergence behavior. As a comparison, there is no work studying how the models trained by RCD would generalize to test examples. In this paper, we initialize the generalization analysis of RCD by leveraging the powerful tool of algorithmic stability. We establish argument stability bounds of RCD for both convex and strongly convex objectives, from which we develop optimal generalization bounds by showing how to early-stop the algorithm to tradeoff the estimation and optimization. Our analysis shows that RCD enjoys better stability as compared to stochastic gradient descent.

----

## [427] Discrete Multiple Kernel k-means

**Authors**: *Rong Wang, Jitao Lu, Yihang Lu, Feiping Nie, Xuelong Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/428](https://doi.org/10.24963/ijcai.2021/428)

**Abstract**:

The multiple kernel k-means (MKKM) and its variants utilize complementary information from different kernels, achieving better performance than kernel k-means (KKM). However, the optimization procedures of previous works all comprise two stages, learning the continuous relaxed label matrix and obtaining the discrete one by extra discretization procedures. Such a two-stage strategy gives rise to a mismatched problem and severe information loss. To address this problem, we elaborate a novel Discrete Multiple Kernel k-means (DMKKM) model solved by an optimization algorithm that directly obtains the cluster indicator matrix without subsequent discretization procedures. Moreover, DMKKM can strictly measure the correlations among kernels, which is capable of enhancing kernel fusion by reducing redundancy and improving diversity. Whatâ€™s more, DMKKM is parameter-free avoiding intractable hyperparameter tuning, which makes it feasible in practical applications. Extensive experiments illustrated the effectiveness and superiority of the proposed model.

----

## [428] Mean Field Equilibrium in Multi-Armed Bandit Game with Continuous Reward

**Authors**: *Xiong Wang, Riheng Jia*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/429](https://doi.org/10.24963/ijcai.2021/429)

**Abstract**:

Mean field game facilitates analyzing multi-armed bandit (MAB) for a large number of agents by approximating their interactions with an average effect. Existing mean field models for multi-agent MAB mostly assume a binary reward function, which leads to tractable analysis but is usually not applicable in practical scenarios. In this paper, we study the mean field bandit game with a continuous reward function. Specifically, we focus on deriving the existence and uniqueness of mean field equilibrium (MFE), thereby guaranteeing the asymptotic stability of the multi-agent system. To accommodate the continuous reward function, we encode the learned reward into an agent state, which is in turn mapped to its stochastic arm playing policy and updated using realized observations. We show that the state evolution is upper semi-continuous, based on which the existence of MFE is obtained. As the Markov analysis is mainly for the case of discrete state, we transform the stochastic continuous state evolution into a deterministic ordinary differential equation (ODE). On this basis, we can characterize a contraction mapping for the ODE to ensure a unique MFE for the bandit game. Extensive evaluations validate our MFE characterization, and exhibit tight empirical regret of the MAB problem.

----

## [429] Demiguise Attack: Crafting Invisible Semantic Adversarial Perturbations with Perceptual Similarity

**Authors**: *Yajie Wang, Shangbo Wu, Wenyi Jiang, Shengang Hao, Yu-an Tan, Quanxin Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/430](https://doi.org/10.24963/ijcai.2021/430)

**Abstract**:

Deep neural networks (DNNs) have been found to be vulnerable to adversarial examples. Adversarial examples are malicious images with visually imperceptible perturbations. While these carefully crafted perturbations restricted with tight Lp norm bounds are small, they are still easily perceivable by humans. These perturbations also have limited success rates when attacking black-box models or models with defenses like noise reduction filters. To solve these problems, we propose Demiguise Attack, crafting "unrestricted" perturbations with Perceptual Similarity. Specifically, we can create powerful and photorealistic adversarial examples by manipulating semantic information based on Perceptual Similarity. Adversarial examples we generate are friendly to the human visual system (HVS), although the perturbations are of large magnitudes. We extend widely-used attacks with our approach, enhancing adversarial effectiveness impressively while contributing to imperceptibility. Extensive experiments show that the proposed method not only outperforms various state-of-the-art attacks in terms of fooling rate, transferability, and robustness against defenses but can also improve attacks effectively. In addition, we also notice that our implementation can simulate illumination and contrast changes that occur in real-world scenarios, which will contribute to exposing the blind spots of DNNs.

----

## [430] Self-Supervised Adversarial Distribution Regularization for Medication Recommendation

**Authors**: *Yanda Wang, Weitong Chen, Dechang Pi, Lin Yue, Sen Wang, Miao Xu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/431](https://doi.org/10.24963/ijcai.2021/431)

**Abstract**:

Medication recommendation is a significant healthcare application due to its promise in effectively prescribing medications. Avoiding fatal side effects related to Drug-Drug Interaction (DDI) is among the critical challenges. Most existing methods try to mitigate the problem by providing models with extra DDI knowledge, making models complicated. While treating all patients with different DDI properties as a single cohort would put forward strict requirements on models' generalization performance. In pursuit of a valuable model for a safe recommendation, we propose the Self-Supervised Adversarial Regularization Model for Medication Recommendation (SARMR). SARMR obtains the target distribution associated with safe medication combinations from raw patient records for adversarial regularization. In this way, the model can shape distributions of patient representations to achieve DDI reduction. To obtain accurate self-supervision information, SARMR models interactions between physicians and patients by building a key-value memory neural network and carrying out multi-hop reading to obtain contextual information for patient representations. SARMR outperforms all baseline methods in the experiment on a real-world clinical dataset. This model can achieve DDI reduction when considering the different number of DDI types, which demonstrates the robustness of adversarial regularization for safe medication recommendation.

----

## [431] Against Membership Inference Attack: Pruning is All You Need

**Authors**: *Yijue Wang, Chenghong Wang, Zigeng Wang, Shanglin Zhou, Hang Liu, Jinbo Bi, Caiwen Ding, Sanguthevar Rajasekaran*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/432](https://doi.org/10.24963/ijcai.2021/432)

**Abstract**:

The large model size, high computational operations, and vulnerability against membership inference attack (MIA) have impeded deep learning or deep neural networks (DNNs) popularity, especially on mobile devices.  To address the challenge, we envision that the weight pruning technique will help DNNs against MIA while reducing model storage and computational operation. In this work, we propose a pruning algorithm, and we show that the proposed algorithm can find a subnetwork that can prevent privacy leakage from MIA and achieves competitive accuracy with the original DNNs. We also verify our theoretical insights with experiments. Our experimental results illustrate that the attack accuracy using model compression is up to 13.6% and 10% lower than that of the baseline and Min-Max game, accordingly.

----

## [432] Layer-Assisted Neural Topic Modeling over Document Networks

**Authors**: *Yiming Wang, Ximing Li, Jihong Ouyang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/433](https://doi.org/10.24963/ijcai.2021/433)

**Abstract**:

Neural topic modeling provides a flexible, efficient, and powerful way to extract topic representations from text documents. Unfortunately, most existing models cannot handle the text data with network links, such as web pages with hyperlinks and scientific papers with citations. To resolve this kind of data, we develop a novel neural topic model , namely Layer-Assisted Neural Topic Model (LANTM), which can be interpreted from the perspective of variational auto-encoders. Our major motivation is to enhance the topic representation encoding by not only using text contents, but also the assisted network links. Specifically, LANTM encodes the texts and network links to the topic representations by an augmented network with graph convolutional modules, and decodes them by maximizing the likelihood of the generative process. The neural variational inference is adopted for efficient inference. Experimental results validate that LANTM significantly outperforms the existing  models on topic quality, text classification and link prediction..

----

## [433] Robust Adversarial Imitation Learning via Adaptively-Selected Demonstrations

**Authors**: *Yunke Wang, Chang Xu, Bo Du*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/434](https://doi.org/10.24963/ijcai.2021/434)

**Abstract**:

The agent in imitation learning (IL) is expected to mimic the behavior of the expert. Its performance relies highly on the quality of given expert demonstrations. However, the assumption that collected demonstrations are optimal cannot always hold in real-world tasks, which would seriously influence the performance of the learned agent. In this paper, we propose a robust method within the framework of Generative Adversarial Imitation Learning (GAIL) to address imperfect demonstration issue, in which good demonstrations can be adaptively selected for training while bad demonstrations are abandoned. Specifically, a binary weight is assigned to each expert demonstration to indicate whether to select it for training. The reward function in GAIL is employed to determine this weight (i.e. higher reward results in higher weight). Compared to some existing solutions that require some auxiliary information about this weight, we set up the connection between weight and model so that we can jointly optimize GAIL and learn the latent weight. Besides hard binary weighting, we also propose a soft weighting scheme. Experiments in the Mujoco demonstrate the proposed method outperforms other GAIL-based methods when dealing with imperfect demonstrations.

----

## [434] Reinforcement Learning Based Sparse Black-box Adversarial Attack on Video Recognition Models

**Authors**: *Zeyuan Wang, Chaofeng Sha, Su Yang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/435](https://doi.org/10.24963/ijcai.2021/435)

**Abstract**:

We explore the black-box adversarial attack on video recognition models.   Attacks are only performed on selected key regions and key frames to reduce the high computation cost of searching adversarial perturbations on a video due to its high dimensionality.   To select key frames, one way is to use heuristic algorithms to evaluate the importance of each frame and choose the essential ones. However, it is time inefficient on sorting and searching. In order to speed up the attack process, we propose a reinforcement learning based frame selection strategy. Specifically, the agent explores the difference between the original class and the target class of videos to make selection decisions. It receives rewards from threat models which indicate the quality of the decisions. Besides, we also use saliency detection to select key regions and only estimate the sign of gradient instead of the gradient itself in zeroth order optimization to further boost the attack process. We can use the trained model directly in the untargeted attack or with little fine-tune in the targeted attack, which saves computation time. A range of empirical results on real datasets demonstrate the effectiveness and efficiency of the proposed method.

----

## [435] Reward-Constrained Behavior Cloning

**Authors**: *Zhaorong Wang, Meng Wang, Jingqi Zhang, Yingfeng Chen, Chongjie Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/436](https://doi.org/10.24963/ijcai.2021/436)

**Abstract**:

Deep reinforcement learning (RL) has demonstrated success in challenging decision-making/control tasks. However, RL methods, which solve tasks through maximizing the expected reward, may generate undesirable behaviors due to inferior local convergence or incompetent reward design. These undesirable behaviors of agents may not reduce the total reward but destroy the user experience of the application. For example, in the autonomous driving task, the policy actuated by speed reward behaves much more sudden brakes while human drivers generally don’t do that. To overcome this problem, we present a novel method named Reward-Constrained Behavior Cloning (RCBC) which synthesizes imitation learning and constrained reinforcement learning. RCBC leverages human demonstrations to induce desirable or human-like behaviors and employs lower-bound reward constraints for policy optimization to maximize the expected reward. Empirical results on popular benchmark environments show that RCBC learns signiﬁcantly more human-desired policies with performance guarantees which meet the lower-bound reward constraints while performing better than or as well as baseline methods in terms of reward maximization.

----

## [436] Closing the BIG-LID: An Effective Local Intrinsic Dimensionality Defense for Nonlinear Regression Poisoning

**Authors**: *Sandamal Weerasinghe, Tamas Abraham, Tansu Alpcan, Sarah M. Erfani, Christopher Leckie, Benjamin I. P. Rubinstein*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/437](https://doi.org/10.24963/ijcai.2021/437)

**Abstract**:

Nonlinear regression, although widely used in engineering, financial and security applications for automated decision making, is known to be vulnerable to training data poisoning. Targeted poisoning attacks may cause learning algorithms to fit decision functions with poor predictive performance. This paper presents a new analysis of local intrinsic dimensionality (LID) of nonlinear regression under such poisoning attacks within a Stackelberg game, leading to a practical defense. After adapting a gradient-based attack on linear regression that significantly impairs prediction capabilities to nonlinear settings, we consider a multi-step unsupervised black-box defense. The first step identifies samples that have the greatest influence on the learner's validation error; we then use the theory of local intrinsic dimensionality, which reveals the degree of being an outlier of data samples, to iteratively identify poisoned samples via a generative probabilistic model, and suppress their influence on the prediction function. Empirical validation demonstrates superior performance compared to a range of recent defenses.

----

## [437] GSPL: A Succinct Kernel Model for Group-Sparse Projections Learning of Multiview Data

**Authors**: *Danyang Wu, Jin Xu, Xia Dong, Meng Liao, Rong Wang, Feiping Nie, Xuelong Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/438](https://doi.org/10.24963/ijcai.2021/438)

**Abstract**:

This paper explores a succinct kernel model for Group-Sparse Projections Learning (GSPL), to handle multiview feature selection task completely. Compared to previous works, our model has the following useful properties: 1) Strictness: GSPL innovatively learns group-sparse projections strictly on multiview data via â€˜2;0-norm constraint, which is different with previous works that encourage group-sparse projections softly. 2) Adaptivity: In GSPL model, when the total number of selected features is given, the numbers of selected features of different views can be determined adaptively,
which avoids artificial settings. Besides, GSPL can capture the differences among multiple views adaptively, which handles the inconsistent problem among different views. 3) Succinctness: Except for the intrinsic parameters of projection-based feature selection task, GSPL does not bring extra parameters, which guarantees the applicability in practice. To solve the optimization problem involved in GSPL, a novel iterative algorithm is proposed with rigorously theoretical guarantees. Experimental results demonstrate the superb performance of GSPL on synthetic and real datasets.

----

## [438] Deep Reinforcement Learning Boosted Partial Domain Adaptation

**Authors**: *Keyu Wu, Min Wu, Jianfei Yang, Zhenghua Chen, Zhengguo Li, Xiaoli Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/439](https://doi.org/10.24963/ijcai.2021/439)

**Abstract**:

Domain adaptation is critical for learning transferable features that effectively reduce the distribution difference among domains. In the era of big data, the availability of large-scale labeled datasets motivates partial domain adaptation (PDA) which deals with adaptation from large source domains to small target domains with less number of classes. In the PDA setting, it is crucial to transfer relevant source samples and eliminate irrelevant ones to mitigate negative transfer. In this paper, we propose a deep reinforcement learning based source data selector for PDA, which is capable of eliminating less relevant source samples automatically to boost existing adaptation methods. It determines to either keep or discard the source instances based on their feature representations so that more effective knowledge transfer across domains can be achieved via filtering out irrelevant samples. As a general module, the proposed DRL-based data selector can be integrated into any existing domain adaptation or partial domain adaptation models. Extensive experiments on several benchmark datasets demonstrate the superiority of the proposed DRL-based data selector which leads to state-of-the-art performance for various PDA tasks.

----

## [439] Learning Deeper Non-Monotonic Networks by Softly Transferring Solution Space

**Authors**: *Zheng-Fan Wu, Hui Xue, Weimin Bai*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/440](https://doi.org/10.24963/ijcai.2021/440)

**Abstract**:

Different from popular neural networks using quasiconvex activations, non-monotonic networks activated by periodic nonlinearities have emerged as a more competitive paradigm, offering revolutionary benefits: 1) compactly characterizing high-frequency patterns; 2) precisely representing high-order derivatives. Nevertheless, they are also well-known for being hard to train, due to easily over-fitting dissonant noise and only allowing for tiny architectures (shallower than 5 layers). The fundamental bottleneck is that the periodicity leads to many poor and dense local minima in solution space. The direction and norm of gradient oscillate continually during error backpropagation. Thus non-monotonic networks are prematurely stuck in these local minima, and leave out effective error feedback. To alleviate the optimization dilemma, in this paper, we propose a non-trivial soft transfer approach. It smooths their solution space close to that of monotonic ones in the beginning, and then improve their representational properties by transferring the solutions from the neural space of monotonic neurons to the Fourier space of non-monotonic neurons as the training continues. The soft transfer consists of two core components: 1) a rectified concrete gate is constructed to characterize the state of each neuron; 2) a variational Bayesian learning framework is proposed to dynamically balance the empirical risk and the intensity of transfer. We provide comprehensive empirical evidence showing that the soft transfer not only reduces the risk of non-monotonic networks on over-fitting noise, but also helps them scale to much deeper architectures (more than 100 layers) achieving the new state-of-the-art performance.

----

## [440] Exploiting Spiking Dynamics with Spatial-temporal Feature Normalization in Graph Learning

**Authors**: *Mingkun Xu, Yujie Wu, Lei Deng, Faqiang Liu, Guoqi Li, Jing Pei*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/441](https://doi.org/10.24963/ijcai.2021/441)

**Abstract**:

Biological spiking neurons with intrinsic dynamics underlie the powerful representation and learning capabilities of the brain for processing multimodal information in complex environments. Despite recent tremendous progress in spiking neural networks (SNNs) for handling Euclidean-space tasks, it still remains challenging to exploit SNNs in processing non-Euclidean-space data represented by graph data, mainly due to the lack of effective modeling framework and useful training techniques. Here we present a general spike-based modeling framework that enables the direct training of SNNs for graph learning. Through spatial-temporal unfolding for spiking data flows of node features, we incorporate graph convolution filters into spiking dynamics and formalize a synergistic learning paradigm. Considering the unique features of spike representation and spiking dynamics, we propose a spatial-temporal feature normalization (STFN) technique suitable for SNN to accelerate convergence. We instantiate our methods into two spiking graph models, including graph convolution SNNs and graph attention SNNs, and validate their performance on three node-classification benchmarks, including Cora, Citeseer, and Pubmed. Our model can achieve comparable performance with the state-of-the-art graph neural network (GNN) models with much lower computation costs, demonstrating great benefits for the execution on neuromorphic hardware and prompting neuromorphic applications in graphical scenarios.

----

## [441] k-Nearest Neighbors by Means of Sequence to Sequence Deep Neural Networks and Memory Networks

**Authors**: *Yiming Xu, Diego Klabjan*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/442](https://doi.org/10.24963/ijcai.2021/442)

**Abstract**:

k-Nearest Neighbors is one of the most fundamental but effective classification models. In this paper, we propose two families of models built on a sequence to sequence model and a memory network model to mimic the k-Nearest Neighbors model, which generate a sequence of labels, a sequence of out-of-sample feature vectors and a final label for classification, and thus they could also function as oversamplers. We also propose 'out-of-core' versions of our models which assume that only a small portion of data can be loaded into memory. Computational experiments show that our models on structured datasets outperform k-Nearest Neighbors, a feed-forward neural network, XGBoost, lightGBM, random forest and a memory network, due to the fact that our models must produce additional output and not just the label. On image and text datasets, the performance of our model is close to many state-of-the-art deep models. As an oversampler on imbalanced datasets, the sequence to sequence kNN model often outperforms Synthetic Minority Over-sampling Technique and Adaptive Synthetic Sampling.

----

## [442] Evolutionary Gradient Descent for Non-convex Optimization

**Authors**: *Ke Xue, Chao Qian, Ling Xu, Xudong Fei*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/443](https://doi.org/10.24963/ijcai.2021/443)

**Abstract**:

Non-convex optimization is often involved in artificial intelligence tasks, which may have many saddle points, and is NP-hard to solve. Evolutionary algorithms (EAs) are general-purpose derivative-free optimization algorithms with a good ability to find the global optimum, which can be naturally applied to non-convex optimization. Their performance is, however, limited due to low efficiency. Gradient descent (GD) runs efficiently, but only converges to a first-order stationary point, which may be a saddle point and thus arbitrarily bad. Some recent efforts have been put into combining EAs and GD. However, previous works either utilized only a specific component of EAs, or just combined them heuristically without theoretical guarantee. In this paper, we propose an evolutionary GD (EGD) algorithm by combining typical components, i.e., population and mutation, of EAs with GD. We prove that EGD can converge to a second-order stationary point by escaping the saddle points, and is more efficient than previous algorithms. Empirical results on non-convex synthetic functions as well as reinforcement learning (RL) tasks also show its superiority.

----

## [443] KDExplainer: A Task-oriented Attention Model for Explaining Knowledge Distillation

**Authors**: *Mengqi Xue, Jie Song, Xinchao Wang, Ying Chen, Xingen Wang, Mingli Song*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/444](https://doi.org/10.24963/ijcai.2021/444)

**Abstract**:

Knowledge distillation (KD) has recently emerged as an efficacious scheme for learning compact deep neural networks (DNNs). Despite the promising results achieved, the rationale that interprets the behavior of KD has yet remained largely understudied. In this paper, we introduce a novel task-oriented attention model, termed as KDExplainer, to shed light on the working mechanism underlying the vanilla KD. At the heart of KDExplainer is a Hierarchical Mixture of Experts (HME), in which a multi-class classification is reformulated as a multi-task binary one. Through distilling knowledge from a free-form pre-trained DNN to KDExplainer, we observe that KD implicitly modulates the knowledge conflicts between different subtasks, and in reality has much more to offer than label smoothing. Based on such findings, we further introduce a portable tool, dubbed as virtual attention module (VAM), that can be seamlessly integrated with various DNNs to enhance their performance under KD. Experimental results demonstrate that with a negligible additional cost, student models equipped with VAM consistently outperform their non-VAM counterparts across different benchmarks. Furthermore, when combined with other KD methods, VAM remains competent in promoting results, even though it is only motivated by vanilla KD. The code is available at https:// github.com/zju-vipa/KDExplainer.

----

## [444] Clustering-Induced Adaptive Structure Enhancing Network for Incomplete Multi-View Data

**Authors**: *Zhe Xue, Junping Du, Changwei Zheng, Jie Song, Wenqi Ren, MeiYu Liang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/445](https://doi.org/10.24963/ijcai.2021/445)

**Abstract**:

Incomplete multi-view clustering aims to cluster samples with missing views, which has drawn more and more research interest. Although several methods have been developed for incomplete multi-view clustering, they fail to extract and exploit the comprehensive global and local structure of multi-view data, so their clustering performance is limited. This paper proposes a Clustering-induced Adaptive Structure Enhancing Network (CASEN) for incomplete multi-view clustering, which is an end-to-end trainable framework that jointly conducts multi-view structure enhancing and data clustering. Our method adopts multi-view autoencoder to infer the missing features of the incomplete samples. Then, we perform adaptive graph learning and graph convolution on the reconstructed complete multi-view data to effectively extract data structure. Moreover, we use multiple kernel clustering to integrate the global and local structure for clustering, and the clustering results in turn are used to enhance the data structure. Extensive experiments on several benchmark datasets demonstrate that our method can comprehensively obtain the structure of incomplete multi-view data and achieve superior performance compared to the other methods.

----

## [445] Differentially Private Pairwise Learning Revisited

**Authors**: *Zhiyu Xue, Shaoyang Yang, Mengdi Huai, Di Wang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/446](https://doi.org/10.24963/ijcai.2021/446)

**Abstract**:

Instead of learning with pointwise loss functions, learning with pairwise loss functions (pairwise learning) has received much attention recently as it is more capable of modeling the relative relationship between pairs of samples.   However, most of the existing algorithms for pairwise learning fail to take into consideration the privacy issue in their design.  To address this  issue, previous work studied pairwise learning in the Differential Privacy (DP) model. However, their utilities (population errors) are far from optimal. To address the sub-optimal utility issue, in this paper, we proposed new pure or approximate DP algorithms for pairwise learning. Specifically, under the assumption that the loss functions are Lipschitz, our algorithms could achieve the optimal expected population risk for both strongly convex and general convex cases. We also conduct extensive experiments on real-world datasets to evaluate the proposed algorithms, experimental results support our theoretical analysis and show the priority of our algorithms.

----

## [446] Decomposable-Net: Scalable Low-Rank Compression for Neural Networks

**Authors**: *Atsushi Yaguchi, Taiji Suzuki, Shuhei Nitta, Yukinobu Sakata, Akiyuki Tanizawa*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/447](https://doi.org/10.24963/ijcai.2021/447)

**Abstract**:

Compressing DNNs is important for the real-world applications operating on resource-constrained devices. However, we typically observe drastic performance deterioration when changing model size after training is completed. Therefore, retraining is required to resume the performance of the compressed models suitable for different devices. In this paper, we propose Decomposable-Net (the network decomposable in any size), which allows flexible changes to model size without retraining. We decompose weight matrices in the DNNs via singular value decomposition and adjust ranks according to the target model size. Unlike the existing low-rank compression methods that specialize the model to a fixed size, we propose a novel backpropagation scheme that jointly minimizes losses for both of full- and low-rank networks. This enables not only to maintain the performance of a full-rank network {\it without retraining} but also to improve low-rank networks in multiple sizes. Additionally, we introduce a simple criterion for rank selection that effectively suppresses approximation error. In experiments on the ImageNet classification task, Decomposable-Net yields superior accuracy in a wide range of model sizes. In particular, Decomposable-Net achieves the top-1 accuracy of 73.2% with 0.27xMACs with ResNet-50, compared to Tucker decomposition (67.4% / 0.30x), Trained Rank Pruning (70.6% / 0.28x), and universally slimmable networks (71.4% / 0.26x).

----

## [447] A Clustering-based framework for Classifying Data Streams

**Authors**: *Xuyang Yan, Abdollah Homaifar, Mrinmoy Sarkar, Abenezer Girma, Edward W. Tunstel*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/448](https://doi.org/10.24963/ijcai.2021/448)

**Abstract**:

The non-stationary nature of data streams strongly challenges traditional machine learning techniques. Although some solutions have been proposed to extend traditional machine learning techniques for handling data streams, these approaches either require an initial label set or rely on specialized design parameters. The overlap among classes and the labeling of data streams constitute other major challenges for classifying data streams. In this paper, we proposed a clustering-based data stream classification framework to handle non-stationary data streams without utilizing an initial label set. A density-based stream clustering procedure is used to capture novel concepts with a dynamic threshold and an effective active label querying strategy is introduced to continuously learn the new concepts from the data streams. The sub-cluster structure of each cluster is explored to handle the overlap among classes. Experimental results and quantitative comparison studies reveal that the proposed method provides statistically better or comparable performance than the existing methods.

----

## [448] Multi-level Generative Models for Partial Label Learning with Non-random Label Noise

**Authors**: *Yan Yan, Yuhong Guo*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/449](https://doi.org/10.24963/ijcai.2021/449)

**Abstract**:

Partial label (PL) learning tackles the problem where each training instance is associated with a set of candidate labels that include both the true label and some irrelevant noise labels. In this paper, we propose a novel multi-level generative model for partial label learning (MGPLL), which tackles the PL problem by learning both a label level adversarial generator and a feature level adversarial generator under a bi-directional mapping framework between the label vectors and the data samples. MGPLL uses a conditional noise label generation network to model the non-random noise labels and perform label denoising, and uses a multi-class predictor to map the training instances to the denoised label vectors, while a conditional data feature generator is used to form an inverse mapping from the denoised label vectors to data samples. Both the noise label generator and the data feature generator are learned in an adversarial manner to match the observed candidate labels and data features respectively. We conduct extensive experiments on both synthesized and real-world partial label datasets. The proposed approach
demonstrates the state-of-the-art performance for partial label learning.

----

## [449] Secure Deep Graph Generation with Link Differential Privacy

**Authors**: *Carl Yang, Haonan Wang, Ke Zhang, Liang Chen, Lichao Sun*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/450](https://doi.org/10.24963/ijcai.2021/450)

**Abstract**:

Many data mining and analytical tasks rely on the abstraction of networks (graphs) to summarize relational structures among individuals (nodes). Since relational data are often sensitive, we aim to seek effective approaches to generate utility-preserved yet privacy-protected structured data.
In this paper, we leverage the differential privacy (DP) framework to formulate and enforce rigorous privacy constraints on deep graph generation models, with a focus on edge-DP to guarantee individual link privacy.
In particular, we enforce edge-DP by injecting designated noise to the gradients of a link reconstruction based graph generation model, while ensuring data utility by improving structure learning with structure-oriented graph discrimination.
Extensive experiments on two real-world network datasets show that our proposed DPGGAN model is able to generate graphs with effectively preserved global structure and rigorously protected individual link privacy.

----

## [450] Progressive Open-Domain Response Generation with Multiple Controllable Attributes

**Authors**: *Haiqin Yang, Xiaoyuan Yao, Yiqun Duan, Jianping Shen, Jie Zhong, Kun Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/451](https://doi.org/10.24963/ijcai.2021/451)

**Abstract**:

It is desirable to include more controllable attributes to enhance the diversity of generated responses in open-domain dialogue systems.  However, existing methods can generate responses with only one controllable attribute or lack a flexible way to generate them with multiple controllable attributes.  In this paper, we propose a Progressively trained Hierarchical Encoder-Decoder (PHED) to tackle this task.  More specifically, PHED deploys Conditional Variational AutoEncoder (CVAE) on Transformer to include one aspect of attributes at one stage.  A vital characteristic of the CVAE is to separate the latent variables at each stage into two types: a global variable capturing the common semantic features and a specific variable absorbing the attribute information at that stage.  PHED then couples the CVAE latent variables with the Transformer encoder and is trained by minimizing a newly derived ELBO and controlled losses to produce the next stage's input and produce responses as required.  Finally, we conduct extensive evaluations to show that PHED significantly outperforms the state-of-the-art neural generation models and produces more diverse responses as expected.

----

## [451] Unsupervised Path Representation Learning with Curriculum Negative Sampling

**Authors**: *Sean Bin Yang, Chenjuan Guo, Jilin Hu, Jian Tang, Bin Yang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/452](https://doi.org/10.24963/ijcai.2021/452)

**Abstract**:

Path representations are critical in a variety of transportation applications, such as estimating path ranking in path recommendation systems and estimating path travel time in navigation systems. Existing studies often learn task-specific path representations in a supervised manner, which require a large amount of labeled training data and generalize poorly to other tasks. We propose an unsupervised learning framework Path InfoMax (PIM) to learn generic path representations that work for different downstream tasks. We first propose a curriculum negative sampling method, for each input path, to generate a small amount of negative paths, by following the principles of curriculum learning. Next, PIM employs mutual information maximization to learn path representations from both a global and a local view.  In the global view, PIM distinguishes the representations of the input paths from those of the negative paths. In the local view, PIM distinguishes the input path representations from the representations of the nodes that appear only in the negative paths. This enables the learned path representations encode both global and local information at different scales. Extensive experiments on two downstream tasks, ranking score estimation and travel time estimation, using two road network datasets suggest that PIM significantly outperforms other unsupervised methods and is also able to be used as a pre-training method to enhance supervised path representation learning.

----

## [452] BESA: BERT-based Simulated Annealing for Adversarial Text Attacks

**Authors**: *Xinghao Yang, Weifeng Liu, Dacheng Tao, Wei Liu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/453](https://doi.org/10.24963/ijcai.2021/453)

**Abstract**:

Modern Natural Language Processing (NLP) models are known immensely brittle towards text adversarial examples. Recent attack algorithms usually adopt word-level substitution strategies following a pre-computed word replacement mechanism. However, their resultant adversarial examples are still imperfect in achieving grammar correctness and semantic similarities, which is largely because of their unsuitable candidate word selections and static optimization methods. In this research, we propose BESA, a BERT-based Simulated Annealing algorithm,  to address these two problems. Firstly, we leverage the BERT Masked Language Model (MLM) to generate contextual-aware candidate words to produce fluent adversarial text and avoid grammar errors. Secondly, we employ Simulated Annealing (SA) to adaptively determine the word substitution order. The SA provides sufficient word replacement options via internal simulations, with an objective to obtain both a high attack success rate and a low word substitution rate.  Besides, our algorithm is able to jump out of local optima with a controlled probability, making it closer to achieve the best possible attack (i.e., the global optima). Experiments on five popular datasets manifest the superiority of BESA compared with existing methods, including TextFooler, BAE, BERT-Attack, PWWS, and PSO.

----

## [453] Rethinking Label-Wise Cross-Modal Retrieval from A Semantic Sharing Perspective

**Authors**: *Yang Yang, Chubing Zhang, Yi-Chu Xu, Dianhai Yu, De-Chuan Zhan, Jian Yang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/454](https://doi.org/10.24963/ijcai.2021/454)

**Abstract**:

The main challenge of cross-modal retrieval is to learn the consistent embedding for heterogeneous modalities. To solve this problem, traditional label-wise cross-modal approaches usually constrain the inter-modal and intra-modal embedding consistency relying on the label ground-truths. However, the experiments reveal that different modal networks actually have various generalization capacities, thereby end-to-end joint training with consistency loss usually leads to sub-optimal uni-modal model, which in turn affects the learning of consistent embedding. Therefore, in this paper, we argue that what really needed for supervised cross-modal retrieval is a good shared classification model. In other words, we learn the consistent embedding by ensuring the classification performance of each modality on the shared model, without the consistency loss. Specifically, we consider a technique called Semantic Sharing, which directly trains the two modalities interactively by adopting a shared self-attention based classification model. We evaluate the proposed approach on three representative datasets. The results validate that the proposed semantic sharing can consistently boost the performance under NDCG metric.

----

## [454] Blocking-based Neighbor Sampling for Large-scale Graph Neural Networks

**Authors**: *Kai-Lang Yao, Wu-Jun Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/455](https://doi.org/10.24963/ijcai.2021/455)

**Abstract**:

The exponential increase in computation and memory complexity with the depth of network has become the main impediment to the successful application of graph neural networks (GNNs) on large-scale graphs like graphs with hundreds of millions of nodes. In this paper, we propose a novel neighbor sampling strategy, dubbed blocking-based neighbor sampling (BNS), for efficient training of GNNs on large-scale graphs. Specifically, BNS adopts a policy to stochastically block the ongoing expansion of neighboring nodes, which can reduce the rate of the exponential increase in computation and memory complexity of GNNs. Furthermore, a reweighted policy is applied to graph convolution, to adjust the contribution of blocked and non-blocked neighbors to central nodes. We theoretically prove that BNS provides an unbiased estimation for the original graph convolution operation. Extensive experiments on three benchmark datasets show that, on large-scale graphs, BNS is 2X~5X faster than state-of-the-art methods when achieving the same accuracy. Moreover, even on the small-scale graphs, BNS also demonstrates the advantage of low time cost.

----

## [455] Understanding the Effect of Bias in Deep Anomaly Detection

**Authors**: *Ziyu Ye, Yuxin Chen, Haitao Zheng*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/456](https://doi.org/10.24963/ijcai.2021/456)

**Abstract**:

Anomaly detection presents a unique challenge in machine learning, due to the scarcity of labeled anomaly data. Recent work attempts to mitigate such problems by augmenting training of deep anomaly detection models with additional labeled anomaly samples. However, the labeled data often does not align with the target distribution and introduces harmful bias to the trained model. In this paper, we aim to understand the effect of a biased anomaly set on anomaly detection. Concretely, we view anomaly detection as a supervised learning task where the objective is to optimize the recall at a given false positive rate. We formally study the relative scoring bias of an anomaly detector, defined as the difference in performance with respect to a baseline anomaly detector. We establish the first finite sample rates for estimating the relative scoring bias for deep anomaly detection, and empirically validate our theoretical results on both synthetic and real-world datasets. We also provide an extensive empirical study on how a biased training anomaly set affects the anomaly score function and therefore the detection performance on different anomaly classes. Our study demonstrates scenarios in which the biased anomaly set can be useful or problematic, and provides a solid benchmark for future research.

----

## [456] Improving Sequential Recommendation Consistency with Self-Supervised Imitation

**Authors**: *Xu Yuan, Hongshen Chen, Yonghao Song, Xiaofang Zhao, Zhuoye Ding*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/457](https://doi.org/10.24963/ijcai.2021/457)

**Abstract**:

Most sequential recommendation models capture the features of consecutive items in a user-item interaction history. Though effective, their representation expressiveness is still hindered by the sparse learning signals. As a result, the sequential recommender is prone to make inconsistent predictions. In this paper, we propose a model, SSI, to improve sequential recommendation consistency with Self-Supervised Imitation. Precisely, we extract the consistency knowledge by utilizing three self-supervised pre-training tasks, where temporal consistency and persona consistency capture user-interaction dynamics in terms of the chronological order and persona sensitivities, respectively. Furthermore, to provide the model with a global perspective, global session consistency is introduced by maximizing the mutual information among global and local interaction sequences. Finally, to comprehensively take advantage of all three independent aspects of consistency-enhanced knowledge, we establish an integrated imitation learning framework. The consistency knowledge is effectively internalized and transferred to the student model by imitating the conventional prediction logit as well as the consistency-enhanced item representations. In addition, the flexible self-supervised imitation framework can also benefit other student recommenders. Experiments on four real-world datasets show that SSI effectively outperforms the state-of-the-art sequential recommendation methods.

----

## [457] Graph Universal Adversarial Attacks: A Few Bad Actors Ruin Graph Learning Models

**Authors**: *Xiao Zang, Yi Xie, Jie Chen, Bo Yuan*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/458](https://doi.org/10.24963/ijcai.2021/458)

**Abstract**:

Deep neural networks, while generalize well, are known to be sensitive to small adversarial perturbations. This phenomenon poses severe security threat and calls for in-depth investigation of the robustness of deep learning models. With the emergence of neural networks for graph structured data, similar investigations are urged to understand their robustness. It has been found that adversarially perturbing the graph structure and/or node features may result in a significant degradation of the model performance. In this work, we show from a different angle that such fragility similarly occurs if the graph contains a few bad-actor nodes, which compromise a trained graph neural network through flipping the connections to any targeted victim. Worse, the bad actors found for one graph model severely compromise other models as well. We call the bad actors ``anchor nodes'' and propose an algorithm, named GUA, to identify them. Thorough empirical investigations suggest an interesting finding that the anchor nodes often belong to the same class; and they also corroborate the intuitive trade-off between the number of anchor nodes and the attack success rate. For the dataset Cora which contains 2708 nodes, as few as six anchor nodes will result in an attack success rate higher than 80% for GCN and other three models.

----

## [458] Hindsight Trust Region Policy Optimization

**Authors**: *Hanbo Zhang, Site Bai, Xuguang Lan, David Hsu, Nanning Zheng*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/459](https://doi.org/10.24963/ijcai.2021/459)

**Abstract**:

Reinforcement Learning (RL) with sparse rewards is a major challenge. We pro-
pose Hindsight Trust Region Policy Optimization (HTRPO), a new RL algorithm
that extends the highly successful TRPO algorithm with hindsight to tackle the
challenge of sparse rewards. Hindsight refers to the algorithmâ€™s ability to learn
from information across goals, including past goals not intended for the current
task. We derive the hindsight form of TRPO, together with QKL, a quadratic
approximation to the KL divergence constraint on the trust region. QKL reduces
variance in KL divergence estimation and improves stability in policy updates. We
show that HTRPO has similar convergence property as TRPO. We also present
Hindsight Goal Filtering (HGF), which further improves the learning performance
for suitable tasks. HTRPO has been evaluated on various sparse-reward tasks,
including Atari games and simulated robot control. Experimental results show that
HTRPO consistently outperforms TRPO, as well as HPG, a state-of-the-art policy
14 gradient algorithm for RL with sparse rewards.

----

## [459] Deep Descriptive Clustering

**Authors**: *Hongjing Zhang, Ian Davidson*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/460](https://doi.org/10.24963/ijcai.2021/460)

**Abstract**:

Recent work on explainable clustering allows describing clusters when the features are interpretable. However, much modern machine learning focuses on complex data such as images, text, and graphs where deep learning is used but the raw features of data are not interpretable. This paper explores a novel setting for performing clustering on complex data while simultaneously generating explanations using interpretable tags. We propose deep descriptive clustering that performs sub-symbolic representation learning on complex data while generating explanations based on symbolic data. We form good clusters by maximizing the mutual information between empirical distribution on the inputs and the induced clustering labels for clustering objectives. We generate explanations by solving an integer linear programming that generates concise and orthogonal descriptions for each cluster. Finally, we allow the explanation to inform better clustering by proposing a novel pairwise loss with self-generated constraints to maximize the clustering and explanation module's consistency. Experimental results on public data demonstrate that our model outperforms competitive baselines in clustering performance while offering high-quality cluster-level explanations.

----

## [460] Independence-aware Advantage Estimation

**Authors**: *Pushi Zhang, Li Zhao, Guoqing Liu, Jiang Bian, Minlie Huang, Tao Qin, Tie-Yan Liu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/461](https://doi.org/10.24963/ijcai.2021/461)

**Abstract**:

Most of existing advantage function estimation methods in reinforcement learning suffer from the problem of high variance, which scales unfavorably with the time horizon. To address this challenge, we propose to identify the independence property between current action and future states in environments, which can be further leveraged to effectively reduce the variance of the advantage estimation. In particular, the recognized independence property can be naturally utilized to construct a novel importance sampling advantage estimator with close-to-zero variance even when the Monte-Carlo return signal yields a large variance. To further remove the risk of the high variance introduced by the new estimator, we combine it with existing Monte-Carlo estimator via a reward decomposition model learned by minimizing the estimation variance. Experiments demonstrate that our method achieves higher sample efficiency compared with existing advantage estimation methods in complex environments.

----

## [461] UNBERT: User-News Matching BERT for News Recommendation

**Authors**: *Qi Zhang, Jingjie Li, Qinglin Jia, Chuyuan Wang, Jieming Zhu, Zhaowei Wang, Xiuqiang He*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/462](https://doi.org/10.24963/ijcai.2021/462)

**Abstract**:

Nowadays, news recommendation has become a popular channel for users to access news of their interests. How to represent rich textual contents of news and precisely match users' interests and candidate news lies in the core of news recommendation. However, existing recommendation methods merely learn textual representations from in-domain news data, which limits their generalization ability to new news that are common in cold-start scenarios. Meanwhile, many of these methods represent each user by aggregating the historically browsed news into a single vector and then compute the matching score with the candidate news vector, which may lose the low-level matching signals. In this paper, we explore the use of the successful BERT pre-training technique in NLP for news recommendation and propose a BERT-based user-news matching model, called UNBERT. In contrast to existing research, our UNBERT model not only leverages the pre-trained model with rich language knowledge to enhance textual representation, but also captures multi-grained user-news matching signals at both word-level and news-level. Extensive experiments on the Microsoft News Dataset (MIND) demonstrate that our approach constantly outperforms the state-of-the-art methods.

----

## [462] Correlation-Guided Representation for Multi-Label Text Classification

**Authors**: *Qian-Wen Zhang, Ximing Zhang, Zhao Yan, Ruifang Liu, Yunbo Cao, Min-Ling Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/463](https://doi.org/10.24963/ijcai.2021/463)

**Abstract**:

Multi-label text classification is an essential task in natural language processing. Existing multi-label classification models generally consider labels as categorical variables and ignore the exploitation of label semantics. In this paper, we view the task as a correlation-guided text representation problem: an attention-based two-step framework is proposed to integrate text information and label semantics by jointly learning words and labels in the same space. In this way, we aim to capture high-order label-label correlations as well as context-label correlations. Specifically, the proposed approach works by learning token-level representations of words and labels globally through a multi-layer Transformer and constructing an attention vector through word-label correlation matrix to generate the text representation. It ensures that relevant words receive higher weights than irrelevant words and thus directly optimizes the classification performance.  Extensive experiments over benchmark multi-label datasets clearly validate the effectiveness of the proposed approach, and further analysis demonstrates that it is competitive in both predicting low-frequency labels and convergence speed.

----

## [463] Private Stochastic Non-convex Optimization with Improved Utility Rates

**Authors**: *Qiuchen Zhang, Jing Ma, Jian Lou, Li Xiong*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/464](https://doi.org/10.24963/ijcai.2021/464)

**Abstract**:

We study the differentially private (DP) stochastic nonconvex optimization with a focus on its under-studied utility measures in terms of the expected excess empirical and population risks. While the excess risks are extensively studied for convex optimization, they are rarely studied for nonconvex optimization, especially the expected population risk. For the convex case, recent studies show that it is possible for private optimization to achieve the same order of excess population risk as to the nonprivate optimization under certain conditions. It still remains an open question for the nonconvex case whether such ideal excess population risk is achievable.
  
In this paper, we progress towards an affirmative answer to this open problem: DP nonconvex optimization is indeed capable of achieving the same excess population risk as to the nonprivate algorithm in most common parameter regimes, under certain conditions (i.e., well-conditioned nonconvexity). We achieve such improved utility rates compared to existing results by designing and analyzing the stagewise DP-SGD with early momentum algorithm. We obtain both excess empirical risk and excess population risk to achieve differential privacy. Our algorithm also features the first known results of excess and population risks for DP-SGD with momentum. Experiment results on both shallow and deep neural networks when respectively applied to simple and complex real datasets corroborate the theoretical results.

----

## [464] Non-IID Multi-Instance Learning for Predicting Instance and Bag Labels with Variational Auto-Encoder

**Authors**: *Weijia Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/465](https://doi.org/10.24963/ijcai.2021/465)

**Abstract**:

Multi-instance learning is a type of weakly supervised learning. It deals with tasks where the data is a set of bags and each bag is a set of instances. Only the bag labels are observed whereas the labels for the instances are unknown. An important advantage of multi-instance learning is that by representing objects as a bag of instances, it is able to preserve the inherent dependencies among parts of the objects. Unfortunately, most existing algorithms assume all instances to be identically and independently distributed, which violates real-world scenarios since the instances within a bag are rarely independent. In this work, we propose the Multi-Instance Variational Autoencoder (MIVAE) algorithm which explicitly models the dependencies among the instances for predicting both bag labels and instance labels. Experimental results on several multi-instance benchmarks and end-to-end medical imaging datasets demonstrate that MIVAE performs better than state-of-the-art algorithms for both instance label and bag label prediction tasks.

----

## [465] Model-based Multi-agent Policy Optimization with Adaptive Opponent-wise Rollouts

**Authors**: *Weinan Zhang, Xihuai Wang, Jian Shen, Ming Zhou*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/466](https://doi.org/10.24963/ijcai.2021/466)

**Abstract**:

This paper investigates the model-based methods in multi-agent reinforcement learning (MARL). We specify the dynamics sample complexity and the opponent sample complexity in MARL, and conduct a theoretic analysis of return discrepancy upper bound. To reduce the upper bound with the intention of low sample complexity during the whole learning process, we propose a novel decentralized model-based MARL method, named Adaptive Opponent-wise Rollout Policy Optimization (AORPO). In AORPO, each agent builds its multi-agent environment model, consisting of a dynamics model and multiple opponent models, and trains its policy with the adaptive opponent-wise rollout. We further prove the theoretic convergence of AORPO under reasonable assumptions. Empirical experiments on competitive and cooperative tasks demonstrate that AORPO can achieve improved sample efficiency with comparable asymptotic performance over the compared MARL methods.

----

## [466] Rethink the Connections among Generalization, Memorization, and the Spectral Bias of DNNs

**Authors**: *Xiao Zhang, Haoyi Xiong, Dongrui Wu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/467](https://doi.org/10.24963/ijcai.2021/467)

**Abstract**:

Over-parameterized deep neural networks (DNNs) with sufficient capacity to memorize random noise can achieve excellent generalization performance, challenging the bias-variance trade-off in classical learning theory. Recent studies claimed that DNNs first learn simple patterns and then memorize noise; some other works showed a phenomenon that DNNs have a spectral bias to learn target functions from low to high frequencies during training. However, we show that the monotonicity of the learning bias does not always hold: under the experimental setup of deep double descent, the high-frequency components of DNNs diminish in the late stage of training, leading to the second descent of the test error. Besides, we find that the spectrum of DNNs can be applied to indicating the second descent of the test error, even though it is calculated from the training set only.

----

## [467] User Retention: A Causal Approach with Triple Task Modeling

**Authors**: *Yang Zhang, Dong Wang, Qiang Li, Yue Shen, Ziqi Liu, Xiaodong Zeng, Zhiqiang Zhang, Jinjie Gu, Derek F. Wong*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/468](https://doi.org/10.24963/ijcai.2021/468)

**Abstract**:

For many Internet companies, it has been an important focus to improve user retention rate. To achieve this goal, we need to recommend proper services in order to meet the demands of users.  Unlike conventional click-through rate (CTR) estimation, there are lots of noise in the collected data when modeling retention, caused by two major issues: 1) implicit impression-revisit effect: users could revisit the APP even if they do not explicitly interact with the recommender system; 2) selection bias: recommender system suffers from selection bias caused by user's self-selection. To address the above challenges, we propose a novel method named UR-IPW (User Retention Modeling with Inverse Propensity Weighting), which 1) makes full use of both explicit and implicit interactions in the observed data. 2) models revisit rate estimation from a causal perspective accounting for the selection bias problem. The experiments on both offline and online environments from different scenarios demonstrate the superiority of UR-IPW over previous methods. To the best of our knowledge, this is the first work to model user retention by estimating the revisit rate from a causal perspective.

----

## [468] Neural Relation Inference for Multi-dimensional Temporal Point Processes via Message Passing Graph

**Authors**: *Yunhao Zhang, Junchi Yan*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/469](https://doi.org/10.24963/ijcai.2021/469)

**Abstract**:

Relation discovery for multi-dimensional temporal point processes (MTPP) has received increasing interest for its importance in prediction and interpretability of the underlying dynamics. Traditional statistical MTPP models like Hawkes Process have difficulty in capturing complex relation due to their limited parametric form of the intensity function. While recent neural-network-based models suffer poor interpretability. In this paper, we propose a neural relation inference model namely TPP-NRI. Given MTPP data, it adopts a variational inference framework to model the posterior relation of MTPP data for probabilistic estimation. Specifically, assuming the prior of the relation is known, the conditional probability of the MTPP conditional on a sampled relation is captured by a message passing graph neural network (GNN) based MTPP model. A variational distribution is introduced to approximate the true posterior. Experiments on synthetic and real-world data show that our model outperforms baseline methods on both inference capability and scalability for high-dimensional data.

----

## [469] Combining Tree Search and Action Prediction for State-of-the-Art Performance in DouDiZhu

**Authors**: *Yunsheng Zhang, Dong Yan, Bei Shi, Haobo Fu, Qiang Fu, Hang Su, Jun Zhu, Ning Chen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/470](https://doi.org/10.24963/ijcai.2021/470)

**Abstract**:

AlphaZero has achieved superhuman performance on various perfect-information games, such as chess, shogi and Go. However, directly applying AlphaZero to imperfect-information games (IIG) is infeasible, due to the fact that traditional MCTS methods cannot handle missing information of other players. Meanwhile, there have been several extensions of MCTS for IIGs, by implicitly or explicitly sampling a state of other players. But, due to the inability to handle private and public information well, the performance of these methods is not satisfactory. In this paper, we extend AlphaZero to multiplayer IIGs by developing a new MCTS method, Action-Prediction MCTS (AP-MCTS). In contrast to traditional MCTS extensions for IIGs, AP-MCTS first builds the search tree based on public information, adopts the policy-value network to generalize between hidden states, and finally predicts other players' actions directly. This design bypasses the inefficiency of sampling and the difficulty of predicting the state of other players.  We conduct extensive experiments on the popular 3-player poker game DouDiZhu to evaluate the performance of AP-MCTS combined with the framework AlphaZero. When playing against experienced human players, AP-MCTS achieved a 65.65\% winning rate, which is almost twice the human's winning rate. When comparing with state-of-the-art DouDiZhu AIs, the Elo rating of AP-MCTS is 50 to 200 higher than them. The ablation study shows that accurate action prediction is the key to AP-MCTS winning.

----

## [470] Uncertainty-Aware Few-Shot Image Classification

**Authors**: *Zhizheng Zhang, Cuiling Lan, Wenjun Zeng, Zhibo Chen, Shih-Fu Chang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/471](https://doi.org/10.24963/ijcai.2021/471)

**Abstract**:

Few-shot image classification learns to recognize new categories from limited labelled data. Metric learning based approaches have been widely investigated, where a query sample is classified by finding the nearest prototype from the support set based on their feature similarities. A neural network has different uncertainties on its calculated similarities of different pairs. Understanding and modeling the uncertainty on the similarity could promote the exploitation of limited samples in few-shot optimization. In this work, we propose Uncertainty-Aware Few-Shot framework for image classification by modeling uncertainty of the similarities of query-support pairs and performing uncertainty-aware optimization. Particularly, we exploit such uncertainty by converting observed similarities to probabilistic representations and incorporate them to the loss for more effective optimization. In order to jointly consider the similarities between a query and the prototypes in a support set, a graph-based model is utilized to estimate the uncertainty of the pairs. Extensive experiments show our proposed method brings significant improvements on top of a strong baseline and achieves the state-of-the-art performance.

----

## [471] Automatic Mixed-Precision Quantization Search of BERT

**Authors**: *Changsheng Zhao, Ting Hua, Yilin Shen, Qian Lou, Hongxia Jin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/472](https://doi.org/10.24963/ijcai.2021/472)

**Abstract**:

Pre-trained language models such as BERT have shown remarkable effectiveness in various natural language processing tasks. However, these models usually contain millions of parameters, which prevent them from the practical deployment on resource-constrained devices. Knowledge distillation, Weight pruning, and Quantization are known to be the main directions in  model compression. However, compact models obtained through knowledge distillation  may suffer from significant accuracy drop even for a relatively small compression ratio. On the other hand, there are only a few attempts based on quantization designed for natural language processing tasks, and they usually require manual setting on hyper-parameters. In this paper, we proposed an automatic mixed-precision quantization framework designed for BERT that can conduct quantization and pruning simultaneously. Specifically, our proposed method leverages Differentiable Neural Architecture Search to assign scale and precision for parameters in each sub-group automatically, and at the same pruning out redundant groups of parameters. Extensive evaluations on BERT downstream tasks reveal that our proposed method beats baselines by providing the same performance with much smaller model size.
We also show the possibility of obtaining the extremely light-weight model by combining our solution with orthogonal methods such as DistilBERT.

----

## [472] Graph Debiased Contrastive Learning with Joint Representation Clustering

**Authors**: *Han Zhao, Xu Yang, Zhenru Wang, Erkun Yang, Cheng Deng*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/473](https://doi.org/10.24963/ijcai.2021/473)

**Abstract**:

By contrasting positive-negative counterparts, graph contrastive learning has become a prominent technique for unsupervised graph representation learning. However, existing methods fail to consider the class information and will introduce false-negative samples in the random negative sampling, causing poor performance. To this end, we propose a graph debiased contrastive learning framework, which can jointly perform representation learning and clustering. Specifically, representations can be optimized by aligning with clustered class information, and simultaneously, the optimized representations can promote clustering, leading to more powerful representations and clustering results. More importantly, we randomly select negative samples from the clusters which are different from the positive sample's cluster. In this way, as the supervisory signals, the clustering results can be utilized to effectively decrease the false-negative samples. Extensive experiments on five datasets demonstrate that our method achieves new state-of-the-art results on graph clustering and classification tasks.

----

## [473] Uncertainty-aware Binary Neural Networks

**Authors**: *Junhe Zhao, Linlin Yang, Baochang Zhang, Guodong Guo, David S. Doermann*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/474](https://doi.org/10.24963/ijcai.2021/474)

**Abstract**:

Binary Neural Networks (BNN) are promising machine learning solutions for deployment on resource-limited devices.  Recent approaches to training BNNs have produced impressive results, but minimizing the drop in accuracy from full precision networks is still challenging. One reason is that conventional BNNs ignore the uncertainty caused by weights that are near zero, resulting in the instability or frequent flip while learning. In this work, we investigate the intrinsic uncertainty of vanishing near-zero weights, making the training vulnerable to instability. We introduce an uncertainty-aware  BNN (UaBNN) by leveraging  a new mapping function called certainty-sign (c-sign)  to reduce these weights' uncertainties. Our c-sign function is the first to train BNNs with a  decreasing uncertainty for binarization.  The approach leads to a  controlled  learning process for BNNs. We also introduce a simple but effective method to measure the uncertainty-based on a Gaussian function. Extensive experiments demonstrate that our method improves multiple BNN methods by maintaining stability  of  training, and achieves a higher performance over prior arts.

----

## [474] Few-Shot Partial-Label Learning

**Authors**: *Yunfeng Zhao, Guoxian Yu, Lei Liu, Zhongmin Yan, Lizhen Cui, Carlotta Domeniconi*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/475](https://doi.org/10.24963/ijcai.2021/475)

**Abstract**:

Partial-label learning (PLL) generally focuses on inducing a noise-tolerant multi-class classifier by training on overly-annotated samples, each of which is annotated with a set of labels, but only one is the valid label. A basic promise of existing PLL solutions is that there are sufficient partial-label (PL) samples for training. However, it is more common than not to have just few PL samples at hand when dealing with new tasks. Furthermore, existing few-shot learning algorithms assume precise labels of the support set; as such,  irrelevant labels may seriously mislead the meta-learner  and thus lead to a compromised performance. How to enable PLL under a few-shot learning setting is an important problem, but not yet well studied. In this paper, we introduce an approach called FsPLL (Few-shot PLL).  FsPLL first performs adaptive distance metric learning by an embedding network and rectifying prototypes on the tasks previously encountered. Next, it calculates the prototype of each class of a new task in the embedding network. An unseen example can then be classified via its distance to each prototype. Experimental results on widely-used few-shot datasets demonstrate that our FsPLL can achieve a superior performance than the state-of-the-art methods, and it needs fewer samples for quickly adapting to new tasks.

----

## [475] Non-decreasing Quantile Function Network with Efficient Exploration for Distributional Reinforcement Learning

**Authors**: *Fan Zhou, Zhoufan Zhu, Qi Kuang, Liwen Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/476](https://doi.org/10.24963/ijcai.2021/476)

**Abstract**:

Although distributional reinforcement learning (DRL) has been widely examined in the past few years, there are two open questions people are still trying to address. One is how to ensure the validity of the learned quantile function, the other is how to efficiently utilize the distribution information. This paper attempts to provide some new perspectives to encourage the future in-depth studies in these two fields. We first propose a non-decreasing quantile function network (NDQFN) to guarantee the monotonicity of the obtained quantile estimates and then design a general exploration framework called distributional prediction error (DPE) for DRL which utilizes the entire distribution of the quantile function. In this paper, we not only discuss the theoretical necessity of our method but also show the performance gain it achieves in practice by comparing with some competitors on Atari 2600 Games especially in some hard-explored games.

----

## [476] Multi-Target Invisibly Trojaned Networks for Visual Recognition and Detection

**Authors**: *Xinzhe Zhou, Wenhao Jiang, Sheng Qi, Yadong Mu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/477](https://doi.org/10.24963/ijcai.2021/477)

**Abstract**:

Visual backdoor attack is a recently-emerging task which aims to implant trojans in a deep neural model. A trojaned model responds to a trojan-invoking trigger in a fully predictable manner while functioning normally otherwise. As a key motivating fact to this work, most triggers adopted in existing methods, such as a learned patterned block that overlays a benigh image, can be easily noticed by human. In this work, we take image recognition and detection as the demonstration tasks, building trojaned networks that are significantly less human-perceptible and can simultaneously attack multiple targets in an image. The main technical contributions are two-folds: first, under a relaxed attack mode, we formulate trigger embedding as an image steganography-and-steganalysis problem that conceals a secret image in another image in a decipherable and almost invisible way. In specific, a variable number of different triggers can be encoded into a same secret image and fed to an encoder module that does steganography. Secondly, we propose a generic split-and-merge scheme for training a trojaned model. Neurons are split into two sets, trained either for normal image recognition / detection or trojaning the model. To merge them, we novelly propose to hide trojan neurons within the nullspace of the normal ones, such that the two sets do not interfere with each other and the resultant model exhibits similar parameter statistics to a clean model. Comprehensive experiments are conducted on the datasets PASCAL VOC and Microsoft COCO (for detection) and a subset of ImageNet (for recognition). All results clearly demonstrate the effectiveness of our proposed visual trojan method.

----

## [477] AutoReCon: Neural Architecture Search-based Reconstruction for Data-free Compression

**Authors**: *Baozhou Zhu, H. Peter Hofstee, Johan Peltenburg, Jinho Lee, Zaid Al-Ars*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/478](https://doi.org/10.24963/ijcai.2021/478)

**Abstract**:

Data-free compression raises a new challenge because the original training dataset for a pre-trained model to be compressed is not available due to privacy or transmission issues. Thus, a common approach is to compute a reconstructed training dataset before compression. The current reconstruction methods compute the reconstructed training dataset with a generator by exploiting information from the pre-trained model. However, current reconstruction methods focus on extracting more information from the pre-trained model but do not leverage network engineering. This work is the first to consider network engineering as an approach to design the reconstruction method. Specifically, we propose the AutoReCon method, which is a neural architecture search-based reconstruction method. In the proposed AutoReCon method, the generator architecture is designed automatically given the pre-trained model for reconstruction. Experimental results show that using generators discovered by the AutoRecon method always improve the performance of data-free compression.

----

## [478] You Get What You Sow: High Fidelity Image Synthesis with a Single Pretrained Network

**Authors**: *Kefeng Zhu, Peilin Tong, Hongwei Kan, Rengang Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/479](https://doi.org/10.24963/ijcai.2021/479)

**Abstract**:

State-of-the-art image synthesis methods are mostly based on generative adversarial networks and require large dataset and extensive training. Although the model-inversion-oriented branch of methods eliminate the training requirement, the quality of the resulting image tends to be limited due to the lack of sufficient natural and class-specific information. In this paper, we introduce a novel strategy for high fidelity image synthesis with a single pretrained classification network. The strategy includes a class-conditional natural regularization design and a corresponding metadata collecting procedure for different scenarios. We show that our method can synthesize high quality natural images that closely follow the features of one or more given seed images. Moreover, our method achieves surprisingly decent results in the task of sketch-based image synthesis without training. Finally, our method further improves the performance in terms of accuracy and efficiency in the data-free knowledge distillation task.

----

## [479] MapGo: Model-Assisted Policy Optimization for Goal-Oriented Tasks

**Authors**: *Menghui Zhu, Minghuan Liu, Jian Shen, Zhicheng Zhang, Sheng Chen, Weinan Zhang, Deheng Ye, Yong Yu, Qiang Fu, Wei Yang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/480](https://doi.org/10.24963/ijcai.2021/480)

**Abstract**:

In Goal-oriented Reinforcement learning, relabeling the raw goals in past experience to provide agents with hindsight ability is a major solution to the reward sparsity problem. In this paper, to enhance the diversity of relabeled goals, we develop FGI (Foresight Goal Inference), a new relabeling strategy that relabels the goals by looking into the future with a learned dynamics model. Besides, to improve sample efficiency, we propose to use the dynamics model to generate simulated trajectories for policy training. By integrating these two improvements, we introduce the MapGo framework (Model-Assisted Policy optimization for Goal-oriented tasks). In our experiments, we first show the effectiveness of the FGI strategy compared with the hindsight one, and then show that the MapGo framework achieves higher sample efficiency when compared to model-free baselines on a set of complicated tasks.

----

## [480] Toward Optimal Solution for the Context-Attentive Bandit Problem

**Authors**: *Djallel Bouneffouf, Raphaël Féraud, Sohini Upadhyay, Irina Rish, Yasaman Khazaeni*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/481](https://doi.org/10.24963/ijcai.2021/481)

**Abstract**:

In various recommender system applications, from medical diagnosis to dialog systems, due to observation costs only a small subset of a potentially large number of context variables can be observed at each iteration; however, the agent has a freedom to choose which variables to observe.  In this paper, we analyze and  extend an online learning framework known as Context-Attentive Bandit, We derive a novel algorithm, called Context-Attentive Thompson Sampling (CATS), which builds upon the Linear Thompson Sampling approach,  adapting it to Context-Attentive Bandit  setting.  We provide a theoretical regret analysis and an extensive empirical evaluation demonstrating  advantages of the proposed approach over several baseline methods  on a variety of real-life datasets.

----

## [481] Sample Efficient Decentralized Stochastic Frank-Wolfe Methods for Continuous DR-Submodular Maximization

**Authors**: *Hongchang Gao, Hanzi Xu, Slobodan Vucetic*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/482](https://doi.org/10.24963/ijcai.2021/482)

**Abstract**:

Continuous DR-submodular maximization is an important machine learning problem, which covers numerous popular applications.  With the emergence of large-scale distributed data, developing efficient algorithms for the continuous DR-submodular maximization, such as the decentralized Frank-Wolfe method, became an important challenge. However, existing decentralized Frank-Wolfe methods for this kind of problem have the sample complexity of $\mathcal{O}(1/\epsilon^3)$, incurring a large computational overhead. In this paper, we propose two novel sample efficient decentralized Frank-Wolfe methods to address this challenge. Our theoretical results demonstrate that the sample complexity of the two proposed methods is $\mathcal{O}(1/\epsilon^2)$, which is better than $\mathcal{O}(1/\epsilon^3)$ of the existing methods. As far as we know, this is the first published result achieving such a favorable sample complexity. Extensive experimental results confirm the effectiveness of the proposed methods.

----

## [482] Self-Guided Community Detection on Networks with Missing Edges

**Authors**: *Dongxiao He, Shuai Li, Di Jin, Pengfei Jiao, Yuxiao Huang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/483](https://doi.org/10.24963/ijcai.2021/483)

**Abstract**:

The vast majority of community detection algorithms assume that the networks are totally observed. However, in reality many networks cannot be fully observed. On such network is edges-missing network, where some relationships (edges) between two entities are missing. Recently, several works have been proposed to solve this problem by combining link prediction and community detection in a two-stage method or in a unified framework. However, the goal of link prediction, which is to predict as many correct edges as possible, is not consistent with the requirement for predicting the important edges for discovering community structure on edges-missing networks. Thus, combining link prediction and community detection cannot work very well in terms of detecting community structure for edges-missing network. In this paper, we propose a community self-guided generative model which jointly completes the edges-missing network and identifies communities. In our new model, completing missing edges and identifying communities are not isolated but closely intertwined. Furthermore, we developed an effective model inference method that combines a nested Expectation-Maximization (EM) algorithm and Metropolis-Hastings Sampling. Extensive experiments on real-world edges-missing networks show that our model can effectively detect community structures while completing missing edges.

----

## [483] Two-Sided Wasserstein Procrustes Analysis

**Authors**: *Kun Jin, Chaoyue Liu, Cathy Xia*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/484](https://doi.org/10.24963/ijcai.2021/484)

**Abstract**:

Learning correspondence between sets of objects is a key component in many machine learning tasks.Recently, optimal Transport (OT) has been successfully applied to such correspondence problems and it  is  appealing  as  a  fully  unsupervised  approach. However,  OT  requires  pairwise  instances  be  directly comparable in a common metric space. This limits its applicability when feature spaces are of different dimensions or not directly comparable. In addition,  OT only focuses on pairwise correspondence without sensing global transformations.  To address these challenges, we propose a new method to jointly learn the optimal coupling between twosets,  and  the  optimal  transformations  (e.g.   rotation,  projection and scaling) of each set based on a two-sided Wassertein Procrustes analysis (TWP). Since the joint problem is a non-convex optimization problem, we present a reformulation that renders the problem component-wise convex. We then propose a novel algorithm to solve the problem harnessing a Gaussâ€“Seidel method. We further present competitive  results  of TWP  on  various  applicationscompared with state-of-the-art methods.

----

## [484] Solving Math Word Problems with Teacher Supervision

**Authors**: *Zhenwen Liang, Xiangliang Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/485](https://doi.org/10.24963/ijcai.2021/485)

**Abstract**:

Math word problems (MWPs) have been recently addressed with Seq2Seq models by `translating' math problems described in natural language to a mathematical expression, following a typical encoder-decoder structure. Although effective in solving classical math problems, these models fail when a subtle variation is applied to the word expression of a math problem, and leads to a remarkably different answer. We find the failure is because MWPs with different answers but similar math formula expression are encoded closely in the latent space. We thus designed a teacher module to make the MWP encoding vector match the correct solution and disaccord from the wrong solutions, which are manipulated from the correct solution. Experimental results on two benchmark MWPs datasets verified that our proposed solution outperforms the state-of-the-art models.

----

## [485] Collaborative Graph Learning with Auxiliary Text for Temporal Event Prediction in Healthcare

**Authors**: *Chang Lu, Chandan K. Reddy, Prithwish Chakraborty, Samantha Kleinberg, Yue Ning*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/486](https://doi.org/10.24963/ijcai.2021/486)

**Abstract**:

Accurate and explainable health event predictions are becoming crucial for healthcare providers to develop care plans for patients. The availability of electronic health records (EHR) has enabled machine learning advances in providing these predictions. However, many deep-learning-based methods are not satisfactory in solving several key challenges: 1) effectively utilizing disease domain knowledge; 2) collaboratively learning representations of patients and diseases; and 3) incorporating unstructured features. To address these issues, we propose a collaborative graph learning model to explore patient-disease interactions and medical domain knowledge. Our solution is able to capture structural features of both patients and diseases. The proposed model also utilizes unstructured text data by employing an attention manipulating strategy and then integrates attentive text features into a sequential learning process. We conduct extensive experiments on two important healthcare problems to show the competitive prediction performance of the proposed method compared with various state-of-the-art models. We also confirm the effectiveness of learned representations and model interpretability by a set of ablation and case studies.

----

## [486] MDNN: A Multimodal Deep Neural Network for Predicting Drug-Drug Interaction Events

**Authors**: *Tengfei Lyu, Jianliang Gao, Ling Tian, Zhao Li, Peng Zhang, Ji Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/487](https://doi.org/10.24963/ijcai.2021/487)

**Abstract**:

The interaction of multiple drugs could lead to serious events, which causes injuries and huge medical costs. Accurate prediction of drug-drug interaction (DDI) events can help clinicians make effective decisions and establish appropriate therapy programs. Recently, many AI-based techniques have been proposed for predicting DDI associated events. However, most existing methods pay less attention to the potential correlations between DDI events and other multimodal data such as targets and enzymes. To address this problem, we propose a Multimodal Deep Neural Network (MDNN) for DDI events prediction. In MDNN, we design a two-pathway framework including drug knowledge graph (DKG) based pathway and heterogeneous feature (HF) based pathway to obtain drug multimodal representations. Finally, a multimodal fusion neural layer is designed to explore the complementary among the drug multimodal representations. We conduct extensive experiments on real-world dataset. The results show that MDNN can accurately predict DDI events and outperform the state-of-the-art models.

----

## [487] SPADE: A Semi-supervised Probabilistic Approach for Detecting Errors in Tables

**Authors**: *Minh Pham, Craig A. Knoblock, Muhao Chen, Binh Vu, Jay Pujara*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/488](https://doi.org/10.24963/ijcai.2021/488)

**Abstract**:

Error detection is one of the most important steps in data cleaning and usually requires extensive human interaction to ensure quality. Existing supervised methods in error detection require a significant amount of training data while unsupervised methods rely on fixed inductive biases, which are usually hard to generalize, to solve the problem. In this paper, we present SPADE, a novel semi-supervised probabilistic approach for error detection. SPADE  introduces a novel probabilistic active learning model, where the system suggests examples to be labeled based on the agreements between user labels and indicative signals, which are designed to capture potential errors.  SPADE uses a two-phase data augmentation process to enrich a dataset before training a deep learning classifier to detect unlabeled errors. In our evaluation, SPADE achieves an average F1-score of 0.91 over five datasets and yields a 10% improvement compared with the state-of-the-art systems.

----

## [488] TEC: A Time Evolving Contextual Graph Model for Speaker State Analysis in Political Debates

**Authors**: *Ramit Sawhney, Shivam Agarwal, Arnav Wadhwa, Rajiv Ratn Shah*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/489](https://doi.org/10.24963/ijcai.2021/489)

**Abstract**:

Political discourses provide a forum for representatives to express their opinions and contribute towards policy making. 
Analyzing these discussions is crucial for recognizing possible delegates and making better voting choices in an independent nation. 
A politician's vote on a proposition is usually associated with their past discourses and impacted by cohesion forces in political parties.
We focus on predicting a speaker's vote on a bill by augmenting linguistic models with temporal and cohesion contexts.
We propose TEC, a time evolving graph based model that jointly employs links between motions, speakers, and temporal politician states. 
TEC outperforms competitive models, illustrating the benefit of temporal and contextual signals for predicting a politician's stance.

----

## [489] Adaptive Residue-wise Profile Fusion for Low Homologous Protein Secondary Structure Prediction Using External Knowledge

**Authors**: *Qin Wang, Jun Wei, Boyuan Wang, Zhen Li, Sheng Wang, Shuguang Cui*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/490](https://doi.org/10.24963/ijcai.2021/490)

**Abstract**:

Protein secondary structure prediction (PSSP) is essential for protein function analysis. However, for low homologous proteins, the PSSP suffers from insufficient input features. In this paper, we explicitly import external self-supervised knowledge for low homologous PSSP under the guidance of residue-wise (amino acid wise) profile fusion. In practice, we firstly demonstrate the superiority of profile over Position-Specific Scoring Matrix (PSSM) for low homologous PSSP. Based on this observation, we introduce the novel self-supervised BERT features as the pseudo profile, which implicitly involves the residue distribution in all native discovered sequences as the complementary features. Furthermore, a novel residue-wise attention is specially designed to adaptively fuse different features (i.e., original low-quality profile, BERT based pseudo profile), which not only takes full advantage of each feature but also avoids noise disturbance. Besides, the feature consistency loss is proposed to accelerate the model learning from multiple semantic levels. Extensive experiments confirm that our method outperforms state-of-the-arts (i.e., 4.7% for extremely low homologous cases on BC40 dataset).

----

## [490] Ordering-Based Causal Discovery with Reinforcement Learning

**Authors**: *Xiaoqiang Wang, Yali Du, Shengyu Zhu, Liangjun Ke, Zhitang Chen, Jianye Hao, Jun Wang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/491](https://doi.org/10.24963/ijcai.2021/491)

**Abstract**:

It is a long-standing question to discover causal relations  among a set of variables in many empirical sciences. 
Recently, Reinforcement Learning (RL)  has achieved promising results in causal discovery from observational data. However, searching the space of directed graphs and enforcing acyclicity by implicit penalties tend to be inefficient and restrict the existing RL-based method to small scale problems. 
In this work, we propose a novel RL-based approach for causal discovery, by incorporating RL into the ordering-based paradigm.
Specifically, we  formulate the ordering search problem as a multi-step Markov decision process, implement the ordering generating process with an encoder-decoder architecture, and finally use 
RL to optimize the proposed model based on the reward mechanisms designed for each ordering. 
A generated ordering would then be processed using variable selection to obtain the final causal graph. 
We  analyze the consistency and computational complexity of  the proposed method, and empirically show that a pretrained model can be exploited to accelerate training. Experimental results on both synthetic and real data sets shows  that the proposed method achieves a much improved performance over existing RL-based method.

----

## [491] Boosting Offline Reinforcement Learning with Residual Generative Modeling

**Authors**: *Hua Wei, Deheng Ye, Zhao Liu, Hao Wu, Bo Yuan, Qiang Fu, Wei Yang, Zhenhui Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/492](https://doi.org/10.24963/ijcai.2021/492)

**Abstract**:

Offline reinforcement learning (RL) tries to learn the near-optimal policy with recorded offline experience without online exploration.Current offline RL research includes: 1) generative modeling, i.e., approximating a policy using fixed data; and 2) learning the state-action value function. While most research focuses on the state-action function part through reducing the bootstrapping error in value function approximation induced by the distribution shift of training data, the effects of error propagation in generative modeling have been neglected. In this paper, we analyze the error in generative modeling. We propose AQL (action-conditioned Q-learning), a residual generative model to reduce policy approximation error for offline RL. We show that our method can learn more accurate policy approximations in different benchmark datasets. In addition, we show that the proposed offline RL method can learn more competitive AI agents in complex control tasks under the multiplayer online battle arena (MOBA) game, Honor of Kings.

----

## [492] Multi-series Time-aware Sequence Partitioning for Disease Progression Modeling

**Authors**: *Xi Yang, Yuan Zhang, Min Chi*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/493](https://doi.org/10.24963/ijcai.2021/493)

**Abstract**:

Electronic healthcare records (EHRs) are comprehensive longitudinal collections of patient data that play a critical role in modeling the disease progression to facilitate clinical decision-making. Based on EHRs, in this work, we focus on sepsis -- a broad syndrome that can develop from nearly all types of infections (e.g., influenza, pneumonia). The symptoms of sepsis, such as elevated heart rate, fever, and shortness of breath, are vague and common to other illnesses, making the modeling of its progression extremely challenging. Motivated by the recent success of a novel subsequence clustering approach: Toeplitz Inverse Covariance-based Clustering (TICC), we model the sepsis progression as a subsequence partitioning problem and propose a Multi-series Time-aware TICC (MT-TICC), which incorporates multi-series nature and irregular time intervals of EHRs. The effectiveness of MT-TICC is first validated via a case study using a real-world hand gesture dataset with ground-truth labels. Then we further apply it for sepsis progression modeling using EHRs. The results suggest that MT-TICC can significantly outperform competitive baseline models, including the TICC. More importantly, it unveils interpretable patterns, which sheds some light on better understanding the sepsis progression.

----

## [493] A Rule Mining-based Advanced Persistent Threats Detection System

**Authors**: *Sidahmed Benabderrahmane, Ghita Berrada, James Cheney, Petko Valtchev*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/494](https://doi.org/10.24963/ijcai.2021/494)

**Abstract**:

Advanced persistent threats (APT) are stealthy cyber-attacks that are aimed at stealing valuable information from target organizations and tend to extend in time. 
Blocking all APTs is impossible, security experts caution, hence the importance of research on early detection and damage limitation. 
Whole-system provenance-tracking and provenance trace mining are considered promising as they can help find causal relationships between activities and flag suspicious event sequences as they occur.
We introduce an unsupervised method that exploits OS-independent features reflecting process activity to detect realistic APT-like attacks from provenance traces. 
Anomalous processes are ranked using both frequent and rare event associations learned from traces. Results are then presented as implications which, since interpretable, help leverage causality in explaining the detected anomalies. When evaluated on Transparent Computing program datasets (DARPA), our method outperformed competing approaches.

----

## [494] Electrocardio Panorama: Synthesizing New ECG views with Self-supervision

**Authors**: *Jintai Chen, Xiangshang Zheng, Hongyun Yu, Danny Z. Chen, Jian Wu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/495](https://doi.org/10.24963/ijcai.2021/495)

**Abstract**:

Multi-lead electrocardiogram (ECG) provides clinical information of heartbeats from several fixed viewpoints determined by the lead positioning. However, it is often not satisfactory to visualize ECG signals in these fixed and limited views, as some clinically useful information is represented only from a few specific ECG viewpoints. For the first time, we propose a new concept, Electrocardio Panorama, which allows visualizing ECG signals from any queried viewpoints. To build Electrocardio Panorama, we assume that an underlying electrocardio field exists, representing locations, magnitudes, and directions of ECG signals. We present a Neural electrocardio field Network (Nef-Net), which first predicts the electrocardio field representation by using a sparse set of one or few input ECG views and then synthesizes Electrocardio Panorama based on the predicted representations. Specially, to better disentangle electrocardio field information from viewpoint biases, a new Angular Encoding is proposed to process viewpoint angles. Also, we propose a self-supervised learning approach called Standin Learning, which helps model the electrocardio field without direct supervision. Further, with very few modifications, Nef-Net can synthesize ECG signals from scratch. Experiments verify that our Nef-Net performs well on Electrocardio Panorama synthesis, and outperforms the previous work on the auxiliary tasks (ECG view transformation and ECG synthesis from scratch). The codes and the division labels of cardiac cycles and ECG deflections on Tianchi ECG and PTB datasets are available at https://github.com/WhatAShot/Electrocardio-Panorama.

----

## [495] A Novel Sequence-to-Subgraph Framework for Diagnosis Classification

**Authors**: *Jun Chen, Quan Yuan, Chao Lu, Haifeng Huang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/496](https://doi.org/10.24963/ijcai.2021/496)

**Abstract**:

Text-based diagnosis classification is a critical problem in AI-enabled healthcare studies, which assists clinicians in making correct decision and lowering the rate of diagnostic errors. Previous studies follow the routine of sequence based deep learning models in NLP literature to deal with clinical notes. However, recent studies find that structural information is important in clinical contents that greatly impacts the predictions. In this paper, a novel sequence-to-subgraph framework is introduced to process clinical texts for classification, which changes the paradigm of managing texts. Moreover, a new classification model under the framework is proposed that incorporates subgraph convolutional network and hierarchical diagnostic attentive network to extract the layered structural features of clinical texts. The evaluation conducted on both the real-world English and Chinese datasets shows that the proposed method outperforms the state-of-the-art deep learning based diagnosis classification models.

----

## [496] Parallel Subtrajectory Alignment over Massive-Scale Trajectory Data

**Authors**: *Lisi Chen, Shuo Shang, Shanshan Feng, Panos Kalnis*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/497](https://doi.org/10.24963/ijcai.2021/497)

**Abstract**:

We study the problem of subtrajectory alignment over massive-scale trajectory data. Given a collection of trajectories, a subtrajectory alignment query returns new targeted trajectories by splitting and aligning existing trajectories. The resulting functionality targets a range of applications, including trajectory data analysis, route planning and recommendation, ridesharing, and general location-based services. To enable efficient and effective subtrajectory alignment computation, we propose a novel search algorithm and filtering techniques that enable the use of the parallel processing capabilities of modern processors. Experiments with large trajectory datasets are conducted for evaluating the performance of our proposal. The results show that our solution to the subtrajectory alignment problem can generate high-quality results and are capable of achieving high efficiency and scalability.

----

## [497] TrafficStream: A Streaming Traffic Flow Forecasting Framework Based on Graph Neural Networks and Continual Learning

**Authors**: *Xu Chen, Junshan Wang, Kunqing Xie*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/498](https://doi.org/10.24963/ijcai.2021/498)

**Abstract**:

With the rapid growth of traffic sensors deployed, a massive amount of traffic flow data are collected, revealing the long-term evolution of traffic flows and the gradual expansion of traffic networks. How to accurately forecasting these traffic flow attracts the attention of researchers as it is of great significance for improving the efficiency of transportation systems. However, existing methods mainly focus on the spatial-temporal correlation of static networks, leaving the problem of efficiently learning models on networks with expansion and evolving patterns less studied. To tackle this problem, we propose a Streaming Traffic Flow Forecasting Framework, TrafficStream,  based on Graph Neural Networks (GNNs) and Continual Learning (CL), achieving accurate predictions and high efficiency. Firstly, we design a traffic pattern fusion method, cleverly integrating the new patterns that emerged during the long-term period into the model. A JS-divergence-based algorithm is proposed to mine new traffic patterns. Secondly, we introduce CL to consolidate the knowledge learned previously and transfer them to the current model. Specifically, we adopt two strategies: historical data replay and parameter smoothing. We construct a streaming traffic data set to verify the efficiency and effectiveness of our model. Extensive experiments demonstrate its excellent potential to extract traffic patterns with high efficiency on long-term streaming network scene. The source code is available at https://github.com/AprLie/TrafficStream.

----

## [498] Predictive Job Scheduling under Uncertain Constraints in Cloud Computing

**Authors**: *Hang Dong, Boshi Wang, Bo Qiao, Wenqian Xing, Chuan Luo, Si Qin, Qingwei Lin, Dongmei Zhang, Gurpreet Virdi, Thomas Moscibroda*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/499](https://doi.org/10.24963/ijcai.2021/499)

**Abstract**:

Capacity management has always been a great challenge for cloud platforms due to massive, heterogeneous on-demand instances running at different times. To better plan the capacity for the whole platform, a class of cloud computing instances have been released to collect computing demands beforehand. To use such instances, users are allowed to submit jobs to run for a pre-specified uninterrupted duration in a flexible range of time in the future with a discount compared to the normal on-demand instances. Proactively scheduling those pre-collected job requests considering the capacity status over the platform can greatly help balance the computing workloads along time. In this work, we formulate the scheduling problem for these pre-collected job requests under uncertain available capacity as a Prediction + Optimization problem with uncertainty in constraints, and propose an effective algorithm called Controlling under Uncertain Constraints (CUC), where the predicted capacity guides the optimization of job scheduling and job scheduling results are leveraged to improve the prediction of capacity through Bayesian optimization. The proposed formulation and solution are commonly applicable for proactively scheduling problems in cloud computing. Our extensive experiments on three public, industrial datasets shows that CUC has great potential for supporting high reliability in cloud platforms.

----

## [499] Fine-tuning Is Not Enough: A Simple yet Effective Watermark Removal Attack for DNN Models

**Authors**: *Shangwei Guo, Tianwei Zhang, Han Qiu, Yi Zeng, Tao Xiang, Yang Liu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/500](https://doi.org/10.24963/ijcai.2021/500)

**Abstract**:

Watermarking has become the tendency in protecting the intellectual property of DNN models. Recent works, from the adversary's perspective, attempted to subvert watermarking mechanisms by designing watermark removal attacks. However, these attacks mainly adopted sophisticated fine-tuning techniques, which have certain fatal drawbacks or unrealistic assumptions. In this paper, we propose a novel watermark removal attack from a different perspective. Instead of just fine-tuning the watermarked models, we design a simple yet powerful transformation algorithm by combining imperceptible pattern embedding and spatial-level transformations, which can effectively and blindly destroy the memorization of watermarked models to the watermark samples. We also introduce a lightweight fine-tuning strategy to preserve the model performance. Our solution requires much less resource or knowledge about the watermarking scheme than prior works. Extensive experimental results indicate that our attack can bypass state-of-the-art watermarking solutions with very high success rates. Based on our attack, we propose watermark augmentation techniques to enhance the robustness of existing watermarks.

----

## [500] Dynamic Lane Traffic Signal Control with Group Attention and Multi-Timescale Reinforcement Learning

**Authors**: *Qize Jiang, Jingze Li, Weiwei Sun, Baihua Zheng*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/501](https://doi.org/10.24963/ijcai.2021/501)

**Abstract**:

Traffic signal control has achieved significant success with the development of reinforcement learning. However, existing works mainly focus on intersections with normal lanes with fixed outgoing directions. It is noticed that some intersections actually implement dynamic lanes, in addition to normal lanes, to adjust the outgoing directions dynamically. Existing methods fail to coordinate the control of traffic signal and that of dynamic lanes effectively. In addition, they lack proper structures and learning algorithms to make full use of traffic flow prediction, which is essential to set the proper directions for dynamic lanes. Motivated by the ineffectiveness of existing approaches when controlling the traffic signal and dynamic lanes simultaneously, we propose a new method, namely MT-GAD, in this paper. It uses a group attention structure to reduce the number of required parameters and to achieve a better generalizability, and uses multi-timescale model training to learn proper strategy that could best control both the traffic signal and the dynamic lanes. The experiments on real datasets demonstrate that MT-GAD outperforms existing approaches significantly.

----

## [501] Differentially Private Correlation Alignment for Domain Adaptation

**Authors**: *Kaizhong Jin, Xiang Cheng, Jiaxi Yang, Kaiyuan Shen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/502](https://doi.org/10.24963/ijcai.2021/502)

**Abstract**:

Domain adaptation solves a learning problem in a target domain by utilizing the training data in a different but related source domain. As a simple and efficient method for domain adaptation, correlation alignment transforms the distribution of the source domain by utilizing the covariance matrix of the target domain, such that a model trained on the transformed source data can be applied to the target data. However, when source and target domains come from different institutes, exchanging information between the two domains might pose a potential privacy risk. In this paper, for the first time, we propose a differentially private correlation alignment approach for domain adaptation called PRIMA, which can provide privacy guarantees for both the source and target data. In PRIMA, to relieve the performance degradation caused by perturbing the covariance matrix in high dimensional setting, we present a random subspace ensemble based covariance estimation method which splits the feature spaces of source and target data into several low dimensional subspaces. Moreover, since perturbing the covariance matrix may destroy its positive semi-definiteness, we develop a shrinking based method for the recovery of positive semi-definiteness of the covariance matrix. Experimental results on standard benchmark datasets confirm the effectiveness of our approach.

----

## [502] Traffic Congestion Alleviation over Dynamic Road Networks: Continuous Optimal Route Combination for Trip Query Streams

**Authors**: *Ke Li, Lisi Chen, Shuo Shang, Panos Kalnis, Bin Yao*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/503](https://doi.org/10.24963/ijcai.2021/503)

**Abstract**:

Route planning and recommendation have attracted much attention for decades. In this paper, we study a continuous optimal route combination problem: Given a dynamic road network and a stream of trip queries, we continuously find an optimal route combination for each new query batch over the query stream such that the total travel time for all routes is minimized. Each route corresponds to a planning result for a particular trip query in the current query batch. Our problem targets a variety of applications, including traffic-flow management, real-time route planning and continuous congestion prevention. The exact algorithm bears exponential time complexity and is computationally prohibitive for application scenarios in dynamic traffic networks.  To address this problem, a self-aware batch processing algorithm is developed in this paper. Extensive experiments offer insight into the accuracy and efficiency of our proposed algorithms.

----

## [503] CFR-MIX: Solving Imperfect Information Extensive-Form Games with Combinatorial Action Space

**Authors**: *Shuxin Li, Youzhi Zhang, Xinrun Wang, Wanqi Xue, Bo An*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/504](https://doi.org/10.24963/ijcai.2021/504)

**Abstract**:

In many real-world scenarios, a team of agents must coordinate with each other to compete against an opponent. The challenge of solving this type of game is that the team's joint action space grows exponentially with the number of agents, which results in the inefficiency of the existing algorithms, e.g., Counterfactual Regret Minimization (CFR). To address this problem, we propose a new framework of CFR: CFR-MIX. Firstly, we propose a new strategy representation that represents a joint action strategy using individual strategies of all agents and a consistency relationship to maintain the cooperation between agents. To compute the equilibrium with individual strategies under the CFR framework, we transform the consistency relationship between strategies to the consistency relationship between the cumulative regret values. Furthermore, we propose a novel decomposition method over cumulative regret values to guarantee the consistency relationship between the cumulative regret values. Finally, we introduce our new algorithm CFR-MIX which employs a mixing layer to estimate cumulative regret values of joint actions as a non-linear combination of cumulative regret values of individual actions. Experimental results show that CFR-MIX outperforms existing algorithms on various games significantly.

----

## [504] Online Credit Payment Fraud Detection via Structure-Aware Hierarchical Recurrent Neural Network

**Authors**: *Wangli Lin, Li Sun, Qiwei Zhong, Can Liu, Jinghua Feng, Xiang Ao, Hao Yang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/505](https://doi.org/10.24963/ijcai.2021/505)

**Abstract**:

Online credit payment fraud detection plays a critical role in financial institutions due to the growing volume of fraudulent transactions. Recently, researchers have shown an increased interest in capturing usersâ€™ dynamic and evolving fraudulent tendencies from their behavior sequences. However, most existing methodologies for sequential modeling overlook the intrinsic structure information of web pages. In this paper, we adopt multi-scale behavior sequence generated from different granularities of web page structures and propose a model named SAH-RNN to consume the multi-scale behavior sequence for online payment fraud detection. The SAH-RNN has stacked RNN layers in which upper layers modeling for compendious behaviors are updated less frequently and receive the summarized representations from lower layers. A dual attention is devised to capture the impacts on both sequential information within the same sequence and structural information among different granularity of web pages. Experimental results on a large-scale real-world transaction dataset from Alibaba show that our proposed model outperforms state-of-the-art models. The code is available at https://github.com/WangliLin/SAH-RNN.

----

## [505] Learning Unknown from Correlations: Graph Neural Network for Inter-novel-protein Interaction Prediction

**Authors**: *Guofeng Lv, Zhiqiang Hu, Yanguang Bi, Shaoting Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/506](https://doi.org/10.24963/ijcai.2021/506)

**Abstract**:

The study of multi-type Protein-Protein Interaction (PPI) is fundamental for understanding biological processes from a systematic perspective and revealing disease mechanisms. Existing methods suffer from significant performance degradation when tested in unseen dataset. In this paper, we investigate the problem and find that it is mainly attributed to the poor performance for inter-novel-protein interaction prediction. However, current evaluations overlook the inter-novel-protein interactions, and thus fail to give an instructive assessment. As a result, we propose to address the problem from both the evaluation and the methodology. Firstly, we design a new evaluation framework that fully respects the inter-novel-protein interactions and gives consistent assessment across datasets. Secondly, we argue that correlations between proteins must provide useful information for analysis of novel proteins, and based on this, we propose a graph neural network based method (GNN-PPI) for better inter-novel-protein interaction prediction. Experimental results on real-world datasets of different scales demonstrate that GNN-PPI significantly outperforms state-of-the-art PPI prediction methods, especially for the inter-novel-protein interaction prediction.

----

## [506] Adapting Meta Knowledge with Heterogeneous Information Network for COVID-19 Themed Malicious Repository Detection

**Authors**: *Yiyue Qian, Yiming Zhang, Yanfang Ye, Chuxu Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/507](https://doi.org/10.24963/ijcai.2021/507)

**Abstract**:

As cyberattacks caused by malware have proliferated during the pandemic, building an automatic system to detect COVID-19 themed malware in social coding platforms is in urgent need. The existing methods mainly rely on file content analysis while ignoring structured information among entities in social coding platforms. Additionally, they usually require sufficient data for model training, impairing their performances over cases with limited data which is common in reality. To address these challenges, we develop Meta-AHIN, a novel model for COVID-19 themed malicious repository detection in GitHub. In Meta-AHIN, we first construct an attributed heterogeneous information network (AHIN) to model the code content and social coding properties in GitHub; and then we exploit attention-based graph convolutional neural network (AGCN) to learn repository embeddings and present a meta-learning framework for model optimization. To utilize unlabeled information in AHIN and to consider task influence of different types of repositories, we further incorporate node attribute-based self-supervised module and task-aware attention weight into AGCN and meta-learning respectively. Extensive experiments on the collected data from GitHub demonstrate that Meta-AHIN outperforms state-of-the-art methods.

----

## [507] Hierarchical Adaptive Temporal-Relational Modeling for Stock Trend Prediction

**Authors**: *Heyuan Wang, Shun Li, Tengjiao Wang, Jiayi Zheng*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/508](https://doi.org/10.24963/ijcai.2021/508)

**Abstract**:

Stock trend prediction is a challenging task due to the non-stationary dynamics and complex market dependencies. Existing methods usually regard each stock as isolated for prediction, or simply detect their correlations based on a fixed predefined graph structure. Genuinely, stock associations stem from diverse aspects, the underlying relation signals should be implicit in comprehensive graphs. On the other hand, the RNN network is mainly used to model stock historical data, while is hard to capture fine-granular volatility patterns implied in different time spans. In this paper, we propose a novel Hierarchical Adaptive Temporal-Relational Network (HATR) to characterize and predict stock evolutions. By stacking dilated causal convolutions and gating paths, short- and long-term transition features are gradually grasped from multi-scale local compositions of stock trading sequences. Particularly, a dual attention mechanism with Hawkes process and target-specific query is proposed to detect significant temporal points and scales conditioned on individual stock traits. Furthermore, we develop a multi-graph interaction module which consolidates prior domain knowledge and data-driven adaptive learning to capture interdependencies among stocks. All components are integrated seamlessly in a unified end-to-end framework. Experiments on three real-world stock market datasets validate the effectiveness of our model.

----

## [508] BACKDOORL: Backdoor Attack against Competitive Reinforcement Learning

**Authors**: *Lun Wang, Zaynah Javed, Xian Wu, Wenbo Guo, Xinyu Xing, Dawn Song*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/509](https://doi.org/10.24963/ijcai.2021/509)

**Abstract**:

Recent research has confirmed the feasibility of backdoor attacks in deep reinforcement learning (RL) systems. However, the existing attacks require the ability to arbitrarily modify an agent's observation, constraining the application scope to simple RL systems such as Atari games. In this paper, we migrate backdoor attacks to more complex RL systems involving multiple agents and explore the possibility of triggering the backdoor without directly manipulating the agent's observation. As a proof of concept, we demonstrate that an adversary agent can trigger the backdoor of the victim agent with its own action in two-player competitive RL systems. We prototype and evaluate BackdooRL in four competitive environments. The results show that when the backdoor is activated, the winning rate of the victim drops by 17% to 37% compared to when not activated. The videos are hosted at https://github.com/wanglun1996/multi_agent_rl_backdoor_videos.

----

## [509] Hiding Numerical Vectors in Local Private and Shuffled Messages

**Authors**: *Shaowei Wang, Jin Li, Yuqiu Qian, Jiachun Du, Wenqing Lin, Wei Yang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/510](https://doi.org/10.24963/ijcai.2021/510)

**Abstract**:

Numerical vector aggregation has numerous applications in privacy-sensitive scenarios, such as distributed gradient estimation in federated learning, and statistical analysis on key-value data. Within the framework of local differential privacy, this work gives tight minimax error bounds of O(d s/(n epsilon^2)), where d is the dimension of the numerical vector and s is the number of non-zero entries. An attainable mechanism is then designed to improve from existing approaches suffering error rate of O(d^2/(n epsilon^2)) or O(d s^2/(n epsilon^2)). To break the error barrier in the local privacy, this work further consider privacy amplification in the shuffle model with anonymous channels, and shows the mechanism satisfies centralized (14 ln(2/delta) (s e^epsilon+2s-1)/(n-1))^0.5, delta)-differential privacy, which is domain independent and thus scales to federated learning of large models. We experimentally validate and compare it with existing approaches,  and demonstrate its significant error reduction.

----

## [510] Solving Large-Scale Extensive-Form Network Security Games via Neural Fictitious Self-Play

**Authors**: *Wanqi Xue, Youzhi Zhang, Shuxin Li, Xinrun Wang, Bo An, Chai Kiat Yeo*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/511](https://doi.org/10.24963/ijcai.2021/511)

**Abstract**:

Securing networked infrastructures is important in the real world. The problem of deploying security resources to protect against an attacker in networked domains can be modeled as Network Security Games (NSGs). Unfortunately, existing approaches, including the deep learning-based approaches, are inefficient to solve large-scale extensive-form NSGs. In this paper, we propose a novel learning paradigm, NSG-NFSP, to solve large-scale extensive-form NSGs based on Neural Fictitious Self-Play (NFSP). Our main contributions include: i) reforming the best response (BR) policy network in NFSP to be a mapping from action-state pair to action-value, to make the calculation of BR possible in NSGs; ii) converting the average policy network of an NFSP agent into a metric-based classifier, helping the agent to assign distributions only on legal actions rather than all actions; iii) enabling NFSP with high-level actions, which can benefit training efficiency and stability in NSGs; and iv) leveraging information contained in graphs of NSGs by learning efficient graph node embeddings. Our algorithm significantly outperforms state-of-the-art algorithms in both scalability and solution quality.

----

## [511] Towards Generating Summaries for Lexically Confusing Code through Code Erosion

**Authors**: *Fan Yan, Ming Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/512](https://doi.org/10.24963/ijcai.2021/512)

**Abstract**:

Code summarization aims to summarize code functionality as high-level nature language descriptions to assist in code comprehension.  Recent approaches in this field mainly focus on generating summaries for code with precise identifier names, in which meaningful words can be found indicating code functionality. When faced with lexically confusing code, current approaches are likely to fail since the correlation between code lexical tokens and summaries is scarce. To tackle this problem, we propose a novel summarization framework named VECOS. VECOS introduces an erosion mechanism to conquer the model's reliance on precisely defined lexical information. To facilitate learning the eroded code's functionality, we force the representation of the eroded code to align with the representation of its original counterpart via variational inference. Experimental results show that our approach outperforms the state-of-the-art approaches to generate coherent and reliable summaries for various lexically confusing code.

----

## [512] Change Matters: Medication Change Prediction with Recurrent Residual Networks

**Authors**: *Chaoqi Yang, Cao Xiao, Lucas Glass, Jimeng Sun*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/513](https://doi.org/10.24963/ijcai.2021/513)

**Abstract**:

Deep learning is revolutionizing predictive healthcare, including recommending medications to patients with complex health conditions. Existing approaches focus on predicting all medications for the current visit, which often overlaps with medications from previous visits. A more clinically relevant task is to identify medication changes.
In this paper, we propose a new recurrent residual networks, named MICRON, for medication change prediction. MICRON takes the changes in patient health records as input and learns to update a hid- den medication vector and the medication set recurrently with a reconstruction design. The medication vector is like the memory cell that encodes longitudinal information of medications. Unlike traditional methods that require the entire patient history for prediction, MICRON has a residual-based inference that allows for sequential updating based only on new patient features (e.g., new diagnoses in the recent visit), which is efficient.
We evaluated MICRON on real inpatient and outpatient datasets. MICRON achieves 3.5% and 7.8% relative improvements over the best baseline in F1 score, respectively. MICRON also requires fewer parameters, which significantly reduces the training time to 38.3s per epoch with 1.5Ã— speed-up.

----

## [513] SafeDrug: Dual Molecular Graph Encoders for Recommending Effective and Safe Drug Combinations

**Authors**: *Chaoqi Yang, Cao Xiao, Fenglong Ma, Lucas Glass, Jimeng Sun*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/514](https://doi.org/10.24963/ijcai.2021/514)

**Abstract**:

Medication recommendation is an essential task of AI for healthcare. Existing works focused on recommending drug combinations for patients with complex health conditions solely based on their electronic health records. Thus, they have the following limitations: (1) some important data such as drug molecule structures have not been utilized in the recommendation process. (2) drug-drug interactions (DDI) are modeled implicitly, which can lead to sub-optimal results. To address these limitations, we propose a DDI-controllable drug recommendation model named SafeDrug to leverage drugs’ molecule structures and model DDIs explicitly. SafeDrug is equipped with a global message passing neural network (MPNN) module and a local bipartite learning module to fully encode the connectivity and functionality of drug molecules. SafeDrug also has a controllable loss function to control DDI level in the recommended drug combinations effectively. On a benchmark dataset, our SafeDrug is relatively shown to reduce DDI by 19.43% and improves 2.88% on Jaccard similarity between recommended and actually prescribed drug combinations over previous approaches. Moreover, SafeDrug also requires much fewer parameters than previous deep learning based approaches, leading to faster training by about 14% and around 2× speed-up in inference.

----

## [514] Real-Time Pricing Optimization for Ride-Hailing Quality of Service

**Authors**: *Enpeng Yuan, Pascal Van Hentenryck*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/515](https://doi.org/10.24963/ijcai.2021/515)

**Abstract**:

When demand increases beyond the system capacity, riders in ride-hailing/ride-sharing systems often experience long waiting time, resulting in poor customer satisfaction. This paper proposes a spatio-temporal pricing framework (AP-RTRS) to alleviate this challenge and shows how it naturally complements state-of-the-art dispatching and routing algorithms. Specifically, the pricing  optimization model regulates demand to ensure that every rider opting to use the system is served within reason-able time: it does so either by reducing demand to meet the capacity constraints or by prompting potential riders to postpone service to a later time. The pricing model is a model-predictive control algorithm that works at a coarser temporal and spatial granularity compared to the real-time dispatching and routing, and naturally integrates vehicle relocations. Simulation experiments indicate that the pricing optimization model achieves short waiting times without sacrificing revenues and geographical fairness.

----

## [515] GraphMI: Extracting Private Graph Data from Graph Neural Networks

**Authors**: *Zaixi Zhang, Qi Liu, Zhenya Huang, Hao Wang, Chengqiang Lu, Chuanren Liu, Enhong Chen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/516](https://doi.org/10.24963/ijcai.2021/516)

**Abstract**:

As machine learning becomes more widely used for critical applications, the need to study its implications in privacy becomes urgent. 
Given access to the target model and auxiliary information, model inversion attack aims to infer sensitive features of the training dataset, which leads to great privacy concerns.
Despite its success in the grid domain, directly applying model inversion techniques on non grid domains such as graph achieves poor attack performance due to the difficulty to fully exploit the intrinsic properties of graphs and attributes of graph nodes used in GNN models.
To bridge this gap, we present Graph Model Inversion attack, which aims to infer edges of the training graph by inverting Graph Neural Networks, one of the most popular graph analysis tools.
Specifically, the projected gradient module in our method can tackle the discreteness of graph edges while preserving the sparsity and smoothness of graph features.
Moreover, a well designed graph autoencoder module can efficiently exploit graph topology, node attributes, and target model parameters.
With the proposed method, we study the connection between model inversion risk and edge influence and show that edges with greater influence are more likely to be recovered.
Extensive experiments over several public datasets demonstrate the effectiveness of our method.
We also show that differential privacy in its canonical form can hardly defend our attack while preserving decent utility.

----

## [516] CSGNN: Contrastive Self-Supervised Graph Neural Network for Molecular Interaction Prediction

**Authors**: *Chengshuai Zhao, Shuai Liu, Feng Huang, Shichao Liu, Wen Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/517](https://doi.org/10.24963/ijcai.2021/517)

**Abstract**:

Molecular interactions are significant resources for analyzing sophisticated biological systems. Identification of multifarious molecular interactions attracts increasing attention in biomedicine, bioinformatics, and human healthcare communities. Recently, a plethora of methods have been proposed to reveal molecular interactions in one specific domain. However, existing methods heavily rely on features or structures involving molecules, which limits the capacity of transferring the models to other tasks. Therefore, generalized models for the multifarious molecular interaction prediction (MIP) are in demand. In this paper, we propose a contrastive self-supervised graph neural network (CSGNN) to predict molecular interactions. CSGNN injects a mix-hop neighborhood aggregator into a graph neural network (GNN) to capture high-order dependency in the molecular interaction networks and leverages a contrastive self-supervised learning task as a regularizer within a multi-task learning paradigm to enhance the generalization ability. Experiments on seven molecular interaction networks show that CSGNN outperforms classic and state-of-the-art models. Comprehensive experiments indicate that the mix-hop aggregator and the self-supervised regularizer can effectively facilitate the link inference in multifarious molecular networks.

----

## [517] Long-term, Short-term and Sudden Event: Trading Volume Movement Prediction with Graph-based Multi-view Modeling

**Authors**: *Liang Zhao, Wei Li, Ruihan Bao, Keiko Harimoto, Yunfang Wu, Xu Sun*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/518](https://doi.org/10.24963/ijcai.2021/518)

**Abstract**:

Trading volume movement prediction is the key in a variety of financial applications.  Despite its importance, there is few  research  on  this  topic  because  of  its  requirement for comprehensive understanding of information  from  different  sources.   For  instance,  the  relation between multiple stocks,  recent transaction data and suddenly released events are all essential for understanding trading market.  However, most of the previous methods only take the fluctuation information  of  the  past  few  weeks  into  consideration, thus yielding poor performance.  To handle this issue, we propose a graph-based approach that can incorporate multi-view information, i.e., long-term stock trend, short-term fluctuation and sudden events information jointly into a temporal heterogeneous graph.   Besides,  our method is equipped with deep canonical analysis to highlight the correlations between different perspectives of fluctuation for better prediction.  Experiment results show that our method outperforms strong baselines by a large margin.

----

## [518] Objective-aware Traffic Simulation via Inverse Reinforcement Learning

**Authors**: *Guanjie Zheng, Hanyang Liu, Kai Xu, Zhenhui Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/519](https://doi.org/10.24963/ijcai.2021/519)

**Abstract**:

Traffic simulators act as an essential component in the operating and planning of transportation systems. Conventional traffic simulators usually employ a calibrated physical car-following model to describe vehicles' behaviors and their interactions with traffic environment. However, there is no universal physical model that can accurately predict the pattern of vehicle's behaviors in different situations. A fixed physical model tends to be less effective in a complicated environment given the non-stationary nature of traffic dynamics. In this paper, we formulate traffic simulation as an inverse reinforcement learning problem, and propose a parameter sharing adversarial inverse reinforcement learning model for dynamics-robust simulation learning. Our proposed model is able to imitate a vehicle's trajectories in the real world while simultaneously recovering the reward function that reveals the vehicle's true objective which is invariant to different dynamics. Extensive experiments on synthetic and real-world datasets show the superior performance of our approach compared to state-of-the-art methods and its robustness to variant dynamics of traffic.

----

## [519] Exemplification Modeling: Can You Give Me an Example, Please?

**Authors**: *Edoardo Barba, Luigi Procopio, Caterina Lacerra, Tommaso Pasini, Roberto Navigli*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/520](https://doi.org/10.24963/ijcai.2021/520)

**Abstract**:

Recently, generative approaches have been used effectively to provide definitions of words in their context. However, the opposite, i.e., generating a usage example given one or more words along with their definitions, has not yet been investigated. In this work, we introduce the novel task of Exemplification Modeling (ExMod), along with a sequence-to-sequence architecture and a training procedure for it. Starting from a set of (word, definition) pairs, our approach is capable of automatically generating high-quality sentences which express the requested semantics. As a result, we can drive the creation of sense-tagged data which cover the full range of meanings in any inventory of interest, and their interactions within sentences. Human annotators agree that the sentences generated are as fluent and semantically-coherent with the input definitions as the sentences in manually-annotated corpora.
Indeed, when employed as training data for Word Sense Disambiguation, our examples enable the current state of the art to be outperformed, and higher results to be achieved than when using gold-standard datasets only. We release the pretrained model, the dataset and the software at https://github.com/SapienzaNLP/exmod.

----

## [520] Generating Senses and RoLes: An End-to-End Model for Dependency- and Span-based Semantic Role Labeling

**Authors**: *Rexhina Blloshmi, Simone Conia, Rocco Tripodi, Roberto Navigli*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/521](https://doi.org/10.24963/ijcai.2021/521)

**Abstract**:

Despite the recent great success of the sequence-to-sequence paradigm in Natural Language Processing, the majority of current studies in Semantic Role Labeling (SRL) still frame the problem as a sequence labeling task.
In this paper we go against the flow and propose GSRL (Generating Senses and RoLes), the first sequence-to-sequence model for end-to-end SRL.
Our approach benefits from recently-proposed decoder-side pretraining techniques to generate both sense and role labels for all the predicates in an input sentence at once, in an end-to-end fashion.
Evaluated on standard gold benchmarks, GSRL achieves state-of-the-art results in both dependency- and span-based English SRL, proving empirically that our simple generation-based model can learn to produce complex predicate-argument structures.
Finally, we propose a framework for evaluating the robustness of an SRL model in a variety of synthetic low-resource scenarios which can aid human annotators in the creation of better, more diverse, and more challenging gold datasets.
We release GSRL at github.com/SapienzaNLP/gsrl.

----

## [521] Improving Context-Aware Neural Machine Translation with Source-side Monolingual Documents

**Authors**: *Linqing Chen, Junhui Li, Zhengxian Gong, Xiangyu Duan, Boxing Chen, Weihua Luo, Min Zhang, Guodong Zhou*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/522](https://doi.org/10.24963/ijcai.2021/522)

**Abstract**:

Document context-aware machine translation remains challenging due to the lack of large-scale document parallel corpora. To make full use of source-side monolingual documents for context-aware NMT, we propose a Pre-training approach with Global Context (PGC). In particular, we first propose a novel self-supervised pre-training task, which contains two training objectives: (1) reconstructing the original sentence from a corrupted version; (2) generating a gap sentence from its left and right neighbouring sentences. Then we design a universal model for PGC which consists of a global context encoder, a sentence encoder and a decoder, with similar architecture to typical context-aware NMT models. We evaluate the effectiveness and generality of our pre-trained PGC model by adapting it to various downstream context-aware NMT models. Detailed experimentation on four different translation tasks demonstrates that our PGC approach significantly improves the translation performance of context-aware NMT. For example, based on the state-of-the-art SAN model, we achieve an averaged improvement of 1.85 BLEU scores and 1.59 Meteor scores on the four translation tasks.

----

## [522] Focus on Interaction: A Novel Dynamic Graph Model for Joint Multiple Intent Detection and Slot Filling

**Authors**: *Zeyuan Ding, Zhihao Yang, Hongfei Lin, Jian Wang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/523](https://doi.org/10.24963/ijcai.2021/523)

**Abstract**:

Intent detection and slot filling are two main tasks for building a spoken language understanding (SLU) system. Since the two tasks are closely related, the joint models for the two tasks always outperform the pipeline models in SLU. However, most joint models directly incorporate multiple intent information for each token, which introduces intent noise into the sentence semantics, causing a decrease in the performance of the joint model. In this paper, we propose a Dynamic Graph Model (DGM) for joint multiple intent detection and slot filling, in which we adopt a sentence-level intent-slot interactive graph to model the correlation between the intents and slot. Besides, we design a novel method of constructing the graph, which can dynamically update the interactive graph and further alleviate the error propagation. Experimental results on several multi-intent and single-intent datasets show that our model not only achieves the state-of-the-art (SOTA) performance but also boosts the speed by three to six times over the SOTA model.

----

## [523] Dialogue Discourse-Aware Graph Model and Data Augmentation for Meeting Summarization

**Authors**: *Xiachong Feng, Xiaocheng Feng, Bing Qin, Xinwei Geng*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/524](https://doi.org/10.24963/ijcai.2021/524)

**Abstract**:

Meeting summarization is a challenging task due to its dynamic interaction nature among multiple speakers and lack of sufficient training data. Existing methods view the meeting as a linear sequence of utterances while ignoring the diverse relations between each utterance. Besides, the limited labeled data further hinders the ability of data-hungry neural models. In this paper, we try to mitigate the above challenges by introducing dialogue-discourse relations. First, we present a Dialogue Discourse-Dware Meeting Summarizer (DDAMS) to explicitly model the interaction between utterances in a meeting by modeling different discourse relations. The core module is a relational graph encoder, where the utterances and discourse relations are modeled in a graph interaction manner. Moreover, we devise a Dialogue Discourse-Aware Data Augmentation (DDADA) strategy to construct a pseudo-summarization corpus from existing input meetings, which is 20 times larger than the original dataset and can be used to pretrain DDAMS. Experimental results on AMI and ICSI meeting datasets show that our full system can achieve SOTA performance. Our codes and outputs are available at https://github.com/xcfcode/DDAMS/.

----

## [524] Automatically Paraphrasing via Sentence Reconstruction and Round-trip Translation

**Authors**: *Zilu Guo, Zhongqiang Huang, Kenny Q. Zhu, Guandan Chen, Kaibo Zhang, Boxing Chen, Fei Huang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/525](https://doi.org/10.24963/ijcai.2021/525)

**Abstract**:

Paraphrase generation plays key roles in NLP tasks such as question answering, machine translation, and information retrieval. In this paper, we propose a novel framework for paraphrase generation. It simultaneously decodes the output sentence using a pretrained wordset-to-sequence model and a round-trip translation model. We evaluate this framework on Quora, WikiAnswers, MSCOCO and Twitter, and show its advantage over previous state-of-the-art unsupervised methods and distantly-supervised methods by significant margins on all datasets. For Quora and WikiAnswers, our framework even performs better than some strongly supervised methods with domain adaptation. Further, we show that the generated paraphrases can be used to augment the training data for machine translation to achieve substantial improvements.

----

## [525] Dialogue Disentanglement in Software Engineering: How Far are We?

**Authors**: *Ziyou Jiang, Lin Shi, Celia Chen, Jun Hu, Qing Wang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/526](https://doi.org/10.24963/ijcai.2021/526)

**Abstract**:

Despite the valuable information contained in software chat messages, disentangling them into distinct conversations is an essential prerequisite for any in-depth analyses that utilize this information. To provide a better understanding of the current state-of-the-art, we evaluate five popular dialog disentanglement approaches on software-related chat. We find that existing approaches do not perform well on disentangling software-related dialogs that discuss technical and complex topics. Further investigation on how well the existing disentanglement measures reflect human satisfaction shows that existing measures cannot correctly indicate human satisfaction on disentanglement results. Therefore, in this paper, we introduce and evaluate a novel measure, named DLD. Using results of human satisfaction, we further summarize four most frequently appeared bad disentanglement cases on software-related chat to insight future improvements. These cases include (i) Ignoring Interaction Patterns, (ii) Ignoring Contextual Information, (iii) Mixing up Topics, and (iv) Ignoring User Relationships. We believe that our findings provide valuable insights on the effectiveness of existing dialog disentanglement approaches and these findings would promote a better application of dialog disentanglement in software engineering.

----

## [526] FedSpeech: Federated Text-to-Speech with Continual Learning

**Authors**: *Ziyue Jiang, Yi Ren, Ming Lei, Zhou Zhao*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/527](https://doi.org/10.24963/ijcai.2021/527)

**Abstract**:

Federated learning enables collaborative training of machine learning models under strict privacy restrictions and federated text-to-speech aims to synthesize natural speech of multiple users with a few audio training samples stored in their devices locally. However, federated text-to-speech faces several challenges: very few training samples from each speaker are available, training samples are all stored in local device of each user, and global model is vulnerable to various attacks. In this paper, we propose a novel federated learning architecture based on continual learning approaches to overcome the difficulties above. Specifically, 1) we use gradual pruning masks to isolate parameters for preserving speakers' tones; 2) we apply selective masks for effectively reusing knowledge from tasks; 3) a private speaker embedding is introduced to keep users' privacy. Experiments on a reduced VCTK dataset demonstrate the effectiveness of FedSpeech: it nearly matches multi-task training in terms of multi-speaker speech quality; moreover, it sufficiently retains the speakers' tones and even outperforms the multi-task training in the speaker similarity experiment.

----

## [527] ALaSca: an Automated approach for Large-Scale Lexical Substitution

**Authors**: *Caterina Lacerra, Tommaso Pasini, Rocco Tripodi, Roberto Navigli*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/528](https://doi.org/10.24963/ijcai.2021/528)

**Abstract**:

The lexical substitution task aims at finding suitable replacements for words in context. It has proved to be useful in several areas, such as word sense induction and text simplification, as well as in more practical applications such as writing-assistant tools.  However, the paucity of annotated data has forced researchers to apply mainly unsupervised approaches,  limiting the applicability of large pre-trained models and thus hampering the potential benefits of supervised approaches to the task. In this paper, we mitigate this issue by proposing ALaSca, a novel approach to automatically creating large-scale datasets for  English lexical substitution.  ALaSca allows examples to be produced for potentially any word in a language vocabulary and to cover most of the meanings it lists.  Thanks to this,  we can unleash the full potential of neural architectures and finetune them on the lexical substitution task. Indeed,  when using our data, a  transformer-based model performs substantially better than when using manually annotated data only. We release  ALaSca at  https://sapienzanlp.github.io/alasca/.

----

## [528] Enhancing Label Representations with Relational Inductive Bias Constraint for Fine-Grained Entity Typing

**Authors**: *Jinqing Li, Xiaojun Chen, Dakui Wang, Yuwei Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/529](https://doi.org/10.24963/ijcai.2021/529)

**Abstract**:

Fine-Grained Entity Typing (FGET) is a task that aims at classifying an entity mention into a wide range of entity label types. Recent researches improve the task performance by imposing the label-relational inductive bias based on the hierarchy of labels or label co-occurrence graph. However, they usually overlook explicit interactions between instances and labels which may limit the capability of label representations. Therefore, we propose a novel method based on a two-phase graph network for the FGET task to enhance the label representations, via imposing the relational inductive biases of instance-to-label and label-to-label. In the phase 1, instance features will be introduced into label representations to make the label representations more representative. In the phase 2, interactions of labels will capture dependency relationships among them thus make label representations more smooth. During prediction, we introduce a pseudo-label generator for the construction of the two-phase graph. The input instances differ from batch to batch so that the label representations are dynamic. Experiments on three public datasets verify the effectiveness and stability of our proposed method and achieve state-of-the-art results on their testing sets.

----

## [529] Modelling General Properties of Nouns by Selectively Averaging Contextualised Embeddings

**Authors**: *Na Li, Zied Bouraoui, José Camacho-Collados, Luis Espinosa Anke, Qing Gu, Steven Schockaert*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/530](https://doi.org/10.24963/ijcai.2021/530)

**Abstract**:

While the success of pre-trained language models has largely eliminated the need for high-quality static word vectors in many NLP applications, static word vectors continue to play an important role in tasks where word meaning needs to be modelled in the absence of linguistic context. In this paper, we explore how the contextualised embeddings predicted by BERT can be used to produce high-quality word vectors for such domains, in particular related to knowledge base completion, where our focus is on capturing the semantic properties of nouns. We find that a simple strategy of averaging the contextualised embeddings of masked word mentions leads to vectors that outperform the static word vectors learned by BERT, as well as those from standard word embedding models, in property induction tasks. We notice in particular that masking target words is critical to achieve this strong performance, as the resulting vectors focus less on idiosyncratic properties and more on general semantic properties. Inspired by this view, we propose a filtering strategy which is aimed at removing the most idiosyncratic mention vectors, allowing us to obtain further performance gains in property induction.

----

## [530] Asynchronous Multi-grained Graph Network For Interpretable Multi-hop Reading Comprehension

**Authors**: *Ronghan Li, Lifang Wang, Shengli Wang, Zejun Jiang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/531](https://doi.org/10.24963/ijcai.2021/531)

**Abstract**:

Multi-hop machine reading comprehension (MRC) task aims to enable models to answer the compound question according to the bridging information. Existing methods that use graph neural networks to represent multiple granularities such as entities and sentences in documents update all nodes synchronously, ignoring the fact that multi-hop reasoning has a certain logical order across granular levels. In this paper, we introduce an Asynchronous Multi-grained Graph Network (AMGN) for multi-hop MRC. First, we construct a multigrained graph containing entity and sentence nodes. Particularly, we use independent parameters to represent relationship groups defined according to the level of granularity. Second, an asynchronous update mechanism based on multi-grained relationships is proposed to mimic human multi-hop reading logic. Besides, we present a question reformulation mechanism to update the latent representation of the compound question with updated graph nodes. We evaluate the proposed model on the HotpotQA dataset and achieve top competitive performance in distractor setting compared with other published models. Further analysis shows that the asynchronous update mechanism can effectively form interpretable reasoning chains at different granularity levels.

----

## [531] Keep the Structure: A Latent Shift-Reduce Parser for Semantic Parsing

**Authors**: *Yuntao Li, Bei Chen, Qian Liu, Yan Gao, Jian-Guang Lou, Yan Zhang, Dongmei Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/532](https://doi.org/10.24963/ijcai.2021/532)

**Abstract**:

Traditional end-to-end semantic parsing models treat a natural language utterance as a holonomic structure. However, hierarchical structures exist in natural languages, which also align with the hierarchical structures of logical forms. In this paper, we propose a latent shift-reduce parser, called LASP, which decomposes both natural language queries and logical form expressions according to their hierarchical structures and finds local alignment between them to enhance semantic parsing. LASP consists of a base parser and a shift-reduce splitter. The splitter dynamically separates an NL query into several spans. The base parser converts the relevant simple spans into logical forms, which are further combined to obtain the final logical form. We conducted empirical studies on two datasets across different domains and different types of logical forms. The results demonstrate that the proposed method significantly improves the performance of semantic parsing, especially on unseen scenarios.

----

## [532] Discourse-Level Event Temporal Ordering with Uncertainty-Guided Graph Completion

**Authors**: *Jian Liu, Jinan Xu, Yufeng Chen, Yujie Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/533](https://doi.org/10.24963/ijcai.2021/533)

**Abstract**:

Learning to order events at discourse-level is a crucial text understanding task.
Despite many efforts for this task, the current state-of-the-art methods rely heavily on manually designed features, which are costly to produce and are often specific to tasks/domains/datasets.
In this paper, we propose a new graph perspective on the task, which does not require complex feature engineering but can assimilate global features and learn inter-dependencies effectively.
Specifically, in our approach, each document is considered as a temporal graph, in which the nodes and edges represent events and event-event relations respectively.
In this sense, the temporal ordering task corresponds to constructing edges for an empty graph.
To train our model, we design a graph mask pre-training mechanism, which can learn inter-dependencies of temporal relations by learning to recover a masked edge following graph topology.
In the testing stage, we design an certain-first strategy based on model uncertainty, which can decide the prediction orders and reduce the risk of error propagation.
The experimental results demonstrate that our approach outperforms previous methods consistently and can meanwhile maintain good global consistency.

----

## [533] Improving Text Generation with Dynamic Masking and Recovering

**Authors**: *Zhidong Liu, Junhui Li, Muhua Zhu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/534](https://doi.org/10.24963/ijcai.2021/534)

**Abstract**:

Due to different types of inputs, diverse text generation tasks may adopt different encoder-decoder frameworks. Thus most existing approaches that aim to improve the robustness of certain generation tasks are input-relevant, and may not work well for other generation tasks. Alternatively, in this paper we present a universal approach to enhance the language representation for text generation on the base of generic encoder-decoder frameworks. This is done from two levels. First, we introduce randomness by randomly masking some percentage of tokens on the decoder side when training the models. In this way, instead of using ground truth history context, we use its corrupted version to predict the next token. Then we propose an auxiliary task to properly recover those masked tokens. Experimental results on several text generation tasks including machine translation (MT), AMR-to-text generation, and image captioning show that the proposed approach can significantly improve over competitive baselines without using any task-specific techniques. This suggests the effectiveness and generality of our proposed approach.

----

## [534] Consistent Inference for Dialogue Relation Extraction

**Authors**: *Xinwei Long, Shuzi Niu, Yucheng Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/535](https://doi.org/10.24963/ijcai.2021/535)

**Abstract**:

Relation Extraction is key to many downstream tasks. Dialogue relation extraction aims at discovering entity relations from multi-turn dialogue scenario. There exist utterance, topic and relation discrepancy mainly due to multi-speakers, utterances, and relations. In this paper, we propose a consistent learning and inference method to minimize possible contradictions from those distinctions. First, we design mask mechanisms to refine utterance-aware and speaker-aware representations respectively from the global dialogue representation for the utterance distinction. Then a gate mechanism is proposed to aggregate such bi-grained representations. Next, mutual attention mechanism is introduced to obtain the entity representation for various relation specific topic structures. Finally, the relational inference is performed through first order logic constraints over the labeled data to decrease logically contradictory predicted relations. Experimental results on two benchmark datasets show that the F1 performance improvement of the proposed method is at least 3.3% compared with SOTA.

----

## [535] Multi-Hop Fact Checking of Political Claims

**Authors**: *Wojciech Ostrowski, Arnav Arora, Pepa Atanasova, Isabelle Augenstein*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/536](https://doi.org/10.24963/ijcai.2021/536)

**Abstract**:

Recent work has proposed multi-hop models and datasets for studying complex natural language reasoning. One notable task requiring multi-hop reasoning is fact checking, where a set of connected evidence pieces leads to the final verdict of a claim. However, existing datasets either do not provide annotations for gold evidence pages, or the only dataset which does (FEVER) mostly consists of claims which can be fact-checked with simple reasoning and is constructed artificially. Here, we study more complex claim verification of naturally occurring claims with multiple hops over interconnected evidence chunks. We: 1) construct a small annotated dataset, PolitiHop, of evidence sentences for claim verification; 2) compare it to existing multi-hop datasets; and 3) study how to transfer knowledge from more extensive in- and out-of-domain resources to PolitiHop. We find that the task is complex and achieve the best performance with an architecture that specifically models reasoning over evidence pieces in combination with in-domain transfer learning.

----

## [536] Laughing Heads: Can Transformers Detect What Makes a Sentence Funny?

**Authors**: *Maxime Peyrard, Beatriz Borges, Kristina Gligoric, Robert West*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/537](https://doi.org/10.24963/ijcai.2021/537)

**Abstract**:

The automatic detection of humor poses a grand challenge for natural language processing.
Transformer-based systems have recently achieved remarkable results on this task, but they usually
(1) were evaluated in setups where serious vs humorous texts came from entirely different sources, and
(2) focused on benchmarking performance without providing insights into how the models work.
We make progress in both respects by training and analyzing transformer-based humor recognition models on a recently introduced dataset consisting of minimal pairs of aligned sentences, one serious, the other humorous.
We find that, although our aligned dataset is much harder than previous datasets, transformer-based models recognize the humorous sentence in an aligned pair with high accuracy (78\%).
In a careful error analysis, we characterize easy vs hard instances.
Finally, by analyzing attention weights, we obtain important insights into the mechanisms by which transformers recognize humor.
Most remarkably, we find clear evidence that one single attention head learns to recognize the words that make a test sentence humorous, even without access to this information at training time.

----

## [537] A Streaming End-to-End Framework For Spoken Language Understanding

**Authors**: *Nihal Potdar, Anderson Raymundo Avila, Chao Xing, Dong Wang, Yiran Cao, Xiao Chen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/538](https://doi.org/10.24963/ijcai.2021/538)

**Abstract**:

End-to-end spoken language understanding (SLU) recently attracted increasing interest. Compared to the conventional tandem-based approach that combines speech recognition and language understanding as separate modules, the new approach extracts users' intentions directly from the speech signals, resulting in joint optimization and low latency. Such an approach, however, is typically designed to process one intent at a time, which leads users to have to take multiple rounds to fulfill their requirements while interacting with a dialogue system. In this paper, we propose a streaming end-to-end framework that can process multiple intentions in an online and incremental way. The backbone of our framework is a unidirectional RNN trained with the connectionist temporal classification (CTC) criterion. By this design, an intention can be identified when sufficient evidence has been accumulated, and  multiple intentions will be identified sequentially. We evaluate our solution on the Fluent Speech Commands (FSC) dataset and the detection accuracy is about 97 % on all multi-intent settings. This result is comparable to the performance of the state-of-the-art non-streaming models, but is achieved in an online and incremental way. We also employ our model to an keyword spotting task using the Google Speech Commands dataset, and the results are also highly promising.

----

## [538] MultiMirror: Neural Cross-lingual Word Alignment for Multilingual Word Sense Disambiguation

**Authors**: *Luigi Procopio, Edoardo Barba, Federico Martelli, Roberto Navigli*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/539](https://doi.org/10.24963/ijcai.2021/539)

**Abstract**:

Word Sense Disambiguation (WSD), i.e., the task of assigning senses to words in context, has seen a surge of interest with the advent of neural models and a considerable increase in performance up to 80% F1 in English. However, when considering other languages, the availability of training data is limited, which hampers scaling WSD to many languages. To address this issue, we put forward MultiMirror, a sense projection approach for multilingual WSD based on a novel neural discriminative model for word alignment: given as input a pair of parallel sentences, our model -- trained with a low number of instances -- is capable of jointly aligning, at the same time, all source and target tokens with each other, surpassing its competitors across several language combinations. We demonstrate that projecting senses from English by leveraging the alignments produced by our model leads a simple mBERT-powered classifier to achieve a new state of the art on established WSD datasets in French, German, Italian, Spanish and Japanese. We release our software and all our datasets at https://github.com/SapienzaNLP/multimirror.

----

## [539] Learning Class-Transductive Intent Representations for Zero-shot Intent Detection

**Authors**: *Qingyi Si, Yuanxin Liu, Peng Fu, Zheng Lin, Jiangnan Li, Weiping Wang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/540](https://doi.org/10.24963/ijcai.2021/540)

**Abstract**:

Zero-shot intent detection (ZSID) aims to deal with the continuously emerging intents without annotated training data. However, existing ZSID systems suffer from two limitations: 1) They are not good at modeling the relationship between seen and unseen intents. 2) They cannot effectively recognize unseen intents under the generalized intent detection (GZSID) setting. A critical problem behind these limitations is that the representations of unseen intents cannot be learned in the training stage. To address this problem, we propose a novel framework that utilizes unseen class labels to learn Class-Transductive Intent Representations (CTIR). Specifically, we allow the model to predict unseen intents during training, with the corresponding label names serving as input utterances. On this basis, we introduce a multi-task learning objective, which encourages the model to learn the distinctions among intents, and a similarity scorer, which estimates the connections among intents more accurately. CTIR is easy to implement and can be integrated with existing ZSID and GZSID methods. Experiments on two real-world datasets show that CTIR brings considerable improvement to the baseline systems.

----

## [540] MEDA: Meta-Learning with Data Augmentation for Few-Shot Text Classification

**Authors**: *Pengfei Sun, Yawen Ouyang, Wenming Zhang, Xinyu Dai*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/541](https://doi.org/10.24963/ijcai.2021/541)

**Abstract**:

Meta-learning has recently emerged as a promising technique to address the challenge of few-shot learning. However, standard meta-learning methods mainly focus on visual tasks, which makes it hard for them to deal with diverse text data directly. In this paper, we introduce a novel framework for few-shot text classification, which is named as MEta-learning with Data Augmentation (MEDA). MEDA is composed of two modules, a ball generator and a meta-learner, which are learned jointly. The ball generator is to increase the number of shots per class by generating more samples, so that meta-learner can be trained with both original and augmented samples. It is worth noting that ball generator is agnostic to the choice of the meta-learning methods. Experiment results show that on both datasets, MEDA outperforms existing state-of-the-art methods and significantly improves the performance of meta-learning on few-shot text classification.

----

## [541] A Sequence-to-Set Network for Nested Named Entity Recognition

**Authors**: *Zeqi Tan, Yongliang Shen, Shuai Zhang, Weiming Lu, Yueting Zhuang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/542](https://doi.org/10.24963/ijcai.2021/542)

**Abstract**:

Named entity recognition (NER) is a widely studied task in natural language processing. Recently, a growing number of studies have focused on the nested NER. The span-based methods, considering the entity recognition as a span classification task, can deal with nested entities naturally. But they suffer from the huge search space and the lack of interactions between entities. To address these issues, we propose a novel sequence-to-set neural network for nested NER. Instead of specifying candidate spans in advance, we provide a fixed set of learnable vectors to learn the patterns of the valuable spans. We utilize a non-autoregressive decoder to predict the final set of entities in one pass, in which we are able to capture dependencies between entities. Compared with the sequence-to-sequence method, our model is more suitable for such unordered recognition task as it is insensitive to the label order. In addition, we utilize the loss function based on bipartite matching to compute the overall training loss. Experimental results show that our proposed model achieves state-of-the-art on three nested NER corpora: ACE 2004, ACE 2005 and KBP 2017. The code is available at https://github.com/zqtan1024/sequence-to-set.

----

## [542] A Structure Self-Aware Model for Discourse Parsing on Multi-Party Dialogues

**Authors**: *Ante Wang, Linfeng Song, Hui Jiang, Shaopeng Lai, Junfeng Yao, Min Zhang, Jinsong Su*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/543](https://doi.org/10.24963/ijcai.2021/543)

**Abstract**:

Conversational discourse structures aim to describe how a dialogue is organized, thus they are helpful for dialogue understanding and response generation. This paper focuses on predicting discourse dependency structures for multi-party dialogues. Previous work adopts incremental methods that take the features from the already predicted discourse relations to help generate the next one. Although the inter-correlations among predictions considered, we find that the error propagation is also very serious and hurts the overall performance. To alleviate error propagation, we propose a Structure Self-Aware (SSA) model, which adopts a novel edge-centric Graph Neural Network (GNN) to update the information between each Elementary Discourse Unit (EDU) pair layer by layer, so that expressive representations can be learned without historical predictions. In addition, we take auxiliary training signals (e.g. structure distillation) for better representation learning. Our model achieves the new state-of-the-art performances on two conversational discourse parsing benchmarks, largely outperforming the previous methods.

----

## [543] Hierarchical Modeling of Label Dependency and Label Noise in Fine-grained Entity Typing

**Authors**: *Junshuang Wu, Richong Zhang, Yongyi Mao, Masoumeh Soflaei Shahrbabak, Jinpeng Huai*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/544](https://doi.org/10.24963/ijcai.2021/544)

**Abstract**:

Fine-grained entity typing (FET) aims to annotate the entity mentions in a sentence with fine-grained type labels. It brings plentiful semantic information for many natural language processing tasks. Existing FET approaches apply hard attention to learn on the noisy labels, and ignore that those noises have structured hierarchical dependency. Despite their successes, these FET models are insufficient in modeling type hierarchy dependencies and handling label noises. In this paper, we directly tackle the structured noisy labels by combining a forward tree module and a backward tree module. Specifically, the forward tree formulates the informative walk that hierarchically represents the type distributions. The backward tree models the erroneous walk that learns the noise confusion matrix. Empirical studies on several benchmark data sets confirm the effectiveness of the proposed framework.

----

## [544] Learn from Syntax: Improving Pair-wise Aspect and Opinion Terms Extraction with Rich Syntactic Knowledge

**Authors**: *Shengqiong Wu, Hao Fei, Yafeng Ren, Donghong Ji, Jingye Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/545](https://doi.org/10.24963/ijcai.2021/545)

**Abstract**:

In this paper, we propose to enhance the pair-wise aspect and opinion terms extraction (PAOTE) task by incorporating rich syntactic knowledge. We first build a syntax fusion encoder for encoding syntactic features, including a label-aware graph convolutional network (LAGCN) for modeling the dependency edges and labels, as well as the POS tags unifiedly, and a local-attention module encoding POS tags for better term boundary detection. During pairing, we then adopt Biaffine and Triaffine scoring for high-order aspect-opinion term pairing, in the meantime re-harnessing the syntax-enriched representations in LAGCN for syntactic-aware scoring. Experimental results on four benchmark datasets demonstrate that our model outperforms current state-of-the-art baselines, meanwhile yielding explainable predictions with syntactic knowledge.

----

## [545] Knowledge-Aware Dialogue Generation via Hierarchical Infobox Accessing and Infobox-Dialogue Interaction Graph Network

**Authors**: *Sixing Wu, Minghui Wang, Dawei Zhang, Yang Zhou, Ying Li, Zhonghai Wu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/546](https://doi.org/10.24963/ijcai.2021/546)

**Abstract**:

Due to limited knowledge carried by queries, traditional dialogue systems often face the dilemma of generating boring responses, leading to poor user experience. To alleviate this issue, this paper proposes a novel infobox knowledge-aware dialogue generation approach, HITA-Graph, with three unique features. First, open-domain infobox tables that describe entities with relevant attributes are adopted as the knowledge source. An order-irrelevance Hierarchical Infobox Table Encoder is proposed to represent an infobox table at three levels of granularity. In addition, an Infobox-Dialogue Interaction Graph Network is built to effectively integrate the infobox context and the dialogue context into a unified infobox representation. Second, a Hierarchical Infobox Attribute Attention mechanism is developed to  access the encoded infobox knowledge at different levels of granularity. Last but not least, a Dynamic Mode Fusion strategy is designed to allow the Decoder to select a vocabulary word or copy a word from the given infobox/query. We extract infobox tables from Chinese Wikipedia and construct an infobox knowledge base. Extensive evaluation on an open-released Chinese corpus demonstrates the superior performance of our approach against several representative methods.

----

## [546] Improving Stylized Neural Machine Translation with Iterative Dual Knowledge Transfer

**Authors**: *Xuanxuan Wu, Jian Liu, Xinjie Li, Jinan Xu, Yufeng Chen, Yujie Zhang, Hui Huang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/547](https://doi.org/10.24963/ijcai.2021/547)

**Abstract**:

Stylized neural machine translation (NMT) aims to translate sentences of one style into sentences of another style, which is essential for the application of machine translation in a real-world scenario. However, a major challenge in this task is the scarcity of high-quality parallel data which is stylized paired. To address this problem, we propose an iterative dual knowledge transfer framework that utilizes informal training data of machine translation and formality style transfer data to create large-scale stylized paired data, for the training of stylized machine translation model. Specifically, we perform bidirectional knowledge transfer between translation model and text style transfer model iteratively through knowledge distillation. Then, we further propose a data-refinement module to process the noisy synthetic parallel data generated during knowledge transfer. Experiment results demonstrate the effectiveness of our method, achieving an improvement over the existing best model by 5 BLEU points on MTFC dataset. Meanwhile, extensive analyses illustrate our method can also improve the accuracy of formality style transfer.

----

## [547] UniMF: A Unified Framework to Incorporate Multimodal Knowledge Bases intoEnd-to-End Task-Oriented Dialogue Systems

**Authors**: *Shiquan Yang, Rui Zhang, Sarah M. Erfani, Jey Han Lau*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/548](https://doi.org/10.24963/ijcai.2021/548)

**Abstract**:

Knowledge bases (KBs) are usually essential for building practical dialogue systems. Recently we have seen rapidly growing interest in integrating knowledge bases into dialogue systems. However, existing approaches mostly deal with knowledge bases of a single modality, typically textual information. As today's knowledge bases become abundant with multimodal information such as images, audios and videos, the limitation of existing approaches greatly hinders the development of dialogue systems. In this paper, we focus on task-oriented dialogue systems and address this limitation by proposing a novel model that integrates external multimodal KB reasoning with pre-trained language models. We further enhance the model via a novel multi-granularity fusion mechanism to capture multi-grained semantics in the dialogue history. To validate the effectiveness of the proposed model, we collect a new large-scale (14K) dialogue dataset MMDialKB, built upon multimodal KB. Both automatic and human evaluation results on MMDialKB demonstrate the superiority of our proposed framework over strong baselines.

----

## [548] MRD-Net: Multi-Modal Residual Knowledge Distillation for Spoken Question Answering

**Authors**: *Chenyu You, Nuo Chen, Yuexian Zou*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/549](https://doi.org/10.24963/ijcai.2021/549)

**Abstract**:

Spoken question answering (SQA) has recently drawn considerable attention in the speech community. It requires systems to find correct answers from the given spoken passages simultaneously. The common SQA systems consist of the automatic speech recognition (ASR) module and text-based question answering module. However, previous methods suffer from severe performance degradation due to ASR errors. To alleviate this problem, this work proposes a novel multi-modal residual knowledge distillation method (MRD-Net), which further distills knowledge at the acoustic level from the audio-assistant (Audio-A). Specifically, we utilize the teacher (T) trained on manual transcriptions to guide the training of the student (S) on ASR transcriptions. We also show that introducing an Audio-A helps this procedure by learning residual errors between T and S. Moreover, we propose a simple yet effective attention mechanism to adaptively leverage audio-text features as the new deep attention knowledge to boost the network performance. Extensive experiments demonstrate that the proposed MRD-Net achieves superior results compared with state-of-the-art methods on three spoken question answering benchmark datasets.

----

## [549] Cross-Domain Slot Filling as Machine Reading Comprehension

**Authors**: *Mengshi Yu, Jian Liu, Yufeng Chen, Jinan Xu, Yujie Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/550](https://doi.org/10.24963/ijcai.2021/550)

**Abstract**:

With task-oriented dialogue systems being widely applied in everyday life, slot filling, the essential component of task-oriented dialogue systems, is required to be quickly adapted to new domains that contain domain-specific slots with few or no training data. Previous methods for slot filling usually adopt sequence labeling framework, which, however, often has limited ability when dealing with the domain-specific slots. In this paper, we take a new perspective on cross-domain slot filling by framing it as a machine reading comprehension (MRC) problem. Our approach firstly transforms slot names into well-designed queries, which contain rich informative prior knowledge and are very helpful for the detection of domain-specific slots. In addition, we utilize the large-scale MRC dataset for pre-training, which further alleviates the data scarcity problem. Experimental results on SNIPS and ATIS datasets show that our approach consistently outperforms the existing state-of-the-art methods by a large margin.

----

## [550] Document-level Relation Extraction as Semantic Segmentation

**Authors**: *Ningyu Zhang, Xiang Chen, Xin Xie, Shumin Deng, Chuanqi Tan, Mosha Chen, Fei Huang, Luo Si, Huajun Chen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/551](https://doi.org/10.24963/ijcai.2021/551)

**Abstract**:

Document-level relation extraction aims to extract relations among multiple entity pairs from a document. Previously proposed graph-based or transformer-based models utilize the entities independently, regardless of global information among relational triples. This paper approaches the problem by predicting an entity-level relation matrix to capture local and global information, parallel to the semantic segmentation task in computer vision. Herein, we propose a Document U-shaped Network for document-level relation extraction. Specifically, we leverage an encoder module to capture the context information of entities and a U-shaped segmentation module over the image-style feature map to capture global interdependency among triples. Experimental results show that our approach can obtain state-of-the-art performance on three benchmark datasets DocRED, CDR, and GDA.

----

## [551] Drop Redundant, Shrink Irrelevant: Selective Knowledge Injection for Language Pretraining

**Authors**: *Ningyu Zhang, Shumin Deng, Xu Cheng, Xi Chen, Yichi Zhang, Wei Zhang, Huajun Chen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/552](https://doi.org/10.24963/ijcai.2021/552)

**Abstract**:

Previous research has demonstrated the power of leveraging prior knowledge to improve the performance of deep models in natural language processing. However, traditional methods neglect the fact that redundant and irrelevant knowledge exists in external knowledge bases. In this study, we launched an in-depth empirical investigation into downstream tasks and found that knowledge-enhanced approaches do not always exhibit satisfactory improvements. To this end, we investigate the fundamental reasons for ineffective knowledge infusion and present selective injection for language pretraining, which constitutes a model-agnostic method and is readily pluggable into previous approaches. Experimental results on benchmark datasets demonstrate that our approach can enhance state-of-the-art knowledge injection methods.

----

## [552] Relational Gating for "What If" Reasoning

**Authors**: *Chen Zheng, Parisa Kordjamshidi*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/553](https://doi.org/10.24963/ijcai.2021/553)

**Abstract**:

This paper addresses the challenge of learning to do procedural reasoning over text to answer "What if..." questions. We propose a novel relational gating network that learns to filter the key entities and relationships and learns contextual and cross representations of both procedure and question for finding the answer. Our relational gating network contains an entity gating module, relation gating module, and contextual interaction module. These modules help in solving the "What if..." reasoning problem. We show that modeling pairwise relationships helps to capture higher-order relations and find the line of reasoning for causes and effects in the procedural descriptions. Our proposed approach achieves the state-of-the-art results on the WIQA dataset.

----

## [553] Efficient Black-Box Planning Using Macro-Actions with Focused Effects

**Authors**: *Cameron Allen, Michael Katz, Tim Klinger, George Konidaris, Matthew Riemer, Gerald Tesauro*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/554](https://doi.org/10.24963/ijcai.2021/554)

**Abstract**:

The difficulty of deterministic planning increases exponentially with search-tree depth. Black-box planning presents an even greater challenge, since planners must operate without an explicit model of the domain. Heuristics can make search more efficient, but goal-aware heuristics for black-box planning usually rely on goal counting, which is often quite uninformative. In this work, we show how to overcome this limitation by discovering macro-actions that make the goal-count heuristic more accurate. Our approach searches for macro-actions with focused effects (i.e. macros that modify only a small number of state variables), which align well with the assumptions made by the goal-count heuristic. Focused macros dramatically improve black-box planning efficiency across a wide range of planning domains, sometimes beating even state-of-the-art planners with access to a full domain model.

----

## [554] ME-MCTS: Online Generalization by Combining Multiple Value Estimators

**Authors**: *Hendrik Baier, Michael Kaisers*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/555](https://doi.org/10.24963/ijcai.2021/555)

**Abstract**:

This paper addresses the challenge of online generalization in tree search. We propose Multiple Estimator Monte Carlo Tree Search (ME-MCTS), with a two-fold contribution: first, we introduce a formalization of online generalization that can represent existing techniques such as "history heuristics", "RAVE", or "OMA" -- contextual action value estimators or abstractors that generalize across specific contexts. Second, we incorporate recent advances in estimator averaging that enable guiding search by combining the online action value estimates of any number of such abstractors or similar types of action value estimators.
Unlike previous work, which usually proposed a single abstractor for either the selection or the rollout phase of MCTS simulations, our approach focuses on the combination of multiple estimators and applies them to all move choices in MCTS simulations. As the MCTS tree itself is just another value estimator -- unbiased, but without abstraction -- this blurs the traditional distinction between action choices inside and outside of the MCTS tree.
Experiments with three abstractors in four board games show significant improvements of ME-MCTS over MCTS using only a single abstractor, both for MCTS with random rollouts as well as for MCTS with static evaluation functions.
While we used deterministic, fully observable games, ME-MCTS naturally extends to more challenging settings.

----

## [555] Learn to Intervene: An Adaptive Learning Policy for Restless Bandits in Application to Preventive Healthcare

**Authors**: *Arpita Biswas, Gaurav Aggarwal, Pradeep Varakantham, Milind Tambe*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/556](https://doi.org/10.24963/ijcai.2021/556)

**Abstract**:

In many public health settings, it is important for patients to adhere to health programs, such as taking medications and periodic health checks. Unfortunately, beneficiaries may gradually disengage from such programs, which is detrimental to their health. A concrete example of gradual disengagement has been observed by an organization that carries out a free automated call-based program for spreading preventive care information among pregnant women. Many women stop picking up calls after being enrolled for a few months. To avoid such disengagements, it is important to provide timely interventions. Such interventions are often expensive and can be provided to only a small fraction of the beneficiaries. We model this scenario as a restless multi-armed bandit (RMAB) problem, where each beneficiary is assumed to transition from one state to another depending on the intervention.  Moreover, since the transition probabilities are unknown a priori, we propose a Whittle index based Q-Learning mechanism and show that it converges to the optimal solution. Our method improves over existing learning-based methods for RMABs on multiple benchmarks from literature and also on the maternal healthcare dataset.

----

## [556] Type-WA*: Using Exploration in Bounded Suboptimal Planning

**Authors**: *Eldan Cohen, Richard Anthony Valenzano, Sheila A. McIlraith*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/557](https://doi.org/10.24963/ijcai.2021/557)

**Abstract**:

Previous work on satisficing planning using greedy best-first search (GBFS) has shown that non-greedy, randomized exploration can help escape uninformative heuristic regions and solve hard problems faster. Despite their success when used with GBFS, such exploration techniques cannot be directly applied to bounded suboptimal algorithms like Weighted A* (WA*) without losing the solution-quality guarantees. In this work, we present Type-WA*, a novel bounded suboptimal planning algorithm that augments WA* with type-based exploration while still satisfying WA*'s theoretical solution-quality guarantee. Our empirical analysis shows that Type-WA* significantly increases the number of solved problems, when used in conjunction with each of three popular heuristics. Our analysis also provides insight into the runtime vs. solution cost trade-off.

----

## [557] Custom-Design of FDR Encodings: The Case of Red-Black Planning

**Authors**: *Daniel Fiser, Daniel Gnad, Michael Katz, Jörg Hoffmann*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/558](https://doi.org/10.24963/ijcai.2021/558)

**Abstract**:

Classical planning tasks are commonly described in PDDL, while most planning systems operate on a grounded finite-domain representation (FDR). The translation of PDDL into FDR is complex and has a lot of choice points---it involves identifying so called mutex groups---but most systems rely on the translator that comes with Fast Downward. Yet the translation choice points can strongly impact performance. Prior work has considered optimizing FDR encodings in terms of the number of variables produced. Here we go one step further by proposing to custom-design FDR encodings, optimizing the encoding to suit particular planning techniques. We develop such a custom design here for red-black planning, a partial delete relaxation technique. The FDR encoding affects the causal graph and the domain transition graph structures, which govern the tractable fragment of red-black planning and hence affects the respective heuristic function. We develop integer linear programming techniques optimizing the scope of that fragment in the resulting FDR encoding. We empirically show that the performance of red-black planning can be improved through such FDR custom design.

----

## [558] Active Goal Recognition Design

**Authors**: *Kevin C. Gall, Wheeler Ruml, Sarah Keren*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/559](https://doi.org/10.24963/ijcai.2021/559)

**Abstract**:

In Goal Recognition Design (GRD), the objective is to modify a domain to facilitate early detection of the goal of a subject agent.  Most previous work studies this problem in the offline setting, in which the observing agent performs its interventions before the subject begins acting.  In this paper, we generalize GRD to the online setting in which time passes and the observer's actions are interleaved with those of the subject. We illustrate weaknesses of existing metrics for GRD and propose an alternative better suited to online settings. We provide a formal definition of this Active GRD (AGRD) problem and study an algorithm for solving it.  AGRD occupies an interesting middle ground between passive goal recognition and strategic two-player game settings.

----

## [559] Stochastic Probing with Increasing Precision

**Authors**: *Martin Hoefer, Kevin Schewior, Daniel Schmand*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/560](https://doi.org/10.24963/ijcai.2021/560)

**Abstract**:

We consider a selection problem with stochastic probing. There is a set of items whose values are drawn from independent distributions. The distributions are known in advance. Each item can be \emph{tested} repeatedly. Each test reduces the uncertainty about the realization of its value. We study a testing model, where the first test reveals if the realized value is smaller or larger than the median of the underlying distribution. Subsequent tests allow to further  narrow down the interval in which the realization is located. There is a limited number of possible tests, and our goal is to design near-optimal testing strategies that allow to maximize the expected value of the chosen item. We study both identical and non-identical distributions and develop polynomial-time algorithms with constant approximation factors in both scenarios.

----

## [560] Incorporating Queueing Dynamics into Schedule-Driven Traffic Control

**Authors**: *Hsu-Chieh Hu, Allen M. Hawkes, Stephen F. Smith*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/561](https://doi.org/10.24963/ijcai.2021/561)

**Abstract**:

Key to the effectiveness of schedule-driven approaches to real-time traffic control is an ability to accurately predict when sensed vehicles will arrive at and pass through the intersection. Prior work in schedule-driven traffic control has assumed a static vehicle arrival model. However, this static predictive model ignores the fact that the queue count and the incurred delay should vary as different partial signal timing schedules (i.e., different possible futures) are explored during the online planning process. In this paper, we propose an alternative arrival time model that incorporates queueing dynamics into this forward search process for a signal timing schedule, to more accurately capture how the intersectionâ€™s queues vary over time. As each search state is generated, an incremental queueing delay is dynamically projected for each vehicle. The resulting total queueing delay is then considered in addition to the cumulative delay caused by signal operations. We demonstrate the potential of this approach through microscopic traffic simulation of a real-world road network, showing a 10-15% reduction in average wait times over the schedule-driven traffic signal control system in heavy traffic scenarios.

----

## [561] Symbolic Dynamic Programming for Continuous State MDPs with Linear Program Transitions

**Authors**: *Jihwan Jeong, Parth Jaggi, Scott Sanner*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/562](https://doi.org/10.24963/ijcai.2021/562)

**Abstract**:

Recent advances in symbolic dynamic programming (SDP) have significantly broadened the class of MDPs for which exact closed-form value functions can be derived. However, no existing solution methods can solve complex discrete and continuous state MDPs where a linear program determines state transitions --- transitions that are often required in problems with underlying constrained flow dynamics arising in problems ranging from traffic signal control to telecommunications bandwidth planning. In this paper, we present a novel SDP solution method for MDPs with LP transitions and continuous piecewise linear dynamics by introducing a novel, fully symbolic argmax operator. On three diverse domains, we show the first automated exact closed-form SDP solution to these challenging problems and the significant advantages of our SDP approach over discretized approximations.

----

## [562] Interference-free Walks in Time: Temporally Disjoint Paths

**Authors**: *Nina Klobas, George B. Mertzios, Hendrik Molter, Rolf Niedermeier, Philipp Zschoche*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/563](https://doi.org/10.24963/ijcai.2021/563)

**Abstract**:

We investigate the computational complexity of finding temporally disjoint paths or walks in temporal graphs. There, the edge set changes over discrete time steps and a temporal path (resp. walk) uses edges that appear at monotonically increasing time steps. Two paths (or walks) are temporally disjoint if they never use the same vertex at the same time; otherwise, they interfere. This reflects applications in robotics, traffic routing, or finding safe pathways in dynamically changing networks.

On the one extreme, we show that on general graphs the problem is computationally hard. The "walk version" is W[1]-hard when parameterized by the number of routes. However, it is polynomial-time solvable for any constant number of walks. The "path version" remains NP-hard even if we want to find only two temporally disjoint paths. On the other extreme, restricting the input temporal graph to have a path as underlying graph, quite counterintuitively, we find NP-hardness in general but also identify natural tractable cases.

----

## [563] Counterfactual Explanations for Optimization-Based Decisions in the Context of the GDPR

**Authors**: *Anton Korikov, Alexander Shleyfman, J. Christopher Beck*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/564](https://doi.org/10.24963/ijcai.2021/564)

**Abstract**:

The General Data Protection Regulations (GDPR) entitle individuals to explanations for automated decisions. The form, comprehensibility, and even existence of such explanations remain open problems, investigated as part of explainable AI. We adopt the approach of counterfactual explanations and apply it to decisions made by declarative optimization models. We argue that inverse combinatorial optimization is particularly suited for counterfactual explanations but that the computational difficulties and relatively nascent literature make its application a challenge. To make progress, we address the case of counterfactual explanations that isolate the minimal differences for an individual. We show that under two common optimization functions, full inverse optimization is unnecessary. In particular, we show that for functions of the form of the sum of weighted binary variables, which includes frameworks such as weighted MaxSAT, a solution can be found by solving a slightly modified version of the original optimization model. In contrast, the sum of weighted integer variables can be solved with a binary search over a series of modifications to the original model.

----

## [564] LTL-Constrained Steady-State Policy Synthesis

**Authors**: *Jan Kretínský*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/565](https://doi.org/10.24963/ijcai.2021/565)

**Abstract**:

Decision-making policies for agents are often synthesized with the constraint that a  formal specification of behaviour is satisfied. Here we focus on infinite-horizon properties. On the one hand,  Linear Temporal Logic (LTL) is a popular example of a formalism for qualitative specifications. On the other hand, Steady-State Policy Synthesis (SSPS) has recently received considerable attention as it provides a more quantitative and more behavioural perspective on specifications, in terms of the frequency with which states are visited. Finally, rewards provide a classic framework for quantitative properties.
In this paper, we study Markov decision processes (MDP) with the specification combining all these three types. The derived policy maximizes the reward among all policies ensuring the LTL specification with the given probability and adhering to the steady-state constraints. To this end, we provide a unified solution reducing the multi-type specification to a multi-dimensional long-run average reward. This is enabled by Limit-Deterministic Büchi Automata (LDBA), recently studied in the context of LTL model checking on MDP, and allows for an elegant solution through a simple linear programme. The algorithm also extends to the general omega-regular properties and runs in time polynomial in the sizes of the MDP as well as the LDBA.

----

## [565] Online Learning of Action Models for PDDL Planning

**Authors**: *Leonardo Lamanna, Alessandro Saetti, Luciano Serafini, Alfonso Gerevini, Paolo Traverso*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/566](https://doi.org/10.24963/ijcai.2021/566)

**Abstract**:

The automated learning of action models is widely recognised as a key and compelling challenge to address the difficulties of the manual specification of planning domains. Most state-of-the-art methods perform this learning offline from an input set of plan traces generated by the execution of (successful) plans.  However, how to generate informative plan traces for learning action models is still an open issue. Moreover, plan traces might not be available for a new environment. In this paper, we propose an algorithm for learning action models online, incrementally during the execution of plans. Such plans are generated to achieve goals that the algorithm decides online in order to obtain informative plan traces and reach states from which useful information can be learned. We show some fundamental theoretical properties of the algorithm, and we experimentally evaluate the online learning of the action models over a large set of IPC domains.

----

## [566] Polynomial-Time in PDDL Input Size: Making the Delete Relaxation Feasible for Lifted Planning

**Authors**: *Pascal Lauer, Álvaro Torralba, Daniel Fiser, Daniel Höller, Julia Wichlacz, Jörg Hoffmann*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/567](https://doi.org/10.24963/ijcai.2021/567)

**Abstract**:

Polynomial-time heuristic functions for planning are commonplace since 20 years. But polynomial-time in which input? Almost all existing approaches are based on a grounded task representation, not on the actual PDDL input which is exponentially smaller. This limits practical applicability to cases where the grounded representation is "small enough". Previous attempts to tackle this problem for the delete relaxation leveraged symmetries to reduce the blow-up. Here we take a more radical approach, applying an additional relaxation to obtain a heuristic function that runs in time polynomial in the size of the PDDL input. Our relaxation splits the predicates into smaller predicates of fixed arity K. We show that computing a relaxed plan is still NP-hard (in PDDL input size) for K>=2, but is polynomial-time for K=1. We implement a heuristic function for K=1 and show that it can improve the state of the art on benchmarks whose grounded representation is large.

----

## [567] Anytime Multi-Agent Path Finding via Large Neighborhood Search

**Authors**: *Jiaoyang Li, Zhe Chen, Daniel Harabor, Peter J. Stuckey, Sven Koenig*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/568](https://doi.org/10.24963/ijcai.2021/568)

**Abstract**:

Multi-Agent Path Finding (MAPF) is the challenging problem of computing collision-free paths for multiple agents. Algorithms for solving MAPF can be categorized on a spectrum. At one end are (bounded-sub)optimal algorithms that can find high-quality solutions for small problems. At the other end are unbounded-suboptimal algorithms that can solve large problems but usually find low-quality solutions. In this paper, we consider a third approach that combines the best of both worlds: anytime algorithms that quickly find an initial solution using efficient MAPF algorithms from the literature, even for large problems, and that subsequently improve the solution quality to near-optimal as time progresses by replanning subgroups of agents using Large Neighborhood Search. We compare our algorithm MAPF-LNS against a range of existing work and report significant gains in scalability, runtime to the initial solution, and speed of improving the solution.

----

## [568] Dynamic Rebalancing Dockless Bike-Sharing System based on Station Community Discovery

**Authors**: *Jingjing Li, Qiang Wang, Wenqi Zhang, Donghai Shi, Zhiwei Qin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/569](https://doi.org/10.24963/ijcai.2021/569)

**Abstract**:

Influenced by the era of the sharing economy and mobile payment, Dockless Bike-Sharing System (Dockless BSS) is expanding in many major cities. The mobility of users constantly leads to supply and demand imbalance, which seriously affects the total profit and customer satisfaction. In this paper, we propose the Spatio-Temporal Mixed Integer Program (STMIP) with Flow-graphed Community Discovery (FCD) approach to rebalancing the system. Different from existing studies that ignore the route of trucks and adopt a centralized rebalancing, our approach considers the spatio-temporal information of trucks and discovers station communities for truck-based rebalancing. First, we propose the FCD algorithm to detect station communities. Significantly, rebalancing communities decomposes the centralized system into a distributed multi-communities system. Then, by considering the routing and velocity of trucks, we design the STMIP model with the objective of maximizing total profit, to find a repositioning policy for each station community. We design a simulator built on real-world data from DiDi Chuxing to test the algorithm performance. The extensive experimental results demonstrate that our approach outperforms in terms of service level, profit, and complexity compared with the state-of-the-art approach.

----

## [569] Synthesizing Good-Enough Strategies for LTLf Specifications

**Authors**: *Yong Li, Andrea Turrini, Moshe Y. Vardi, Lijun Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/570](https://doi.org/10.24963/ijcai.2021/570)

**Abstract**:

We consider the problem of synthesizing good-enough (GE)-strategies for linear temporal logic (LTL) over finite traces or LTLf for short.
The problem of synthesizing GE-strategies for an LTL formula φ over infinite traces reduces to the problem of synthesizing winning strategies for the formula (∃Oφ)⇒φ where O is the set of propositions controlled by the system.
We first prove that this reduction does not work for LTLf formulas.
Then we show how to synthesize GE-strategies for LTLf formulas via the Good-Enough (GE)-synthesis of LTL formulas.
Unfortunately, this requires to construct deterministic parity automata on infinite words, which is computationally expensive.
We then show how to synthesize GE-strategies for LTLf formulas by a reduction to solving games played on deterministic Büchi automata, based on an easier construction of deterministic automata on finite words.
We show empirically that our specialized synthesis algorithm for GE-strategies outperforms the algorithms going through GE-synthesis of LTL formulas by orders of magnitude.

----

## [570] Change the World - How Hard Can that Be? On the Computational Complexity of Fixing Planning Models

**Authors**: *Songtuan Lin, Pascal Bercher*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/571](https://doi.org/10.24963/ijcai.2021/571)

**Abstract**:

Incorporating humans into AI planning is an important feature of flexible planning technology. Such human integration allows to incorporate previously unknown constraints, and is also an integral part of automated modeling assistance. As a foundation for integrating user requests, we study the computational complexity of determining the existence of changes to an existing model, such that the resulting model allows for specific user-provided solutions. We are provided with a planning problem modeled either in the classical (non-hierarchical) or hierarchical task network (HTN) planning formalism, as well as with a supposed-to-be solution plan, which is actually not a solution for the current model. Considering changing decomposition methods as well as preconditions and effects of actions, we show that most change requests are NP-complete though some turn out to be tractable.

----

## [571] Learning Temporal Plan Preferences from Examples: An Empirical Study

**Authors**: *Valentin Seimetz, Rebecca Eifler, Jörg Hoffmann*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/572](https://doi.org/10.24963/ijcai.2021/572)

**Abstract**:

Temporal plan preferences are natural and important in a variety of applications. Yet users often find it difficult to formalize their preferences. Here we explore the possibility to learn preferences from example plans. Focusing on one preference at a time, the user is asked to annotate examples as good/bad. We leverage prior work on LTL formula learning to extract a preference from these examples. We conduct an empirical study of this approach in an oversubscription planning context, using hidden target formulas to emulate the user preferences. We explore four different methods for generating example plans, and evaluate performance as a function of domain and formula size. Overall, we find that reasonable-size target formulas can often be learned effectively.

----

## [572] On Weak Stubborn Sets in Classical Planning

**Authors**: *Silvan Sievers, Martin Wehrle*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/573](https://doi.org/10.24963/ijcai.2021/573)

**Abstract**:

Stubborn sets are a pruning technique for state-space search which is well established in optimal classical planning. In this paper, we show that weak stubborn sets introduced in recent work in planning are actually not weak stubborn sets in Valmari's original sense. Based on this finding, we introduce weak stubborn sets in the original sense for planning by providing a generalized definition analogously to generalized strong stubborn sets in previous work. We discuss the relationship of strong, weak and the previously called weak stubborn sets, thus providing a further step in getting an overall picture of the stubborn set approach in planning.

----

## [573] Learning Generalized Unsolvability Heuristics for Classical Planning

**Authors**: *Simon Ståhlberg, Guillem Francès, Jendrik Seipp*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/574](https://doi.org/10.24963/ijcai.2021/574)

**Abstract**:

Recent work in classical planning has introduced dedicated techniques for detecting unsolvable states, i.e., states from which no goal state can be reached. We approach the problem from a generalized planning perspective and learn first-order-like formulas that characterize unsolvability for entire planning domains. We show how to cast the problem as a self-supervised classification task. Our training data is automatically generated and labeled by exhaustive exploration of small instances of each domain, and candidate features are automatically computed from the predicates used to define the domain. We investigate three learning algorithms with different properties and compare them to heuristics from the literature. Our empirical results show that our approach often captures important classes of unsolvable states with high classification accuracy. Additionally, the logical form of our heuristics makes them easy to interpret and reason about, and can be used to show that the characterizations learned in some domains capture exactly all unsolvable states of the domain.

----

## [574] Solving Partially Observable Stochastic Shortest-Path Games

**Authors**: *Petr Tomásek, Karel Horák, Aditya Aradhye, Branislav Bosanský, Krishnendu Chatterjee*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/575](https://doi.org/10.24963/ijcai.2021/575)

**Abstract**:

We study the two-player zero-sum extension of the partially observable stochastic shortest-path problem where one agent has only partial information about the environment.
We formulate this problem as a partially observable stochastic game (POSG): given a set of target states and negative rewards for each transition, the player with imperfect information maximizes the expected undiscounted total reward until a target state is reached. The second player with the perfect information aims for the opposite. 
We base our formalism on POSGs with one-sided observability (OS-POSGs) and give the following contributions:
(1) we introduce a novel heuristic search value iteration algorithm that iteratively solves depth-limited variants of the game,
(2) we derive the bound on the depth guaranteeing an arbitrary precision, (3) we propose a novel upper-bound estimation that allows early terminations, and
(4) we experimentally evaluate the algorithm on a pursuit-evasion game.

----

## [575] The Fewer the Merrier: Pruning Preferred Operators with Novelty

**Authors**: *Alexander Tuisov, Michael Katz*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/576](https://doi.org/10.24963/ijcai.2021/576)

**Abstract**:

Heuristic search is among the best performing approaches to classical satisficing planning, with its performance heavily relying on informative and fast heuristics, as well as search-boosting and pruning techniques. While both heuristics and pruning techniques have gained much attention recently, search-boosting techniques in general, and preferred operators in particular have received less attention in the last decade.  
Our work aims at bringing the light back to preferred operators research, with the introduction of preferred operators pruning technique, based on the concept of novelty. Continuing the research on novelty with respect to an underlying heuristic, we present the definition of preferred operators for such novelty heuristics. For that, we extend the previously defined concepts to operators, allowing us to reason about the novelty of the preferred operators.

Our experimental evaluation shows the practical benefit of our suggested
approach, compared to the currently used methods.

----

## [576] TANGO: Commonsense Generalization in Predicting Tool Interactions for Mobile Manipulators

**Authors**: *Shreshth Tuli, Rajas Bansal, Rohan Paul, Mausam*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/577](https://doi.org/10.24963/ijcai.2021/577)

**Abstract**:

Robots assisting us in factories or homes must learn to make use of objects as tools to perform tasks, e.g., a tray for carrying objects. We consider the problem of learning commonsense knowledge of when a tool may be useful and how its use may be composed with other tools to accomplish a high-level task instructed by a human. We introduce TANGO, a novel neural model for predicting task-specific tool interactions. TANGO is trained using demonstrations obtained from human teachers instructing a virtual robot in a physics simulator. TANGO encodes the world state consisting of objects and symbolic relationships between them using a graph neural network. The model learns to attend over the scene using knowledge of the goal and the action history, finally decoding the symbolic action to execute. Crucially, we address generalization to unseen environments where some known tools are missing, but alternative unseen tools are present. We show that by augmenting the representation of the environment with pre-trained embeddings derived from a knowledge-base, the model can generalize effectively to novel environments. Experimental results show a 60.5-78.9% improvement over the baseline in predicting successful symbolic plans in unseen settings for a simulated mobile manipulator.

----

## [577] The Traveling Tournament Problem with Maximum Tour Length Two: A Practical Algorithm with An Improved Approximation Bound

**Authors**: *Jingyang Zhao, Mingyu Xiao*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/578](https://doi.org/10.24963/ijcai.2021/578)

**Abstract**:

The Traveling Tournament Problem is a well-known benchmark problem in tournament timetabling, which asks us to design a schedule of home/away games of n teams (n is even) under some feasibility requirements such that the total traveling distance of all the n teams is minimized. In this paper, we study TTP-2, the traveling tournament problem where at most two consecutive home games or away games are allowed, and give an effective algorithm for n/2 being odd. Experiments on the well-known benchmark sets show that we can beat previously known solutions for all instances with n/2 being odd by an average improvement of 2.66%. Furthermore, we improve the theoretical approximation ratio from 3/2+O(1/n) to 1+O(1/n) for n/2 being odd, answering a challenging open problem in this area.

----

## [578] Non-Parametric Stochastic Sequential Assignment With Random Arrival Times

**Authors**: *Danial Dervovic, Parisa Hassanzadeh, Samuel Assefa, Prashant Reddy*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/579](https://doi.org/10.24963/ijcai.2021/579)

**Abstract**:

We consider a problem wherein jobs arrive at random times and assume random values. Upon each job arrival, the decision-maker must decide immediately whether or not to accept the job and gain the value on offer as a reward, with the constraint that they may only accept at most n jobs over some reference time period. The decision-maker only has access to M independent realisations of the job arrival process. We propose an algorithm, Non-Parametric Sequential Allocation (NPSA), for solving this problem. Moreover, we prove that the expected reward returned by the NPSA algorithm converges in probability to optimality as M grows large. We demonstrate the effectiveness of the algorithm empirically on synthetic data and on public fraud-detection datasets, from where the motivation for this work is derived.

----

## [579] On the Parameterized Complexity of Polytree Learning

**Authors**: *Niels Grüttemeier, Christian Komusiewicz, Nils Morawietz*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/580](https://doi.org/10.24963/ijcai.2021/580)

**Abstract**:

A Bayesian network is a directed acyclic graph that represents statistical dependencies between variables of a joint probability distribution. A fundamental task in data science is to learn a Bayesian network from observed data. Polytree Learning is the problem of learning an optimal Bayesian network that fulfills the additional property that its underlying undirected graph is a forest. In this work, we revisit the complexity of Polytree Learning. We show that Polytree Learning can be solved in single-exponential FPT time for the number of variables. Moreover, we consider the influence of d, the number of variables that might receive a nonempty parent set in the final DAG on the complexity of Polytree Learning. We show that Polytree Learning is presumably not fixed-parameter tractable for d, unlike Bayesian network learning which is fixed-parameter tractable for d. Finally, we show that if d and the maximum parent set size are bounded, then we can obtain efficient algorithms.

----

## [580] Handling Overlaps When Lifting Gaussian Bayesian Networks

**Authors**: *Mattis Hartwig, Tanya Braun, Ralf Möller*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/581](https://doi.org/10.24963/ijcai.2021/581)

**Abstract**:

Gaussian Bayesian networks are widely used for modeling the behavior of continuous random variables. Lifting exploits symmetries when dealing with large numbers of isomorphic random variables. It provides a more compact representation for more efficient query answering by encoding the symmetries using logical variables. This paper improves on an existing lifted representation of the joint distribution represented by a Gaussian Bayesian network (lifted joint), allowing overlaps between the logical variables. Handling overlaps without grounding a model is critical for modelling real-world scenarios.
Specifically, this paper contributes (i) a lifted joint that allows overlaps in logical variables and (ii) a lifted query answering algorithm using the lifted joint. Complexity analyses and experimental results show that - despite overlaps - constructing a lifted joint and answering queries on the lifted joint outperform their grounded counterparts significantly.

----

## [581] Deep Bucket Elimination

**Authors**: *Yasaman Razeghi, Kalev Kask, Yadong Lu, Pierre Baldi, Sakshi Agarwal, Rina Dechter*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/582](https://doi.org/10.24963/ijcai.2021/582)

**Abstract**:

Bucket Elimination (BE) is a universal inference scheme that can solve most tasks over probabilistic and deterministic graphical models exactly.
However, it often requires exponentially high levels of memory (in the induced-width) preventing its execution. In the spirit of exploiting Deep Learning for inference tasks, in this paper, we will use neural networks to approximate BE. 
The resulting Deep Bucket Elimination (DBE) algorithm is developed for computing the partition function.
We provide a proof-of-concept empirically using instances from several different benchmarks, showing that DBE can be a more accurate approximation than current state-of-the-art approaches for approximating  BE (e.g. the mini-bucket schemes), especially when problems are sufficiently hard.

----

## [582] BKT-POMDP: Fast Action Selection for User Skill Modelling over Tasks with Multiple Skills

**Authors**: *Nicole Salomons, Emir Akdere, Brian Scassellati*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/583](https://doi.org/10.24963/ijcai.2021/583)

**Abstract**:

Creating an accurate model of a user's skills is necessary for intelligent tutoring systems. Without an accurate model, sample problems or tasks must be selected haphazardly by the tutor. Once an accurate model has been trained, the tutor can selectively focus on training essential or deficient skills. Prior work offers mechanisms for optimizing the training of a single skill or for multiple skills when individual tasks involve testing only a single skill at a time, but not for multiple skills when individual tasks can contain evidence for multiple skills. In this paper, we present a system that estimates user skill models for multiple skills by selecting tasks which maximize the information gain across the entire skill model. We compare our system's policy against several baselines and an optimal policy in both simulated and real tasks. Our system outperforms baselines and performs almost on par with the optimal policy.

----

## [583] Improved Acyclicity Reasoning for Bayesian Network Structure Learning with Constraint Programming

**Authors**: *Fulya Trösser, Simon de Givry, George Katsirelos*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/584](https://doi.org/10.24963/ijcai.2021/584)

**Abstract**:

Bayesian networks are probabilistic graphical models with a wide
range of application areas including gene regulatory networks inference, risk
analysis and image processing. Learning the structure of a Bayesian
network (BNSL) from discrete data is known to be an NP-hard task with
a superexponential search space of directed acyclic graphs.  In this
work, we propose a new polynomial time algorithm for discovering a
subset of all possible cluster cuts, a greedy algorithm for
approximately solving the resulting linear program, and a generalized
arc consistency algorithm for the acyclicity constraint.  We embed
these in the constraint programming-based branch-and-bound solver
CPBayes and show that, despite being suboptimal, they improve
performance by orders of magnitude. The resulting solver also compares
favorably with GOBNILP, a state-of-the-art solver for the BNSL
problem which solves an NP-hard problem to discover each cut and
solves the linear program exactly.

----

## [584] Provable Guarantees on the Robustness of Decision Rules to Causal Interventions

**Authors**: *Benjie Wang, Clare Lyle, Marta Kwiatkowska*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/585](https://doi.org/10.24963/ijcai.2021/585)

**Abstract**:

Robustness of decision rules to shifts in the data-generating process is crucial to the successful deployment of decision-making systems. Such shifts can be viewed as interventions on a causal graph, which capture (possibly hypothetical) changes in the data-generating process, whether due to natural reasons or by the action of an adversary. We consider causal Bayesian networks and formally define the interventional robustness problem, a novel model-based notion of robustness for decision functions that measures worst-case performance with respect to a set of interventions that denote changes to parameters and/or causal influences. By relying on a tractable representation of Bayesian networks as arithmetic circuits, we provide efficient algorithms for computing guaranteed upper and lower bounds on the interventional robustness probabilities. Experimental results demonstrate that the methods yield useful and interpretable bounds for a range of practical networks, paving the way towards provably causally robust decision-making systems.

----

## [585] Fast Algorithms for Relational Marginal Polytopes

**Authors**: *Yuanhong Wang, Timothy van Bremen, Juhua Pu, Yuyi Wang, Ondrej Kuzelka*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/586](https://doi.org/10.24963/ijcai.2021/586)

**Abstract**:

We study the problem of constructing the relational marginal polytope (RMP) of a given set of first-order formulas. Past work has shown that the RMP construction problem can be reduced to weighted first-order model counting (WFOMC). However, existing reductions in the literature are intractable in practice, since they typically require an infeasibly large number of calls to a WFOMC oracle. In this paper, we propose an algorithm to construct RMPs using fewer oracle calls. As an application, we also show how to apply this new algorithm to improve an existing approximation scheme for WFOMC. We demonstrate the efficiency of the proposed approaches experimentally, and find that our method provides speed-ups over the baseline for RMP construction of a full order of magnitude.

----

## [586] Partition Function Estimation: A Quantitative Study

**Authors**: *Durgesh Agrawal, Yash Pote, Kuldeep S. Meel*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/587](https://doi.org/10.24963/ijcai.2021/587)

**Abstract**:

Probabilistic graphical models have emerged as a powerful modeling tool for several real-world scenarios where one needs to reason under uncertainty. A graphical model's partition function is a central quantity of interest, and its computation is key to several probabilistic reasoning tasks. Given the #P-hardness of computing the partition function, several techniques have been proposed over the years with varying guarantees on the quality of estimates and their runtime behavior. This paper seeks to present a survey of 18 techniques and a rigorous empirical study of their behavior across an extensive set of benchmarks. Our empirical study draws up a surprising observation: exact techniques are as efficient as the approximate ones, and therefore, we conclude with an optimistic view of opportunities for the design of approximate techniques with enhanced scalability. Motivated by the observation of an order of magnitude difference between the Virtual Best Solver and the best performing tool, we envision an exciting line of research focused on the development of portfolio solvers.

----

## [587] A Survey of Machine Learning-Based Physics Event Generation

**Authors**: *Yasir Alanazi, Nobuo Sato, Pawel Ambrozewicz, Astrid N. Hiller Blin, Wally Melnitchouk, Marco Battaglieri, Tianbo Liu, Yaohang Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/588](https://doi.org/10.24963/ijcai.2021/588)

**Abstract**:

Event generators in high-energy nuclear and particle physics play an important role in facilitating studies of particle reactions. We survey the state of the art of machine learning (ML) efforts at building physics event generators. We review ML generative models used in ML-based event generators and their specific challenges, and discuss various approaches of incorporating physics into the ML model designs to overcome these challenges. Finally, we explore some open questions related to super-resolution, fidelity, and extrapolation for physics event generation based on ML technology.

----

## [588] Distortion in Social Choice Problems: The First 15 Years and Beyond

**Authors**: *Elliot Anshelevich, Aris Filos-Ratsikas, Nisarg Shah, Alexandros A. Voudouris*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/589](https://doi.org/10.24963/ijcai.2021/589)

**Abstract**:

The notion of distortion in social choice problems has been defined to measure the loss in efficiency---typically measured by the utilitarian social welfare, the sum of utilities of the participating agents---due to having access only to limited information about the preferences of the agents. We survey the most significant results of the literature on distortion from the past 15 years, and highlight important open problems and the most promising avenues of ongoing and future work.

----

## [589] Building Affordance Relations for Robotic Agents - A Review

**Authors**: *Paola Ardón, Èric Pairet, Katrin S. Lohan, Subramanian Ramamoorthy, Ron P. A. Petrick*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/590](https://doi.org/10.24963/ijcai.2021/590)

**Abstract**:

Affordances describe the possibilities for an agent to perform actions with an object. While the significance of the affordance concept has been previously studied from varied perspectives, such as psychology and cognitive science, these approaches are not always sufficient to enable direct transfer, in the sense of implementations, to artificial intelligence (AI)-based systems and robotics. However, many efforts have been made to pragmatically employ the concept of affordances, as it represents great potential for AI agents to effectively bridge perception to action. In this survey, we review and find common ground amongst different strategies that use the concept of affordances within robotic tasks, and build on these methods to provide guidance for including affordances as a mechanism to improve autonomy. To this end, we outline common design choices for building representations of affordance relations, and their implications on the generalisation capabilities of an agent when facing previously unseen scenarios. Finally, we identify and discuss a range of interesting research directions involving affordances that have the potential to improve the capabilities of an AI agent.

----

## [590] Recent Advances in Adversarial Training for Adversarial Robustness

**Authors**: *Tao Bai, Jinqi Luo, Jun Zhao, Bihan Wen, Qian Wang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/591](https://doi.org/10.24963/ijcai.2021/591)

**Abstract**:

Adversarial training is one of the most effective approaches for deep learning models to defend against adversarial examples.
Unlike other defense strategies, adversarial training aims to enhance the robustness of models intrinsically.
During the past few years, adversarial training has been studied and discussed from various aspects, which deserves a comprehensive review.
For the first time in this survey, we systematically review the recent progress on adversarial training for adversarial robustness with a novel taxonomy.
Then we discuss the generalization problems in adversarial training from three perspectives and highlight the challenges which are not fully tackled.
Finally, we present potential future directions.

----

## [591] Hardware-Aware Neural Architecture Search: Survey and Taxonomy

**Authors**: *Hadjer Benmeziane, Kaoutar El Maghraoui, Hamza Ouarnoughi, Smaïl Niar, Martin Wistuba, Naigang Wang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/592](https://doi.org/10.24963/ijcai.2021/592)

**Abstract**:

There is no doubt that making AI mainstream by bringing powerful, yet power hungry deep neural networks (DNNs) to resource-constrained devices would required an efficient co-design of algorithms, hardware and software. The increased popularity of DNN applications deployed on a wide variety of platforms, from tiny microcontrollers to data-centers, have resulted in multiple questions and challenges related to constraints introduced by the hardware. In this survey on hardware-aware neural architecture search (HW-NAS), we present some of the existing answers proposed in the literature for the following questions: "Is it possible to build an efficient DL model that meets the latency and energy constraints of tiny edge devices?", "How can we reduce the trade-off between the accuracy of a DL model and its ability to be deployed in a variety of platforms?". The survey provides a new taxonomy of HW-NAS and assesses the hardware cost estimation strategies. We also highlight the challenges and limitations of existing approaches and potential future directions. 
We hope that this survey will help to fuel the research towards efficient deep learning.

----

## [592] Recent Trends in Word Sense Disambiguation: A Survey

**Authors**: *Michele Bevilacqua, Tommaso Pasini, Alessandro Raganato, Roberto Navigli*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/593](https://doi.org/10.24963/ijcai.2021/593)

**Abstract**:

Word Sense Disambiguation (WSD) aims at making explicit the semantics of a word in context by identifying the most suitable meaning from a predefined sense inventory. Recent breakthroughs in representation learning have fueled intensive WSD research, resulting in considerable performance improvements, breaching the 80% glass ceiling set by the inter-annotator agreement. In this survey, we provide an extensive overview of current advances in WSD, describing the state of the art in terms of i) resources for the task, i.e., sense inventories and reference datasets for training and testing, as well as ii) automatic disambiguation approaches, detailing their peculiarities, strengths and weaknesses. Finally, we highlight the current limitations of the task itself, but also point out recent trends that could help expand the scope and applicability of WSD, setting up new promising directions for the future.

----

## [593] When Computational Representation Meets Neuroscience: A Survey on Brain Encoding and Decoding

**Authors**: *Lu Cao, Dandan Huang, Yue Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/594](https://doi.org/10.24963/ijcai.2021/594)

**Abstract**:

Real human language mechanisms and the artificial intelligent language processing methods are two independent systems. Exploring the relationship between the two can help develop human-like language models and is also beneficial to reveal the neuroscience of the reading brain. The flourishing research in this interdisciplinal research field calls for surveys to systemically study and analyze the recent successes. However, such a comprehensive review still cannot be found, which motivates our work. This article first briefly introduces the interdisciplinal research progress, then systematically discusses the task of brain decoding from the perspective of simple concepts and complete sentences, and also describes main limitations in this field and put forward with possible solutions. Finally, we conclude this survey with certain open research questions that will stimulate further studies.

----

## [594] Combinatorial Optimization and Reasoning with Graph Neural Networks

**Authors**: *Quentin Cappart, Didier Chételat, Elias B. Khalil, Andrea Lodi, Christopher Morris, Petar Velickovic*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/595](https://doi.org/10.24963/ijcai.2021/595)

**Abstract**:

Combinatorial optimization is a well-established area in operations research and computer science. Until recently, its methods have mostly focused on solving problem instances in isolation, ignoring the fact that they often stem from related data distributions in practice. However, recent years have seen a surge of interest in using machine learning, especially graph neural networks, as a key building block for combinatorial tasks, either directly as solvers or by enhancing the former. This paper presents a conceptual review of recent key advancements in this emerging field, aiming at researchers in both optimization and machine learning.

----

## [595] Mechanism Design for Facility Location Problems: A Survey

**Authors**: *Hau Chan, Aris Filos-Ratsikas, Bo Li, Minming Li, Chenhao Wang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/596](https://doi.org/10.24963/ijcai.2021/596)

**Abstract**:

The study of approximate mechanism design for facility location has been in the center of research at the intersection of artificial intelligence and economics for the last decade, largely due to its practical importance in various domains, such as social planning and clustering.  At a high level, the goal is to select a number of locations on which to build a set of  facilities, aiming to optimize some social objective based on the preferences of strategic agents, who might have incentives to misreport their private information. This paper presents a comprehensive survey of the significant progress that has been made since the introduction of the problem, highlighting all the different variants and methodologies, as well as the most interesting directions for future research.

----

## [596] Knowledge-aware Zero-Shot Learning: Survey and Perspective

**Authors**: *Jiaoyan Chen, Yuxia Geng, Zhuo Chen, Ian Horrocks, Jeff Z. Pan, Huajun Chen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/597](https://doi.org/10.24963/ijcai.2021/597)

**Abstract**:

Zero-shot learning (ZSL) which aims at predicting classes that have never appeared during the training using external knowledge (a.k.a. side information) has been widely investigated. In this paper we present a literature review towards ZSL in the perspective of external knowledge, where we categorize the external knowledge, review their  methods and compare different external knowledge. With the literature review, we further discuss and outlook the role of symbolic knowledge in addressing ZSL and other machine learning sample shortage issues.

----

## [597] Causal Learning for Socially Responsible AI

**Authors**: *Lu Cheng, Ahmadreza Mosallanezhad, Paras Sheth, Huan Liu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/598](https://doi.org/10.24963/ijcai.2021/598)

**Abstract**:

There have been increasing concerns about Artificial Intelligence (AI) due to its unfathomable potential power. To make AI address ethical challenges and shun undesirable outcomes, researchers proposed to develop socially responsible AI (SRAI). One of these approaches is causal learning (CL). We survey state-of-the-art methods of CL for SRAI. We begin by examining the seven CL tools to enhance the social responsibility of AI,  then review how existing works have succeeded using these tools to tackle issues in developing SRAI such as fairness. The goal of this survey is to bring forefront the potentials and promises of CL for SRAI.

----

## [598] Understanding the Relationship between Interactions and Outcomes in Human-in-the-Loop Machine Learning

**Authors**: *Yuchen Cui, Pallavi Koppol, Henny Admoni, Scott Niekum, Reid G. Simmons, Aaron Steinfeld, Tesca Fitzgerald*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/599](https://doi.org/10.24963/ijcai.2021/599)

**Abstract**:

Human-in-the-loop Machine Learning (HIL-ML) is a widely adopted paradigm for instilling human knowledge in autonomous agents. Many design choices influence the efficiency and effectiveness of such interactive learning processes, particularly the interaction type through which the human teacher may provide feedback. While different interaction types (demonstrations, preferences, etc.) have been proposed and evaluated in the HIL-ML literature, there has been little discussion of how these compare or how they should be selected to best address a particular learning problem. In this survey, we propose an organizing principle for HIL-ML that provides a way to analyze the effects of interaction types on human performance and training data. We also identify open problems in understanding the effects of interaction types.

----

## [599] Argumentative XAI: A Survey

**Authors**: *Kristijonas Cyras, Antonio Rago, Emanuele Albini, Pietro Baroni, Francesca Toni*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/600](https://doi.org/10.24963/ijcai.2021/600)

**Abstract**:

Explainable AI (XAI) has been investigated for decades and, together with AI itself, has witnessed unprecedented growth in recent years. Among various approaches to XAI, argumentative models have been advocated in both the AI and social science literature, as their dialectical nature appears to match some basic desirable features of the explanation activity. In this survey we overview XAI approaches built using methods from the field of computational argumentation, leveraging its wide array of reasoning abstractions and explanation delivery methods. We overview the literature focusing on different types of explanation (intrinsic and post-hoc), different models with which argumentation-based explanations are deployed, different forms of delivery, and different argumentation frameworks they use. We also lay out a roadmap for future work.

----



[Go to the previous page](IJCAI-2021-list02.md)

[Go to the next page](IJCAI-2021-list04.md)

[Go to the catalog section](README.md)