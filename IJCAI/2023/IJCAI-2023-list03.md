## [400] DeepPSL: End-to-End Perception and Reasoning

**Authors**: *Sridhar Dasaratha, Sai Akhil Puranam, Karmvir Singh Phogat, Sunil Reddy Tiyyagura, Nigel P. Duffy*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/401](https://doi.org/10.24963/ijcai.2023/401)

**Abstract**:

We introduce DeepPSL a variant of probabilistic soft logic (PSL) to produce an end-to-end trainable system that integrates reasoning and perception. PSL represents first-order logic in terms of a convex graphical model â€“ hinge-loss Markov random fields (HL-MRFs). PSL stands out among probabilistic logic frameworks due to its tractability having been applied to systems of more than 1 billion ground rules. The key to our approach is to represent predicates in first-order logic using deep neural networks and then to approximately back-propagate through the HL-MRF and thus train every aspect of the first-order system being represented. We believe that this approach represents an interesting direction for the integration of deep learning and reasoning techniques with applications to knowledge base learning, multi-task learning, and explainability. Evaluation on three different tasks demonstrates that DeepPSL significantly outperforms state-of-the-art neuro-symbolic methods on scalability while achieving comparable or better accuracy.

----

## [401] Scalable Coupling of Deep Learning with Logical Reasoning

**Authors**: *Marianne Defresne, Sophie Barbe, Thomas Schiex*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/402](https://doi.org/10.24963/ijcai.2023/402)

**Abstract**:

In the ongoing quest for hybridizing discrete reasoning with neural nets, there is an increasing interest in neural architectures that can learn how to solve discrete reasoning or optimization problems from natural inputs. In this paper, we introduce a scalable neural architecture and loss function dedicated to learning the constraints and criteria of NP-hard reasoning problems expressed as discrete Graphical Models. We empirically show our loss function is able to efficiently learn how to solve NP-hard reasoning problems from natural inputs as the symbolic, visual or many-solutions Sudoku problems as well as the energy optimization formulation of the protein design problem, providing data efficiency, interpretability, and a posteriori control over predictions.

----

## [402] Neuro-Symbolic Class Expression Learning

**Authors**: *Caglar Demir, Axel-Cyrille Ngonga Ngomo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/403](https://doi.org/10.24963/ijcai.2023/403)

**Abstract**:

Models computed using deep learning have been effectively applied to tackle various problems in many disciplines. Yet, the predictions of these models are often at most post-hoc and locally explainable. 
In contrast, class expressions in description logics are ante-hoc and globally explainable. Although state-of-the-art symbolic machine learning approaches are being successfully applied to learn class expressions, their application at large scale has been hindered by their impractical runtimes. Arguably, the reliance on myopic heuristic functions contributes to this limitation. We propose a novel neuro-symbolic class expression learning model, DRILL, to mitigate this limitation. By learning non-myopic heuristic functions with deep Q-learning, DRILL efficiently steers the standard search procedure in a quasi-ordered search space towards goal states. Our extensive experiments on 4 benchmark datasets and 390 learning problems suggest that DRILL converges to goal states at least 2.7 times faster than state-of-the-art models on all learning problems. The results of our statistical significance test confirms that DRILL converges to goal states significantly faster (p-value <1%) than state-of-the-art models on all benchmark datasets. We provide an open-source implementation of DRILL, including pre-trained models, training and evaluation scripts.

----

## [403] Bidirectional Dilation Transformer for Multispectral and Hyperspectral Image Fusion

**Authors**: *Shangqi Deng, Liang-Jian Deng, Xiao Wu, Ran Ran, Rui Wen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/404](https://doi.org/10.24963/ijcai.2023/404)

**Abstract**:

Transformer-based methods have proven to be effective in achieving long-distance modeling, capturing the spatial and spectral information, and exhibiting strong inductive bias in various computer vision tasks. Generally, the Transformer model includes two common modes of multi-head self-attention (MSA): spatial MSA (Spa-MSA) and spectral MSA (Spe-MSA). However, Spa-MSA is computationally efficient but limits the global spatial response within a local window. On the other hand, Spe-MSA can calculate channel self-attention to accommodate high-resolution images, but it disregards the crucial local information that is essential for low-level vision tasks. In this study, we propose a bidirectional dilation Transformer (BDT) for multispectral and hyperspectral image fusion (MHIF), which aims to leverage the advantages of both MSA and the latent multiscale information specific to MHIF tasks. The BDT consists of two designed modules: the dilation Spa-MSA (D-Spa), which dynamically expands the spatial receptive field through a given hollow strategy, and the grouped Spe-MSA (G-Spe), which extracts latent features within the feature map and learns local data behavior. Additionally, to fully exploit the multiscale information from both inputs with different spatial resolutions, we employ a bidirectional hierarchy strategy in the BDT, resulting in improved performance. Finally, extensive experiments on two commonly used datasets, CAVE and Harvard, demonstrate the superiority of BDT both visually and quantitatively. Furthermore, the related code will be available at the GitHub page of the authors.

----

## [404] Understanding the Generalization Ability of Deep Learning Algorithms: A Kernelized Rényi's Entropy Perspective

**Authors**: *Yuxin Dong, Tieliang Gong, Hong Chen, Chen Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/405](https://doi.org/10.24963/ijcai.2023/405)

**Abstract**:

Recently, information-theoretic analysis has become a popular framework for understanding the generalization behavior of deep neural networks. It allows a direct analysis for stochastic gradient / Langevin descent (SGD/SGLD) learning algorithms without strong assumptions such as Lipschitz or convexity conditions. However, the current generalization error bounds within this framework are still far from optimal, while substantial improvements on these bounds are quite challenging due to the intractability of high-dimensional information quantities. To address this issue,  we first propose a novel information theoretical measure: kernelized Rényi's entropy, by utilizing operator representation in Hilbert space. It inherits the properties of Shannon's entropy and can be effectively calculated via simple random sampling, while remaining independent of the input dimension. We then establish the generalization error bounds for SGD/SGLD under kernelized Rényi's entropy, where the mutual information quantities can be directly calculated, enabling evaluation of the tightness of each intermediate step. We show that our information-theoretical bounds depend on the statistics of the stochastic gradients evaluated along with the iterates, and are rigorously tighter than the current state-of-the-art (SOTA) results. The theoretical findings are also supported by large-scale empirical studies.

----

## [405] ActUp: Analyzing and Consolidating tSNE and UMAP

**Authors**: *Andrew Draganov, Jakob Rødsgaard Jørgensen, Katrine Scheel, Davide Mottin, Ira Assent, Tyrus Berry, Çigdem Aslay*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/406](https://doi.org/10.24963/ijcai.2023/406)

**Abstract**:

TSNE and UMAP are popular dimensionality reduction algorithms due to their speed and interpretable low-dimensional embeddings. Despite their popularity, however, little work has been done to study their full span of differences. We theoretically and experimentally evaluate the space of parameters in the TSNE and UMAP algorithms and observe that a single one -- the normalization -- is responsible for switching between them. This, in turn, implies that a majority of the algorithmic differences can be toggled without affecting the embeddings. We discuss the implications this has on several theoretic claims behind UMAP, as well as how to reconcile them with existing TSNE interpretations.

Based on our analysis, we provide a method (GDR) that combines previously incompatible techniques from TSNE and UMAP and can replicate the results of either algorithm. This allows our method to incorporate further improvements, such as an acceleration that obtains either method's outputs faster than UMAP. We release improved versions of TSNE, UMAP, and GDR that are fully plug-and-play with the traditional libraries.

----

## [406] Automatic Truss Design with Reinforcement Learning

**Authors**: *Weihua Du, Jinglun Zhao, Chao Yu, Xingcheng Yao, Zimeng Song, Siyang Wu, Ruifeng Luo, Zhiyuan Liu, Xianzhong Zhao, Yi Wu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/407](https://doi.org/10.24963/ijcai.2023/407)

**Abstract**:

Truss layout design, namely finding a lightweight truss layout satisfying all the physical constraints, is a fundamental problem in the building industry. Generating the optimal layout is a challenging combinatorial optimization problem, which can be extremely expensive to solve by exhaustive search. Directly applying end-to-end reinforcement learning (RL) methods to truss layout design is infeasible either, since only a tiny portion of the entire layout space is valid under the physical constraints, leading to particularly sparse rewards for RL training.
In this paper, we develop AutoTruss, a two-stage framework to efficiently generate both lightweight and valid truss layouts. AutoTruss first adopts Monte Carlo tree search to discover a diverse collection of valid layouts. Then RL is applied to iteratively refine the valid solutions. We conduct experiments and ablation studies in popular truss layout design test cases in both 2D and 3D settings.  AutoTruss outperforms the best-reported layouts by 25.1% in the most challenging 3D test cases, resulting in the first effective deep-RL-based approach in the truss layout design literature.

----

## [407] A Logic-based Approach to Contrastive Explainability for Neurosymbolic Visual Question Answering

**Authors**: *Thomas Eiter, Tobias Geibinger, Nelson Higuera, Johannes Oetsch*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/408](https://doi.org/10.24963/ijcai.2023/408)

**Abstract**:

Visual Question Answering (VQA) is a well-known problem for which deep-learning is key. This poses a challenge for explaining answers to questions, the more if advanced notions like contrastive explanations (CEs) should be provided. The latter explain why an answer has been reached in contrast to a different one and are attractive as they focus on reasons necessary to flip a query answer. We present a CE framework for VQA that uses a neurosymbolic VQA architecture which disentangles perception from reasoning. Once the reasoning part is provided as logical theory, we use answer-set programming, in which CE generation can be framed as an abduction problem. We validate our approach on the CLEVR dataset, which we extend by more sophisticated questions to further demonstrate the robustness of the modular architecture. While we achieve top performance compared to related approaches, we can also produce CEs for explanation, model debugging, and validation tasks, showing the versatility of the declarative approach to reasoning.

----

## [408] Cardinality-Minimal Explanations for Monotonic Neural Networks

**Authors**: *Ouns El Harzli, Bernardo Cuenca Grau, Ian Horrocks*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/409](https://doi.org/10.24963/ijcai.2023/409)

**Abstract**:

In recent years, there has been increasing interest in explanation methods for neural model predictions that offer precise formal guarantees. These include abductive (respectively, contrastive) methods, which aim to compute  minimal subsets of input features that are sufficient for a given prediction to hold (respectively, to change a given prediction). The corresponding decision problems are, however, known to be intractable. In this paper, we investigate whether  tractability can be regained by focusing on neural models implementing a monotonic function. Although the relevant decision problems remain intractable, we can show that they become solvable in polynomial time by means of greedy algorithms if we additionally assume that the activation functions are continuous everywhere and differentiable almost everywhere. Our experiments suggest favourable performance of our algorithms.

----

## [409] Neural Capacitated Clustering

**Authors**: *Jonas K. Falkner, Lars Schmidt-Thieme*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/410](https://doi.org/10.24963/ijcai.2023/410)

**Abstract**:

Recent work on deep clustering has found new promising methods also for constrained clustering problems. 
Their typically pairwise constraints often can be used to guide the partitioning of the data.
Many problems however, feature cluster-level constraints, e.g. the Capacitated Clustering Problem (CCP), where each point has a weight and the total weight sum of all points in each cluster is bounded by a prescribed capacity. 
In this paper we propose a new method for the CCP, Neural Capacited Clustering, that learns a neural network to predict the assignment probabilities of points to cluster centers from a data set of optimal or near optimal past solutions of other problem instances. 
During inference, the resulting scores are then used in an iterative k-means like procedure to refine the assignment under capacity constraints. 
In our experiments on artificial data and two real world datasets our approach outperforms several state-of-the-art mathematical and heuristic solvers from the literature. 
Moreover, we apply our method in the context of a cluster-first-route-second approach to the Capacitated Vehicle Routing Problem (CVRP) and show competitive results on the well-known Uchoa benchmark.

----

## [410] A Fast Adaptive Randomized PCA Algorithm

**Authors**: *Xu Feng, Wenjian Yu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/411](https://doi.org/10.24963/ijcai.2023/411)

**Abstract**:

It is desirable to adaptively determine the number of dimensions (rank) for PCA according to a given tolerance of low-rank approximation error. In this work, we aim to develop a fast algorithm solving this adaptive PCA problem. We propose to replace the QR factorization in randQB_EI algorithm with matrix multiplication and inversion of small matrices, and propose a new error indicator to incrementally evaluate approximation error in Frobenius norm. Combining the shifted power iteration technique for better accuracy, we finally build up an algorithm named farPCA. Experimental results show that farPCA is much faster than the baseline methods (randQB_EI, randUBV and svds) in practical setting of multi-thread computing, while producing nearly optimal results of adpative PCA.

----

## [411] FedHGN: A Federated Framework for Heterogeneous Graph Neural Networks

**Authors**: *Xinyu Fu, Irwin King*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/412](https://doi.org/10.24963/ijcai.2023/412)

**Abstract**:

Heterogeneous graph neural networks (HGNNs) can learn from typed and relational graph data more effectively than conventional GNNs. With larger parameter spaces, HGNNs may require more training data, which is often scarce in real-world applications due to privacy regulations (e.g., GDPR). Federated graph learning (FGL) enables multiple clients to train a GNN collaboratively without sharing their local data. However, existing FGL methods mainly focus on homogeneous GNNs or knowledge graph embeddings; few have considered heterogeneous graphs and HGNNs. In federated heterogeneous graph learning, clients may have private graph schemas. Conventional FL/FGL methods attempting to define a global HGNN model would violate schema privacy. To address these challenges, we propose FedHGN, a novel and general FGL framework for HGNNs. FedHGN adopts schema-weight decoupling to enable schema-agnostic knowledge sharing and employs coefficients alignment to stabilize the training process and improve HGNN performance. With better privacy preservation, FedHGN consistently outperforms local training and conventional FL methods on three widely adopted heterogeneous graph datasets with varying client numbers. The code is available at https://github.com/cynricfu/FedHGN.

----

## [412] Autonomous Exploration for Navigating in MDPs Using Blackbox RL Algorithms

**Authors**: *Pratik Gajane, Peter Auer, Ronald Ortner*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/413](https://doi.org/10.24963/ijcai.2023/413)

**Abstract**:

We consider the problem of navigating in a Markov decision process where extrinsic rewards are either absent or ignored. In this setting, the objective is to learn policies to reach all the states that are reachable within a given number of steps (in expectation) from a starting state. We introduce a novel meta-algorithm which can use any online reinforcement learning algorithm (with appropriate regret guarantees) as a black-box. Our algorithm demonstrates a method for transforming the output of online algorithms to a batch setting. We prove an upper bound on the sample complexity of our algorithm in terms of the regret bound of the used black-box RL algorithm. Furthermore, we provide experimental results to validate the effectiveness of our algorithm and correctness of our theoretical results.

----

## [413] Self-Recover: Forecasting Block Maxima in Time Series from Predictors with Disparate Temporal Coverage Using Self-Supervised Learning

**Authors**: *Asadullah Hill Galib, Andrew McDonald, Pang-Ning Tan, Lifeng Luo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/414](https://doi.org/10.24963/ijcai.2023/414)

**Abstract**:

Forecasting the block maxima of a future time window is a challenging task due to the difficulty in inferring the tail distribution of a target variable. As the historical observations alone may not be sufficient to train robust models to predict the block maxima, domain-driven process models are often available in many scientific domains to supplement the observation data and improve the forecast accuracy. Unfortunately, coupling the historical observations with process model outputs is a challenge due to their disparate temporal coverage. This paper presents Self-Recover, a deep learning framework to predict the block maxima of a time window by employing self-supervised learning to address the varying temporal data coverage problem. Specifically Self-Recover uses a combination of contrastive and generative self-supervised learning schemes along with a denoising autoencoder to impute the missing values. The framework also combines representations of the historical observations with process model outputs via a residual learning approach and learns the generalized extreme value (GEV) distribution characterizing the block maxima values. This enables the framework to reliably estimate the block maxima of each time window along with its confidence interval. Extensive experiments on real-world datasets demonstrate the superiority of Self-Recover compared to other state-of-the-art forecasting methods.

----

## [414] Unbiased Risk Estimator to Multi-Labeled Complementary Label Learning

**Authors**: *Yi Gao, Miao Xu, Min-Ling Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/415](https://doi.org/10.24963/ijcai.2023/415)

**Abstract**:

Multi-label learning (MLL) usually requires assigning multiple relevant labels to each instance. While a fully supervised MLL dataset needs a large amount of labeling effort, using complementary labels can help alleviate this burden. However, current approaches to learning from complementary labels are mainly designed for multi-class learning and assume that each instance has a single relevant label. This means that these approaches cannot be easily applied to MLL when only complementary labels are provided, where the number of relevant labels is unknown and can vary across instances. In this paper, we first propose the unbiased risk estimator for the multi-labeled complementary label learning (MLCLL) problem. We also provide an estimation error bound to ensure the convergence of the empirical risk estimator. In some cases, the unbiased estimator may give unbounded gradients for certain loss functions and result in overfitting. To mitigate this problem, we improve the risk estimator by minimizing a proper loss function, which has been shown to improve gradient updates. Our experimental results demonstrate the effectiveness of the proposed approach on various datasets.

----

## [415] Modeling with Homophily Driven Heterogeneous Data in Gossip Learning

**Authors**: *Abhirup Ghosh, Cecilia Mascolo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/416](https://doi.org/10.24963/ijcai.2023/416)

**Abstract**:

Training deep learning models on data distributed and local to edge devices such as mobile phones is a prominent recent research direction. In a Gossip Learning (GL) system, each participating device maintains a model trained on its local data and iteratively aggregates it with the models from its neighbours in a communication network. While the fully distributed operation in GL comes with natural advantages over the centralized orchestration in Federated Learning (FL), its convergence becomes particularly slow when the data distribution is heterogeneous and aligns with the clustered structure of the communication network. These characteristics are pervasive across practical applications as people with similar interests (thus producing similar data) tend to create communities.

This paper proposes a data-driven neighbor weighting strategy for aggregating the models: this enables faster diffusion of knowledge across the communities in the network and leads to quicker convergence. We augment the method to make it computationally efficient and fair: the devices quickly converge to the same model. We evaluate our model on real and synthetic datasets that we generate using a novel generative model for communication networks with heterogeneous data. Our exhaustive empirical evaluation  verifies that our proposed method attains a faster convergence rate than the baselines. For example, the median test accuracy for a decentralized bird image classifier application reaches 81% with our proposed method within 80 rounds, whereas the baseline only reaches 46%.

----

## [416] Adaptive Estimation Q-learning with Uncertainty and Familiarity

**Authors**: *Xiaoyu Gong, Shuai Lü, Jiayu Yu, Sheng Zhu, Zongze Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/417](https://doi.org/10.24963/ijcai.2023/417)

**Abstract**:

One of the key problems in model-free deep reinforcement learning is how to obtain more accurate value estimations. Current most widely-used off-policy algorithms suffer from over- or underestimation bias which may lead to unstable policy. In this paper, we propose a novel method, Adaptive Estimation Q-learning (AEQ), which uses uncertainty and familiarity to control the value estimation naturally and can adaptively change for specific state-action pair. We theoretically prove the property of our familiarity term which can even keep the expected estimation bias approximate to 0, and experimentally demonstrate our dynamic estimation can improve the performance and prevent the bias continuously increasing. We evaluate AEQ on several continuous control tasks, outperforming state-of-the-art performance. Moreover, AEQ is simple to implement and can be applied in any off-policy actor-critic algorithm.

----

## [417] FedPass: Privacy-Preserving Vertical Federated Deep Learning with Adaptive Obfuscation

**Authors**: *Hanlin Gu, Jiahuan Luo, Yan Kang, Lixin Fan, Qiang Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/418](https://doi.org/10.24963/ijcai.2023/418)

**Abstract**:

Vertical federated learning (VFL) allows an active party with labeled data to leverage auxiliary features from the passive parties to improve model performance. Concerns about the private feature and label leakage in both the training and inference phases of VFL have drawn wide research attention. In this paper, we propose a general privacy-preserving vertical federated deep learning framework called FedPass, which leverages adaptive obfuscation to protect the feature and label simultaneously.  Strong privacy-preserving capabilities about private features and labels are theoretically proved (in Theorems 1 and 2). 
Extensive experimental results with different datasets and network architectures also justify the superiority of FedPass against existing methods in light of its near-optimal trade-off between privacy and model performance.

----

## [418] Globally Consistent Federated Graph Autoencoder for Non-IID Graphs

**Authors**: *Kun Guo, Yutong Fang, Qingqing Huang, Yuting Liang, Ziyao Zhang, Wenyu He, Liu Yang, Kai Chen, Ximeng Liu, Wenzhong Guo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/419](https://doi.org/10.24963/ijcai.2023/419)

**Abstract**:

Graph neural networks (GNNs) have been applied successfully in many machine learning tasks due to their advantages in utilizing neighboring information. Recently, with the global enactment of privacy protection regulations, federated GNNs have gained increasing attention in academia and industry. However, the graphs owned by different participants could be non-independently-and-identically distributed (non-IID), leading to the deterioration of federated GNNs' accuracy. In this paper, we propose a globally consistent federated graph autoencoder (GCFGAE) to overcome the non-IID problem in unsupervised federated graph learning via three innovations. First, by integrating federated learning with split learning, we train a unique global model instead of FedAvg-styled global and local models, yielding results consistent with that of the centralized GAE. Second, we design a collaborative computation mechanism considering overlapping vertices to reduce communication overhead during forward propagation. Third, we develop a layer-wise and block-wise gradient computation strategy to reduce the space and communication complexity during backward propagation. Experiments on real-world datasets demonstrate that GCFGAE achieves not only higher accuracy but also around 500 times lower communication overhead and 1000 times smaller space overhead than existing federated GNN models.

----

## [419] Generalization Guarantees of Self-Training of Halfspaces under Label Noise Corruption

**Authors**: *Lies Hadjadj, Massih-Reza Amini, Sana Louhichi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/420](https://doi.org/10.24963/ijcai.2023/420)

**Abstract**:

We investigate the generalization properties of a self-training algorithm with halfspaces. The approach learns a list of halfspaces iteratively from labeled and unlabeled training data, in which each iteration consists of two steps: exploration and pruning. In the exploration phase, the halfspace is found sequentially by maximizing the unsigned-margin among  unlabeled examples and then assigning pseudo-labels to those that have a distance higher than the current threshold. These pseudo-labels are allegedly corrupted by noise. The training set is then augmented with noisy pseudo-labeled examples, and a new classifier is trained.  This process is repeated until no more unlabeled examples remain for pseudo-labeling. In the pruning phase, pseudo-labeled samples that have a distance to the last halfspace greater than the associated  unsigned-margin are then discarded. We prove that the misclassification error of the resulting sequence of classifiers is bounded and show that the resulting semi-supervised approach never degrades performance compared to the  classifier learned using only the initial labeled training set. Experiments carried out on a variety of benchmarks demonstrate the efficiency of the proposed approach compared to state-of-the-art methods.

----

## [420] Learning Preference Models with Sparse Interactions of Criteria

**Authors**: *Margot Herin, Patrice Perny, Nataliya Sokolovska*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/421](https://doi.org/10.24963/ijcai.2023/421)

**Abstract**:

Multicriteria decision making requires defining the result of conflicting and possibly interacting criteria. Allowing criteria interactions in a decision model increases the complexity of the preference learning task due to the combinatorial nature of the possible interactions. In this paper, we propose an approach to learn a decision model in which the interaction pattern is revealed from preference data and kept as simple as possible.  We consider weighted aggregation functions like multilinear utilities or Choquet integrals, admitting representations including non-linear terms measuring the joint benefit or penalty attached to some combinations of criteria. The weighting coefficients known as Möbius masses model positive or negative synergies among criteria. We propose an approach to learn the Möbius masses, based on iterative reweighted least square for sparse recovery, and dualization to improve scalability. This approach is applied to learn sparse representations of the multilinear utility model and conjunctive/disjunctive forms of the discrete Choquet integral from preferences examples, in aggregation problems possibly involving more than 20 criteria.

----

## [421] BRExIt: On Opponent Modelling in Expert Iteration

**Authors**: *Daniel Hernandez, Hendrik Baier, Michael Kaisers*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/422](https://doi.org/10.24963/ijcai.2023/422)

**Abstract**:

Finding a best response policy is a central objective in game theory and multi-agent learning, with modern population-based training approaches employing reinforcement learning algorithms as best-response oracles to improve play against candidate opponents (typically previously learnt policies). We propose Best Response Expert Iteration (BRExIt), which accelerates learning in games by incorporating opponent models into the state-of-the-art learning algorithm Expert Iteration (ExIt). BRExIt aims to (1) improve feature shaping in the apprentice, with a policy head predicting opponent policies as an auxiliary task, and (2) bias opponent moves in planning towards the given or learnt opponent model, to generate apprentice targets that better approximate a best response. In an empirical ablation on BRExIt's algorithmic variants against a set of fixed test agents, we provide statistical evidence that BRExIt learns better performing policies than ExIt. Code available at: https://github.com/Danielhp95/on-opponent-modelling-in-expert-iteration-code. Supplementary material available
at https://arxiv.org/abs/2206.00113.

----

## [422] Dynamic Flows on Curved Space Generated by Labeled Data

**Authors**: *Xinru Hua, Truyen Nguyen, Tam Le, Jose H. Blanchet, Viet Anh Nguyen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/423](https://doi.org/10.24963/ijcai.2023/423)

**Abstract**:

The scarcity of labeled data is a long-standing challenge for many machine learning tasks. We propose our gradient flow method to leverage the existing dataset (i.e., source) to generate new samples that are close to the dataset of interest (i.e., target). We lift both datasets to the space of probability distributions on the feature-Gaussian manifold, and then develop a gradient flow method that minimizes the maximum mean discrepancy loss. To perform the gradient flow of distributions on the curved feature-Gaussian space, we unravel the Riemannian structure of the space and compute explicitly the  Riemannian gradient of the loss function induced by the optimal transport metric. For practical applications, we also propose a discretized flow, and provide conditional results guaranteeing the global convergence of the flow to the optimum. We illustrate the results of our proposed gradient flow method on several real-world datasets and show our method can improve the accuracy of classification models in transfer learning settings.

----

## [423] DiffAR: Adaptive Conditional Diffusion Model for Temporal-augmented Human Activity Recognition

**Authors**: *Shuokang Huang, Po-Yu Chen, Julie Ann McCann*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/424](https://doi.org/10.24963/ijcai.2023/424)

**Abstract**:

Human activity recognition (HAR) is a fundamental sensing and analysis technique that supports diverse applications, such as smart homes and healthcare. In device-free and non-intrusive HAR, WiFi channel state information (CSI) captures wireless signal variations caused by human interference without the need for video cameras or on-body sensors. However, current CSI-based HAR performance is hampered by incomplete CSI recordings due to fixed window sizes in CSI collection and human/machine errors that incur missing values in CSI. To address these issues, we propose DiffAR, a temporal-augmented HAR approach that improves HAR performance by augmenting CSI. DiffAR devises a novel Adaptive Conditional Diffusion Model (ACDM) to synthesize augmented CSI, which tackles the issue of fixed windows by forecasting and handles missing values with imputation. Compared to existing diffusion models, ACDM improves the synthesis quality by guiding progressive synthesis with step-specific conditions. DiffAR further exploits an ensemble classifier for activity recognition using both raw and augmented CSI. Extensive experiments on four public datasets show that DiffAR achieves the best synthesis quality of augmented CSI and outperforms state-of-the-art CSI-based HAR methods in recognition performance. The source code of DiffAR is available at https://github.com/huangshk/DiffAR.

----

## [424] Progressive Label Propagation for Semi-Supervised Multi-Dimensional Classification

**Authors**: *Teng Huang, Bin-Bin Jia, Min-Ling Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/425](https://doi.org/10.24963/ijcai.2023/425)

**Abstract**:

In multi-dimensional classification (MDC), each training example is associated with multiple class variables from different class spaces. However, it is rather costly to collect labeled MDC examples which have to be annotated from several dimensions (class spaces). To reduce the labeling cost, we attempt to deal with the MDC problem under the semi-supervised learning setting. Accordingly, a novel MDC approach named PLAP is proposed to solve the resulting semi-supervised MDC problem. Overall, PLAP works under the label propagation framework to utilize unlabeled data. To further consider dependencies among class spaces, PLAP deals with each class space in a progressive manner, where the previous propagation results will be used to initialize the current propagation procedure and all processed class spaces and the current one will be regarded as an entirety. Experiments validate the effectiveness of the proposed approach.

----

## [425] Federated Graph Semantic and Structural Learning

**Authors**: *Wenke Huang, Guancheng Wan, Mang Ye, Bo Du*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/426](https://doi.org/10.24963/ijcai.2023/426)

**Abstract**:

Federated graph learning collaboratively learns a global graph neural network with distributed graphs, where the non-independent and identically distributed property is one of the major challenge. Most relative arts focus on traditional distributed tasks like images and voices, incapable of the graph structures. This paper firstly reveals that local client distortion is brought by both node-level semantics and graph-level structure. First, for node-level semantic, we find that contrasting nodes from distinct classes is beneficial to provide a well-performing discrimination. We pull the local node towards the global node of the same class and push them away from the global node of different classes. Second, we postulate that a well-structural graph neural network possesses similarity for neighbors due to the inherent adjacency relationships. However, aligning each node with adjacent nodes hinders discrimination due to the potential class inconsistency. We transform the adjacency relationships into the similarity distribution and leverage the global model to distill the relation knowledge into the local model, which preserves the structural information and discriminability of the local model. Empirical results on three graph datasets manifest the superiority of the proposed method over counterparts.

----

## [426] Enabling Abductive Learning to Exploit Knowledge Graph

**Authors**: *Yu-Xuan Huang, Zequn Sun, Guangyao Li, Xiaobin Tian, Wang-Zhou Dai, Wei Hu, Yuan Jiang, Zhi-Hua Zhou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/427](https://doi.org/10.24963/ijcai.2023/427)

**Abstract**:

Most systems integrating data-driven machine learning with knowledge-driven reasoning usually rely on a specifically designed knowledge base to enable efficient symbolic inference. However, it could be cumbersome for the nonexpert end-users to prepare such a knowledge base in real tasks. Recent years have witnessed the success of large-scale knowledge graphs, which could be ideal domain knowledge resources for real-world machine learning tasks. However, these large-scale knowledge graphs usually contain much information that is irrelevant to a specific learning task. Moreover, they often contain a certain degree of noise. Existing methods can hardly make use of them because the large-scale probabilistic logical inference is usually intractable. To address these problems, we present ABductive Learning with Knowledge Graph (ABL-KG) that can automatically mine logic rules from knowledge graphs during learning, using a knowledge forgetting mechanism for filtering out irrelevant information. Meanwhile, these rules can form a logic program that enables efficient joint optimization of the machine learning model and logic inference within the Abductive Learning (ABL) framework. Experiments on four different tasks show that ABL-KG can automatically extract useful rules from large-scale and noisy knowledge graphs, and significantly improve the performance of machine learning with only a handful of labeled data.

----

## [427] Latent Processes Identification From Multi-View Time Series

**Authors**: *Zenan Huang, Haobo Wang, Junbo Zhao, Nenggan Zheng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/428](https://doi.org/10.24963/ijcai.2023/428)

**Abstract**:

Understanding the dynamics of time series data typically requires identifying the unique latent factors for data generation, a.k.a., latent processes identification. Driven by the independent assumption, existing works have made great progress in handling single-view data. However, it is a non-trivial problem that extends them to multi-view time series data because of two main challenges: (i) the complex data structure, such as temporal dependency, can result in violation of the independent assumption; (ii) the factors from different views are generally overlapped and are hard to be aggregated to a complete set. In this work, we propose a novel framework MuLTI that employs the contrastive learning technique to invert the data generative process for enhanced identifiability. Additionally, MuLTI integrates a permutation mechanism that merges corresponding overlapped variables by the establishment of an optimal transport formula. Extensive experimental results on synthetic and real-world datasets demonstrate the superiority of our method in recovering identifiable latent variables on multi-view time series. The code is available on https://github.com/lccurious/MuLTI.

----

## [428] Multi-Modality Deep Network for JPEG Artifacts Reduction

**Authors**: *Xuhao Jiang, Weimin Tan, Qing Lin, Chenxi Ma, Bo Yan, Liquan Shen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/429](https://doi.org/10.24963/ijcai.2023/429)

**Abstract**:

In recent years, many convolutional neural network-based models are designed for JPEG artifacts reduction, and have achieved notable progress. However, few methods are suitable for extreme low-bitrate image compression artifacts reduction. The main challenge is that the highly compressed image loses too much information, resulting in reconstructing high-quality image difficultly. To address this issue, we propose a multimodal fusion learning method for text-guided JPEG artifacts reduction, in which the corresponding text description not only provides the potential prior information of the highly compressed image, but also serves as supplementary information to assist in image deblocking. We fuse image features and text semantic features from the global and local perspectives respectively, and design a contrastive loss built upon contrastive learning to produce visually pleasing results. Extensive experiments, including a user study, prove that our method can obtain better deblocking results compared to the state-of-the-art methods.

----

## [429] Musical Voice Separation as Link Prediction: Modeling a Musical Perception Task as a Multi-Trajectory Tracking Problem

**Authors**: *Emmanouil Karystinaios, Francesco Foscarin, Gerhard Widmer*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/430](https://doi.org/10.24963/ijcai.2023/430)

**Abstract**:

This paper targets the perceptual task of separating the different interacting voices, i.e., monophonic melodic streams, in a polyphonic musical piece. We target symbolic music, where notes are explicitly encoded, and model this task as a Multi-Trajectory Tracking (MTT) problem from discrete observations, i.e., notes in a pitch-time space. Our approach builds a graph from a musical piece, by creating one node for every note, and separates the melodic trajectories by predicting a link between two notes if they are consecutive in the same voice/stream. This kind of local, greedy prediction is made possible by node embeddings created by a heterogeneous graph neural network that can capture inter- and intra-trajectory information. Furthermore, we propose a new regularization loss that encourages the output to respect the MTT premise of at most one incoming and one outgoing link for every node, favoring monophonic (voice) trajectories; this loss function might also be useful in other general MTT scenarios. Our approach does not use domain-specific heuristics, is scalable to longer sequences and a higher number of voices, and can handle complex cases such as voice inversions and overlaps. We reach new state-of-the-art results for the voice separation task on classical music of different styles. All code, data, and pretrained models are available on https://github.com/manoskary/vocsep_ijcai2023

----

## [430] A Unification Framework for Euclidean and Hyperbolic Graph Neural Networks

**Authors**: *Mehrdad Khatir, Nurendra Choudhary, Sutanay Choudhury, Khushbu Agarwal, Chandan K. Reddy*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/431](https://doi.org/10.24963/ijcai.2023/431)

**Abstract**:

Hyperbolic neural networks can effectively capture the inherent hierarchy of graph datasets, and consequently a powerful choice of GNNs. However, they entangle multiple incongruent (gyro-)vector spaces within a layer, which makes them limited in terms of generalization and scalability. 

In this work, we propose the Poincar√© disk model as our search space, and apply all approximations on the disk (as if the disk is a tangent space derived from the origin), thus getting rid of all inter-space transformations. Such an approach enables us to propose a hyperbolic normalization layer and to further simplify the entire hyperbolic model to a Euclidean model cascaded with our hyperbolic normalization layer. We applied our proposed nonlinear hyperbolic normalization to the current state-of-the-art homogeneous and multi-relational graph networks. We demonstrate that our model not only leverages the power of Euclidean networks such as interpretability and efficient execution of various model components, but also outperforms both Euclidean and hyperbolic counterparts on various benchmarks. Our code is made publicly available at https://github.com/oom-debugger/ijcai23.

----

## [431] SeRO: Self-Supervised Reinforcement Learning for Recovery from Out-of-Distribution Situations

**Authors**: *Chan Kim, JaeKyung Cho, Christophe Bobda, Seung-Woo Seo, Seong-Woo Kim*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/432](https://doi.org/10.24963/ijcai.2023/432)

**Abstract**:

Robotic agents trained using reinforcement learning have the problem of taking unreliable actions in an out-of-distribution (OOD) state. Agents can easily become OOD in real-world environments because it is almost impossible for them to visit and learn the entire state space during training. Unfortunately, unreliable actions do not ensure that agents perform their original tasks successfully. Therefore, agents should be able to recognize whether they are in OOD states and learn how to return to the learned state distribution rather than continue to take unreliable actions. In this study, we propose a novel method for retraining agents to recover from OOD situations in a self-supervised manner when they fall into OOD states. Our in-depth experimental results demonstrate that our method substantially improves the agentâ€™s ability to recover from OOD situations in terms of sample efficiency and restoration of the performance for the original tasks. Moreover, we show that our method can retrain the agent to recover from OOD situations even when in-distribution states are difficult to visit through exploration. Code and supplementary materials are available at https://github.com/SNUChanKim/SeRO.

----

## [432] MultiPar-T: Multiparty-Transformer for Capturing Contingent Behaviors in Group Conversations

**Authors**: *Dong Won Lee, Yubin Kim, Rosalind W. Picard, Cynthia Breazeal, Hae Won Park*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/433](https://doi.org/10.24963/ijcai.2023/433)

**Abstract**:

As we move closer to real-world social AI systems, AI agents must be able to deal with multiparty (group) conversations. Recognizing and interpreting multiparty behaviors is challenging, as the system must recognize individual behavioral cues, deal with the complexity of multiple streams of data from multiple people, and recognize the subtle contingent social exchanges that take place amongst group members. To tackle this challenge, we propose the Multiparty-Transformer (Multipar- T), a transformer model for multiparty behavior modeling. The core component of our proposed approach is Crossperson Attention, which is specifically designed to detect contingent behavior between pairs of people. We verify the effectiveness of Multipar-T on a publicly available video-based group engagement detection benchmark, where it outperforms state-of-the-art approaches in average F-1 scores by 5.2% and individual class F-1 scores by up to 10.0%. Through qualitative analysis, we show that our Crossperson Attention module is able to discover contingent behaviors.

----

## [433] Stochastic Feature Averaging for Learning with Long-Tailed Noisy Labels

**Authors**: *Hao-Tian Li, Tong Wei, Hao Yang, Kun Hu, Chong Peng, Li-Bo Sun, Xun-Liang Cai, Min-Ling Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/434](https://doi.org/10.24963/ijcai.2023/434)

**Abstract**:

Deep neural networks have shown promising results on a wide variety of tasks using large-scale and well-annotated training datasets. However, data collected from real-world applications can suffer from two prevalent biases, i.e., long-tailed class distribution and label noise. Previous efforts on long-tailed learning and label-noise learning can only address a single type of data bias, leading to a severe deterioration of their performance. In this paper, we propose a distance-based sample selection algorithm called Stochastic Feature Averaging (SFA), which fits a Gaussian using the exponential running average of class centroids to capture uncertainty in representation space due to label noise and data scarcity. With SFA, we detect noisy samples based on their distances to class centroids sampled from this Gaussian distribution. Based on the identified clean samples, we then propose to train an auxiliary balanced classifier to improve the generalization for the minority class and facilitate the update of Gaussian parameters. Extensive experimental results show that SFA can enhance the performance of existing methods on both simulated and real-world datasets. Further, we propose to combine SFA with the sample-selection approach, distribution-robust, and noise-robust loss functions, resulting in significant improvement in performance over the baselines. Our code is available at https://github.com/HotanLee/SFA

----

## [434] Incomplete Multi-view Clustering via Prototype-based Imputation

**Authors**: *Haobin Li, Yunfan Li, Mouxing Yang, Peng Hu, Dezhong Peng, Xi Peng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/435](https://doi.org/10.24963/ijcai.2023/435)

**Abstract**:

In this paper, we study how to achieve two characteristics highly-expected by incomplete multi-view clustering (IMvC). Namely, i) instance commonality refers to that within-cluster instances should share a common pattern, and ii) view versatility refers to that cross-view samples should own view-specific patterns. To this end, we design a novel dual-stream model which employs a dual attention layer and a dual contrastive learning loss to learn view-specific prototypes and model the sample-prototype relationship. When the view is missed, our model performs data recovery using the prototypes in the missing view and the sample-prototype relationship inherited from the observed view. Thanks to our dual-stream model, both cluster- and view-specific information could be captured, and thus the instance commonality and view versatility could be preserved to facilitate IMvC. Extensive experiments demonstrate the superiority of our method on five challenging benchmarks compared with 11 approaches. The code could be accessed from https://pengxi.me.

----

## [435] Towards Sharp Analysis for Distributed Learning with Random Features

**Authors**: *Jian Li, Yong Liu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/436](https://doi.org/10.24963/ijcai.2023/436)

**Abstract**:

In recent studies, the generalization properties for distributed learning and random features assumed the existence of the target concept over the hypothesis space. However, this strict condition is not applicable to the more common non-attainable case. In this paper, using refined proof techniques, we first extend the optimal rates for distributed learning with random features to the non-attainable case. Then, we reduce the number of required random features via data-dependent generating strategy, and improve the allowed number of partitions with additional unlabeled data. Theoretical analysis shows these techniques remarkably reduce computational cost while preserving the optimal generalization accuracy under standard assumptions. Finally, we conduct several experiments on both simulated and real-world datasets, and the empirical results validate our theoretical findings.

----

## [436] IID-GAN: an IID Sampling Perspective for Regularizing Mode Collapse

**Authors**: *Yang Li, Liangliang Shi, Junchi Yan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/437](https://doi.org/10.24963/ijcai.2023/437)

**Abstract**:

Despite its success, generative adversarial networks (GANs) still suffer from mode collapse, i.e., the generator can only map latent variables to a partial set of modes in the target distribution. In this paper, we analyze and seek to regularize this issue with an independent and identically distributed (IID) sampling perspective and emphasize that holding the IID property referring to the target distribution for generation can naturally avoid mode collapse. This is based on the basic IID assumption for real data in machine learning. However, though the source samples {z} obey IID, the generations {G(z)} may not necessarily be IID sampling from the target distribution. Based on this observation, considering a necessary condition of IID generation, we propose a new loss to encourage the closeness between the inverse samples of real data and the Gaussian source in the latent space to regularize the generation to be IID from the target distribution. The logic is that the inverse samples from target data should also be IID in the source distribution. Experiments on both synthetic and real-world data show the effectiveness of our model.

----

## [437] Generative Flow Networks for Precise Reward-Oriented Active Learning on Graphs

**Authors**: *Yinchuan Li, Zhigang Li, Wenqian Li, Yunfeng Shao, Yan Zheng, Jianye Hao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/438](https://doi.org/10.24963/ijcai.2023/438)

**Abstract**:

Many score-based active learning methods have been successfully applied to graph-structured data, aiming to reduce the number of labels and achieve better performance of graph neural networks based on predefined score functions. However, these algorithms struggle to learn policy distributions that are proportional to rewards and have limited exploration capabilities. In this paper, we innovatively formulate the graph active learning problem as a generative process, named GFlowGNN, which generates various samples through sequential actions with probabilities precisely proportional to a predefined reward function. Furthermore, we propose the concept of flow nodes and flow features to efficiently model graphs as flows based on generative flow networks, where the policy network is trained with specially designed rewards. Extensive experiments on real datasets show that the proposed approach has good exploration capability and transferability, outperforming various state-of-the-art methods.

----

## [438] Teacher Assistant-Based Knowledge Distillation Extracting Multi-level Features on Single Channel Sleep EEG

**Authors**: *Heng Liang, Yucheng Liu, Haichao Wang, Ziyu Jia*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/439](https://doi.org/10.24963/ijcai.2023/439)

**Abstract**:

Sleep stage classification is of great significance to the diagnosis of sleep disorders. However, existing sleep stage classification models based on deep learning are usually relatively large in size (wider and deeper), which makes them hard to be deployed on wearable devices. Therefore, it is a challenge to lighten the existing sleep stage classification models. In this paper, we propose a novel general knowledge distillation framework for sleep stage classification tasks called SleepKD. Our SleepKD, composed of the multi-level module, teacher assistant module, and other knowledge distillation modules, aims to lighten large-scale sleep stage classification models. Specifically, the multi-level module is able to transfer the multi-level knowledge extracted from sleep signals by the teacher model (large-scale model) to the student model (lightweight model). Moreover, the teacher assistant module bridges the large gap between the teacher and student network, and further improves the distillation. We evaluate our method on two public sleep datasets (Sleep-EDF and ISRUC-III). Compared to the baseline methods, the results show that our knowledge distillation framework achieves state-of-the-art performance. SleepKD can significantly lighten the sleep model while maintaining its classification performance. The source code is available at https://github.com/HychaoWang/SleepKD.

----

## [439] HyperFed: Hyperbolic Prototypes Exploration with Consistent Aggregation for Non-IID Data in Federated Learning

**Authors**: *Xinting Liao, Weiming Liu, Chaochao Chen, Pengyang Zhou, Huabin Zhu, Yanchao Tan, Jun Wang, Yue Qi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/440](https://doi.org/10.24963/ijcai.2023/440)

**Abstract**:

Federated learning (FL) collaboratively models user data in a decentralized way. However, in the real world, non-identical and independent data distributions (non-IID) among clients hinder the performance of FL due to three issues, i.e., (1) the class statistics shifting, (2) the insufficient hierarchical information utilization, and (3) the inconsistency in aggregating clients. To address the above issues, we propose HyperFed which contains three main modules, i.e., hyperbolic prototype Tammes initialization (HPTI), hyperbolic prototype learning (HPL), and consistent aggregation (CA). Firstly, HPTI in the server constructs uniformly distributed and fixed class prototypes, and shares them with clients to match class statistics, further guiding consistent feature representation for local clients. Secondly, HPL in each client captures the hierarchical information in local data with the supervision of shared class prototypes in the hyperbolic model space. Additionally, CA in the server mitigates the impact of the inconsistent deviations from clients to server. Extensive studies of four datasets prove that HyperFed is effective in enhancing the performance of FL under the non-IID setting.

----

## [440] Contrastive Learning and Reward Smoothing for Deep Portfolio Management

**Authors**: *Yun-Hsuan Lien, Yuan-kui Li, Yu-Shuen Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/441](https://doi.org/10.24963/ijcai.2023/441)

**Abstract**:

In this study, we used reinforcement learning (RL) models to invest assets in order to earn returns. The models were trained to interact with a simulated environment based on historical market data and learn trading strategies. However, using deep neural networks based on the returns of each period can be challenging due to the unpredictability of financial markets. As a result, the policies learned from training data may not be effective when tested in real-world situations. To address this issue, we incorporated contrastive learning and reward smoothing into our training process. Contrastive learning allows the RL models to recognize patterns in asset states that may indicate future price movements. Reward smoothing, on the other hand, serves as a regularization technique to prevent the models from seeking immediate but uncertain profits. We tested our method against various traditional financial techniques and other deep RL methods, and found it to be effective in both the U.S. stock market and the cryptocurrency market. Our source code is available at https://github.com/sophialien/FinTech-DPM.

----

## [441] Learning Survival Distribution with Implicit Survival Function

**Authors**: *Yu Ling, Weimin Tan, Bo Yan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/442](https://doi.org/10.24963/ijcai.2023/442)

**Abstract**:

Survival analysis aims at modeling the relationship between covariates and event occurrence with some untracked (censored) samples. In implementation, existing methods model the survival distribution with strong assumptions or in a discrete time space for likelihood estimation with censorship, which leads to weak generalization. In this paper, we propose Implicit Survival Function (ISF) based on Implicit Neural Representation for survival distribution estimation without strong assumptions, and employ numerical integration to approximate the cumulative distribution function for prediction and optimization. Experimental results show that ISF outperforms the state-of-the-art methods in three public datasets and has robustness to the hyperparameter controlling estimation precision.

----

## [442] FedET: A Communication-Efficient Federated Class-Incremental Learning Framework Based on Enhanced Transformer

**Authors**: *Chenghao Liu, Xiaoyang Qu, Jianzong Wang, Jing Xiao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/443](https://doi.org/10.24963/ijcai.2023/443)

**Abstract**:

Federated Learning (FL) has been widely concerned for it enables decentralized learning while ensuring data privacy. However, most existing methods unrealistically assume that the classes encountered by local clients are fixed over time. After learning new classes, this impractical assumption will make the model's catastrophic forgetting of old classes significantly severe. Moreover, due to the limitation of communication cost, it is challenging to use large-scale models in FL, which will affect the prediction accuracy. To address these challenges, we propose a novel framework, Federated Enhanced Transformer (FedET), which simultaneously achieves high accuracy and low communication cost. Specifically, FedET uses Enhancer, a tiny module, to absorb and communicate new knowledge, and applies pre-trained Transformers combined with different Enhancers to ensure high precision on various tasks. To address local forgetting caused by new classes of new tasks and global forgetting brought by non-i.i.d class imbalance across different local clients, we proposed an Enhancer distillation method to modify the imbalance between old and new knowledge and repair the non-i.i.d. problem. Experimental results demonstrate that FedET's average accuracy on a representative benchmark dataset is 14.1% higher than the state-of-the-art method, while FedET saves 90% of the communication cost compared to the previous method.

----

## [443] FedDWA: Personalized Federated Learning with Dynamic Weight Adjustment

**Authors**: *Jiahao Liu, Jiang Wu, Jinyu Chen, Miao Hu, Yipeng Zhou, Di Wu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/444](https://doi.org/10.24963/ijcai.2023/444)

**Abstract**:

Different from conventional federated learning, personalized federated learning (PFL) is able to train a customized model for each individual client according to its unique requirement. The mainstream approach is to adopt a kind of weighted aggregation method to generate personalized models, in which weights are determined by the loss value or model parameters among different clients. However, such kinds of methods require clients to download others' models. It not only sheer increases communication traffic but also potentially infringes data privacy. In this paper, we propose a new PFL algorithm called FedDWA (Federated Learning with Dynamic Weight Adjustment) to address the above problem, which leverages the parameter server (PS) to compute personalized aggregation weights based on collected models from clients. In this way, FedDWA can capture similarities between clients with much less communication overhead. More specifically, we formulate the PFL problem as an optimization problem by minimizing the distance between personalized models and  guidance models, so as to  customize aggregation weights for each client. Guidance models are obtained by  the local one-step ahead adaptation on individual clients. Finally,  we conduct extensive experiments using five real datasets and the results demonstrate that FedDWA can significantly reduce the communication traffic and achieve much higher model accuracy than the state-of-the-art approaches.

----

## [444] Open-world Semi-supervised Novel Class Discovery

**Authors**: *Jiaming Liu, Yangqiming Wang, Tongze Zhang, Yulu Fan, Qinli Yang, Junming Shao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/445](https://doi.org/10.24963/ijcai.2023/445)

**Abstract**:

Traditional semi-supervised learning tasks assume that both labeled and unlabeled data follow the same class distribution, but the realistic open-world scenarios are of more complexity with unknown novel classes mixed in the unlabeled set. Therefore, it is of great challenge to not only recognize samples from known classes but also discover the unknown number of novel classes within the unlabeled data. In this paper, we introduce a new open-world semi-supervised novel class discovery approach named OpenNCD, a progressive bi-level contrastive learning method over multiple prototypes. The proposed method is composed of two reciprocally enhanced parts. First, a bi-level contrastive learning method is introduced, which maintains the pair-wise similarity of the prototypes and the prototype group levels for better representation learning. Then, a reliable prototype similarity metric is proposed based on the common representing instances. Prototypes with high similarities will be grouped progressively for known class recognition and novel class discovery. Extensive experiments on three image datasets are conducted and the results show the effectiveness of the proposed method in open-world scenarios, especially with scarce known classes and labels.

----

## [445] Bayesian Optimization with Switching Cost: Regret Analysis and Lookahead Variants

**Authors**: *Peng Liu, Haowei Wang, Wei Qiyu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/446](https://doi.org/10.24963/ijcai.2023/446)

**Abstract**:

Bayesian Optimization (BO) has recently received increasing attention due to its efficiency in optimizing expensive-to-evaluate functions.  For some practical problems, it is essential to consider the path-dependent switching cost between consecutive sampling locations given a total traveling budget. For example, when using a drone to locate cracks in a building wall or search for lost survivors in the wild, the search path needs to be efficiently planned given the limited battery power of the drone. Tackling such problems requires a careful cost-benefit analysis of candidate locations and balancing exploration and exploitation.  In this work, we formulate such a problem as a constrained Markov Decision Process (MDP) and solve it by proposing a new distance-adjusted multi-step look-ahead acquisition function, the distUCB, and using rollout approximation. We also provide a theoretical regret analysis of the distUCB-based Bayesian optimization algorithm. In addition, the empirical performance of the proposed algorithm is tested based on both synthetic and real data experiments, and it shows that our cost-aware non-myopic algorithm performs better than other popular alternatives.

----

## [446] Label Enhancement via Joint Implicit Representation Clustering

**Authors**: *Yunan Lu, Weiwei Li, Xiuyi Jia*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/447](https://doi.org/10.24963/ijcai.2023/447)

**Abstract**:

Label distribution is an effective label form to portray label polysemy (i.e., the cases that an instance can be described by multiple labels simultaneously). However, the expensive annotating cost of label distributions limits its application to a wider range of practical tasks. Therefore, LE (label enhancement) techniques are extensively studied to solve this problem. Existing LE algorithms mostly estimate label distributions by the instance relation or the label relation. However, they suffer from biased instance relations, limited model capabilities, or suboptimal local label correlations. Therefore, in this paper, we propose a deep generative model called JRC to simultaneously learn and cluster the joint implicit representations of both features and labels, which can be used to improve any existing LE algorithm involving the instance relation or local label correlations. Besides, we develop a novel label distribution recovery module, and then integrate it with JRC model, thus constituting a novel generative label enhancement model that utilizes the learned joint implicit representations and instance clusters in a principled way. Finally, extensive experiments validate our proposal.

----

## [447] Recognizable Information Bottleneck

**Authors**: *Yilin Lyu, Xin Liu, Mingyang Song, Xinyue Wang, Yaxin Peng, Tieyong Zeng, Liping Jing*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/448](https://doi.org/10.24963/ijcai.2023/448)

**Abstract**:

Information Bottlenecks (IBs) learn representations that generalize to unseen data by information compression. However, existing IBs are practically unable to guarantee generalization in real-world scenarios due to the vacuous generalization bound. The recent PAC-Bayes IB uses information complexity instead of information compression to establish a connection with the mutual information generalization bound. However, it requires the computation of expensive second-order curvature, which hinders its practical application. In this paper, we establish the connection between the recognizability of representations and the recent functional conditional mutual information (f-CMI) generalization bound, which is significantly easier to estimate. On this basis we propose a Recognizable Information Bottleneck (RIB) which regularizes the recognizability of representations through a recognizability critic optimized by density ratio matching under the Bregman divergence. Extensive experiments on several commonly used datasets demonstrate the effectiveness of the proposed method in regularizing the model and estimating the generalization gap.

----

## [448] Multi-View Robust Graph Representation Learning for Graph Classification

**Authors**: *Guanghui Ma, Chunming Hu, Ling Ge, Hong Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/449](https://doi.org/10.24963/ijcai.2023/449)

**Abstract**:

The robustness of graph classification models plays an essential role in providing highly reliable applications. Previous studies along this line primarily focus on seeking the stability of the model in terms of  overall data metrics (e.g., accuracy)  when facing data perturbations, such as removing edges. Empirically, we find that these graph classification models also suffer from semantic bias and confidence collapse issues, which substantially hinder their applicability in real-world scenarios. To address these issues, we present  MGRL, a multi-view representation learning model for graph classification tasks that achieves robust results. Firstly, we proposes an instance-view consistency representation learning method, which utilizes multi-granularity contrastive learning technique to perform semantic constraints on instance representations at both the node and graph levels, thus alleviating the semantic bias issue. Secondly, we proposes a class-view discriminative representation learning method, which employs the prototype-driven class distance optimization technique to adjust intra- and inter-class distances, thereby mitigating the confidence collapse issue.Finally, extensive experiments and visualizations on eight benchmark dataset demonstrate the effectiveness of MGRL.

----

## [449] CTW: Confident Time-Warping for Time-Series Label-Noise Learning

**Authors**: *Peitian Ma, Zhen Liu, Junhao Zheng, Linghao Wang, Qianli Ma*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/450](https://doi.org/10.24963/ijcai.2023/450)

**Abstract**:

Noisy labels seriously degrade the generalization ability of Deep Neural Networks (DNNs) in various classification tasks. Existing studies on label-noise learning mainly focus on computer vision, while time series also suffer from the same issue. Directly applying the methods from computer vision to time series may reduce the temporal dependency due to different data characteristics. How to make use of the properties of time series to enable DNNs to learn robust representations in the presence of noisy labels has not been fully explored. To this end, this paper proposes a method that expands the distribution of Confident instances by Time-Warping (CTW) to learn robust representations of time series. Specifically, since applying the augmentation method to all data may introduce extra mislabeled data, we select confident instances to implement Time-Warping. In addition, we normalize the distribution of the training loss of each class to eliminate the model's selection preference for instances of different classes, alleviating the class imbalance caused by sample selection. Extensive experimental results show that CTW achieves state-of-the-art performance on the UCR datasets when dealing with different types of noise. Besides, the t-SNE visualization of our method verifies that augmenting confident data improves the generalization ability. Our code is available at https://github.com/qianlima-lab/CTW.

----

## [450] Label Specific Multi-Semantics Metric Learning for Multi-Label Classification: Global Consideration Helps

**Authors**: *Junxiang Mao, Wei Wang, Min-Ling Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/451](https://doi.org/10.24963/ijcai.2023/451)

**Abstract**:

In multi-label classification, it is critical to capitalize on complicated data structures and semantic relationships. Metric learning serves as an effective strategy to provide a better measurement of distances between examples. Existing works on metric learning for multi-label classification mainly learn one single global metric that characterizes latent semantic similarity between multi-label instances. However, such single-semantics metric exploitation approaches can not capture the intrinsic properties of multi-label data possessed of rich semantics. In this paper, the first attempt towards multi-semantics metric learning for multi-label classification is investigated. Specifically, the proposed LIMIC approach simultaneously learns one global and multiple label-specific local metrics by exploiting label-specific side information. The global metric is learned to capture the commonality across all the labels and label-specific local metrics characterize the individuality of each semantic space. The combination of global metric and label-specific local metrics is utilized to construct latent semantic space for each label, in which similar intra-class instances are pushed closer and inter-class instances are pulled apart. Furthermore, metric-based label correlation regularization is constructed to maintain similarity between correlated label spaces. Extensive experiments on benchmark multi-label data sets validate the superiority of our proposed approach in learning effective distance metrics for multi-label classification.

----

## [451] Doubly Stochastic Graph-based Non-autoregressive Reaction Prediction

**Authors**: *Ziqiao Meng, Peilin Zhao, Yang Yu, Irwin King*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/452](https://doi.org/10.24963/ijcai.2023/452)

**Abstract**:

Organic reaction prediction is a critical task in drug discovery. Recently, researchers have achieved non-autoregressive reaction prediction by modeling the redistribution of electrons, resulting in state-of-the-art top-1 accuracy, and enabling parallel sampling. However, the current non-autoregressive decoder does not satisfy two essential rules of electron redistribution modeling simultaneously: the electron-counting rule and the symmetry rule. This violation of the physical constraints of chemical reactions impairs model performance. In this work, we propose a new framework called ReactionSink that combines two doubly stochastic self-attention mappings to obtain electron redistribution predictions that follow both constraints. We further extend our solution to a general multi-head attention mechanism with augmented constraints. To achieve this, we apply Sinkhorn's algorithm to iteratively update self-attention mappings, which imposes doubly conservative constraints as additional informative priors on electron redistribution modeling. We theoretically demonstrate that our ReactionSink can simultaneously satisfy both rules, which the current decoder mechanism cannot do. Empirical results show that our approach consistently improves the predictive performance of non-autoregressive models and does not bring an unbearable additional computational cost.

----

## [452] Overlooked Implications of the Reconstruction Loss for VAE Disentanglement

**Authors**: *Nathan Michlo, Richard Klein, Steven James*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/453](https://doi.org/10.24963/ijcai.2023/453)

**Abstract**:

Learning disentangled representations with variational autoencoders (VAEs) is often attributed to the regularisation component of the loss. In this work, we highlight the interaction between data and the reconstruction term of the loss as the main contributor to disentanglement in VAEs. We show that standard benchmark datasets have unintended correlations between their subjective ground-truth factors and perceived axes in the data according to typical VAE reconstruction losses. Our work exploits this relationship to provide a theory for what constitutes an adversarial dataset under a given reconstruction loss. We verify this by constructing an example dataset that prevents disentanglement in state-of-the-art frameworks while maintaining human-intuitive ground-truth factors. Finally, we re-enable disentanglement by designing an example reconstruction loss that is once again able to perceive the ground-truth factors. Our findings demonstrate the subjective nature of disentanglement and the importance of considering the interaction between the ground-truth factors, data and notably, the reconstruction loss, which is under-recognised in the literature.

----

## [453] Social Motivation for Modelling Other Agents under Partial Observability in Decentralised Training

**Authors**: *Dung Nguyen, Hung Le, Kien Do, Svetha Venkatesh, Truyen Tran*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/454](https://doi.org/10.24963/ijcai.2023/454)

**Abstract**:

Understanding other agents is a key challenge in constructing artificial social agents. Current works focus on centralised training, wherein agents are allowed to know all the information about others and the environmental state during training. In contrast, this work studies decentralised training, wherein agents must learn the model of other agents in order to cooperate with them under partially-observable conditions, even during training, i.e. learning agents are myopic. The intrinsic motivation for artificial agents is modelled on the concept of human social motivation that entices humans to meet and understand each other, especially when experiencing a utility loss. Our intrinsic motivation encourages agents to stay near each other to obtain better observations and construct a model of others. They do so when their model of other agents is poor, or the overall task performance is bad during the learning phase. This simple but effective method facilitates the processes of modelling others, resulting in an improvement of the performance in cooperative tasks significantly. Our experiments demonstrate that the socially-motivated agent can model others better and promote cooperation across different tasks.

----

## [454] Efficient NLP Model Finetuning via Multistage Data Filtering

**Authors**: *Xu Ouyang, Shahina Mohd Azam Ansari, Felix Xiaozhu Lin, Yangfeng Ji*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/455](https://doi.org/10.24963/ijcai.2023/455)

**Abstract**:

As model finetuning is central to the modern NLP, we set to maximize its efficiency. Motivated by redundancy in training examples and the sheer sizes of pretrained models, we exploit a key opportunity: training only on important data. To this end, we set to filter training examples in a streaming fashion, in tandem with training the target model. Our key techniques are two: (1) automatically determine a training loss threshold for skipping backward training passes; (2) run a meta predictor for further skipping forward training passes. We integrate the above techniques in a holistic, three-stage training pro- cess. On a diverse set of benchmarks, our method reduces the required training examples by up to 5.3× and training time by up to 6.8×, while only seeing minor accuracy degradation. Our method is effective even for training one epoch, where each training example is encountered only once. It is simple to implement and is compatible with the existing finetuning techniques. Code is available at: https://github.com/xo28/efficient-NLP-multistage-training

----

## [455] Mitigating Disparity while Maximizing Reward: Tight Anytime Guarantee for Improving Bandits

**Authors**: *Vishakha Patil, Vineet Nair, Ganesh Ghalme, Arindam Khan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/456](https://doi.org/10.24963/ijcai.2023/456)

**Abstract**:

We study the Improving Multi-Armed Bandit problem, where the reward obtained from an arm increases with the number of pulls it receives. This model provides an elegant abstraction for many real-world problems in domains such as education and employment, where decisions about the distribution of opportunities can affect the future capabilities of communities and the disparity between them. A decision-maker in such settings must consider the impact of her decisions on future rewards in addition to the standard objective of maximizing her cumulative reward at any time. We study the tension between two seemingly conflicting objectives in the horizon-unaware setting: a) maximizing the cumulative reward at any time and b) ensuring that arms with better long-term rewards get sufficient pulls even if they initially have low rewards. We show that, surprisingly, the two objectives are aligned with each other. Our main contribution is an anytime algorithm for the IMAB problem that achieves the best possible cumulative reward while ensuring that the arms reach their true potential given sufficient time. Our algorithm mitigates the initial disparity due to lack of opportunity and continues pulling an arm until it stops improving. We prove the optimality of our algorithm by showing that a) any algorithm for the IMAB problem, no matter how utilitarian, must suffer Omega(T) policy regret and Omega(k) competitive ratio with respect to the optimal offline policy, and b) the competitive ratio of our algorithm is O(k).

----

## [456] An Empirical Study on the Language Modal in Visual Question Answering

**Authors**: *Daowan Peng, Wei Wei, Xian-Ling Mao, Yuanyuan Fu, Dangyang Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/457](https://doi.org/10.24963/ijcai.2023/457)

**Abstract**:

Generalization beyond in-domain experience to out-of-distribution data is of paramount significance in the AI domain. Of late, state-of-the-art Visual Question Answering (VQA) models have shown impressive performance on in-domain data, partially due to the language prior bias which, however, hinders the generalization ability in practice. This paper attempts to provide new insights into the influence of language modality on VQA performance from an empirical study perspective. To achieve this, we conducted a series of experiments on six models. The results of these experiments revealed that, 1) apart from prior bias caused by question types, there is a notable influence of postfix-related bias in inducing biases, and 2) training VQA models with word-sequence-related variant questions demonstrated improved performance on the out-of-distribution benchmark, and the LXMERT even achieved a 10-point gain without adopting any debiasing methods. We delved into the underlying reasons behind these experimental results and put forward some simple proposals to reduce the models' dependency on language priors. The experimental results demonstrated the effectiveness of our proposed method in improving performance on the out-of-distribution benchmark, VQA-CPv2.  We hope this study can inspire novel insights for future research on designing bias-reduction approaches.

----

## [457] RAIN: RegulArization on Input and Network for Black-Box Domain Adaptation

**Authors**: *Qucheng Peng, Zhengming Ding, Lingjuan Lyu, Lichao Sun, Chen Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/458](https://doi.org/10.24963/ijcai.2023/458)

**Abstract**:

Source-Free domain adaptation transits the source-trained model towards target domain without exposing the source data, trying to dispel these concerns about data privacy and security. However, this paradigm is still at risk of data leakage due to adversarial attacks on the source model. Hence, the Black-Box setting only allows to use the outputs of source model, but still suffers from overfitting on the source domain more severely due to source model's unseen weights. In this paper, we propose a novel approach named RAIN (RegulArization on Input and Network) for Black-Box domain adaptation from both input-level and network-level regularization. For the input-level, we design a new data augmentation technique as Phase MixUp, which highlights task-relevant objects in the interpolations, thus enhancing input-level regularization and class consistency for target models. For network-level, we develop a Subnetwork Distillation mechanism to transfer knowledge from the target subnetwork to the full target network via knowledge distillation, which thus alleviates overfitting on the source domain by learning diverse target representations. Extensive experiments show that our method achieves state-of-the-art performance on several cross-domain benchmarks under both single- and multi-source black-box domain adaptation.

----

## [458] Linear Query Approximation Algorithms for Non-monotone Submodular Maximization under Knapsack Constraint

**Authors**: *Canh V. Pham, Tan D. Tran, Dung K. T. Ha, My T. Thai*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/459](https://doi.org/10.24963/ijcai.2023/459)

**Abstract**:

This work, for the first time, introduces two constant factor approximation algorithms with linear query complexity for non-monotone submodular maximization over a ground set of size n subject to a knapsack constraint, DLA  and RLA. DLA is a deterministic algorithm that provides an approximation factor of nearly 6 while  RLA  is a randomized algorithm with an approximation factor of nearly 4. Both run in linear query complexity. The key idea to obtain a constant approximation ratio with linear query lies in: (1) dividing the ground set into two appropriate subsets to find the near-optimal solution over these subsets with linear queries, and (2) combining a threshold greedy with properties of two disjoint sets or a random selection process to improve solution quality. In addition to the theoretical analysis, we have evaluated our proposed solutions with three applications: Revenue Maximization, Image Summarization, and Maximum Weighted Cut, showing that our algorithms not only return comparative results to state-of-the-art algorithms but also require significantly fewer queries.

----

## [459] On Conditional and Compositional Language Model Differentiable Prompting

**Authors**: *Jonathan Pilault, Can Liu, Mohit Bansal, Markus Dreyer*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/460](https://doi.org/10.24963/ijcai.2023/460)

**Abstract**:

Prompts have been shown to be an effective method to adapt a frozen Pretrained Language Model (PLM) to perform well on downstream tasks. Prompts can be represented by a human-engineered word sequence or by a learned continuous embedding. 
In this work, we investigate conditional and compositional differentiable prompting.
We propose a new model, Prompt Production System (ProPS), which learns to transform task instructions or input metadata, into continuous prompts that elicit task-specific outputs from the PLM. 
Our model uses a modular network structure based on our neural formulation of Production Systems, which allows the model to learn discrete rules -- neural functions that learn to specialize in transforming particular prompt input patterns, making it suitable for compositional transfer learning and few-shot learning. 
We present extensive empirical and theoretical analysis and show that ProPS consistently surpasses other PLM adaptation techniques, and often improves upon fully fine-tuned models, on compositional generalization tasks, controllable summarization and multilingual translation, while needing fewer trainable parameters.

----

## [460] NeuPSL: Neural Probabilistic Soft Logic

**Authors**: *Connor Pryor, Charles Dickens, Eriq Augustine, Alon Albalak, William Yang Wang, Lise Getoor*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/461](https://doi.org/10.24963/ijcai.2023/461)

**Abstract**:

In this paper, we introduce Neural Probabilistic Soft Logic (NeuPSL), a novel neuro-symbolic (NeSy) framework that unites state-of-the-art symbolic reasoning with the low-level perception of deep neural networks. To model the boundary between neural and symbolic representations, we propose a family of energy-based models, NeSy Energy-Based Models, and show that they are general enough to include NeuPSL and many other NeSy approaches. Using this framework, we show how to seamlessly integrate neural and symbolic parameter learning and inference in NeuPSL. Through an extensive empirical evaluation, we demonstrate the benefits of using NeSy methods, achieving upwards of 30% improvement over independent neural network models. On a well-established NeSy task, MNIST-Addition, NeuPSL demonstrates its joint reasoning capabilities by outperforming existing NeSy approaches by up to 10% in low-data settings. Furthermore, NeuPSL achieves a 5% boost in performance over state-of-the-art NeSy methods in a canonical citation network task with up to a 40 times speed up.

----

## [461] FedSampling: A Better Sampling Strategy for Federated Learning

**Authors**: *Tao Qi, Fangzhao Wu, Lingjuan Lyu, Yongfeng Huang, Xing Xie*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/462](https://doi.org/10.24963/ijcai.2023/462)

**Abstract**:

Federated learning (FL) is an important technique for learning models from decentralized data in a privacy-preserving way. Existing FL methods usually uniformly sample clients for local model learning in each round. However, different clients may have significantly different data sizes, and the clients with more data cannot have more opportunities to contribute to model training, which may lead to inferior performance. In this paper, instead of client uniform sampling, we propose a novel data uniform sampling strategy for federated learning (FedSampling), which can effectively improve the performance of federated learning especially when client data size distribution is highly imbalanced across clients. In each federated learning round, local data on each client is randomly sampled for local model learning according to a probability based on the server desired sample size and the total sample size on all available clients. Since the data size on each client is privacy-sensitive, we propose a privacy-preserving way to estimate the total sample size with a differential privacy guarantee. Experiments on four benchmark datasets show that FedSampling can effectively improve the performance of federated learning.

----

## [462] Efficient Online Decision Tree Learning with Active Feature Acquisition

**Authors**: *Arman Rahbar, Ziyu Ye, Yuxin Chen, Morteza Haghir Chehreghani*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/463](https://doi.org/10.24963/ijcai.2023/463)

**Abstract**:

Constructing decision trees online is a classical machine learning problem. Existing works often assume that features are readily available for each incoming data point. However, in many real world applications, both feature values and the labels are unknown a priori and can only be obtained at a cost. For example, in medical diagnosis, doctors have to choose which tests to perform (i.e., making costly feature queries) on a patient in order to make a diagnosis decision (i.e., predicting labels). We provide a fresh perspective to tackle this practical challenge. Our framework consists of an active planning oracle embedded in an online learning scheme for which we investigate several information acquisition functions. Specifically, we employ a surrogate information acquisition function based on adaptive submodularity to actively query feature values with a minimal cost, while using a posterior sampling scheme to maintain a low regret for online prediction. We demonstrate the efficiency and effectiveness of our framework via extensive experiments on various real-world datasets. Our framework also naturally adapts to the challenging setting of online learning with concept drift and is shown to be competitive with baseline models while being more flexible.

----

## [463] Some Might Say All You Need Is Sum

**Authors**: *Eran Rosenbluth, Jan Tönshoff, Martin Grohe*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/464](https://doi.org/10.24963/ijcai.2023/464)

**Abstract**:

The expressivity of Graph Neural Networks (GNNs) is dependent on the aggregation functions they employ. Theoretical works have pointed towards Sum aggregation GNNs subsuming every other GNNs, while certain practical works have observed a clear advantage to using Mean and Max. An examination of the theoretical guarantee identifies two caveats. First, it is size-restricted, that is, the power of every specific GNN is limited to graphs of a specific size. Successfully processing larger graphs may require an other GNN, and so on. Second, it concerns the power to distinguish non-isomorphic graphs, not the power to approximate general functions on graphs, and the former does not necessarily imply the latter.
It is desired that a GNN's usability will not be limited to graphs of any specific size. Therefore, we explore the realm of unrestricted-size expressivity. We prove that basic functions, which can be computed exactly by Mean or Max GNNs, are inapproximable by any Sum GNN. We prove that under certain restrictions, every Mean or Max GNN can be approximated by a Sum GNN, but even there, a combination of (Sum, [Mean/Max]) is more expressive than Sum alone. Lastly, we prove further expressivity limitations for GNNs with a broad class of aggregations.

----

## [464] Sample Efficient Model-free Reinforcement Learning from LTL Specifications with Optimality Guarantees

**Authors**: *Daqian Shao, Marta Kwiatkowska*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/465](https://doi.org/10.24963/ijcai.2023/465)

**Abstract**:

Linear Temporal Logic (LTL) is widely used to specify high-level objectives for system policies, and it is highly desirable for autonomous systems to learn the optimal policy with respect to such specifications. However, learning the optimal policy from LTL specifications is not trivial. We present a model-free Reinforcement Learning (RL) approach that efficiently learns an optimal policy for an unknown stochastic system, modelled using Markov Decision Processes (MDPs). We propose a novel and more general product MDP, reward structure and discounting mechanism that, when applied in conjunction with off-the-shelf model-free RL algorithms, efficiently learn the optimal policy that maximizes the probability of satisfying a given LTL specification with optimality guarantees. We also provide improved theoretical results on choosing the key parameters in RL to ensure optimality. To directly evaluate the learned policy, we adopt probabilistic model checker PRISM to compute the probability of the policy satisfying such specifications. Several experiments on various tabular MDP environments across different LTL tasks demonstrate the improved sample efficiency and optimal policy convergence.

----

## [465] Graph-based Semi-supervised Local Clustering with Few Labeled Nodes

**Authors**: *Zhaiming Shen, Ming-Jun Lai, Sheng Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/466](https://doi.org/10.24963/ijcai.2023/466)

**Abstract**:

Local clustering aims at extracting a local structure inside a graph without the necessity of knowing the entire graph structure. As the local structure is usually small in size compared to the entire graph, one can think of it as a compressive sensing problem where the indices of target cluster can be thought as a sparse solution to a linear system. In this paper, we apply this idea based on two pioneering works under the same framework and propose a new semi-supervised local clustering approach using only few labeled nodes. Our approach improves the existing works by making the initial cut to be the entire graph and hence overcomes a major limitation of the existing works, which is the low quality of initial cut. Extensive experimental results on various datasets demonstrate the effectiveness of our approach.

----

## [466] Co-training with High-Confidence Pseudo Labels for Semi-supervised Medical Image Segmentation

**Authors**: *Zhiqiang Shen, Peng Cao, Hua Yang, Xiaoli Liu, Jinzhu Yang, Osmar R. Zaïane*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/467](https://doi.org/10.24963/ijcai.2023/467)

**Abstract**:

Consistency regularization and pseudo labeling-based semi-supervised methods perform co-training using the pseudo labels from multi-view inputs. However, such co-training models tend to converge early to a consensus, degenerating to the self-training ones, and produce low-confidence pseudo labels from the perturbed inputs during training. 
    To address these issues, we propose an Uncertainty-guided Collaborative Mean-Teacher (UCMT) for semi-supervised semantic segmentation with the high-confidence pseudo labels. Concretely, UCMT consists of two main components: 1) collaborative mean-teacher (CMT) for encouraging model disagreement and performing co-training between the sub-networks, and 2) uncertainty-guided region mix (UMIX) for manipulating the input images according to the uncertainty maps of CMT and facilitating CMT to produce high-confidence pseudo labels. 
    Combining the strengths of UMIX with CMT, UCMT can retain model disagreement and enhance the quality of pseudo labels for the co-training segmentation.
    Extensive experiments on four public medical image datasets including 2D and 3D modalities demonstrate the superiority of UCMT over the state-of-the-art. 
    Code is available at: https://github.com/Senyh/UCMT.

----

## [467] Unreliable Partial Label Learning with Recursive Separation

**Authors**: *Yu Shi, Ning Xu, Hua Yuan, Xin Geng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/468](https://doi.org/10.24963/ijcai.2023/468)

**Abstract**:

Partial label learning (PLL) is a typical weakly supervised learning problem in which each instance is associated with a candidate label set, and among which only one is true. However, the assumption that the ground-truth label is always among the candidate label set would be unrealistic, as the reliability of the candidate label sets in real-world applications cannot be guaranteed by annotators. Therefore, a generalized PLL named Unreliable Partial Label Learning (UPLL) is proposed, in which the true label may not be in the candidate label set. Due to the challenges posed by unreliable labeling, previous PLL methods will experience a marked decline in performance when applied to UPLL. To address the issue, we propose a two-stage framework named Unreliable Partial Label Learning with Recursive Separation (UPLLRS). In the first stage, the self-adaptive recursive separation strategy is proposed to separate the training set into a reliable subset and an unreliable subset. In the second stage, a disambiguation strategy is employed to progressively identify the ground-truth labels in the reliable subset. Simultaneously, semi-supervised learning methods are adopted to extract valuable information from the unreliable subset. Our method demonstrates state-of-the-art performance as evidenced by experimental results, particularly in situations of high unreliability. Code and supplementary materials are available at https://github.com/dhiyu/UPLLRS.

----

## [468] Guide to Control: Offline Hierarchical Reinforcement Learning Using Subgoal Generation for Long-Horizon and Sparse-Reward Tasks

**Authors**: *Wonchul Shin, Yusung Kim*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/469](https://doi.org/10.24963/ijcai.2023/469)

**Abstract**:

Reinforcement learning (RL) has achieved considerable success in many fields, but applying it to real-world problems can be costly and risky because it requires a lot of online interaction. Recently, offline RL has shown the possibility of extracting a solution through existing logged data without online interaction. In this work, we propose an offline hierarchical RL method, Guider (Guide to Control), that can efficiently solve long-horizon and sparse-reward tasks from offline data. The high-level policy sequentially generates a subgoal that can guide the agent to arrive at the final goal, and the lower-level policy learns how to reach each given guided subgoal. In the process of learning from offline data, the key is to make the low-level policy reachable to the generated subgoals. We show that high-quality subgoal generation is possible through pre-training a latent subgoal prior model. The well-regulated subgoal generation improves performance while avoiding distributional shifts in offline RL by breaking down long, complex tasks into shorter, easier ones. For evaluations, Guider outperforms prior offline RL methods in long-horizon robot navigation and complex manipulation benchmarks. Our code is available at https://github.com/gckor/Guider.

----

## [469] MA2CL: Masked Attentive Contrastive Learning for Multi-Agent Reinforcement Learning

**Authors**: *Haolin Song, Mingxiao Feng, Wengang Zhou, Houqiang Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/470](https://doi.org/10.24963/ijcai.2023/470)

**Abstract**:

Recent approaches have utilized self-supervised auxiliary tasks as representation learning to improve the performance and sample efficiency of vision-based reinforcement learning algorithms in single-agent settings. However, in multi-agent reinforcement learning (MARL), these techniques face challenges because each agent only receives partial observation from an environment influenced by others, resulting in correlated observations in the agent dimension. So it is necessary to consider agent-level information in representation learning for MARL. In this paper, we propose an effective framework called Multi-Agent Masked Attentive Contrastive Learning (MA2CL), which encourages learning representation to be both temporal and agent-level predictive by reconstructing the masked agent observation in latent space. Specifically, we use an attention reconstruction model for recovering and the model is trained via contrastive learning. MA2CL allows better utilization of contextual information at the agent level, facilitating the training of MARL agents for cooperation tasks. Extensive experiments demonstrate that our method significantly improves the performance and sample efficiency of different MARL algorithms and outperforms other methods in various vision-based and state-based scenarios.

----

## [470] Handling Learnwares Developed from Heterogeneous Feature Spaces without Auxiliary Data

**Authors**: *Peng Tan, Zhi-Hao Tan, Yuan Jiang, Zhi-Hua Zhou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/471](https://doi.org/10.24963/ijcai.2023/471)

**Abstract**:

The learnware paradigm proposed by Zhou [2016] devotes to constructing a market of numerous well-performed models, enabling users to solve problems by reusing existing efforts rather than starting from scratch. A learnware comprises a trained model and the specification which enables the model to be adequately identified according to the user's requirement. Previous studies concentrated on the homogeneous case where models share the same feature space based on Reduced Kernel Mean Embedding (RKME) specification. However, in real-world scenarios, models are typically constructed from different feature spaces. If such a scenario can be handled by the market, all models built for a particular task even with different feature spaces can be identified and reused for a new user task. Generally, this problem would be easier if there were additional auxiliary data connecting different feature spaces, however, obtaining such data in reality is challenging. In this paper, we present a general framework for accommodating heterogeneous learnwares without requiring additional auxiliary data. The key idea is to utilize the submitted RKME specifications to establish the relationship between different feature spaces. Additionally, we give a matrix factorization-based implementation and propose the overall procedure for constructing and exploiting the heterogeneous learnware market. Experiments on real-world tasks validate the efficacy of our method.

----

## [471] Improving Heterogeneous Model Reuse by Density Estimation

**Authors**: *Anke Tang, Yong Luo, Han Hu, Fengxiang He, Kehua Su, Bo Du, Yixin Chen, Dacheng Tao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/472](https://doi.org/10.24963/ijcai.2023/472)

**Abstract**:

This paper studies multiparty learning, aiming to learn a model using the private data of different participants. Model reuse is a promising solution for multiparty learning, assuming that a local model has been trained for each party. Considering the potential sample selection bias among different parties, some heterogeneous model reuse approaches have been developed. However, although pre-trained local classifiers are utilized in these approaches, the characteristics of the local data are not well exploited. This motivates us to estimate the density of local data and design an auxiliary model together with the local classifiers for reuse. To address the scenarios where some local models are not well pre-trained, we further design a multiparty cross-entropy loss for calibration. Upon existing works, we address a challenging problem of heterogeneous model reuse from a decision theory perspective and take advantage of recent advances in density estimation. Experimental results on both synthetic and benchmark data demonstrate the superiority of the proposed method.

----

## [472] Spike Count Maximization for Neuromorphic Vision Recognition

**Authors**: *Jianxiong Tang, Jian-Huang Lai, Xiaohua Xie, Lingxiao Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/473](https://doi.org/10.24963/ijcai.2023/473)

**Abstract**:

Spiking Neural Networks (SNNs) are the promising models of neuromorphic vision recognition. The mean square error (MSE) and cross-entropy (CE) losses are widely applied to supervise the training of SNNs on neuromorphic datasets. However, the relevance between the output spike counts and predictions is not well modeled by the existing loss functions. This paper proposes a Spike Count Maximization (SCM) training approach for the SNN-based neuromorphic vision recognition model based on optimizing the output spike counts. The SCM is achieved by structural risk minimization (SRM) and a specially designed spike counting loss. The spike counting loss counts the output spikes of the SNN by using the L0-norm, and the SRM maximizes the distance between the margin boundaries of the classifier to ensure the generalization of the model. The SCM is non-smooth and non-differentiable, and we design a two-stage algorithm with fast convergence to solve the problem. Experiment results demonstrate that the SCM performs satisfactorily in most cases. Using the output spikes for prediction, the accuracies of SCM are 2.12%~16.50% higher than the popular training losses on the CIFAR10-DVS dataset. The code is available at https://github.com/TJXTT/SCM-SNN.

----

## [473] Competitive-Cooperative Multi-Agent Reinforcement Learning for Auction-based Federated Learning

**Authors**: *Xiaoli Tang, Han Yu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/474](https://doi.org/10.24963/ijcai.2023/474)

**Abstract**:

Auction-based Federated Learning (AFL) enables open collaboration among self-interested data consumers and data owners. Existing AFL approaches cannot manage the mutual influence among multiple data consumers competing to enlist data owners. Moreover, they cannot support a single data owner to join multiple data consumers simultaneously. To bridge these gaps, we propose the Multi-Agent Reinforcement Learning for AFL (MARL-AFL) approach to steer data consumers to bid strategically
towards an equilibrium with desirable overall system characteristics. We design a temperature-based reward reassignment scheme to make tradeoffs between cooperation and competition among AFL data consumers. In this way, it can reach an equilibrium state that ensures individual data consumers can achieve good utility, while preserving system-level social welfare. To circumvent potential collusion behaviors among data consumers, we introduce a bar agent to set a personalized bidding
lower bound for each data consumer. Extensive experiments on six commonly adopted benchmark datasets show that MARL-AFL is significantly more advantageous compared to six state-of-the-art approaches, outperforming the best by 12.2%, 1.9% and 3.4% in terms of social welfare, revenue and accuracy, respectively.

----

## [474] Calibrating a Deep Neural Network with Its Predecessors

**Authors**: *Linwei Tao, Minjing Dong, Daochang Liu, Changming Sun, Chang Xu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/475](https://doi.org/10.24963/ijcai.2023/475)

**Abstract**:

Confidence calibration - the process to calibrate the output probability distribution of neural networks - is essential for safety-critical applications of such networks. Recent works verify the link between mis-calibration and overfitting. However, early stopping, as a well-known technique to mitigate overfitting, fails to calibrate networks. In this work, we study the limitions of early stopping and comprehensively analyze the overfitting problem of a network considering each individual block. We then propose a novel regularization method, predecessor combination search (PCS), to improve calibration by searching a combination of best-fitting block predecessors, where block predecessors are the corresponding network blocks with weight parameters from earlier training stages. PCS achieves the state-of-the-art calibration performance on multiple datasets and architectures. In addition, PCS improves model robustness under dataset distribution shift. Supplementary material and code are available at https://github.com/Linwei94/PCS

----

## [475] One Model, Any CSP: Graph Neural Networks as Fast Global Search Heuristics for Constraint Satisfaction

**Authors**: *Jan Tönshoff, Berke Kisin, Jakob Lindner, Martin Grohe*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/476](https://doi.org/10.24963/ijcai.2023/476)

**Abstract**:

We propose a universal Graph Neural Network architecture which can be trained as an end-2-end search heuristic for any Constraint Satisfaction Problem (CSP). Our architecture can be trained unsupervised with policy gradient descent to generate problem specific heuristics for any CSP in a purely data driven manner. 
The approach is based on a novel graph representation for CSPs that is both generic and compact and enables us to process every possible CSP instance with one GNN, regardless of constraint arity, relations or domain size. Unlike previous RL-based methods, we operate on a global search action space and allow our GNN to modify any number of variables in every step of the stochastic search. This enables our method to properly leverage the inherent parallelism of GNNs. 
We perform a thorough empirical evaluation where we learn heuristics for well known and important CSPs, both decision and optimisation problems, from random data, including graph coloring, MAXCUT, and MAX-k-SAT, and the general RB model. Our approach significantly outperforms prior end-2-end approaches for neural combinatorial optimization. It can compete with conventional heuristics and solvers on test instances that are several orders of magnitude larger and structurally more complex than those seen during training.

----

## [476] DEIR: Efficient and Robust Exploration through Discriminative-Model-Based Episodic Intrinsic Rewards

**Authors**: *Shanchuan Wan, Yujin Tang, Yingtao Tian, Tomoyuki Kaneko*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/477](https://doi.org/10.24963/ijcai.2023/477)

**Abstract**:

Exploration is a fundamental aspect of reinforcement learning (RL), and its effectiveness is a deciding factor in the performance of RL algorithms, especially when facing sparse extrinsic rewards. Recent studies have shown the effectiveness of encouraging exploration with intrinsic rewards estimated from novelties in observations. However, there is a gap between the novelty of an observation and an exploration, as both the stochasticity in the environment and the agent's behavior may affect the observation. To evaluate exploratory behaviors accurately, we propose DEIR, a novel method in which we theoretically derive an intrinsic reward with a conditional mutual information term that principally scales with the novelty contributed by agent explorations, and then implement the reward with a discriminative forward model. Extensive experiments on both standard and advanced exploration tasks in MiniGrid show that DEIR quickly learns a better policy than the baselines. Our evaluations on ProcGen demonstrate both the generalization capability and the general applicability of our intrinsic reward.

----

## [477] DeLELSTM: Decomposition-based Linear Explainable LSTM to Capture Instantaneous and Long-term Effects in Time Series

**Authors**: *Chaoqun Wang, Yijun Li, Xiangqian Sun, Qi Wu, Dongdong Wang, Zhixiang Huang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/478](https://doi.org/10.24963/ijcai.2023/478)

**Abstract**:

Time series forecasting is prevalent in various real-world applications. Despite the promising results of deep learning models in time series forecasting, especially the Recurrent Neural Networks (RNNs), the explanations of time series models, which are critical in high-stakes applications, have received little attention. In this paper, we propose a Decomposition-based Linear Explainable LSTM (DeLELSTM) to improve the interpretability of LSTM. Conventionally, the interpretability of RNNs only concentrates on the variable importance and time importance. We additionally distinguish between the instantaneous influence of new coming data and the long-term effects of historical data. Specifically, DeLELSTM consists of two components, i.e., standard LSTM and tensorized LSTM. The tensorized LSTM assigns each variable with a unique hidden state making up a matrix h(t), and the standard LSTM models all the variables with a shared hidden state H(t). By decomposing the H(t) into the linear combination of past information h(t-1) and the fresh information h(t)-h(t-1), we can get the instantaneous influence and the long-term effect of each feature. In addition, the advantage of linear regression also makes the explanation transparent and clear. We demonstrate the effectiveness and interpretability of DeLELSTM on three empirical datasets. Extensive experiments show that the proposed method achieves competitive performance against the baseline methods and provides a reliable explanation relative to domain knowledge.

----

## [478] Deep Partial Multi-Label Learning with Graph Disambiguation

**Authors**: *Haobo Wang, Shisong Yang, Gengyu Lyu, Weiwei Liu, Tianlei Hu, Ke Chen, Songhe Feng, Gang Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/479](https://doi.org/10.24963/ijcai.2023/479)

**Abstract**:

In partial multi-label learning (PML), each data example is equipped with a candidate label set, which consists of multiple ground-truth labels and other false-positive labels. Recently, graph-based methods, which demonstrate a good ability to estimate accurate confidence scores from candidate labels, have been prevalent to deal with PML problems. However, we observe that existing graph-based PML methods typically adopt linear multi-label classifiers and thus fail to achieve superior performance. In this work, we attempt to remove several obstacles for extending them to deep models and propose a novel deep Partial multi-Label model with grAph-disambIguatioN (PLAIN). Specifically, we introduce the instance-level and label-level similarities to recover label confidences as well as exploit label dependencies. At each training epoch, labels are propagated on the instance and label graphs to produce relatively accurate pseudo-labels; then, we train the deep model to fit the numerical labels. Moreover, we provide a careful analysis of the risk functions to guarantee the robustness of the proposed model. Extensive experiments on various synthetic datasets and three real-world PML datasets demonstrate that PLAIN achieves significantly superior results to state-of-the-art methods.

----

## [479] Context-Aware Feature Selection and Classification

**Authors**: *Juanyan Wang, Mustafa Bilgic*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/480](https://doi.org/10.24963/ijcai.2023/480)

**Abstract**:

We propose a joint model that performs instance-level feature selection and classification. For a given case, the joint model first skims the full feature vector, decides which features are relevant for that case, and makes a classification decision using only the selected features, resulting in compact, interpretable, and case-specific classification decisions. Because the selected features depend on the case at hand, we refer to this approach as context-aware feature selection and classification. The model can be trained on instances that are annotated by experts with both class labels and instance-level feature selections, so it can select instance-level features that humans would use. Experiments on several datasets demonstrate that the proposed model outperforms eight baselines on a combined classification and feature selection measure, and is able to better emulate the ground-truth instance-level feature selections. The supplementary materials are available at https://github.com/IIT-ML/IJCAI23-CFSC.

----

## [480] From Association to Generation: Text-only Captioning by Unsupervised Cross-modal Mapping

**Authors**: *Junyang Wang, Ming Yan, Yi Zhang, Jitao Sang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/481](https://doi.org/10.24963/ijcai.2023/481)

**Abstract**:

With the development of Vision-Language Pre-training Models (VLPMs) represented by CLIP and ALIGN, significant breakthroughs have been achieved for association-based visual tasks such as image classification and image-text retrieval by the zero-shot capability of CLIP without fine-tuning. However, CLIP is hard to apply to generation-based tasks. This is due to the lack of decoder architecture and pre-training tasks for generation. Although previous works have created generation capacity for CLIP through additional language models, a modality gap between the CLIP representations of different modalities and the inability of CLIP to model the offset of this gap, which results in the failure of the concept to transfer across modes. To solve the problem, we try to map images/videos to the language modality and generate captions from the language modality. In this paper, we propose the K-nearest-neighbor Cross-modality Mapping (Knight), a zero-shot method from association to generation. With vision-free unsupervised training, Knight achieves state-of-the-art performance in zero-shot methods for image captioning and video captioning.

----

## [481] Multi-objective Optimization-based Selection for Quality-Diversity by Non-surrounded-dominated Sorting

**Authors**: *Ren-Jian Wang, Ke Xue, Haopu Shang, Chao Qian, Haobo Fu, Qiang Fu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/482](https://doi.org/10.24963/ijcai.2023/482)

**Abstract**:

Quality-Diversity (QD) algorithms, a subset of evolutionary algorithms, maintain an archive (i.e., a set of solutions) and simulate the natural evolution process through iterative selection and reproduction, with the goal of generating a set of high-quality and diverse solutions. Though having found many successful applications in reinforcement learning, QD algorithms often select the parent solutions uniformly at random, which lacks selection pressure and may limit the performance. Recent studies have treated each type of behavior of a solution as an objective, and selected the parent solutions based on Multi-objective Optimization (MO), which is a natural idea, but has not lead to satisfactory performance as expected. This paper gives the reason for the first time, and then proposes a new MO-based selection method by non-surrounded-dominated sorting (NSS), which considers all possible directions of the behaviors, and thus can generate diverse solutions over the whole behavior space. By combining NSS with the most widespread QD algorithm, MAP-Elites, we perform experiments on synthetic functions and several complex tasks (i.e., QDGym, robotic arm, and Mario environment generation), showing that NSS achieves better performance than not only other MO-based selection methods but also state-of-the-art selection methods in QD.

----

## [482] FedBFPT: An Efficient Federated Learning Framework for Bert Further Pre-training

**Authors**: *Xin'ao Wang, Huan Li, Ke Chen, Lidan Shou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/483](https://doi.org/10.24963/ijcai.2023/483)

**Abstract**:

This study proposes FEDBFPT (Federated BERT Further Pre-Training), a Federated Learning (FL) framework for further pre-training the BERT language model in specialized domains while addressing privacy concerns. FEDBFPT enables multiple clients to collaboratively train the shallower layers of BERT, which are crucial in the pre-training stage, without the need to share private data. To achieve this, FEDBFPT involves building a local model for each client, progressively training the shallower layers of local models while sampling deeper layers, and aggregating trained parameters on a server to create the final global model. This approach utilizes multiple smaller local models to further pre-train a global model targeted at specific tasks via fine-tuning, resulting in a reduction in resource usage while maintaining model accuracy. Theoretical analysis is conducted to support the efficiency of FEDBFPT, and experiments are conducted on corpora across domains such as medicine, biology, and computer science. Results indicate that FEDBFPT achieves performance levels comparable to traditional FL methods while reducing computation and communication costs by 46.70% and 7.04%, respectively, even approaching the performance of centralized training models. The Source code is released at https://github.com/Hanzhouu/FedBFPT.

----

## [483] Contrastive Label Enhancement

**Authors**: *Yifei Wang, Yiyang Zhou, Jihua Zhu, Xinyuan Liu, Wenbiao Yan, Zhiqiang Tian*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/484](https://doi.org/10.24963/ijcai.2023/484)

**Abstract**:

Label distribution learning (LDL) is a new machine learning paradigm for solving label ambiguity. Since it is difficult to directly obtain label distributions, many studies are focusing on how to recover label distributions from logical labels, dubbed label enhancement (LE). Existing LE methods estimate label distributions by simply building a mapping relationship between features and label distributions under the supervision of logical labels. They typically overlook the fact that both features and logical labels are descriptions of the instance from different views. Therefore, we propose a novel method called Contrastive Label Enhancement (ConLE) which integrates features and logical labels into the unified projection space to generate high-level features by contrastive learning strategy. In this approach, features and logical labels belonging to the same sample are pulled closer, while those of different samples are projected farther away from each other in the projection space. Subsequently, we leverage the obtained high-level features to gain label distributions through a well-designed training strategy that considers the consistency of label attributes. Extensive experiments on LDL benchmark datasets demonstrate the effectiveness and superiority of our method.

----

## [484] Scalable Optimal Margin Distribution Machine

**Authors**: *Yilin Wang, Nan Cao, Teng Zhang, Xuanhua Shi, Hai Jin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/485](https://doi.org/10.24963/ijcai.2023/485)

**Abstract**:

Optimal margin Distribution Machine (ODM) is a newly proposed statistical learning framework rooting in the novel margin theory, which demonstrates better generalization performance than the traditional large margin based counterparts. Nonetheless, it suffers from the ubiquitous scalability problem regarding both computation time and memory as other kernel methods. This paper proposes a scalable ODM, which can achieve nearly ten times speedup compared to the original ODM training method. For nonlinear kernels, we propose a novel distribution-aware partition method to make the local ODM trained on each partition be close and converge faster to the global one. When linear kernel is applied, we extend a communication efficient SVRG method to accelerate the training further. Extensive empirical studies validate that our proposed method is highly computational efficient and almost never worsen the generalization.

----

## [485] c-TPE: Tree-structured Parzen Estimator with Inequality Constraints for Expensive Hyperparameter Optimization

**Authors**: *Shuhei Watanabe, Frank Hutter*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/486](https://doi.org/10.24963/ijcai.2023/486)

**Abstract**:

Hyperparameter optimization (HPO) is crucial for strong performance of deep learning algorithms and real-world applications often impose some constraints, such as memory usage, or latency on top of the performance requirement. In this work, we propose constrained TPE (c-TPE), an extension of the widely-used versatile Bayesian optimization method, tree-structured Parzen estimator (TPE), to handle these constraints. Our proposed extension goes beyond a simple combination of an existing acquisition function and the original TPE, and instead includes modifications that address issues that cause poor performance. We thoroughly analyze these modifications both empirically and theoretically, providing insights into how they effectively overcome these challenges. In the experiments, we demonstrate that c-TPE exhibits the best average rank performance among existing methods with statistical significance on 81 expensive HPO with inequality constraints. Due to the lack of baselines, we only discuss the applicability of our method to hard-constrained optimization in Appendix D. See https://arxiv.org/abs/2211.14411 for the latest version with Appendix.

----

## [486] Speeding Up Multi-Objective Hyperparameter Optimization by Task Similarity-Based Meta-Learning for the Tree-Structured Parzen Estimator

**Authors**: *Shuhei Watanabe, Noor H. Awad, Masaki Onishi, Frank Hutter*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/487](https://doi.org/10.24963/ijcai.2023/487)

**Abstract**:

Hyperparameter optimization (HPO) is a vital step in improving performance in deep learning (DL). Practitioners are often faced with the trade-off between multiple criteria, such as accuracy and latency. Given the high computational needs of DL and the growing demand for efficient HPO, the acceleration of multi-objective (MO) optimization becomes ever more important. Despite the significant body of work on meta-learning for HPO, existing methods are inapplicable to MO tree-structured Parzen estimator (MO-TPE), a simple yet powerful MO-HPO algorithm. In this paper, we extend TPE’s acquisition function to the meta-learning setting using a task similarity defined by the overlap of top domains between tasks. We also theoretically analyze and address the limitations of our task similarity. In the experiments, we demonstrate that our method speeds up MO-TPE on tabular HPO benchmarks and attains state-of-the-art performance. Our method was also validated externally by winning the AutoML 2022 competition on “Multiobjective Hyperparameter Optimization for Transformers”. See https://arxiv.org/abs/2212.06751 for the latest version with Appendix.

----

## [487] PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces

**Authors**: *Shuhei Watanabe, Archit Bansal, Frank Hutter*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/488](https://doi.org/10.24963/ijcai.2023/488)

**Abstract**:

The recent rise in popularity of Hyperparameter Optimization (HPO) for deep learning has highlighted the role that good hyperparameter (HP) space design can play in training strong models. In turn, designing a good HP space is critically dependent on understanding the role of different HPs. This motivates research on HP Importance (HPI), e.g., with the popular method of functional ANOVA (f-ANOVA). However, the original f-ANOVA formulation is inapplicable to the subspaces most relevant to algorithm designers, such as those defined by top performance. To overcome this issue, we derive a novel formulation of f-ANOVA for arbitrary subspaces and propose an algorithm that uses Pearson divergence (PED) to enable a closed-form calculation of HPI. We demonstrate that this new algorithm, dubbed PED-ANOVA, is able to successfully identify important HPs in different subspaces while also being extremely computationally efficient. See https://arxiv.org/abs/2304.10255 for the latest version with Appendix.

----

## [488] Generalization Bounds for Adversarial Metric Learning

**Authors**: *Wen Wen, Han Li, Hong Chen, Rui Wu, Lingjuan Wu, Liangxuan Zhu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/489](https://doi.org/10.24963/ijcai.2023/489)

**Abstract**:

Recently, adversarial metric learning has been proposed to enhance the robustness of the learned distance metric against adversarial perturbations. Despite rapid progress in validating its effectiveness empirically, theoretical guarantees on adversarial robustness and generalization are far less understood. To fill this gap, this paper focuses on unveiling the generalization properties of adversarial metric learning by developing the uniform convergence analysis techniques. Based on the capacity estimation of covering numbers, we establish the first high-probability generalization bounds with order O(n^{-1/2}) for adversarial metric learning with pairwise perturbations and general losses, where n is the number of training samples. Moreover, we obtain the refined generalization bounds with order O(n^{-1}) for the smooth loss by using local Rademacher complexity, which is faster than the previous result of adversarial pairwise learning, e.g., adversarial bipartite ranking. Experimental evaluation on real-world datasets validates our theoretical findings.

----

## [489] More for Less: Safe Policy Improvement with Stronger Performance Guarantees

**Authors**: *Patrick Wienhöft, Marnix Suilen, Thiago D. Simão, Clemens Dubslaff, Christel Baier, Nils Jansen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/490](https://doi.org/10.24963/ijcai.2023/490)

**Abstract**:

In an offline reinforcement learning setting, the safe policy improvement (SPI) problem aims to improve the performance of a behavior policy according to which sample data has been generated.
State-of-the-art approaches to SPI require a high number of samples to provide practical probabilistic guarantees on the improved policy's performance.
We present a novel approach to the SPI problem that provides the means to require less data for such guarantees. 
Specifically, to prove the correctness of these guarantees, we devise implicit transformations on the data set and the underlying environment model that serve as theoretical foundations to derive tighter improvement bounds for SPI.
Our empirical evaluation, using the well-established SPI with baseline bootstrapping (SPIBB) algorithm, on standard benchmarks shows that our method indeed significantly reduces the sample complexity of the SPIBB algorithm.

----

## [490] Not Only Pairwise Relationships: Fine-Grained Relational Modeling for Multivariate Time Series Forecasting

**Authors**: *Jinming Wu, Qi Qi, Jingyu Wang, Haifeng Sun, Zhikang Wu, Zirui Zhuang, Jianxin Liao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/491](https://doi.org/10.24963/ijcai.2023/491)

**Abstract**:

Recent graph-based methods achieve significant success in multivariate time series modeling and forecasting due to their ability to handle relationships among time series variables. However, only pairwise relationships are considered in most existing works. They ignore beyond-pairwise relationships and their potential categories in practical scenarios, which leads to incomprehensive relationship learning for multivariate time series forecasting. In this paper, we present ReMo, a Relational Modeling-based method, to promote fine-grained relational learning among multivariate time series data. Firstly, by treating time series variables and complex relationships as nodes and hyperedges, we extract multi-view hypergraphs from data to capture beyond-pairwise relationships. Secondly, a novel hypergraph message passing strategy is designed to characterize both nodes and hyperedges by inferring the potential categories of relationships and further distinguishing their impacts on time series variables. By integrating these two modules into the time series forecasting framework, ReMo effectively improves the performance of multivariate time series forecasting. The experimental results on seven commonly used datasets from different domains demonstrate the superiority of our model.

----

## [491] FedNoRo: Towards Noise-Robust Federated Learning by Addressing Class Imbalance and Label Noise Heterogeneity

**Authors**: *Nannan Wu, Li Yu, Xuefeng Jiang, Kwang-Ting Cheng, Zengqiang Yan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/492](https://doi.org/10.24963/ijcai.2023/492)

**Abstract**:

Federated noisy label learning (FNLL) is emerging as a promising tool for privacy-preserving multi-source decentralized learning. Existing research, relying on the assumption of class-balanced global data, might be incapable to model complicated label noise, especially in medical scenarios. In this paper, we first formulate a new and more realistic federated label noise problem where global data is class-imbalanced and label noise is heterogeneous, and then propose a two-stage framework named FedNoRo for noise-robust federated learning. Specifically, in the first stage of FedNoRo, per-class loss indicators followed by Gaussian Mixture Model are deployed for noisy client identification. In the second stage, knowledge distillation and a distance-aware aggregation function are jointly adopted for noise-robust federated model updating. Experimental results on the widely-used ICH and ISIC2019 datasets demonstrate the superiority of FedNoRo against the state-of-the-art FNLL methods for addressing class imbalance and label noise heterogeneity in real-world FL scenarios.

----

## [492] Singularformer: Learning to Decompose Self-Attention to Linearize the Complexity of Transformer

**Authors**: *Yifan Wu, Shichao Kan, Min Zeng, Min Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/493](https://doi.org/10.24963/ijcai.2023/493)

**Abstract**:

Transformers achieve excellent performance in a variety of domains since they can capture long-distance dependencies through the self-attention mechanism. However, self-attention is computationally costly due to its quadratic complexity and high memory consumption. In this paper, we propose a novel Transformer variant (Singularformer) that uses neural networks to learn the singular value decomposition process of the attention matrix to design a linear-complexity and memory-efficient global self-attention mechanism. Specifically, we decompose the attention matrix into the product of three matrix factors based on singular value decomposition and design neural networks to learn these matrix factors, then the associative law of matrix multiplication is used to linearize the calculation of self-attention. The above procedure allows us to compute self-attention as two-dimensional reduction processes in the first and second token dimensional spaces, followed by a multi-head self-attention computational process on the first dimensional reduced token features. Experimental results on 8 real-world datasets demonstrate that Singularformer performs favorably against the other Transformer variants with lower time and space complexity. Our source code is publicly available at https://github.com/CSUBioGroup/Singularformer.

----

## [493] ProMix: Combating Label Noise via Maximizing Clean Sample Utility

**Authors**: *Ruixuan Xiao, Yiwen Dong, Haobo Wang, Lei Feng, Runze Wu, Gang Chen, Junbo Zhao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/494](https://doi.org/10.24963/ijcai.2023/494)

**Abstract**:

Learning with Noisy Labels (LNL) has become an appealing topic, as imperfectly annotated data are relatively cheaper to obtain. Recent state-of-the-art approaches employ specific selection mechanisms to separate clean and noisy samples and then apply Semi-Supervised Learning (SSL) techniques for improved performance. However, the selection step mostly provides a medium-sized and decent-enough clean subset, which overlooks a rich set of clean samples. To fulfill this, we propose a novel LNL framework ProMix that attempts to maximize the utility of clean samples for boosted performance. Key to our method, we propose a matched high confidence selection technique that selects those examples with high confidence scores and matched predictions with given labels to dynamically expand a base clean sample set. To overcome the potential side effect of excessive clean set selection procedure, we further devise a novel SSL framework that is able to train balanced and unbiased classifiers on the separated clean and noisy samples. Extensive experiments demonstrate that ProMix significantly advances the current state-of-the-art results on multiple benchmarks with different types and levels of noise. It achieves an average improvement of 2.48% on the CIFAR-N dataset.

----

## [494] Violin: Virtual Overbridge Linking for Enhancing Semi-supervised Learning on Graphs with Limited Labels

**Authors**: *Siyue Xie, Da Sun Handason Tam, Wing Cheong Lau*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/495](https://doi.org/10.24963/ijcai.2023/495)

**Abstract**:

Graph Neural Networks (GNNs) is a family of promising tools for graph semi-supervised learning. However, in training, most existing GNNs rely heavily on a large amount of labeled data, which is rare in real-world scenarios. Unlabeled data with useful information are usually under-exploited, which limits the representation power of GNNs. To handle these problems, we propose Virtual Overbridge Linking (Violin), a generic framework to enhance the learning capacity of common GNNs. By learning to add virtual overbridges between two nodes that are estimated to be semantic-consistent, labeled and unlabeled data can be correlated. Supervised information can be well utilized in training while simultaneously inducing the model to learn from unlabeled data. Discriminative relation patterns extracted from unlabeled nodes can also be shared with other nodes even if they are remote from each other. Motivated by recent advances in data augmentations, we additionally integrate Violin with the consistency regularized training. Such a scheme yields node representations with better robustness, which significantly enhances a GNN. Violin can be readily extended to a wide range of GNNs without introducing additional learnable parameters. Extensive experiments on six datasets demonstrate that our method is effective and robust under low-label rate scenarios, where Violin can boost some GNNs' performance by over 10% on node classifications.

----

## [495] Distilling Universal and Joint Knowledge for Cross-Domain Model Compression on Time Series Data

**Authors**: *Qing Xu, Min Wu, Xiaoli Li, Kezhi Mao, Zhenghua Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/496](https://doi.org/10.24963/ijcai.2023/496)

**Abstract**:

For many real-world time series tasks, the computational complexity of prevalent deep leaning models often hinders the deployment on resource limited environments (e.g., smartphones). Moreover, due to the inevitable domain shift between model training (source) and deploying (target) stages, compressing those deep models under cross-domain scenarios becomes more challenging. Although some of existing works have already explored cross-domain knowledge distillation for model compression, they are either biased to source data or heavily tangled between source and target data. To this end, we design a novel end-to-end framework called UNiversal and joInt Knowledge Distillation (UNI-KD) for cross-domain model compression. In particular, we propose to transfer both the universal feature-level knowledge across source and target domains and the joint logit-level knowledge shared by both domains from the teacher to the student model via an adversarial learning scheme. More specifically, a feature-domain discriminator is employed to align teacher’s and student’s representations for universal knowledge transfer. A data-domain discriminator is utilized to prioritize the domain-shared samples for joint knowledge transfer. Extensive experimental results on four time series datasets demonstrate the superiority of our proposed method over state-of-the-art (SOTA) benchmarks. The source code is available at https://github.com/ijcai2023/UNI KD.

----

## [496] Expanding the Hyperbolic Kernels: A Curvature-aware Isometric Embedding View

**Authors**: *Meimei Yang, Pengfei Fang, Hui Xue*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/497](https://doi.org/10.24963/ijcai.2023/497)

**Abstract**:

Modeling data relation as a hierarchical structure has proven beneficial for many learning scenarios, and the hyperbolic space, with negative curvature, can encode such data hierarchy without distortion. Several recent studies also show that the representation power of the hyperbolic space can be further improved by endowing the kernel methods. Unfortunately, the known kernel methods, developed in hyperbolic space, are limited by the adaptation capacity or distortion issues. This paper addresses the issues through a novel embedding function. To this end, we propose a curvature-aware isometric embedding, which establishes an isometry from the Poincar\'e model to a special reproducing kernel Hilbert space (RKHS). Then we can further define a series of kernels on this RKHS, including several positive definite kernels and an indefinite kernel. Thorough experiments are conducted to demonstrate the superiority of our proposals over existing-known hyperbolic and Euclidean kernels in various learning tasks, e.g., graph learning and zero-shot learning.

----

## [497] BARA: Efficient Incentive Mechanism with Online Reward Budget Allocation in Cross-Silo Federated Learning

**Authors**: *Yunchao Yang, Yipeng Zhou, Miao Hu, Di Wu, Quan Z. Sheng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/498](https://doi.org/10.24963/ijcai.2023/498)

**Abstract**:

Federated learning (FL) is a prospective distributed machine learning  framework that can preserve data privacy. In particular, cross-silo FL can complete model training by making isolated data islands of different organizations collaborate with a parameter server (PS) via exchanging model parameters for multiple communication rounds. In cross-silo FL, an incentive mechanism is indispensable  for motivating data owners to contribute their models to FL training. However, how to allocate the reward budget among different rounds is an essential but complicated problem  largely overlooked by existing works. The challenge of this problem lies in the opaque feedback between reward budget allocation and model utility improvement of FL, making the optimal reward budget allocation complicated. To address this problem, we design an online reward budget allocation algorithm using Bayesian optimization named BARA (Budget Allocation for Reverse Auction). Specifically, BARA can model the complicated relationship between reward budget allocation and final model accuracy in FL based on historical training records so that the reward budget allocated to each communication round is dynamically optimized so as to maximize the final model utility. We further incorporate the BARA algorithm into reverse auction-based incentive mechanisms to illustrate its effectiveness. Extensive experiments are conducted on real datasets to demonstrate that BARA significantly outperforms competitive baselines by improving  model utility with the same amount of reward budget.

----

## [498] Generalized Discriminative Deep Non-Negative Matrix Factorization Based on Latent Feature and Basis Learning

**Authors**: *Zijian Yang, Zhiwei Li, Lu Sun*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/499](https://doi.org/10.24963/ijcai.2023/499)

**Abstract**:

As a powerful tool for data representation, deep NMF has attracted much attention in recent years. Current deep NMF builds the multi-layer structure by decomposing either basis matrix or feature matrix into multiple factors, and probably complicates the learning process when data is insufficient or exhibits simple structure. To overcome the limitations, a novel method called Generalized Deep Non-negative Matrix Factorization (GDNMF) is proposed, which generalizes several NMF and deep NMF methods in a unified framework. GDNMF simultaneously performs decomposition on both features and bases, which learns a hierarchical data representation based on multi-level basis. To further improve the latent representation and enhance its flexibility, GDNMF mutually reinforces shallow linear model and deep non-linear model. Moreover, semi-supervised GDNMF is proposed by treating partial label information as soft constraints in the multi-layer structure. An efficient two-phase optimization algorithm is developed, and experiments on five real-world datesets verify its superior performance compared with state-of-the-art methods.

----

## [499] Multi-Task Learning via Time-Aware Neural ODE

**Authors**: *Feiyang Ye, Xuehao Wang, Yu Zhang, Ivor W. Tsang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/500](https://doi.org/10.24963/ijcai.2023/500)

**Abstract**:

Multi-Task Learning (MTL) is a well-established paradigm for learning shared models for a diverse set of tasks. Moreover, MTL improves data efficiency by jointly training all tasks simultaneously. However, directly optimizing the losses of all the tasks may lead to imbalanced performance on all the tasks due to the competition among tasks for the shared parameters in MTL models. Many MTL methods try to mitigate this problem by dynamically weighting task losses or manipulating task gradients. Different from existing studies, in this paper, we propose a Neural Ordinal diffeRential equation based Multi-tAsk Learning (NORMAL) method to alleviate this issue by modeling task-specific feature transformations from the perspective of dynamic flows built on the Neural Ordinary Differential Equation (NODE). Specifically, the proposed NORMAL model designs a time-aware neural ODE block to learn task-specific time information, which determines task positions of feature transformations in the dynamic flow, in NODE automatically via gradient descent methods. In this way, the proposed NORMAL model handles the problem of competing shared parameters by learning task positions. Moreover, the learned task positions can be used to measure the relevance among different tasks. Extensive experiments show that the proposed NORMAL model outperforms state-of-the-art MTL models.

----

## [500] LGI-GT: Graph Transformers with Local and Global Operators Interleaving

**Authors**: *Shuo Yin, Guoqiang Zhong*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/501](https://doi.org/10.24963/ijcai.2023/501)

**Abstract**:

Since Transformers can alleviate some critical and fundamental problems of graph neural networks (GNNs), such as over-smoothing, over-squashing and limited expressiveness, they have been successfully applied to graph representation learning and achieved impressive results. However, although there are many works dedicated to make graph Transformers (GTs) aware of the structure and edge information by specifically tailored attention forms or graph-related positional and structural encodings, few works address the problem of how to construct high-performing GTs with modules of GNNs and Transformers. In this paper, we propose a novel graph Transformer with local and global operators interleaving (LGI-GT), in which we further design a new method propagating embeddings of the [CLS] token for global information representation. Additionally, we propose an effective message passing module called edge enhanced local attention (EELA), which makes LGI-GT a full-attention GT. Extensive experiments demonstrate that LGI-GT performs consistently better than previous state-of-the-art GNNs and GTs, while ablation studies show the effectiveness of the proposed LGI scheme and EELA. The source code of LGI-GT is available at https://github.com/shuoyinn/LGI-GT.

----

## [501] On the Reuse Bias in Off-Policy Reinforcement Learning

**Authors**: *Chengyang Ying, Zhongkai Hao, Xinning Zhou, Hang Su, Dong Yan, Jun Zhu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/502](https://doi.org/10.24963/ijcai.2023/502)

**Abstract**:

Importance sampling (IS) is a popular technique in off-policy evaluation, which re-weights the return of trajectories in the replay buffer to boost sample efficiency. However, training with IS can be unstable and previous attempts to address this issue mainly focus on analyzing the variance of IS. In this paper, we reveal that the instability is also related to a new notion of Reuse Bias of IS --- the bias in off-policy evaluation caused by the reuse of the replay buffer for evaluation and optimization. We theoretically show that the off-policy evaluation and optimization of the current policy with the data from the replay buffer result in an overestimation of the objective, which may cause an erroneous gradient update and degenerate the performance. We further provide a high-probability upper bound of the Reuse Bias and show that controlling one term of the upper bound can control the Reuse Bias by introducing the concept of stability for off-policy algorithms. Based on these analyses, we present a novel yet simple Bias-Regularized Importance Sampling (BIRIS) framework along with practical algorithms, which can alleviate the negative impact of the Reuse Bias, and show that our BIRIS can significantly reduce the Reuse Bias empirically. Moreover, extensive experimental results show that our BIRIS-based methods can significantly improve the sample efficiency on a series of continuous control tasks in MuJoCo.

----

## [502] Adversarial Amendment is the Only Force Capable of Transforming an Enemy into a Friend

**Authors**: *Chong Yu, Tao Chen, Zhongxue Gan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/503](https://doi.org/10.24963/ijcai.2023/503)

**Abstract**:

Adversarial attack is commonly regarded as a huge threat to neural networks because of misleading behavior. This paper presents an opposite perspective: adversarial attacks can be harnessed to improve neural models if amended correctly. Unlike traditional adversarial defense or adversarial training schemes that aim to improve the adversarial robustness, the proposed adversarial amendment (AdvAmd) method aims to improve the original accuracy level of neural models on benign samples. We thoroughly analyze the distribution mismatch between the benign and adversarial samples. This distribution mismatch and the mutual learning mechanism with the same learning ratio applied in prior art defense strategies is the main cause leading the accuracy degradation for benign samples. The proposed AdvAmd is demonstrated to steadily heal the accuracy degradation and even leads to a certain accuracy boost of common neural models on benign classification, object detection, and segmentation tasks. The efficacy of the AdvAmd is contributed by three key components: mediate samples (to reduce the influence of distribution mismatch with a fine-grained amendment), auxiliary batch norm (to solve the mutual learning mechanism and the smoother judgment surface), and AdvAmd loss (to adjust the learning ratios according to different attack vulnerabilities) through quantitative and ablation experiments.

----

## [503] CLE-ViT: Contrastive Learning Encoded Transformer for Ultra-Fine-Grained Visual Categorization

**Authors**: *Xiaohan Yu, Jun Wang, Yongsheng Gao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/504](https://doi.org/10.24963/ijcai.2023/504)

**Abstract**:

Ultra-fine-grained visual classification (ultra-FGVC) targets at classifying sub-grained categories of fine-grained objects. This inevitably requires discriminative representation learning within a limited training set. Exploring intrinsic features from the object itself, e.g., predicting the rotation of a given image, has demonstrated great progress towards learning discriminative representation. Yet none of these works consider explicit supervision for learning mutual information at instance level. To this end, this paper introduces CLE-ViT, a novel contrastive learning encoded transformer, to address the fundamental problem in ultra-FGVC. The core design is a self-supervised module that performs self-shuffling and masking and then distinguishes these altered images from other images. This drives the model to learn an optimized feature space that has a large inter-class distance while remaining tolerant to intra-class variations. By incorporating this self-supervised module, the network acquires more knowledge from the intrinsic structure of the input data, which improves the generalization ability without requiring extra manual annotations. CLE-ViT demonstrates strong performance on 7 publicly available datasets, demonstrating its effectiveness in the ultra-FGVC task. The code is available at https://github.com/Markin-Wang/CLEViT.

----

## [504] Explainable Reinforcement Learning via a Causal World Model

**Authors**: *Zhongwei Yu, Jingqing Ruan, Dengpeng Xing*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/505](https://doi.org/10.24963/ijcai.2023/505)

**Abstract**:

Generating explanations for reinforcement learning (RL) is challenging as actions may produce long-term effects on the future. In this paper, we develop a novel framework for explainable RL by learning a causal world model without prior knowledge of the causal structure of the environment. The model captures the influence of actions, allowing us to interpret the long-term effects of actions through causal chains, which present how actions influence environmental variables and finally lead to rewards. Different from most explanatory models which suffer from low accuracy, our model remains accurate while improving explainability, making it applicable in model-based learning. As a result, we demonstrate that our causal model can serve as the bridge between explainability and learning.

----

## [505] Hierarchical State Abstraction based on Structural Information Principles

**Authors**: *Xianghua Zeng, Hao Peng, Angsheng Li, Chunyang Liu, Lifang He, Philip S. Yu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/506](https://doi.org/10.24963/ijcai.2023/506)

**Abstract**:

State abstraction optimizes decision-making by ignoring irrelevant environmental information in reinforcement learning with rich observations. Nevertheless, recent approaches focus on adequate representational capacities resulting in essential information loss, affecting their performances on challenging tasks. In this article, we propose a novel mathematical Structural Information principles-based State Abstraction framework, namely SISA, from the information-theoretic perspective. Specifically, an unsupervised, adaptive hierarchical state clustering method without requiring manual assistance is presented, and meanwhile, an optimal encoding tree is generated. On each non-root tree node, a new aggregation function and condition structural entropy are designed to achieve hierarchical state abstraction and compensate for sampling-induced essential information loss in state abstraction. Empirical evaluations on a visual gridworld domain and six continuous control benchmarks demonstrate that, compared with five SOTA state abstraction approaches, SISA significantly improves mean episode reward and sample efficiency up to 18.98 and 44.44%, respectively. Besides, we experimentally show that SISA is a general framework that can be flexibly integrated with different representation-learning objectives to improve their performances further.

----

## [506] Dual Personalization on Federated Recommendation

**Authors**: *Chunxu Zhang, Guodong Long, Tianyi Zhou, Peng Yan, Zijian Zhang, Chengqi Zhang, Bo Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/507](https://doi.org/10.24963/ijcai.2023/507)

**Abstract**:

Federated recommendation is a new Internet service architecture that aims to provide privacy-preserving recommendation services in federated settings. Existing solutions are used to combine distributed recommendation algorithms and privacy-preserving mechanisms. Thus it inherently takes the form of heavyweight models at the server and hinders the deployment of on-device intelligent models to end-users. This paper proposes a novel Personalized Federated Recommendation (PFedRec) framework to learn many user-specific lightweight models to be deployed on smart devices rather than a heavyweight model on a server. Moreover, we propose a new dual personalization mechanism to effectively learn fine-grained personalization on both users and items. The overall learning process is formulated into a unified federated optimization framework. Specifically, unlike previous methods that share exactly the same item embeddings across users in a federated system, dual personalization allows mild finetuning of item embeddings for each user to generate user-specific views for item representations which can be integrated into existing federated recommendation methods to gain improvements immediately. Experiments on multiple benchmark datasets have demonstrated the effectiveness of PFedRec and the dual personalization mechanism. Moreover, we provide visualizations and in-depth analysis of the personalization techniques in item embedding, which shed novel insights on the design of recommender systems in federated settings. The code is available.

----

## [507] Denial-of-Service or Fine-Grained Control: Towards Flexible Model Poisoning Attacks on Federated Learning

**Authors**: *Hangtao Zhang, Zeming Yao, Leo Yu Zhang, Shengshan Hu, Chao Chen, Alan Wee-Chung Liew, Zhetao Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/508](https://doi.org/10.24963/ijcai.2023/508)

**Abstract**:

Federated learning (FL) is vulnerable to poisoning attacks, where adversaries corrupt the global aggregation results and cause denial-of-service (DoS). Unlike recent model poisoning attacks that optimize the amplitude of malicious perturbations along certain prescribed directions to cause DoS, we propose a flexible model poisoning attack (FMPA) that can achieve versatile attack goals. We consider a practical threat scenario where no extra knowledge about the FL system (e.g., aggregation rules or updates on benign devices) is available to adversaries. FMPA exploits the global historical information to construct an estimator that predicts the next round of the global model as a benign reference. It then fine-tunes the reference model to obtain the desired poisoned model with low accuracy and small perturbations. Besides the goal of causing DoS, FMPA can be naturally extended to launch a fine-grained controllable attack, making it possible to precisely reduce the global accuracy. Armed with precise control, malicious FL service providers can gain advantages over their competitors without getting noticed, hence opening a new attack surface in FL other than DoS. Even for the purpose of DoS, experiments show that FMPA significantly decreases the global accuracy, outperforming six state-of-the-art attacks.

----

## [508] G2Pxy: Generative Open-Set Node Classification on Graphs with Proxy Unknowns

**Authors**: *Qin Zhang, Zelin Shi, Xiaolin Zhang, Xiaojun Chen, Philippe Fournier-Viger, Shirui Pan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/509](https://doi.org/10.24963/ijcai.2023/509)

**Abstract**:

Node classification is the task of predicting the labels of unlabeled nodes in a graph. State-of-the-art methods based on graph neural networks achieve  excellent performance when all labels are available
 during training. But in real-life, models are of ten applied on data with new classes, which can lead to massive misclassification and thus significantly degrade performance. Hence, developing
 open-set classification methods is crucial to determine if a given sample belongs to a known class. Existing methods for open-set node classification generally use transductive learning with part or all
 of the features of real unseen class nodes to help with open-set classification. In this paper, we propose a novel generative open-set node classification method, i.e., G2Pxy, which follows a stricter inductive learning setting where no information about unknown classes is available during training and validation. Two kinds of proxy unknown nodes, inter-class unknown proxies and external unknown proxies are generated via mixup to efficiently anticipate the distribution of novel classes. Using the generated proxies, a closed-set classifier can be transformed into an open-set one, by augmenting it with an extra proxy classifier. Under the constraints
 of both cross entropy loss and complement entropy loss, G2Pxy achieves superior effectiveness for unknown class detection and known class classification, which is validated by experiments on bench
mark graph datasets. Moreover, G2Pxy does not have specific requirement on the GNN architecture and shows good generalizations.

----

## [509] Learning to Binarize Continuous Features for Neuro-Rule Networks

**Authors**: *Wei Zhang, Yongxiang Liu, Zhuo Wang, Jianyong Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/510](https://doi.org/10.24963/ijcai.2023/510)

**Abstract**:

Neuro-Rule Networks (NRNs) emerge as a promising neuro-symbolic method, enjoyed by the ability to equate fully-connected neural networks with logic rules. To support learning logic rules consisting of boolean variables, converting input features into binary representations is required. Different from discrete features that could be directly transformed by one-hot encodings, continuous features need to be binarized based on some numerical intervals. Existing studies usually select the bound values of intervals based on empirical strategies (e.g., equal-width interval). However, it is not optimal since the bounds are fixed and cannot be optimized to accommodate the ultimate training target. In this paper, we propose AutoInt, an approach that automatically binarizes continuous features and enables the intervals to be optimized with NRNs in an end-to-end fashion. Specifically, AutoInt automatically selects an interval for a given continuous feature in a soft manner to enable a differentiable learning procedure of interval-related parameters. Moreover, it introduces an additional soft K-means clustering loss to make the interval centres approach the original feature value distribution, thus reducing the risk of overfitting intervals. We conduct comprehensive experiments on public datasets and demonstrate the effectiveness of AutoInt in boosting the performance of NRNs.

----

## [510] SSML-QNet: Scale-Separative Metric Learning Quadruplet Network for Multi-modal Image Patch Matching

**Authors**: *Xiuwei Zhang, Yi Sun, Yamin Han, Yanping Li, Hanlin Yin, Yinghui Xing, Yanning Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/511](https://doi.org/10.24963/ijcai.2023/511)

**Abstract**:

Multi-modal image matching is very challenging due to the significant diversities in visual appearance of different modal images. Typically, the existing well-performed methods mainly focus on learning invariant and discriminative features for measuring the relation between multi-modal image pairs. However, these methods often take the features as a whole and largely overlook the fact that different scale features for a same image pair may have different similarity, which may lead to sub-optimal results only. In this work, we propose a Scale-Separative Metric Learning Quadruplet network (SSML-QNet) for multi-modal image patch matching. Specifically, SSML-QNet can extract both relevant and irrelevant features of imaging modality with the proposed quadruplet network architecture. Then, the proposed Scale-Separative Metric Learning module separately encodes the similarity of different scale features with the pyramid structure. And for each scale, cross-modal consistent features are extracted and measured by coordinate and channel-wise attention sequentially. This makes our network robust to appearance divergence caused by different imaging mechanism. Experiments on the benchmark dataset (VIS-NIR, VIS-LWIR, Optical-SAR, and Brown) have verified that the proposed SSML-QNet is able to outperform other state-of-the-art methods. Furthermore, the cross-dataset transferring experiments on these four datasets also have shown that the proposed method has powerful ability of cross-dataset transferring.

----

## [511] Communication-Efficient Stochastic Gradient Descent Ascent with Momentum Algorithms

**Authors**: *Yihan Zhang, Meikang Qiu, Hongchang Gao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/512](https://doi.org/10.24963/ijcai.2023/512)

**Abstract**:

Numerous machine learning models can be formulated as a stochastic minimax optimization problem, such as imbalanced data classification with AUC maximization. 
Developing efficient algorithms to optimize such kinds of problems is of importance and necessity. However, most existing algorithms restrict their focus on the single-machine setting so that they are incapable of dealing with the large communication overhead in a distributed training system. Moreover, most existing communication-efficient optimization algorithms only focus on the traditional minimization problem, failing to handle the minimax optimization problem. To address these challenging issues, in this paper, we develop two novel communication-efficient stochastic gradient descent ascent with momentum algorithms for the distributed minimax optimization problem, which can significantly reduce the communication cost via the two-way compression scheme. However, the compressed momentum makes it considerably challenging to investigate the convergence rate of our algorithms, especially in the presence of the interaction between the minimization and maximization subproblems. In this paper, we successfully addressed these challenges and established the convergence rate of our algorithms for nonconvex-strongly-concave problems. To the best of our knowledge, our algorithms are the first communication-efficient algorithm with theoretical guarantees for the minimax optimization problem. Finally, we apply our algorithm to the distributed AUC maximization problem for the imbalanced data classification task. Extensive experimental results confirm the efficacy of our algorithm in saving communication costs.

----

## [512] Multi-level Graph Contrastive Prototypical Clustering

**Authors**: *Yuchao Zhang, Yuan Yuan, Qi Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/513](https://doi.org/10.24963/ijcai.2023/513)

**Abstract**:

Recently, graph neural networks (GNNs) have drawn a surge of investigations in deep graph clustering. Nevertheless, existing approaches predominantly are inclined to semantic-agnostic since GNNs exhibit inherent limitations in capturing global underlying semantic structures. Meanwhile, multiple objectives are imposed within one latent space, whereas representations from different granularities may presumably conflict with each other, yielding severe performance degradation for clustering. To this end, we propose a novel Multi-Level Graph Contrastive Prototypical Clustering (MLG-CPC) framework for end-to-end clustering. Specifically, a Prototype Discrimination (ProDisc) objective function is proposed to explicitly capture semantic information via cluster assignments. Moreover, to alleviate the issue of objectives conflict, we introduce to perceive representations of different granularities within individual feature-, prototypical-, and cluster-level spaces by the feature decorrelation, prototype contrast, and cluster space consistency respectively. Extensive experiments on four benchmarks demonstrate the superiority of the proposed MLG-CPC against the state-of-the-art graph clustering approaches.

----

## [513] Adaptive Reward Shifting Based on Behavior Proximity for Offline Reinforcement Learning

**Authors**: *Zhe Zhang, Xiaoyang Tan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/514](https://doi.org/10.24963/ijcai.2023/514)

**Abstract**:

One of the major challenges of the current offline reinforcement learning research is to deal with the distribution shift problem due to the change in state-action visitations for the new policy. To address this issue, we present a novel reward shifting-based method. Specifically, to regularize the behavior of the new policy at each state, we modify the reward to be received by the new policy by shifting it adaptively according to its proximity to the behavior policy, and apply the reward shifting along opposite directions for in-distribution actions and the ones not. In this way we are able to guide the learning procedure of the new policy itself by influencing the consequence of its actions explicitly, helping it to achieve a better balance between behavior constraints and policy improvement. Empirical results on the popular D4RL benchmarks show that the proposed method obtains competitive performance compared to the state-of-art baselines.

----

## [514] Unbiased Gradient Boosting Decision Tree with Unbiased Feature Importance

**Authors**: *Zheyu Zhang, Tianping Zhang, Jian Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/515](https://doi.org/10.24963/ijcai.2023/515)

**Abstract**:

Gradient Boosting Decision Tree (GBDT) has achieved remarkable success in a wide variety of applications. The split finding algorithm, which determines the tree construction process, is one of the most crucial components of GBDT. However, the split finding algorithm has long been criticized for its bias towards features with a large number of potential splits. This bias introduces severe interpretability and overfitting issues in GBDT. To this end, we provide a fine-grained analysis of bias in GBDT and demonstrate that the bias originates from 1) the systematic bias in the gain estimation of each split and 2) the bias in the split finding algorithm resulting from the use of the same data to evaluate the split improvement and determine the best split. Based on the analysis, we propose unbiased gain, a new unbiased measurement of gain importance using out-of-bag samples. Moreover, we incorporate the unbiased property into the split finding algorithm and develop UnbiasedGBM to solve the overfitting issue of GBDT. We assess the performance of UnbiasedGBM and unbiased gain in a large-scale empirical study comprising 60 datasets and show that: 1) UnbiasedGBM exhibits better performance than popular GBDT implementations such as LightGBM, XGBoost, and Catboost on average on the 60 datasets and 2) unbiased gain achieves better average performance in feature selection than popular feature importance methods.

----

## [515] DPMAC: Differentially Private Communication for Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Canzhe Zhao, Yanjie Ze, Jing Dong, Baoxiang Wang, Shuai Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/516](https://doi.org/10.24963/ijcai.2023/516)

**Abstract**:

Communication lays the foundation for cooperation in human society and in multi-agent reinforcement learning (MARL). Humans also desire to maintain their privacy when communicating with others, yet such privacy concern has not been considered in existing works in MARL. We propose the differentially private multi-agent communication (DPMAC) algorithm, which protects the sensitive information of individual agents by equipping each agent with a local message sender with rigorous (epsilon, delta)-differential privacy (DP) guarantee. In contrast to directly perturbing the messages with predefined DP noise as commonly done in privacy-preserving scenarios, we adopt a stochastic message sender for each agent respectively and incorporate the DP requirement into the sender, which automatically adjusts the learned message distribution to alleviate the instability caused by DP noise. Further, we prove the existence of a Nash equilibrium in cooperative MARL with privacy-preserving communication, which suggests that this problem is game-theoretically learnable. Extensive experiments demonstrate a clear advantage of DPMAC over baseline methods in privacy-preserving scenarios.

----

## [516] LGPConv: Learnable Gaussian Perturbation Convolution for Lightweight Pansharpening

**Authors**: *Chen-Yu Zhao, Tian-Jing Zhang, Ran Ran, Zhi-Xuan Chen, Liang-Jian Deng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/517](https://doi.org/10.24963/ijcai.2023/517)

**Abstract**:

Pansharpening is a crucial and challenging task that aims to obtain a high spatial resolution image by merging a multispectral (MS) image and a panchromatic (PAN) image. Current methods use CNNs with standard convolution, but we've observed strong correlation among channel dimensions in the kernel, leading to computational burden and redundancy. To address this, we propose Learnable Gaussian Perturbation Convolution (LGPConv), surpassing standard convolution. LGPConv leverages two properties of standard convolution kernels: 1) correlations within channels, learning a premier kernel as a base to reduce parameters and training difficulties caused by redundancy; 2) introducing Gaussian noise perturbations to simulate randomness and enhance nonlinear representation within channels. We incorporate LGPConv into a well-designed pansharpening network and demonstrate its superiority through extensive experiments, achieving state-of-the-art performance with minimal parameters (27K). Code is available on the GitHub page of the authors.

----

## [517] Graph Neural Convection-Diffusion with Heterophily

**Authors**: *Kai Zhao, Qiyu Kang, Yang Song, Rui She, Sijie Wang, Wee Peng Tay*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/518](https://doi.org/10.24963/ijcai.2023/518)

**Abstract**:

Graph neural networks (GNNs) have shown promising results across various graph learning tasks, but they often assume homophily, which can result in poor performance on heterophilic graphs. The connected nodes are likely to be from different classes or have dissimilar features on heterophilic graphs. In this paper, we propose a novel GNN that incorporates the principle of heterophily by modeling the flow of information on nodes using the convection-diffusion equation (CDE). This allows the CDE to take into account both the diffusion of information due to homophily and the ``convection'' of information due to heterophily. We conduct extensive experiments, which suggest that our framework can achieve competitive performance on node classification tasks for heterophilic graphs, compared to the state-of-the-art methods. The code is available at https://github.com/zknus/Graph-Diffusion-CDE.

----

## [518] Reducing Communication for Split Learning by Randomized Top-k Sparsification

**Authors**: *Fei Zheng, Chaochao Chen, Lingjuan Lyu, Binhui Yao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/519](https://doi.org/10.24963/ijcai.2023/519)

**Abstract**:

Split learning is a simple solution for Vertical Federated Learning (VFL), which has drawn substantial attention in both research and application due to its simplicity and efficiency. However, communication efficiency is still a crucial issue for split learning. In this paper, we investigate multiple communication reduction methods for split learning, including cut layer size reduction, top-k sparsification, quantization, and L1 regularization. Through analysis of the cut layer size reduction and top-k sparsification, we further propose randomized top-k sparsification, to make the model generalize and converge better. This is done by selecting top-k elements with a large probability while also having a small probability to select non-top-k elements. Empirical results show that compared with other communication-reduction methods, our proposed randomized top-k sparsification achieves a better model performance under the same compression level.

----

## [519] MAT: Mixed-Strategy Game of Adversarial Training in Fine-tuning

**Authors**: *Zhehua Zhong, Tianyi Chen, Zhen Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/520](https://doi.org/10.24963/ijcai.2023/520)

**Abstract**:

Fine-tuning large-scale pre-trained language models has been demonstrated effective for various natural language processing (NLP) tasks. Previous studies have established that incorporating adversarial training during the fine-tuning stage can significantly enhance model generalization and robustness. However, from the perspective of game theory, such utilizations of adversarial training correspond to pure-strategy games, which are inherently limited in terms of the scope of their strategies, thereby still having room for improvement. In order to push the performance boundaries, we propose a novel Mixed-strategy Adversarial Training algorithm (MAT). Methodologically, we derive the Nash equilibrium of a mixed-strategy game for adversarial training using Entropy Mirror Descent to establish MAT by sampling method. To verify the effectiveness of MAT, we conducted extensive benchmark experiments on large-scale pre-trained models, such as BERT and RoBERTa. MAT significantly outperforms the state-of-the-art methods on both the GLUE and ANLI benchmarks in terms of generalization and robustness.

----

## [520] pTSE: A Multi-model Ensemble Method for Probabilistic Time Series Forecasting

**Authors**: *Yunyi Zhou, Zhixuan Chu, Yijia Ruan, Ge Jin, Yuchen Huang, Sheng Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/521](https://doi.org/10.24963/ijcai.2023/521)

**Abstract**:

Various probabilistic time series forecasting models have sprung up and shown remarkably good performance. However, the choice of model highly relies on the characteristics of the input time series and the fixed distribution that model is based on. Due to the fact that the probability distributions cannot be averaged over different models straightforwardly, the current time series model ensemble methods cannot be directly applied to improve the robustness and accuracy of forecasting. To address this issue, we propose pTSE, a multi-model distribution ensemble method for probabilistic forecasting based on Hidden Markov Model (HMM). pTSE only takes off-the-shelf outputs from member models without requiring further information about each model. Besides, we provide a complete theoretical analysis of pTSE to prove that the empirical distribution of time series subject to an HMM will converge to the stationary distribution almost surely. Experiments on benchmarks show the superiority of pTSE over all member models and competitive ensemble methods.

----

## [521] Towards Long-delayed Sparsity: Learning a Better Transformer through Reward Redistribution

**Authors**: *Tianchen Zhu, Yue Qiu, Haoyi Zhou, Jianxin Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/522](https://doi.org/10.24963/ijcai.2023/522)

**Abstract**:

Recently, Decision Transformer (DT) pioneered the offline RL into a contextual conditional sequence modeling paradigm, which leverages self-attended autoregression to learn from global target rewards, states, and actions. However, many applications have a severe delay of the above signals, such as the agent can only obtain a reward signal at the end of each trajectory. This delay causes an unwanted bias cumulating in autoregressive learning global signals. In this paper, we focused its virtual example on episodic reinforcement learning with trajectory feedback. We propose a new reward redistribution algorithm for learning parameterized reward functions, and it decomposes the long-delayed reward onto each timestep. To improve the redistributing's adaptation ability, we formulate the previous decomposition as a bi-level optimization problem for global optimal. We extensively evaluate the proposed method on various benchmarks and demonstrate an overwhelming performance improvement under long-delayed settings.

----

## [522] Hierarchical Transformer for Scalable Graph Learning

**Authors**: *Wenhao Zhu, Tianyu Wen, Guojie Song, Xiaojun Ma, Liang Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/523](https://doi.org/10.24963/ijcai.2023/523)

**Abstract**:

Graph Transformer is gaining increasing attention in the field of machine learning and has demonstrated state-of-the-art performance on benchmarks for graph representation learning. However, as current implementations of Graph Transformer primarily focus on learning representations of small-scale graphs, the quadratic complexity of the global self-attention mechanism presents a challenge for full-batch training when applied to larger graphs. Additionally, conventional sampling-based methods fail to capture necessary high-level contextual information, resulting in a significant loss of performance. In this paper, we introduce the Hierarchical Scalable Graph Transformer (HSGT) as a solution to these challenges. HSGT successfully scales the Transformer architecture to node representation learning tasks on large-scale graphs, while maintaining high performance. By utilizing graph hierarchies constructed through coarsening techniques, HSGT efficiently updates and stores multi-scale information in node embeddings at different levels. Together with sampling-based training methods, HSGT effectively captures and aggregates multi-level information on the hierarchical graph using only Transformer blocks. Empirical evaluations demonstrate that HSGT achieves state-of-the-art performance on large-scale benchmarks with graphs containing millions of nodes with high efficiency.

----

## [523] Causal Deep Reinforcement Learning Using Observational Data

**Authors**: *Wenxuan Zhu, Chao Yu, Qiang Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/524](https://doi.org/10.24963/ijcai.2023/524)

**Abstract**:

Deep reinforcement learning (DRL) requires the collection of interventional data, which is sometimes expensive and even unethical in the real world, such as in the autonomous driving and the medical field. Offline reinforcement learning promises to alleviate this issue by exploiting the vast amount of observational data available in the real world. However, observational data may mislead the learning agent to undesirable outcomes if the behavior policy that generates the data depends on unobserved random variables (i.e., confounders). In this paper, we propose two deconfounding methods in DRL to address this problem. The methods first calculate the importance degree of different samples based on the causal inference technique, and then adjust the impact of different samples on the loss function by reweighting or resampling the offline dataset to ensure its unbiasedness. These deconfounding methods can be flexibly combined with existing model-free DRL algorithms such as soft actor-critic and deep Q-learning, provided that a weak condition can be satisfied by the loss functions of these algorithms. We prove the effectiveness of our deconfounding methods and validate them experimentally.

----

## [524] Prediction with Incomplete Data under Agnostic Mask Distribution Shift

**Authors**: *Yichen Zhu, Jian Yuan, Bo Jiang, Tao Lin, Haiming Jin, Xinbing Wang, Chenghu Zhou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/525](https://doi.org/10.24963/ijcai.2023/525)

**Abstract**:

Data with missing values is ubiquitous in many applications. Recent years have witnessed increasing attention on prediction with only incomplete data consisting of observed features and a mask that indicates the missing pattern. Existing methods assume that the training and testing distributions are the same, which may be violated in real-world scenarios. In this paper, we consider prediction with incomplete data in the presence of distribution shift. We focus on the case where the underlying joint distribution of complete features and label is invariant, but the missing pattern, i.e., mask distribution may shift agnostically between training and testing. To achieve generalization, we leverage the observation that for each mask, there is an invariant optimal predictor. To avoid the exponential explosion when learning them separately, we approximate the optimal predictors jointly using a double parameterization technique. This has the undesirable side effect of allowing the learned predictors to rely on the intra-mask correlation and that between features and mask. We perform decorrelation to minimize this effect. Combining the techniques above, we propose a novel prediction method called StableMiss. Extensive experiments on both synthetic and real-world datasets show that StableMiss is robust and outperforms state-of-the-art methods under agnostic mask distribution shift.

----

## [525] Graph Sampling-based Meta-Learning for Molecular Property Prediction

**Authors**: *Xiang Zhuang, Qiang Zhang, Bin Wu, Keyan Ding, Yin Fang, Huajun Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/526](https://doi.org/10.24963/ijcai.2023/526)

**Abstract**:

Molecular property is usually observed with a limited number of samples, and researchers have considered property prediction as a few-shot problem. One important fact that has been ignored by prior works is that each molecule can be recorded with several different properties simultaneously. To effectively utilize many-to-many correlations of molecules and properties, we propose a Graph Sampling-based Meta-learning (GS-Meta) framework for few-shot molecular property prediction. First, we construct a Molecule-Property relation Graph (MPG): molecule and properties are nodes, while property labels decide edges. Then, to utilize the topological information of MPG,  we reformulate an episode in meta-learning as a subgraph of the MPG, containing a target property node, molecule nodes, and auxiliary property nodes. Third, as episodes in the form of subgraphs are no longer independent of each other, we propose to schedule the subgraph sampling process with a contrastive loss function, which considers the consistency and discrimination of subgraphs. Extensive experiments on 5 commonly-used benchmarks show GS-Meta consistently outperforms state-of-the-art methods by  5.71%-6.93% in ROC-AUC and verify the effectiveness of each proposed module. Our code is available at https://github.com/HICAI-ZJU/GS-Meta.

----

## [526] A Noisy-Label-Learning Formulation for Immune Repertoire Classification and Disease-Associated Immune Receptor Sequence Identification

**Authors**: *Mingcai Chen, Yu Zhao, Zhonghuang Wang, Bing He, Jianhua Yao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/527](https://doi.org/10.24963/ijcai.2023/527)

**Abstract**:

Immune repertoire classification, a typical multiple instance learning (MIL) problem, is a frontier research topic in computational biology that makes transformative contributions to new vaccines and immune therapies. However, the traditional instance-space MIL, directly assigning bag-level labels to instances, suffers from the massive amount of noisy labels and extremely low witness rate. In this work, we propose a noisy-label-learning formulation to solve the immune repertoire classification task. To remedy the inaccurate supervision of repertoire-level labels for a sequence-level classifier, we design a robust training strategy: The initial labels are smoothed to be asymmetric and are progressively corrected using the model's predictions throughout the training process. Furthermore, two models with the same architecture but different parameter initialization are co-trained simultaneously to remedy the known ``confirmation bias'' problem in the self-training-like schema. As a result, we obtain accurate sequence-level classification and, subsequently, repertoire-level classification. Experiments on the Cytomegalovirus (CMV) and Cancer datasets demonstrate our method's effectiveness and superior performance on sequence-level and repertoire-level tasks. Code available at https://github.com/TencentAILabHealthcare/NLL-IRC.

----

## [527] Specifying and Testing k-Safety Properties for Machine-Learning Models

**Authors**: *Maria Christakis, Hasan Ferit Eniser, Jörg Hoffmann, Adish Singla, Valentin Wüstholz*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/528](https://doi.org/10.24963/ijcai.2023/528)

**Abstract**:

Machine-learning models are becoming increasingly prevalent in our lives, for instance assisting in image-classification or decision-making tasks. Consequently, the reliability of these models is of critical importance and has resulted in the development of numerous approaches for validating and verifying their robustness and fairness. However, beyond such specific properties, it is challenging to specify, let alone check, general functional-correctness expectations from models. In this paper, we take inspiration from specifications used in formal methods, expressing functional-correctness properties by reasoning about k different executions---so-called k-safety properties. Considering a credit-screening model of a bank, the expected property that "if a person is denied a loan and their income decreases, they should still be denied the loan" is a 2-safety property. Here, we show the wide applicability of k-safety properties for machine-learning models and present the first specification language for expressing them. We also operationalize the language in a framework for automatically validating such properties using metamorphic testing. Our experiments show that our framework is effective in identifying property violations, and that detected bugs could be used to train better models.

----

## [528] A Generalized Deep Markov Random Fields Framework for Fake News Detection

**Authors**: *Yiqi Dong, Dongxiao He, Xiaobao Wang, Yawen Li, Xiaowen Su, Di Jin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/529](https://doi.org/10.24963/ijcai.2023/529)

**Abstract**:

Recently, the wanton dissemination of fake news on social media has adversely affected our lives, rendering automatic fake news detection a pressing issue. Current methods are often fully supervised and typically employ deep neural networks (DNN) to learn implicit relevance from labeled data, ignoring explicitly shared properties (e.g., inflammatory expressions) across fake news. To address this limitation, we propose a graph-theoretic framework, called Generalized Deep Markov Random Fields Framework (GDMRFF), that inherits the capability of deep learning while at the same time exploiting the correlations among the news articles (including labeled and unlabeled data). Specifically, we first leverage a DNN-based module to learn implicit relations, which we then reveal as the unary function of MRF. Pairwise functions with refining effects to encapsulate human insights are designed to capture the explicit association among all samples. Meanwhile, an event removal module is introduced to remove event impact on pairwise functions. Note that we train GDMRFF with the semi-supervised setting, which decreases the reliance on labeled data while maximizing the potential of unlabeled data. We further develop an Ambiguity Learning Guided MRF (ALGM) model as a concretization of GDMRFF.  Experiments show that ALGM outperforms the compared methods significantly on two datasets, especially when labeled data is limited.

----

## [529] StockFormer: Learning Hybrid Trading Machines with Predictive Coding

**Authors**: *Siyu Gao, Yunbo Wang, Xiaokang Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/530](https://doi.org/10.24963/ijcai.2023/530)

**Abstract**:

Typical RL-for-finance solutions directly optimize trading policies over the noisy market data, such as stock prices and trading volumes, without explicitly considering the future trends and correlations of different investment assets as we humans do. In this paper, we present StockFormer, a hybrid trading machine that integrates the forward modeling capabilities of predictive coding with the advantages of RL agents in policy flexibility. The predictive coding part consists of three Transformer branches with modified structures, which respectively extract effective latent states of long-/short-term future dynamics and asset relations. The RL agent adaptively fuses these states and then executes an actor-critic algorithm in the unified state space. The entire model is jointly trained by propagating the critic's gradients back to the predictive coding module. StockFormer significantly outperforms existing approaches across three publicly available financial datasets in terms of portfolio returns and Sharpe ratios.

----

## [530] Pseudo-Labeling Enhanced by Privileged Information and Its Application to In Situ Sequencing Images

**Authors**: *Marzieh Haghighi, Mario C. Cruz, Erin Weisbart, Beth A. Cimini, Avtar Singh, Julia Bauman, Maria E. Lozada, Sanam L. Kavari, James T. Neal, Paul C. Blainey, Anne E. Carpenter, Shantanu Singh*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/531](https://doi.org/10.24963/ijcai.2023/531)

**Abstract**:

Various strategies for label-scarce object detection have been explored by the computer vision research community. These strategies mainly rely on assumptions that are specific to natural images and not directly applicable to the biological and biomedical vision domains. For example, most semi-supervised learning strategies rely on a small set of labeled data as a confident source of ground truth. In many biological vision applications, however, the ground truth is unknown and indirect information might be available in the form of noisy estimations or orthogonal evidence. In this work, we frame a crucial problem in spatial transcriptomics - decoding barcodes from In-Situ-Sequencing (ISS) images - as a semi-supervised object detection (SSOD) problem. Our proposed framework incorporates additional available sources of information into a semi-supervised learning framework in the form of privileged information. The privileged information is incorporated into the teacher's pseudo-labeling in a teacher-student self-training iteration. Although the available privileged information could be data domain specific, we have introduced a general strategy of pseudo-labeling enhanced by privileged information (PLePI) and exemplified the concept using ISS images, 
as well on the COCO benchmark using extra evidence provided by CLIP.

----

## [531] Relation-enhanced DETR for Component Detection in Graphic Design Reverse Engineering

**Authors**: *Xixuan Hao, Danqing Huang, Jieru Lin, Chin-Yew Lin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/532](https://doi.org/10.24963/ijcai.2023/532)

**Abstract**:

It is a common practice for designers to create digital prototypes from a mock-up/screenshot. Reverse engineering graphic design by detecting its components (e.g., text, icon, button) helps expedite this process. This paper first conducts a statistical analysis to emphasize the importance of relations in graphic layouts, which further motivates us to incorporate relation modeling into component detection.  Built on the current state-of-the-art DETR (DEtection TRansformer), we introduce a learnable relation matrix to model class correlations. Specifically, the matrix will be added in the DETR decoder to update the query-to-query self-attention. Experiment results on three public datasets show that our approach achieves better performance than several strong baselines. We further visualize the learnt relation matrix and observe some reasonable patterns. Moreover, we show an application of component detection where we leverage the detection outputs as augmented training data for layout generation, which achieves promising results.

----

## [532] Sequential Attention Source Identification Based on Feature Representation

**Authors**: *Dongpeng Hou, Zhen Wang, Chao Gao, Xuelong Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/533](https://doi.org/10.24963/ijcai.2023/533)

**Abstract**:

Snapshot observation based source localization has been widely studied due to its accessibility and low cost. However, the interaction of users in existing methods does not be addressed in time-varying infection scenarios. So these methods have a decreased accuracy in heterogeneous interaction scenarios. To solve this critical issue, this paper proposes a sequence-to-sequence based localization framework called Temporal-sequence based Graph Attention Source Identification (TGASI) based on an inductive learning idea. More specifically, the encoder focuses on generating multiple features by estimating the influence probability between two users, and the decoder distinguishes the importance of prediction sources in different timestamps by a designed temporal attention mechanism. It's worth mentioning that the inductive learning idea ensures that TGASI can detect the sources in new scenarios without knowing other prior knowledge, which proves the scalability of TGASI. Comprehensive experiments with the SOTA methods demonstrate the higher detection performance and scalability in different scenarios of TGASI.

----

## [533] Differentially Private Partial Set Cover with Applications to Facility Location

**Authors**: *George Z. Li, Dung Nguyen, Anil Vullikanti*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/534](https://doi.org/10.24963/ijcai.2023/534)

**Abstract**:

Set Cover is a fundamental problem in combinatorial optimization which has been studied for many decades due to its various applications across multiple domains. In many of these domains, the input data consists of locations, relationships, and other sensitive information of individuals which may leaked due to the set cover output. Attempts have been made to design privacy-preserving algorithms to solve the Set Cover under privacy constraints. Under differential privacy, it has been proved that the Set Cover problem has strong impossibility results and no explicit forms of the output can be released to the public.

In this work, we observe that these hardness results dissolve when we turn to the Partial Set Cover problem, where we only need to cover a ρ ∈ (0,1) fraction of the elements. We show that this relaxation enables us to avoid the impossibility results, and give the first algorithm which outputs an explicit form of set cover with non-trivial utility guarantees under differential privacy. Using our algorithm as a subroutine, we design a differentially private bicriteria algorithm to solve a recently proposed facility location problem for vaccine distribution which generalizes the k-supplier with outliers. Our analysis shows that relaxing the covering requirement to serve only a ρ ∈ (0,1) fraction of the population/universe also allows us to circumvent the inherent hardness of k-supplier and give the first non-trivial guarantees.

----

## [534] Voice Guard: Protecting Voice Privacy with Strong and Imperceptible Adversarial Perturbation in the Time Domain

**Authors**: *Jingyang Li, Dengpan Ye, Long Tang, Chuanxi Chen, Shengshan Hu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/535](https://doi.org/10.24963/ijcai.2023/535)

**Abstract**:

Adversarial example is a rising tool for voice privacy protection. By adding imperceptible noise to public audio, it prevents tampers from using zero-shot Voice Conversion (VC) to synthesize high quality speech with target speaker identity. However, many existing studies ignore the human perception characteristics of audio data, and it is challenging to generate strong and imperceptible adversarial audio. In this paper, we propose the Voice Guard defense method, which uses a novel method to advance the adversarial perturbation to the time domain to avoid the loss caused by cross-domain conversion. And the psychoacoustic model is introduced into the defense of VC for the first time, which greatly improves the disruption ability and concealment of adversarial audio. We also standardize the evaluation metrics of adversarial audio for the first time, combining multi-dimensional metrics to define the criteria for defense. We evaluate Voice Guard on several state-of-the-art zero-shot VC models. The experimental results show that our method can ensure the perceptual quality of adversarial audio while having a strong defense capability, and is far superior to previous works in terms of disruption ability and concealment.

----

## [535] GLPocket: A Multi-Scale Representation Learning Approach for Protein Binding Site Prediction

**Authors**: *Peiying Li, Yongchang Liu, Shikui Tu, Lei Xu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/536](https://doi.org/10.24963/ijcai.2023/536)

**Abstract**:

Protein binding site prediction is an important prerequisite for the discovery of new drugs. Usually, natural 3D U-Net is adopted as the standard site prediction framework to do per-voxel binary mask classification. However, this scheme only performs feature extraction for single-scale samples, which may bring the loss of global or local information, resulting in incomplete, artifacted or even missed predictions. To tackle this issue, we propose a network called GLPocket, which is based on the Lmser (Least mean square error reconstruction) network and utilizes multi-scale representation to predict binding sites. Firstly, GLPocket uses Target Cropping Block (TCB) for targeted prediction. TCB selects the local interested feature from the global representations to perform concentrated prediction, and reduces the volume of feature maps to be calculated by 82% without adding additional parameters. It integrates global distribution information into local regions, making prediction more concentrated on decoding stage. Secondly, GLPocket establishes long-range relationship of patches within the local region with Transformer Block (TB), to enrich local context semantic information. Experiments show that GLPocket improves by 0.5%-4% on DCA Top-n prediction compared with previous state-of-the-art methods on four datasets. Our code has been released in https://github.com/CMACH508/GLPocket.

----

## [536] Multi-view Contrastive Learning Hypergraph Neural Network for Drug-Microbe-Disease Association Prediction

**Authors**: *Luotao Liu, Feng Huang, Xuan Liu, Zhankun Xiong, Menglu Li, Congzhi Song, Wen Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/537](https://doi.org/10.24963/ijcai.2023/537)

**Abstract**:

Identifying the potential associations among drugs, microbes and diseases is of great significance in exploring the pathogenesis and improving precision medicine. There are plenty of computational methods for pair-wise association prediction, such as drug-microbe and microbe-disease associations, but few methods focus on the higher-order triple-wise drug-microbe-disease (DMD) associations. Driven by the advancement of hypergraph neural networks (HGNNs), we expect them to fully capture high-order interaction patterns behind the hypergraph formulated by DMD associations and realize sound prediction performance. However, the confirmed DMD associations are insufficient due to the high cost of in vitro screening, which forms a sparse DMD hypergraph and thus brings in suboptimal generalization ability. To mitigate the limitation, we propose a Multi-view Contrastive Learning Hypergraph Neural Network, named MCHNN, for DMD association prediction. We design a novel multi-view contrastive learning on the DMD hypergraph as an auxiliary task, which guides the HGNN to learn more discriminative representations and enhances the generalization ability. Extensive computational experiments show that MCHNN achieves satisfactory performance in DMD association prediction and, more importantly, demonstrate the effectiveness of our devised multi-view contrastive learning on the sparse DMD hypergraph.

----

## [537] Robust Steganography without Embedding Based on Secure Container Synthesis and Iterative Message Recovery

**Authors**: *Ziping Ma, Yuesheng Zhu, Guibo Luo, Xiyao Liu, Gerald Schaefer, Hui Fang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/538](https://doi.org/10.24963/ijcai.2023/538)

**Abstract**:

Synthesis-based steganography without embedding (SWE) methods transform secret messages to container images synthesised by generative networks, which eliminates distortions of container images and thus can fundamentally resist typical steganalysis tools. However, existing methods suffer from weak message recovery robustness, synthesis fidelity, and the risk of message leakage. To address these problems, we propose a novel robust steganography without embedding method in this paper. In particular, we design a secure weight modulation-based generator by introducing secure factors to hide secret messages in synthesised container images. In this manner, the synthesised results are modulated by secure factors and thus the secret messages are inaccessible when using fake factors, thus reducing the risk of message leakage. Furthermore, we design a difference predictor via the reconstruction of tampered container images together with an adversarial training strategy to iteratively update the estimation of hidden messages. This ensures robustness of recovering hidden messages, while degradation of synthesis fidelity is reduced since the generator is not included in the adversarial training. Extensive experimental results convincingly demonstrate that our proposed method is effective in avoiding message leakage and superior to other existing methods in terms of recovery robustness and synthesis fidelity.

----

## [538] Choosing Well Your Opponents: How to Guide the Synthesis of Programmatic Strategies

**Authors**: *Rubens O. Moraes, David S. Aleixo, Lucas N. Ferreira, Levi H. S. Lelis*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/539](https://doi.org/10.24963/ijcai.2023/539)

**Abstract**:

This paper introduces Local Learner (2L), an algorithm for providing a set of reference strategies to guide the search for programmatic strategies in two-player zero-sum games. Previous learning algorithms, such as Iterated Best Response (IBR), Fictitious Play (FP), and Double-Oracle (DO), can be computationally expensive or miss important information for guiding search algorithms. 2L actively selects a set of reference strategies to improve the search signal. We empirically demonstrate the advantages of our approach while guiding a local search algorithm for synthesizing strategies in three games, including MicroRTS, a challenging real-time strategy game. Results show that 2L learns reference strategies that provide a stronger search signal than IBR, FP, and DO. We also simulate a tournament of MicroRTS, where a synthesizer using 2L outperformed the winners of the two latest MicroRTS competitions, which were programmatic strategies written by human programmers.

----

## [539] Toward Convex Manifolds: A Geometric Perspective for Deep Graph Clustering of Single-cell RNA-seq Data

**Authors**: *Nairouz Mrabah, Mohamed Mahmoud Amar, Mohamed Bouguessa, Abdoulaye Banire Diallo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/540](https://doi.org/10.24963/ijcai.2023/540)

**Abstract**:

The deep clustering paradigm has shown great potential for discovering complex patterns that can reveal cell heterogeneity in single-cell RNA sequencing data. This paradigm involves two training phases: pretraining based on a pretext task and fine-tuning using pseudo-labels. Although current models yield promising results, they overlook the geometric distortions that regularly occur during the training process. More precisely, the transition between the two phases results in a coarse flattening of the latent structures, which can deteriorate the clustering performance. In this context, existing methods perform euclidean-based embedding clustering without ensuring the flatness and convexity of the latent manifolds. To address this problem, we incorporate two mechanisms. First, we introduce an overclustering loss to flatten the local curves. Second, we propose an adversarial mechanism to adjust the global geometric configuration. The second mechanism gradually transforms the latent structures into convex ones. Empirical results on a variety of gene expression datasets show that our model outperforms state-of-the-art methods.

----

## [540] Unveiling Concepts Learned by a World-Class Chess-Playing Agent

**Authors**: *Aðalsteinn Pálsson, Yngvi Björnsson*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/541](https://doi.org/10.24963/ijcai.2023/541)

**Abstract**:

In recent years, the state-of-the-art agents for playing abstract board games, like chess and others, have moved from using intricate hand-crafted models for evaluating the merits of individual game states toward using neural networks (NNs). This development has eased the encapsulation of the relevant domain-specific knowledge and resulted in much-improved playing strength. However, this has come at the cost of making the resulting models ill-interpretable and challenging to understand and use for enhancing human knowledge. Using a world-class superhuman-strength chess-playing engine as our testbed, we show how recent model probing interpretability techniques can shed light on concepts learned by the engine's NN. Furthermore, to gain additional insight, we contrast the game-state evaluations of the NN to that of its counterpart hand-crafted evaluation model and identify and explain some of the main differences.

----

## [541] Revisiting the Evaluation of Deep Learning-Based Compiler Testing

**Authors**: *Yongqiang Tian, Zhenyang Xu, Yiwen Dong, Chengnian Sun, Shing-Chi Cheung*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/542](https://doi.org/10.24963/ijcai.2023/542)

**Abstract**:

A high-quality program generator is essential to effective automated compiler testing. Engineering such a program generator is difficult, time-consuming, and specific to the language under testing, thus requiring tremendous efforts from human experts with language-specific domain knowledge. To avoid repeatedly writing program generators for different languages, researchers recently proposed a language-agnostic approach based on deep learning techniques to automatically learn a program generator (referred to as DLG) from existing programs. Evaluations show that DLGs outperform Language-Specific Program Generators (LSGs) in testing compilers.
However, we argue that it is unfair to use LSGs as baselines to evaluate DLGs. LSGs aim to validate compiler optimizations by only generating compilable, well-defined test programs; this restriction inevitably impairs the diversity of the language features used in the generated programs. In contrast, DLGs do not aim to validate the correctness of compiler optimizations, and its generated programs are not guaranteed to be well-defined or even compilable. Therefore, it is not surprising that DLG-generated programs are more diverse in terms of used language features than LSG-generated ones. 

This study revisits the evaluation of DLGs, and proposes a new, fair, simple yet strong baseline named Kitten for evaluating DLGs. Given a dataset consisting of human-written programs, instead of using deep learning techniques to learn a program generator, Kitten directly derives new programs by mutating the programs in the dataset. Extensive experiments with more than 1,500 CPU-hours demonstrate that the state-of-the-art DLGs fail to compete against such a simple baseline: 3 v.s. 1,750 hang bugs, 1 v.s. 34 distinct compiler crashes. We believe that DLGs still have a large room for improvement.

----

## [542] Transferable Curricula through Difficulty Conditioned Generators

**Authors**: *Sidney Tio, Pradeep Varakantham*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/543](https://doi.org/10.24963/ijcai.2023/543)

**Abstract**:

Advancements in reinforcement learning (RL) have demonstrated superhuman performance in complex tasks such as Starcraft, Go, Chess etc. However, knowledge transfer from Artificial ``Experts" to humans remain a significant challenge. A promising avenue for such transfer would be the use of curricula. Recent methods in curricula generation focuses on training RL agents efficiently, yet such methods rely on surrogate measures to track student progress, and are not suited for training robots in the real world (or more ambitiously humans). In this paper, we introduce a method named Parameterized Environment Response Model (PERM) that shows promising results in training RL agents in parameterized environments. Inspired by Item Response Theory, PERM seeks to model difficulty of environments and ability of RL agents directly. Given that RL agents and humans are trained more efficiently under the ``zone of proximal development", our method generates a curriculum by matching the difficulty of an environment to the current ability of the student.  In addition, PERM can be trained offline and does not employ non-stationary measures of student ability, making it suitable for transfer between students. We demonstrate PERM's ability to represent the environment parameter space, and training with RL agents with PERM produces a strong performance in deterministic environments. Lastly, we show that our method is transferable between students, without any sacrifice in training quality.

----

## [543] JEPOO: Highly Accurate Joint Estimation of Pitch, Onset and Offset for Music Information Retrieval

**Authors**: *Haojie Wei, Jun Yuan, Rui Zhang, Yueguo Chen, Gang Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/544](https://doi.org/10.24963/ijcai.2023/544)

**Abstract**:

Melody extraction is a core task in music information retrieval, and the estimation of pitch, onset and offset are key sub-tasks in melody extraction. Existing methods have limited accuracy, and work for only one type of data, either single-pitch or multi-pitch. In this paper, we propose a highly accurate method for joint estimation of pitch, onset and offset, named JEPOO. We address the challenges of joint learning optimization and handling both single-pitch and multi-pitch data through novel model design and a new optimization technique named Pareto modulated loss with loss weight regularization. This is the first method that can accurately handle both single-pitch and multi-pitch music data, and even a mix of them. A comprehensive experimental study on a wide range of real datasets shows that JEPOO outperforms state-of-the-art methods by up to 10.6\%, 8.3\% and 10.3\% for the prediction of Pitch, Onset and Offset, respectively, and JEPOO is robust for various types of data and instruments. The ablation study validates the effectiveness of each component of JEPOO.

----

## [544] HireVAE: An Online and Adaptive Factor Model Based on Hierarchical and Regime-Switch VAE

**Authors**: *Zikai Wei, Anyi Rao, Bo Dai, Dahua Lin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/545](https://doi.org/10.24963/ijcai.2023/545)

**Abstract**:

Factor model is a fundamental investment tool in quantitative investment, which can be empowered by deep learning to become more flexible and efficient in practical complicated investing situations. However, it is still an open question to build a factor model that can conduct stock prediction in an online and adaptive setting, where the model can adapt itself to match the current market regime identified based on only point-in-time market information. To tackle this problem, we propose the first deep learning based online and adaptive factor model, HireVAE, at the core of which is a hierarchical latent space that embeds the underlying relationship between the market situation and stock-wise latent factors, so that HireVAE can effectively estimate useful latent factors given only historical market information and subsequently predict accurate stock returns. Across four commonly used real stock market benchmarks, the proposed HireVAE demonstrate superior performance in terms of active returns over previous methods, verifying the potential of such online and adaptive factor model.

----

## [545] A Diffusion Model with Contrastive Learning for ICU False Arrhythmia Alarm Reduction

**Authors**: *Feng Wu, Guoshuai Zhao, Xueming Qian, Li-Wei H. Lehman*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/546](https://doi.org/10.24963/ijcai.2023/546)

**Abstract**:

The high rate of false arrhythmia alarms in intensive care units (ICUs) can negatively impact patient care and lead to slow staff response time due to alarm fatigue. To reduce false alarms in ICUs, previous works proposed conventional supervised learning methods which have inherent limitations in dealing with high-dimensional, sparse, unbalanced, and limited data. We propose a deep generative approach based on the conditional denoising diffusion model to detect false arrhythmia alarms in the ICUs. Conditioning on past waveform data of a patient, our approach generates waveform predictions of the patient during an actual arrhythmia event, and uses the distance between the generated and the observed samples to classify the alarm. We design a network with residual links and self-attention mechanism to capture long-term dependencies in signal sequences, and leverage the contrastive learning mechanism to maximize distances between true and false arrhythmia alarms. We demonstrate the effectiveness of our approach on the MIMIC II arrhythmia dataset for detecting false alarms in both retrospective and real-time settings.

----

## [546] VecoCare: Visit Sequences-Clinical Notes Joint Learning for Diagnosis Prediction in Healthcare Data

**Authors**: *Yongxin Xu, Kai Yang, Chaohe Zhang, Peinie Zou, Zhiyuan Wang, Hongxin Ding, Junfeng Zhao, Yasha Wang, Bing Xie*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/547](https://doi.org/10.24963/ijcai.2023/547)

**Abstract**:

Due to the insufficiency of electronic health records (EHR) data utilized in practical diagnosis prediction scenarios, most works are devoted to learning powerful patient representations either from structured EHR data (e.g., temporal medical events, lab test results, etc.) or unstructured data (e.g., clinical notes, etc.). However, synthesizing rich information from both of them still needs to be explored. Firstly, the heterogeneous semantic biases across them heavily hinder the synthesis of representation spaces, which is critical for diagnosis prediction. Secondly, the intermingled quality of partial clinical notes leads to inadequate representations of to-be-predicted patients. Thirdly, typical attention mechanisms mainly focus on aggregating information from similar patients, ignoring important auxiliary information from others. To tackle these challenges, we propose a novel visit sequences-clinical notes joint learning approach, dubbed VecoCare. It performs a Gromov-Wasserstein Distance (GWD)-based contrastive learning task and an adaptive masked language model task in a sequential pre-training manner to reduce heterogeneous semantic biases. After pre-training, VecoCare further aggregates information from both similar and dissimilar patients through a dual-channel retrieval mechanism. We conduct diagnosis prediction experiments on two real-world datasets, which indicates that VecoCare outperforms state-of-the-art approaches. Moreover, the findings discovered by VecoCare are consistent with the medical researches.

----

## [547] Spotlight News Driven Quantitative Trading Based on Trajectory Optimization

**Authors**: *Mengyuan Yang, Mengying Zhu, Qianqiao Liang, Xiaolin Zheng, Menghan Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/548](https://doi.org/10.24963/ijcai.2023/548)

**Abstract**:

News-driven quantitative trading (NQT) has been popularly studied in recent years. Most existing NQT methods are performed in a two-step paradigm, i.e., first analyzing markets by a financial prediction task and then making trading decisions, which is doomed to failure due to the nearly futile financial prediction task. To bypass the financial prediction task, in this paper, we focus on reinforcement learning (RL) based NQT paradigm, which leverages news to make profitable trading decisions directly. In this paper, we propose a novel NQT framework SpotlightTrader based on decision trajectory optimization, which can effectively stitch together a continuous and flexible sequence of trading decisions to maximize profits. In addition, we enhance this framework by constructing a spotlight-driven state trajectory that obeys a stochastic process with irregular abrupt jumps caused by spotlight news. Furthermore, in order to adapt to non-stationary financial markets, we propose an effective training pipeline for this framework, which blends offline pretraining with online finetuning to balance exploration and exploitation effectively during online tradings. Extensive experiments on three real-world datasets demonstrate our proposed modelâ€™s superiority over the state-of-the-art NQT methods.

----

## [548] GPMO: Gradient Perturbation-Based Contrastive Learning for Molecule Optimization

**Authors**: *Xixi Yang, Li Fu, Yafeng Deng, Yuansheng Liu, Dongsheng Cao, Xiangxiang Zeng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/549](https://doi.org/10.24963/ijcai.2023/549)

**Abstract**:

Optimizing molecules with desired properties is a crucial step in de novo drug design. While translation-based methods have achieved initial success, they continue to face the challenge of the “exposure bias” problem. The challenge of preventing the “exposure bias” problem of molecule
optimization lies in the need for both positive and negative molecules of contrastive learning. That is because generating positive molecules through data augmentation requires domain-specific knowledge, and randomly sampled negative molecules are easily distinguished from the real molecules. Hence, in this work, we propose a molecule optimization method called GPMO, which leverages a gradient perturbation-based contrastive learning method to prevent the “exposure bias” problem in translation-based molecule optimization. With the assistance of positive and negative molecules, GPMO is able to effectively handle both real and artificial molecules. GPMO is a molecule optimization method that is conditioned on matched molecule pairs for drug discovery. Our empirical studies show that GPMO outperforms the state-of-the- art molecule optimization methods. Furthermore,  the negative and positive perturbations improve the robustness of GPMO.

----

## [549] InitLight: Initial Model Generation for Traffic Signal Control Using Adversarial Inverse Reinforcement Learning

**Authors**: *Yutong Ye, Yingbo Zhou, Jiepin Ding, Ting Wang, Mingsong Chen, Xiang Lian*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/550](https://doi.org/10.24963/ijcai.2023/550)

**Abstract**:

Due to repetitive trial-and-error style interactions between agents and a fixed traffic environment during the policy learning, existing Reinforcement Learning (RL)-based Traffic Signal Control (TSC) methods greatly suffer from long RL training time and poor adaptability of RL agents to other complex traffic environments. To address these problems, we propose a novel Adversarial Inverse Reinforcement Learning (AIRL)-based pre-training method named InitLight, which enables effective initial model generation for TSC agents. Unlike traditional RL-based TSC approaches that train a large number of agents simultaneously for a specific multi-intersection environment, InitLight pre-trains only one single initial model based on multiple single-intersection environments together with their expert trajectories. Since the reward function learned by InitLight can recover ground-truth TSC rewards for different intersections at optimality, the pre-trained agent can be deployed at intersections of any traffic environments as initial models to accelerate subsequent overall global RL training. Comprehensive experimental results show that, the initial model generated by InitLight can not only significantly accelerate the convergence with much fewer episodes, but also own superior generalization ability to accommodate various kinds of complex traffic environments.

----

## [550] Don't Ignore Alienation and Marginalization: Correlating Fraud Detection

**Authors**: *Yilong Zang, Ruimin Hu, Zheng Wang, Danni Xu, Jia Wu, Dengshi Li, Junhang Wu, Lingfei Ren*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/551](https://doi.org/10.24963/ijcai.2023/551)

**Abstract**:

The anonymity of online networks makes tackling fraud increasingly costly. Thanks to the superiority of graph representation learning, graph-based fraud detection has made significant progress in recent years. However, upgrading fraudulent strategies produces more advanced and difficult scams. One common strategy is synergistic camouflage —— combining multiple means to deceive others. Existing methods mostly investigate the differences between relations on individual frauds, that neglect the correlation among multi-relation fraudulent behaviors. In this paper, we design several statistics to validate the existence of synergistic camouflage of fraudsters by exploring the correlation among multi-relation interactions. From the perspective of multi-relation, we find two distinctive features of fraudulent behaviors, i.e., alienation and marginalization. Based on the finding, we propose COFRAUD, a correlation-aware fraud detection model, which innovatively incorporates synergistic camouflage into fraud detection. It captures the correlation among multi-relation fraudulent behaviors. Experimental results on two public datasets demonstrate that COFRAUD achieves significant improvements over state-of-the-art methods.

----

## [551] Realistic Cell Type Annotation and Discovery for Single-cell RNA-seq Data

**Authors**: *Yuyao Zhai, Liang Chen, Minghua Deng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/552](https://doi.org/10.24963/ijcai.2023/552)

**Abstract**:

The rapid development of single-cell RNA sequencing (scRNA-seq) technologies allows us to explore tissue heterogeneity at the cellular level. Cell type annotation plays an essential role in the substantial downstream analysis of scRNA-seq data. Existing methods usually classify the novel cell types in target data as an “unassigned” group and rarely discover the fine-grained cell type structure among them. Besides, these methods carry risks, such as susceptibility to batch effect between reference and target data, thus further compromising of inherent discrimination of target data. Considering these limitations, here we propose a new and practical task called realistic cell type annotation and discovery for scRNA-seq data. In this task, cells from seen cell types are given class labels, while cells from novel cell types are given cluster labels. To tackle this problem, we propose an end-to-end algorithm framework called scPOT from the perspective of optimal transport (OT). Specifically,  we first design an OT-based prototypical representation learning paradigm to encourage both global discriminations of clusters and local consistency of cells to uncover the intrinsic structure of target data. Then we propose an unbalanced OT-based partial alignment strategy with statistical filling to detect the cells from the seen cell types across reference and target data. Notably, scPOT also introduces an easy yet effective solution to automatically estimate the overall cell type number in target data. Extensive results on our carefully designed evaluation benchmarks demonstrate the superiority of scPOT over various state-of-the-art clustering and annotation methods.

----

## [552] Towards Generalizable Reinforcement Learning for Trade Execution

**Authors**: *Chuheng Zhang, Yitong Duan, Xiaoyu Chen, Jianyu Chen, Jian Li, Li Zhao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/553](https://doi.org/10.24963/ijcai.2023/553)

**Abstract**:

Optimized trade execution is to sell (or buy) a given amount of assets in a given time with the lowest possible trading cost. Recently, reinforcement learning (RL) has been applied to optimized trade execution to learn smarter policies from market data. However, we find that many existing RL methods exhibit considerable overfitting which prevents them from real deployment. In this paper, we provide an extensive study on the overfitting problem in optimized trade execution. First, we model the optimized trade execution as offline RL with dynamic context (ORDC), where the context represents market variables that cannot be influenced by the trading policy and are collected in an offline manner.  Under this framework, we derive the generalization bound and find that the overfitting issue is caused by large context space and limited context samples in the offline setting. Accordingly, we propose to learn compact representations for context to address the overfitting problem, either by leveraging prior knowledge or in an end-to-end manner. To evaluate our algorithms, we also implement a carefully designed simulator based on historical limit order book (LOB) data to provide a high-fidelity benchmark for different algorithms. Our experiments on the high-fidelity simulator demonstrate that our algorithms can effectively alleviate overfitting and achieve better performance.

----

## [553] SemiGNN-PPI: Self-Ensembling Multi-Graph Neural Network for Efficient and Generalizable Protein-Protein Interaction Prediction

**Authors**: *Ziyuan Zhao, Peisheng Qian, Xulei Yang, Zeng Zeng, Cuntai Guan, Tam Wai Leong, Xiaoli Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/554](https://doi.org/10.24963/ijcai.2023/554)

**Abstract**:

Protein-protein interactions (PPIs) are crucial in various biological processes and their study has significant implications for drug development and disease diagnosis. Existing deep learning methods suffer from significant performance degradation under complex real-world scenarios due to various factors, e.g., label scarcity and domain shift. In this paper, we propose a self-ensembling multi-graph neural network (SemiGNN-PPI) that can effectively predict PPIs while being both efficient and generalizable. In SemiGNN-PPI, we not only model the protein correlations but explore the label dependencies by constructing and processing multiple graphs from the perspectives of both features and labels in the graph learning process. We further marry GNN with Mean Teacher to effectively leverage unlabeled graph-structured PPI data for self-ensemble graph learning. We also design multiple graph consistency constraints to align the student and teacher graphs in the feature embedding space, enabling the student model to better learn from the teacher model by incorporating more relationships. Extensive experiments on PPI datasets of different scales with different evaluation settings demonstrate that SemiGNN-PPI outperforms state-of-the-art PPI prediction methods, particularly in challenging scenarios such as training with limited annotations and testing on unseen data.

----

## [554] Deep Hashing-based Dynamic Stock Correlation Estimation via Normalizing Flow

**Authors**: *Xiaolin Zheng, Mengpu Liu, Mengying Zhu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/555](https://doi.org/10.24963/ijcai.2023/555)

**Abstract**:

In financial scenarios, influenced by common factors such as global macroeconomic and sector-specific factors, stocks exhibit varying degrees of correlations with each other, which is essential in risk-averse portfolio allocation. Because the real risk matrix is unobservable, the covariance-based correlation matrix is widely used for constructing diversified stock portfolios. However, studies have seldom focused on dynamic correlation matrix estimation under the non-stationary financial market. Moreover, as the number of stocks in the market grows, existing correlation matrix estimation methods face more serious challenges with regard to efficiency and effectiveness. In this paper, we propose a novel hash-based dynamic correlation forecasting model (HDCF) to estimate dynamic stock correlations. Under structural assumptions on the correlation matrix, HDCF learns the hash representation based on normalizing flows instead of the real-valued representation, which performs extremely efficiently in high-dimensional settings. Experiments show that our proposed model outperforms baselines on portfolio decisions in terms of effectiveness and efficiency.

----

## [555] MolHF: A Hierarchical Normalizing Flow for Molecular Graph Generation

**Authors**: *Yiheng Zhu, Zhenqiu Ouyang, Ben Liao, Jialu Wu, Yixuan Wu, Chang-Yu Hsieh, Tingjun Hou, Jian Wu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/556](https://doi.org/10.24963/ijcai.2023/556)

**Abstract**:

Molecular de novo design is a critical yet challenging task in scientific fields, aiming to design novel molecular structures with desired property profiles. Significant progress has been made by resorting to generative models for graphs. However, limited attention is paid to hierarchical generative models, which can exploit the inherent hierarchical structure (with rich semantic information) of the molecular graphs and generate complex molecules of larger size that we shall demonstrate to be difficult for most existing models. The primary challenge to hierarchical generation is the non-differentiable issue caused by the generation of intermediate discrete coarsened graph structures. To sidestep this issue, we cast the tricky hierarchical generation problem over discrete spaces as the reverse process of hierarchical representation learning and propose MolHF, a new hierarchical flow-based model that generates molecular graphs in a coarse-to-fine manner. Specifically, MolHF first generates bonds through a multi-scale architecture, then generates atoms based on the coarsened graph structure at each scale. We demonstrate that MolHF achieves state-of-the-art performance in random generation and property optimization, implying its high capacity to model data distribution. Furthermore, MolHF is the first flow-based model that can be applied to model larger molecules (polymer) with more than 100 heavy atoms. The code and models are available at https://github.com/violet-sto/MolHF.

----

## [556] Keep Skills in Mind: Understanding and Implementing Skills in Commonsense Question Answering

**Authors**: *Meikai Bao, Qi Liu, Kai Zhang, Ye Liu, Linan Yue, Longfei Li, Jun Zhou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/557](https://doi.org/10.24963/ijcai.2023/557)

**Abstract**:

Commonsense Question Answering (CQA) aims to answer questions that require human commonsense. Closed-book CQA, as one of the subtasks, requires the model to answer questions without retrieving external knowledge, which emphasizes the importance of the model's problem-solving ability. Most previous methods relied on large-scale pre-trained models to generate question-related knowledge while ignoring the crucial role of skills in the process of answering commonsense questions. Generally, skills refer to the learned ability in performing a specific task or activity, which are derived from knowledge and experience. In this paper, we introduce a new approach named Dynamic Skill-aware Commonsense Question Answering (DSCQA), which transcends the limitations of traditional methods by informing the model about the need for each skill in questions and utilizes skills as a critical driver in CQA process. To be specific, DSCQA first employs commonsense skill extraction module to generate various skill representations. Then, DSCQA utilizes dynamic skill module to generate dynamic skill representations. Finally, in perception and emphasis module, various skills and dynamic skill representations are used to help question-answering process. Experimental results on two publicly available CQA datasets show the effectiveness of our proposed model and the considerable impact of introducing skills.

----

## [557] An Effective and Efficient Time-aware Entity Alignment Framework via Two-aspect Three-view Label Propagation

**Authors**: *Li Cai, Xin Mao, Youshao Xiao, Changxu Wu, Man Lan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/558](https://doi.org/10.24963/ijcai.2023/558)

**Abstract**:

Entity alignment (EA) aims to find the equivalent entity pairs between different knowledge graphs (KGs), which is crucial to promote knowledge fusion. With the wide use of temporal knowledge graphs (TKGs), time-aware EA (TEA) methods appear to enhance EA. Existing TEA models are based on Graph Neural Networks (GNN) and achieve state-of-the-art (SOTA) performance, but it is difficult to transfer them to large-scale TKGs due to the scalability issue of GNN. In this paper, we propose an effective and efficient non-neural EA framework between TKGs, namely LightTEA, which consists of four essential components: (1) Two-aspect Three-view Label Propagation, (2) Sparse Similarity with Temporal Constraints, (3) Sinkhorn Operator, and (4) Temporal Iterative Learning. All of these modules work together to improve the performance of EA while reducing the time consumption of the model. Extensive experiments on public datasets indicate that our proposed model significantly outperforms the SOTA methods for EA between TKGs, and the time consumed by LightTEA is only dozens of seconds at most, no more than 10% of the most efficient TEA method.

----

## [558] One Model for All Domains: Collaborative Domain-Prefix Tuning for Cross-Domain NER

**Authors**: *Xiang Chen, Lei Li, Shuofei Qiao, Ningyu Zhang, Chuanqi Tan, Yong Jiang, Fei Huang, Huajun Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/559](https://doi.org/10.24963/ijcai.2023/559)

**Abstract**:

Cross-domain NER is a challenging task to address the low-resource problem in practical scenarios. Previous typical solutions mainly obtain a NER model by pre-trained language models (PLMs) with data from a rich-resource domain and adapt it to the target domain. Owing to the mismatch issue among entity types in different domains, previous approaches normally tune all parameters of PLMs, ending up with an entirely new NER model for each domain. Moreover, current models only focus on leveraging knowledge in one general source domain while failing to successfully transfer knowledge from multiple sources to the target. To address these issues, we introduce Collaborative Domain-Prefix Tuning for cross-domain NER (CP-NER) based on text-to-text generative PLMs. Specifically, we present text-to-text generation grounding domain-related instructors to transfer knowledge to new domain NER tasks without structural modifications. We utilize frozen PLMs and conduct collaborative domain-prefix tuning to stimulate the potential of PLMs to handle NER tasks across various domains. Experimental results on the Cross-NER benchmark show that the proposed approach has flexible transfer ability and performs better on both one-source and multiple-source cross-domain NER tasks.

----

## [559] Less Learn Shortcut: Analyzing and Mitigating Learning of Spurious Feature-Label Correlation

**Authors**: *Yanrui Du, Jing Yan, Yan Chen, Jing Liu, Sendong Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang, Bing Qin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/560](https://doi.org/10.24963/ijcai.2023/560)

**Abstract**:

Recent research has revealed that deep neural networks often take dataset biases as a shortcut to make decisions rather than understand tasks, leading to failures in real-world applications. In this study, we focus on the spurious correlation between word features and labels that models learn from the biased data distribution of training data. In particular, we define the word highly co-occurring with a specific label as biased word, and the example containing biased word as biased example. Our analysis shows that biased examples are easier for models to learn, while at the time of prediction, biased words make a significantly higher contribution to the models' predictions, and models tend to assign predicted labels over-relying on the spurious correlation between words and labels. To mitigate models' over-reliance on the shortcut (i.e. spurious correlation), we propose a training strategy Less-Learn-Shortcut (LLS): our strategy quantifies the biased degree of the biased examples and down-weights them accordingly. Experimental results on Question Matching, Natural Language Inference and Sentiment Analysis tasks show that LLS is a task-agnostic strategy and can improve the model performance on adversarial data while maintaining good performance on in-domain data.

----

## [560] KEST: Kernel Distance Based Efficient Self-Training for Improving Controllable Text Generation

**Authors**: *Yuxi Feng, Xiaoyuan Yi, Laks V. S. Lakshmanan, Xing Xie*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/561](https://doi.org/10.24963/ijcai.2023/561)

**Abstract**:

Self-training (ST) has come to fruition in language understanding tasks by producing pseudo labels, which reduces the labeling bottleneck of language model fine-tuning. Nevertheless, in facilitating semi-supervised controllable language generation, ST faces two key challenges. First, augmented by self-generated pseudo text, generation models tend to over-exploit the previously learned text distribution, suffering from mode collapse and poor generation diversity. Second, generating pseudo text in each iteration is time-consuming, severely decelerating the training process. In this work, we propose KEST, a novel and efficient self-training framework to handle these problems. KEST utilizes a kernel-based loss, rather than standard cross entropy, to learn from the soft pseudo text produced by a shared non-autoregressive generator. We demonstrate both theoretically and empirically that KEST can benefit from more diverse pseudo text in an efficient manner, which allows not only refining and exploiting the previously fitted distribution but also enhanced exploration towards a larger potential text space, providing a guarantee of improved performance. Experiments on three controllable generation tasks demonstrate that KEST significantly improves control accuracy while maintaining comparable text fluency and generation diversity against several strong baselines.

----

## [561] Regularisation for Efficient Softmax Parameter Generation in Low-Resource Text Classifiers

**Authors**: *Daniel Grießhaber, Johannes Maucher, Ngoc Thang Vu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/562](https://doi.org/10.24963/ijcai.2023/562)

**Abstract**:

Meta-learning has made tremendous progress in recent years and was demonstrated to be particularly suitable in low-resource settings where training data is very limited. However, meta-learning models still require large amounts of training tasks to achieve good generalisation. Since labelled training data may be sparse, self-supervision-based approaches are able to further improve performance on downstream tasks. Although no labelled data is necessary for this training, a large corpus of unlabelled text needs to be available.  
In this paper, we improve on recent advances in meta-learning for natural language models that allow training on a diverse set of training tasks for few-shot, low-resource target tasks. We introduce a way to generate new training data with the need for neither more supervised nor unsupervised datasets. We evaluate the method on a diverse set of NLP tasks and show that the model decreases in performance when trained on this data without further adjustments. Therefore, we introduce and evaluate two methods for regularising the training process and show that they not only improve performance when used in conjunction with the new training data but also improve average performance when training only on the original data, compared to the baseline.

----

## [562] SmartBERT: A Promotion of Dynamic Early Exiting Mechanism for Accelerating BERT Inference

**Authors**: *Boren Hu, Yun Zhu, Jiacheng Li, Siliang Tang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/563](https://doi.org/10.24963/ijcai.2023/563)

**Abstract**:

Dynamic early exiting has been proven to improve the inference speed of the pre-trained language model like BERT. However, all samples must go through all consecutive layers before early exiting and more complex samples usually go through more layers, which still exists redundant computation. In this paper, we propose a novel dynamic early exiting combined with layer skipping for BERT inference named SmartBERT, which adds a skipping gate and an exiting operator into each layer of BERT. SmartBERT can adaptively skip some layers and adaptively choose whether to exit. Besides, we propose cross-layer contrastive learning and combine it into our training phases to boost the intermediate layers and classifiers which would be beneficial for early exiting. To keep the inconsistent usage of skipping gates between training and inference phases, we propose a hard weight mechanism during training phase. We conduct experiments on eight classification datasets of the GLUE benchmark. Experimental results show that SmartBERT achieves 2-3Ã— computation reduction with minimal accuracy drops compared with BERT and our method outperforms previous methods in both efficiency and accuracy. Moreover, in some complex datasets, we prove that the early exiting based on entropy hardly works, and the skipping mechanism is essential for reducing computation.

----

## [563] Cross-Modal Global Interaction and Local Alignment for Audio-Visual Speech Recognition

**Authors**: *Yuchen Hu, Ruizhe Li, Chen Chen, Heqing Zou, Qiushi Zhu, Eng Siong Chng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/564](https://doi.org/10.24963/ijcai.2023/564)

**Abstract**:

Audio-visual speech recognition (AVSR) research has gained a great success recently by improving the noise-robustness of audio-only automatic speech recognition (ASR) with noise-invariant visual information. However, most existing AVSR approaches simply fuse the audio and visual features by concatenation, without explicit interactions to capture the deep correlations between them, which results in sub-optimal multimodal representations for downstream speech recognition task. In this paper, we propose a cross-modal global interaction and local alignment (GILA) approach for AVSR, which captures the deep audio-visual (A-V) correlations from both global and local perspectives. Specifically, we design a global interaction model to capture the A-V complementary relationship on modality level, as well as a local alignment approach to model the A-V temporal consistency on frame level. Such a holistic view of cross-modal correlations enable better multimodal representations for AVSR. Experiments on public benchmarks LRS3 and LRS2 show that our GILA outperforms the supervised learning state-of-the-art. Code is at https://github.com/YUCHEN005/GILA.

----

## [564] Explainable Text Classification via Attentive and Targeted Mixing Data Augmentation

**Authors**: *Songhao Jiang, Yan Chu, Zhengkui Wang, Tianxing Ma, Hanlin Wang, Wenxuan Lu, Tianning Zang, Bo Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/565](https://doi.org/10.24963/ijcai.2023/565)

**Abstract**:

Mixing data augmentation methods have been widely used in text classification recently. However, existing methods do not control the quality of augmented data and have low model explainability. To tackle these issues, this paper proposes an explainable text classification solution based on attentive and targeted mixing data augmentation, ATMIX. Instead of selecting data for augmentation without control, ATMIX focuses on the misclassified training samples as the target for augmentation to better improve the model's capability. Meanwhile, to generate meaningful augmented samples, it adopts a self-attention mechanism to understand the importance of the subsentences in a text, and cut and mix the subsentences between the misclassified and correctly classified samples wisely. Furthermore, it employs a novel dynamic augmented data selection framework based on the loss function gradient to dynamically optimize the augmented samples for model training. In the end, we develop a new model explainability evaluation method based on subsentence attention and conduct extensive evaluations over multiple real-world text datasets. The results indicate that ATMIX is more effective with higher explainability than the typical classification models, hidden-level, and input-level mixup models.

----

## [565] ScriptWorld: Text Based Environment for Learning Procedural Knowledge

**Authors**: *Abhinav Joshi, Areeb Ahmad, Umang Pandey, Ashutosh Modi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/566](https://doi.org/10.24963/ijcai.2023/566)

**Abstract**:

Text-based games provide a framework for developing natural language understanding and commonsense knowledge about the world in reinforcement learning based agents. Existing text-based environments often rely on fictional situations and characters to create a gaming framework and are far from real-world scenarios. In this paper, we introduce ScriptWorld: a text-based environment for teaching agents about real-world daily chores and hence imparting commonsense knowledge. To the best of our knowledge, it is the first interactive text-based gaming framework that consists of daily real-world human activities designed using scripts dataset. We provide gaming environments for 10 daily activities and perform a detailed analysis of the proposed environment. We develop RL-based baseline models/agents to play the games in ScriptWorld. To understand the role of language models in such environments, we leverage features obtained from pre-trained language models in the RL agents. Our experiments show that prior knowledge obtained from a pre-trained language model helps to solve real-world text-based gaming environments.

----

## [566] Towards Incremental NER Data Augmentation via Syntactic-aware Insertion Transformer

**Authors**: *Wenjun Ke, Zongkai Tian, Qi Liu, Peng Wang, Jinhua Gao, Rui Qi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/567](https://doi.org/10.24963/ijcai.2023/567)

**Abstract**:

Named entity recognition (NER) aims to locate and classify named entities in natural language texts. Most existing high-performance NER models employ a supervised paradigm, which requires a large quantity of high-quality annotated data during training. In order to help NER models perform well in few-shot scenarios, data augmentation approaches attempt to build extra data by means of random editing or by using end-to-end generation with PLMs. 
However, these methods focus on only the fluency of generated sentences, ignoring the syntactic correlation between the new and raw sentences. Such uncorrelation also brings low diversity and inconsistent labeling of synthetic samples. To fill this gap, we present SAINT (Syntactic-Aware InsertioN Transformer), a hard-constraint controlled text generation model that incorporates syntactic information. The proposed method operates by inserting new tokens between existing entities in a parallel manner. During insertion procedure, new tokens will be added taking both semantic and syntactic factors into account. Hence the resulting sentence can retain the syntactic correctness with respect to the raw data. Experimental results on two benchmark datasets, i.e., Ontonotes and Wikiann, demonstrate the comparable performance of SAINT over the state-of-the-art baselines.

----

## [567] Towards Lossless Head Pruning through Automatic Peer Distillation for Language Models

**Authors**: *Bingbing Li, Zigeng Wang, Shaoyi Huang, Mikhail A. Bragin, Ji Li, Caiwen Ding*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/568](https://doi.org/10.24963/ijcai.2023/568)

**Abstract**:

Pruning has been extensively studied in Transformer-based language models to improve efficiency. Typically, we zero (prune) unimportant model weights and train a derived compact model to improve final accuracy. For pruned weights, we treat them as useless and discard them. This usually leads to significant model accuracy degradation. In this paper, we focus on attention head pruning as head attention is a key component of the transformer-based language models and provides interpretable knowledge meaning. We reveal the relationship between pruned attention heads and retained heads and provide a solution to recycle the discarded knowledge from the pruned heads, named peer distillation. We also develop an automatic framework to locate the to-be-pruned attention heads in each layer, freeing the time-consuming human labor in tuning hyperparameters.Experimental results on the General Language Understanding Evaluation  (GLUE)  benchmark are provided using BERT model. By recycling discarded knowledge from pruned heads, the proposed method maintains model performance across all nine tasks while reducing heads by over 58% on average and outperforms state-of-the-art techniques (e.g., Random, HISP, L0 Norm, SMP).

----

## [568] Annealing Genetic-based Preposition Substitution for Text Rubbish Example Generation

**Authors**: *Chen Li, Xinghao Yang, Baodi Liu, Weifeng Liu, Honglong Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/569](https://doi.org/10.24963/ijcai.2023/569)

**Abstract**:

Modern Natural Language Processing (NLP) models expose under-sensitivity towards text rubbish examples. The text rubbish example is the heavily modified input text which is nonsensical to humans but does not change the model’s prediction. Prior work crafts rubbish examples by iteratively deleting words and determining the deletion order with beam search. However, the produced rubbish examples usually cause a reduction in model confidence and sometimes deliver human-readable text. To address these problems, we propose an Annealing Genetic based Preposition Substitution (AGPS) algorithm for text rubbish sample generation with two major merits. Firstly, the AGPS crafts rubbish text examples by substituting input words with meaningless prepositions instead of directly removing them, which brings less degradation to the model’s confidence. Secondly, we design an Annealing Genetic algorithm to optimize the word replacement priority, which allows the Genetic Algorithm (GA) to jump out the local optima with probabilities. This is significant in achieving better objectives, i.e., a high word modification rate and a high model confidence. Experimental results on five popular datasets manifest the superiority of AGPS compared with the baseline and expose the fact: the NLP models can not really understand the semantics of sentences, as they give the same prediction with even higher confidence for the nonsensical preposition sequences.

----

## [569] iRe2f: Rethinking Effective Refinement in Language Structure Prediction via Efficient Iterative Retrospecting and Reasoning

**Authors**: *Zuchao Li, Xingyi Guo, Letian Peng, Lefei Zhang, Hai Zhao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/570](https://doi.org/10.24963/ijcai.2023/570)

**Abstract**:

Refinement plays a critical role in language structure prediction, a process that deals with complex situations such as structural edge interdependencies. Since language structure prediction usually modeled as graph parsing, typical refinement methods involve taking an initial parsing graph as input and refining it using language input and other relevant information. Intuitively, a refinement component, i.e., refiner, should be lightweight and efficient, as it is only responsible for correcting faults in the initial graph. However, current refiners add a significant burden to the parsing process due to their reliance on time-consuming encoding-decoding procedure on the language input and graph. To make the refiner more practical for real-world applications, this paper proposes a lightweight but effective iterative refinement framework, iRe^2f, based on iterative retrospecting and reasoning without involving the re-encoding process on the graph. iRe^2f iteratively refine the parsing graph based on interaction between graph and sequence and efficiently learns the shortcut to update the sequence and graph representations in each iteration.  The shortcut is calculated based on the graph representation in the latest iteration.  iRe^2f reduces the number of refinement parameters by 90% compared to the previous smallest refiner. Experiments on a variety of language structure prediction tasks show that iRe^2f performs comparably or better than current state-of-the-art refiners, with a significant increase in efficiency.

----

## [570] Local and Global: Temporal Question Answering via Information Fusion

**Authors**: *Yonghao Liu, Di Liang, Mengyu Li, Fausto Giunchiglia, Ximing Li, Sirui Wang, Wei Wu, Lan Huang, Xiaoyue Feng, Renchu Guan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/571](https://doi.org/10.24963/ijcai.2023/571)

**Abstract**:

Many models that leverage knowledge graphs (KGs) have recently demonstrated remarkable success in question answering (QA) tasks. In the real world, many facts contained in KGs are time-constrained thus temporal KGQA has received increasing attention. Despite the fruitful efforts of previous models in temporal KGQA, they still have several limitations. (I) They neither emphasize the graph structural information between entities in KGs nor explicitly utilize a multi-hop relation path through graph neural networks to enhance answer prediction. (II) They adopt pre-trained language models (LMs) to obtain question representations, focusing merely on the global information related to the question while not highlighting the local information of the entities in KGs. To address these limitations, we introduce a novel model that simultaneously explores both Local information and Global information for the task of temporal KGQA (LGQA). Specifically, we first introduce an auxiliary task in the temporal KG embedding procedure to make timestamp embeddings time-order aware. Then, we design information fusion layers that effectively incorporate local and global information to deepen question understanding. We conduct extensive experiments on two benchmarks, and LGQA significantly outperforms previous state-of-the-art models, especially in difficult questions. Moreover, LGQA can generate interpretable and trustworthy predictions.

----

## [571] PPAT: Progressive Graph Pairwise Attention Network for Event Causality Identification

**Authors**: *Zhenyu Liu, Baotian Hu, Zhenran Xu, Min Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/572](https://doi.org/10.24963/ijcai.2023/572)

**Abstract**:

Event Causality Identification (ECI) aims to identify the causality between a pair of event mentions in a document, which is composed of sentence-level ECI (SECI) and document-level ECI (DECI). Previous work applies various reasoning models to identify the implicit event causality. However, they indiscriminately reason all event causality in the same way, ignoring that most inter-sentence event causality depends on intra-sentence event causality to infer.  In this paper, we propose a progressive graph pairwise attention network (PPAT) to consider the above dependence. PPAT applies a progressive reasoning strategy, as it first predicts the intra-sentence event causality, and then infers the more implicit inter-sentence event causality based on the SECI result. We construct a sentence boundary event relational graph, and PPAT leverages a simple pairwise attention mechanism, which attends to different reasoning chains on the graph. In addition, we propose a causality-guided training strategy for assisting PPAT in learning causality-related representations on every layer. Extensive experiments show that our model achieves state-of-the-art performance on three benchmark datasets (5.5%, 2.2% and 4.5% F1 gains on EventStoryLine, MAVEN-ERE and Causal-TimeBank). Code is available at https://github.com/HITsz-TMG/PPAT.

----

## [572] Meta-Tsallis-Entropy Minimization: A New Self-Training Approach for Domain Adaptation on Text Classification

**Authors**: *Menglong Lu, Zhen Huang, Zhiliang Tian, Yunxiang Zhao, Xuanyu Fei, Dongsheng Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/573](https://doi.org/10.24963/ijcai.2023/573)

**Abstract**:

Text classification is a fundamental task for natural language processing, and adapting text classification models across domains has broad applications. 
Self-training generates pseudo-examples from the model's predictions and iteratively trains on the pseudo-examples, i.e., minimizes the loss on the source domain and the Gibbs entropy on the target domain. However, Gibbs entropy is sensitive to prediction errors, and thus, self-training tends to fail when the domain shift is large. In this paper, we propose Meta-Tsallis Entropy minimization (MTEM). MTEM uses an instance adaptive Tsallis entropy to replace the Gibbs entropy and a meta-learning algorithm to optimize the instance adaptive Tsallis entropy on the target domain. To reduce the computation cost of MTEM, we propose an approximation technique to approximate the second-order derivation involved in the meta-learning. To efficiently generate pseudo labels, we propose an annealing sampling mechanism for exploring the model's prediction probability. Theoretically, we prove the convergence of the meta-learning algorithm in MTEM and analyze the effectiveness of MTEM in achieving domain adaptation. Experimentally, MTEM improves the adaptation performance of BERT with an average of 4 percent on the benchmark dataset.

----

## [573] ODEE: A One-Stage Object Detection Framework for Overlapping and Nested Event Extraction

**Authors**: *Jinzhong Ning, Zhihao Yang, Zhizheng Wang, Yuanyuan Sun, Hongfei Lin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/574](https://doi.org/10.24963/ijcai.2023/574)

**Abstract**:

The task of extracting overlapping and nested events has received significant attention in recent times, as prior research has primarily focused on extracting flat events, overlooking the intricacies of overlapping and nested occurrences. In this work, we present a new approach to Event Extraction (EE) by reformulating it as an object detection task on a table of token pairs. Our proposed one-stage event extractor, called ODEE, can handle overlapping and nested events. The model is designed with a vertex-based tagging scheme and two auxiliary tasks of predicting the spans and types of event trigger words and argument entities, leveraging the full span information of event elements. Furthermore, in the training stage, we introduce a negative sampling method for table cells to address the imbalance problem of positive and negative table cell tags, meanwhile improving computational efficiency. Empirical evaluations demonstrate that ODEE achieves the state-of-the-art performance on three benchmarks for overlapping and nested EE (i.e., FewFC, Genia11, and Genia13). Furthermore, ODEE outperforms current state-of-the-art methods in terms of both number of parameters and inference speed, indicating its high computational efficiency. To facilitate future research in this area, the codes are publicly available at https://github.com/NingJinzhong/ODEE.

----

## [574] Learning to Speak from Text: Zero-Shot Multilingual Text-to-Speech with Unsupervised Text Pretraining

**Authors**: *Takaaki Saeki, Soumi Maiti, Xinjian Li, Shinji Watanabe, Shinnosuke Takamichi, Hiroshi Saruwatari*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/575](https://doi.org/10.24963/ijcai.2023/575)

**Abstract**:

While neural text-to-speech (TTS) has achieved human-like natural synthetic speech, multilingual TTS systems are limited to resource-rich languages due to the need for paired text and studio-quality audio data. This paper proposes a method for zero-shot multilingual TTS using text-only data for the target language. The use of text-only data allows the development of TTS systems for low-resource languages for which only textual resources are available, making TTS accessible to thousands of languages. Inspired by the strong cross-lingual transferability of multilingual language models, our framework first performs masked language model pretraining with multilingual text-only data. Then we train this model with a paired data in a supervised manner, while freezing a language-aware embedding layer. This allows inference even for languages not included in the paired data but present in the text-only data. Evaluation results demonstrate highly intelligible zero-shot TTS with a character error rate of less than 12% for an unseen language.

----

## [575] Case-Based Reasoning with Language Models for Classification of Logical Fallacies

**Authors**: *Zhivar Sourati, Filip Ilievski, Hông-Ân Sandlin, Alain Mermoud*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/576](https://doi.org/10.24963/ijcai.2023/576)

**Abstract**:

The ease and speed of spreading misinformation and propaganda on the Web motivate the need to develop trustworthy technology for detecting fallacies in natural language arguments. However, state-of-the-art language modeling methods exhibit a lack of robustness on tasks like logical fallacy classification that require complex reasoning. In this paper, we propose a Case-Based Reasoning method that classifies new cases of logical fallacy by language-modeling-driven retrieval and adaptation of historical cases. We design four complementary strategies to enrich input representation for our model, based on external information about goals, explanations, counterarguments, and argument structure. Our experiments in in-domain and out-of-domain settings indicate that Case-Based Reasoning improves the accuracy and generalizability of language models. Our ablation studies suggest that representations of similar cases have a strong impact on the model performance, that models perform well with fewer retrieved cases, and that the size of the case database has a negligible effect on the performance. Finally, we dive deeper into the relationship between the properties of the retrieved cases and the model performance.

----

## [576] Fine-tuned vs. Prompt-tuned Supervised Representations: Which Better Account for Brain Language Representations?

**Authors**: *Jingyuan Sun, Marie-Francine Moens*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/577](https://doi.org/10.24963/ijcai.2023/577)

**Abstract**:

To decipher the algorithm underlying the human brain's language representation, previous work probed brain responses to language input with pre-trained artificial neural network (ANN) models fine-tuned on NLU tasks.  However, full fine-tuning generally updates the entire parametric space and distorts pre-trained features, cognitively inconsistent with the brain's robust multi-task learning ability. Prompt-tuning, in contrast, protects pre-trained weights and learns task-specific embeddings to fit a task. Could prompt-tuning generate representations that better account for the brain's language representations than fine-tuning? If so, what kind of NLU task leads a pre-trained model to better decode the information represented in the human brain? We investigate these questions by comparing prompt-tuned and fine-tuned representations in neural decoding, that is predicting the linguistic stimulus from the brain activities evoked by the stimulus. We find that on none of the 10 NLU tasks, full fine-tuning significantly outperforms prompt-tuning in neural decoding, implicating that a more brain-consistent tuning method yields representations that better correlate with brain data. Moreover, we identify that tasks dealing with fine-grained concept meaning yield representations that better decode brain activation patterns than other tasks, especially the syntactic chunking task. This indicates that our brain encodes more fine-grained concept information than shallow syntactic information when representing languages.

----

## [577] SQuAD-SRC: A Dataset for Multi-Accent Spoken Reading Comprehension

**Authors**: *Yixuan Tang, Anthony K. H. Tung*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/578](https://doi.org/10.24963/ijcai.2023/578)

**Abstract**:

Spoken Reading Comprehension (SRC) is a challenging problem in spoken natural language retrieval, which automatically extracts the answer from the text-form contents according to the audio-form question. However, the existing spoken question answering approaches are mainly based on synthetically generated audio-form data, which may be ineffectively applied for multi-accent spoken question answering directly in many real-world applications. In this paper, we construct a large-scale multi-accent human spoken dataset SQuAD-SRC, in order to study the problem of multi-accent spoken reading comprehension. We choose 24 native English speakers from six different countries with various English accents and construct audio-form questions to the correspondent text-form contents by the chosen speakers. The dataset consists of 98,169 spoken question answering pairs and 20,963 passages from the popular machine reading comprehension dataset SQuAD. We present a statistical analysis of our SQuAD-SRC dataset and conduct extensive experiments on it by comparing cascaded SRC approaches and the enhanced end-to-end ones. Moreover, we explore various adaption strategies to improve the SRC performance, especially for multi-accent spoken questions.

----

## [578] PasCore: A Chinese Overlapping Relation Extraction Model Based on Global Pointer Annotation Strategy

**Authors**: *Peng Wang, Jiafeng Xie, Xiye Chen, Guozheng Li, Wei Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/579](https://doi.org/10.24963/ijcai.2023/579)

**Abstract**:

Recent work for extracting relations from texts has achieved excellent performance. However, existing studies mainly focus on simple relation extraction, these methods perform not well on overlapping triple problem because the tags of shared entities would conflict with each other.  Especially, overlapping entities are common and indispensable in Chinese. To address this issue, this paper proposes PasCore, which utilizes a global pointer annotation strategy for overlapping relation extraction in Chinese. PasCore first obtains the sentence vector via general pre-training model encoder, and uses classifier to predicate relations. Subsequently, it uses global pointer annotation strategy for head entity annotation, which uses global tags to label the start and end positions of the entities. Finally, PasCore integrates the relation, head entity and its type to mark the tail entity. Furthermore, PasCore performs conditional layer normalization to fuse features, which connects all stages and greatly enriches the association between relations and entities. Experimental results on both Chinese and English real-world datasets demonstrate that PasCore outperforms strong baselines on relation extraction and, especially, shows superior performance on overlapping relation extraction.

----

## [579] Privacy-Preserving End-to-End Spoken Language Understanding

**Authors**: *Yinggui Wang, Wei Huang, Le Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/580](https://doi.org/10.24963/ijcai.2023/580)

**Abstract**:

Spoken language understanding (SLU), one of the key enabling technologies for human-computer interaction in IoT devices, provides an easy-to-use user interface. Human speech can contain a lot of user-sensitive information, such as gender, identity, and sensitive content. New types of security and privacy breaches have thus emerged. Users do not want to expose their personal sensitive information to malicious attacks by untrusted third parties. Thus, the SLU system needs to ensure that a potential malicious attacker cannot deduce the sensitive attributes of the users, while it should avoid greatly compromising the SLU accuracy. To address the above challenge, this paper proposes a novel SLU multi-task privacy-preserving model to prevent both the speech recognition (ASR) and identity recognition (IR) attacks. The model uses the hidden layer separation technique so that SLU information is distributed only in a specific portion of the hidden layer, and the other two types of information are removed to obtain a privacy-secure hidden layer. In order to achieve good balance between efficiency and privacy, we introduce a new mechanism of model pre-training, namely joint adversarial training, to further enhance the user privacy. Experiments over two SLU datasets show that the proposed method can reduce the accuracy of both the ASR and IR attacks close to that of a random guess, while leaving the SLU performance largely unaffected.

----

## [580] Beyond Pure Text: Summarizing Financial Reports Based on Both Textual and Tabular Data

**Authors**: *Ziao Wang, Zelin Jiang, Xiaofeng Zhang, Jaehyeon Soon, Jialu Zhang, Wang Xiaoyao, Hongwei Du*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/581](https://doi.org/10.24963/ijcai.2023/581)

**Abstract**:

Abstractive text summarization is to generate concise summaries that well preserve both salient information and the overall semantic meanings of the given documents. However, real-world documents, e.g., financial reports, generally contain rich data such as charts and tabular data which invalidates most existing text summarization approaches. This paper is thus motivated to propose this novel approach to simultaneously summarize both textual and tabular data. Particularly, we first manually construct a “table+text → summary” dataset. Then, the tabular data is respectively embedded in a row-wise and column-wise manner, and the textual data is encoded at the sentence-level via an employed pre-trained model. We propose a salient detector gate respectively performed between each pair of row/column and sentence embeddings. The highly correlated content is considered as salient information that must be summarized. Extensive experiments have been performed on our constructed dataset and the promising results demonstrate the effectiveness of the proposed approach w.r.t. a number of both automatic and human evaluation criteria.

----

## [581] Learning Summary-Worthy Visual Representation for Abstractive Summarization in Video

**Authors**: *Zenan Xu, Xiaojun Meng, Yasheng Wang, Qinliang Su, Zexuan Qiu, Xin Jiang, Qun Liu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/582](https://doi.org/10.24963/ijcai.2023/582)

**Abstract**:

Multimodal abstractive summarization for videos (MAS) requires generating a concise textual summary to describe the highlights of a video according to multimodal resources, in our case, the video content and its transcript. Inspired by the success of the large-scale generative pre-trained language model (GPLM) in generating high-quality textual content (e.g., summary), recent MAS methods have proposed to adapt the GPLM to this task by equipping it with the visual information, which is often obtained through a general-purpose visual feature extractor. However, the generally extracted visual features may overlook some summary-worthy visual information, which impedes model performance. In this work, we propose a novel approach to learning the summary-worthy visual representation that facilitates abstractive summarization. Our method exploits the summary-worthy information from both the cross-modal transcript data and the knowledge that distills from the pseudo summary. Extensive experiments on three public multimodal datasets show that our method outperforms all competing baselines. Furthermore, with the advantages of summary-worthy visual information, our model can have a significant improvement on small datasets or even datasets with limited training data.

----

## [582] TITAN : Task-oriented Dialogues with Mixed-Initiative Interactions

**Authors**: *Sitong Yan, Shengli Song, Jingyang Li, Shiqi Meng, Guangneng Hu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/583](https://doi.org/10.24963/ijcai.2023/583)

**Abstract**:

In multi-domain task-oriented dialogue systems, users proactively propose a series of domain-specific requests that can often be under-or over-specified, sometimes with ambiguous and cross-domain demands. System-sided initiative would be necessary to identify certain situations and appropriately interact with users to resolve them. However, most existing task-oriented dialogue systems fail to consider such mixed-initiative interaction strategies, performing low efficiency and poor collaboration ability in human-computer conversation. In this paper, we construct a multi-domain task-oriented dialogue dataset with mixed-initiative strategies named TITAN from the large-scale dialogue corpus MultiWOZ 2.1. It contains a total of 1,800 human-human conversations where the system can either ask clarification questions actively or provides relevant information to address failure situations and implicit user requests. We report the results of several baseline models on system response generation and dialogue act prediction to assess the performance of SOTA methods on TITAN. These models can capture mixed-initiative dialogue acts, while remaining the deficiency to actively generate implicit requests and accurately provide alternative information, suggesting ample room for improvement in future studies.

----

## [583] Efficient Sign Language Translation with a Curriculum-based Non-autoregressive Decoder

**Authors**: *Pei Yu, Liang Zhang, Biao Fu, Yidong Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/584](https://doi.org/10.24963/ijcai.2023/584)

**Abstract**:

Most existing studies on Sign Language Translation (SLT) employ AutoRegressive  Decoding Mechanism (AR-DM) to generate target sentences. However, the main disadvantage of the AR-DM is high inference latency. To address this problem, we introduce Non-AutoRegressive Decoding Mechanism (NAR-DM) into SLT, which generates the whole sentence at once. Meanwhile, to improve its decoding ability, we integrate the advantages of curriculum learning and NAR-DM and propose a Curriculum-based NAR Decoder (CND). Specifically, the lower layers of the CND are expected to predict simple tokens that could be predicted correctly using source-side information solely. Meanwhile, the upper layers could predict complex tokens based on the lower layers' predictions. Therefore, our CND significantly reduces the model's inference latency while maintaining its competitive performance. Moreover, to further boost the performance of our CND, we propose a mutual learning framework, containing two decoders, i.e., an AR decoder and our CND. We jointly train the two decoders and minimize the KL divergence between their outputs, which enables our CND to learn the forward sequential knowledge from the strengthened AR decoder. Experimental results on PHOENIX2014T and CSL-Daily demonstrate that our model consistently outperforms all competitive baselines and achieves 7.92/8.02Ã— speed-up compared to the AR SLT model respectively. Our source code is available at https://github.com/yp20000921/CND.

----

## [584] Fast-StrucTexT: An Efficient Hourglass Transformer with Modality-guided Dynamic Token Merge for Document Understanding

**Authors**: *Mingliang Zhai, Yulin Li, Xiameng Qin, Chen Yi, Qunyi Xie, Chengquan Zhang, Kun Yao, Yuwei Wu, Yunde Jia*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/585](https://doi.org/10.24963/ijcai.2023/585)

**Abstract**:

Transformers achieve promising performance in document understanding because of their high effectiveness and still suffer from quadratic computational complexity dependency on the sequence length. General efficient transformers are challenging to be directly adapted to model document. They are unable to handle the layout representation in documents, e.g. word, line and paragraph, on different granularity levels and seem hard to achieve a good trade-off between efficiency and performance. To tackle the concerns, we propose Fast-StrucTexT, an efficient multi-modal framework based on the StrucTexT algorithm with an hourglass transformer architecture, for visual document understanding. Specifically, we design a modality-guided dynamic token merging block to make the model learn multi-granularity representation and prunes redundant tokens. Additionally, we present a multi-modal interaction module called Symmetry Cross-Attention (SCA) to consider multi-modal fusion and efficiently guide the token mergence. The SCA allows one modality input as query to calculate cross attention with another modality in a dual phase. Extensive experiments on FUNSD, SROIE, and CORD datasets demonstrate that our model achieves the state-of-the-art performance and almost 1.9x faster inference time than the state-of-the-art methods.

----

## [585] Exploring Effective Inter-Encoder Semantic Interaction for Document-Level Relation Extraction

**Authors**: *Liang Zhang, Zijun Min, Jinsong Su, Pei Yu, Ante Wang, Yidong Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/586](https://doi.org/10.24963/ijcai.2023/586)

**Abstract**:

In document-level relation extraction (RE), the models are required to correctly predict implicit relations in documents via relational reasoning. To this end, many graph-based methods have been proposed  for this task. Despite their success, these methods still suffer from several drawbacks: 1) their interaction between document encoder and graph encoder is usually unidirectional and insufficient; 2) their graph encoders often fail to capture the global context of nodes in document graph. In this paper, we propose a document-level RE model with a Graph-Transformer Network (GTN). The GTN includes two core sublayers: 1) the graph-attention sublayer that simultaneously models global and local contexts of nodes in the document graph; 2) the cross-attention sublayer, enabling GTN to capture the non-entity clue information from the document encoder. Furthermore, we introduce two auxiliary training tasks to enhance the bidirectional semantic interaction between the document encoder and GTN: 1) the graph node reconstruction that can effectively train our cross-attention sublayer to enhance the semantic transition from the document encoder to GTN; 2) the structure-aware adversarial knowledge distillation, by which we can effectively transfer the structural information of GTN to the document encoder. Experimental results on four benchmark datasets prove the effectiveness of our model. Our source code is available at https://github.com/DeepLearnXMU/DocRE-BSI.

----

## [586] NerCo: A Contrastive Learning Based Two-Stage Chinese NER Method

**Authors**: *Zai Zhang, Bin Shi, Haokun Zhang, Huang Xu, Yaodong Zhang, Yuefei Wu, Bo Dong, Qinghua Zheng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/587](https://doi.org/10.24963/ijcai.2023/587)

**Abstract**:

Sequence labeling serves as the most commonly used scheme for Chinese named entity recognition(NER). However, traditional sequence labeling methods classify tokens within an entity into different classes according to their positions. As a result, different tokens in the same entity may be learned with representations that are isolated and unrelated in target representation space, which could finally negatively affect the subsequent performance of token classification. In this paper, we point out and define this problem as Entity Representation Segmentation in Label-semantics. And then we present NerCo: Named entity recognition with Contrastive learning, a novel NER framework which can better exploit labeled data and avoid the above problem. Following the pretrain-finetune paradigm, NerCo firstly guides the encoder to learn powerful label-semantics based representations by gathering the encoded token representations of the same Semantic Class while pushing apart that of different. Subsequently, NerCo finetunes the learned encoder for final entity prediction. Extensive experiments on several datasets demonstrate that our framework can consistently improve the baseline and achieve state-of-the-art performance.

----

## [587] Genetic Prompt Search via Exploiting Language Model Probabilities

**Authors**: *Jiangjiang Zhao, Zhuoran Wang, Fangchun Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/588](https://doi.org/10.24963/ijcai.2023/588)

**Abstract**:

Prompt tuning for large-scale pretrained language models (PLMs) has shown remarkable potential, especially in low-resource scenarios such as few-shot learning. Moreover, derivative-free optimisation (DFO) techniques make it possible to tune prompts for a black-box PLM to better fit downstream tasks. However, there are usually preconditions to apply existing DFO-based prompt tuning methods, e.g. the backbone PLM needs to provide extra APIs so that hidden states (and/or embedding vectors) can be injected into it as continuous prompts, or carefully designed (discrete) manual prompts need to be available beforehand, serving as the initial states of the tuning algorithm. To waive such preconditions and make DFO-based prompt tuning ready for general use, this paper introduces a novel genetic algorithm (GA) that evolves from empty prompts, and uses the predictive probabilities derived from the backbone PLM(s) on the basis of a (few-shot) training set to guide the token selection process during prompt mutations. Experimental results on diverse benchmark datasets show that the proposed precondition-free method significantly outperforms the existing DFO-style counterparts that require preconditions, including black-box tuning, genetic prompt search and gradient-free instructional prompt search.

----

## [588] Learning Few-shot Sample-set Operations for Noisy Multi-label Aspect Category Detection

**Authors**: *Shiman Zhao, Wei Chen, Tengjiao Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/589](https://doi.org/10.24963/ijcai.2023/589)

**Abstract**:

Multi-label Aspect Category Detection (MACD) is essential for aspect-based sentiment analysis, which aims to identify multiple aspect categories in a given sentence. Few-shot MACD is critical due to the scarcity of labeled data. However, MACD is a high-noise task, and existing methods fail to address it with only two or three training samples per class, which limits the application in practice. To solve above issues, we propose a group of Few-shot Sample-set Operations (FSO) to solve noisy MACD in fewer sample scenarios by identifying the semantic contents of samples. Learning interactions among intersection, subtraction, and union networks, the FSO imitates arithmetic operations on samples to distinguish relevant and irrelevant aspect contents. Eliminating the negative effect caused by noises, the FSO extracts discriminative prototypes and customizes a dedicated query vector for each class. Besides, we design a multi-label architecture, which integrates with score-wise loss and multi-label loss to optimize the FSO for multi-label prediction, avoiding complex threshold training or selection. Experiments show that our method achieves considerable performance. Significantly, it improves by 11.01% at most and an average of 8.59% Macro-F in fewer sample scenarios.

----

## [589] COOL, a Context Outlooker, and Its Application to Question Answering and Other Natural Language Processing Tasks

**Authors**: *Fangyi Zhu, See-Kiong Ng, Stéphane Bressan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/590](https://doi.org/10.24963/ijcai.2023/590)

**Abstract**:

Vision outlooker improves the performance of vision transformers, which implements a self-attention mechanism by adding an outlook attention, a form of local attention.  

In natural language processing, as has been the case in computer vision and other domains, transformer-based models constitute the state-of-the-art for most processing tasks. In this domain, too, many authors have argued and demonstrated the importance of local context.

We present an outlook attention mechanism, COOL, for natural language processing. COOL, added on top of the self-attention layers of a transformer-based model, encodes local syntactic context considering word proximity and more pair-wise constraints than dynamic convolution used by existing approaches.

A comparative empirical performance evaluation of an implementation of COOL with different transformer-based models confirms the opportunity for improvement over a baseline using the original models alone for various natural language processing tasks, including question answering. The proposed approach achieves competitive performance with existing state-of-the-art methods on some tasks.

----

## [590] DiSProD: Differentiable Symbolic Propagation of Distributions for Planning

**Authors**: *Palash Chatterjee, Ashutosh Chapagain, Weizhe Chen, Roni Khardon*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/591](https://doi.org/10.24963/ijcai.2023/591)

**Abstract**:

The paper introduces DiSProD, an online planner developed for
environments with probabilistic transitions in continuous state and
action spaces. DiSProD builds a symbolic graph that captures the
distribution of future trajectories, conditioned on a given policy,
using independence assumptions and approximate propagation of
distributions. The symbolic graph provides a differentiable
representation of the policy's value, enabling efficient gradient-based
optimization for long-horizon search. The propagation of approximate
distributions can be seen as an aggregation of many trajectories, making
it well-suited for dealing with sparse rewards and stochastic
environments. An extensive experimental evaluation compares DiSProD to
state-of-the-art planners in discrete-time planning and real-time
control of robotic systems. The proposed method improves over existing
planners in handling stochastic environments, sensitivity to search
depth, sparsity of rewards, and large action spaces. Additional
real-world experiments demonstrate that DiSProD can control ground
vehicles and surface vessels to successfully navigate around obstacles.

----

## [591] Minimizing Reachability Times on Temporal Graphs via Shifting Labels

**Authors**: *Argyrios Deligkas, Eduard Eiben, George Skretas*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/592](https://doi.org/10.24963/ijcai.2023/592)

**Abstract**:

We study how we can accelerate the spreading of information in temporal graphs via shifting operations; a problem that captures real-world applications varying from information flows to distribution schedules. In a temporal graph there is a set of fixed vertices and the available connections between them change over time in a predefined manner.  We observe that, in some cases, shifting some connections, i.e., advancing or delaying them, can decrease the time required to reach from some vertex (source) to another vertex. We study how we can minimize the maximum time a set of sources needs to reach every vertex, when we are allowed to shift some of the connections. If we restrict the allowed number of changes, we prove that, already for a single source, the problem is NP-hard, and W[2]-hard when parameterized by the number of changes. Then we focus on unconstrained number of changes. We derive a polynomial-time algorithm when there is one source. When there are two sources, we show that the problem becomes NP-hard; on the other hand, we design an FPT algorithm parameterized by the treewidth of the graph plus the lifetime of the optimal solution, that works for any number of sources. Finally, we provide polynomial-time algorithms for several graph classes.

----

## [592] On the Compilability of Bounded Numeric Planning

**Authors**: *Nicola Gigante, Enrico Scala*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/593](https://doi.org/10.24963/ijcai.2023/593)

**Abstract**:

Bounded numeric planning, where each numeric variable domain is bounded, is PSPACE-complete, but such a complexity result does not capture how hard it really is, since the same holds even for the practically much easier STRIPS fragment. A finer way to compare the difficulty of planning formalisms is through the notion of compilability, which has been however extensively studied only for classical planning by Nebel. This paper extends  Nebel's framework to the setting of bounded numeric planning. First, we identify a variety of numeric fragments differing on the degree of the polynomials involved and the availability of features such as conditional effects and Boolean conditions; then we study the compilability of these fragments to each other and to the classical fragments. Surprisingly, numeric and classical planning with conditional effects and Boolean conditions can be compiled both ways preserving plan size exactly, while the same does not hold when targeting pure STRIPS. Our study reveals also that numeric fragments cluster into two equivalence classes separated by the availability of incomplete initial state specifications, a feature allowing to specify uncertainty in the initial state.

----

## [593] On the Study of Curriculum Learning for Inferring Dispatching Policies on the Job Shop Scheduling

**Authors**: *Zangir Iklassov, Dmitrii Medvedev, Ruben Solozabal Ochoa de Retana, Martin Takác*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/594](https://doi.org/10.24963/ijcai.2023/594)

**Abstract**:

This paper studies the use of Curriculum Learning on Reinforcement Learning (RL) to improve the performance of the dispatching policies learned on the Job-shop Scheduling Problem (JSP). Current works in the literature present a large optimality gap when learning end-to-end solutions on this problem. In this regard, we identify the difficulty for RL to learn directly on large instances as part of the issue and use Curriculum Learning (CL) to mitigate this effect. Particularly, CL sequences the learning process in a curriculum of increasing complexity tasks, which allows learning on large instances that otherwise would be impossible to learn from scratch. In this paper, we present a size-agnostic model that enables us to demonstrate that current curriculum strategies have a major impact on the quality of the solution inferred. In addition, we introduce a novel Reinforced Adaptive Staircase Curriculum Learning (RASCL) strategy, which adjusts the difficulty level during the learning process by revisiting the worst-performing instances. Conducted experiments on Taillard’s and Demirkol’s datasets show that the presented approach significantly improves the current stateof-the-art models on the JSP. It reduces the average optimality gap from 19.35% to 10.46% on Taillard’s instances and from 38.43% to 18.85% on Demirkol’s instances.

----

## [594] Simulation-Assisted Optimization for Large-Scale Evacuation Planning with Congestion-Dependent Delays

**Authors**: *Kazi Ashik Islam, Da Qi Chen, Madhav V. Marathe, Henning S. Mortveit, Samarth Swarup, Anil Vullikanti*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/595](https://doi.org/10.24963/ijcai.2023/595)

**Abstract**:

Evacuation planning is a crucial part of disaster management. However, joint optimization of its two essential components, routing and scheduling, with objectives such as minimizing average evacuation time or evacuation completion time, is a computationally hard problem. To approach it, we present MIP-LNS, a scalable optimization method that utilizes heuristic search with mathematical optimization and can optimize a variety of objective functions. We also present the method MIP-LNS-SIM, where we combine agent-based simulation with MIP-LNS to estimate delays due to congestion, as well as, find optimized plans considering such delays. We use Harris County in Houston, Texas, as our study area. We show that, within a given time limit, MIP-LNS finds better solutions than existing methods in terms of three different metrics. However, when congestion dependent delay is considered, MIP-LNS-SIM outperforms MIP-LNS in multiple performance metrics. In addition, MIP-LNS-SIM has a significantly lower percent error in estimated evacuation completion time compared to MIP-LNS.

----

## [595] K∗ Search over Orbit Space for Top-k Planning

**Authors**: *Michael Katz, Junkyu Lee*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/596](https://doi.org/10.24963/ijcai.2023/596)

**Abstract**:

Top-k planning, the task of finding k top-cost plans, is a key formalism for many planning applications and K* search is a well-established approach to top-k planning.  The algorithm iteratively runs A* search and Eppstein’s algorithm until a sufficient number of plans is found.  The performance of K* algorithm is therefore inherently limited by the performance of A*, and in order to improve K* performance, that of A* must be improved.  In cost-optimal planning, orbit space search improves A* performance by exploiting symmetry pruning, essentially performing A* in the orbit space instead of state space.  In this work, we take a similar approach to top-k planning.  We show theoretical equivalence between the goal paths in the state space and in the orbit space, allowing to perform K* search in the orbit space instead, reconstructing plans from the found paths in the orbit space.  We prove that our algorithm is sound and complete for top-k planning and empirically show it to achieve state-of-the-art performance, overtaking all existing to date top-k planners.  The code is available at https://github.com/IBM/kstar.

----

## [596] Helpful Information Sharing for Partially Informed Planning Agents

**Authors**: *Sarah Keren, David Wies, Sara Bernardini*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/597](https://doi.org/10.24963/ijcai.2023/597)

**Abstract**:

In many real-world settings, an autonomous agent may not have sufficient information or sensory capabilities to accomplish its goals, even when they are achievable. In some cases, the needed information can be provided by another agent, but information sharing might be costly due to limited communication bandwidth and other constraints. We address the problem of Helpful Information Sharing (HIS), which focuses on selecting minimal information to reveal to a partially informed agent in order to guarantee it can achieve its goal. We offer a novel compilation of HIS to a classical planning problem, which can be solved efficiently by any off-the-shelf planner. We provide guarantees of optimality for our approach and describe its extensions to maximize robustness and support settings in which the agent needs to decide which sensors to deploy in the environment. We demonstrate the power of our approaches on a set of standard benchmarks as well as on a novel benchmark.

----

## [597] Mean Payoff Optimization for Systems of Periodic Service and Maintenance

**Authors**: *David Klaska, Antonín Kucera, Vít Musil, Vojtech Rehák*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/598](https://doi.org/10.24963/ijcai.2023/598)

**Abstract**:

Consider oriented graph nodes requiring periodic visits by a service agent. The agent moves among the nodes and receives a payoff for each completed service task, depending on the time elapsed since the previous visit to a node. We consider the problem of finding a suitable schedule for the agent to maximize its long-run average payoff per time unit. We show that the problem of constructing an epsilon-optimal schedule is PSPACE-hard for every fixed non-negative epsilon, and that there exists an optimal periodic schedule of exponential length. We propose randomized finite-memory (RFM) schedules as a compact description of the agent's strategies and design an efficient algorithm for constructing RFM schedules. Furthermore, we construct deterministic periodic schedules by sampling from RFM schedules.

----

## [598] Action Space Reduction for Planning Domains

**Authors**: *Harsha Kokel, Junkyu Lee, Michael Katz, Kavitha Srinivas, Shirin Sohrabi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/599](https://doi.org/10.24963/ijcai.2023/599)

**Abstract**:

Planning tasks succinctly represent labeled transition systems, with each ground action corresponding to a label. This granularity, however, is not necessary for solving planning tasks and can be harmful, especially for model-free methods. In order to apply such methods, the label sets are often manually reduced. In this work, we propose automating this manual process. We characterize a valid label reduction for classical planning tasks and propose an automated way of obtaining such valid reductions by leveraging lifted mutex groups. Our experiments show a significant reduction in the action label space size across a wide collection of planning domains. We demonstrate the benefit of our automated label reduction in two separate use cases: improved sample complexity of model-free reinforcement learning algorithms and speeding up successor generation in lifted planning. The code and supplementary material are available at https://github.com/IBM/Parameter-Seed-Set.

----

## [599] Recursive Small-Step Multi-Agent A* for Dec-POMDPs

**Authors**: *Wietze Koops, Nils Jansen, Sebastian Junges, Thiago D. Simão*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/600](https://doi.org/10.24963/ijcai.2023/600)

**Abstract**:

We present recursive small-step multi-agent A* (RS-MAA*), an exact algorithm that optimizes the expected reward in decentralized partially observable Markov decision processes (Dec-POMDPs). RS-MAA* builds on multi-agent A* (MAA*), an algorithm that finds policies by exploring a search tree, but tackles two major scalability concerns. First, we employ a modified, small-step variant of the search tree that avoids the double exponential outdegree of the classical formulation. Second, we use a tight and recursive heuristic that we compute on-the-fly, thereby avoiding an expensive precomputation. The resulting algorithm is conceptually simple, yet it shows superior performance on a rich set of standard benchmarks.

----



[Go to the previous page](IJCAI-2023-list02.md)

[Go to the next page](IJCAI-2023-list04.md)

[Go to the catalog section](README.md)