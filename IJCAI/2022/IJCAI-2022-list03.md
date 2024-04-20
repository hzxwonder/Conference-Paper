## [400] Geometric Transformer for End-to-End Molecule Properties Prediction

**Authors**: *Yoni Choukroun, Lior Wolf*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/401](https://doi.org/10.24963/ijcai.2022/401)

**Abstract**:

Transformers have become methods of choice in many applications thanks to their ability to represent complex interactions between elements. 
However, extending the Transformer architecture to non-sequential data such as molecules and enabling its training on small datasets remains a challenge. 
In this work, we introduce a Transformer-based architecture for molecule property prediction, which is able to capture the geometry of the molecule. 
We modify the classical positional encoder by an initial encoding of the molecule geometry, as well as a learned gated self-attention mechanism. 
We further suggest an augmentation scheme for molecular data capable of avoiding the overfitting induced by the overparameterized architecture. 
The proposed framework outperforms the state-of-the-art methods while being based on pure machine learning solely, i.e. the method does not incorporate domain knowledge from quantum chemistry and does not use extended geometric inputs besides the pairwise atomic distances.

----

## [401] Multiband VAE: Latent Space Alignment for Knowledge Consolidation in Continual Learning

**Authors**: *Kamil Deja, Pawel Wawrzynski, Wojciech Masarczyk, Daniel Marczak, Tomasz Trzcinski*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/402](https://doi.org/10.24963/ijcai.2022/402)

**Abstract**:

We propose a new method for unsupervised generative continual learning through realignment of Variational Autoencoder's latent space. Deep generative models suffer from catastrophic forgetting in the same way as other neural structures. Recent generative continual learning works approach this problem and try to learn from new data without forgetting previous knowledge.  However, those methods usually focus on artificial scenarios where examples share almost no similarity between subsequent portions of data - an assumption not realistic in the real-life applications of continual learning. In this work, we identify this limitation and posit the goal of generative continual learning as a knowledge accumulation task. We solve it by continuously aligning latent representations of new data that we call bands in additional latent space where examples are encoded independently of their source task. In addition, we introduce a method for controlled forgetting of past data that simplifies this process. On top of the standard continual learning benchmarks, we propose a novel challenging knowledge consolidation scenario and show that the proposed approach outperforms state-of-the-art by up to twofold across all experiments and additional real-life evaluation. To our knowledge, Multiband VAE is the first method to show forward and backward knowledge transfer in generative continual learning.

----

## [402] Reinforcement Learning with Option Machines

**Authors**: *Floris den Hengst, Vincent François-Lavet, Mark Hoogendoorn, Frank van Harmelen*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/403](https://doi.org/10.24963/ijcai.2022/403)

**Abstract**:

Reinforcement learning (RL) is a powerful framework for learning complex behaviors, but lacks adoption in many settings due to sample size requirements. We introduce a framework for increasing sample efficiency of RL algorithms. Our approach focuses on optimizing environment rewards with high-level instructions. These are modeled as a high-level controller over temporally extended actions known as options. These options can be looped, interleaved and partially ordered with a rich language for high-level instructions. Crucially, the instructions may be underspecified in the sense that following them does not guarantee high reward in the environment. We present an algorithm for control with these so-called option machines (OMs), discuss option selection for the partially ordered case and describe an algorithm for learning with OMs. We compare our approach in zero-shot, single- and multi-task settings in an environment with fully specified and underspecified instructions. We find that OMs perform significantly better than or comparable to the state-of-art in all environments and learning settings.

----

## [403] Coherent Probabilistic Aggregate Queries on Long-horizon Forecasts

**Authors**: *Prathamesh Deshpande, Sunita Sarawagi*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/404](https://doi.org/10.24963/ijcai.2022/404)

**Abstract**:

Long range forecasts are the starting point of many decision support systems that need to draw inference from high-level aggregate patterns on forecasted values. State of the art time-series forecasting methods are either subject to concept drift on long-horizon forecasts, or fail to accurately predict coherent and accurate high-level aggregates.

In this work, we present a novel probabilistic forecasting method that produces forecasts that are coherent in terms of base level and predicted aggregate statistics. We achieve the coherency between predicted base-level and aggregate statistics using a novel inference method based on KL-divergence that can be solved efficiently in closed form. We show that our method improves forecast performance across both base level and unseen aggregates post inference on real datasets ranging three diverse domains. (Project URL)

----

## [404] Taylor-Lagrange Neural Ordinary Differential Equations: Toward Fast Training and Evaluation of Neural ODEs

**Authors**: *Franck Djeumou, Cyrus Neary, Eric Goubault, Sylvie Putot, Ufuk Topcu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/405](https://doi.org/10.24963/ijcai.2022/405)

**Abstract**:

Neural ordinary differential equations (NODEs) -- parametrizations of differential equations using neural networks -- have shown tremendous promise in learning models of unknown continuous-time dynamical systems from data. However, every forward evaluation of a NODE requires numerical integration of the neural network used to capture the system dynamics, making their training prohibitively expensive. Existing works rely on off-the-shelf adaptive step-size numerical integration schemes, which often require an excessive number of evaluations of the underlying dynamics network to obtain sufficient accuracy for training. By contrast, we accelerate the evaluation and the training of NODEs by proposing a data-driven approach to their numerical integration. The proposed Taylor-Lagrange NODEs (TL-NODEs) use a fixed-order Taylor expansion for numerical integration, while also learning to estimate the expansion's approximation error. As a result, the proposed approach achieves the same accuracy as adaptive step-size schemes while employing only low-order Taylor expansions, thus greatly reducing the computational cost necessary to integrate the NODE. A suite of numerical experiments, including modeling dynamical systems, image classification, and density estimation, demonstrate that TL-NODEs can be trained more than an order of magnitude faster than state-of-the-art approaches, without any loss in performance.

----

## [405] Residual Contrastive Learning for Image Reconstruction: Learning Transferable Representations from Noisy Images

**Authors**: *Nanqing Dong, Matteo Maggioni, Yongxin Yang, Eduardo Pérez-Pellitero, Ales Leonardis, Steven McDonagh*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/406](https://doi.org/10.24963/ijcai.2022/406)

**Abstract**:

This paper is concerned with contrastive learning (CL) for low-level image restoration and enhancement tasks. We propose a new label-efficient learning paradigm based on residuals, residual contrastive learning (RCL), and derive an unsupervised visual representation learning framework, suitable for low-level vision tasks with noisy inputs. While supervised image reconstruction aims to minimize residual terms directly, RCL alternatively builds a connection between residuals and CL by defining a novel instance discrimination pretext task, using residuals as the discriminative feature. Our formulation mitigates the severe task misalignment between instance discrimination pretext tasks and downstream image reconstruction tasks, present in existing CL frameworks. Experimentally, we find that RCL can learn robust and transferable representations that improve the performance of various downstream tasks, such as denoising and super resolution, in comparison with recent self-supervised methods designed specifically for noisy inputs. Additionally, our unsupervised pre-training can significantly reduce annotation costs whilst maintaining performance competitive with fully-supervised image reconstruction.

----

## [406] Function-words Adaptively Enhanced Attention Networks for Few-Shot Inverse Relation Classification

**Authors**: *Chunliu Dou, Shaojuan Wu, Xiaowang Zhang, Zhiyong Feng, Kewen Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/407](https://doi.org/10.24963/ijcai.2022/407)

**Abstract**:

The relation classification is to identify semantic relations between two entities in a given text. While existing models perform well for classifying inverse relations with large datasets, their performance is significantly reduced for few-shot learning. In this paper, we propose a function words adaptively enhanced attention framework (FAEA) for few-shot inverse relation classification, in which a hybrid attention model is designed to attend class-related function words based on meta-learning. As the involvement of function words brings in significant intra-class redundancy, an adaptive message passing mechanism is introduced to capture and transfer inter-class differences.We mathematically analyze the negative impact of function words from dot-product measurement, which explains why the message passing mechanism effectively reduces the impact. Our experimental results show that FAEA outperforms strong baselines, especially the inverse relation accuracy is improved by 14.33% under 1-shot setting in FewRel1.0.

----

## [407] Multi-Vector Embedding on Networks with Taxonomies

**Authors**: *Yue Fan, Xiuli Ma*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/408](https://doi.org/10.24963/ijcai.2022/408)

**Abstract**:

A network can effectively depict close relationships among its nodes, with labels in a taxonomy describing the nodes' rich attributes. Network embedding aims at learning a representation vector for each node and label to preserve their proximity, while most existing methods suffer from serious underfitting when dealing with datasets with dense node-label links. For instance, a node could have dozens of labels describing its diverse properties, causing the single node vector overloaded and hard to fit all the labels. We propose HIerarchical Multi-vector Embedding (HIME), which solves the underfitting problem by adaptively learning multiple 'branch vectors' for each node to dynamically fit separate sets of labels in a hierarchy-aware embedding space. Moreover, a 'root vector' is learned for each node based on its branch vectors to better predict the sparse but valuable node-node links with the knowledge of its labels. Experiments reveal HIMEâ€™s comprehensive advantages over existing methods on tasks such as proximity search, link prediction and hierarchical classification.

----

## [408] Fixed-Budget Pure Exploration in Multinomial Logit Bandits

**Authors**: *Boli Fang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/409](https://doi.org/10.24963/ijcai.2022/409)

**Abstract**:

In this paper we investigate pure exploration problem in Multinomial Logit bandit(MNL-bandit) under fxed budget settings, a problem motivated by real-time applications in online advertising and retailing. Given an MNL-bandit instance and a fixed exploration budget, our goal is to minimize the misidentifcation error of the optimal assortment. Towards such an end we propose an algorithm that achieve gap-dependent complexities, and complement our investigation with a discussion on recent studies and a minimax lower bound on misidentifcation probability. To the best of our knowledge, our paper is the frst to address the recently proposed open problem of fxed-budget pure exploration problem for MNL-bandits.

----

## [409] Learning Unforgotten Domain-Invariant Representations for Online Unsupervised Domain Adaptation

**Authors**: *Cheng Feng, Chaoliang Zhong, Jie Wang, Ying Zhang, Jun Sun, Yasuto Yokota*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/410](https://doi.org/10.24963/ijcai.2022/410)

**Abstract**:

Existing unsupervised domain adaptation (UDA) studies focus on transferring knowledge in an offline manner. However, many tasks involve online requirements, especially in real-time systems. In this paper, we discuss Online UDA (OUDA) which assumes that the target samples are arriving sequentially as a small batch. OUDA tasks are challenging for prior UDA methods since online training suffers from catastrophic forgetting which leads to poor generalization. Intuitively, a good memory is a crucial factor in the success of OUDA. We formalize this intuition theoretically with a generalization bound where the OUDA target error can be bounded by the source error, the domain discrepancy distance, and a novel metric on forgetting in continuous online learning. Our theory illustrates the tradeoffs inherent in learning and remembering representations for OUDA. To minimize the proposed forgetting metric, we propose a novel source feature distillation (SFD) method which utilizes the source-only model as a teacher to guide the online training. In the experiment, we modify three UDA algorithms, i.e., DANN, CDAN, and MCC, and evaluate their performance on OUDA tasks with real-world datasets. By applying SFD, the performance of all baselines is significantly improved.

----

## [410] Comparison Knowledge Translation for Generalizable Image Classification

**Authors**: *Zunlei Feng, Tian Qiu, Sai Wu, Xiaotuan Jin, Zengliang He, Mingli Song, Huiqiong Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/411](https://doi.org/10.24963/ijcai.2022/411)

**Abstract**:

Deep learning has recently achieved remarkable performance in image classification tasks, which depends heavily on massive annotation. However, the classification mechanism of existing deep learning models seems to contrast to humans' recognition mechanism. With only a glance at an image of the object even unknown type, humans can quickly and precisely find other same category objects from massive images, which benefits from daily recognition of various objects. In this paper, we attempt to build a generalizable framework that emulates the humans' recognition mechanism in the image classification task, hoping to improve the classification performance on unseen categories with the support of annotations of other categories. Specifically, we investigate a new task termed Comparison Knowledge Translation (CKT). Given a set of fully labeled categories, CKT aims to translate the comparison knowledge learned from the labeled categories to a set of novel categories. To this end, we put forward a Comparison Classification Translation Network (CCT-Net), which comprises a comparison classifier and a matching discriminator. The comparison classifier is devised to classify whether two images belong to the same category or not, while the matching discriminator works together in an adversarial manner to ensure whether classified results match the truth. Exhaustive experiments show that CCT-Net achieves surprising generalization ability on unseen categories and SOTA performance on target categories.

----

## [411] Non-Cheating Teaching Revisited: A New Probabilistic Machine Teaching Model

**Authors**: *Cèsar Ferri, José Hernández-Orallo, Jan Arne Telle*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/412](https://doi.org/10.24963/ijcai.2022/412)

**Abstract**:

Over the past decades in the field of machine teaching, several restrictions have been  introduced to avoid ‘cheating’, such as collusion-free or non-clashing teaching. However, these restrictions forbid several teaching situations  that  we  intuitively consider natural and fair, especially those ‘changes of mind’ of the learner as more evidence is given, affecting the likelihood of concepts and ultimately their  posteriors. Under a new generalised probabilistic teaching, not only do these non-cheating constraints look too narrow but we also show that the most relevant machine teaching models are particular cases of this framework: the consistency graph between concepts and elements simply becomes a joint probability distribution. We show a simple procedure that builds the witness joint distribution from the ground joint distribution. We prove a chain of relations, also with a theoretical lower bound, on the teaching dimension of the old and new models. Overall, this new setting is more general than the traditional machine teaching models, yet at the same time more intuitively capturing a less abrupt notion of non-cheating teaching.

----

## [412] DeepExtrema: A Deep Learning Approach for Forecasting Block Maxima in Time Series Data

**Authors**: *Asadullah Hill Galib, Andrew McDonald, Tyler Wilson, Lifeng Luo, Pang-Ning Tan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/413](https://doi.org/10.24963/ijcai.2022/413)

**Abstract**:

Accurate forecasting of extreme values in time series is critical due to the significant impact of extreme events on human and natural systems. This paper presents DeepExtrema, a novel framework that combines a deep neural network (DNN) with generalized extreme value (GEV) distribution to forecast the block maximum value of a time series. Implementing such a network is a challenge as the framework must preserve the inter-dependent constraints among the GEV model parameters even when the DNN is initialized. We describe our approach to address this challenge and present an architecture that enables both conditional mean and quantile prediction of the block maxima. The extensive experiments performed on both real-world and synthetic data demonstrated the superiority of DeepExtrema compared to other baseline methods.

----

## [413] Multi-view Unsupervised Graph Representation Learning

**Authors**: *Jiangzhang Gan, Rongyao Hu, Mengmeng Zhan, Yujie Mo, Yingying Wan, Xiaofeng Zhu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/414](https://doi.org/10.24963/ijcai.2022/414)

**Abstract**:

Both data augmentation and contrastive loss are the key components of contrastive learning. In this paper, we design a new multi-view unsupervised graph representation learning method including adaptive data augmentation and multi-view contrastive learning, to address some issues of contrastive learning ignoring the information from feature space. Specifically, the adaptive data augmentation  first builds a feature graph from the feature space, and then designs a deep graph learning model on the original representation and the topology graph to update the feature graph and the new representation. As a result, the adaptive data augmentation outputs multi-view information, which is fed into two GCNs to generate multi-view embedding features. Two kinds of contrastive losses are further designed on multi-view embedding features to explore the complementary information among the topology and feature graphs. Additionally, adaptive data augmentation and contrastive learning are embedded in a unified framework to form an end-to-end model. Experimental results verify the effectiveness of our proposed method, compared to  state-of-the-art methods.

----

## [414] A Reinforcement Learning-Informed Pattern Mining Framework for Multivariate Time Series Classification

**Authors**: *Ge Gao, Qitong Gao, Xi Yang, Miroslav Pajic, Min Chi*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/415](https://doi.org/10.24963/ijcai.2022/415)

**Abstract**:

Multivariate time series (MTS) classification is a challenging and important task in various domains and real-world applications. Much of prior work on MTS can be roughly divided into neural network (NN)- and pattern-based methods. The former can lead to robust classification performance, but many of the generated patterns are challenging to interpret; while the latter often produce interpretable patterns that may not be helpful for the classification task. In this work, we propose a reinforcement learning (RL) informed PAttern Mining framework (RLPAM) to identify interpretable yet important patterns for MTS classification. Our framework has been validated by 30 benchmark datasets as well as real-world large-scale electronic health records (EHRs) for an extremely challenging task: sepsis shock early prediction. We show that RLPAM outperforms the state-of-the-art NN-based methods on 14 out of 30 datasets as well as on the EHRs. Finally, we show how RL informed patterns can be interpretable and can improve our understanding of septic shock progression.

----

## [415] Bootstrapping Informative Graph Augmentation via A Meta Learning Approach

**Authors**: *Hang Gao, Jiangmeng Li, Wenwen Qiang, Lingyu Si, Fuchun Sun, Changwen Zheng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/416](https://doi.org/10.24963/ijcai.2022/416)

**Abstract**:

Recent works explore learning graph representations in a self-supervised manner. In graph contrastive learning, benchmark methods apply various graph augmentation approaches. However, most of the augmentation methods are non-learnable, which causes the issue of generating unbeneficial augmented graphs. Such augmentation may degenerate the representation ability of graph contrastive learning methods. Therefore, we motivate our method to generate augmented graph with a learnable graph augmenter, called MEta Graph Augmentation (MEGA). We then clarify that a "good" graph augmentation must have uniformity at the instance-level and informativeness at the feature-level. To this end, we propose a novel approach to learning a graph augmenter that can generate an augmentation with uniformity and informativeness. The objective of the graph augmenter is to promote our feature extraction network to learn a more discriminative feature representation, which motivates us to propose a meta-learning paradigm. Empirically, the experiments across multiple benchmark datasets demonstrate that MEGA outperforms the state-of-the-art methods in graph self-supervised learning tasks. Further experimental studies prove the effectiveness of different terms of MEGA. Our codes are available at https://github.com/hang53/MEGA.

----

## [416] Learning First-Order Rules with Differentiable Logic Program Semantics

**Authors**: *Kun Gao, Katsumi Inoue, Yongzhi Cao, Hanpin Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/417](https://doi.org/10.24963/ijcai.2022/417)

**Abstract**:

Learning first-order logic programs (LPs) from relational facts which yields intuitive insights into the data is a challenging topic in neuro-symbolic research. We introduce a novel differentiable inductive logic programming (ILP) model, called differentiable first-order rule learner (DFOL), which finds the correct LPs from relational facts by searching for the interpretable matrix representations of LPs. These interpretable matrices are deemed as trainable tensors in neural networks (NNs). The NNs are devised according to the differentiable semantics of LPs. Specifically, we first adopt a novel propositionalization method that transfers facts to NN-readable vector pairs representing interpretation pairs. We replace the immediate consequence operator with NN constraint functions consisting of algebraic operations and a sigmoid-like activation function. We map the symbolic forward-chained format of LPs into NN constraint functions consisting of operations between subsymbolic vector representations of atoms. By applying gradient descent, the trained well parameters of NNs can be decoded into precise symbolic LPs in forward-chained logic format. We demonstrate that DFOL can perform on several standard ILP datasets, knowledge bases, and probabilistic relation facts and outperform several well-known differentiable ILP models. Experimental results indicate that DFOL is a precise, robust, scalable, and computationally cheap differentiable ILP model.

----

## [417] Attributed Graph Clustering with Dual Redundancy Reduction

**Authors**: *Lei Gong, Sihang Zhou, Wenxuan Tu, Xinwang Liu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/418](https://doi.org/10.24963/ijcai.2022/418)

**Abstract**:

Attributed graph clustering is a basic yet essential method for graph data exploration. Recent efforts over graph contrastive learning
have achieved impressive clustering performance. However, we observe that the commonly adopted InfoMax operation tends to capture redundant information, limiting the downstream clustering performance. To this end, we develop a novel method termed attributed graph clustering with dual redundancy reduction (AGC-DRR) to reduce the information redundancy in both input space and latent feature space. Specifically, for the input space redundancy reduction, we introduce an adversarial learning mechanism to adaptively learn a redundant edge-dropping matrix to ensure the diversity of the compared sample pairs. To reduce the redundancy in the latent space, we force the correlation matrix of the cross-augmentation sample embedding to approximate an identity matrix. Consequently, the learned network is forced to be robust against perturbation while discriminative against different samples. Extensive experiments have demonstrated that AGC-DRR outperforms the state-of-the-art clustering methods on most of our benchmarks. The corresponding code is available at https://github.com/gongleii/AGC-DRR.

----

## [418] Sample Complexity Bounds for Robustly Learning Decision Lists against Evasion Attacks

**Authors**: *Pascale Gourdeau, Varun Kanade, Marta Kwiatkowska, James Worrell*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/419](https://doi.org/10.24963/ijcai.2022/419)

**Abstract**:

A fundamental problem in adversarial machine learning is to quantify how much training data is needed in the presence of evasion attacks. In this paper we address this issue within the framework of PAC learning, focusing on the class of decision lists. Given that distributional assumptions are essential in the adversarial setting, we work with probability distributions on the input data that satisfy a Lipschitz condition: nearby points have similar probability. Our key results illustrate that the adversary's budget (that is, the number of bits it can perturb on each input) is a fundamental quantity in determining the sample complexity of robust learning. Our first main result is a sample-complexity lower bound: the class of monotone conjunctions (essentially the simplest non-trivial hypothesis class on the Boolean hypercube) and any superclass  has sample complexity at least exponential in the adversary's budget. Our second main result is a corresponding upper bound: for every fixed k the class of k-decision lists has polynomial sample complexity against a log(n)-bounded adversary. This sheds further light on the question of whether an efficient PAC learning algorithm can always be used as an efficient log(n)-robust learning algorithm under the uniform distribution.

----

## [419] RoboGNN: Robustifying Node Classification under Link Perturbation

**Authors**: *Sheng Guan, Hanchao Ma, Yinghui Wu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/420](https://doi.org/10.24963/ijcai.2022/420)

**Abstract**:

Graph neural networks (GNNs) have emerged as powerful approaches for graph representation learning and node classification. Nevertheless, they can be vulnerable (sensitive) to link perturbations due to structural noise or adversarial attacks. This paper introduces RoboGNN, a novel framework that simultaneously robustifies an input classifier to a counterpart with certifiable robustness, and suggests desired graph representation with auxiliary links to ensure the robustness guarantee. (1) We introduce (p,θ)-robustness, which characterizes the robustness guarantee of a GNN-based classifier if its performance is insensitive for at least θ fraction of a targeted set of nodes under any perturbation of a set of vulnerable links up to a bounded size p. (2) We present a co-learning framework that interacts model learning with graph structural learning to robustify an input model M to a (p,θ)-robustness counterpart. The framework also outputs the desired graph structures that ensure the robustness. Using real-world benchmark graphs, we experimentally verify that roboGNN can effectively robustify representative GNNs with guaranteed robustness, and desirable gains on accuracy.

----

## [420] Option Transfer and SMDP Abstraction with Successor Features

**Authors**: *Dongge Han, Sebastian Tschiatschek*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/421](https://doi.org/10.24963/ijcai.2022/421)

**Abstract**:

Abstraction plays an important role in the generalisation of knowledge and skills and is key to sample efficient learning. In this work, we study joint temporal and state abstraction in reinforcement learning, where temporally-extended actions in the form of options induce temporal abstractions, while aggregation of similar states with respect to abstract options induces state abstractions. Many existing abstraction schemes ignore the interplay of state and temporal abstraction. Consequently, the considered option policies often cannot be directly transferred to new environments due to changes in the state space and transition dynamics. To address this issue, we propose a novel abstraction scheme building on successor features. This includes an algorithm for transferring abstract options across different environments and a state abstraction mechanism that allows us to perform efficient planning with the transferred options.

----

## [421] To Trust or Not To Trust Prediction Scores for Membership Inference Attacks

**Authors**: *Dominik Hintersdorf, Lukas Struppek, Kristian Kersting*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/422](https://doi.org/10.24963/ijcai.2022/422)

**Abstract**:

Membership inference attacks (MIAs) aim to determine whether a specific sample was used to train a predictive model. Knowing this may indeed lead to a privacy breach. Most MIAs, however, make use of the model's prediction scores - the probability of each output given some input - following the intuition that the trained model tends to behave differently on its training data. We argue that this is a fallacy for many modern deep network architectures. Consequently, MIAs will miserably fail since overconfidence leads to high false-positive rates not only on known domains but also on out-of-distribution data and implicitly acts as a defense against MIAs. Specifically, using generative adversarial networks, we are able to produce a potentially infinite number of samples falsely classified as part of the training data. In other words, the threat of MIAs is overestimated, and less information is leaked than previously assumed. Moreover, there is actually a trade-off between the overconfidence of models and their susceptibility to MIAs: the more classifiers know when they do not know, making low confidence predictions, the more they reveal the training data.

----

## [422] Leveraging Class Abstraction for Commonsense Reinforcement Learning via Residual Policy Gradient Methods

**Authors**: *Niklas Höpner, Ilaria Tiddi, Herke van Hoof*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/423](https://doi.org/10.24963/ijcai.2022/423)

**Abstract**:

Enabling reinforcement learning (RL) agents to leverage a knowledge base while learning from experience promises to advance RL in knowledge intensive domains. However, it has proven difficult to leverage knowledge that is not manually tailored to the environment. We propose to use the subclass relationships present in open-source knowledge graphs to abstract away from specific objects. We develop a residual policy gradient method that is able to integrate knowledge across different abstraction levels in the class hierarchy. Our method results in improved sample efficiency and generalisation to unseen objects in commonsense games, but we also investigate failure modes, such as excessive noise in the extracted class knowledge or environments with little class structure.

----

## [423] Learning Continuous Graph Structure with Bilevel Programming for Graph Neural Networks

**Authors**: *Minyang Hu, Hong Chang, Bingpeng Ma, Shiguang Shan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/424](https://doi.org/10.24963/ijcai.2022/424)

**Abstract**:

Learning graph structure for graph neural networks (GNNs) is crucial to facilitate the GNN-based downstream learning tasks. It is challenging due to the non-differentiable discrete graph structure and lack of ground-truth. In this paper, we address these problems and propose a novel graph structure learning framework for GNNs. Firstly, we directly model the continuous graph structure with dual-normalization, which implicitly imposes sparse constraint and reduces the influence of noisy edges. Secondly, we formulate the whole training process as a bilevel programming problem, where the inner objective is to optimize the GNNs given learned graphs, while the outer objective is to optimize the graph structure to minimize the generalization error of downstream task. Moreover, for bilevel optimization, we propose an improved Neumann-IFT algorithm to obtain an approximate solution, which is more stable and accurate than existing optimization methods. Besides, it makes the bilevel optimization process memory-efficient and scalable to large graphs. Experiments on node classification and scene graph generation show that our method can outperform related methods, especially with noisy graphs.

----

## [424] SHAPE: An Unified Approach to Evaluate the Contribution and Cooperation of Individual Modalities

**Authors**: *Pengbo Hu, Xingyu Li, Yi Zhou*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/425](https://doi.org/10.24963/ijcai.2022/425)

**Abstract**:

As deep learning advances, there is an ever-growing demand for models capable of synthesizing information from multi-modal resources to address the complex tasks raised from real-life applications. Recently, many large multi-modal datasets have been collected, on which researchers actively explore different methods of fusing multi-modal information. However, little attention has been paid to quantifying the contribution of different modalities within the proposed models. 
In this paper, we propose the SHapley vAlue-based PErceptual (SHAPE) scores that measure the marginal contribution of individual modalities and the degree of cooperation across modalities. Using these scores, we systematically evaluate different fusion methods on different multi-modal datasets for different tasks. Our experiments suggest that for some tasks where different modalities are complementary, the multi-modal models still tend to use the dominant modality alone and ignore the cooperation across modalities. On the other hand, models learn to exploit cross-modal cooperation when different modalities are indispensable for the task. In this case, the scores indicate it is better to fuse different modalities at relatively early stages. We hope our scores can help improve the understanding of how the present multi-modal models operate on different modalities and encourage more sophisticated methods of integrating multiple modalities.

----

## [425] Enhancing Unsupervised Domain Adaptation via Semantic Similarity Constraint for Medical Image Segmentation

**Authors**: *Tao Hu, Shiliang Sun, Jing Zhao, Dongyu Shi*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/426](https://doi.org/10.24963/ijcai.2022/426)

**Abstract**:

This work proposes a novel unsupervised cross-modality adaptive segmentation method for medical images to tackle the performance degradation caused by the severe domain shift when neural networks are being deployed to unseen modalities. The proposed method is an end-2-end framework, which conducts appearance transformation via a domain-shared shallow content encoder and two domain-specific decoders. The feature extracted from the encoder is enhanced to be more domain-invariant by a similarity learning task using the proposed Semantic Similarity Mining (SSM) module which has a strong help of domain adaptation. The domain-invariant latent feature is then fused into the target domain segmentation sub-network, trained using the original target domain images and the translated target images from the source domain in the framework of adversarial training. The adversarial training is effective to narrow the remaining gap between domains in semantic space after appearance alignment. Experimental results on two challenging datasets demonstrate that our method outperforms the state-of-the-art approaches.

----

## [426] Type-aware Embeddings for Multi-Hop Reasoning over Knowledge Graphs

**Authors**: *Zhiwei Hu, Víctor Gutiérrez-Basulto, Zhiliang Xiang, Xiaoli Li, Ru Li, Jeff Z. Pan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/427](https://doi.org/10.24963/ijcai.2022/427)

**Abstract**:

Multi-hop reasoning over real-life knowledge graphs (KGs) is a highly challenging  problem as traditional subgraph matching methods are not capable to deal with noise and missing information. Recently, to address this problem a promising approach based on jointly embedding logical queries and KGs into a low-dimensional space to identify answer entities has emerged. However,  existing proposals ignore critical semantic knowledge inherently available in KGs, such  as  type  information. To  leverage type  information, we  propose a novel type-aware  model, TypE-aware Message Passing (TEMP), which enhances the entity and relation representation in queries, and simultaneously improves generalization, and deductive and inductive reasoning. Remarkably, TEMP is a plug-and-play model that can be easily incorporated into existing embedding-based models to improve their performance. Extensive experiments on three real-world datasets demonstrate TEMPâ€™s effectiveness.

----

## [427] Reconstructing Diffusion Networks from Incomplete Data

**Authors**: *Hao Huang, Keqi Han, Beicheng Xu, Ting Gan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/428](https://doi.org/10.24963/ijcai.2022/428)

**Abstract**:

To reconstruct the topology of a diffusion network, existing approaches customarily demand not only eventual infection statuses of nodes, but also the exact times when infections occur. In real-world settings, such as the spread of epidemics, tracing the exact infection times is often infeasible; even obtaining the eventual infection statuses of all nodes is a challenging task. In this work, we study topology reconstruction of a diffusion network with incomplete observations of the node infection statuses. To this end, we iteratively infer the network topology based on observed infection statuses and estimated values for unobserved infection statuses by investigating the correlation of node infections, and learn the most probable probabilities of the infection propagations among nodes w.r.t. current inferred topology, as well as the corresponding probability distribution of each unobserved infection status, which in turn helps update the estimate of unobserved data. Extensive experimental results on both synthetic and real-world networks verify the effectiveness and efficiency of our approach.

----

## [428] FLS: A New Local Search Algorithm for K-means with Smaller Search Space

**Authors**: *Junyu Huang, Qilong Feng, Ziyun Huang, Jinhui Xu, Jianxin Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/429](https://doi.org/10.24963/ijcai.2022/429)

**Abstract**:

The k-means problem is an extensively studied unsupervised learning problem with various applications in decision making and data mining. In this paper, we propose a fast and practical local search algorithm for the k-means problem. Our method reduces the search space of swap pairs from O(nk) to O(k^2), and applies random mutations to find potentially better solutions when local search falls into poor local optimum. With the assumption of data distribution that each optimal cluster has "average" size of \Omega(n/k), which is common in many datasets and k-means benchmarks, we prove that our proposed algorithm gives a (100+\epsilon)-approximate solution in expectation. Empirical experiments show that our algorithm achieves better performance compared to existing state-of-the-art local search methods on k-means benchmarks and large datasets.

----

## [429] Robust Reinforcement Learning as a Stackelberg Game via Adaptively-Regularized Adversarial Training

**Authors**: *Peide Huang, Mengdi Xu, Fei Fang, Ding Zhao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/430](https://doi.org/10.24963/ijcai.2022/430)

**Abstract**:

Robust Reinforcement Learning (RL) focuses on improving performances under model errors or adversarial attacks, which facilitates the real-life deployment of RL agents. Robust Adversarial Reinforcement Learning (RARL) is one of the most popular frameworks for robust RL. However, most of the existing literature models RARL as a zero-sum simultaneous game with Nash equilibrium as the solution concept, which could overlook the sequential nature of RL deployments, produce overly conservative agents, and induce training instability. In this paper, we introduce a novel hierarchical formulation of robust RL -- a general-sum Stackelberg game model called RRL-Stack -- to formalize the sequential nature and provide extra flexibility for robust training. We develop the Stackelberg Policy Gradient algorithm to solve RRL-Stack, leveraging the Stackelberg learning dynamics by considering the adversary's response. Our method generates challenging yet solvable adversarial environments which benefit RL agents' robust learning. Our algorithm demonstrates better training stability and robustness against different testing conditions in the single-agent robotics control and multi-agent highway merging tasks.

----

## [430] On the Channel Pruning using Graph Convolution Network for Convolutional Neural Network Acceleration

**Authors**: *Di Jiang, Yuan Cao, Qiang Yang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/431](https://doi.org/10.24963/ijcai.2022/431)

**Abstract**:

Network pruning is considered efficient for sparsification and acceleration of Convolutional Neural Network (CNN) based models that can be adopted in re-source-constrained environments. Inspired by two popular pruning criteria, i.e. magnitude and similarity, this paper proposes a novel structural pruning method based on Graph Convolution Network (GCN) to further promote compression performance. The channel features are firstly extracted by Global Average Pooling (GAP) from a batch of samples, and a graph model for each layer is generated based on the similarity of features. A set of agents for individual CNN layers are implemented by GCN and utilized to synthesize comprehensive channel information and determine the pruning scheme for the overall CNN model. The training process of each agent is carried out using Reinforcement Learning (RL) to ensure their convergence and adaptability to various network architectures. The proposed solution is assessed based on a range of image classification datasets i.e., CIFAR and Tiny-ImageNet. The numerical results indicate that the proposed pruning method outperforms the pure magnitude-based or similarity-based pruning solutions and other SOTA methods (e.g., HRank and SCP). For example, the proposed method can prune VGG16 by removing 93% of the model parameters without any accuracy reduction in the CIFAR10 dataset.

----

## [431] Graph Masked Autoencoder Enhanced Predictor for Neural Architecture Search

**Authors**: *Kun Jing, Jungang Xu, Pengfei Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/432](https://doi.org/10.24963/ijcai.2022/432)

**Abstract**:

Performance estimation of neural architecture is a crucial component of neural architecture search (NAS). Meanwhile, neural predictor is a current mainstream performance estimation method. However, it is a challenging task to train the predictor with few architecture evaluations for efficient NAS. In this paper, we propose a graph masked autoencoder (GMAE) enhanced predictor, which can reduce the dependence on supervision data by self-supervised pre-training with untrained architectures. We compare our GMAE-enhanced predictor with existing predictors in different search spaces, and experimental results show that our predictor has high query utilization. Moreover, GMAE-enhanced predictor with different search strategies can discover competitive architectures in different search spaces. Code and supplementary materials are available at https://github.com/kunjing96/GMAENAS.git.

----

## [432] Online Evasion Attacks on Recurrent Models: The Power of Hallucinating the Future

**Authors**: *Byunggill Joe, Insik Shin, Jihun Hamm*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/433](https://doi.org/10.24963/ijcai.2022/433)

**Abstract**:

Recurrent models are frequently being used in online tasks such as autonomous driving, and a comprehensive study of their vulnerability is called for. Existing research is limited in generality only addressing application-specific vulnerability or making implausible assumptions such as the knowledge of future input. In this paper, we present a general attack framework for online tasks incorporating the unique constraints of the online setting different from offline tasks. Our framework is versatile in that it covers time-varying adversarial objectives and various optimization constraints, allowing for a comprehensive study of robustness. Using the framework, we also present a novel white-box attack called Predictive Attack that `hallucinates' the future. The attack achieves 98 percent of the performance of the ideal but infeasible clairvoyant attack on average. We validate the effectiveness of the proposed framework and attacks through various experiments.

----

## [433] Set Interdependence Transformer: Set-to-Sequence Neural Networks for Permutation Learning and Structure Prediction

**Authors**: *Mateusz Jurewicz, Leon Derczynski*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/434](https://doi.org/10.24963/ijcai.2022/434)

**Abstract**:

The task of learning to map an input set onto a permuted sequence of its elements is challenging for neural networks. Set-to-sequence problems occur in natural language processing, computer vision and structure prediction, where interactions between elements of large sets define the optimal output. Models must exhibit relational reasoning, handle varying cardinalities and manage combinatorial complexity. Previous attention-based methods require n layers of their set transformations to explicitly represent n-th order relations. Our aim is to enhance their ability to efficiently model higher-order interactions through an additional interdependence component. We propose a novel neural set encoding method called the Set Interdependence Transformer, capable of relating the set's permutation invariant representation to its elements within sets of any cardinality. We combine it with a permutation learning module into a complete, 3-part set-to-sequence model and demonstrate its state-of-the-art performance on a number of tasks. These range from combinatorial optimization problems, through permutation learning challenges on both synthetic and established NLP datasets for sentence ordering, to a novel domain of product catalog structure prediction. Additionally, the network's ability to generalize to unseen sequence lengths is investigated and a comparative empirical analysis of the existing methods' ability to learn higher-order interactions is provided.

----

## [434] Relational Abstractions for Generalized Reinforcement Learning on Symbolic Problems

**Authors**: *Rushang Karia, Siddharth Srivastava*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/435](https://doi.org/10.24963/ijcai.2022/435)

**Abstract**:

Reinforcement learning in problems with symbolic state spaces is challenging due to the need for reasoning over long horizons. This paper presents a new approach that utilizes relational abstractions in conjunction with deep learning to learn a generalizable Q-function for such problems. The learned Q-function can be efficiently transferred to related problems that have different object names and object quantities, and thus, entirely different state spaces. We show that the learned, generalized Q-function can be utilized for zero-shot transfer to related problems without an explicit, hand-coded curriculum. Empirical evaluations on a range of problems show that our method facilitates efficient zero-shot transfer of learned knowledge to much larger problem instances containing many objects.

----

## [435] Data Augmentation for Learning to Play in Text-Based Games

**Authors**: *Jinhyeon Kim, Kee-Eung Kim*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/436](https://doi.org/10.24963/ijcai.2022/436)

**Abstract**:

Improving generalization in text-based games serves as a useful stepping-stone towards reinforcement learning (RL) agents with generic linguistic ability. Data augmentation for generalization in RL has shown to be very successful in classic control and visual tasks, but there is no prior work for text-based games. We propose Transition-Matching Permutation, a novel data augmentation technique for text-based games, where we identify phrase permutations that match as many transitions in the trajectory data. We show that applying this technique results in state-of-the-art performance in the Cooking Game benchmark suite for text-based games.

----

## [436] Self-Predictive Dynamics for Generalization of Vision-based Reinforcement Learning

**Authors**: *Kyungsoo Kim, Jeongsoo Ha, Yusung Kim*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/437](https://doi.org/10.24963/ijcai.2022/437)

**Abstract**:

Vision-based reinforcement learning requires efficient and robust representations of image-based observations, especially when the images contain distracting (task-irrelevant) elements such as shadows, clouds, and light. It becomes more important if those distractions are not exposed during training. We design a Self-Predictive Dynamics (SPD) method to extract task-relevant features efficiently, even in unseen observations after training. SPD uses weak and strong augmentations in parallel, and learns representations by predicting inverse and forward transitions across the two-way augmented versions. In a set of MuJoCo visual control tasks and an autonomous driving task (CARLA), SPD outperforms previous studies in complex observations, and significantly improves the generalization performance for unseen observations. Our code is available at https://github.com/unigary/SPD.

----

## [437] DyGRAIN: An Incremental Learning Framework for Dynamic Graphs

**Authors**: *Seoyoon Kim, Seongjun Yun, Jaewoo Kang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/438](https://doi.org/10.24963/ijcai.2022/438)

**Abstract**:

Graph-structured data provide a powerful representation of complex relations or interactions. Many variants of graph neural networks (GNNs) have emerged to learn graph-structured data where underlying graphs are static, although graphs in various real-world applications are dynamic (e.g., evolving structure). To consider the dynamic nature that a graph changes over time, the need for applying incremental learning (i.e., continual learning or lifelong learning) to the graph domain has been emphasized. However, unlike incremental learning on Euclidean data, graph-structured data contains dependency between the existing nodes and newly appeared nodes, resulting in the phenomenon that receptive fields of existing nodes vary by new inputs (e.g., nodes and edges). In this paper, we raise a crucial challenge of incremental learning for dynamic graphs as time-varying receptive fields, and propose a novel incremental learning framework, DyGRAIN, to mitigate time-varying receptive fields and catastrophic forgetting. Specifically, our proposed method incrementally learns dynamic graph representations by reflecting the influential change in receptive fields of existing nodes and maintaining previous knowledge of informational nodes prone to be forgotten. Our experiments on large-scale graph datasets demonstrate that our proposed method improves the performance by effectively capturing pivotal nodes and preventing catastrophic forgetting.

----

## [438] Thompson Sampling for Bandit Learning in Matching Markets

**Authors**: *Fang Kong, Junming Yin, Shuai Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/439](https://doi.org/10.24963/ijcai.2022/439)

**Abstract**:

The problem of two-sided matching markets has a wide range of real-world applications and has been extensively studied in the literature. A line of recent works have focused on the problem setting where the preferences of one-side market participants are unknown a priori and are learned by iteratively interacting with the other side of participants. All these works are based on explore-then-commit (ETC) and upper confidence bound (UCB) algorithms, two common strategies in multi-armed bandits (MAB). Thompson sampling (TS) is another popular approach, which attracts lots of attention due to its easier implementation and better empirical performances. In many problems, even when UCB and ETC-type algorithms have already been analyzed, researchers are still trying to study TS for its benefits. However, the convergence analysis of TS is much more challenging and remains open in many problem settings. In this paper, we provide the first regret analysis for TS in the new setting of iterative matching markets. Extensive experiments demonstrate the practical advantages of the TS-type algorithm over the ETC and UCB-type baselines.

----

## [439] Multi-policy Grounding and Ensemble Policy Learning for Transfer Learning with Dynamics Mismatch

**Authors**: *Hyun-Rok Lee, Ram Ananth Sreenivasan, Yeonjeong Jeong, Jongseong Jang, Dongsub Shim, Chi-Guhn Lee*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/440](https://doi.org/10.24963/ijcai.2022/440)

**Abstract**:

We propose a new transfer learning algorithm between tasks with different dynamics. The proposed algorithm solves an Imitation from Observation problem (IfO) to ground the source environment to the target task before learning an optimal policy in the grounded environment. The learned policy is deployed in the target task without additional training. A particular feature of our algorithm is the employment of multiple rollout policies during training with a goal to ground the environment more globally; hence, it is named as Multi-Policy Grounding (MPG). The quality of final policy is further enhanced via ensemble policy learning. We demonstrate the superiority of the proposed algorithm analytically and numerically. Numerical studies show that the proposed multi-policy approach allows comparable grounding with single policy approach with a fraction of target samples, hence the algorithm is able to maintain the quality of obtained policy even as the number of interactions with the target environment becomes extremely small.

----

## [440] Pseudo-spherical Knowledge Distillation

**Authors**: *Kyungmin Lee, Hyeongkeun Lee*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/441](https://doi.org/10.24963/ijcai.2022/441)

**Abstract**:

Knowledge distillation aims to transfer the information by minimizing the cross-entropy between the probabilistic outputs of the teacher and student network.
In this work, we propose an alternative distillation objective by maximizing the scoring rule, which quantitatively measures the agreement of a distribution to the reference distribution. 
We demonstrate that the proper and homogeneous scoring rule exhibits more preferable properties for distillation than the original cross entropy based approach. 
To that end, we present an efficient implementation of the distillation objective based on a pseudo-spherical scoring rule, which is a family of proper and homogeneous scoring rules. We refer to it as pseudo-spherical knowledge distillation. 
Through experiments on various model compression tasks, we validate the effectiveness of our method by showing its superiority over the original knowledge distillation. 
Moreover, together with structural distillation methods such as contrastive representation distillation, we achieve state of the art results in CIFAR100 benchmarks.

----

## [441] Libra-CAM: An Activation-Based Attribution Based on the Linear Approximation of Deep Neural Nets and Threshold Calibration

**Authors**: *Sangkyun Lee, Sungmin Han*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/442](https://doi.org/10.24963/ijcai.2022/442)

**Abstract**:

Universal application of AI has increased the need to explain why an AI model makes a specific decision in a human-understandable form. Among many related works, the class activation map (CAM)-based methods have been successful recently, creating input attribution based on the weighted sum of activation maps in convolutional neural networks. However, existing methods use channel-wise importance weights with specific architectural assumptions, relying on arbitrarily chosen attribution threshold values in their quality assessment: we think these can degrade the quality of attribution. In this paper, we propose Libra-CAM, a new CAM-style attribution method based on the best linear approximation of the layer (as a function) between the penultimate activation and the target-class score output. From the approximation, we derive the base formula of Libra-CAM, which is applied with multiple reference activations from a pre-built library. We construct Libra-CAM by averaging these base attribution maps, taking a threshold calibration procedure to optimize its attribution quality. Our experiments show that Libra-CAM can be computed in a reasonable time and is superior to the existing attribution methods in quantitative and qualitative attribution quality evaluations.

----

## [442] SGAT: Simplicial Graph Attention Network

**Authors**: *See Hian Lee, Feng Ji, Wee Peng Tay*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/443](https://doi.org/10.24963/ijcai.2022/443)

**Abstract**:

Heterogeneous graphs have multiple node and edge types and are semantically richer than homogeneous graphs. To learn such complex semantics, many graph neural network approaches for heterogeneous graphs use metapaths to capture multi-hop interactions between nodes. Typically, features from non-target nodes are not incorporated into the learning procedure. However, there can be nonlinear, high-order interactions involving multiple nodes or edges. In this paper, we present Simplicial Graph Attention Network (SGAT), a simplicial complex approach to represent such high-order interactions by placing features from non-target nodes on the simplices. We then use attention mechanisms and upper adjacencies to generate representations. We empirically demonstrate the efficacy of our approach with node classification tasks on heterogeneous graph datasets and further show SGAT's ability in extracting structural information by employing random node features. Numerical experiments indicate that SGAT performs better than other current state-of-the-art heterogeneous graph learning methods.

----

## [443] Learning General Gaussian Mixture Model with Integral Cosine Similarity

**Authors**: *Guanglin Li, Bin Li, Changsheng Chen, Shunquan Tan, Guoping Qiu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/444](https://doi.org/10.24963/ijcai.2022/444)

**Abstract**:

Gaussian mixture model (GMM) is a powerful statistical tool in data modeling, especially for unsupervised learning tasks. Traditional learning methods for GMM such as expectation maximization (EM) require the covariance of the Gaussian components to be non-singular, a condition that is often not satisfied in real-world applications. This paper presents a new learning method called G$^2$M$^2$ (General Gaussian Mixture Model) by fitting an unnormalized Gaussian mixture function (UGMF) to a data distribution. At the core of G$^2$M$^2$ is the introduction of an integral cosine similarity (ICS) function for comparing the UGMF and the unknown data density distribution without having to explicitly estimate it. By maximizing the ICS through Monte Carlo sampling, the UGMF can be made to overlap with the unknown data density distribution such that the two only differ by a constant scalar, and the UGMF can be normalized to obtain the data density distribution. A Siamese convolutional neural network is also designed for optimizing the ICS function. Experimental results show that our method is more competitive in modeling data having correlations that may lead to singular covariance matrices in GMM, and it outperforms state-of-the-art methods in unsupervised anomaly detection.

----

## [444] Ridgeless Regression with Random Features

**Authors**: *Jian Li, Yong Liu, Yingying Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/445](https://doi.org/10.24963/ijcai.2022/445)

**Abstract**:

Recent theoretical studies illustrated that kernel ridgeless regression can guarantee good generalization ability without an explicit regularization. In this paper, we investigate the statistical properties of ridgeless regression with random features and stochastic gradient descent. We explore the effect of factors in the stochastic gradient and random features, respectively. Specifically, random features error exhibits the double-descent curve. Motivated by the theoretical findings, we propose a tunable kernel algorithm that optimizes the spectral density of kernel during training. Our work bridges the interpolation theory and practical algorithm.

----

## [445] Learning from Students: Online Contrastive Distillation Network for General Continual Learning

**Authors**: *Jin Li, Zhong Ji, Gang Wang, Qiang Wang, Feng Gao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/446](https://doi.org/10.24963/ijcai.2022/446)

**Abstract**:

The goal of General Continual Learning (GCL) is to preserve learned knowledge and learn new knowledge with constant memory from an infinite data stream where task boundaries are blurry. Distilling the model's response of reserved samples between the old and the new models is an effective way to achieve promise performance on GCL. However, it accumulates the inherent old model's response bias and is not robust to model changes. To this end, we propose an Online Contrastive Distillation Network (OCD-Net) to tackle these problems, which explores the merit of the student model in each time step to guide the training process of the student model. Concretely, the teacher model is devised to help the student model to consolidate the learned knowledge, which is trained online via integrating the model weights of the student model to accumulate the new knowledge. Moreover, our OCD-Net incorporates both relation and adaptive response to help the student model alleviate the catastrophic forgetting, which is also beneficial for the teacher model preserves the learned knowledge. Extensive experiments on six benchmark datasets demonstrate that our proposed OCD-Net significantly outperforms state-of-the-art approaches in 3.26%~8.71% with various buffer sizes. Our code is available at https://github.com/lijincm/OCD-Net.

----

## [446] Cross-modal Representation Learning and Relation Reasoning for Bidirectional Adaptive Manipulation

**Authors**: *Lei Li, Kai Fan, Chun Yuan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/447](https://doi.org/10.24963/ijcai.2022/447)

**Abstract**:

Since single-modal controllable manipulation typically requires supervision of information from other modalities or cooperation with complex software and experts, this paper addresses the problem of cross-modal adaptive manipulation (CAM). The novel task performs cross-modal semantic alignment from mutual supervision and implements bidirectional exchange of attributes, relations, or objects in parallel, benefiting both modalities while significantly reducing manual effort. We introduce a robust solution for CAM, which includes two essential modules, namely Heterogeneous Representation Learning (HRL) and Cross-modal Relation Reasoning (CRR). The former is designed to perform representation learning for cross-modal semantic alignment on heterogeneous graph nodes. The latter is adopted to identify and exchange the focused attributes, relations, or objects in both modalities. Our method produces pleasing cross-modal outputs on CUB and Visual Genome.

----

## [447] Neural PCA for Flow-Based Representation Learning

**Authors**: *Shen Li, Bryan Hooi*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/448](https://doi.org/10.24963/ijcai.2022/448)

**Abstract**:

Of particular interest is to discover useful representations solely from observations in an unsupervised generative manner. However, the question of whether existing normalizing flows provide effective representations for downstream tasks remains mostly unanswered despite their strong ability for sample generation and density estimation. This paper investigates this problem for such a family of generative models that admits exact invertibility. We propose Neural Principal Component Analysis (Neural-PCA) that operates in full dimensionality while capturing principal components in descending order. Without exploiting any label information, the principal components recovered store the most informative elements in their leading dimensions and leave the negligible in the trailing ones, allowing for clear performance improvements of 5%-10% in downstream tasks. Such improvements are empirically found consistent irrespective of the number of latent trailing dimensions dropped. Our work suggests that necessary inductive bias be introduced into generative modeling when representation quality is of interest.

----

## [448] Pruning-as-Search: Efficient Neural Architecture Search via Channel Pruning and Structural Reparameterization

**Authors**: *Yanyu Li, Pu Zhao, Geng Yuan, Xue Lin, Yanzhi Wang, Xin Chen*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/449](https://doi.org/10.24963/ijcai.2022/449)

**Abstract**:

Neural architecture search (NAS) and network pruning are widely studied efficient AI techniques, but not yet perfect.
NAS performs exhaustive candidate architecture search, incurring tremendous search cost.
Though (structured) pruning can simply shrink model dimension, it remains unclear how to decide the per-layer sparsity automatically and optimally.
In this work, we revisit the problem of layer-width optimization and propose Pruning-as-Search (PaS), an end-to-end channel pruning method to  search out desired sub-network automatically and efficiently.
Specifically, we add a depth-wise binary convolution to learn pruning policies directly through gradient descent.
By combining the structural reparameterization and PaS, we successfully searched out a new family of VGG-like and lightweight networks, which enable the flexibility of arbitrary width with respect to each layer instead of each stage.
Experimental results show that our proposed architecture outperforms prior arts by around 1.0% top-1 accuracy under similar inference speed on ImageNet-1000 classification task.
Furthermore, we demonstrate the effectiveness of our width search on complex tasks including instance segmentation and image translation.
Code and models are released.

----

## [449] Rethinking the Setting of Semi-supervised Learning on Graphs

**Authors**: *Ziang Li, Ming Ding, Weikai Li, Zihan Wang, Ziyu Zeng, Yukuo Cen, Jie Tang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/450](https://doi.org/10.24963/ijcai.2022/450)

**Abstract**:

We argue that the present setting of semisupervised learning on graphs may result in unfair comparisons, due to its potential risk of over-tuning hyper-parameters for models. In this paper, we highlight the significant influence of tuning hyper-parameters, which leverages the label information in the validation set to improve the performance. To explore the limit of over-tuning hyperparameters, we propose ValidUtil, an approach to fully utilize the label information in the validation set through an extra group of hyper-parameters. With ValidUtil, even GCN can easily get high accuracy of 85.8% on Cora.
To avoid over-tuning, we merge the training set and the validation set and construct an i.i.d. graph benchmark (IGB) consisting of 4 datasets. Each dataset contains 100 i.i.d. graphs sampled from a large graph to reduce the evaluation variance. Our experiments suggest that IGB is a more stable benchmark than previous datasets for semisupervised learning on graphs. Our code and data are released at https://github.com/THUDM/IGB/.

----

## [450] Contrastive Multi-view Hyperbolic Hierarchical Clustering

**Authors**: *Fangfei Lin, Bing Bai, Kun Bai, Yazhou Ren, Peng Zhao, Zenglin Xu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/451](https://doi.org/10.24963/ijcai.2022/451)

**Abstract**:

Hierarchical clustering recursively partitions data at an increasingly finer granularity. In real-world applications, multi-view data have become increasingly important. This raises a less investigated problem, i.e., multi-view hierarchical clustering, to better understand the hierarchical structure of multi-view data. To this end, we propose a novel neural network-based model, namely Contrastive Multi-view Hyperbolic Hierarchical Clustering(CMHHC). It consists of three components, i.e., multi-view alignment learning, aligned feature similarity learning, and continuous hyperbolic hierarchical clustering. First, we align sample-level representations across multiple views in a contrastive way to capture the view-invariance information. Next, we utilize both the manifold and Euclidean similarities to improve the metric property. Then, we embed the representations into a hyperbolic space and optimize the hyperbolic embeddings via a continuous relaxation of hierarchical clustering loss. Finally, a binary clustering tree is decoded from optimized hyperbolic embeddings. Experimental results on five real-world datasets demonstrate the effectiveness of the proposed method and its components.

----

## [451] JueWu-MC: Playing Minecraft with Sample-efficient Hierarchical Reinforcement Learning

**Authors**: *Zichuan Lin, Junyou Li, Jianing Shi, Deheng Ye, Qiang Fu, Wei Yang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/452](https://doi.org/10.24963/ijcai.2022/452)

**Abstract**:

Learning rational behaviors in open-world games like Minecraft remains to be challenging for Reinforcement Learning (RL) research due to the compound challenge of partial observability, high-dimensional visual perception and delayed reward. To address this, we propose JueWu-MC, a sample-efficient hierarchical RL approach equipped with representation learning and imitation learning to deal with perception and exploration. Specifically, our approach includes two levels of hierarchy, where the high-level controller learns a policy to control over options and the low-level workers learn to solve each sub-task. To boost the learning of sub-tasks, we propose a combination of techniques including 1) action-aware representation learning which captures underlying relations between action and representation, 2) discriminator-based self-imitation learning for efficient exploration, and 3) ensemble behavior cloning with consistency filtering for policy robustness. Extensive experiments show that JueWu-MC significantly improves sample efficiency and outperforms a set of baselines by a large margin. Notably, we won the championship of the NeurIPS MineRL 2021 research competition and achieved the highest performance score ever.

----

## [452] Declaration-based Prompt Tuning for Visual Question Answering

**Authors**: *Yuhang Liu, Wei Wei, Daowan Peng, Feida Zhu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/453](https://doi.org/10.24963/ijcai.2022/453)

**Abstract**:

In recent years, the pre-training-then-fine-tuning paradigm has yielded immense success on a wide spectrum of cross-modal tasks, such as visual question answering (VQA), in which a visual-language (VL) model is first optimized via self-supervised task objectives, e.g., masked language modeling (MLM) and image-text matching (ITM), and then fine-tuned to adapt to downstream task (e.g., VQA) via a brand-new objective function, e.g., answer prediction. However, the inconsistency of the objective forms not only severely limits the generalization of pre-trained VL models to downstream tasks, but also requires a large amount of labeled data for fine-tuning. To alleviate the problem, we propose an innovative VL fine-tuning paradigm (named Declaration-based Prompt Tuning, abbreviated as DPT), which fine-tunes the model for downstream VQA using the pre-training objectives, boosting the effective adaptation of pre-trained
models to the downstream task. Specifically, DPT reformulates the VQA task via (1) textual adaptation, which converts the given questions into declarative sentence form for prompt-tuning, and (2) task adaptation, which optimizes the objective function of VQA problem in the manner of pre-training phase. Experimental results on GQA dataset show that DPT outperforms the fine-tuned counterpart by a large margin regarding accuracy in both fully-supervised (2.68%) and zero-shot/fewshot (over 31%) settings. All the data and codes will be available to facilitate future research.

----

## [453] Projected Gradient Descent Algorithms for Solving Nonlinear Inverse Problems with Generative Priors

**Authors**: *Zhaoqiang Liu, Jun Han*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/454](https://doi.org/10.24963/ijcai.2022/454)

**Abstract**:

In this paper, we propose projected gradient descent (PGD) algorithms for signal estimation from noisy nonlinear measurements. We assume that the unknown signal lies near the range of a Lipschitz continuous generative model with bounded inputs. In particular, we consider two cases when the nonlinear link function is either unknown or known. For unknown nonlinearity, we make the assumption of sub-Gaussian observations and propose a linear least-squares estimator. We show that when there is no representation error, the sensing vectors are Gaussian, and the number of samples is sufficiently large, with high probability, a PGD algorithm converges linearly to a point achieving the optimal statistical rate using arbitrary initialization. For known nonlinearity, we assume monotonicity, and make much weaker assumptions on the sensing vectors and allow for representation error. We propose a nonlinear least-squares estimator that is guaranteed to enjoy an optimal statistical rate. A corresponding PGD algorithm is provided and is shown to also converge linearly to the estimator using arbitrary initialization. In addition, we present experimental results on image datasets to demonstrate the performance of our PGD algorithms.

----

## [454] SELC: Self-Ensemble Label Correction Improves Learning with Noisy Labels

**Authors**: *Yangdi Lu, Wenbo He*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/455](https://doi.org/10.24963/ijcai.2022/455)

**Abstract**:

Deep neural networks are prone to overfitting noisy labels, resulting in poor generalization performance. To overcome this problem, we present a simple and effective method self-ensemble label correction (SELC) to progressively correct noisy labels and refine the model. We look deeper into the memorization behavior in training with noisy labels and observe that the network outputs are reliable in the early stage. To retain this reliable knowledge, SELC uses ensemble predictions formed by an exponential moving average of network outputs to update the original noisy labels. We show that training with SELC refines the model by gradually reducing supervision from noisy labels and increasing supervision from ensemble predictions. Despite its simplicity, compared with many state-of-the-art methods, SELC obtains more promising and stable results in the presence of class-conditional, instance-dependent, and real-world label noise. The code is available at https://github.com/MacLLL/SELC.

----

## [455] Exploring Binary Classification Hidden within Partial Label Learning

**Authors**: *Hengheng Luo, Yabin Zhang, Suyun Zhao, Hong Chen, Cuiping Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/456](https://doi.org/10.24963/ijcai.2022/456)

**Abstract**:

Partial label learning (PLL) is to learn a discriminative model under incomplete supervision, where each instance is annotated with a candidate label set. The basic principle of PLL is that the unknown correct label y of an instance x resides in its candidate label set s, i.e., P(y ∈ s | x) = 1. On which basis, current researches either directly model P(x | y) under different data generation assumptions or propose various surrogate multiclass losses, which all aim to encourage the model-based Pθ(y ∈ s | x)→1 implicitly. In this work, instead, we explicitly construct a binary classification task toward P(y ∈ s | x) based on the discriminative model, that is to predict whether the model-output label of x is one of its candidate labels. We formulate a novel risk estimator with estimation error bound for the proposed PLL binary classification risk. By applying logit adjustment based on disambiguation strategy, the practical approach directly maximizes Pθ(y ∈ s | x) while implicitly disambiguating the correct one from candidate labels simultaneously. Thorough experiments validate that the proposed approach achieves competitive performance against the state-of-the-art PLL methods.

----

## [456] Teaching LTLf Satisfiability Checking to Neural Networks

**Authors**: *Weilin Luo, Hai Wan, Jianfeng Du, Xiaoda Li, Yuze Fu, Rongzhen Ye, Delong Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/457](https://doi.org/10.24963/ijcai.2022/457)

**Abstract**:

Linear temporal logic over finite traces (LTLf) satisfiability checking is a fundamental and hard (PSPACE-complete) problem in the artificial intelligence community. We explore teaching end-to-end neural networks to check satisfiability in polynomial time. It is a challenge to characterize the syntactic and semantic features of LTLf via neural networks. To tackle this challenge, we propose LTLfNet, a recursive neural network that captures syntactic features of LTLf by recursively combining the embeddings of sub-formulae. LTLfNet models permutation invariance and sequentiality in the semantics of LTLf through different aggregation mechanisms of sub-formulae. Experimental results demonstrate that LTLfNet achieves good performance in synthetic datasets and generalizes across large-scale datasets. They also show that LTLfNet is competitive with state-of-the-art symbolic approaches such as nuXmv and CDLSC.

----

## [457] Towards Robust Unsupervised Disentanglement of Sequential Data - A Case Study Using Music Audio

**Authors**: *Yin-Jyun Luo, Sebastian Ewert, Simon Dixon*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/458](https://doi.org/10.24963/ijcai.2022/458)

**Abstract**:

Disentangled sequential autoencoders (DSAEs) represent a class of probabilistic graphical models that describes an observed sequence with dynamic latent variables and a static latent variable. The former encode information at a frame rate identical to the observation, while the latter globally governs the entire sequence. This introduces an inductive bias and facilitates unsupervised disentanglement of the underlying local and global factors. In this paper, we show that the vanilla DSAE suffers from being sensitive to the choice of model architecture and capacity of the dynamic latent variables, and is prone to collapse the static latent variable. As a countermeasure, we propose TS-DSAE, a two-stage training framework that first learns sequence-level prior distributions, which are subsequently employed to regularise the model and facilitate auxiliary objectives to promote disentanglement. The proposed framework is fully unsupervised and robust against the global factor collapse problem across a wide range of model configurations. It also avoids typical solutions such as adversarial training which usually involves laborious parameter tuning, and domain-specific data augmentation. We conduct quantitative and qualitative evaluations to demonstrate its robustness in terms of disentanglement on both artificial and real-world music audio datasets.

----

## [458] Deep Graph Matching for Partial Label Learning

**Authors**: *Gengyu Lyu, Yanan Wu, Songhe Feng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/459](https://doi.org/10.24963/ijcai.2022/459)

**Abstract**:

Partial Label Learning (PLL) aims to learn from training data where each instance is associated with a set of candidate labels, among which only one is correct. In this paper, we formulate the task of PLL problem as an ``instance-label'' matching selection problem, and propose a DeepGNN-based graph matching PLL approach to solve it. Specifically, we first construct all instances and labels as graph nodes into two different graphs respectively, and then integrate them into a unified matching graph by connecting each instance to its candidate labels. Afterwards, the graph attention mechanism is adopted to aggregate and update all nodes state on the instance graph to form structural representations for each instance. Finally, each candidate label is embedded into its corresponding instance and derives a matching affinity score for each instance-label correspondence with a progressive cross-entropy loss. Extensive experiments on various data sets have demonstrated the superiority of our proposed method.

----

## [459] Locally Normalized Soft Contrastive Clustering for Compact Clusters

**Authors**: *Xin Ma, Won Hwa Kim*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/460](https://doi.org/10.24963/ijcai.2022/460)

**Abstract**:

Recent deep clustering algorithms take advantage of self-supervised learning and self-training techniques to map the original data into a latent space, where the data embedding and clustering assignment can be jointly optimized. However, as many recent datasets are enormous and noisy, getting a clear boundary between different clusters is challenging with existing methods that mainly focus on contracting similar samples together and overlooking samples near boundary of clusters in the latent space. In this regard, we propose an end-to-end deep clustering algorithm, i.e., Locally Normalized Soft Contrastive Clustering (LNSCC). It takes advantage of similarities among each sample's local neighborhood and globally disconnected samples to leverage positiveness and negativeness of sample pairs in a contrastive way to separate different clusters. Experimental results on various datasets illustrate that our proposed approach achieves outstanding clustering performance over most of the state-of-the-art clustering methods for both image and non-image data even without convolution.

----

## [460] Game Redesign in No-regret Game Playing

**Authors**: *Yuzhe Ma, Young Wu, Xiaojin Zhu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/461](https://doi.org/10.24963/ijcai.2022/461)

**Abstract**:

We study the game redesign problem in which an external designer has the ability to change the payoff function in each round, but incurs a design cost for deviating from the original game. The players apply no-regret learning algorithms to repeatedly play the changed games with limited feedback. The goals of the designer are to (i) incentivize players to take a specific target action profile frequently; (ii) incur small cumulative design cost. We present game redesign algorithms with the guarantee that the target action profile is played in T-o(T) rounds while incurring only o(T) cumulative design cost. Simulations on four classic games confirm the ef- fectiveness of our proposed redesign algorithms.

----

## [461] COMET Flows: Towards Generative Modeling of Multivariate Extremes and Tail Dependence

**Authors**: *Andrew McDonald, Pang-Ning Tan, Lifeng Luo*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/462](https://doi.org/10.24963/ijcai.2022/462)

**Abstract**:

Normalizing flows—a popular class of deep generative models—often fail to represent extreme phenomena observed in real-world processes. In particular, existing normalizing flow architectures struggle to model multivariate extremes, characterized by heavy-tailed marginal distributions and asymmetric tail dependence among variables. In light of this shortcoming, we propose COMET (COpula Multivariate ExTreme) Flows, which decompose the process of modeling a joint distribution into two parts: (i) modeling its marginal distributions, and (ii) modeling its copula distribution. COMET Flows capture heavy-tailed marginal distributions by combining a parametric tail belief at extreme quantiles of the marginals with an empirical kernel density function at mid-quantiles. In addition, COMET Flows capture asymmetric tail dependence among multivariate extremes by viewing such dependence as inducing a low-dimensional manifold structure in feature space. Experimental results on both synthetic and real-world datasets demonstrate the effectiveness of COMET flows in capturing both heavy-tailed marginals and asymmetric tail dependence compared to other state-of-the-art baseline architectures. All code is available at https://github.com/andrewmcdonald27/COMETFlows.

----

## [462] Tessellation-Filtering ReLU Neural Networks

**Authors**: *Bernhard Alois Moser, Michal Lewandowski, Somayeh Kargaran, Werner Zellinger, Battista Biggio, Christoph Koutschan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/463](https://doi.org/10.24963/ijcai.2022/463)

**Abstract**:

We identify tessellation-filtering  ReLU  neural networks that, when composed with another ReLU network, keep its non-redundant tessellation unchanged or reduce it.The additional network complexity modifies the shape  of the decision surface without increasing the number of linear regions. We provide a mathematical understanding of the related additional expressiveness by means of a novel measure of shape complexity by counting deviations from convexity which results in a Boolean algebraic characterization of this special class.  A local representation theorem gives rise to novel approaches for pruning and decision surface analysis.

----

## [463] A Few Seconds Can Change Everything: Fast Decision-based Attacks against DNNs

**Authors**: *Ningping Mou, Baolin Zheng, Qian Wang, Yunjie Ge, Binqing Guo*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/464](https://doi.org/10.24963/ijcai.2022/464)

**Abstract**:

Previous researches have demonstrated deep learning models' vulnerabilities to decision-based adversarial attacks, which craft adversarial examples based solely on information from output decisions (top-1 labels). However, existing decision-based attacks have two major limitations, i.e., expensive query cost and being easy to detect. To bridge the gap and enlarge real threats to commercial applications, we propose a novel and efficient decision-based attack against black-box models, dubbed FastDrop, which only requires a few queries and work well under strong defenses. The crux of the innovation is that, unlike existing adversarial attacks that rely on gradient estimation and additive noise, FastDrop generates adversarial examples by dropping information in the frequency domain. Extensive experiments on three datasets demonstrate that FastDrop can escape the detection of the state-of-the-art (SOTA) black-box defenses and reduce the number of queries by 13~133Ã— under the same level of perturbations compared with the SOTA attacks. FastDrop only needs 10~20 queries to conduct an attack against various black-box models within 1s. Besides, on commercial vision APIs provided by Baidu and Tencent, FastDrop achieves an attack success rate (ASR) of 100% with 10 queries on average, which poses a real and severe threat to real-world applications.

----

## [464] Escaping Feature Twist: A Variational Graph Auto-Encoder for Node Clustering

**Authors**: *Nairouz Mrabah, Mohamed Bouguessa, Riadh Ksantini*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/465](https://doi.org/10.24963/ijcai.2022/465)

**Abstract**:

Most recent graph clustering methods rely on pretraining graph auto-encoders using self-supervision techniques (pretext task) and finetuning based on pseudo-supervision (main task). However, the transition from self-supervision to pseudo-supervision has never been studied from a geometric perspective. Herein, we establish the first systematic exploration of the latent manifolds' geometry under the deep clustering paradigm; we study the evolution of their intrinsic dimension and linear intrinsic dimension. We find that the embedded manifolds undergo coarse geometric transformations under the transition regime: from curved low-dimensional to flattened higher-dimensional. Moreover, we find that this inappropriate flattening leads to clustering deterioration by twisting the curved structures. To address this problem, which we call Feature Twist, we propose a variational graph auto-encoder that can smooth the local curves before gradually flattening the global structures. Our results show a notable improvement over multiple state-of-the-art approaches by escaping Feature Twist.

----

## [465] Composing Neural Learning and Symbolic Reasoning with an Application to Visual Discrimination

**Authors**: *Adithya Murali, Atharva Sehgal, Paul Krogmeier, P. Madhusudan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/466](https://doi.org/10.24963/ijcai.2022/466)

**Abstract**:

We consider the problem of combining machine learning models to perform higher-level cognitive tasks with clear specifications. We propose the novel problem of Visual Discrimination Puzzles (VDP) that requires finding interpretable discriminators that classify images according to a logical specification. Humans can solve these puzzles with ease and they give robust, verifiable, and interpretable discriminators as answers. We propose a compositional neurosymbolic framework that combines a neural network to detect objects and relationships with a symbolic learner that finds interpretable discriminators. We create large classes of VDP datasets involving natural and artificial images and show that our neurosymbolic framework performs favorably compared to several purely neural approaches.

----

## [466] Certified Robustness via Randomized Smoothing over Multiplicative Parameters of Input Transformations

**Authors**: *Nikita Muravev, Aleksandr Petiushko*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/467](https://doi.org/10.24963/ijcai.2022/467)

**Abstract**:

Currently the most popular method of providing robustness certificates is randomized smoothing where an input is smoothed via some probability distribution. We propose a novel approach to randomized smoothing over multiplicative parameters. Using this method we construct certifiably robust classifiers with respect to a gamma correction perturbation and compare the result with classifiers obtained via other smoothing distributions (Gaussian, Laplace, uniform). The experiments show that asymmetrical Rayleigh distribution allows to obtain better certificates for some values of perturbation parameters. To the best of our knowledge it is the first work concerning certified robustness against the multiplicative gamma correction transformation and the first to study effects of asymmetrical distributions in randomized smoothing.

----

## [467] Weakly-supervised Text Classification with Wasserstein Barycenters Regularization

**Authors**: *Jihong Ouyang, Yiming Wang, Ximing Li, Changchun Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/468](https://doi.org/10.24963/ijcai.2022/468)

**Abstract**:

Weakly-supervised text classification aims to train predictive models with unlabeled texts and a few representative words of classes, referred to as category words, rather than labeled texts. These weak supervisions are much more cheaper and easy to collect in real-world scenarios. To resolve this task, we propose a novel deep classification model, namely Weakly-supervised Text Classification with Wasserstein Barycenter Regularization (WTC-WBR). Specifically, we initialize the pseudo-labels of texts by using the category word occurrences, and formulate a weakly self-training framework to iteratively update the weakly-supervised targets by combining the pseudo-labels with the sharpened predictions. Most importantly, we suggest a Wasserstein barycenter regularization with the weakly-supervised targets on the deep feature space. The intuition is that the texts tend to be close to the corresponding Wasserstein barycenter indicated by weakly-supervised targets. Another benefit is that the regularization can capture the geometric information of deep feature space to boost the discriminative power of deep features. Experimental results demonstrate that WTC-WBR outperforms the existing weakly-supervised baselines, and achieves comparable performance to semi-supervised and supervised baselines.

----

## [468] Search-based Reinforcement Learning through Bandit Linear Optimization

**Authors**: *Milan Peelman, Antoon Bronselaer, Guy De Tré*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/469](https://doi.org/10.24963/ijcai.2022/469)

**Abstract**:

The development of AlphaZero was a breakthrough in search-based reinforcement learning, by employing a given world model in a Monte-Carlo tree search (MCTS) algorithm to incrementally learn both an action policy and a value estimation. When extending this paradigm to the setting of simultaneous move games we find that the selection strategy of AlphaZero has theoretical shortcomings, including that convergence to a Nash equilibrium is not guaranteed. By analyzing these shortcomings, we find that the selection strategy corresponds to an approximated version of bandit linear optimization using Tsallis entropy regularization with α parameter set to zero, which is equivalent to log-barrier regularization. This observation allows us to refine the search method used by AlphaZero to obtain an algorithm that has theoretically optimal regret as well as superior empirical performance on our evaluation benchmark.

----

## [469] On the Optimization of Margin Distribution

**Authors**: *Meng-Zhang Qian, Zheng Ai, Teng Zhang, Wei Gao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/470](https://doi.org/10.24963/ijcai.2022/470)

**Abstract**:

Margin has played an important role on the design and analysis of learning algorithms during the past years, mostly working with the maximization of the minimum margin. Recent years have witnessed the increasing empirical studies on the optimization of margin distribution according to different statistics such as medium margin, average margin, margin variance, etc., whereas there is a relative paucity of theoretical understanding.

In this work, we take one step on this direction by providing a new generalization error bound, which is heavily relevant to margin distribution by incorporating ingredients such as average margin and semi-variance, a new margin statistics for the characterization of margin distribution. Inspired by the theoretical findings, we propose the MSVMAv, an efficient approach to achieve better performance by optimizing margin distribution in terms of its empirical average margin and semi-variance. We finally conduct extensive experiments to show the superiority of the proposed MSVMAv approach.

----

## [470] Understanding the Limits of Poisoning Attacks in Episodic Reinforcement Learning

**Authors**: *Anshuka Rangi, Haifeng Xu, Long Tran-Thanh, Massimo Franceschetti*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/471](https://doi.org/10.24963/ijcai.2022/471)

**Abstract**:

To understand the security threats to reinforcement learning (RL) algorithms,  this paper studies poisoning attacks to manipulate any order-optimal learning algorithm towards a targeted policy in episodic RL and examines the potential damage of two natural types of poisoning attacks, i.e., the manipulation of reward or action. We discover that the effect of attacks crucially depends on whether the rewards are bounded or unbounded. In bounded reward settings, we show that only reward manipulation or only action manipulation cannot guarantee a successful attack. However,  by combining reward and action manipulation,  the adversary can manipulate any order-optimal learning algorithm to follow any targeted policy with \Theta(\sqrt{T})  total attack cost, which is order-optimal, without any knowledge of the underlying MDP. In contrast, in unbounded reward settings, we show that reward manipulation attacks are sufficient for an adversary to successfully manipulate any order-optimal learning algorithm to follow any targeted policy using \tilde{O}(\sqrt{T}) amount of contamination. Our results reveal useful insights about what can or cannot be achieved by poisoning attacks, and are set to spur more work on the design of robust RL algorithms.

----

## [471] Multi-Armed Bandit Problem with Temporally-Partitioned Rewards: When Partial Feedback Counts

**Authors**: *Giulia Romano, Andrea Agostini, Francesco Trovò, Nicola Gatti, Marcello Restelli*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/472](https://doi.org/10.24963/ijcai.2022/472)

**Abstract**:

There is a rising interest in industrial online applications where data becomes available sequentially. Inspired by the recommendation of playlists to users where their preferences can be collected during the listening of the entire playlist, we study a novel bandit setting, namely Multi-Armed Bandit with Temporally-Partitioned Rewards (TP-MAB), in which the stochastic reward associated with the pull of an arm is partitioned over a finite number of consecutive rounds following the pull. This setting, unexplored so far to the best of our knowledge, is a natural extension of delayed-feedback bandits to the case in which rewards may be dilated over a finite-time span after the pull instead of being fully disclosed in a single, potentially delayed round. We provide two algorithms to address TP-MAB problems, namely, TP-UCB-FR and TP-UCB-EW, which exploit the partial information disclosed by the reward collected over time. We show that our algorithms provide better asymptotical regret upper bounds than delayed-feedback bandit algorithms when a property characterizing a broad set of reward structures of practical interest, namely α-smoothness, holds. We also empirically evaluate their performance across a wide range of settings, both synthetically generated and from a real-world media recommendation problem.

----

## [472] Markov Abstractions for PAC Reinforcement Learning in Non-Markov Decision Processes

**Authors**: *Alessandro Ronca, Gabriel Paludo Licks, Giuseppe De Giacomo*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/473](https://doi.org/10.24963/ijcai.2022/473)

**Abstract**:

Our work aims at developing reinforcement learning algorithms that do not rely on the Markov assumption. We consider the class of Non-Markov Decision Processes where histories can be abstracted into a finite set of states while preserving the dynamics. We call it a Markov abstraction since it induces a Markov Decision Process over a set of states that encode the non-Markov dynamics. This phenomenon underlies the recently introduced Regular Decision Processes (as well as POMDPs where only a finite number of belief states is reachable). In all such kinds of decision process, an agent that uses a Markov abstraction can rely on the Markov property to achieve optimal behaviour. We show that Markov abstractions can be learned during reinforcement learning. Our approach combines automata learning and classic reinforcement learning. For these two tasks, standard algorithms can be employed. We show that our approach has PAC guarantees when the employed algorithms have PAC guarantees, and we also provide an experimental evaluation.

----

## [473] PAnDR: Fast Adaptation to New Environments from Offline Experiences via Decoupling Policy and Environment Representations

**Authors**: *Tong Sang, Hongyao Tang, Yi Ma, Jianye Hao, Yan Zheng, Zhaopeng Meng, Boyan Li, Zhen Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/474](https://doi.org/10.24963/ijcai.2022/474)

**Abstract**:

Deep Reinforcement Learning (DRL) has been a promising solution to many complex decision-making problems. Nevertheless, the notorious weakness in generalization among environments prevent widespread application of DRL agents in real-world scenarios. Although advances have been made recently, most prior works assume sufficient online interaction on training environments, which can be costly in practical cases.
To this end, we focus on an offline-training-online-adaptation setting,
in which the agent first learns from offline experiences collected in environments with different dynamics and then performs online policy adaptation in environments with new dynamics. In this paper, we propose Policy Adaptation with Decoupled Representations (PAnDR) for fast policy adaptation.
In offline training phase, the environment representation and policy representation are learned through contrastive learning and policy recovery, respectively. The representations are further refined by mutual information optimization to make them more decoupled and complete. With learned representations, a Policy-Dynamics Value Function (PDVF)  network is trained to approximate the values for different combinations of policies and environments from offline experiences. In online adaptation phase, with the environment context inferred from few experiences collected in new environments, the policy is optimized by gradient ascent with respect to the PDVF. Our experiments show that PAnDR outperforms existing algorithms in several representative policy adaptation problems.

----

## [474] Federated Multi-Task Attention for Cross-Individual Human Activity Recognition

**Authors**: *Qiang Shen, Haotian Feng, Rui Song, Stefano Teso, Fausto Giunchiglia, Hao Xu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/475](https://doi.org/10.24963/ijcai.2022/475)

**Abstract**:

Federated Learning (FL) is an emerging privacy-aware machine learning technique that applies successfully to the collaborative learning of global models for Human Activity Recognition (HAR). As of now, the applications of FL for HAR assume that the data associated with diverse individuals follow the same distribution. However, this assumption is impractical in real-world scenarios where the same activity is frequently performed differently by different individuals. To tackle this issue, we propose FedMAT, a Federated Multi-task ATtention framework for HAR, which extracts and fuses shared as well as individual-specific multi-modal sensor data features. Specifically, we treat the HAR problem associated with each individual as a different task and train a federated multi-task model, composed of a shared feature representation network in a central server plus multiple individual-specific networks with attention modules stored in decentralized nodes. In this architecture, the attention module operates as a mask that allows to learn individual-specific features from the global model, whilst simultaneously allowing for features to be shared among different individuals. We conduct extensive experiments based on publicly available HAR datasets, which are collected in both controlled environments and real-world scenarios. Numeric results verify that our proposed FedMAT significantly outperforms baselines not only in generalizing to existing individuals but also in adapting to new individuals.

----

## [475] Lexicographic Multi-Objective Reinforcement Learning

**Authors**: *Joar Skalse, Lewis Hammond, Charlie Griffin, Alessandro Abate*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/476](https://doi.org/10.24963/ijcai.2022/476)

**Abstract**:

In this work we introduce reinforcement learning techniques for solving lexicographic multi-objective problems.  These are problems that involve multiple reward signals, and where the goal is to learn a policy that maximises the first reward signal, and subject to this constraint also maximises the second reward signal, and so on. We present a family of both action-value and policy gradient algorithms that can be used to solve such problems, and prove that they converge to policies that are lexicographically optimal. We evaluate the scalability and performance of these algorithms empirically, and demonstrate their applicability in practical settings. As a more specific application, we show how our algorithms can be used to impose safety constraints on the behaviour of an agent, and compare their performance in this context with that of other constrained reinforcement learning algorithms.

----

## [476] Dynamic Sparse Training for Deep Reinforcement Learning

**Authors**: *Ghada Sokar, Elena Mocanu, Decebal Constantin Mocanu, Mykola Pechenizkiy, Peter Stone*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/477](https://doi.org/10.24963/ijcai.2022/477)

**Abstract**:

Deep reinforcement learning (DRL) agents are trained through trial-and-error interactions with the environment. This leads to a long training time for dense neural networks to achieve good performance. Hence, prohibitive computation and memory resources are consumed. Recently, learning efficient DRL agents has received increasing attention. Yet, current methods focus on accelerating inference time. In this paper, we introduce for the first time a dynamic sparse training approach for deep reinforcement learning to accelerate the training process. The proposed approach trains a sparse neural network from scratch and dynamically adapts its topology to the changing data distribution during training. Experiments on continuous control tasks show that our dynamic sparse agents achieve higher performance than the equivalent dense methods, reduce the parameter count and floating-point operations (FLOPs) by 50%, and have a faster learning speed that enables reaching the performance of dense agents with 40âˆ’50% reduction in the training steps.

----

## [477] CCLF: A Contrastive-Curiosity-Driven Learning Framework for Sample-Efficient Reinforcement Learning

**Authors**: *Chenyu Sun, Hangwei Qian, Chunyan Miao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/478](https://doi.org/10.24963/ijcai.2022/478)

**Abstract**:

In reinforcement learning (RL), it is challenging to learn directly from high-dimensional observations, where data augmentation has recently remedied it via encoding invariances from raw pixels. Nevertheless, we empirically find that not all samples are equally important and hence simply injecting more augmented inputs may instead cause instability in Q-learning. In this paper, we approach this problem systematically by developing a model-agnostic Contrastive-Curiosity-driven Learning Framework (CCLF), which can fully exploit sample importance and improve learning efficiency in a self-supervised manner. Facilitated by the proposed contrastive curiosity, CCLF is capable of prioritizing the experience replay, selecting the most informative augmented inputs, and more importantly regularizing the Q-function as well as the encoder to concentrate more on under-learned data. Moreover, it encourages the agent to explore with a curiosity-based reward. As a result, the agent can focus on more informative samples and learn representation invariances more efficiently, with significantly reduced augmented inputs. We apply CCLF to several base RL algorithms and evaluate on the DeepMind Control Suite, Atari, and MiniGrid benchmarks, where our approach demonstrates superior sample efficiency and learning performances compared with other state-of-the-art methods. Our code is available at https://github.com/csun001/CCLF.

----

## [478] Memory Augmented State Space Model for Time Series Forecasting

**Authors**: *Yinbo Sun, Lintao Ma, Yu Liu, Shijun Wang, James Zhang, Yangfei Zheng, Hu Yun, Lei Lei, Yulin Kang, Llinbao Ye*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/479](https://doi.org/10.24963/ijcai.2022/479)

**Abstract**:

State space model (SSM) provides a general and flexible forecasting framework for time series. Conventional SSM with fixed-order Markovian assumption often falls short in handling the long-range temporal dependencies and/or highly non-linear correlation in time-series data, which is crucial for accurate forecasting. To this extend, we present External Memory Augmented State Space Model (EMSSM) within the sequential Monte Carlo (SMC) framework. Unlike the common fixed-order Markovian SSM, our model features an external memory system, in which we store informative latent state experience, whereby to create ``memoryful" latent dynamics modeling complex long-term dependencies. Moreover, conditional normalizing flows are incorporated in our emission model, enabling the adaptation to a broad class of underlying data distributions. We further propose a Monte Carlo Objective that employs an efficient variational proposal distribution, which fuses the filtering and the dynamic prior information, to approximate the posterior state with proper particles. Our results demonstrate the competitiveness of forecasting performance of our proposed model comparing with other state-of-the-art SSMs.

----

## [479] MMT: Multi-way Multi-modal Transformer for Multimodal Learning

**Authors**: *Jiajia Tang, Kang Li, Ming Hou, Xuanyu Jin, Wanzeng Kong, Yu Ding, Qibin Zhao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/480](https://doi.org/10.24963/ijcai.2022/480)

**Abstract**:

The heart of multimodal learning research lies the challenge of effectively exploiting fusion representations among multiple modalities.However, existing two-way cross-modality unidirectional attention could only exploit the intermodal interactions from one source to one target modality. This indeed fails to unleash the complete expressive power of multimodal fusion with restricted number of modalities and fixed interactive direction.In this work, the multiway multimodal transformer (MMT) is proposed to simultaneously explore multiway multimodal intercorrelations for each modality via single block rather than multiple stacked cross-modality blocks. The core idea of MMT is the multiway multimodal attention, where the multiple modalities are leveraged to compute the multiway attention tensor. This naturally benefits us to exploit comprehensive many-to-many multimodal interactive paths. Specifically, the multiway tensor is comprised of multiple interconnected modality-aware core tensors that consist of the intramodal interactions. Additionally, the tensor contraction operation is utilized to investigate intermodal dependencies between distinct core tensors.Essentially, our tensor-based multiway structure allows for easily extending MMT to the case associated with an arbitrary number of modalities. Taking MMT as the basis, the hierarchical network is further established to recursively transmit the low-level multiway multimodal interactions to high-level ones. The experiments demonstrate that MMT can achieve state-of-the-art or comparable performance.

----

## [480] RecipeRec: A Heterogeneous Graph Learning Model for Recipe Recommendation

**Authors**: *Yijun Tian, Chuxu Zhang, Zhichun Guo, Chao Huang, Ronald A. Metoyer, Nitesh V. Chawla*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/481](https://doi.org/10.24963/ijcai.2022/481)

**Abstract**:

Recipe recommendation systems play an essential role in helping people decide what to eat. Existing recipe recommendation systems typically focused on content-based or collaborative filtering approaches, ignoring the higher-order collaborative signal such as relational structure information among users, recipes and food items. In this paper, we formalize the problem of recipe recommendation with graphs to incorporate the collaborative signal into recipe recommendation through graph modeling. In particular, we first present URI-Graph, a new and large-scale user-recipe-ingredient graph. We then propose RecipeRec, a novel heterogeneous graph learning model for recipe recommendation. The proposed model can capture recipe content and collaborative signal through a heterogeneous graph neural network with hierarchical attention and an ingredient set transformer. We also introduce a graph contrastive augmentation strategy to extract informative graph knowledge in a self-supervised manner. Finally, we design a joint objective function of recommendation and contrastive learning to optimize the model. Extensive experiments demonstrate that RecipeRec outperforms state-of-the-art methods for recipe recommendation. Dataset and codes are available at https://github.com/meettyj/RecipeRec.

----

## [481] Recipe2Vec: Multi-modal Recipe Representation Learning with Graph Neural Networks

**Authors**: *Yijun Tian, Chuxu Zhang, Zhichun Guo, Yihong Ma, Ronald A. Metoyer, Nitesh V. Chawla*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/482](https://doi.org/10.24963/ijcai.2022/482)

**Abstract**:

Learning effective recipe representations is essential in food studies. Unlike what has been developed for image-based recipe retrieval or learning structural text embeddings, the combined effect of multi-modal information (i.e., recipe images, text, and relation data) receives less attention. In this paper, we formalize the problem of multi-modal recipe representation learning to integrate the visual, textual, and relational information into recipe embeddings. In particular, we first present Large-RG, a new recipe graph data with over half a million nodes, making it the largest recipe graph to date. We then propose Recipe2Vec, a novel graph neural network based recipe embedding model to capture multi-modal information. Additionally, we introduce an adversarial attack strategy to ensure stable learning and improve performance. Finally, we design a joint objective function of node classification and adversarial learning to optimize the model. Extensive experiments demonstrate that Recipe2Vec outperforms state-of-the-art baselines on two classic food study tasks, i.e., cuisine category classification and region prediction. Dataset and codes are available at https://github.com/meettyj/Recipe2Vec.

----

## [482] Stochastic Coherence Over Attention Trajectory For Continuous Learning In Video Streams

**Authors**: *Matteo Tiezzi, Simone Marullo, Lapo Faggi, Enrico Meloni, Alessandro Betti, Stefano Melacci*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/483](https://doi.org/10.24963/ijcai.2022/483)

**Abstract**:

Devising intelligent agents able to live in an environment and learn by observing the surroundings is a longstanding goal of Artificial Intelligence. From a bare Machine Learning perspective, challenges arise when the agent is prevented from leveraging large fully-annotated dataset, but rather the interactions with supervisory signals are sparsely distributed over space and time.
This paper proposes a novel neural-network-based approach to progressively and autonomously develop pixel-wise representations in a video stream. The proposed method is based on a human-like attention mechanism that allows the agent to learn by observing what is moving in the attended locations. Spatio-temporal stochastic coherence along the attention trajectory, paired with a contrastive term, leads to an unsupervised learning criterion that naturally copes with the considered setting. Differently from most existing works, the learned representations are used in open-set class-incremental classification of each frame pixel, relying on few supervisions. Our experiments leverage 3D virtual environments and they show that the proposed agents can learn to distinguish objects just by observing the video stream. Inheriting features from state-of-the art models is not as powerful as one might expect.

----

## [483] Approximate Exploitability: Learning a Best Response

**Authors**: *Finbarr Timbers, Nolan Bard, Edward Lockhart, Marc Lanctot, Martin Schmid, Neil Burch, Julian Schrittwieser, Thomas Hubert, Michael Bowling*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/484](https://doi.org/10.24963/ijcai.2022/484)

**Abstract**:

Researchers have shown that neural networks are vulnerable to adversarial examples and subtle environment changes.  The resulting errors can look like blunders to humans, eroding trust in these agents. In prior games research, agent evaluation often focused on the in-practice game outcomes. Such evaluation typically fails to evaluate robustness to worst-case outcomes. Computer poker research has examined how to assess such worst-case performance. Unfortunately, exact computation is infeasible with larger domains, and existing approximations are poker-specific. We introduce ISMCTS-BR, a scalable search-based deep reinforcement learning algorithm for learning a best response to an agent,  approximating worst-case performance. We demonstrate the technique in several games against a variety of agents, including several AlphaZero-based agents. Supplementary material is available at https://arxiv.org/abs/2004.09677.

----

## [484] Initializing Then Refining: A Simple Graph Attribute Imputation Network

**Authors**: *Wenxuan Tu, Sihang Zhou, Xinwang Liu, Yue Liu, Zhiping Cai, En Zhu, Changwang Zhang, Jieren Cheng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/485](https://doi.org/10.24963/ijcai.2022/485)

**Abstract**:

Representation learning on the attribute-missing graphs, whose connection information is complete while the attribute information of some nodes is missing, is an important yet challenging task. To impute the missing attributes, existing methods isolate the learning processes of attribute and structure information embeddings, and force both resultant representations to align with a common in-discriminative normal distribution, leading to inaccurate imputation. To tackle these issues, we propose a novel graph-oriented imputation framework called initializing then refining (ITR), where we first employ the structure information for initial imputation, and then leverage observed attribute and structure information to adaptively refine the imputed latent variables. Specifically, we first adopt the structure embeddings of attribute-missing samples as the embedding initialization, and then refine these initial values by aggregating the reliable and informative embeddings of attribute-observed samples according to the affinity structure. Specially, in our refining process, the affinity structure is adaptively updated through iterations by calculating the sample-wise correlations upon the recomposed embeddings. Extensive experiments on four benchmark datasets verify the superiority of ITR against state-of-the-art methods.

----

## [485] Bounded Memory Adversarial Bandits with Composite Anonymous Delayed Feedback

**Authors**: *Zongqi Wan, Xiaoming Sun, Jialin Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/486](https://doi.org/10.24963/ijcai.2022/486)

**Abstract**:

We study the adversarial bandit problem with composite anonymous delayed feedback. In this setting, losses of an action are split into d components, spreading over consecutive rounds after the action is chosen. And in each round, the algorithm observes the aggregation of losses that come from the latest d rounds. Previous works focus on oblivious adversarial setting, while we investigate the harder nonoblivious setting. We show nonoblivious setting incurs Omega(T) pseudo regret even when the loss sequence is bounded memory. However, we propose a wrapper algorithm which enjoys o(T) policy regret on many adversarial bandit problems with the assumption that the loss sequence is bounded memory. Especially, for K armed bandit and bandit convex optimization, our policy regret bound is in the order of T to the two third. We also prove a matching lower bound for K armed bandit. Our lower bound works even when the loss sequence is oblivious but the delay is nonoblivious. It answers the open problem proposed in [Wang, Wang, Huang 2021], showing that nonoblivious delay is enough to incur the regret in the order of T to the two third.

----

## [486] Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration

**Authors**: *Di Wang, Jinyuan Liu, Xin Fan, Risheng Liu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/487](https://doi.org/10.24963/ijcai.2022/487)

**Abstract**:

Recent learning-based image fusion methods have marked numerous progress in pre-registered multi-modality data, but suffered serious ghosts dealing with misaligned multi-modality data, due to the spatial deformation and the difficulty narrowing cross-modality discrepancy.
To overcome the obstacles, in this paper, we present a robust cross-modality generation-registration paradigm for unsupervised misaligned infrared and visible image fusion (IVIF).
Specifically, we propose a Cross-modality Perceptual Style Transfer Network (CPSTN) to generate a pseudo infrared image taking a visible image as input.
Benefiting from the favorable geometry preservation ability of the CPSTN, the generated pseudo infrared image embraces a sharp structure, which is more conducive to transforming cross-modality image alignment into mono-modality registration coupled with the structure-sensitive of the infrared image. 
In this case, we introduce a Multi-level Refinement Registration Network (MRRN) to predict the displacement vector field between distorted and pseudo infrared images and reconstruct registered infrared image under the mono-modality setting.
Moreover, to better fuse the registered infrared images and visible images, we present a feature Interaction Fusion Module (IFM) to adaptively select more meaningful features for fusion in the Dual-path Interaction Fusion Network (DIFN).
Extensive experimental results suggest that the proposed method performs superior capability on misaligned cross-modality image fusion.

----

## [487] Multi-Task Personalized Learning with Sparse Network Lasso

**Authors**: *Jiankun Wang, Lu Sun*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/488](https://doi.org/10.24963/ijcai.2022/488)

**Abstract**:

Multi-task learning learns multiple related tasks together, in order to improve the generalization performance. Existing methods typically build a global model shared by all the samples, which saves the homogeneity but ignores the individuality (heterogeneity) of samples. Personalized learning is recently proposed to learn sample-specific local models by utilizing sample heterogeneity, however, directly applying it in the multi-task learning setting poses three key challenges: 1) model sample homogeneity, 2) prevent from over-parameterization and 3) capture task correlations. In this paper, we propose a novel multi-task personalized learning method to handle these challenges. For 1), each model is decomposed into a sum of global and local components, that saves sample homogeneity and sample heterogeneity, respectively. For 2), regularized by sparse network Lasso, the joint models are embedded into a low-dimensional subspace and exhibit sparse group structures, leading to a significantly reduced number of effective parameters. For 3), the subspace is further separated into two parts, so as to save both commonality and specificity of tasks.  We develop an alternating algorithm to solve the proposed optimization problem, and extensive experiments on various synthetic and real-world datasets demonstrate its robustness and effectiveness.

----

## [488] IMO^3: Interactive Multi-Objective Off-Policy Optimization

**Authors**: *Nan Wang, Hongning Wang, Maryam Karimzadehgan, Branislav Kveton, Craig Boutilier*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/489](https://doi.org/10.24963/ijcai.2022/489)

**Abstract**:

Most real-world optimization problems have multiple objectives. A system designer needs to find a policy that trades off these objectives to reach a desired operating point. This problem has been studied extensively in the setting of known objective functions. However, we consider a more practical but challenging setting of unknown objective functions. In industry, optimization under this setting is mostly approached with online A/B testing, which is often costly and inefficient. As an alternative, we propose Interactive Multi-Objective Off-policy Optimization (IMO^3). The key idea of IMO^3 is to interact with a system designer using policies evaluated in an off-policy fashion to uncover which policy maximizes her unknown utility function. We theoretically show that IMO^3 identifies a near-optimal policy with high probability, depending on the amount of designer's feedback and training data for off-policy estimation. We demonstrate its effectiveness empirically on several multi-objective optimization problems.

----

## [489] Modeling Spatio-temporal Neighbourhood for Personalized Point-of-interest Recommendation

**Authors**: *Xiaolin Wang, Guohao Sun, Xiu Fang, Jian Yang, Shoujin Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/490](https://doi.org/10.24963/ijcai.2022/490)

**Abstract**:

Point-of-interest (POI) recommendations can help users explore attractive locations, which is playing an important role in location-based social networks (LBSNs). In POI recommendations, the results are largely impacted by users' preferences. However, the existing POI methods model user and location almost separately, which cannot capture users' personal and dynamic preferences to location. In addition, they also ignore users' acceptance to distance/time of location. To overcome the limitations of the existing methods, we first introduce Knowledge Graph with temporal information (known as TKG) into POI recommendation, including both user and location with timestamps. Then, based on TKG, we propose a Spatial-Temporal Graph Convolutional Attention Network (STGCAN), a novel network that learns users' preferences on TKG by dynamically capturing the spatial-temporal neighbourhoods. Specifically, in STGCAN, we construct receptive fields on TKG to aggregate neighbourhoods of user and location respectively at each timestamp. And we measure the spatial-temporal interval as users' acceptance to distance/time with self-attention. Experiments on three real-world datasets demonstrate that the proposed model outperforms the state-of-the-art POI recommendation approaches.

----

## [490] Multi-Player Multi-Armed Bandits with Finite Shareable Resources Arms: Learning Algorithms & Applications

**Authors**: *Xuchuang Wang, Hong Xie, John C. S. Lui*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/491](https://doi.org/10.24963/ijcai.2022/491)

**Abstract**:

Multi-player multi-armed bandits (MMAB) study how decentralized players cooperatively play the same multi-armed bandit so as to maximize their total cumulative rewards. Existing MMAB models mostly assume when more than one player pulls the same arm, they either have a collision and obtain zero rewards or have no collision and gain independent rewards, both of which are usually too restrictive in practical scenarios. In this paper, we propose an MMAB with shareable resources as an extension of the collision and non-collision settings. Each shareable arm has finite shareable resources and a “per-load” reward random variable, both of which are unknown to players. The reward from a shareable arm is equal to the “per-load” reward multiplied by the minimum between the number of players pulling the arm and the arm’s maximal shareable resources. We consider two types of feedback: sharing demand information (SDI) and sharing demand awareness (SDA), each of which provides different signals of resource sharing. We design the DPE-SDI and SIC-SDA algorithms to address the shareable arm problem under these two cases of feedback respectively and prove that both algorithms have logarithmic regrets that are tight in the number of rounds. We conduct simulations to validate both algorithms’ performance and show their utilities in wireless networking and edge computing.

----

## [491] Estimation and Comparison of Linear Regions for ReLU Networks

**Authors**: *Yuan Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/492](https://doi.org/10.24963/ijcai.2022/492)

**Abstract**:

We study the relationship between the arrangement of neurons and the complexity of the ReLU-activated neural networks measured by the number of linear regions. More specifically, we provide both theoretical and empirical evidence for the point of view that shallow networks tend to have higher complexity than deep ones when the total number of neurons is fixed. In the theoretical part, we prove that this is the case for networks whose neurons in the hidden layers are arranged in the forms of 1x2n, 2xn and nx2; in the empirical part, we implement an algorithm that precisely tracks (hence counts) all the linear regions, and run it on networks with various structures. Although the time complexity of the algorithm is quite high, we verify that the problem of calculating the number of linear regions of a ReLU network is itself NP-hard. So currently there is no surprisingly efficient way to solve it. Roughly speaking, in the algorithm we divide the linear regions into subregions called the "activation regions", which are convex and easy to propagate through the network. The relationship between the number of the linear regions and that of the activation regions is also discussed.

----

## [492] Self-paced Supervision for Multi-source Domain Adaptation

**Authors**: *Zengmao Wang, Chaoyang Zhou, Bo Du, Fengxiang He*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/493](https://doi.org/10.24963/ijcai.2022/493)

**Abstract**:

Multi-source domain adaptation has attracted great attention in machine learning community. Most of these methods focus on weighting the predictions produced by the adaptation networks of different domains. Thus the domain shifts between certain of domains and target domain are not effectively relieved, resulting in that these domains are not fully exploited and even may have a negative influence on multi-source domain adaptation task. To address such challenge, we propose a multi-source domain adaptation method to gradually improve the adaptation ability of each source domain by producing more high-confident pseudo-labels with self-paced learning for conditional distribution alignment. The proposed method first trains several separate domain branch networks with single domains and an ensemble branch network with all domains. Then we obtain some high-confident pseudo-labels with the branch networks and learn the branch specific pseudo-labels with self-paced learning. Each branch network reduces the domain gap by aligning the conditional distribution with its branch specific pseudo-labels and the pseudo-labels provided by all branch networks. Experiments on Office31, Office-Home and DomainNet show that the proposed method outperforms the state-of-the-art methods.

----

## [493] Value Refinement Network (VRN)

**Authors**: *Jan Wöhlke, Felix Schmitt, Herke van Hoof*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/494](https://doi.org/10.24963/ijcai.2022/494)

**Abstract**:

In robotic tasks, we encounter the unique strengths of (1) reinforcement learning (RL) that can handle high-dimensional observations as well as unknown, complex dynamics and (2) planning that can handle sparse and delayed rewards given a dynamics model. Combining these strengths of RL and planning, we propose the Value Refinement Network (VRN), in this work. Our VRN is an RL-trained neural network architecture that learns to locally refine an initial (value-based) plan in a simplified (2D) problem abstraction based on detailed local sensory observations. We evaluate the VRN on simulated robotic (navigation) tasks and demonstrate that it can successfully refine sub-optimal plans to match the performance of more costly planning in the non-simplified problem. Furthermore, in a dynamic environment, the VRN still enables high task completion without global re-planning.

----

## [494] EMGC²F: Efficient Multi-view Graph Clustering with Comprehensive Fusion

**Authors**: *Danyang Wu, Jitao Lu, Feiping Nie, Rong Wang, Yuan Yuan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/495](https://doi.org/10.24963/ijcai.2022/495)

**Abstract**:

This paper proposes an Efficient Multi-view Graph Clustering with Comprehensive Fusion (EMGC²F) model and a corresponding efficient optimization algorithm to address multi-view graph clustering tasks effectively and efficiently. Compared to existing works, our proposals have the following highlights: 1) EMGC²F directly finds a consistent cluster indicator matrix with a Super Nodes Similarity Minimization module from multiple views, which avoids time-consuming spectral decomposition in previous works. 2) EMGC²F comprehensively mines information from multiple views. More formally, it captures the consistency of multiple views via a Cross-view Nearest Neighbors Voting (CN²V) mechanism, meanwhile capturing the importance of multiple views via an adaptive weighted-learning mechanism. 3) EMGC²F is a parameter-free model and the time complexity of the proposed algorithm is far less than existing works, demonstrating the practicability. Empirical results on several benchmark datasets demonstrate that our proposals outperform SOTA competitors both in effectiveness and efficiency.

----

## [495] A Unified Meta-Learning Framework for Dynamic Transfer Learning

**Authors**: *Jun Wu, Jingrui He*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/496](https://doi.org/10.24963/ijcai.2022/496)

**Abstract**:

Transfer learning refers to the transfer of knowledge or information from a relevant source task to a target task. However, most existing works assume both tasks are sampled from a stationary task distribution, thereby leading to the sub-optimal performance for dynamic tasks drawn from a non-stationary task distribution in real scenarios. To bridge this gap, in this paper, we study a more realistic and challenging transfer learning setting with dynamic tasks, i.e., source and target tasks are continuously evolving over time. We theoretically show that the expected error on the dynamic target task can be tightly bounded in terms of source knowledge and consecutive distribution discrepancy across tasks. This result motivates us to propose a generic meta-learning framework L2E for modeling the knowledge transferability on dynamic tasks. It is centered around a task-guided meta-learning problem with a group of meta-pairs of tasks, based on which we are able to learn the prior model initialization for fast adaptation on the newest target task. L2E enjoys the following properties: (1) effective knowledge transferability across dynamic tasks; (2) fast adaptation to the new target task; (3) mitigation of catastrophic forgetting on historical target tasks; and (4) flexibility in incorporating any existing static transfer learning algorithms. Extensive experiments on various image data sets demonstrate the effectiveness of the proposed L2E framework.

----

## [496] A Simple yet Effective Method for Graph Classification

**Authors**: *Junran Wu, Shangzhe Li, Jianhao Li, Yicheng Pan, Ke Xu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/497](https://doi.org/10.24963/ijcai.2022/497)

**Abstract**:

In deep neural networks, better results can often be obtained by increasing the complexity of previously developed basic models. However, it is unclear whether there is a way to boost performance by decreasing the complexity of such models. Intuitively, given a problem, a simpler data structure comes with a simpler algorithm. Here, we investigate the feasibility of improving graph classification performance while simplifying the learning process. Inspired by structural entropy on graphs, we transform the data sample from graphs to coding trees, which is a simpler but essential structure for graph data. Furthermore, we propose a novel message passing scheme, termed hierarchical reporting, in which features are transferred from leaf nodes to root nodes by following the hierarchical structure of coding trees. We then present a tree kernel and a convolutional network to implement our scheme for graph classification. With the designed message passing scheme, the tree kernel and convolutional network have a lower runtime complexity of O(n) than Weisfeiler-Lehman subtree kernel and other graph neural networks of at least O(hm). We empirically validate our methods with several graph classification benchmarks and demonstrate that they achieve better performance and lower computational consumption than competing approaches.

----

## [497] Stabilizing and Enhancing Link Prediction through Deepened Graph Auto-Encoders

**Authors**: *Xinxing Wu, Qiang Cheng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/498](https://doi.org/10.24963/ijcai.2022/498)

**Abstract**:

Graph neural networks have been widely used for a variety of learning tasks. Link prediction is a relatively under-studied graph learning task, with current state-of-the-art models based on one- or two-layer shallow graph auto-encoder (GAE) architectures. In this paper, we overcome the limitation of current methods for link prediction of non-Euclidean network data, which can only use shallow GAEs and variational GAEs. Our proposed methods innovatively incorporate standard auto-encoders (AEs) into the architectures of GAEs to capitalize on the intimate coupling of node and edge information in complex network data. Empirically, extensive experiments on various datasets demonstrate the competitive performance of our proposed approach. Theoretically, we prove that our deep extensions can inclusively express multiple polynomial filters with different orders. The codes of this paper are available at https://github.com/xinxingwu-uk/DGAE.

----

## [498] Automatically Gating Multi-Frequency Patterns through Rectified Continuous Bernoulli Units with Theoretical Principles

**Authors**: *Zheng-Fan Wu, Yi-Nan Feng, Hui Xue*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/499](https://doi.org/10.24963/ijcai.2022/499)

**Abstract**:

Different nonlinearities are only suitable for responding to different frequency signals. The locally-responding ReLU is incapable of modeling high-frequency features due to the spectral bias, whereas the globally-responding sinusoidal function is intractable to represent low-frequency concepts cheaply owing to the optimization dilemma. Moreover, nearly all the practical tasks are composed of complex multi-frequency patterns, whereas there is little prospect of designing or searching a heterogeneous network containing various types of neurons matching the frequencies, because of their exponentially-increasing combinatorial states. In this paper, our contributions are three-fold: 1) we propose a general Rectified Continuous Bernoulli (ReCB) unit paired with an efficient variational Bayesian learning paradigm, to automatically detect/gate/represent different frequency responses; 2) our numerically-tight theoretical framework proves that ReCB-based networks can achieve the optimal representation ability, which is O(m^{η/(d^2)}) times better than that of popular neural networks, for a hidden dimension of m, an input dimension of d, and a Lipschitz constant of η; 3) we provide comprehensive empirical evidence showing that ReCB-based networks can keenly learn multi-frequency patterns and push the state-of-the-art performance.

----

## [499] Information Augmentation for Few-shot Node Classification

**Authors**: *Zongqian Wu, Peng Zhou, Guoqiu Wen, Yingying Wan, Junbo Ma, Debo Cheng, Xiaofeng Zhu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/500](https://doi.org/10.24963/ijcai.2022/500)

**Abstract**:

Although meta-learning and metric learning have been widely applied for few-shot node classification (FSNC), some limitations still need to be addressed, such as expensive time costs for the meta-train and difficult of exploring the complex structure inherent the graph data. To address in issues, this paper proposes a new data augmentation method to conduct FSNC on the graph data including parameter initialization and parameter fine-tuning. Specifically, parameter initialization only conducts a multi-classification task on the base classes, resulting in good generalization ability and less time cost. Parameter fine-tuning designs two data augmentation methods (i.e., support augmentation and shot augmentation) on the novel classes to generate sufficient node features so that any traditional supervised classifiers can be used to classify the query set. As a result, the proposed method is the first work of data augmentation for FSNC. Experiment results show the effectiveness and the efficiency of our proposed method, compared to state-of-the-art methods, in terms of different classification tasks.

----

## [500] Adversarial Bi-Regressor Network for Domain Adaptive Regression

**Authors**: *Haifeng Xia, Pu Wang, Toshiaki Koike-Akino, Ye Wang, Philip V. Orlik, Zhengming Ding*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/501](https://doi.org/10.24963/ijcai.2022/501)

**Abstract**:

Domain adaptation (DA) aims to transfer the knowledge of a well-labeled source domain to facilitate unlabeled target learning. When turning to specific tasks such as indoor (Wi-Fi) localization, it is essential to learn a cross-domain regressor to mitigate the domain shift. This paper proposes a novel method Adversarial Bi-Regressor Network (ABRNet) to seek more effective cross- domain regression model. Specifically, a discrepant bi-regressor architecture is developed to maximize the difference of bi-regressor to discover uncertain target instances far from the source distribution, and then an adversarial training mechanism is adopted between feature extractor and dual regressors to produce domain-invariant representations. To further bridge the large domain gap, a domain- specific augmentation module is designed to synthesize two source-similar and target-similar inter- mediate domains to gradually eliminate the original domain mismatch. The empirical studies on two cross-domain regressive benchmarks illustrate the power of our method on solving the domain adaptive regression (DAR) problem.

----

## [501] Ambiguity-Induced Contrastive Learning for Instance-Dependent Partial Label Learning

**Authors**: *Shiyu Xia, Jiaqi Lv, Ning Xu, Xin Geng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/502](https://doi.org/10.24963/ijcai.2022/502)

**Abstract**:

Partial label learning (PLL) learns from a typical weak supervision, where each training instance is labeled with a set of ambiguous candidate labels (CLs) instead of its exact ground-truth label. Most existing PLL works directly eliminate, rather than exploiting the label ambiguity, since they explicitly or implicitly assume that incorrect CLs are noise independent of the instance. While a more practical setting in the wild should be instance-dependent, namely, the CLs depend on both the true label and the instance itself, such that each CL may describe the instance from some sensory channel, thereby providing some noisy but really valid information about the instance. In this paper, we leverage such additional information acquired from the ambiguity and propose AmBiguity-induced contrastive LEarning (ABLE) under the framework of contrastive learning. Specifically, for each CL of an anchor, we select a group of samples currently predicted as that class as ambiguity-induced positives, based on which we synchronously learn a representor (RP) that minimizes the weighted sum of contrastive losses of all groups and a classifier (CS) that minimizes a classification loss. Although they are circularly dependent: RP requires the ambiguity-induced positives on-the-fly induced by CS, and CS needs the first half of RP as the representation extractor, ABLE still enables RP and CS to be trained simultaneously within a coherent framework. Experiments on benchmark datasets demonstrate its substantial improvements over state-of-the-art methods for learning from the instance-dependent partially labeled data.

----

## [502] Neuro-Symbolic Verification of Deep Neural Networks

**Authors**: *Xuan Xie, Kristian Kersting, Daniel Neider*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/503](https://doi.org/10.24963/ijcai.2022/503)

**Abstract**:

Formal verification has emerged as a powerful approach to ensure the safety and reliability of deep neural networks. However, current verification tools are limited to only a handful of properties that can be expressed as first-order constraints over the inputs and output of a network. While adversarial robustness and fairness fall under this category, many real-world properties (e.g., "an autonomous vehicle has to stop in front of a stop sign") remain outside the scope of existing verification technology. To mitigate this severe practical restriction, we introduce a novel framework for verifying neural networks, named neuro-symbolic verification. The key idea is to use neural networks as part of the otherwise logical specification, enabling the verification of a wide variety of complex, real-world properties, including the one above. A defining feature of our framework is that it can be implemented on top of existing verification infrastructure for neural networks, making it easily accessible to researchers and practitioners.

----

## [503] MultiQuant: Training Once for Multi-bit Quantization of Neural Networks

**Authors**: *Ke Xu, Qiantai Feng, Xingyi Zhang, Dong Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/504](https://doi.org/10.24963/ijcai.2022/504)

**Abstract**:

Quantization has become a popular technique to compress deep neural networks (DNNs) and reduce computational costs, but most prior work focuses on training DNNs at each individual fixed bit-width and accuracy trade-off point. How to produce a model with flexible precision is largely unexplored. This work proposes a multi-bit quantization framework (MultiQuant) to make the learned DNNs robust for different precision configuration during inference by adopting Lowest-Random-Highest bit-width co-training method. Meanwhile, we propose an online adaptive label generation strategy to alleviate the problem of vicious competition under different precision caused by one-hot labels in the supernet training. The trained supernet model can be flexibly set to different bit widths to support dynamic speed and accuracy trade-off. Furthermore, we adopt the Monte Carlo sampling-based genetic algorithm search strategy with quantization-aware accuracy predictor as evaluation criterion to incorporate the mixed precision technology in our framework. Experiment results on ImageNet datasets demonstrate MultiQuant method can attain the quantization results under different bit-widths comparable with quantization-aware training without retraining.

----

## [504] MemREIN: Rein the Domain Shift for Cross-Domain Few-Shot Learning

**Authors**: *Yi Xu, Lichen Wang, Yizhou Wang, Can Qin, Yulun Zhang, Yun Fu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/505](https://doi.org/10.24963/ijcai.2022/505)

**Abstract**:

Few-shot learning aims to enable models generalize to new categories (query instances) with only limited labeled samples (support instances) from each category. Metric-based mechanism is a promising direction which compares feature embeddings via different metrics. However, it always fail to generalize to unseen domains due to the considerable domain gap challenge. In this paper, we propose a novel framework, MemREIN, which considers Memorized, Restitution, and Instance Normalization for cross-domain few-shot learning. Specifically, an instance normalization algorithm is explored to alleviate feature dissimilarity, which provides the initial model generalization ability. However, naively normalizing the feature would lose fine-grained discriminative knowledge between different classes. To this end, a memorized module is further proposed to separate the most refined knowledge and remember it. Then, a restitution module is utilized to restitute the discrimination ability from the learned knowledge. A novel reverse contrastive learning strategy is proposed to stabilize the distillation process. Extensive experiments on five popular benchmark datasets demonstrate that MemREIN well addresses the domain shift challenge, and significantly improves the performance up to 16.43% compared with state-of-the-art baselines.

----

## [505] Active Contrastive Set Mining for Robust Audio-Visual Instance Discrimination

**Authors**: *Hanyu Xuan, Yihong Xu, Shuo Chen, Zhiliang Wu, Jian Yang, Yan Yan, Xavier Alameda-Pineda*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/506](https://doi.org/10.24963/ijcai.2022/506)

**Abstract**:

The recent success of audio-visual representation learning can be largely attributed to their pervasive property of audio-visual synchronization, which can be used as self-annotated supervision. As a state-of-the-art solution, Audio-Visual Instance Discrimination (AVID) extends instance discrimination to the audio-visual realm. Existing AVID methods construct the contrastive set by random sampling based on the assumption that the audio and visual clips from all other videos are not semantically related. We argue that this assumption is rough, since the resulting contrastive sets have a large number of faulty negatives. In this paper, we overcome this limitation by proposing a novel Active Contrastive Set Mining (ACSM) that aims to mine the contrastive sets with informative and diverse negatives for robust AVID. Moreover, we also integrate a semantically-aware hard-sample mining strategy into our ACSM. The proposed ACSM is implemented into two most recent state-of-the-art AVID methods and significantly improves their performance. Extensive experiments conducted on both action and  sound recognition on multiple datasets show the remarkably improved performance of our method.

----

## [506] On the (In)Tractability of Reinforcement Learning for LTL Objectives

**Authors**: *Cambridge Yang, Michael L. Littman, Michael Carbin*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/507](https://doi.org/10.24963/ijcai.2022/507)

**Abstract**:

In recent years, researchers have made significant progress in devising reinforcement-learning algorithms for optimizing linear temporal logic (LTL) objectives and LTL-like objectives.
Despite these advancements, there are fundamental limitations to how well this problem can be solved. Previous studies have alluded to this fact but have not examined it in depth.
In this paper, we address the tractability of reinforcement learning for general LTL objectives from a theoretical perspective.
We formalize the problem under the probably approximately correct learning in Markov decision processes (PAC-MDP) framework, a standard framework for measuring sample complexity in reinforcement learning.
In this formalization, we prove that the optimal policy for any LTL formula is PAC-MDP-learnable if and only if the formula is in the most limited class in the LTL hierarchy, consisting of formulas that are decidable within a finite horizon.
Practically, our result implies that it is impossible for a reinforcement-learning algorithm to obtain a PAC-MDP guarantee on the performance of its learned policy after finitely many interactions with an unconstrained environment for LTL objectives that are not decidable within a finite horizon.

----

## [507] Towards Applicable Reinforcement Learning: Improving the Generalization and Sample Efficiency with Policy Ensemble

**Authors**: *Zhengyu Yang, Kan Ren, Xufang Luo, Minghuan Liu, Weiqing Liu, Jiang Bian, Weinan Zhang, Dongsheng Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/508](https://doi.org/10.24963/ijcai.2022/508)

**Abstract**:

It is challenging for reinforcement learning (RL) algorithms to succeed in real-world applications. Take financial trading as an example, the market information is noisy yet imperfect and the macroeconomic regulation or other factors may shift between training and evaluation, thus it requires both generalization and high sample efficiency for resolving the task. However, directly applying typical RL algorithms can lead to poor performance in such scenarios. To derive a robust and applicable RL algorithm, in this work, we design a simple but effective method named Ensemble Proximal Policy Optimization (EPPO), which learns ensemble policies in an end-to-end manner. Notably, EPPO combines each policy and the policy ensemble organically and optimizes both simultaneously. In addition, EPPO adopts a diversity enhancement regularization over the policy space which helps to generalize to unseen states and promotes exploration. We theoretically prove that EPPO can increase exploration efficacy, and through comprehensive experimental evaluations on various tasks, we demonstrate that EPPO achieves higher efficiency and is robust for real-world applications compared with vanilla policy optimization algorithms and other ensemble methods. Code and supplemental materials are available at https://seqml.github.io/eppo.

----

## [508] Online ECG Emotion Recognition for Unknown Subjects via Hypergraph-Based Transfer Learning

**Authors**: *Yalan Ye, Tongjie Pan, Qianhe Meng, Jingjing Li, Li Lu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/509](https://doi.org/10.24963/ijcai.2022/509)

**Abstract**:

Electrocardiogram (ECG) signal based cross-subject emotion recognition methods reduce the influence of individual differences using domain adaptation (DA) techniques. These methods generally assume that the entire unlabeled data of unknown target subjects are available in training phase.  However, this assumption does not hold in some practical scenarios where the data of target subjects arrive one by one in an online manner instead of being acquired at a time. Thus, existing DA methods cannot be directly applied in this case since the unknown target data is inaccessible in training phase. To tackle the problem, we propose a novel online cross-subject ECG emotion recognition method leveraging hypergraph-based online transfer learning (HOTL). Specifically, the proposed hypergraph structure is capable of learning the high-order correlation among data, such that the recognition model trained on source subjects can be more effectively generalized to target subjects. Meanwhile, the structure can be easily updated by adding a hyperedge which connects a newly coming sample with the current hypergraph, resulting in further reduce the individual differences in online manner without re-training the model. Consequently, HOTL can effectively deal with the online cross-subject scenario where unknown target ECG data arrive one by one and varying overtime. Extensive experiments conducted on the Amigos dataset validate the superiority of the proposed method.

----

## [509] Towards Safe Reinforcement Learning via Constraining Conditional Value-at-Risk

**Authors**: *Chengyang Ying, Xinning Zhou, Hang Su, Dong Yan, Ning Chen, Jun Zhu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/510](https://doi.org/10.24963/ijcai.2022/510)

**Abstract**:

Though deep reinforcement learning (DRL) has obtained substantial success, it may encounter catastrophic failures due to the intrinsic uncertainty of both transition and observation. Most of the existing methods for safe reinforcement learning can only handle transition disturbance or observation disturbance since these two kinds of disturbance affect different parts of the agent;   besides, the popular worst-case return may lead to overly pessimistic policies. To address these issues, we first theoretically prove that the performance degradation under transition disturbance and observation disturbance depends on a novel metric of Value Function Range (VFR), which corresponds to the gap in the value function between the best state and the worst state. Based on the analysis, we adopt conditional value-at-risk (CVaR) as an assessment of risk and propose a novel reinforcement learning algorithm of CVaR-Proximal-Policy-Optimization (CPPO) which formalizes the risk-sensitive constrained optimization problem by keeping its CVaR under a given threshold. Experimental results show that CPPO achieves a higher cumulative reward and is more robust against both observation and transition disturbances on a series of continuous control tasks in MuJoCo.

----

## [510] EGCN: An Ensemble-based Learning Framework for Exploring Effective Skeleton-based Rehabilitation Exercise Assessment

**Authors**: *Bruce X. B. Yu, Yan Liu, Xiang Zhang, Gong Chen, Keith C. C. Chan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/511](https://doi.org/10.24963/ijcai.2022/511)

**Abstract**:

Recently, some skeleton-based physical therapy systems have been attempted to automatically evaluate the correctness or quality of an exercise performed by rehabilitation subjects. However, in terms of algorithms and evaluation criteria, the task remains not fully explored regarding making full use of different skeleton features. To advance the prior work, we propose a learning framework called Ensemble-based Graph Convolutional Network (EGCN) for skeleton-based rehabilitation exercise assessment. As far as we know, this is the first attempt that utilizes both two skeleton feature groups and investigates different ensemble strategies for the task. We also examine the properness of existing evaluation criteria and focus on evaluating the prediction ability of our proposed method. We then conduct extensive cross-validation experiments on two latest public datasets: UI-PRMD and KIMORE. Results indicate that the model-level ensemble scheme of our EGCN achieves better performance than existing methods. Code is available: https://github.com/bruceyo/EGCN.

----

## [511] Robust Weight Perturbation for Adversarial Training

**Authors**: *Chaojian Yu, Bo Han, Mingming Gong, Li Shen, Shiming Ge, Du Bo, Tongliang Liu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/512](https://doi.org/10.24963/ijcai.2022/512)

**Abstract**:

Overfitting widely exists in adversarial robust training of deep networks. An effective remedy is adversarial weight perturbation, which injects the worst-case weight perturbation during network training by maximizing the classification loss on adversarial examples. Adversarial weight perturbation helps reduce the robust generalization gap; however, it also undermines the robustness improvement. A criterion that regulates the weight perturbation is therefore crucial for adversarial training. In this paper, we propose such a criterion, namely Loss Stationary Condition (LSC) for constrained perturbation. With LSC, we find that it is essential to conduct weight perturbation on adversarial data with small classification loss to eliminate robust overfitting. Weight perturbation on adversarial data with large classification loss is not necessary and may even lead to poor robustness. Based on these observations, we propose a robust perturbation strategy to constrain the extent of weight perturbation. The perturbation strategy prevents deep networks from overfitting while avoiding the side effect of excessive weight perturbation,  significantly improving the robustness of adversarial training. Extensive experiments demonstrate the superiority of the proposed method over the state-of-the-art adversarial training methods.

----

## [512] Masked Feature Generation Network for Few-Shot Learning

**Authors**: *Yunlong Yu, Dingyi Zhang, Zhong Ji*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/513](https://doi.org/10.24963/ijcai.2022/513)

**Abstract**:

In this paper, we present a feature-augmentation approach called Masked Feature Generation Network (MFGN) for Few-Shot Learning (FSL), a challenging task that attempts to recognize the novel classes with a few visual instances for each class. Most of the feature-augmentation approaches tackle FSL tasks via modeling the intra-class distributions. We extend this idea further to explicitly capture the intra-class variations in a one-to-many manner. Specifically, MFGN consists of an encoder-decoder architecture, with an encoder that performs as a feature extractor and extracts the feature embeddings of the available visual instances (the unavailable instances are seen to be masked), along with a decoder that performs as a feature generator and reconstructs the feature embeddings of the unavailable visual instances from both the available feature embeddings and the masked tokens. Equipped with this generative architecture, MFGN produces nontrivial visual features for the novel classes with limited visual instances. In extensive experiments on four FSL benchmarks, MFGN performs competitively and outperforms the state-of-the-art competitors on most of the few-shot classification tasks.

----

## [513] Don't Touch What Matters: Task-Aware Lipschitz Data Augmentation for Visual Reinforcement Learning

**Authors**: *Zhecheng Yuan, Guozheng Ma, Yao Mu, Bo Xia, Bo Yuan, Xueqian Wang, Ping Luo, Huazhe Xu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/514](https://doi.org/10.24963/ijcai.2022/514)

**Abstract**:

One of the key challenges in visual Reinforcement Learning (RL) is to learn policies that can generalize to unseen environments. Recently, data augmentation techniques aiming at enhancing data diversity have demonstrated proven performance in improving the generalization ability of learned policies. However, due to the sensitivity of RL training, naively applying data augmentation, which transforms each pixel in a task-agnostic manner, may suffer from instability and damage the sample efficiency, thus further exacerbating the generalization performance. At the heart of this phenomenon is the diverged action distribution and high-variance value estimation in the face of augmented images. To alleviate this issue, we propose Task-aware Lipschitz Data Augmentation (TLDA) for visual RL, which  explicitly identifies the task-correlated pixels with large Lipschitz constants, and only augments the task-irrelevant pixels for stability. We verify the effectiveness of our approach on DeepMind Control suite, CARLA and DeepMind Manipulation tasks. The extensive empirical results show that TLDA improves both sample efficiency and generalization; it outperforms previous state-of-the-art methods across 3 different visual control benchmarks.

----

## [514] Improved Pure Exploration in Linear Bandits with No-Regret Learning

**Authors**: *Mohammadi Zaki, Avi Mohan, Aditya Gopalan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/515](https://doi.org/10.24963/ijcai.2022/515)

**Abstract**:

We study the best arm identification problem in linear multi-armed bandits (LMAB) in the fixed confidence setting ; this is also the problem of optimizing an unknown linear function over a discrete domain with noisy, zeroth-order access. We propose an explicitly implementable and provably order-optimal sample-complexity algorithm for best arm identification. Existing approaches that achieve optimal sample complexity assume either access to a nontrivial minimax optimization oracle (e.g. RAGE and Lazy-TS) or knowledge of an upper-bound on the norm of the unknown parameter vector(e.g. LinGame). Our algorithm, which we call the Phased Elimination Linear Exploration Game (PELEG), maintains a high-probability confidence ellipsoid containing the unknown vector, and uses it to eliminate suboptimal arms in phases, where a minimax problem is essentially solved in each phase using two-player low regret learners. PELEG does not require to know a bound on norm of the unknown vector, and is asymptotically-optimal, settling an open problem. We show that the sample complexity of PELEG matches, up to order and in a non-asymptotic sense, an instance-dependent universal lower bound for linear bandits. PELEG is thus the first algorithm to achieve both order-optimal sample complexity and explicit implementability for this setting. We also provide numerical results for the proposed algorithm consistent with its theoretical guarantees.

----

## [515] Model-Based Offline Planning with Trajectory Pruning

**Authors**: *Xianyuan Zhan, Xiangyu Zhu, Haoran Xu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/516](https://doi.org/10.24963/ijcai.2022/516)

**Abstract**:

The recent offline reinforcement learning (RL) studies have achieved much progress to make RL usable in real-world systems by learning policies from pre-collected datasets without environment interaction. Unfortunately, existing offline RL methods still face many practical challenges in real-world system control tasks, such as computational restriction during agent training and the requirement of extra control flexibility. The model-based planning framework provides an attractive alternative. However, most model-based planning algorithms are not designed for offline settings. Simply combining the ingredients of offline RL with existing methods either provides over-restrictive planning or leads to inferior performance. We propose a new light-weighted model-based offline planning framework, namely MOPP, which tackles the dilemma between the restrictions of offline learning and high-performance planning. MOPP encourages more aggressive trajectory rollout guided by the behavior policy learned from data, and prunes out problematic trajectories to avoid potential out-of-distribution samples. Experimental results show that MOPP provides competitive performance compared with existing model-based offline planning and RL approaches.

----

## [516] Hyperbolic Knowledge Transfer with Class Hierarchy for Few-Shot Learning

**Authors**: *Baoquan Zhang, Hao Jiang, Shanshan Feng, Xutao Li, Yunming Ye, Rui Ye*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/517](https://doi.org/10.24963/ijcai.2022/517)

**Abstract**:

Few-shot learning (FSL) aims to recognize a novel class with very few instances, which is a challenging task since it suffers from a data scarcity issue. One way to effectively alleviate this issue is introducing explicit knowledge summarized from human past experiences to achieve knowledge transfer for FSL. Based on this idea, in this paper, we introduce the explicit knowledge of class hierarchy (i.e., the hierarchy relations between classes) as FSL priors and propose a novel hyperbolic knowledge transfer framework for FSL, namely, HyperKT. Our insight is, in the hyperbolic space, the hierarchy relation between classes can be well preserved by resorting to the exponential growth characters of hyperbolic volume, so that better knowledge transfer can be achieved for FSL. Specifically, we first regard the class hierarchy as a tree-like structure. Then, 1) a hyperbolic representation learning module and a hyperbolic prototype inference module are employed to encode/infer each image and class prototype to the hyperbolic space, respectively; and 2) a novel hierarchical classification and relation reconstruction loss are carefully designed to learn the class hierarchy. Finally, the novel class prediction is performed in a nearest-prototype manner. Extensive experiments on three datasets show our method achieves superior performance over state-of-the-art methods, especially on 1-shot tasks.

----

## [517] Fine-Tuning Graph Neural Networks via Graph Topology Induced Optimal Transport

**Authors**: *Jiying Zhang, Xi Xiao, Long-Kai Huang, Yu Rong, Yatao Bian*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/518](https://doi.org/10.24963/ijcai.2022/518)

**Abstract**:

Recently, the pretrain-finetuning paradigm has attracted tons of attention in graph learning community due to its power of alleviating the lack of labels problem in many real-world applications. Current studies use existing techniques, such as weight constraint, representation constraint, which are derived from images or text data, to transfer the invariant knowledge from the pre-train stage to fine-tuning stage. However, these methods failed to preserve invariances from graph structure and Graph Neural Network (GNN) style models. In this paper, we present a novel optimal transport-based fine-tuning framework called GTOT-Tuning, namely, Graph Topology induced Optimal Transport fine-Tuning, for GNN style backbones. GTOT-Tuning is required to utilize the property of graph data to enhance the preservation of representation produced by fine-tuned networks. Toward this goal, we formulate graph local knowledge transfer as an Optimal Transport (OT) problem with a structural prior and construct the GTOT regularizer to constrain the fine-tuned model behaviors. By using the adjacency relationship amongst nodes, the GTOT regularizer achieves node-level optimal transport procedures and reduces redundant transport procedures, resulting in efficient knowledge transfer from the pre-trained models. We evaluate GTOT-Tuning on eight downstream tasks with various GNN backbones and demonstrate that it achieves state-of-the-art fine-tuning performance for GNNs.

----

## [518] Hierarchical Diffusion Scattering Graph Neural Network

**Authors**: *Ke Zhang, Xinyan Pu, Jiaxing Li, Jiasong Wu, Huazhong Shu, Youyong Kong*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/519](https://doi.org/10.24963/ijcai.2022/519)

**Abstract**:

Graph neural network (GNN) is popular now to solve the tasks in non-Euclidean space and most of them learn deep embeddings by aggregating the neighboring nodes. However, these methods are prone to some problems such as over-smoothing because of the single-scale perspective field and the nature of low-pass filter. To address these limitations, we introduce diffusion scattering network (DSN) to exploit high-order patterns. With observing the complementary relationship between multi-layer GNN and DSN, we propose Hierarchical Diffusion Scattering Graph Neural Network (HDS-GNN) to efficiently bridge DSN and GNN layer by layer to supplement GNN with multi-scale information and band-pass signals. Our model extracts node-level scattering representations by intercepting the low-pass filtering, and adaptively tunes the different scales to regularize multi-scale information. Then we apply hierarchical representation enhancement to improve GNN with the scattering features. We benchmark our model on nine real-world networks on the transductive semi-supervised node classification task. The experimental results demonstrate the effectiveness of our method.

----

## [519] Penalized Proximal Policy Optimization for Safe Reinforcement Learning

**Authors**: *Linrui Zhang, Li Shen, Long Yang, Shixiang Chen, Xueqian Wang, Bo Yuan, Dacheng Tao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/520](https://doi.org/10.24963/ijcai.2022/520)

**Abstract**:

Safe reinforcement learning aims to learn the optimal policy while satisfying safety constraints, which is essential in real-world applications. However, current algorithms still struggle for efficient policy updates with hard constraint satisfaction. In this paper, we propose Penalized Proximal Policy Optimization (P3O), which solves the cumbersome constrained policy iteration via a single minimization of an equivalent unconstrained problem. Specifically, P3O utilizes a simple yet effective penalty approach to eliminate cost constraints and removes the trust-region constraint by the clipped surrogate objective. We theoretically prove the exactness of the penalized method with a finite penalty factor and provide a worst-case analysis for approximate error when evaluated on sample trajectories. Moreover, we extend P3O to more challenging multi-constraint and multi-agent scenarios which are less studied in previous work. Extensive experiments show that P3O outperforms state-of-the-art algorithms with respect to both reward improvement and constraint satisfaction on a set of constrained locomotive tasks.

----

## [520] Next Point-of-Interest Recommendation with Inferring Multi-step Future Preferences

**Authors**: *Lu Zhang, Zhu Sun, Ziqing Wu, Jie Zhang, Yew Soon Ong, Xinghua Qu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/521](https://doi.org/10.24963/ijcai.2022/521)

**Abstract**:

Existing studies on next point-of-interest (POI) recommendation mainly attempt to learn user preference from the past and current sequential behaviors. They, however, completely ignore the impact of future behaviors on the decision-making, thus hindering the quality of user preference learning. Intuitively, users' next POI visits may also be affected by their multi-step future behaviors, as users may often have activity planning in mind. To fill this gap, we propose a novel Context-aware Future Preference inference Recommender (CFPRec) to help infer user future preference in a self-ensembling manner. In particular, it delicately derives multi-step future preferences from the learned past preference thanks to the periodic property of users' daily check-ins, so as to implicitly mimic userâ€™s activity planning before her next visit. The inferred future preferences are then seamlessly integrated with the current preference for more expressive user preference learning. Extensive experiments on three datasets demonstrate the superiority of CFPRec against state-of-the-arts.

----

## [521] Het2Hom: Representation of Heterogeneous Attributes into Homogeneous Concept Spaces for Categorical-and-Numerical-Attribute Data Clustering

**Authors**: *Yiqun Zhang, Yiu-ming Cheung, An Zeng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/522](https://doi.org/10.24963/ijcai.2022/522)

**Abstract**:

Data sets composed of a mixture of categorical and numerical attributes (also called mixed data hereinafter) are common in real-world cluster analysis. However, insightful analysis of such data under an unsupervised scenario using clustering is extremely challenging because the information provided by the two different types of attributes is heterogeneous, being at different concept hierarchies. That is, the values of a categorical attribute represent a set of different concepts (e.g., professor, lawyer, and doctor of the attribute "occupation"), while the values of a numerical attribute describe the tendencies toward two different concepts (e.g., low and high of the attribute "income"). To appropriately use such heterogeneous information in clustering, this paper therefore proposes a novel attribute representation learning method called Het2Hom, which first converts the heterogeneous attributes into a homogeneous form, and then learns attribute representations and data partitions on such a homogeneous basis. Het2Hom features low time complexity and intuitive interpretability. Extensive experiments show that Het2Hom outperforms the state-of-the-art counterparts.

----

## [522] Learning Mixture of Neural Temporal Point Processes for Multi-dimensional Event Sequence Clustering

**Authors**: *Yunhao Zhang, Junchi Yan, Xiaolu Zhang, Jun Zhou, Xiaokang Yang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/523](https://doi.org/10.24963/ijcai.2022/523)

**Abstract**:

Multi-dimensional event sequence clustering applies to many scenarios e.g. e-Commerce and electronic health. Traditional clustering models fail to characterize complex real-world processes due to the strong parametric assumption. While Neural Temporal Point Processes (NTPPs) mainly focus on modeling similar sequences instead of clustering. To fill the gap, we propose Mixture of Neural Temporal Point Processes (NTPP-MIX), a  general framework that can utilize many existing NTPPs for multi-dimensional event sequence clustering. In NTPP-MIX, the prior distribution of coefficients for cluster assignment is modeled by a Dirichlet distribution. When the assignment is given, the conditional probability of a sequence is modeled by the mixture of a series of NTPPs. We combine variational EM algorithm and Stochastic Gradient Descent (SGD) to efficiently train the framework. In E-step, we fix parameters for NTPPs and approximate the true posterior with variational distributions. In M-step, we fix variational distributions and use SGD to update parameters of NTPPs. Extensive experimental results on four synthetic datasets and three real-world datasets show the effectiveness of NTPP-MIX against state-of-the-arts.

----

## [523] Fusion Label Enhancement for Multi-Label Learning

**Authors**: *Xingyu Zhao, Yuexuan An, Ning Xu, Xin Geng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/524](https://doi.org/10.24963/ijcai.2022/524)

**Abstract**:

Multi-label learning (MLL) refers to the problem of tagging a given instance with a set of relevant labels. In MLL, the implicit relative importance of different labels representing a single instance is generally different, which recently gained considerable attention and should be fully leveraged. Therefore, label enhancement (LE) has been widely applied in various MLL tasks as the ability to effectively mine the implicit relative importance information of different labels. However, due to the fact that the label enhancement process in previous LE-based MLL methods is decoupled from the training process on the predictive models, the objective of LE does not match the training process and finally affects the whole learning system. In this paper, we propose a novel approach named Fusion Label Enhancement for Multi-label learning (FLEM) to effectively integrate the LE process and the training process. Specifically, we design a matching and interaction mechanism which leverages a novel interaction label enhancement loss to avoid that the recovered label distribution does not match the need of the predictive model. In the meantime, we present a unified label distribution loss for establishing the corresponding relationship between the recovered label distribution and the training of the predictive model. With the proposed loss, the label distributions recovered from the LE process can be efficiently utilized for training the predictive model. Experimental results on multiple benchmark datasets validate the effectiveness of the proposed approach.

----

## [524] Learning Mixtures of Random Utility Models with Features from Incomplete Preferences

**Authors**: *Zhibing Zhao, Ao Liu, Lirong Xia*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/525](https://doi.org/10.24963/ijcai.2022/525)

**Abstract**:

Random Utility Models (RUMs), which subsume Plackett-Luce model (PL) as a special case, are among the most popular models for preference learning. In this paper, we consider RUMs with features and their mixtures, where each alternative has a vector of features, possibly different across agents. Such models significantly generalize the standard PL and RUMs, but are not as well investigated in the literature. We extend mixtures of RUMs with features to models that generate incomplete preferences and characterize their identifiability. For PL, we prove that when PL with features is identifiable, its MLE is consistent with a strictly concave objective function under mild assumptions, by characterizing a bound on root-mean-square-error (RMSE), which naturally leads to a sample complexity bound. We also characterize identifiability of more general RUMs with features and propose a generalized RBCML to learn them. Our experiments on synthetic data demonstrate the effectiveness of MLE on PL with features with tradeoffs between statistical efficiency and computational efficiency. Our experiments on real-world data show the prediction power of PL with features and its mixtures.

----

## [525] Unsupervised Voice-Face Representation Learning by Cross-Modal Prototype Contrast

**Authors**: *Boqing Zhu, Kele Xu, Changjian Wang, Zheng Qin, Tao Sun, Huaimin Wang, Yuxing Peng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/526](https://doi.org/10.24963/ijcai.2022/526)

**Abstract**:

We present an approach to learn voice-face representations from the talking face videos, without any identity labels. Previous works employ cross-modal instance discrimination tasks to establish the correlation of voice and face. These methods neglect the semantic content of different videos, introducing false-negative pairs as training noise. Furthermore, the positive pairs are constructed based on the natural correlation between audio clips and visual frames. However, this correlation might be weak or inaccurate in a large amount of real-world data, which leads to deviating positives into the contrastive paradigm. To address these issues, we propose the cross-modal prototype contrastive learning (CMPC), which takes advantage of contrastive methods and resists adverse effects of false negatives and deviate positives. On one hand, CMPC could learn the intra-class invariance by constructing semantic-wise positives via unsupervised clustering in different modalities. On the other hand, by comparing the similarities of cross-modal instances from that of cross-modal prototypes, we dynamically recalibrate the unlearnable instances' contribution to overall loss. Experiments show that the proposed approach outperforms state-of-the-art unsupervised methods on various voice-face association evaluation protocols. Additionally, in the low-shot supervision setting, our method also has a significant improvement compared to previous instance-wise contrastive learning.

----

## [526] RoSA: A Robust Self-Aligned Framework for Node-Node Graph Contrastive Learning

**Authors**: *Yun Zhu, Jianhao Guo, Fei Wu, Siliang Tang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/527](https://doi.org/10.24963/ijcai.2022/527)

**Abstract**:

Graph contrastive learning has gained significant progress recently. However, existing works have rarely explored non-aligned node-node contrasting. In this paper, we propose a novel graph contrastive learning method named RoSA that focuses on utilizing non-aligned augmented views for node-level representation learning. First, we leverage the earth mover's distance to model the minimum effort to transform the distribution of one view to the other as our contrastive objective, which does not require alignment between views. Then we introduce adversarial training as an auxiliary method to increase sampling diversity and enhance the robustness of our model. Experimental results show that RoSA outperforms a series of graph contrastive learning frameworks on homophilous, non-homophilous and dynamic graphs, which validates the effectiveness of our work. To the best of our awareness, RoSA is the first work focuses on the non-aligned node-node graph contrastive learning problem. Our codes are available at: https://github.com/ZhuYun97/RoSA

----

## [527] Multi-Constraint Deep Reinforcement Learning for Smooth Action Control

**Authors**: *Guangyuan Zou, Ying He, F. Richard Yu, Longquan Chen, Weike Pan, Zhong Ming*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/528](https://doi.org/10.24963/ijcai.2022/528)

**Abstract**:

Deep reinforcement learning (DRL) has been studied in a variety of challenging decision-making tasks, e.g., autonomous driving. \textcolor{black}{However, DRL typically suffers from the action shaking problem, which means that agents can select actions with big difference even though states only slightly differ.} One of the crucial reasons for this issue is the inappropriate design of the reward in DRL. In this paper, to address this issue, we propose a novel way to incorporate the smoothness of actions in the reward. Specifically, we introduce sub-rewards and add multiple constraints related to these sub-rewards. In addition, we propose a multi-constraint proximal policy optimization (MCPPO) method to solve the multi-constraint DRL problem. Extensive simulation results show that the proposed MCPPO method has better action smoothness compared with the traditional proportional-integral-differential (PID) and mainstream DRL algorithms. The video is available at https://youtu.be/F2jpaSm7YOg.

----

## [528] Subsequence-based Graph Routing Network for Capturing Multiple Risk Propagation Processes

**Authors**: *Rui Cheng, Qing Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/529](https://doi.org/10.24963/ijcai.2022/529)

**Abstract**:

In finance, the risk of an entity depends not only on its historical information 
but also on the risk propagated by its related peers. Pilot studies rely on Graph Neural Networks (GNNs) to model this risk propagation, where each entity is treated as a node and represented by its time-series information.
However, conventional GNNs are constrained by their unified messaging mechanism with an assumption that the risk of a given entity only propagates to its related peers with the same time lag and has the same effect, which is against the ground truth. In this study, we propose the subsequence-based graph routing network (S-GRN) for capturing the variant risk propagation processes among different time-series represented entities. In S-GRN, the messaging mechanism between each node pair is dynamically and independently selected from multiple messaging mechanisms based on the dependencies of variant subsequence patterns. The S-GRN is extensively evaluated on two synthetic tasks and three real-world datasets and demonstrates state-of-the-art performance.

----

## [529] 3E-Solver: An Effortless, Easy-to-Update, and End-to-End Solver with Semi-Supervised Learning for Breaking Text-Based Captchas

**Authors**: *Xianwen Deng, Ruijie Zhao, Yanhao Wang, Libo Chen, Yijun Wang, Zhi Xue*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/530](https://doi.org/10.24963/ijcai.2022/530)

**Abstract**:

Text-based captchas are the most widely used security mechanism currently. Due to the limitations and specificity of the segmentation algorithm, the early segmentation-based attack method has been unable to deal with the current captchas with newly introduced security features (e.g., occluding lines and overlapping). Recently, some works have designed captcha solvers based on deep learning methods with powerful feature extraction capabilities, which have greater generality and higher accuracy. However, these works still suffer from two main intrinsic limitations: (1) many labor costs are required to label the training data, and (2) the solver cannot be updated with unlabeled data to recognize captchas more accurately. In this paper, we present a novel solver using improved FixMatch for semi-supervised captcha recognition to tackle these problems. Specifically, we first build an end-to-end baseline model to effectively break text-based captchas by leveraging encoder-decoder architecture and attention mechanism. Then we construct our solver with a few labeled samples and many unlabeled samples by improved FixMatch, which introduces teacher forcing, adaptive batch normalization, and consistency loss to achieve more effective training. Experiment results show that our solver outperforms state-of-the-arts by a large margin on current captcha schemes. We hope that our work can help security experts to revisit the design and usability of text-based captchas. The source code of this work is available at https://github.com/SJTU-dxw/3E-Solver-CAPTCHA.

----

## [530] Placing Green Bridges Optimally, with Habitats Inducing Cycles

**Authors**: *Maike Herkenrath, Till Fluschnik, Francesco Grothe, Leon Kellerhals*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/531](https://doi.org/10.24963/ijcai.2022/531)

**Abstract**:

Choosing the placement of wildlife crossings (i.e., green bridges) to reconnect animal species' fragmented habitats is among the 17 goals towards sustainable development by the UN. We consider the following established model: Given a graph whose vertices represent the fragmented habitat areas and whose weighted edges represent possible green bridge locations, as well as the habitable vertex set for each species, find the cheapest set of edges such that each species' habitat is connected. We study this problem from a theoretical (algorithms and complexity) and an experimental perspective, while focusing on the case where habitats induce cycles. We prove that the NP-hardness persists in this case even if the graph structure is restricted. If the habitats additionally induce faces in plane graphs however, the problem becomes efficiently solvable. In our empirical evaluation we compare this algorithm as well as ILP formulations for more general variants and an approximation algorithm with another. Our evaluation underlines that each specialization is beneficial in terms of running time, whereas the approximation provides highly competitive solutions in practice.

----

## [531] Membership Inference via Backdooring

**Authors**: *Hongsheng Hu, Zoran Salcic, Gillian Dobbie, Jinjun Chen, Lichao Sun, Xuyun Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/532](https://doi.org/10.24963/ijcai.2022/532)

**Abstract**:

Recently issued data privacy regulations like GDPR (General Data Protection Regulation) grant individuals the right to be forgotten. In the context of machine learning, this requires a model to forget about a training data sample if requested by the data owner (i.e., machine unlearning). As an essential step prior to machine unlearning, it is still a challenge for a data owner to tell whether or not her data have been used by an unauthorized party to train a machine learning model. Membership inference is a recently emerging technique to identify whether a data sample was used to train a target model, and seems to be a promising solution to this challenge. However, straightforward adoption of existing membership inference approaches fails to address the challenge effectively due to being originally designed for attacking membership privacy and suffering from several severe limitations such as low inference accuracy on well-generalized models. In this paper, we propose a novel membership inference approach inspired by the backdoor technology to address the said challenge. Specifically, our approach of Membership Inference via Backdooring (MIB) leverages the key observation that a backdoored model behaves very differently from a clean model when predicting on deliberately marked samples created by a data owner. Appealingly, MIB requires data owners' marking a small number of samples for membership inference and only black-box access to the target model, with theoretical guarantees for inference results. We perform extensive experiments on various datasets and deep neural network architectures, and the results validate the efficacy of our approach, e.g., marking only 0.1% of the training dataset is practically sufficient for effective membership inference.

----

## [532] A Universal PINNs Method for Solving Partial Differential Equations with a Point Source

**Authors**: *Xiang Huang, Hongsheng Liu, Beiji Shi, Zidong Wang, Kang Yang, Yang Li, Min Wang, Haotian Chu, Jing Zhou, Fan Yu, Bei Hua, Bin Dong, Lei Chen*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/533](https://doi.org/10.24963/ijcai.2022/533)

**Abstract**:

In recent years, deep learning technology has been used to solve partial differential equations (PDEs), among which the physics-informed neural networks (PINNs)method emerges to be a promising method for solving both forward and inverse PDE problems. PDEs with a point source that is expressed as a Dirac delta function in the governing equations are mathematical models of many physical processes. However, they cannot be solved directly by conventional PINNs method due to the singularity brought by the Dirac delta function. In this paper, we propose a universal solution to tackle this problem by proposing three novel techniques.  Firstly the Dirac delta function is modeled as a continuous probability density function to eliminate the singularity at the point source; secondly a lower bound constrained uncertainty weighting algorithm is proposed to balance the physics-informed loss terms of point source area and the remaining areas; and thirdly a multi-scale deep neural network with periodic activation function is used to improve the accuracy and convergence speed. We evaluate the proposed method with three representative PDEs, and the experimental results show that our method outperforms existing deep learning based methods with respect to the accuracy, the efficiency and the versatility.

----

## [533] A Polynomial-time Decentralised Algorithm for Coordinated Management of Multiple Intersections

**Authors**: *Tatsuya Iwase, Sebastian Stein, Enrico H. Gerding, Archie Chapman*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/534](https://doi.org/10.24963/ijcai.2022/534)

**Abstract**:

Autonomous intersection management has the potential to reduce road traffic congestion and energy consumption. To realize this potential, efficient algorithms are needed. 
However, most existing studies locally optimize one intersection at a time, and this can cause negative externalities on the traffic network as a whole.
Here, we focus on coordinating multiple intersections,
and formulate the problem as a distributed constraint optimisation problem (DCOP). We consider three utility design approaches that trade off efficiency and fairness.
Our polynomial-time algorithm for coordinating multiple intersections reduces the traffic delay by about 41 percentage points compared to independent single intersection management approaches.

----

## [534] Multi-Agent Reinforcement Learning for Traffic Signal Control through Universal Communication Method

**Authors**: *Qize Jiang, Minhao Qin, Shengmin Shi, Weiwei Sun, Baihua Zheng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/535](https://doi.org/10.24963/ijcai.2022/535)

**Abstract**:

How to coordinate the communication among intersections effectively in real complex traffic scenarios with multi-intersection is challenging.  Existing approaches only enable the communication in a heuristic manner without considering the content/importance of information to be shared. In this paper, we propose a universal communication form UniComm between intersections. UniComm embeds massive observations collected at one agent into crucial predictions of their impact on its neighbors, which improves the communication efficiency and is universal across existing methods. We also propose a concise network UniLight to make full use of communications enabled by UniComm.  Experimental results on real datasets demonstrate that UniComm universally improves the performance of existing state-of-the-art methods, and UniLight significantly outperforms existing methods on a wide range of traffic situations. Source codes are available at https://github.com/zyr17/UniLight.

----

## [535] Cumulative Stay-time Representation for Electronic Health Records in Medical Event Time Prediction

**Authors**: *Takayuki Katsuki, Kohei Miyaguchi, Akira Koseki, Toshiya Iwamori, Ryosuke Yanagiya, Atsushi Suzuki*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/536](https://doi.org/10.24963/ijcai.2022/536)

**Abstract**:

We address the problem of predicting when a disease will develop, i.e., medical event time (MET), from a patient's electronic health record (EHR).
The MET of non-communicable diseases like diabetes is highly correlated to cumulative health conditions, more specifically, how much time the patient spent with specific health conditions in the past.
The common time-series representation is indirect in extracting such information from EHR because it focuses on detailed dependencies between values in successive observations, not cumulative information.
We propose a novel data representation for EHR called cumulative stay-time representation (CTR), which directly models such cumulative health conditions.
We derive a trainable construction of CTR based on neural networks that has the flexibility to fit the target data and scalability to handle high-dimensional EHR.
Numerical experiments using synthetic and real-world datasets demonstrate that CTR alone achieves a high prediction performance, and it enhances the performance of existing models when combined with them.

----

## [536] Self-Supervised Learning with Attention-based Latent Signal Augmentation for Sleep Staging with Limited Labeled Data

**Authors**: *Harim Lee, Eunseon Seong, Dong-Kyu Chae*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/537](https://doi.org/10.24963/ijcai.2022/537)

**Abstract**:

Sleep staging is an important task that enables sleep quality assessment and disorder diagnosis. Due to dependency on manually labeled data, many researches have turned from supervised approaches to self-supervised learning (SSL) for sleep staging. While existing SSL methods have made significant progress in terms of its comparable performance to supervised methods, there are still some limitations. Contrastive learning could potentially lead to false negative pair assignments in sleep signal data. Moreover, existing data augmentation techniques directly modify the original signal data, making it likely to lose important information. To mitigate these issues, we propose Self-Supervised Learning with Attention-aided Positive Pairs (SSLAPP). Instead of the contrastive learning, SSLAPP carefully draws high-quality positive pairs and exploits them in representation learning. Here, we propose attention-based latent signal augmentation, which plays a key role by capturing important features without losing valuable signal information. Experimental results show that our proposed method achieves state-of-the-art performance in sleep stage classification with limited labeled data. The code is available at: https://github.com/DILAB-HYU/SSLAPP

----

## [537] Learning Curricula for Humans: An Empirical Study with Puzzles from The Witness

**Authors**: *Levi H. S. Lelis, João Gabriel Gama Vila Nova, Eugene Chen, Nathan R. Sturtevant, Carrie Demmans Epp, Michael Bowling*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/538](https://doi.org/10.24963/ijcai.2022/538)

**Abstract**:

The combination of tree search and neural networks has achieved super-human performance in challenging domains. We are interested in transferring to humans the knowledge these learning systems generate. We hypothesize the process in which neural-guided tree search algorithms learn how to solve a set of problems can be used to generate curricula for helping human learners. In this paper we show how the Bootstrap learning system can be modified to learn curricula for humans in a puzzle domain. We evaluate our system in two curriculum learning settings. First, given a small set of problem instances, our system orders the instances to ease the learning process of human learners. Second, given a large set of problem instances, our system returns a small ordered subset of the initial set that can be presented to human learners. We evaluate our curricula with a user study where participants learn how to solve a class of puzzles from the game `The Witness.' The user-study results suggest one of the curricula our system generates compares favorably with simple baselines and is competitive with the curriculum from the original `The Witness' game in terms of user retention and effort.

----

## [538] Transformer-based Objective-reinforced Generative Adversarial Network to Generate Desired Molecules

**Authors**: *Chen Li, Chikashige Yamanaka, Kazuma Kaitoh, Yoshihiro Yamanishi*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/539](https://doi.org/10.24963/ijcai.2022/539)

**Abstract**:

Deep generative models of sequence-structure data have attracted widespread attention in drug discovery. However, such models cannot fully extract the semantic features of molecules from sequential representations. Moreover, mode collapse reduces the diversity of the generated molecules. This paper proposes a transformer-based objective-reinforced generative adversarial network (TransORGAN) to generate molecules. TransORGAN leverages a transformer architecture as a generator and uses a stochastic policy gradient for reinforcement learning to generate plausible molecules with rich semantic features. The discriminator grants rewards that guide the policy update of the generator, while an objective-reinforced penalty encourages the generation of diverse molecules. Experiments were performed using the ZINC chemical dataset, and the results demonstrated the usefulness of TransORGAN in terms of uniqueness, novelty, and diversity of the generated molecules.

----

## [539] Towards Controlling the Transmission of Diseases: Continuous Exposure Discovery over Massive-Scale Moving Objects

**Authors**: *Ke Li, Lisi Chen, Shuo Shang, Haiyan Wang, Yang Liu, Panos Kalnis, Bin Yao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/540](https://doi.org/10.24963/ijcai.2022/540)

**Abstract**:

Infectious diseases have been recognized as major public health concerns for decades. Close contact  discovery is playing an indispensable role in preventing epidemic transmission. In this light, we study the continuous exposure search problem: Given a collection of moving objects and a collection of moving queries, we continuously discover all objects that have been directly and indirectly exposed to at least one query over a period of time. Our problem targets a variety of applications, including but not limited to disease control, epidemic pre-warning, information spreading, and co-movement mining. To answer this problem, we develop an exact group processing algorithm with optimization strategies. Further, we propose an approximate algorithm that substantially improves the efficiency without false dismissal. Extensive experiments offer insight into effectiveness and efficiency of our proposed algorithms.

----

## [540] Distilling Governing Laws and Source Input for Dynamical Systems from Videos

**Authors**: *Lele Luan, Yang Liu, Hao Sun*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/541](https://doi.org/10.24963/ijcai.2022/541)

**Abstract**:

Distilling interpretable physical laws from videos has led to expanded interest in the computer vision community recently thanks to the advances in deep learning, but still remains a great challenge. This paper introduces an end-to-end unsupervised deep learning framework to uncover the explicit governing equations of dynamics presented by moving object(s), based on recorded videos. Instead in the pixel (spatial) coordinate system of image space, the physical law is modeled in a regressed underlying physical coordinate system where the physical states follow potential explicit governing equations. A numerical integrator-based sparse regression module is designed and serves as a physical constraint to the autoencoder and coordinate system regression, and, in the meanwhile, uncover the parsimonious closed-form governing equations from the learned physical states. Experiments on simulated dynamical scenes show that the proposed method is able to distill closed-form governing equations and simultaneously identify unknown excitation input for several dynamical systems recorded by videos, which fills in the gap in literature where no existing methods are available and applicable for solving this type of problem.

----

## [541] Monolith to Microservices: Representing Application Software through Heterogeneous Graph Neural Network

**Authors**: *Alex Mathai, Sambaran Bandyopadhyay, Utkarsh Desai, Srikanth Tamilselvam*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/542](https://doi.org/10.24963/ijcai.2022/542)

**Abstract**:

Monolithic software encapsulates all functional capabilities into a single deployable unit. But managing it becomes harder as the demand for new functionalities grow. Microservice architecture is seen as an alternative as it advocates building an application through a set of loosely coupled small services wherein each service owns a single functional responsibility. But the challenges associated with the separation of functional modules, slows down the migration of a monolithic code into microservices. In this work, we propose a representation learning based solution to tackle this problem. We use a heterogeneous graph to jointly represent software artifacts (like programs and resources) and the different relationships they share (function calls, inheritance, etc.), and perform a constraint-based clustering through a novel heterogeneous graph neural network. Experimental studies show that our approach is effective on monoliths of different types.

----

## [542] Learn Continuously, Act Discretely: Hybrid Action-Space Reinforcement Learning For Optimal Execution

**Authors**: *Feiyang Pan, Tongzhe Zhang, Ling Luo, Jia He, Shuoling Liu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/543](https://doi.org/10.24963/ijcai.2022/543)

**Abstract**:

Optimal execution is a sequential decision-making problem for cost-saving in algorithmic trading. Studies have found that reinforcement learning (RL) can help decide the order-splitting sizes. However, a problem remains unsolved: how to place limit orders at appropriate limit prices?
The key challenge lies in the ``continuous-discrete duality'' of the action space. On the one hand, the continuous action space using percentage changes in prices is preferred for generalization. On the other hand, the trader eventually needs to choose limit prices discretely due to the existence of the tick size, which requires specialization for every single stock with different characteristics (e.g., the liquidity and the price range). So we need continuous control for generalization and discrete control for specialization. To this end, we propose a hybrid RL method to combine the advantages of both of them. We first use a continuous control agent to scope an action subset, then deploy a fine-grained agent to choose a specific limit price. Extensive experiments show that our method has higher sample efficiency and better training stability than existing RL algorithms and significantly outperforms previous learning-based methods for order execution.

----

## [543] Communicative Subgraph Representation Learning for Multi-Relational Inductive Drug-Gene Interaction Prediction

**Authors**: *Jiahua Rao, Shuangjia Zheng, Sijie Mai, Yuedong Yang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/544](https://doi.org/10.24963/ijcai.2022/544)

**Abstract**:

Illuminating the interconnections between drugs and genes is an important topic in drug development and precision medicine.  Currently, computational predictions of drug-gene interactions mainly focus on the binding interactions without considering other relation types like agonist, antagonist, etc. In addition,  existing methods either heavily rely on high-quality domain features or are intrinsically transductive, which limits the capacity of models to generalize to drugs/genes that lack external information or are unseen during the training process. To address these problems, we propose a novel Communicative Subgraph representation learning for Multi-relational Inductive drug-Gene interactions prediction (CoSMIG), where the predictions of drug-gene relations are made through subgraph patterns, and thus are naturally inductive for unseen drugs/genes without retraining or utilizing external domain features. Moreover, the model strengthened the relations on the drug-gene graph through a communicative message passing mechanism. To evaluate our method, we compiled two new benchmark datasets from DrugBank and DGIdb. The comprehensive experiments on the two datasets showed that our method outperformed state-of-the-art baselines in the transductive scenarios and achieved superior performance in the inductive ones. Further experimental analysis including LINCS experimental validation and literature verification also demonstrated the value of our model.

----

## [544] FOGS: First-Order Gradient Supervision with Learning-based Graph for Traffic Flow Forecasting

**Authors**: *Xuan Rao, Hao Wang, Liang Zhang, Jing Li, Shuo Shang, Peng Han*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/545](https://doi.org/10.24963/ijcai.2022/545)

**Abstract**:

Traffic flow forecasting plays a vital role in the transportation domain. 
Existing studies usually manually construct correlation graphs and design sophisticated models for learning spatial and temporal features to predict future traffic states.
However, manually constructed correlation graphs cannot accurately extract the complex patterns hidden in the traffic data. 
In addition, it is challenging for the prediction model to fit traffic data due to its irregularly-shaped distribution.
To solve the above-mentioned problems, in this paper, we propose a novel learning-based method to learn a spatial-temporal correlation graph, which could make good use of the traffic flow data. 
Moreover, we propose First-Order Gradient Supervision (FOGS), a novel method for traffic flow forecasting. 
FOGS utilizes first-order gradients, rather than specific flows, to train prediction model, which effectively avoids the problem of fitting irregularly-shaped distributions. Comprehensive numerical evaluations on four real-world datasets reveal that the proposed methods achieve state-of-the-art performance and significantly outperform the benchmarks.

----

## [545] Offline Vehicle Routing Problem with Online Bookings: A Novel Problem Formulation with Applications to Paratransit

**Authors**: *Amutheezan Sivagnanam, Salah Uddin Kadir, Ayan Mukhopadhyay, Philip Pugliese, Abhishek Dubey, Samitha Samaranayake, Aron Laszka*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/546](https://doi.org/10.24963/ijcai.2022/546)

**Abstract**:

Vehicle routing problems (VRPs) can be divided into two major categories: offline VRPs, which consider a given set of trip requests to be served, and online VRPs, which consider requests as they arrive in real-time. Based on discussions with public transit agencies, we identify a real-world problem that is not addressed by existing formulations: booking trips with flexible pickup windows (e.g., 3 hours) in advance (e.g., the day before) and confirming tight pickup windows (e.g., 30 minutes) at the time of booking. Such a service model is often required in paratransit service settings, where passengers typically book trips for the next day over the phone. To address this gap between offline and online problems, we introduce a novel formulation, the offline vehicle routing problem with online bookings. This problem is very challenging computationally since it faces the complexity of considering large sets of requests—similar to offline VRPs—but must abide by strict constraints on running time—similar to online VRPs. To solve this problem, we propose a novel computational approach, which combines an anytime algorithm with a learning-based policy for real-time decisions. Based on a paratransit dataset obtained from the public transit agency of Chattanooga, TN, we demonstrate that our novel formulation and computational approach lead to significantly better outcomes in this setting than existing algorithms.

----

## [546] Local Differential Privacy Meets Computational Social Choice - Resilience under Voter Deletion

**Authors**: *Liangde Tao, Lin Chen, Lei Xu, Weidong Shi*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/547](https://doi.org/10.24963/ijcai.2022/547)

**Abstract**:

The resilience of a voting system has been a central topic in computational social choice. Many voting rules, like plurality, are shown to be vulnerable as the attacker can target specific voters to manipulate the result. What if a local differential privacy (LDP) mechanism is adopted such that the true preference of a voter is never revealed in pre-election polls? In this case, the attacker can only infer stochastic information about a voter's true preference, and this may cause the manipulation of the electoral result significantly harder. The goal of this paper is to provide a quantitative study on the effect of adopting LDP mechanisms on a voting system. We introduce the metric PoLDP (power of LDP) that quantitatively measures the difference between the attacker's manipulation cost under LDP mechanisms and that without LDP mechanisms. The larger PoLDP is, the more robustness LDP mechanisms can add to a voting system. We give a full characterization of PoLDP for the voting system with plurality rule and provide general guidance towards the application of LDP mechanisms.

----

## [547] Private Stochastic Convex Optimization and Sparse Learning with Heavy-tailed Data Revisited

**Authors**: *Youming Tao, Yulian Wu, Xiuzhen Cheng, Di Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/548](https://doi.org/10.24963/ijcai.2022/548)

**Abstract**:

In this paper, we revisit the problem of Differentially Private Stochastic Convex Optimization (DP-SCO) with heavy-tailed data, where the gradient of the loss function has bounded moments. Instead of the case where the loss function is Lipschitz or each coordinate of the gradient has bounded second moment studied previously, we consider a relaxed scenario where  each coordinate of the gradient only has bounded (1+v)-th moment with some vâˆˆ(0, 1]. Firstly, we start from the one dimensional private mean estimation for heavy-tailed distributions. We propose a novel robust and private mean estimator which is optimal. Based on its idea, we then extend to the general d-dimensional space and study DP-SCO with general convex and strongly convex loss functions. We also provide  lower bounds for these two classes of loss under our setting and show that our upper bounds are optimal up to a factor of  O(Poly(d)). To address the high dimensionality issue, we also study DP-SCO with heavy-tailed gradient under some sparsity constraint (DP sparse learning). We propose a new method and show it is also optimal up to a factor of O(s*), where s* is the underlying sparsity of the constraint.

----

## [548] Exploring the Vulnerability of Deep Reinforcement Learning-based Emergency Control for Low Carbon Power Systems

**Authors**: *Xu Wan, Lanting Zeng, Mingyang Sun*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/549](https://doi.org/10.24963/ijcai.2022/549)

**Abstract**:

Decarbonization of global power systems significantly increases the operational uncertainty and modeling complexity that drive the necessity of widely exploiting cutting-edge Deep Reinforcement Learning (DRL) technologies to realize adaptive and real-time emergency control, which is the last resort for system stability and resiliency. The vulnerability of the DRL-based emergency control scheme may lead to severe real-world security issues if it can not be fully explored before implementing it practically. To this end, this is the first work that comprehensively investigates adversarial attacks and defense mechanisms for DRL-based power system emergency control. In particular, recovery-targeted (RT) adversarial attacks are designed for gradient-based approaches, aiming to dramatically degrade the effectiveness of the conducted emergency control actions to prevent the system from restoring to a stable state. Furthermore, the corresponding robust defense (RD) mechanisms are proposed to actively modify the observations based on the distances of sequential states. Experiments are conducted based on the standard IEEE reliability test system, and the results show that security risks indeed exist in the state-of-the-art DRL-based power system emergency control models. The effectiveness, stealthiness, instantaneity, and transferability of the proposed attacks and defense mechanisms are demonstrated with both white-box and black-box settings.

----

## [549] Heterogeneous Interactive Snapshot Network for Review-Enhanced Stock Profiling and Recommendation

**Authors**: *Heyuan Wang, Tengjiao Wang, Shun Li, Shijie Guan, Jiayi Zheng, Wei Chen*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/550](https://doi.org/10.24963/ijcai.2022/550)

**Abstract**:

Stock recommendation plays a critical role in modern quantitative trading. The large volumes of social media information such as investment reviews that delegate emotion-driven factors, together with price technical indicators formulate a “snapshot” of the evolving stock market profile. However, previous studies usually model the temporal trajectories of price and media modalities separately while losing their interrelated influences. Moreover, they mainly extract review semantics via sequential or attentive models, whereas the rich text associated knowledge is largely neglected. In this paper, we propose a novel heterogeneous interactive snapshot network for stock profiling and recommendation. We model investment reviews in each snapshot as a heterogeneous document graph, and develop a flexible hierarchical attentive propagation framework to capture fine-grained proximity features. Further, to learn stock embedding for ranking, we introduce a novel twins-GRU method, which tightly couples the media and price parallel sequences in a cross-interactive fashion to catch dynamic dependencies between successive snapshots. Our approach excels state-of-the-arts over 7.6% in terms of cumulative and risk-adjusted returns in trading simulations on both English and Chinese benchmarks.

----

## [550] Adaptive Long-Short Pattern Transformer for Stock Investment Selection

**Authors**: *Heyuan Wang, Tengjiao Wang, Shun Li, Jiayi Zheng, Shijie Guan, Wei Chen*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/551](https://doi.org/10.24963/ijcai.2022/551)

**Abstract**:

Stock investment selection is a hard issue in the Fintech field due to non-stationary dynamics and complex market interdependencies. Existing studies are mostly based on RNNs, which struggle to capture interactive information among fine granular volatility patterns. Besides, they either treat stocks as isolated, or presuppose a fixed graph structure heavily relying on prior domain knowledge. In this paper, we propose a novel Adaptive Long-Short Pattern Transformer (ALSP-TF) for stock ranking in terms of expected returns. Specifically, we overcome the limitations of canonical self-attention including context and position agnostic, with two additional capacities: (i) fine-grained pattern distiller to contextualize queries and keys based on localized feature scales, and (ii) time-adaptive modulator to let the dependency modeling among pattern pairs sensitive to different time intervals. Attention heads in stacked layers gradually harvest short- and long-term transition traits, spontaneously boosting the diversity of representations. Moreover, we devise a graph self-supervised regularization, which helps automatically assimilate the collective synergy of stocks and improve the generalization ability of overall model. Experiments on three exchange market datasets show ALSP-TFâ€™s superiority over state-of-the-art stock forecast methods.

----

## [551] Bridging the Gap between Reality and Ideality of Entity Matching: A Revisting and Benchmark Re-Constrcution

**Authors**: *Tianshu Wang, Hongyu Lin, Cheng Fu, Xianpei Han, Le Sun, Feiyu Xiong, Hui Chen, Minlong Lu, Xiuwen Zhu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/552](https://doi.org/10.24963/ijcai.2022/552)

**Abstract**:

Entity matching (EM) is the most critical step for entity resolution (ER). While current deep learning-based methods achieve very impressive performance on standard EM benchmarks, their real-world application performance is much frustrating. In this paper, we highlight that such the gap between reality and ideality stems from the unreasonable benchmark construction process, which is inconsistent with the nature of entity matching and therefore leads to biased evaluations of current EM approaches. To this end, we build a new EM corpus and re-construct EM benchmarks to challenge critical assumptions implicit in the previous benchmark construction process by step-wisely changing the restricted entities, balanced labels, and single-modal records in previous benchmarks into open entities, imbalanced labels, and multi-modal records in an open environment. Experimental results demonstrate that the assumptions made in the previous benchmark construction process are not coincidental with the open environment, which conceal the main challenges of the task and therefore significantly overestimate the current progress of entity matching. The constructed benchmarks and code are publicly released at https://github.com/tshu-w/ember.

----

## [552] Learnability of Competitive Threshold Models

**Authors**: *Yifan Wang, Guangmo Tong*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/553](https://doi.org/10.24963/ijcai.2022/553)

**Abstract**:

Modeling the spread of social contagions is central to various applications in social computing. In this paper, we study the learnability of the competitive threshold model from a theoretical perspective. We demonstrate how competitive threshold models can be seamlessly simulated by artificial neural networks with finite VC dimensions, which enables analytical sample complexity and generalization bounds. Based on the proposed hypothesis space, we design efficient algorithms under the empirical risk minimization scheme. The theoretical insights are finally translated into practical and explainable modeling methods, the effectiveness of which is verified through a sanity check over a few synthetic and real datasets. The experimental results promisingly show that our method enjoys a decent performance without using excessive data points, outperforming off-the-shelf methods.

----

## [553] Data-Efficient Backdoor Attacks

**Authors**: *Pengfei Xia, Ziqiang Li, Wei Zhang, Bin Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/554](https://doi.org/10.24963/ijcai.2022/554)

**Abstract**:

Recent studies have proven that deep neural networks are vulnerable to backdoor attacks. Specifically, by mixing a small number of poisoned samples into the training set, the behavior of the trained model can be maliciously controlled. Existing attack methods construct such adversaries by randomly selecting some clean data from the benign set and then embedding a trigger into them. However, this selection strategy ignores the fact that each poisoned sample contributes inequally to the backdoor injection, which reduces the efficiency of poisoning. In this paper, we formulate improving the poisoned data efficiency by the selection as an optimization problem and propose a Filtering-and-Updating Strategy (FUS) to solve it. The experimental results on CIFAR-10 and ImageNet-10 indicate that the proposed method is effective: the same attack success rate can be achieved with only 47% to 75% of the poisoned sample volume compared to the random selection strategy. More importantly, the adversaries selected according to one setting can generalize well to other settings, exhibiting strong transferability. The prototype code of our method is now available at https://github.com/xpf/Data-Efficient-Backdoor-Attacks.

----

## [554] TinyLight: Adaptive Traffic Signal Control on Devices with Extremely Limited Resources

**Authors**: *Dong Xing, Qian Zheng, Qianhui Liu, Gang Pan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/555](https://doi.org/10.24963/ijcai.2022/555)

**Abstract**:

Recent advances in deep reinforcement learning (DRL) have largely promoted the performance of adaptive traffic signal control (ATSC). Nevertheless, regarding the implementation, most works are cumbersome in terms of storage and computation. This hinders their deployment on scenarios where resources are limited. In this work, we propose TinyLight, the first DRL-based ATSC model that is designed for devices with extremely limited resources. TinyLight first constructs a super-graph to associate a rich set of candidate features with a group of light-weighted network blocks. Then,  to diminish the model's resource consumption, we ablate edges in the super-graph automatically with a novel entropy-minimized objective function. This enables TinyLight to work on a standalone microcontroller with merely 2KB RAM and 32KB ROM. We evaluate TinyLight on multiple road networks with real-world traffic demands. Experiments show that even with extremely limited resources, TinyLight still achieves competitive performance. The source code and appendix of this work can be found at https://bit.ly/38hH8t8.

----

## [555] ARCANE: An Efficient Architecture for Exact Machine Unlearning

**Authors**: *Haonan Yan, Xiaoguang Li, Ziyao Guo, Hui Li, Fenghua Li, Xiaodong Lin*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/556](https://doi.org/10.24963/ijcai.2022/556)

**Abstract**:

Recently users’ right-to-be-forgotten is stipulated by many laws and regulations. However, only removing the data from the dataset is not enough, as machine learning models would memorize the training data once the data is involved in model training, increasing the risk of exposing users’ privacy. To solve this problem, currently, the straightforward method, naive retraining, is to discard these data and retrain the model from scratch, which is reliable but brings much computational and time overhead. In this paper, we propose an exact unlearning architecture called ARCANE. Based on ensemble learning, we transform the naive retraining into multiple one-class classiﬁcation tasks to reduce retraining cost while ensuring model performance, especially in the case of a large number of unlearning requests not considered by previous works. Then we further introduce data preprocessing methods to reduce the retraining overhead and speed up the unlearning, which includes representative data selection for redundancy removal, training state saving to reuse previous calculation results, and sorting to cope with unlearning requests of different distributions. We extensively evaluate ARCANE on three typical datasets with three common model architectures. Experiment results show the effectiveness and superiority of ARCANE over both the naive retraining and the state-of-the-art method in terms of model performance and unlearning speed.

----

## [556] A Smart Trader for Portfolio Management based on Normalizing Flows

**Authors**: *Mengyuan Yang, Xiaolin Zheng, Qianqiao Liang, Bing Han, Mengying Zhu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/557](https://doi.org/10.24963/ijcai.2022/557)

**Abstract**:

In this paper, we study a new kind of portfolio problem, named trading point aware portfolio optimization (TPPO), which aims to obtain excess intraday profit by deciding the portfolio weights and their trading points simultaneously based on microscopic information. However, a strategy for the TPPO problem faces two challenging problems, i.e., modeling the ever-changing and irregular microscopic stock price time series and deciding the scattering candidate trading points. To address these problems, we propose a novel TPPO strategy named STrader based on normalizing flows. STrader is not only promising in reversibly transforming the geometric Brownian motion process to the unobservable and complicated stochastic process of the microscopic stock price time series for modeling such series, but also has the ability to earn excess intraday profit by capturing the appropriate trading points of the portfolio. Extensive experiments conducted on three public datasets demonstrate STrader's superiority over the state-of-the-art portfolio strategies.

----

## [557] Temporality Spatialization: A Scalable and Faithful Time-Travelling Visualization for Deep Classifier Training

**Authors**: *Xianglin Yang, Yun Lin, Ruofan Liu, Jin Song Dong*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/558](https://doi.org/10.24963/ijcai.2022/558)

**Abstract**:

Time-travelling visualization answers how the predictions of a deep classifier are formed during the training. It visualizes in two or three dimensional space how the classification boundaries and sample embeddings are evolved during training. 

In this work, we propose TimeVis, a novel time-travelling visualization solution for deep classifiers. Comparing to the state-of-the-art solution DeepVisualInsight (DVI), TimeVis can significantly (1) reduce visualization errors for rendering samplesâ€™ travel across different training epochs, and (2) improve the visualization efficiency. To this end, we design a technique called temporality spatialization, which unifies the spatial relation (e.g., neighbouring samples in single epoch) and temporal relation (e.g., one identical sample in neighbouring training epochs) into one high-dimensional topological complex. Such spatio-temporal complex can be used to efficiently train one visualization model to accurately project and inverse-project any high and low dimensional data across epochs. Our extensive experiment shows that, in comparison to DVI, TimeVis not only is more accurate to preserve the visualized time-travelling semantics, but 15X faster in visualization efficiency, achieving a new state-of-the-art in time-travelling visualization.

----

## [558] Post-processing of Differentially Private Data: A Fairness Perspective

**Authors**: *Keyu Zhu, Ferdinando Fioretto, Pascal Van Hentenryck*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/559](https://doi.org/10.24963/ijcai.2022/559)

**Abstract**:

Post-processing immunity is a fundamental property of differential privacy: it enables arbitrary data-independent transformations to differentially private outputs without affecting their privacy guarantees. Post-processing is routinely applied in data-release applications, including census data, which are then used to make allocations with substantial societal impacts. This paper shows that post-processing causes disparate impacts on individuals or groups and analyzes two critical settings: the release of differentially private datasets and the use of such private datasets for downstream decisions, such as the allocation of funds  informed by US Census data. In the first setting, the paper proposes tight bounds on the unfairness for traditional post-processing mechanisms, giving a unique tool to decision makers to quantify the disparate impacts introduced by their release. In the second setting, this paper proposes a novel post-processing mechanism that is (approximately) optimal under different fairness metrics, either reducing fairness issues substantially or reducing the cost of privacy. The theoretical analysis is complemented with numerical simulations on Census data.

----

## [559] Enhancing Entity Representations with Prompt Learning for Biomedical Entity Linking

**Authors**: *Tiantian Zhu, Yang Qin, Qingcai Chen, Baotian Hu, Yang Xiang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/560](https://doi.org/10.24963/ijcai.2022/560)

**Abstract**:

Biomedical entity linking aims to map mentions in biomedical text to standardized concepts or entities in a curated knowledge base (KB) such as Unified Medical Language System (UMLS). The latest research tends to solve this problem in a unified framework solely based on surface form matching between mentions and entities. Specifically, these methods focus on addressing the variety challenge of the heterogeneous naming of biomedical concepts. Yet, the ambiguity challenge that the same word under different contexts may refer to distinct entities is usually ignored. To address this challenge, we propose a two-stage linking algorithm to enhance the entity representations based on prompt learning. The first stage includes a coarser-grained retrieval from a representation space defined by a bi-encoder that independently embeds the mention and entityâ€™s surface forms. Unlike previous one-model-fits-all systems, each candidate is then re-ranked with a finer-grained encoder based on prompt-tuning that utilizes the contextual information. Extensive experiments show that our model achieves promising performance improvements compared with several state-of-the-art techniques on the largest biomedical public dataset MedMentions and the NCBI disease corpus. We also observe by cases that the proposed prompt-tuning strategy is effective in solving both the variety and ambiguity challenges in the linking task.

----

## [560] Aspect-based Sentiment Analysis with Opinion Tree Generation

**Authors**: *Xiaoyi Bao, Zhongqing Wang, Xiaotong Jiang, Rong Xiao, Shoushan Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/561](https://doi.org/10.24963/ijcai.2022/561)

**Abstract**:

Existing studies usually extract these sentiment elements by decomposing the complex structure prediction task into multiple subtasks. Despite their effectiveness, these methods ignore the semantic structure in ABSA problems and require extensive task-specific designs. In this study, we introduce an Opinion Tree Generation task, which aims to jointly detect all sentiment elements in a tree. We believe that the opinion tree can reveal a more comprehensive and complete aspect-level sentiment structure. Furthermore, we propose a pre-trained model to integrate both syntax and semantic features for opinion tree generation. On one hand, a pre-trained model with large-scale unlabeled data is important for the tree generation model. On the other hand, the syntax and semantic features are very effective for forming the opinion tree structure.  Extensive experiments show the superiority of our proposed method. The results also validate the tree structure is effective to generate sentimental elements.

----

## [561] Speaker-Guided Encoder-Decoder Framework for Emotion Recognition in Conversation

**Authors**: *Yinan Bao, Qianwen Ma, Lingwei Wei, Wei Zhou, Songlin Hu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/562](https://doi.org/10.24963/ijcai.2022/562)

**Abstract**:

The emotion recognition in conversation (ERC) task aims to predict the emotion label of an utterance in a conversation. Since the dependencies between speakers are complex and dynamic, which consist of intra- and inter-speaker dependencies, the modeling of speaker-specific information is a vital role in ERC. Although existing researchers have proposed various methods of speaker interaction modeling, they cannot explore dynamic intra- and inter-speaker dependencies jointly, leading to the insufficient comprehension of context and further hindering emotion prediction. To this end, we design a novel speaker modeling scheme that explores intra- and inter-speaker dependencies jointly in a dynamic manner. Besides, we propose a Speaker-Guided Encoder-Decoder (SGED) framework for ERC, which fully exploits speaker information for the decoding of emotion. We use different existing methods as the conversational context encoder of our framework, showing the high scalability and flexibility of the proposed framework. Experimental results demonstrate the superiority and effectiveness of SGED.

----

## [562] Learning Meta Word Embeddings by Unsupervised Weighted Concatenation of Source Embeddings

**Authors**: *Danushka Bollegala*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/563](https://doi.org/10.24963/ijcai.2022/563)

**Abstract**:

Given multiple source word embeddings learnt using diverse algorithms and lexical resources, meta word embedding learning methods attempt to learn more accurate and wide-coverage word embeddings. 
	Prior work on meta-embedding has repeatedly discovered that simple vector concatenation of the source embeddings to be a competitive baseline. 
	However, it remains unclear as to why and when simple vector concatenation can produce accurate meta-embeddings. 
	We show that weighted concatenation can be seen as a spectrum matching operation between each source embedding and the meta-embedding, minimising the pairwise inner-product loss.
	Following this theoretical analysis, we propose two \emph{unsupervised} methods to learn the optimal concatenation weights for creating meta-embeddings from a given set of source embeddings.
	Experimental results on multiple benchmark datasets show that the proposed weighted concatenated meta-embedding methods outperform previously proposed meta-embedding learning methods.

----

## [563] PCVAE: Generating Prior Context for Dialogue Response Generation

**Authors**: *Zefeng Cai, Zerui Cai*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/564](https://doi.org/10.24963/ijcai.2022/564)

**Abstract**:

Conditional Variational AutoEncoder (CVAE) is promising for modeling one-to-many relationships in dialogue generation, as it can naturally generate many responses from a given context. However, the conventional used continual latent variables in CVAE are more likely to generate generic rather than distinct and specific responses. To resolve this problem, we introduce a novel discrete variable called prior context which enables the generation of favorable responses. Specifically, we present Prior Context VAE (PCVAE), a hierarchical VAE that learns prior context from data automatically for dialogue generation. Meanwhile, we design Active Codeword Transport (ACT) to help the model actively discover potential prior context. Moreover, we propose Autoregressive Compatible Arrangement (ACA) that enables modeling prior context in autoregressive style, which is crucial for selecting appropriate prior context according to a given context. Extensive experiments demonstrate that PCVAE can generate distinct responses and significantly outperforms strong baselines.

----

## [564] Towards Joint Intent Detection and Slot Filling via Higher-order Attention

**Authors**: *Dongsheng Chen, Zhiqi Huang, Xian Wu, Shen Ge, Yuexian Zou*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/565](https://doi.org/10.24963/ijcai.2022/565)

**Abstract**:

Recently, attention-based models for joint intent detection and slot filling have achieved state-of-the-art performance. However, we think the conventional attention can only capture the first-order feature interaction between two tasks and is insufficient. To address this issue, we propose a unified BiLinear attention block, which leverages bilinear pooling to synchronously explore both the contextual and channel-wise bilinear attention distributions to capture the second-order interactions between the input intent and slot features. Higher-order interactions are constructed by combining many such blocks and exploiting Exponential Linear activations. Furthermore, we present a Higher-order Attention Network (HAN) to jointly model them. The experimental results show that our approach outperforms the state-of-the-art results. We also conduct experiments on the new SLURP dataset, and give a discussion on HANâ€™s properties, i.e., robustness and generalization.

----

## [565] Effective Graph Context Representation for Document-level Machine Translation

**Authors**: *Kehai Chen, Muyun Yang, Masao Utiyama, Eiichiro Sumita, Rui Wang, Min Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/566](https://doi.org/10.24963/ijcai.2022/566)

**Abstract**:

Document-level neural machine translation (DocNMT) universally encodes several local sentences or the entire document. Thus, DocNMT does not consider the relevance of document-level contextual information, for example, some context (i.e., content words, logical order, and co-occurrence relation) is more effective than another auxiliary context  (i.e., functional and auxiliary words). To address this issue, we first utilize the word frequency information to recognize content words in the input document, and then use heuristical relations to summarize content words and sentences as a graph structure without relying on external syntactic knowledge. Furthermore, we apply graph attention networks to this graph structure to learn its feature representation, which allows DocNMT to more effectively capture the document-level context. Experimental results on several widely-used document-level benchmarks demonstrated the effectiveness of the proposed approach.

----

## [566] DictBERT: Dictionary Description Knowledge Enhanced Language Model Pre-training via Contrastive Learning

**Authors**: *Qianglong Chen, Feng-Lin Li, Guohai Xu, Ming Yan, Ji Zhang, Yin Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/567](https://doi.org/10.24963/ijcai.2022/567)

**Abstract**:

Although pre-trained language models (PLMs) have achieved state-of-the-art performance on various natural language processing (NLP) tasks, they are shown to be lacking in knowledge when dealing with knowledge driven tasks. Despite the many efforts made for injecting knowledge into PLMs, this problem remains open. To address the challenge, we propose DictBERT, a novel approach that enhances PLMs with dictionary knowledge which is easier to acquire than knowledge graph (KG). During pre-training, we present two novel pre-training tasks to inject dictionary knowledge into PLMs via contrastive learning: dictionary entry prediction and entry description discrimination. In fine-tuning, we use the pre-trained DictBERT as a plugin knowledge base (KB) to retrieve implicit knowledge for identified entries in an input sequence, and infuse the retrieved knowledge into the input to enhance its representation via a novel extra-hop attention mechanism. We evaluate our approach on a variety of knowledge driven and language understanding tasks, including NER, relation extraction, CommonsenseQA, OpenBookQA and GLUE. Experimental results demonstrate that our model can significantly improve typical PLMs: it gains a substantial improvement of 0.5%, 2.9%, 9.0%, 7.1% and 3.3% on BERT-large respectively, and is also effective on RoBERTa-large.

----

## [567] Interpretable AMR-Based Question Decomposition for Multi-hop Question Answering

**Authors**: *Zhenyun Deng, Yonghua Zhu, Yang Chen, Michael Witbrock, Patricia Riddle*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/568](https://doi.org/10.24963/ijcai.2022/568)

**Abstract**:

Effective multi-hop question answering (QA) requires reasoning over multiple scattered paragraphs and providing explanations for answers. Most existing approaches cannot provide an interpretable reasoning process to illustrate how these models arrive at an answer. In this paper, we propose a Question Decomposition method based on Abstract Meaning Representation (QDAMR) for multi-hop QA, which achieves interpretable reasoning by decomposing a multi-hop question into simpler subquestions and answering them in order. Since annotating the decomposition is expensive, we first delegate the complexity of understanding the multi-hop question to an AMR parser. We then achieve decomposition of a multi-hop question via segmentation of the corresponding AMR graph based on the required reasoning type. Finally, we generate sub-questions using an AMR-to-Text generation model and answer them with an off-the-shelf QA model. Experimental results on HotpotQA demonstrate that our approach is competitive for interpretable reasoning and that the sub-questions generated by QDAMR are well-formed, outperforming existing question-decomposition-based multihop QA approaches.

----

## [568] Interactive Information Extraction by Semantic Information Graph

**Authors**: *Siqi Fan, Yequan Wang, Jing Li, Zheng Zhang, Shuo Shang, Peng Han*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/569](https://doi.org/10.24963/ijcai.2022/569)

**Abstract**:

Information extraction (IE) mainly focuses on three highly correlated subtasks, i.e., entity extraction, relation extraction and event extraction. Recently, there are studies using Abstract Meaning Representation (AMR) to utilize the intrinsic correlations among these three subtasks. AMR based models are capable of building the relationship of arguments. However, they are hard to deal with relations. In addition, the noises of AMR (i.e., tags unrelated to IE tasks, nodes with unconcerned conception, and edge types with complicated hierarchical structures) disturb the decoding processing of IE. As a result, the decoding processing limited by the AMR cannot be worked effectively. To overcome the shortages, we propose an Interactive Information Extraction (InterIE) model based on a novel Semantic Information Graph (SIG). SIG can guide our InterIE model to tackle the three subtasks jointly. Furthermore, the well-designed SIG without noise is capable of enriching entity and event trigger representation, and capturing the edge connection between the information types. Experimental results show that our InterIE achieves state-of-the-art performance on all IE subtasks on the benchmark dataset (i.e., ACE05-E+ and ACE05-E). More importantly, the proposed model is not sensitive to the decoding order, which goes beyond the limitations of AMR based methods.

----

## [569] Global Inference with Explicit Syntactic and Discourse Structures for Dialogue-Level Relation Extraction

**Authors**: *Hao Fei, Jingye Li, Shengqiong Wu, Chenliang Li, Donghong Ji, Fei Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/570](https://doi.org/10.24963/ijcai.2022/570)

**Abstract**:

Recent research attention for relation extraction has been paid to the dialogue scenario, i.e., dialogue-level relation extraction (DiaRE). Existing DiaRE methods either simply concatenate the utterances in a dialogue into a long piece of text, or employ naive words, sentences or entities to build dialogue graphs, while the structural characteristics in dialogues have not been fully utilized. In this work, we investigate a novel dialogue-level mixed dependency graph (D2G) and an argument reasoning graph (ARG) for DiaRE with a global relation reasoning mechanism. First, we model the entire dialogue into a unified and coherent D2G by explicitly integrating both syntactic and discourse structures, which enables richer semantic and feature learning for relation extraction. Second, we stack an ARG graph on top of D2G to further focus on argument inter-dependency learning and argument representation refinement, for sufficient argument relation inference. In our global reasoning framework, D2G and ARG work collaboratively, iteratively performing lexical, syntactic and semantic information exchange and representation learning over the entire dialogue context. On two DiaRE benchmarks, our framework shows considerable improvements over the current state-of-the-art baselines. Further analyses show that the model effectively solves the long-range dependence issue, and meanwhile gives explainable predictions.

----

## [570] Conversational Semantic Role Labeling with Predicate-Oriented Latent Graph

**Authors**: *Hao Fei, Shengqiong Wu, Meishan Zhang, Yafeng Ren, Donghong Ji*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/571](https://doi.org/10.24963/ijcai.2022/571)

**Abstract**:

Conversational semantic role labeling (CSRL) is a newly proposed task that uncovers the shallow semantic structures in a dialogue text. Unfortunately several important characteristics of the CSRL task have been overlooked by the existing works, such as the structural information integration, near-neighbor influence. In this work, we investigate the integration of a latent graph for CSRL. We propose to automatically induce a predicate-oriented latent graph (POLar) with a predicate-centered gaussian mechanism, by which the nearer and informative words to the predicate will be allocated with more attention. The POLar structure is then dynamically pruned and refined so as to best fit the task need. We additionally introduce an effective dialogue-level pre-trained language model, CoDiaBERT, for better supporting multiple utterance sentences and handling the speaker coreference issue in CSRL. Our system outperforms best-performing baselines on three benchmark CSRL datasets with big margins, especially achieving over 4% F1 score improvements on the cross-utterance argument detection. Further analyses are presented to better understand the effectiveness of our proposed methods.

----

## [571] Inheriting the Wisdom of Predecessors: A Multiplex Cascade Framework for Unified Aspect-based Sentiment Analysis

**Authors**: *Hao Fei, Fei Li, Chenliang Li, Shengqiong Wu, Jingye Li, Donghong Ji*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/572](https://doi.org/10.24963/ijcai.2022/572)

**Abstract**:

So far, aspect-based sentiment analysis (ABSA) has involved with total seven subtasks, in which, however the interactions among them have been left unexplored sufficiently. This work presents a novel multiplex cascade framework for unified ABSA and maintaining such interactions. First, we model total seven subtasks as a hierarchical dependency in the easy-to-hard order, based on which we then propose a multiplex decoding mechanism, transferring the sentiment layouts and clues in lower tasks to upper ones. The multiplex strategy enables highly-efficient subtask interflows and avoids repetitive training; meanwhile it sufficiently utilizes the existing data without requiring any further annotation. Further, based on the characteristics of aspect-opinion term extraction and pairing, we enhance our multiplex framework by integrating POS tag and syntactic dependency information for term boundary and pairing identification. The proposed Syntax-aware Multiplex (SyMux) framework enhances the ABSA performances on 28 subtasks (7Ã—4 datasets) with big margins.

----

## [572] Logically Consistent Adversarial Attacks for Soft Theorem Provers

**Authors**: *Alexander Gaskell, Yishu Miao, Francesca Toni, Lucia Specia*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/573](https://doi.org/10.24963/ijcai.2022/573)

**Abstract**:

Recent efforts within the AI community have yielded impressive results towards “soft theorem proving” over natural language sentences using language models. We propose a novel, generative adversarial framework for probing and improving these models’ reasoning capabilities. Adversarial attacks in this domain suffer from the logical inconsistency problem, whereby perturbations to the input may alter the label. Our Logically consistent AdVersarial Attacker, LAVA, addresses this by combining a structured generative process with a symbolic solver, guaranteeing logical consistency. Our framework successfully generates adversarial attacks and identifies global weaknesses common across multiple target models. Our analyses reveal naive heuristics and vulnerabilities in these models’ reasoning capabilities, exposing an incomplete grasp of logical deduction under logic programs. Finally, in addition to effective probing of these models, we show that training on the generated samples improves the target model’s performance.

----

## [573] Leveraging the Wikipedia Graph for Evaluating Word Embeddings

**Authors**: *Joachim Giesen, Paul Kahlmeyer, Frank Nussbaum, Sina Zarrieß*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/574](https://doi.org/10.24963/ijcai.2022/574)

**Abstract**:

Deep learning models for different NLP tasks often rely on pre-trained word embeddings, that is, vector representations of words. Therefore, it is crucial to evaluate pre-trained word embeddings independently of downstream tasks. Such evaluations try to assess whether the geometry induced by a word embedding captures connections made in natural language, such as, analogies, clustering of words, or word similarities. Here, traditionally, similarity is measured by comparison to human judgment. However, explicitly annotating word pairs with similarity scores by surveying humans is expensive. We tackle this problem by formulating a similarity measure that is based on an agent for routing the Wikipedia hyperlink graph. In this graph, word similarities are implicitly encoded by edges between articles. We show on the English Wikipedia that our measure correlates well with a large group of traditional similarity measures, while covering a much larger proportion of words and avoiding explicit human labeling. Moreover, since Wikipedia is available in more than 300 languages, our measure can easily be adapted to other languages, in contrast to traditional similarity measures.

----

## [574] Fallacious Argument Classification in Political Debates

**Authors**: *Pierpaolo Goffredo, Shohreh Haddadan, Vorakit Vorakitphan, Elena Cabrio, Serena Villata*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/575](https://doi.org/10.24963/ijcai.2022/575)

**Abstract**:

Fallacies play a prominent role in argumentation since antiquity due to their contribution to argumentation in critical thinking education. Their role is even more crucial nowadays as contemporary argumentation technologies face challenging tasks as misleading and manipulative information detection in news articles and political discourse, and counter-narrative generation.  Despite some work in this direction, the issue of classifying arguments as being fallacious largely remains a challenging and an unsolved task. Our contribution is twofold: first, we present a novel annotated resource of 31 political debates from the U.S. Presidential Campaigns, where we annotated six main categories of fallacious arguments (i.e., ad hominem, appeal to authority, appeal to emotion, false cause, slogan, slippery slope) leading to 1628 annotated fallacious arguments; second, we tackle this novel task of fallacious argument classification and we define a neural architecture based on transformers outperforming state-of-the-art results and standard baselines. Our results show the important role played by argument components and relations in this task.

----

## [575] Improving Few-Shot Text-to-SQL with Meta Self-Training via Column Specificity

**Authors**: *Xinnan Guo, Yongrui Chen, Guilin Qi, Tianxing Wu, Hao Xu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/576](https://doi.org/10.24963/ijcai.2022/576)

**Abstract**:

The few-shot problem is an urgent challenge for single-table text-to-SQL. Existing methods ignore the potential value of unlabeled data, and merely rely on a coarse-grained Meta-Learning (ML) algorithm that neglects the differences of column contributions to the optimization object. This paper proposes a Meta Self-Training text-to-SQL (MST-SQL) method to solve the problem. Specifically, MST-SQL is based on column-wise HydraNet and adopts self-training as an effective mechanism to learn from readily available unlabeled samples. During each epoch of training, it first predicts pseudo-labels for unlabeled samples and then leverages them to update the parameters. A fine-grained ML algorithm is used in updating, which weighs the contribution of columns by their specificity, in order to further improve the generalizability. Extensive experimental results on both open-domain and domain-specific benchmarks reveal that our MST-SQL has significant advantages in few-shot scenarios,  and is also competitive in standard supervised settings.

----

## [576] FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis

**Authors**: *Rongjie Huang, Max W. Y. Lam, Jun Wang, Dan Su, Dong Yu, Yi Ren, Zhou Zhao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/577](https://doi.org/10.24963/ijcai.2022/577)

**Abstract**:

Denoising diffusion probabilistic models (DDPMs) have recently achieved leading performances in many generative tasks. However, the inherited iterative sampling process costs hindered their applications to speech synthesis. This paper proposes FastDiff, a fast conditional diffusion model for high-quality speech synthesis. FastDiff employs a stack of time-aware location-variable convolutions of diverse receptive field patterns to efficiently model long-term time dependencies with adaptive conditions. A noise schedule predictor is also adopted to reduce the sampling steps without sacrificing the generation quality. Based on FastDiff, we design an end-to-end text-to-speech synthesizer, FastDiff-TTS, which generates high-fidelity speech waveforms without any intermediate feature (e.g., Mel-spectrogram). Our evaluation of FastDiff demonstrates the state-of-the-art results with higher-quality (MOS 4.28) speech samples. Also, FastDiff enables a sampling speed of 58x faster than real-time on a V100 GPU, making diffusion models practically applicable to speech synthesis deployment for the first time. We further show that FastDiff generalized well to the mel-spectrogram inversion of unseen speakers, and FastDiff-TTS outperformed other competing methods in end-to-end text-to-speech synthesis. Audio samples are available at https://FastDiff.github.io/.

----

## [577] MuiDial: Improving Dialogue Disentanglement with Intent-Based Mutual Learning

**Authors**: *Ziyou Jiang, Lin Shi, Celia Chen, Fangwen Mu, Yumin Zhang, Qing Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/578](https://doi.org/10.24963/ijcai.2022/578)

**Abstract**:

The main goal of dialogue disentanglement is to separate the mixed utterances from a chat slice into independent dialogues. Existing models often utilize either an utterance-to-utterance (U2U) prediction to determine whether two utterances that have the “reply-to” relationship belong to one dialogue, or an utterance-to-thread (U2T) prediction to determine which dialogue-thread a given utterance should belong to. Inspired by mutual leaning, we propose MuiDial, a novel dialogue disentanglement model, to exploit the intent of each utterance and feed the intent to a mutual learning U2U-U2T disentanglement model. Experimental results and in-depth analysis on several benchmark datasets demonstrate the effectiveness and generalizability of our approach.

----

## [578] AdMix: A Mixed Sample Data Augmentation Method for Neural Machine Translation

**Authors**: *Chang Jin, Shigui Qiu, Nini Xiao, Hao Jia*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/579](https://doi.org/10.24963/ijcai.2022/579)

**Abstract**:

In Neural Machine Translation (NMT), data augmentation methods such as back-translation have proven their effectiveness in improving translation performance. In this paper, we propose a novel data augmentation approach for NMT, which is independent of any additional training data. Our approach, AdMix, consists of two parts: 1) introduce faint discrete noise (word replacement, word dropping, word swapping) into the original sentence pairs to form augmented samples; 2) generate new synthetic training data by softly mixing the augmented samples with their original samples in training corpus. Experiments on three translation datasets of different scales show that AdMix achieves significant improvements (1.0 to 2.7 BLEU points) over strong Transformer baseline. When combined with other data augmentation techniques (e.g., back-translation), our approach can obtain further improvements.

----

## [579] Curriculum-Based Self-Training Makes Better Few-Shot Learners for Data-to-Text Generation

**Authors**: *Pei Ke, Haozhe Ji, Zhenyu Yang, Yi Huang, Junlan Feng, Xiaoyan Zhu, Minlie Huang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/580](https://doi.org/10.24963/ijcai.2022/580)

**Abstract**:

Despite the success of text-to-text pre-trained models in various natural language generation (NLG) tasks, the generation performance is largely restricted by the number of labeled data in downstream tasks, particularly in data-to-text generation tasks. Existing works mostly utilize abundant unlabeled structured data to conduct unsupervised pre-training for task adaption, which fail to model the complex relationship between source structured data and target texts. Thus, we introduce self-training as a better few-shot learner than task-adaptive pre-training, which explicitly captures this relationship via pseudo-labeled data generated by the pre-trained model. To alleviate the side-effect of low-quality pseudo-labeled data during self-training, we propose a novel method called Curriculum-Based Self-Training (CBST) to effectively leverage unlabeled data in a rearranged order determined by the difficulty of text generation. Experimental results show that our method can outperform fine-tuning and task-adaptive pre-training methods, and achieve state-of-the-art performance in the few-shot setting of data-to-text generation.

----

## [580] Deexaggeration

**Authors**: *Li Kong, Chuanyi Li, Vincent Ng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/581](https://doi.org/10.24963/ijcai.2022/581)

**Abstract**:

We introduce a new task in hyperbole processing, deexaggeration, which concerns the recovery of the meaning of what is being exaggerated in a hyperbolic sentence in the form of a structured representation. In this paper, we lay the groundwork for the computational study of understanding hyperbole by (1) defining a structured representation to encode what is being exaggerated in a hyperbole in a non-hyperbolic manner, (2) annotating the hyperbolic sentences in two existing datasets, HYPO and HYPO-cn, using this structured representation, (3) conducting an empirical analysis of our annotated corpora, and (4) presenting preliminary results on the deexaggeration task.

----

## [581] Taylor, Can You Hear Me Now? A Taylor-Unfolding Framework for Monaural Speech Enhancement

**Authors**: *Andong Li, Shan You, Guochen Yu, Chengshi Zheng, Xiaodong Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/582](https://doi.org/10.24963/ijcai.2022/582)

**Abstract**:

While the deep learning techniques promote the rapid development of the speech enhancement (SE) community, most schemes only pursue the performance in a black-box manner and lack adequate model interpretability. Inspired by Taylor's approximation theory, we propose an interpretable decoupling-style SE framework, which disentangles the complex spectrum recovery into two separate optimization problems i.e., magnitude and complex residual estimation. Specifically, serving as the 0th-order term in Taylor's series, a filter network is delicately devised to suppress the noise component only in the magnitude domain and obtain a coarse spectrum. To refine the phase distribution, we estimate the sparse complex residual, which is defined as the difference between target and coarse spectra, and measures the phase gap. In this study, we formulate the residual component as the combination of various high-order Taylor terms and propose a lightweight trainable module to replace the complicated derivative operator between adjacent terms. Finally, following Taylor's formula, we can reconstruct the target spectrum by the superimposition between 0th-order and high-order terms. Experimental results on two benchmark datasets show that our framework achieves state-of-the-art performance over previous competing baselines in various evaluation metrics. The source code is available at https://github.com/Andong-Li-speech/TaylorSENet.

----

## [582] FastRE: Towards Fast Relation Extraction with Convolutional Encoder and Improved Cascade Binary Tagging Framework

**Authors**: *Guozheng Li, Xu Chen, Peng Wang, Jiafeng Xie, Qiqing Luo*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/583](https://doi.org/10.24963/ijcai.2022/583)

**Abstract**:

Recent work for extracting relations from texts has achieved excellent performance. However, most existing methods pay less attention to the efficiency, making it still challenging to quickly extract relations from massive or streaming text data in realistic scenarios. The main efficiency bottleneck is that these methods use a Transformer-based pre-trained language model for encoding, which heavily affects the training speed and inference speed. To address this issue, we propose a fast relation extraction model (FastRE) based on convolutional encoder and improved cascade binary tagging framework. Compared to previous work, FastRE employs several innovations to improve efficiency while also keeping promising performance. Concretely, FastRE adopts a novel convolutional encoder architecture combined with dilated convolution, gated unit and residual connection, which significantly reduces the computation cost of training and inference, while maintaining the satisfactory performance. Moreover, to improve the cascade binary tagging framework, FastRE first introduces a type-relation mapping mechanism to accelerate tagging efficiency and alleviate relation redundancy, and then utilizes a position-dependent adaptive thresholding strategy to obtain higher tagging accuracy and better model generalization. Experimental results demonstrate that FastRE is well balanced between efficiency and performance, and achieves 3-10$\times$ training speed, 7-15$\times$ inference speed faster, and 1/100 parameters compared to the state-of-the-art models, while the performance is still competitive. Our code is available at \url{https://github.com/seukgcode/FastRE}.

----

## [583] Neutral Utterances are Also Causes: Enhancing Conversational Causal Emotion Entailment with Social Commonsense Knowledge

**Authors**: *Jiangnan Li, Fandong Meng, Zheng Lin, Rui Liu, Peng Fu, Yanan Cao, Weiping Wang, Jie Zhou*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/584](https://doi.org/10.24963/ijcai.2022/584)

**Abstract**:

Conversational Causal Emotion Entailment aims to detect causal utterances for a non-neutral targeted utterance from a conversation. In this work, we build conversations as graphs to overcome implicit contextual modelling of the original entailment style. Following the previous work, we further introduce the emotion information into graphs. Emotion information can markedly promote the detection of causal utterances whose emotion is the same as the targeted utterance. However, it is still hard to detect causal utterances with different emotions, especially neutral ones. The reason is that models are limited in reasoning causal clues and passing them between utterances. To alleviate this problem, we introduce social commonsense knowledge (CSK) and propose a Knowledge Enhanced Conversation graph (KEC). KEC propagates the CSK between two utterances. As not all CSK is emotionally suitable for utterances, we therefore propose a sentiment-realized knowledge selecting strategy to filter CSK. To process KEC, we further construct the Knowledge Enhanced Directed Acyclic Graph networks. Experimental results show that our method outperforms baselines and infers more causes with different emotions from the targeted utterance.

----

## [584] Domain-Adaptive Text Classification with Structured Knowledge from Unlabeled Data

**Authors**: *Tian Li, Xiang Chen, Zhen Dong, Kurt Keutzer, Shanghang Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/585](https://doi.org/10.24963/ijcai.2022/585)

**Abstract**:

Domain adaptive text classification is a challenging problem for the large-scale pretrained language models because they often require expensive additional labeled data to adapt to new domains. Existing works usually fails to leverage the implicit relationships among words across domains. In this paper, we propose a novel method, called Domain Adaptation with Structured Knowledge (DASK), to enhance domain adaptation by exploiting word-level semantic relationships. DASK first builds a knowledge graph to capture the relationship between pivot terms (domain-independent words) and non-pivot terms in the target domain. Then during training, DASK injects pivot-related knowledge graph information into source domain texts. For the downstream task, these knowledge-injected texts are fed into a BERT variant capable of processing knowledge-injected textual data. Thanks to the knowledge injection, our model learns domain-invariant features for non-pivots according to their relationships with pivots. DASK ensures the pivots to have domain-invariant behaviors by dynamically inferring via the polarity scores of candidate pivots during training with pseudo-labels. We validate DASK on a wide range of cross-domain sentiment classification tasks and observe up to 2.9% absolute performance improvement over baselines for 20 different domain pairs. Code is available at https://github.com/hikaru-nara/DASK.

----

## [585] Parameter-Efficient Sparsity for Large Language Models Fine-Tuning

**Authors**: *Yuchao Li, Fuli Luo, Chuanqi Tan, Mengdi Wang, Songfang Huang, Shen Li, Junjie Bai*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/586](https://doi.org/10.24963/ijcai.2022/586)

**Abstract**:

With the dramatically increased number of parameters in language models, sparsity methods have received ever-increasing research focus to compress and accelerate the models. While most research focuses on how to accurately retain appropriate weights while maintaining the performance of the compressed model, there are challenges in the computational overhead and memory footprint of sparse training when compressing large-scale language models. To address this problem, we propose a Parameter-efficient Sparse Training (PST) method to reduce the number of trainable parameters during sparse-aware training in downstream tasks. Specifically, we first combine the data-free and data-driven criteria to efficiently and accurately measure the importance of weights. Then we investigate the intrinsic redundancy of data-driven weight importance and derive two obvious characteristics i.e. low-rankness and structuredness. Based on that, two groups of small matrices are introduced to compute the data-driven importance of weights, instead of using the original large importance score matrix, which therefore makes the sparse training resource-efficient and parameter-efficient. Experiments with diverse networks (i.e. BERT, RoBERTa and GPT-2) on dozens of datasets demonstrate PST performs on par or better than previous sparsity methods, despite only training a small number of parameters. For instance, compared with previous sparsity methods, our PST only requires 1.5% trainable parameters to achieve comparable performance on BERT.

----

## [586] Explicit Alignment Learning for Neural Machine Translation

**Authors**: *Zuchao Li, Hai Zhao, Fengshun Xiao, Masao Utiyama, Eiichiro Sumita*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/587](https://doi.org/10.24963/ijcai.2022/587)

**Abstract**:

Even though neural machine translation (NMT) has become the state-of-the-art solution for end-to-end translation, it still suffers from a lack of translation interpretability, which may be conveniently enhanced by explicit alignment learning (EAL), as performed in traditional statistical machine translation (SMT). To provide the benefits of both NMT and SMT, this paper presents a novel model design that enhances NMT with an additional training process for EAL, in addition to the end-to-end translation training. 
Thus, we propose two approaches an explicit alignment learning approach, in which we further remove the need for the additional alignment model, and perform embedding mixup with the alignment based on encoder--decoder attention weights in the NMT model. We conducted experiments on both small-scale (IWSLT14 De->En and IWSLT13 Fr->En) and large-scale (WMT14 En->De, En->Fr, WMT17 Zh->En) benchmarks. Evaluation results show that our EAL methods significantly outperformed strong baseline methods, which shows the effectiveness of EAL. Further explorations show that the translation improvements are due to a better spatial alignment of the source and target language embeddings. Our method improves translation performance without the need to increase model parameters and training data, which verifies that the idea of incorporating techniques of SMT into NMT is worthwhile.

----

## [587] Lyra: A Benchmark for Turducken-Style Code Generation

**Authors**: *Qingyuan Liang, Zeyu Sun, Qihao Zhu, Wenjie Zhang, Lian Yu, Yingfei Xiong, Lu Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/588](https://doi.org/10.24963/ijcai.2022/588)

**Abstract**:

Recently, neural techniques have been used to generate source code automatically. While promising for declarative languages, these approaches achieve much poorer performance on datasets for imperative languages. Since a declarative language is typically embedded in an imperative language (i.e., the turducken-style programming) in real-world software development, the promising results on declarative languages can hardly lead to significant reduction of manual software development efforts.

In this paper, we define a new code generation task: given a natural language comment, this task aims to generate a program in a base imperative language with an embedded declarative language. To our knowledge, this is the first turducken-style code generation task. For this task, we present Lyra: a dataset in Python with embedded SQL. This dataset contains 2,000 carefully annotated database manipulation programs from real usage projects. Each program is paired with both a Chinese comment and an English comment. In our experiment, we adopted Transformer, BERT-style, and GPT-style models as baselines. In the best setting, GPT-style model can achieve 24% and 25.5% AST exact matching accuracy using Chinese and English comments, respectively. Therefore, we believe that Lyra provides a new challenge for code generation. Yet, overcoming this challenge may significantly boost the applicability of code generation techniques for real-world software development.

----

## [588] CUP: Curriculum Learning based Prompt Tuning for Implicit Event Argument Extraction

**Authors**: *Jiaju Lin, Qin Chen, Jie Zhou, Jian Jin, Liang He*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/589](https://doi.org/10.24963/ijcai.2022/589)

**Abstract**:

Implicit event argument extraction (EAE) aims to identify arguments that could scatter over the document. Most previous work focuses on learning the direct relations between arguments and the given trigger, while the implicit relations with long-range dependency are not well studied. Moreover, recent neural network based approaches rely on a large amount of labeled data for training, which is unavailable due to the high labelling cost. In this paper, we propose a Curriculum learning based Prompt tuning (CUP) approach, which resolves implicit EAE by four learning stages. The stages are defined according to the relations with the trigger node in a semantic graph, which well captures the long-range dependency between arguments and the trigger. In addition, we integrate a prompt-based encoder-decoder model to elicit related knowledge from pre-trained language models (PLMs) in each stage, where the prompt templates are adapted with the learning progress to enhance the reasoning for arguments. Experimental results on two well-known benchmark datasets show the great advantages of our proposed approach. In particular, we outperform the state-of-the-art models in both fully-supervised and low-data scenarios.

----

## [589] Low-Resource NER by Data Augmentation With Prompting

**Authors**: *Jian Liu, Yufeng Chen, Jinan Xu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/590](https://doi.org/10.24963/ijcai.2022/590)

**Abstract**:

Named entity recognition (NER) is a fundamental information extraction task that seeks to identify entity mentions of certain types in text.
Despite numerous advances, the existing NER methods rely on extensive supervision for model training, which struggle in a low-resource scenario with limited training data.
In this paper, we propose a new data augmentation method for low-resource NER, by eliciting knowledge from BERT with prompting strategies.
Particularly, we devise a label-conditioned word replacement strategy that can produce more label-consistent examples by capturing the underlying word-label dependencies, and a prompting with question answering method to generate new training data from unlabeled texts. 
The experimental results have widely confirmed the effectiveness of our approach.
Particularly, in a low-resource scenario with only 150 training sentences, our approach outperforms previous methods without data augmentation by over 40% in F1 and prior best data augmentation methods by over 2.0% in F1. Furthermore, our approach also fits with a zero-shot scenario, yielding promising results without using any human-labeled data for the task.

----

## [590] Generating a Structured Summary of Numerous Academic Papers: Dataset and Method

**Authors**: *Shuaiqi Liu, Jiannong Cao, Ruosong Yang, Zhiyuan Wen*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/591](https://doi.org/10.24963/ijcai.2022/591)

**Abstract**:

Writing a survey paper on one research topic usually needs to cover the salient content from numerous related papers, which can be modeled as a multi-document summarization (MDS) task. Existing MDS datasets usually focus on producing the structureless summary covering a few input documents. Meanwhile, previous structured summary generation works focus on summarizing a single document into a multi-section summary. These existing datasets and methods cannot meet the requirements of summarizing numerous academic papers into a structured summary. To deal with the scarcity of available data, we propose BigSurvey, the first large-scale dataset for generating comprehensive summaries of numerous academic papers on each topic. We collect target summaries from more than seven thousand survey papers and utilize their 430 thousand reference papersâ€™ abstracts as input documents. To organize the diverse content from dozens of input documents and ensure the efficiency of processing long text sequences, we propose a summarization method named category-based alignment and sparse transformer (CAST). The experimental results show that our CAST method outperforms various advanced summarization methods.

----

## [591] "My nose is running" "Are you also coughing?": Building A Medical Diagnosis Agent with Interpretable Inquiry Logics

**Authors**: *Wenge Liu, Yi Cheng, Hao Wang, Jianheng Tang, Yafei Liu, Ruihui Zhao, Wenjie Li, Yefeng Zheng, Xiaodan Liang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/592](https://doi.org/10.24963/ijcai.2022/592)

**Abstract**:

With the rise of telemedicine, the task of developing Dialogue Systems for Medical Diagnosis (DSMD) has received much attention in recent years. Different from early researches that needed to rely on extra human resources and expertise to build the system, recent researches focused on how to build DSMD in a data-driven manner. However, the previous data-driven DSMD methods largely overlooked the system interpretability, which is critical for a medical application, and they also suffered from the data sparsity issue at the same time. In this paper, we explore how to bring interpretability to data-driven DSMD. Specifically, we propose a more interpretable decision process to implement the dialogue manager of DSMD by reasonably mimicking real doctors' inquiry logics, and we devise a model with highly transparent components to conduct the inference. Moreover, we collect a new DSMD dataset, which has a much larger scale, more diverse patterns, and is of higher quality than the existing ones. The experiments show that our method obtains 7.7%, 10.0%, 3.0% absolute improvement in diagnosis accuracy respectively on three datasets, demonstrating the effectiveness of its rational decision process and model design. Our codes and the GMD-12 dataset are available at https://github.com/lwgkzl/BR-Agent.

----

## [592] Abstract Rule Learning for Paraphrase Generation

**Authors**: *Xianggen Liu, Wenqiang Lei, Jiancheng Lv, Jizhe Zhou*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/593](https://doi.org/10.24963/ijcai.2022/593)

**Abstract**:

In early years, paraphrase generation typically adopts rule-based methods, which are interpretable and able to make global transformations to the original sentence. But they struggle to produce fluent paraphrases. Recently, deep neural networks have shown impressive performances in generating paraphrases. However, the current neural models are black boxes and are prone to make local modifications to the inputs. In this work, we combine these two approaches into RULER, a novel approach that performs abstract rule learning for paraphrasing. The key idea is to explicitly learn generalizable rules that could enhance the paraphrase generation process of neural networks. In RULER, we first propose a rule generalizability metric to guide the model to generate rules underlying the paraphrasing. Then, we leverage neural networks to generate paraphrases by refining the sentences transformed by the learned rules. Extensive experimental results demonstrate the superiority of RULER over previous state-of-the-art methods in terms of paraphrase quality, generalization ability and interpretability.

----

## [593] Graph-based Dynamic Word Embeddings

**Authors**: *Yuyin Lu, Xin Cheng, Ziran Liang, Yanghui Rao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/594](https://doi.org/10.24963/ijcai.2022/594)

**Abstract**:

As time goes by, language evolves with word semantics changing. Unfortunately, traditional word embedding methods neglect the evolution of language and assume that word representations are static. Although contextualized word embedding models can capture the diverse representations of polysemous words, they ignore temporal information as well. To tackle the aforementioned challenges, we propose a graph-based dynamic word embedding (GDWE) model, which focuses on capturing the semantic drift of words continually. We introduce word-level knowledge graphs (WKGs) to store short-term and long-term knowledge. WKGs can provide rich structural information as supplement of lexical information, which help enhance the word embedding quality and capture semantic drift quickly. Theoretical analysis and extensive experiments validate the effectiveness of our GDWE on dynamic word embedding learning.

----

## [594] Searching for Optimal Subword Tokenization in Cross-domain NER

**Authors**: *Ruotian Ma, Yiding Tan, Xin Zhou, Xuanting Chen, Di Liang, Sirui Wang, Wei Wu, Tao Gui*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/595](https://doi.org/10.24963/ijcai.2022/595)

**Abstract**:

Input distribution shift is one of the vital problems in unsupervised domain adaptation (UDA). The most popular UDA approaches focus on domain-invariant representation learning, trying to align the features from different domains into a similar feature distribution. However, these approaches ignore the direct alignment of input word distributions between domains, which is a vital factor in word-level classification tasks such as cross-domain NER. In this work, we shed new light on cross-domain NER by introducing a subword-level solution, X-Piece, for input word-level distribution shift in NER. Specifically, we re-tokenize the input words of the source domain to approach the target subword distribution, which is formulated and solved as an optimal transport problem. As this approach focuses on the input level, it can also be combined with previous DIRL methods for further improvement. Experimental results show the effectiveness of the proposed method based on BERT-tagger on four benchmark NER datasets. Also, the proposed method is proved to benefit DIRL methods such as DANN.

----

## [595] Prompting to Distill: Boosting Data-Free Knowledge Distillation via Reinforced Prompt

**Authors**: *Xinyin Ma, Xinchao Wang, Gongfan Fang, Yongliang Shen, Weiming Lu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/596](https://doi.org/10.24963/ijcai.2022/596)

**Abstract**:

Data-free knowledge distillation (DFKD) conducts knowledge distillation via eliminating the dependence of original training data, and has recently achieved impressive results in accelerating pre-trained language models.  At the heart of DFKD is to reconstruct a synthetic dataset by inverting the parameters of the uncompressed model. Prior DFKD approaches, however, have largely relied on hand-crafted priors of the target data distribution for the reconstruction, which can be inevitably biased and often incompetent to capture the intrinsic distributions. To address this problem, we propose a prompt-based method, termed as PromptDFD, that allows us to take advantage of learned language priors, which effectively harmonizes the synthetic sentences to be semantically and grammatically correct. Specifically, PromptDFD leverages a pre-trained generative model to provide language priors and introduces a reinforced topic prompter to control data synthesis, making the generated samples thematically relevant and semantically plausible, and thus friendly to downstream tasks. As shown in our experiments, the proposed method substantially improves the synthesis quality and achieves considerable improvements on distillation performance. In some cases, PromptDFD even gives rise to results on par with those from the data-driven knowledge distillation with access to the original training data.

----

## [596] Variational Learning for Unsupervised Knowledge Grounded Dialogs

**Authors**: *Mayank Mishra, Dhiraj Madan, Gaurav Pandey, Danish Contractor*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/597](https://doi.org/10.24963/ijcai.2022/597)

**Abstract**:

Recent methods for knowledge grounded dialogs generate responses by incorporating information from an external textual document. These methods do not require the exact document to be known during training and rely on the use of a retrieval system to fetch relevant documents from a large index. The documents used to generate the responses are modeled as latent variables whose prior probabilities need to be estimated. Models such as RAG, marginalize the document probabilities over the documents retrieved from the index to define the log-likelihood loss function which is optimized end-to-end. 

In this paper, we develop a variational approach to the above technique wherein, we instead maximize the Evidence Lower bound (ELBO). Using a collection of three publicly available open-conversation datasets, we demonstrate how the posterior distribution, which has information from the ground-truth response, allows for a better approximation of the objective function during training.  To overcome the challenges associated with sampling over a large knowledge collection, we develop an efficient approach to approximate the ELBO. 
To the best of our knowledge, we are the first to apply variational training for open-scale unsupervised knowledge grounded dialog systems.

----

## [597] Enhancing Text Generation via Multi-Level Knowledge Aware Reasoning

**Authors**: *Feiteng Mu, Wenjie Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/598](https://doi.org/10.24963/ijcai.2022/598)

**Abstract**:

How to generate high-quality textual content is a non-trivial task. Existing methods generally generate text by grounding on word-level knowledge. However, word-level knowledge cannot express multi-word text units, hence existing methods may generate low-quality and unreasonable text. 
In this paper, we leverage event-level knowledge to enhance text generation. 
However, event knowledge is very sparse. To solve this problem, we split a coarse-grained event into fine-grained word components to obtain the word-level knowledge among event components. The word-level knowledge models the interaction among event components, which makes it possible to reduce the sparsity of events. Based on the event-level and the word-level knowledge, we devise a multi-level knowledge aware reasoning framework. Specifically, we first utilize event knowledge to make event-based content planning, i.e., select reasonable event sketches conditioned by the input text. Then, we combine the selected event sketches with the word-level knowledge for text generation. We validate our method on two widely used datasets, experimental results demonstrate the effectiveness of our framework to text generation.

----

## [598] Automatic Noisy Label Correction for Fine-Grained Entity Typing

**Authors**: *Weiran Pan, Wei Wei, Feida Zhu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/599](https://doi.org/10.24963/ijcai.2022/599)

**Abstract**:

Fine-grained entity typing (FET) aims to assign proper semantic types to entity mentions according to their context, which is a fundamental task in various entity-leveraging applications. Current FET systems usually establish on large-scale weaklysupervised/distantly annotation data, which may contain abundant noise and thus severely hinder the performance of the FET task. Although previous studies have made great success in automatically identifying the noisy labels in FET, they usually rely on some auxiliary resources which may be unavailable in real-world applications (e.g., pre-defined hierarchical type structures, humanannotated subsets). In this paper, we propose a novel approach to automatically correct noisy labels for FET without external resources. Specifically, it first identifies the potentially noisy labels by estimating the posterior probability of a label being positive or negative according to the logits output by the model, and then relabel candidate noisy labels by training a robust model over the remaining clean labels. Experiments on two popular benchmarks prove the effectiveness of our method. Our source code can be obtained from https://github.com/CCIIPLab/DenoiseFET.

----

## [599] Control Globally, Understand Locally: A Global-to-Local Hierarchical Graph Network for Emotional Support Conversation

**Authors**: *Wei Peng, Yue Hu, Luxi Xing, Yuqiang Xie, Yajing Sun, Yunpeng Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/600](https://doi.org/10.24963/ijcai.2022/600)

**Abstract**:

Emotional support conversation aims at reducing the emotional distress of the help-seeker, which is a new and challenging task. It requires the system to explore the cause of help-seeker's emotional distress and understand their psychological intention to provide supportive responses. However, existing methods mainly focus on the sequential contextual information, ignoring the hierarchical relationships with the global cause and local psychological intention behind conversations, thus leads to a weak ability of emotional support. In this paper, we propose a Global-to-Local Hierarchical Graph Network to capture the multi-source information (global cause, local intentions and dialog history) and model hierarchical relationships between them, which consists of a multi-source encoder, a hierarchical graph reasoner, and a global-guide decoder. Furthermore, a novel training objective is designed to monitor semantic information of the global cause. Experimental results on the emotional support conversation dataset, ESConv, confirm that the proposed GLHG has achieved the state-of-the-art performance on the automatic and human evaluations.

----



[Go to the previous page](IJCAI-2022-list02.md)

[Go to the next page](IJCAI-2022-list04.md)

[Go to the catalog section](README.md)