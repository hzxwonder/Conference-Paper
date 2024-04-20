## [1200] The decomposition of the higher-order homology embedding constructed from the $k$-Laplacian

        **Authors**: *Yu-Chia Chen, Marina Meila*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/842424a1d0595b76ec4fa03c46e8d755-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/842424a1d0595b76ec4fa03c46e8d755-Abstract.html)

        **Abstract**:

        The null space of the $k$-th order Laplacian $\mathbf{\mathcal L}_k$, known as the {\em $k$-th homology vector space}, encodes the non-trivial topology of a manifold or a network. Understanding the structure of the homology embedding can thus disclose geometric or topological information from the data. The study of the null space embedding of the graph Laplacian $\mathbf{\mathcal L}_0$ has spurred new research and applications, such as spectral clustering algorithms with theoretical guarantees and estimators of the Stochastic Block Model. In this work, we investigate the geometry of the $k$-th homology embedding and focus on cases reminiscent of spectral clustering. Namely, we analyze the {\em connected sum} of manifolds as a perturbation to the direct sum of their homology embeddings. We propose an algorithm to factorize the homology embedding into subspaces corresponding to a manifold's simplest topological components. The proposed framework is applied to the {\em shortest homologous loop detection} problem, a problem known to be NP-hard in general. Our spectral loop detection algorithm scales better than existing methods and is effective on diverse data such as point clouds and images.

        ----

        ## [1201] Breaking the Moments Condition Barrier: No-Regret Algorithm for Bandits with Super Heavy-Tailed Payoffs

        **Authors**: *Han Zhong, Jiayi Huang, Lin Yang, Liwei Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/843a4d7fb5b1641b0bb8e3c2b2e75231-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/843a4d7fb5b1641b0bb8e3c2b2e75231-Abstract.html)

        **Abstract**:

        Despite a large amount of effort in dealing with heavy-tailed error in machine learning, little is known when moments of the error can become non-existential: the random noise $\eta$ satisfies Pr$\left[|\eta| > |y|\right] \le 1/|y|^{\alpha}$ for some $\alpha > 0$. We make the first attempt to actively handle such super heavy-tailed noise in bandit learning problems:  We propose a novel robust statistical estimator, mean of medians, which estimates a random variable by computing the empirical mean of a sequence of empirical medians. We then present a generic reductionist algorithmic framework for solving bandit learning problems (including multi-armed and linear bandit problem): the mean of medians estimator can be applied to nearly any bandit learning algorithm as a black-box filtering for its reward signals and obtain similar regret bound as if the reward is sub-Gaussian. We show that the regret bound is near-optimal even with very heavy-tailed noise. We also empirically demonstrate the effectiveness of the proposed algorithm, which further corroborates our theoretical results.

        ----

        ## [1202] A nonparametric method for gradual change problems with statistical guarantees

        **Authors**: *Lizhen Nie, Dan Nicolae*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8452a95c40e2b232acd9b8a8712935d7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8452a95c40e2b232acd9b8a8712935d7-Abstract.html)

        **Abstract**:

        We consider the detection and localization of gradual changes in the distribution of a sequence of time-ordered observations. Existing literature focuses mostly on the simpler abrupt setting which assumes a discontinuity jump in distribution, and is unrealistic for some applied settings. We propose a general method for detecting and localizing gradual changes that does not require any specific data generating model, any particular data type, or any prior knowledge about which features of the distribution are subject to change. Despite relaxed assumptions, the proposed method possesses proven theoretical guarantees for both detection and localization.

        ----

        ## [1203] Nested Graph Neural Networks

        **Authors**: *Muhan Zhang, Pan Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8462a7c229aea03dde69da754c3bbcc4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8462a7c229aea03dde69da754c3bbcc4-Abstract.html)

        **Abstract**:

        Graph neural network (GNN)'s success in graph classification is closely related to the Weisfeiler-Lehman (1-WL) algorithm. By iteratively aggregating neighboring node features to a center node, both 1-WL and GNN obtain a node representation that encodes a rooted subtree around the center node. These rooted subtree representations are then pooled into a single representation to represent the whole graph. However, rooted subtrees are of limited expressiveness to represent a non-tree graph. To address it, we propose Nested Graph Neural Networks (NGNNs). NGNN represents a graph with rooted subgraphs instead of rooted subtrees, so that two graphs sharing many identical subgraphs (rather than subtrees) tend to have similar representations. The key is to make each node representation encode a subgraph around it more than a subtree. To achieve this, NGNN extracts a local subgraph around each node and applies a base GNN to each subgraph to learn a subgraph representation. The whole-graph representation is then obtained by pooling these subgraph representations. We provide a rigorous theoretical analysis showing that NGNN is strictly more powerful than 1-WL. In particular, we proved that NGNN can discriminate almost all r-regular graphs, where 1-WL always fails. Moreover, unlike other more powerful GNNs, NGNN only introduces a constant-factor higher time complexity than standard GNNs. NGNN is a plug-and-play framework that can be combined with various base GNNs. We test NGNN with different base GNNs on several benchmark datasets. NGNN uniformly improves their performance and shows highly competitive performance on all datasets.

        ----

        ## [1204] Multimodal and Multilingual Embeddings for Large-Scale Speech Mining

        **Authors**: *Paul-Ambroise Duquenne, Hongyu Gong, Holger Schwenk*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8466f9ace6a9acbe71f75762ffc890f1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8466f9ace6a9acbe71f75762ffc890f1-Abstract.html)

        **Abstract**:

        We present an approach to encode a speech signal into a fixed-size representation which minimizes the cosine loss with the existing massively multilingual LASER text embedding space. Sentences are close in this embedding space, independently of their language and modality, either text or audio. Using a similarity metric in that multimodal embedding space, we perform mining of audio in German, French, Spanish and English from Librivox against billions of sentences from Common Crawl. This yielded more than twenty thousand hours of aligned speech translations.  To evaluate the automatically mined speech/text corpora, we train neural speech translation systems for several languages pairs. Adding the mined data, achieves significant improvements in the BLEU score on the CoVoST2 and the MUST-C test sets with respect to a very competitive baseline. Our approach can also be used to directly perform speech-to-speech mining, without the need to first transcribe or translate the data. We obtain more than one thousand three hundred hours of aligned speech in French, German, Spanish and English. This speech corpus has the potential to boost research in speech-to-speech translation which suffers from scarcity of natural end-to-end training data. All the mined multimodal corpora will be made freely available.

        ----

        ## [1205] Necessary and sufficient graphical conditions for optimal adjustment sets in causal graphical models with hidden variables

        **Authors**: *Jakob Runge*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8485ae387a981d783f8764e508151cd9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8485ae387a981d783f8764e508151cd9-Abstract.html)

        **Abstract**:

        The problem of selecting optimal backdoor adjustment sets to estimate causal effects in graphical models with hidden and conditioned variables is addressed. Previous work has defined optimality as achieving the smallest asymptotic estimation variance and derived an optimal set for the case without hidden variables. For the case with hidden variables there can be settings where no optimal set exists and currently only a sufficient graphical optimality criterion of limited applicability has been derived. In the present work optimality is characterized as maximizing a certain adjustment information which allows to derive a necessary and sufficient graphical criterion for the existence of an optimal adjustment set and a definition and algorithm to construct it. Further, the optimal set is valid if and only if a valid adjustment set exists and has higher (or equal) adjustment information than the Adjust-set proposed in Perkovi{\'c} et~al. [Journal of Machine Learning Research, 18: 1--62, 2018] for any graph. The results translate to minimal asymptotic estimation variance for a class of estimators whose asymptotic variance follows a certain information-theoretic relation. Numerical experiments indicate that the asymptotic results also hold for relatively small sample sizes and that the optimal adjustment set or minimized variants thereof often yield better variance also beyond that estimator class. Surprisingly, among the randomly created setups more than 90\% fulfill the optimality conditions indicating that also in many real-world scenarios graphical optimality may hold.

        ----

        ## [1206] On Blame Attribution for Accountable Multi-Agent Sequential Decision Making

        **Authors**: *Stelios Triantafyllou, Adish Singla, Goran Radanovic*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/848c4965359e617d5e16c924b4a85fd9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/848c4965359e617d5e16c924b4a85fd9-Abstract.html)

        **Abstract**:

        Blame attribution is one of the key aspects of accountable decision making, as it provides means to quantify the responsibility of an agent for a decision making outcome. In this paper, we study blame attribution in the context of cooperative multi-agent sequential decision making. As a particular setting of interest, we focus on cooperative decision making formalized by Multi-Agent Markov Decision Processes (MMDPs), and we analyze different blame attribution methods derived from or inspired by existing concepts in cooperative game theory. We formalize desirable properties of blame attribution in the setting of interest, and we analyze the relationship between these properties and the studied blame attribution methods. Interestingly, we show that some of the well known blame attribution methods, such as Shapley value, are not performance-incentivizing, while others, such as Banzhaf index, may over-blame agents. To mitigate these value misalignment and fairness issues, we introduce a novel blame attribution method, unique in the set of properties it satisfies, which trade-offs explanatory power (by under-blaming agents) for the aforementioned properties.  We further show how to account for uncertainty about agents' decision making policies, and we experimentally: a) validate the qualitative properties of the studied blame attribution methods, and b) analyze their robustness to uncertainty.

        ----

        ## [1207] FLEX: Unifying Evaluation for Few-Shot NLP

        **Authors**: *Jonathan Bragg, Arman Cohan, Kyle Lo, Iz Beltagy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8493eeaccb772c0878f99d60a0bd2bb3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8493eeaccb772c0878f99d60a0bd2bb3-Abstract.html)

        **Abstract**:

        Few-shot NLP research is highly active, yet conducted in disjoint research threads with evaluation suites that lack challenging-yet-realistic testing setups and fail to employ careful experimental design. Consequently, the community does not know which techniques perform best or even if they outperform simple baselines. In response, we formulate the FLEX Principles, a set of requirements and best practices for unified, rigorous, valid, and cost-sensitive few-shot NLP evaluation. These principles include Sample Size Design, a novel approach to benchmark design that optimizes statistical accuracy and precision while keeping evaluation costs manageable. Following the principles, we release the FLEX benchmark, which includes four few-shot transfer settings, zero-shot evaluation, and a public leaderboard that covers diverse NLP tasks. In addition, we present UniFew, a prompt-based model for few-shot learning that unifies pretraining and finetuning prompt formats, eschewing complex machinery of recent prompt-based approaches in adapting downstream task formats to language model pretraining objectives. We demonstrate that despite simplicity, UniFew achieves results competitive with both popular meta-learning and prompt-based approaches.

        ----

        ## [1208] A flow-based latent state generative model of neural population responses to natural images

        **Authors**: *Mohammad Bashiri, Edgar Y. Walker, Konstantin-Klemens Lurz, Akshay Jagadish, Taliah Muhammad, Zhiwei Ding, Zhuokun Ding, Andreas S. Tolias, Fabian H. Sinz*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/84a529a92de322be42dd3365afd54f91-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/84a529a92de322be42dd3365afd54f91-Abstract.html)

        **Abstract**:

        We present a joint deep neural system identification model for two major sources of neural variability: stimulus-driven and stimulus-conditioned fluctuations. To this end, we combine (1) state-of-the-art deep networks for stimulus-driven activity and (2) a flexible, normalizing flow-based generative model to capture the stimulus-conditioned variability including noise correlations. This allows us to train the model end-to-end without the need for sophisticated probabilistic approximations associated with many latent state models for stimulus-conditioned fluctuations. We train the model on the responses of thousands of neurons from multiple areas of the mouse visual cortex to natural images. We show that our model outperforms previous state-of-the-art models in predicting the distribution of neural population responses to novel stimuli, including shared stimulus-conditioned variability. Furthermore, it successfully learns known latent factors of the population responses that are related to behavioral variables such as pupil dilation, and other factors that vary systematically with brain area or retinotopic location. Overall, our model accurately accounts for two critical sources of neural variability while avoiding several complexities associated with many existing latent state models. It thus provides a useful tool for uncovering the interplay between different factors that contribute to variability in neural activity.

        ----

        ## [1209] Learnable Fourier Features for Multi-dimensional Spatial Positional Encoding

        **Authors**: *Yang Li, Si Si, Gang Li, Cho-Jui Hsieh, Samy Bengio*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/84c2d4860a0fc27bcf854c444fb8b400-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/84c2d4860a0fc27bcf854c444fb8b400-Abstract.html)

        **Abstract**:

        Attentional mechanisms are order-invariant. Positional encoding is a crucial component to allow attention-based deep model architectures such as Transformer to address sequences or images where the position of information matters. In this paper, we propose a novel positional encoding method based on learnable Fourier features. Instead of hard-coding each position as a token or a vector, we represent each position, which can be multi-dimensional, as a trainable encoding based on learnable Fourier feature mapping, modulated  with  a  multi-layer perceptron. The representation is particularly advantageous for a spatial multi-dimensional position, e.g., pixel positions on an image, where $L_2$ distances or more complex positional relationships need to be captured. Our experiments based on several public benchmark tasks show that our learnable Fourier feature representation for multi-dimensional positional encoding outperforms existing methods by both improving the accuracy and allowing faster convergence.

        ----

        ## [1210] Doubly Robust Thompson Sampling with Linear Payoffs

        **Authors**: *Wonyoung Kim, Gi-Soo Kim, Myunghee Cho Paik*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/84d5711e9bf5547001b765878e7b0157-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/84d5711e9bf5547001b765878e7b0157-Abstract.html)

        **Abstract**:

        A challenging aspect of the bandit problem is that a stochastic reward is observed only for the chosen arm and the rewards of other arms remain missing.    The dependence of the arm choice on the past context and reward pairs compounds the complexity of regret analysis.We propose a novel multi-armed contextual bandit algorithm called Doubly Robust Thompson Sampling (DRTS) employing the doubly-robust estimator used in missing data literature to Thompson Sampling with contexts (\texttt{LinTS}).Different from previous works relying on missing data techniques (Dimakopoulou et al. [2019], Kim and Paik [2019]), the proposed algorithm is designed to allow a novel additive regret decomposition leading to an improved regret bound with the order of $\tilde{O}(\phi^{-2}\sqrt{T})$, where $\phi^2$ is the minimum eigenvalue of the covariance matrix of contexts.This is the first regret bound of \texttt{LinTS} using $\phi^2$ without $d$,  where $d$ is the dimension of the context.Applying the relationship between $\phi^2$ and $d$, the regret bound of the proposed algorithm is $\tilde{O}(d\sqrt{T})$ in many practical scenarios, improving the bound of \texttt{LinTS} by a factor of $\sqrt{d}$.A benefit of the proposed method is that it uses all the context data, chosen or not chosen, thus allowing to circumvent the technical definition of unsaturated arms used in theoretical analysis of \texttt{LinTS}.Empirical studies show the advantage of the proposed algorithm over \texttt{LinTS}.

        ----

        ## [1211] A Computationally Efficient Method for Learning Exponential Family Distributions

        **Authors**: *Abhin Shah, Devavrat Shah, Gregory W. Wornell*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/84f7e69969dea92a925508f7c1f9579a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/84f7e69969dea92a925508f7c1f9579a-Abstract.html)

        **Abstract**:

        We consider the question of learning the natural parameters of a $k$ parameter \textit{minimal} exponential family from i.i.d. samples in a computationally and statistically efficient manner. We focus on the setting where the support as well as the natural parameters are appropriately bounded. While the traditional maximum likelihood estimator for this class of exponential family is consistent, asymptotically normal, and asymptotically efficient, evaluating it is computationally hard. In this work, we propose a computationally efficient estimator that is consistent as well as asymptotically normal under mild conditions. We provide finite sample guarantees to achieve an ($\ell_2$) error of $\alpha$ in the parameter estimation  with sample complexity $O(\mathrm{poly}(k/\alpha))$ and computational complexity ${O}(\mathrm{poly}(k/\alpha))$. To establish these results, we show that, at the population level, our method can be viewed as the maximum likelihood estimation of a re-parameterized distribution belonging to the same class of exponential family.

        ----

        ## [1212] Rethinking Neural Operations for Diverse Tasks

        **Authors**: *Nicholas Roberts, Mikhail Khodak, Tri Dao, Liam Li, Christopher Ré, Ameet Talwalkar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/84fdbc3ac902561c00871c9b0c226756-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/84fdbc3ac902561c00871c9b0c226756-Abstract.html)

        **Abstract**:

        An important goal of AutoML is to automate-away the design of neural networks on new tasks in under-explored domains. Motivated by this goal, we study the problem of enabling users to discover the right neural operations given data from their specific domain. We introduce a search space of operations called XD-Operations that mimic the inductive bias of standard multi-channel convolutions while being much more expressive: we prove that it includes many named operations across multiple application areas. Starting with any standard backbone such as ResNet, we show how to transform it into a search space over XD-operations and how to traverse the space using a simple weight sharing scheme. On a diverse set of tasks—solving PDEs, distance prediction for protein folding, and music modeling—our approach consistently yields models with lower error than baseline networks and often even lower error than expert-designed domain-specific approaches.

        ----

        ## [1213] Motif-based Graph Self-Supervised Learning for Molecular Property Prediction

        **Authors**: *Zaixi Zhang, Qi Liu, Hao Wang, Chengqiang Lu, Chee-Kong Lee*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/85267d349a5e647ff0a9edcb5ffd1e02-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/85267d349a5e647ff0a9edcb5ffd1e02-Abstract.html)

        **Abstract**:

        Predicting molecular properties with data-driven methods has drawn much attention in recent years. Particularly, Graph Neural Networks (GNNs) have demonstrated remarkable success in various molecular generation and prediction tasks. In cases where labeled data is scarce, GNNs can be pre-trained on unlabeled molecular data to first learn the general semantic and structural information before being finetuned for specific tasks. However, most existing self-supervised pretraining frameworks for GNNs only focus on node-level or graph-level tasks. These approaches cannot capture the rich information in subgraphs or graph motifs. For example, functional groups (frequently-occurred subgraphs in molecular graphs)  often carry indicative information about the molecular properties. To bridge this gap, we propose Motif-based Graph Self-supervised Learning (MGSSL) by introducing a novel self-supervised motif generation framework for GNNs. First, for motif extraction from molecular graphs, we design a molecule fragmentation method that leverages a retrosynthesis-based algorithm BRICS and additional rules for controlling the size of motif vocabulary. Second, we design a general motif-based generative pretraining framework in which GNNs are asked to make topological and label predictions. This generative framework can be implemented in two different ways, i.e., breadth-first or depth-first. Finally, to take the multi-scale information in molecular graphs into consideration, we introduce a multi-level self-supervised pre-training. Extensive experiments on various downstream benchmark tasks show that our methods outperform all state-of-the-art baselines.

        ----

        ## [1214] On Inductive Biases for Heterogeneous Treatment Effect Estimation

        **Authors**: *Alicia Curth, Mihaela van der Schaar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8526e0962a844e4a2f158d831d5fddf7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8526e0962a844e4a2f158d831d5fddf7-Abstract.html)

        **Abstract**:

        We investigate how to exploit structural similarities of an individual's potential outcomes (POs) under different treatments to obtain better estimates of conditional average treatment effects in finite samples. Especially when it is unknown whether a treatment has an effect at all, it is natural to hypothesize that the POs are similar -- yet, some existing strategies for treatment effect estimation employ regularization schemes that implicitly encourage heterogeneity even when it does not exist and fail to fully make use of shared structure. In this paper, we investigate and compare three end-to-end learning strategies to overcome this problem -- based on regularization, reparametrization and a flexible multi-task architecture -- each encoding inductive bias favoring shared behavior across POs. To build understanding of their relative strengths, we implement all strategies using neural networks and conduct a wide range of semi-synthetic experiments. We observe that all three approaches can lead to substantial improvements upon numerous baselines and gain insight into performance differences across various experimental settings.

        ----

        ## [1215] DP-SSL: Towards Robust Semi-supervised Learning with A Few Labeled Samples

        **Authors**: *Yi Xu, Jiandong Ding, Lu Zhang, Shuigeng Zhou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/854d6fae5ee42911677c739ee1734486-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/854d6fae5ee42911677c739ee1734486-Abstract.html)

        **Abstract**:

        The scarcity of labeled data is a critical obstacle to deep learning. Semi-supervised learning (SSL) provides a promising way to leverage unlabeled data by pseudo labels. However, when the size of labeled data is very small (say a few labeled samples per class), SSL performs poorly and unstably, possibly due to the low quality of learned pseudo labels. In this paper, we propose a new SSL method called DP-SSL that adopts an innovative data programming (DP) scheme to generate probabilistic labels for unlabeled data. Different from existing DP methods that rely on human experts to provide initial labeling functions (LFs), we develop a multiple-choice learning~(MCL) based approach to automatically generate LFs from scratch in SSL style. With the noisy labels produced by the LFs, we design a label model to resolve the conflict and overlap among the noisy labels, and finally infer probabilistic labels for unlabeled samples. Extensive experiments on four standard SSL benchmarks show that DP-SSL can provide reliable labels for unlabeled data and achieve better classification performance on test sets than existing SSL methods, especially when only a small number of labeled samples are available. Concretely, for CIFAR-10 with only 40 labeled samples, DP-SSL achieves 93.82% annotation accuracy on unlabeled data and 93.46% classification accuracy on test data, which are higher than the SOTA results.

        ----

        ## [1216] Transformer in Transformer

        **Authors**: *Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, Yunhe Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/854d9fca60b4bd07f9bb215d59ef5561-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/854d9fca60b4bd07f9bb215d59ef5561-Abstract.html)

        **Abstract**:

        Transformer is a new kind of neural architecture which encodes the input data as powerful features via the attention mechanism. Basically, the visual transformers first divide the input images into several local patches and then calculate both representations and their relationship. Since natural images are of high complexity with abundant detail and color information, the granularity of the patch dividing is not fine enough for excavating features of objects in different scales and locations. In this paper, we point out that the attention inside these local patches are also essential for building visual transformers with high performance and we explore a new architecture, namely, Transformer iN Transformer (TNT). Specifically, we regard the local patches (\eg, 16$\times$16) as “visual sentences” and present to further divide them into smaller patches (\eg, 4$\times$4) as “visual words”. The attention of each word will be calculated with other words in the given visual sentence with negligible computational costs. Features of both words and sentences will be aggregated to enhance the representation ability. Experiments on several benchmarks demonstrate the effectiveness of the proposed TNT architecture, \eg, we achieve an 81.5\% top-1 accuracy on the ImageNet, which is about 1.7\% higher than that of the state-of-the-art visual transformer with similar computational cost. The PyTorch code is available at \url{https://github.com/huawei-noah/CV-Backbones}, and the MindSpore code is available at \url{https://gitee.com/mindspore/models/tree/master/research/cv/TNT}.

        ----

        ## [1217] Adversarial Graph Augmentation to Improve Graph Contrastive Learning

        **Authors**: *Susheel Suresh, Pan Li, Cong Hao, Jennifer Neville*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/854f1fb6f65734d9e49f708d6cd84ad6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/854f1fb6f65734d9e49f708d6cd84ad6-Abstract.html)

        **Abstract**:

        Self-supervised learning of graph neural networks (GNN) is in great need because of the widespread label scarcity issue in real-world graph/network data. Graph contrastive learning (GCL), by training GNNs to maximize the correspondence between the representations of the same graph in its different augmented forms, may yield robust and transferable GNNs even without using labels. However, GNNs trained by traditional GCL often risk capturing redundant graph features and thus may be brittle and provide sub-par performance in downstream tasks. Here, we propose a novel principle, termed adversarial-GCL (\textit{AD-GCL}), which enables GNNs to avoid capturing redundant information during the training by optimizing adversarial graph augmentation strategies used in GCL. We pair AD-GCL with theoretical explanations and design a practical instantiation based on trainable edge-dropping graph augmentation. We experimentally validate AD-GCL by comparing with the state-of-the-art GCL methods and achieve performance gains of up-to~14\% in unsupervised, ~6\% in transfer and~3\% in semi-supervised learning settings overall with 18 different benchmark datasets for the tasks of molecule property regression and classification, and social network classification.

        ----

        ## [1218] Online Control of Unknown Time-Varying Dynamical Systems

        **Authors**: *Edgar Minasyan, Paula Gradu, Max Simchowitz, Elad Hazan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/856b503e276cc491e7e6e0ac1b9f4b17-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/856b503e276cc491e7e6e0ac1b9f4b17-Abstract.html)

        **Abstract**:

        We study online control of time-varying linear systems with unknown dynamics in the nonstochastic control model. At a high level, we demonstrate that this setting is \emph{qualitatively harder} than that of either unknown time-invariant or known time-varying dynamics, and complement our negative results with algorithmic upper bounds in regimes where sublinear regret is possible. More specifically, we study regret bounds with respect to common classes of policies: Disturbance Action (SLS), Disturbance Response (Youla), and linear feedback policies. While these three classes are essentially equivalent for LTI systems, we demonstrate that these equivalences break down for time-varying systems. We prove a lower bound that no algorithm can obtain sublinear regret with respect to the first two classes unless a certain measure of system variability also scales sublinearly in the horizon. Furthermore, we show that offline planning over the state linear feedback policies is NP-hard, suggesting hardness of the online learning problem. On the positive side, we give an efficient algorithm that attains a sublinear regret bound against the class of Disturbance Response policies up to the aforementioned system variability term. In fact, our algorithm enjoys sublinear \emph{adaptive} regret bounds, which is a strictly stronger metric than standard regret and is more appropriate for time-varying systems. We sketch extensions to Disturbance Action policies and partial observation, and propose an inefficient algorithm for regret against linear state feedback policies.

        ----

        ## [1219] Contrastive Reinforcement Learning of Symbolic Reasoning Domains

        **Authors**: *Gabriel Poesia, Wenxin Dong, Noah D. Goodman*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/859555c74e9afd45ab771c615c1e49a6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/859555c74e9afd45ab771c615c1e49a6-Abstract.html)

        **Abstract**:

        Abstract symbolic reasoning, as required in domains such as mathematics and logic, is a key component of human intelligence. Solvers for these domains have important applications, especially to computer-assisted education. But learning to solve symbolic problems is challenging for machine learning algorithms. Existing models either learn from human solutions or use hand-engineered features, making them expensive to apply in new domains. In this paper, we instead consider symbolic domains as simple environments where states and actions are given as unstructured text, and binary rewards indicate whether a problem is solved. This flexible setup makes it easy to specify new domains, but search and planning become challenging. We introduce five environments inspired by the Mathematics Common Core Curriculum, and observe that existing Reinforcement Learning baselines perform poorly. We then present a novel learning algorithm, Contrastive Policy Learning (ConPoLe) that explicitly optimizes the InfoNCE loss, which lower bounds the mutual information between the current state and next states that continue on a path to the solution. ConPoLe successfully solves all four domains. Moreover, problem representations learned by ConPoLe enable accurate prediction of the categories of problems in a real mathematics curriculum. Our results suggest new directions for reinforcement learning in symbolic domains, as well as applications to mathematics education.

        ----

        ## [1220] Spatial Ensemble: a Novel Model Smoothing Mechanism for Student-Teacher Framework

        **Authors**: *Tengteng Huang, Yifan Sun, Xun Wang, Haotian Yao, Chi Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8597a6cfa74defcbde3047c891d78f90-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8597a6cfa74defcbde3047c891d78f90-Abstract.html)

        **Abstract**:

        Model smoothing is of central importance for obtaining a reliable teacher model in the student-teacher framework, where the teacher generates surrogate supervision signals to train the student. A popular model smoothing method is the Temporal Moving Average (TMA), which continuously averages the teacher parameters with the up-to-date student parameters. In this paper, we propose ''Spatial Ensemble'', a novel model smoothing mechanism in parallel with TMA. Spatial Ensemble randomly picks up a small fragment of the student model to directly replace the corresponding fragment of the teacher model. Consequentially, it stitches different fragments of historical student models into a unity, yielding the ''Spatial Ensemble'' effect. Spatial Ensemble obtains comparable student-teacher learning performance by itself and demonstrates valuable complementarity with temporal moving average. Their integration, named Spatial-Temporal Smoothing, brings general (sometimes significant) improvement to the student-teacher learning framework on a variety of state-of-the-art methods. For example, based on the self-supervised method BYOL, it yields +0.9% top-1 accuracy improvement on ImageNet, while based on the semi-supervised approach FixMatch, it increases the top-1 accuracy by around +6% on CIFAR-10 when only few training labels are available. Codes and models are available at: https://github.com/tengteng95/Spatial_Ensemble.

        ----

        ## [1221] Probabilistic Tensor Decomposition of Neural Population Spiking Activity

        **Authors**: *Hugo Soulat, Sepiedeh Keshavarzi, Troy W. Margrie, Maneesh Sahani*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/859b755563f548d008f936906a959c8f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/859b755563f548d008f936906a959c8f-Abstract.html)

        **Abstract**:

        The firing of neural populations is coordinated across cells, in time, and across experimentalconditions or repeated experimental trials; and so a full understanding of the computationalsignificance of neural responses must be based on a separation of these different contributions tostructured activity.Tensor decomposition is an approach to untangling the influence of multiple factors in data that iscommon in many fields.  However, despite some recent interest in neuroscience, wider applicabilityof the approach is hampered by the lack of a full probabilistic treatment allowing principledinference of a decomposition from non-Gaussian spike-count data.Here, we extend the PÃ³lya-Gamma (PG) augmentation, previously used in sampling-based Bayesianinference, to implement scalable variational inference in non-conjugate spike-count models.Using this new approach, we develop techniques related to automatic relevance determination to inferthe most appropriate tensor rank, as well as to incorporate priors based on known brain anatomy suchas the segregation of cell response properties by brain area.We apply the model to neural recordings taken under conditions of visual-vestibular sensoryintegration, revealing how the encoding of self- and visual-motion signals is modulated by thesensory information available to the animal.

        ----

        ## [1222] Recurrent Bayesian Classifier Chains for Exact Multi-Label Classification

        **Authors**: *Walter Gerych, Thomas Hartvigsen, Luke Buquicchio, Emmanuel Agu, Elke A. Rundensteiner*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/859bf1416b8b8761c5d588dee78dc65f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/859bf1416b8b8761c5d588dee78dc65f-Abstract.html)

        **Abstract**:

        Exact multi-label classification is the task of assigning each datapoint a set of class labels such that the assigned set exactly matches the ground truth. Optimizing for exact multi-label classification is important in domains where missing a single label can be especially costly, such as in object detection for autonomous vehicles or symptom classification for disease diagnosis. Recurrent Classifier Chains (RCCs), a recurrent neural network extension of ensemble-based classifier chains, are the state-of-the-art exact multi-label classification method for maximizing subset accuracy. However, RCCs iteratively predict classes with an unprincipled ordering, and therefore indiscriminately condition class probabilities. These disadvantages make RCCs prone to predicting inaccurate label sets. In this work we propose Recurrent Bayesian Classifier Chains (RBCCs), which learn a Bayesian network of class dependencies and leverage this network in order to condition the prediction of child nodes only on their parents. By conditioning predictions in this way, we perform principled and non-noisy class prediction.  We demonstrate the effectiveness of our RBCC method on a variety of real-world multi-label datasets, where we routinely outperform the state of the art methods for exact multi-label classification.

        ----

        ## [1223] Wasserstein Flow Meets Replicator Dynamics: A Mean-Field Analysis of Representation Learning in Actor-Critic

        **Authors**: *Yufeng Zhang, Siyu Chen, Zhuoran Yang, Michael I. Jordan, Zhaoran Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/85a4413ecea7122bcc399cf0a53bba26-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/85a4413ecea7122bcc399cf0a53bba26-Abstract.html)

        **Abstract**:

        Actor-critic  (AC) algorithms, empowered by neural networks, have had significant empirical success in recent years. However, most of the existing theoretical support for AC algorithms focuses on the case of linear function approximations, or linearized neural networks, where the feature representation is fixed throughout training. Such a limitation fails to capture the key aspect of representation learning in neural AC, which is pivotal in practical problems. In this work, we take a mean-field perspective on the evolution and convergence of feature-based neural AC. Specifically, we consider a version of  AC where the actor and critic are represented by overparameterized two-layer neural networks and are updated with two-timescale learning rates. The critic is updated by temporal-difference (TD) learning with a larger stepsize while the actor is updated via proximal policy optimization (PPO) with a smaller stepsize. In the continuous-time and infinite-width limiting regime, when the timescales are properly separated, we prove that neural AC finds the globally optimal policy at a sublinear rate. Additionally,  we prove that the feature representation induced by the critic network is allowed to evolve within a neighborhood of the initial one.

        ----

        ## [1224] Assessing Fairness in the Presence of Missing Data

        **Authors**: *Yiliang Zhang, Qi Long*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/85dca1d270f7f9aef00c9d372f114482-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/85dca1d270f7f9aef00c9d372f114482-Abstract.html)

        **Abstract**:

        Missing data are prevalent and present daunting challenges in real data analysis. While there is a growing body of literature on fairness in analysis of fully observed data, there has been little theoretical work on investigating fairness in analysis of incomplete data. In practice, a popular analytical approach for dealing with missing data is to use only the set of complete cases, i.e., observations with all features fully observed to train a prediction algorithm. However, depending on the missing data mechanism, the distribution of complete cases and the distribution of the complete data may be substantially different. When the goal is to develop a fair algorithm in the complete data domain where there are no missing values, an algorithm that is fair in the complete case domain may show disproportionate bias towards some marginalized groups in the complete data domain. To fill this significant gap, we study the problem of estimating fairness in the complete data domain for an arbitrary model evaluated merely using complete cases. We provide upper and lower bounds on the fairness estimation error and conduct numerical experiments to assess our theoretical results. Our work provides the first known theoretical results on fairness guarantee in analysis of incomplete data.

        ----

        ## [1225] Adversarial Attack Generation Empowered by Min-Max Optimization

        **Authors**: *Jingkang Wang, Tianyun Zhang, Sijia Liu, Pin-Yu Chen, Jiacen Xu, Makan Fardad, Bo Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/85e5526a360b0bcf082d8d42e7bf100b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/85e5526a360b0bcf082d8d42e7bf100b-Abstract.html)

        **Abstract**:

        The worst-case training principle that minimizes the maximal adversarial loss, also known as adversarial training (AT), has shown to be a state-of-the-art approach for enhancing adversarial robustness. Nevertheless, min-max optimization beyond the purpose of AT has not been rigorously explored in the adversarial context. In this paper, we show how a general notion of min-max optimization over multiple domains can be leveraged to the design of different types of adversarial attacks. In particular, given a set of risk sources, minimizing the worst-case attack loss can be reformulated as a min-max problem by introducing domain weights that are maximized over the probability simplex of the domain set. We showcase this unified framework in three attack generation problems -- attacking model ensembles, devising universal perturbation under multiple inputs, and crafting attacks resilient to data transformations. Extensive experiments demonstrate that our approach leads to substantial attack improvement over the existing heuristic strategies as well as robustness improvement over state-of-the-art defense methods against multiple perturbation types. Furthermore, we find that the self-adjusted domain weights learned from min-max optimization can provide a holistic tool to explain the difficulty level of attack across domains.

        ----

        ## [1226] Safe Pontryagin Differentiable Programming

        **Authors**: *Wanxin Jin, Shaoshuai Mou, George J. Pappas*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/85ea6fd7a2ca3960d0cf5201933ac998-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/85ea6fd7a2ca3960d0cf5201933ac998-Abstract.html)

        **Abstract**:

        We propose a Safe Pontryagin Differentiable Programming (Safe PDP) methodology, which establishes a theoretical and algorithmic  framework to solve a broad class of safety-critical learning and control tasks---problems that require the guarantee of safety constraint satisfaction at any stage of the learning and control progress.  In the spirit of interior-point methods,   Safe PDP handles different types of system  constraints on states and inputs by incorporating them into the cost or loss through barrier functions. We prove three fundamentals  of the proposed  Safe PDP:  first, both the  solution and its gradient in the backward pass can be approximated by solving their  more efficient unconstrained counterparts;  second,   the approximation for both the  solution and its gradient can be controlled for arbitrary accuracy by a  barrier parameter;   and third,   importantly, all intermediate results throughout the approximation and optimization  strictly respect the  constraints,  thus guaranteeing safety throughout the entire learning and control process. We demonstrate the capabilities of   Safe PDP in solving various safety-critical tasks,  including safe policy optimization, safe motion planning, and learning MPCs from demonstrations, on different challenging systems such as 6-DoF maneuvering quadrotor and 6-DoF rocket powered landing.

        ----

        ## [1227] Class-Disentanglement and Applications in Adversarial Detection and Defense

        **Authors**: *Kaiwen Yang, Tianyi Zhou, Yonggang Zhang, Xinmei Tian, Dacheng Tao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8606f35ec6c77858dfb80a385d0d1151-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8606f35ec6c77858dfb80a385d0d1151-Abstract.html)

        **Abstract**:

        What is the minimum necessary information required by a neural net $D(\cdot)$ from an image $x$ to accurately predict its class? Extracting such information in the input space from $x$ can allocate the areas $D(\cdot)$ mainly attending to and shed novel insights to the detection and defense of adversarial attacks. In this paper, we propose ''class-disentanglement'' that trains a variational autoencoder $G(\cdot)$ to extract this class-dependent information as $x - G(x)$ via a trade-off between reconstructing $x$ by $G(x)$ and classifying $x$ by $D(x-G(x))$, where the former competes with the latter in decomposing $x$ so the latter retains only necessary information for classification in $x-G(x)$. We apply it to both clean images and their adversarial images and discover that the perturbations generated by adversarial attacks mainly lie in the class-dependent part $x-G(x)$. The decomposition results also provide novel interpretations to classification and attack models. Inspired by these observations, we propose to conduct adversarial detection and adversarial defense respectively on $x - G(x)$ and $G(x)$, which consistently outperform the results on the original $x$. In experiments, this simple approach substantially improves the detection and defense against different types of adversarial attacks.

        ----

        ## [1228] Active 3D Shape Reconstruction from Vision and Touch

        **Authors**: *Edward J. Smith, David Meger, Luis Pineda, Roberto Calandra, Jitendra Malik, Adriana Romero-Soriano, Michal Drozdzal*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8635b5fd6bc675033fb72e8a3ccc10a0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8635b5fd6bc675033fb72e8a3ccc10a0-Abstract.html)

        **Abstract**:

        Humans build 3D understandings of the world through active object exploration, using jointly their senses of vision and touch. However, in 3D shape reconstruction, most recent progress has relied on static datasets of limited sensory data such as RGB images, depth maps or haptic readings, leaving the active exploration of the shape largely unexplored. In active touch sensing for 3D reconstruction, the goal is to actively select the tactile readings that maximize the improvement in shape reconstruction accuracy. However, the development of deep learning-based active touch models is largely limited by the lack of frameworks for shape exploration. In this paper, we focus on this problem and introduce a system composed of: 1) a haptic simulator leveraging high spatial resolution vision-based tactile sensors for active touching of 3D objects; 2) a mesh-based 3D shape reconstruction model that relies on tactile or visuotactile signals; and 3) a set of data-driven solutions with either tactile or visuotactile priors to guide the shape exploration. Our framework enables the development of the first fully data-driven solutions to active touch on top of learned models for object understanding. Our experiments show the benefits of such solutions in the task of 3D shape understanding where our models consistently outperform natural baselines. We provide our framework as a tool to foster future research in this direction.

        ----

        ## [1229] CAPE: Encoding Relative Positions with Continuous Augmented Positional Embeddings

        **Authors**: *Tatiana Likhomanenko, Qiantong Xu, Gabriel Synnaeve, Ronan Collobert, Alex Rogozhnikov*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/865bf46435bd84fa5d89f64cf3ba7347-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/865bf46435bd84fa5d89f64cf3ba7347-Abstract.html)

        **Abstract**:

        Without positional information, attention-based Transformer neural networks are permutation-invariant. Absolute or relative positional embeddings are the most popular ways to feed Transformer models with positional information. Absolute positional embeddings are simple to implement, but suffer from generalization issues when evaluating on sequences longer than seen at training time. Relative positions are more robust to input length change, but are more complex to implement and yield inferior model throughput due to extra computational and memory costs. In this paper, we propose an augmentation-based approach (CAPE) for absolute positional embeddings, which keeps the advantages of both absolute (simplicity and speed) and relative positional embeddings (better generalization). In addition, our empirical evaluation on state-of-the-art models in machine translation, image and speech recognition demonstrates that CAPE leads to better generalization performance as well as increased stability with respect to training hyper-parameters.

        ----

        ## [1230] Multi-armed Bandit Requiring Monotone Arm Sequences

        **Authors**: *Ningyuan Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/865dfbde8a344b44095495f3591f7407-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/865dfbde8a344b44095495f3591f7407-Abstract.html)

        **Abstract**:

        In many online learning or multi-armed bandit problems, the taken actions or pulled arms are ordinal and required to be monotone over time. Examples include dynamic pricing, in which the firms use markup pricing policies to please early adopters and deter strategic waiting, and clinical trials, in which the dose allocation usually follows the dose escalation principle to prevent dose limiting toxicities. We consider the continuum-armed bandit problem when the arm sequence is required to be monotone. We show that when the unknown objective function is Lipschitz continuous, the regret is $O(T)$. When in addition the objective function is unimodal or quasiconcave, the regret is $\tilde O(T^{3/4})$ under the proposed algorithm, which is also shown to be the optimal rate. This deviates from the optimal rate $\tilde O(T^{2/3})$ in the continuous-armed bandit literature and demonstrates the cost to the learning efficiency brought by the monotonicity requirement.

        ----

        ## [1231] Gradient Driven Rewards to Guarantee Fairness in Collaborative Machine Learning

        **Authors**: *Xinyi Xu, Lingjuan Lyu, Xingjun Ma, Chenglin Miao, Chuan Sheng Foo, Bryan Kian Hsiang Low*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8682cc30db9c025ecd3fee433f8ab54c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8682cc30db9c025ecd3fee433f8ab54c-Abstract.html)

        **Abstract**:

        In collaborative machine learning(CML), multiple agents pool their resources(e.g., data) together for a common learning task. In realistic CML settings where the agents are self-interested and not altruistic, they may be unwilling to share data or model information without adequate rewards. Furthermore, as the data/model information shared by the agents may differ in quality, designing rewards which are fair to them is important so that they would not feel exploited nor discouraged from sharing. In this paper, we adopt federated learning as the CML paradigm, propose a novel cosine gradient Shapley value(CGSV) to fairly evaluate the expected marginal contribution of each agent’s uploaded model parameter update/gradient without needing an auxiliary validation dataset, and based on the CGSV, design a novel training-time gradient reward mechanism with a fairness guarantee by sparsifying the aggregated parameter update/gradient downloaded from the server as reward to each agent such that its resulting quality is commensurate to that of the agent’s uploaded parameter update/gradient. We empirically demonstrate the effectiveness of our fair gradient reward mechanism on multiple benchmark datasets in terms of fairness, predictive performance, and time overhead.

        ----

        ## [1232] Generalizable Imitation Learning from Observation via Inferring Goal Proximity

        **Authors**: *Youngwoon Lee, Andrew Szot, Shao-Hua Sun, Joseph J. Lim*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/868b7df964b1af24c8c0a9e43a330c6a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/868b7df964b1af24c8c0a9e43a330c6a-Abstract.html)

        **Abstract**:

        Task progress is intuitive and readily available task information that can guide an agent closer to the desired goal. Furthermore, a task progress estimator can generalize to new situations. From this intuition, we propose a simple yet effective imitation learning from observation method for a goal-directed task using a learned goal proximity function as a task progress estimator for better generalization to unseen states and goals. We obtain this goal proximity function from expert demonstrations and online agent experience, and then use the learned goal proximity as a dense reward for policy training. We demonstrate that our proposed method can robustly generalize compared to prior imitation learning methods on a set of goal-directed tasks in navigation, locomotion, and robotic manipulation, even with demonstrations that cover only a part of the states.

        ----

        ## [1233] DualNet: Continual Learning, Fast and Slow

        **Authors**: *Quang Pham, Chenghao Liu, Steven C. H. Hoi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/86a1fa88adb5c33bd7a68ac2f9f3f96b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/86a1fa88adb5c33bd7a68ac2f9f3f96b-Abstract.html)

        **Abstract**:

        According to Complementary Learning Systems (CLS) theory~\cite{mcclelland1995there} in neuroscience, humans do effective \emph{continual learning} through two complementary systems: a fast learning system centered on the hippocampus for rapid learning of the specifics and individual experiences, and a slow learning system located in the neocortex for the gradual acquisition of structured knowledge about the environment. Motivated by this theory, we propose a novel continual learning framework named ``DualNet", which comprises a fast learning system for supervised learning of pattern-separated representation from specific tasks and a slow learning system for unsupervised representation learning of task-agnostic general representation via a Self-Supervised Learning (SSL) technique. The two fast and slow learning systems are complementary and work seamlessly in a holistic continual learning framework. Our extensive experiments on two challenging continual learning benchmarks of CORE50 and miniImageNet show that DualNet outperforms state-of-the-art continual learning methods by a large margin. We further conduct ablation studies of different SSL objectives to validate DualNet's efficacy, robustness, and scalability. Code is publicly available at \url{https://github.com/phquang/DualNet}.

        ----

        ## [1234] Deformable Butterfly: A Highly Structured and Sparse Linear Transform

        **Authors**: *Rui Lin, Jie Ran, King Hung Chiu, Graziano Chesi, Ngai Wong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/86b122d4358357d834a87ce618a55de0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/86b122d4358357d834a87ce618a55de0-Abstract.html)

        **Abstract**:

        We introduce a new kind of linear transform named Deformable Butterfly (DeBut) that generalizes the conventional butterfly matrices and can be adapted to various input-output dimensions. It inherits the fine-to-coarse-grained learnable hierarchy of traditional butterflies and when deployed to neural networks, the prominent structures and sparsity in a DeBut layer constitutes a new way for network compression. We apply DeBut as a drop-in replacement of standard fully connected and convolutional layers, and demonstrate its superiority in homogenizing a neural network and rendering it favorable properties such as light weight and low inference complexity, without compromising accuracy. The natural complexity-accuracy tradeoff arising from the myriad deformations of a DeBut layer also opens up new rooms for analytical and practical research. The codes and Appendix are publicly available at: https://github.com/ruilin0212/DeBut.

        ----

        ## [1235] Why Do Pretrained Language Models Help in Downstream Tasks? An Analysis of Head and Prompt Tuning

        **Authors**: *Colin Wei, Sang Michael Xie, Tengyu Ma*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/86b3e165b8154656a71ffe8a327ded7d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/86b3e165b8154656a71ffe8a327ded7d-Abstract.html)

        **Abstract**:

        Pretrained language models have achieved state-of-the-art performance when adapted to a downstream NLP task. However, theoretical analysis of these models is scarce and challenging since the pretraining and downstream tasks can be very different. We propose an analysis framework that links the pretraining and downstream tasks with an underlying latent variable generative model of text -- the downstream classifier must recover a function of the posterior distribution over the latent variables. We analyze head tuning (learning a classifier on top of the frozen pretrained model) and prompt tuning in this setting. The generative model in our analysis is either a Hidden Markov Model (HMM) or an HMM augmented with a latent memory component, motivated by long-term dependencies in natural language. We show that 1) under certain non-degeneracy conditions on the HMM, simple classification heads can solve the downstream task, 2) prompt tuning obtains downstream guarantees with weaker non-degeneracy conditions, and 3) our recovery guarantees for the memory-augmented HMM are stronger than for the vanilla HMM because task-relevant information is easier to recover from the long-term memory. Experiments on synthetically generated data from HMMs back our theoretical findings.

        ----

        ## [1236] Learning Diverse Policies in MOBA Games via Macro-Goals

        **Authors**: *Yiming Gao, Bei Shi, Xueying Du, Liang Wang, Guangwei Chen, Zhenjie Lian, Fuhao Qiu, Guoan Han, Weixuan Wang, Deheng Ye, Qiang Fu, Wei Yang, Lanxiao Huang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/86dba86754c0ad93997a11fa947d97b2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/86dba86754c0ad93997a11fa947d97b2-Abstract.html)

        **Abstract**:

        Recently, many researchers have made successful progress in building the AI systems for MOBA-game-playing with deep reinforcement learning, such as on Dota 2 and Honor of Kings. Even though these AI systems have achieved or even exceeded human-level performance, they still suffer from the lack of policy diversity. In this paper, we propose a novel Macro-Goals Guided framework, called MGG, to learn diverse policies in MOBA games. MGG abstracts strategies as macro-goals from human demonstrations and trains a Meta-Controller to predict these macro-goals. To enhance policy diversity, MGG samples macro-goals from the Meta-Controller prediction and guides the training process towards these goals. Experimental results on the typical MOBA game Honor of Kings demonstrate that MGG can execute diverse policies in different matches and lineups, and also outperform the state-of-the-art methods over 102 heroes.

        ----

        ## [1237] Evaluation of Human-AI Teams for Learned and Rule-Based Agents in Hanabi

        **Authors**: *Ho Chit Siu, Jaime Daniel Peña, Edenna Chen, Yutai Zhou, Victor J. Lopez, Kyle Palko, Kimberlee C. Chang, Ross E. Allen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)

        **Abstract**:

        Deep reinforcement learning has generated superhuman AI in competitive games such as Go and StarCraft. Can similar learning techniques create a superior AI teammate for human-machine collaborative games? Will humans prefer AI teammates that improve objective team performance or those that improve subjective metrics of trust? In this study, we perform a single-blind evaluation of teams of humans and AI agents in the cooperative card game Hanabi, with both rule-based and learning-based agents. In addition to the game score, used as an objective metric of the human-AI team performance, we also quantify subjective measures of the human's perceived performance, teamwork, interpretability, trust, and overall preference of AI teammate. We find that humans have a clear preference toward a rule-based AI teammate (SmartBot) over a state-of-the-art learning-based AI teammate (Other-Play) across nearly all subjective metrics, and generally view the learning-based agent negatively, despite no statistical difference in the game score. This result has implications for future AI design and reinforcement learning benchmarking, highlighting the need to incorporate subjective metrics of human-AI teaming rather than a singular focus on objective task performance.

        ----

        ## [1238] Counterfactual Invariance to Spurious Correlations in Text Classification

        **Authors**: *Victor Veitch, Alexander D'Amour, Steve Yadlowsky, Jacob Eisenstein*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8710ef761bbb29a6f9d12e4ef8e4379c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8710ef761bbb29a6f9d12e4ef8e4379c-Abstract.html)

        **Abstract**:

        Informally, a 'spurious correlation' is the dependence of a model on some aspect of the input data that an analyst thinks shouldn't matter. In machine learning, these have a know-it-when-you-see-it character; e.g., changing the gender of a sentence's subject changes a sentiment predictor's output. To check for spurious correlations, we can 'stress test' models by perturbing irrelevant parts of input data and seeing if model predictions change. In this paper, we study stress testing using the tools of causal inference. We introduce counterfactual invariance as a formalization of the requirement that changing irrelevant parts of the input shouldn't change model predictions. We connect counterfactual invariance to out-of-domain model performance, and provide practical schemes for learning (approximately) counterfactual invariant predictors (without access to counterfactual examples). It turns out that both the means and implications of counterfactual invariance depend fundamentally on the true underlying causal structure of the data---in particular, whether the label causes the features or the features cause the label. Distinct causal structures require distinct regularization schemes to induce counterfactual invariance. Similarly, counterfactual invariance implies different domain shift guarantees depending on the underlying causal structure. This theory is supported by empirical results on text classification.

        ----

        ## [1239] Better Safe Than Sorry: Preventing Delusive Adversaries with Adversarial Training

        **Authors**: *Lue Tao, Lei Feng, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8726bb30dc7ce15023daa8ff8402bcfd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8726bb30dc7ce15023daa8ff8402bcfd-Abstract.html)

        **Abstract**:

        Delusive attacks aim to substantially deteriorate the test accuracy of the learning model by slightly perturbing the features of correctly labeled training examples. By formalizing this malicious attack as finding the worst-case training data within a specific $\infty$-Wasserstein ball, we show that minimizing adversarial risk on the perturbed data is equivalent to optimizing an upper bound of natural risk on the original data. This implies that adversarial training can serve as a principled defense against delusive attacks. Thus, the test accuracy decreased by delusive attacks can be largely recovered by adversarial training. To further understand the internal mechanism of the defense, we disclose that adversarial training can resist the delusive perturbations by preventing the learner from overly relying on non-robust features in a natural setting. Finally, we complement our theoretical findings with a set of experiments on popular benchmark datasets, which show that the defense withstands six different practical attacks. Both theoretical and empirical results vote for adversarial training when confronted with delusive adversaries.

        ----

        ## [1240] Determinantal point processes based on orthogonal polynomials for sampling minibatches in SGD

        **Authors**: *Rémi Bardenet, Subhroshekhar Ghosh, Meixia Lin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8744cf92c88433f8cb04a02e6db69a0d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8744cf92c88433f8cb04a02e6db69a0d-Abstract.html)

        **Abstract**:

        Stochastic gradient descent (SGD) is a cornerstone of machine learning. When the number $N$ of data items is large, SGD relies on constructing an unbiased estimator of the gradient of the empirical risk using a small subset of the original dataset, called a minibatch. Default minibatch construction involves uniformly sampling a subset of the desired size, but alternatives have been explored for variance reduction. In particular, experimental evidence suggests drawing minibatches from determinantal point processes (DPPs), tractable distributions over minibatches that favour diversity among selected items. However, like in recent work on DPPs for coresets, providing a systematic and principled understanding of how and why DPPs help has been difficult. In this work, we contribute an orthogonal polynomial-based determinantal point process paradigm for performing minibatch sampling in SGD. Our approach leverages the specific data distribution at hand, which endows it with greater sensitivity and power over existing data-agnostic methods. We substantiate our method via a detailed theoretical analysis of its convergence properties, interweaving between the discrete data set and the underlying continuous domain. In particular, we show how specific DPPs and a string of controlled approximations can lead to gradient estimators with a variance that decays faster with the batchsize than under uniform sampling. Coupled with existing finite-time guarantees for SGD on convex objectives, this entails that, for a large enough batchsize and a fixed budget of item-level gradients to evaluate, DPP minibatches lead to a smaller bound on the mean square approximation error than uniform minibatches. Moreover, our estimators are amenable to a recent algorithm that directly samples linear statistics of DPPs (i.e., the gradient estimator) without sampling the underlying DPP (i.e., the minibatch), thereby reducing computational overhead. We provide detailed synthetic as well as real data experiments to substantiate our theoretical claims.

        ----

        ## [1241] Revisiting Contrastive Methods for Unsupervised Learning of Visual Representations

        **Authors**: *Wouter Van Gansbeke, Simon Vandenhende, Stamatios Georgoulis, Luc Van Gool*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8757150decbd89b0f5442ca3db4d0e0e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8757150decbd89b0f5442ca3db4d0e0e-Abstract.html)

        **Abstract**:

        Contrastive self-supervised learning has outperformed supervised pretraining on many downstream tasks like segmentation and object detection. However, current methods are still primarily applied to curated datasets like ImageNet. In this paper, we first study how biases in the dataset affect existing methods. Our results show that an approach like MoCo works surprisingly well across: (i) object- versus scene-centric, (ii) uniform versus long-tailed and (iii) general versus domain-specific datasets. Second, given the generality of the approach, we try to realize further gains with minor modifications. We show that learning additional invariances - through the use of multi-scale cropping, stronger augmentations and nearest neighbors - improves the representations. Finally, we observe that MoCo learns spatially structured representations when trained with a multi-crop strategy. The representations can be used for semantic segment retrieval and video instance segmentation without finetuning. Moreover, the results are on par with specialized models. We hope this work will serve as a useful study for other researchers.

        ----

        ## [1242] Neural Analysis and Synthesis: Reconstructing Speech from Self-Supervised Representations

        **Authors**: *Hyeong-Seok Choi, Juheon Lee, Wansoo Kim, Jie Lee, Hoon Heo, Kyogu Lee*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html)

        **Abstract**:

        We present a neural analysis and synthesis (NANSY) framework that can manipulate the voice, pitch, and speed of an arbitrary speech signal.  Most of the previous works have focused on using information bottleneck to disentangle analysis features for controllable synthesis, which usually results in poor reconstruction quality. We address this issue by proposing a novel training strategy based on information perturbation. The idea is to perturb information in the original input signal (e.g., formant, pitch, and frequency response), thereby letting synthesis networks selectively take essential attributes to reconstruct the input signal. Because NANSY does not need any bottleneck structures, it enjoys both high reconstruction quality and controllability. Furthermore, NANSY does not require any labels associated with speech data such as text and speaker information, but rather uses a new set of analysis features, i.e., wav2vec feature and newly proposed pitch feature, Yingram, which allows for fully self-supervised training. Taking advantage of fully self-supervised training, NANSY can be easily extended to a multilingual setting by simply training it with a multilingual dataset. The experiments show that NANSY can achieve significant improvement in performance in several applications such as zero-shot voice conversion, pitch shift, and time-scale modification.

        ----

        ## [1243] Auto-Encoding Knowledge Graph for Unsupervised Medical Report Generation

        **Authors**: *Fenglin Liu, Chenyu You, Xian Wu, Shen Ge, Sheng Wang, Xu Sun*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/876e1c59023b1a0e95808168e1a8ff89-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/876e1c59023b1a0e95808168e1a8ff89-Abstract.html)

        **Abstract**:

        Medical report generation, which aims to automatically generate a long and coherent report of a given medical image, has been receiving growing research interests. Existing approaches mainly adopt a supervised manner and heavily rely on coupled image-report pairs. However, in the medical domain, building a large-scale image-report paired dataset is both time-consuming and expensive. To relax the dependency on paired data, we propose an unsupervised model Knowledge Graph Auto-Encoder (KGAE) which accepts independent sets of images and reports in training. KGAE consists of a pre-constructed knowledge graph, a knowledge-driven encoder and a knowledge-driven decoder. The knowledge graph works as the shared latent space to bridge the visual and textual domains; The knowledge-driven encoder projects medical images and reports to the corresponding coordinates in this latent space and the knowledge-driven decoder generates a medical report given a coordinate in this space. Since the knowledge-driven encoder and decoder can be trained with independent sets of images and reports, KGAE is unsupervised. The experiments show that the unsupervised KGAE generates desirable medical reports without using any image-report training pairs. Moreover, KGAE can also work in both semi-supervised and supervised settings, and accept paired images and reports in training. By further fine-tuning with image-report pairs, KGAE consistently outperforms the current state-of-the-art models on two datasets.

        ----

        ## [1244] Diffusion Normalizing Flow

        **Authors**: *Qinsheng Zhang, Yongxin Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/876f1f9954de0aa402d91bb988d12cd4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/876f1f9954de0aa402d91bb988d12cd4-Abstract.html)

        **Abstract**:

        We present a novel generative modeling method called diffusion normalizing flow based on stochastic differential equations (SDEs). The algorithm consists of two neural SDEs: a forward SDE that gradually adds noise to the data to transform the data into Gaussian random noise, and a backward SDE that gradually removes the noise to sample from the data distribution. By jointly training the two neural SDEs to minimize a common cost function that quantifies the difference between the two, the backward SDE converges to a diffusion process the starts with a Gaussian distribution and ends with the desired data distribution. Our method is closely related to normalizing flow and diffusion probabilistic models, and can be viewed as a combination of the two. Compared with normalizing flow, diffusion normalizing flow is able to learn distributions with sharp boundaries. Compared with diffusion probabilistic models, diffusion normalizing flow requires fewer discretization steps and thus has better sampling efficiency. Our algorithm demonstrates competitive performance in both high-dimension data density estimation and image generation tasks.

        ----

        ## [1245] Introspective Distillation for Robust Question Answering

        **Authors**: *Yulei Niu, Hanwang Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/878d5691c824ee2aaf770f7d36c151d6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/878d5691c824ee2aaf770f7d36c151d6-Abstract.html)

        **Abstract**:

        Question answering (QA) models are well-known to exploit data bias, e.g., the language prior in visual QA and the position bias in reading comprehension. Recent debiasing methods achieve good out-of-distribution (OOD) generalizability with a considerable sacrifice of the in-distribution (ID) performance. Therefore, they are only applicable in domains where the test distribution is known in advance. In this paper, we present a novel debiasing method called Introspective Distillation (IntroD) to make the best of both worlds for QA. Our key technical contribution is to blend the inductive bias of OOD and ID by introspecting whether a training sample fits in the factual ID world or the counterfactual OOD one. Experiments on visual QA datasets VQA v2, VQA-CP, and reading comprehension dataset SQuAD demonstrate that our proposed IntroD maintains the competitive OOD performance compared to other debiasing methods, while sacrificing little or even achieving better ID performance compared to the non-debiasing ones.

        ----

        ## [1246] Rethinking the Pruning Criteria for Convolutional Neural Network

        **Authors**: *Zhongzhan Huang, Wenqi Shao, Xinjiang Wang, Liang Lin, Ping Luo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/87ae6fb631f7c8a627e8e28785d9992d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/87ae6fb631f7c8a627e8e28785d9992d-Abstract.html)

        **Abstract**:

        Channel pruning is a popular technique for compressing convolutional neural networks (CNNs), where various pruning criteria have been proposed to remove the redundant filters. From our comprehensive experiments, we found two blind spots of pruning criteria: (1) Similarity: There are some strong similarities among several primary pruning criteria that are widely cited and compared. According to these criteria, the ranks of filtersâ€™ Importance Score are almost identical, resulting in similar pruned structures. (2) Applicability: The filters' Importance Score measured by some pruning criteria are too close to distinguish the network redundancy well. In this paper, we analyze the above blind spots on different types of pruning criteria with layer-wise pruning or global pruning. We also break some stereotypes, such as that the results of $\ell_1$ and $\ell_2$ pruning are not always similar. These analyses are based on the empirical experiments and our assumption (Convolutional Weight Distribution Assumption) that the well-trained convolutional filters in each layer approximately follow a Gaussian-alike distribution. This assumption has been verified through systematic and extensive statistical tests.

        ----

        ## [1247] Adaptive Machine Unlearning

        **Authors**: *Varun Gupta, Christopher Jung, Seth Neel, Aaron Roth, Saeed Sharifi-Malvajerdi, Chris Waites*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/87f7ee4fdb57bdfd52179947211b7ebb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/87f7ee4fdb57bdfd52179947211b7ebb-Abstract.html)

        **Abstract**:

        Data deletion algorithms aim to remove the influence of deleted data points from trained models at a cheaper computational cost than fully retraining those models. However, for sequences of deletions, most prior work in the non-convex setting gives valid guarantees only for sequences that are chosen independently of the models that are published. If people choose to delete their data as a function of the published models (because they donâ€™t like what the models reveal about them, for example), then the update sequence is adaptive. In this paper, we give a general reduction from deletion guarantees against adaptive sequences to deletion guarantees against non-adaptive sequences, using differential privacy and its connection to max information. Combined with ideas from prior work which give guarantees for non-adaptive deletion sequences, this leads to extremely flexible algorithms able to handle arbitrary model classes and training methodologies, giving strong provable deletion guarantees for adaptive deletion sequences. We show in theory how prior work for non-convex models fails against adaptive deletion sequences, and use this intuition to design a practical attack against the SISA algorithm of Bourtoule et al. [2021] on CIFAR-10, MNIST, Fashion-MNIST.

        ----

        ## [1248] EditGAN: High-Precision Semantic Image Editing

        **Authors**: *Huan Ling, Karsten Kreis, Daiqing Li, Seung Wook Kim, Antonio Torralba, Sanja Fidler*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/880610aa9f9de9ea7c545169c716f477-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/880610aa9f9de9ea7c545169c716f477-Abstract.html)

        **Abstract**:

        Generative adversarial networks (GANs) have recently found applications in image editing. However, most GAN-based image editing methods often require large-scale datasets with semantic segmentation annotations for training, only provide high-level control, or merely interpolate between different images. Here, we propose EditGAN, a novel method for high-quality, high-precision semantic image editing, allowing users to edit images by modifying their highly detailed part segmentation masks, e.g., drawing a new mask for the headlight of a car. EditGAN builds on a GAN framework that jointly models images and their semantic segmentation, requiring only a handful of labeled examples – making it a scalable tool for editing. Specifically, we embed an image into the GAN’s latent space and perform conditional latent code optimization according to the segmentation edit, which effectively also modifies the image. To amortize optimization, we find “editing vectors” in latent space that realize the edits. The framework allows us to learn an arbitrary number of editing vectors, which can then be directly applied on other images at interactive rates. We experimentally show that EditGAN can manipulate images with an unprecedented level of detail and freedom while preserving full image quality. We can also easily combine multiple edits and perform plausible edits beyond EditGAN’s training data. We demonstrate EditGAN on a wide variety of image types and quantitatively outperform several previous editing methods on standard editing benchmark tasks.

        ----

        ## [1249] Deep Molecular Representation Learning via Fusing Physical and Chemical Information

        **Authors**: *Shuwen Yang, Ziyao Li, Guojie Song, Lingsheng Cai*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/884d247c6f65a96a7da4d1105d584ddd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/884d247c6f65a96a7da4d1105d584ddd-Abstract.html)

        **Abstract**:

        Molecular representation learning is the first yet vital step in combining deep learning and molecular science. To push the boundaries of molecular representation learning, we present PhysChem, a novel neural architecture that learns molecular representations via fusing physical and chemical information of molecules. PhysChem is composed of a physicist network (PhysNet) and a chemist network (ChemNet). PhysNet is a neural physical engine that learns molecular conformations through simulating molecular dynamics with parameterized forces; ChemNet implements geometry-aware deep message-passing to learn chemical / biomedical properties of molecules. Two networks specialize in their own tasks and cooperate by providing expertise to each other. By fusing physical and chemical information, PhysChem achieved state-of-the-art performances on MoleculeNet, a standard molecular machine learning benchmark. The effectiveness of PhysChem was further corroborated on cutting-edge datasets of SARS-CoV-2.

        ----

        ## [1250] Neural optimal feedback control with local learning rules

        **Authors**: *Johannes Friedrich, Siavash Golkar, Shiva Farashahi, Alexander Genkin, Anirvan M. Sengupta, Dmitri B. Chklovskii*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/88591b4d3219675bdeb33584b755f680-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/88591b4d3219675bdeb33584b755f680-Abstract.html)

        **Abstract**:

        A major problem in motor control is understanding how the brain plans and executes proper movements in the face of delayed and noisy stimuli. A prominent framework for addressing such control problems is Optimal Feedback Control (OFC). OFC generates control actions that optimize behaviorally relevant criteria by integrating noisy sensory stimuli and the predictions of an internal model using the Kalman filter or its extensions. However, a satisfactory neural model of Kalman filtering and control is lacking because existing proposals have the following  limitations: not considering the delay of sensory feedback, training in alternating phases, requiring knowledge of the noise covariance matrices, as well as that of systems dynamics. Moreover, the majority of these studies considered Kalman filtering in isolation, and not jointly with control. To address these shortcomings, we introduce a novel online algorithm which combines adaptive Kalman filtering with a model free control approach  (i.e., policy gradient algorithm). We implement this algorithm in a biologically plausible neural network with local synaptic plasticity rules. This network, with local synaptic plasticity rules, performs system identification, Kalman filtering and control with delayed noisy sensory feedback. This network performs system identification and Kalman filtering, without the need for multiple phases with distinct update rules or the knowledge of the noise covariances. It can perform state estimation  with delayed sensory feedback, with the help of an internal model. It learns the control policy without requiring any knowledge of the dynamics, thus avoiding the need for weight transport. In this way, our implementation of OFC solves the credit assignment problem needed to produce the appropriate sensory-motor control in the presence of stimulus delay.

        ----

        ## [1251] Reinforcement Learning in Linear MDPs: Constant Regret and Representation Selection

        **Authors**: *Matteo Papini, Andrea Tirinzoni, Aldo Pacchiano, Marcello Restelli, Alessandro Lazaric, Matteo Pirotta*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8860e834a67da41edd6ffe8a1c58fa55-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8860e834a67da41edd6ffe8a1c58fa55-Abstract.html)

        **Abstract**:

        We study the role of the representation of state-action value functions in regret minimization in finite-horizon Markov Decision Processes (MDPs) with linear structure. We first derive a necessary condition on the representation, called universally spanning optimal features (UNISOFT), to achieve constant regret in any MDP with linear reward function. This result encompasses the well-known settings of low-rank MDPs and, more generally, zero inherent Bellman error (also known as the Bellman closure assumption). We then demonstrate that this condition is also sufficient for these classes of problems by deriving a constant regret bound for two optimistic algorithms (LSVI-UCB and ELEANOR). Finally, we propose an algorithm for representation selection and we prove that it achieves constant regret when one of the given representations, or a suitable combination of them, satisfies the UNISOFT condition.

        ----

        ## [1252] Noether Networks: meta-learning useful conserved quantities

        **Authors**: *Ferran Alet, Dylan Doblar, Allan Zhou, Josh Tenenbaum, Kenji Kawaguchi, Chelsea Finn*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/886ad506e0c115cf590d18ebb6c26561-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/886ad506e0c115cf590d18ebb6c26561-Abstract.html)

        **Abstract**:

        Progress in machine learning (ML) stems from a combination of data availability, computational resources, and an appropriate encoding of inductive biases. Useful biases often exploit symmetries in the prediction problem, such as convolutional networks relying on translation equivariance. Automatically discovering these useful symmetries holds the potential to greatly improve the performance of ML systems, but still remains a challenge. In this work, we focus on sequential prediction problems and take inspiration from Noether's theorem to reduce the problem of finding inductive biases to meta-learning useful conserved quantities. We propose Noether Networks: a new type of architecture where a meta-learned conservation loss is optimized inside the prediction function. We show, theoretically and experimentally, that Noether Networks improve prediction quality, providing a general framework for discovering inductive biases in sequential problems.

        ----

        ## [1253] Uncertainty-Driven Loss for Single Image Super-Resolution

        **Authors**: *Qian Ning, Weisheng Dong, Xin Li, Jinjian Wu, Guangming Shi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/88a199611ac2b85bd3f76e8ee7e55650-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/88a199611ac2b85bd3f76e8ee7e55650-Abstract.html)

        **Abstract**:

        In low-level vision such as single image super-resolution (SISR), traditional MSE or L1 loss function treats every pixel equally with the assumption that the importance of all pixels is the same. However, it has been long recognized that texture and edge areas carry more important visual information than smooth areas in photographic images. How to achieve such spatial adaptation in a principled manner has been an open problem in both traditional model-based and modern learning-based approaches toward SISR. In this paper, we propose a new adaptive weighted loss for SISR to train deep networks focusing on challenging situations such as textured and edge pixels with high uncertainty. Specifically, we introduce variance estimation characterizing the uncertainty on a pixel-by-pixel basis into SISR solutions so the targeted pixels in a high-resolution image (mean) and their corresponding uncertainty (variance) can be learned simultaneously. Moreover, uncertainty estimation allows us to leverage conventional wisdom such as sparsity prior for regularizing SISR solutions. Ultimately, pixels with large certainty (e.g., texture and edge pixels) will be prioritized for SISR according to their importance to visual quality. For the first time, we demonstrate that such uncertainty-driven loss can achieve better results than MSE or L1 loss for a wide range of network architectures. Experimental results on three popular SISR networks show that our proposed uncertainty-driven loss has achieved better PSNR performance than traditional loss functions without any increased computation during testing. The code is available at https://see.xidian.edu.cn/faculty/wsdong/Projects/UDL-SR.htm

        ----

        ## [1254] GradInit: Learning to Initialize Neural Networks for Stable and Efficient Training

        **Authors**: *Chen Zhu, Renkun Ni, Zheng Xu, Kezhi Kong, W. Ronny Huang, Tom Goldstein*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/88ae6372cfdc5df69a976e893f4d554b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/88ae6372cfdc5df69a976e893f4d554b-Abstract.html)

        **Abstract**:

        Innovations in neural architectures have fostered significant breakthroughs in language modeling and computer vision. Unfortunately, novel architectures often result in challenging hyper-parameter choices and training instability if the network parameters are not properly initialized. A number of architecture-specific initialization schemes have been proposed, but these schemes are not always portable to new architectures. This paper presents GradInit, an automated and architecture agnostic method for initializing neural networks. GradInit is based on a simple heuristic; the norm of each network layer is adjusted so that a single step of SGD or Adam with prescribed hyperparameters results in the smallest possible loss value. This adjustment is done by introducing a scalar multiplier variable in front of each parameter block, and then optimizing these variables using a simple numerical scheme. GradInit accelerates the convergence and test performance of many convolutional architectures, both with or without skip connections, and even without normalization layers. It also improves the stability of the original Transformer architecture for machine translation, enabling training it without learning rate warmup using either Adam or SGD under a wide range of learning rates and momentum coefficients. Code is available at https://github.com/zhuchen03/gradinit.

        ----

        ## [1255] Capacity and Bias of Learned Geometric Embeddings for Directed Graphs

        **Authors**: *Michael Boratko, Dongxu Zhang, Nicholas Monath, Luke Vilnis, Kenneth L. Clarkson, Andrew McCallum*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/88d25099b103efd638163ecb40a55589-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/88d25099b103efd638163ecb40a55589-Abstract.html)

        **Abstract**:

        A wide variety of machine learning tasks such as knowledge base completion, ontology alignment, and multi-label classification can benefit from incorporating into learning differentiable representations of graphs or taxonomies.  While vectors in Euclidean space can theoretically represent any graph, much recent work shows that alternatives such as complex, hyperbolic, order, or box embeddings have geometric properties better suited to modeling real-world graphs. Experimentally these gains are seen only in lower dimensions, however, with performance benefits diminishing in higher dimensions. In this work, we introduce a novel variant of box embeddings that uses a learned smoothing parameter to achieve better representational capacity than vector models in low dimensions, while also avoiding performance saturation common to other geometric models in high dimensions. Further, we present theoretical results that prove box embeddings can represent any DAG. We perform rigorous empirical evaluations of vector, hyperbolic, and region-based geometric representations on several families of synthetic and real-world directed graphs. Analysis of these results exposes correlations between different families of graphs, graph characteristics, model size, and embedding geometry, providing useful insights into the inductive biases of various differentiable graph representations.

        ----

        ## [1256] Online Learning Of Neural Computations From Sparse Temporal Feedback

        **Authors**: *Mikio Ludwig Braun, Tim P. Vogels*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/88e1ce84f9feef5a08d0df0334c53468-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/88e1ce84f9feef5a08d0df0334c53468-Abstract.html)

        **Abstract**:

        Neuronal computations depend on synaptic connectivity and intrinsic electrophysiological properties. Synaptic connectivity determines which inputs from presynaptic neurons are integrated, while cellular properties determine how inputs are filtered over time. Unlike their biological counterparts, most computational approaches to learning in simulated neural networks are limited to changes in synaptic connectivity. However, if intrinsic parameters change, neural computations are altered drastically. Here, we include the parameters that determine the intrinsic properties, e.g., time constants and reset potential, into the learning paradigm. Using sparse feedback signals that indicate target spike times, and gradient-based parameter updates, we show that the intrinsic parameters can be learned along with the synaptic weights to produce specific input-output functions. Specifically, we use  a teacher-student paradigm in which a randomly initialised leaky integrate-and-fire or resonate-and-fire neuron must recover the parameters of a teacher neuron. We show that complex temporal functions can be learned online and without backpropagation through time, relying on event-based updates only. Our results are a step towards online learning of neural computations from ungraded and unsigned sparse feedback signals with a biologically inspired learning mechanism.

        ----

        ## [1257] Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style

        **Authors**: *Julius von Kügelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Schölkopf, Michel Besserve, Francesco Locatello*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8929c70f8d710e412d38da624b21c3c8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8929c70f8d710e412d38da624b21c3c8-Abstract.html)

        **Abstract**:

        Self-supervised representation learning has shown remarkable success in a number of domains. A common practice is to perform data augmentation via hand-crafted transformations intended to leave the semantics of the data invariant. We seek to understand the empirical success of this approach from a theoretical perspective. We formulate the augmentation process as a latent variable model by postulating a partition of the latent representation into a content component, which is assumed invariant to augmentation, and a style component, which is allowed to change. Unlike prior work on disentanglement and independent component analysis, we allow for both nontrivial statistical and causal dependencies in the latent space. We study the identifiability of the latent representation based on pairs of views of the observations and prove sufficient conditions that allow us to identify the invariant content partition up to an invertible mapping in both generative and discriminative settings. We find numerical simulations with dependent latent variables are consistent with our theory. Lastly, we introduce Causal3DIdent, a dataset of high-dimensional, visually complex images with rich causal dependencies, which we use to study the effect of data augmentations performed in practice.

        ----

        ## [1258] Instance-Conditional Knowledge Distillation for Object Detection

        **Authors**: *Zijian Kang, Peizhen Zhang, Xiangyu Zhang, Jian Sun, Nanning Zheng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/892c91e0a653ba19df81a90f89d99bcd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/892c91e0a653ba19df81a90f89d99bcd-Abstract.html)

        **Abstract**:

        Knowledge distillation has shown great success in classification, however, it is still challenging for detection. In a typical image for detection, representations from different locations may have different contributions to detection targets, making the distillation hard to balance. In this paper, we propose a conditional distillation framework to distill the desired knowledge, namely knowledge that is beneficial in terms of both classification and localization for every instance. The framework introduces a learnable conditional decoding module, which retrieves information given each target instance as query. Specifically, we encode the condition information as query and use the teacher's representations as key. The attention between query and key is used to measure the contribution of different features, guided by a localization-recognition-sensitive auxiliary task. Extensive experiments demonstrate the efficacy of our method: we observe impressive improvements under various settings. Notably, we boost RetinaNet with ResNet-50 backbone from $37.4$ to $40.7$ mAP ($+3.3$) under $1\times$ schedule, that even surpasses the teacher ($40.4$ mAP) with ResNet-101 backbone under $3\times$ schedule. Code has been released on https://github.com/megvii-research/ICD.

        ----

        ## [1259] Self-Supervised Representation Learning on Neural Network Weights for Model Characteristic Prediction

        **Authors**: *Konstantin Schürholt, Dimche Kostadinov, Damian Borth*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/89562dccfeb1d0394b9ae7e09544dc70-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/89562dccfeb1d0394b9ae7e09544dc70-Abstract.html)

        **Abstract**:

        Self-Supervised Learning (SSL) has been shown to learn useful and information-preserving representations. Neural Networks (NNs) are widely applied, yet their weight space is still not fully understood. Therefore, we propose to use SSL to learn hyper-representations of the weights of populations of NNs. To that end, we introduce domain specific data augmentations and an adapted attention architecture.  Our empirical evaluation demonstrates that self-supervised representation learning in this domain is able to recover diverse NN model characteristics. Further, we show that the proposed learned representations outperform prior work for predicting hyper-parameters, test accuracy, and generalization gap as well as transfer to out-of-distribution settings.

        ----

        ## [1260] Multimodal Virtual Point 3D Detection

        **Authors**: *Tianwei Yin, Xingyi Zhou, Philipp Krähenbühl*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/895daa408f494ad58006c47a30f51c1f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/895daa408f494ad58006c47a30f51c1f-Abstract.html)

        **Abstract**:

        Lidar-based sensing drives current autonomous vehicles. Despite rapid progress, current Lidar sensors still lag two decades behind traditional color cameras in terms of resolution and cost. For autonomous driving, this means that large objects close to the sensors are easily visible, but far-away or small objects comprise only one measurement or two. This is an issue, especially when these objects turn out to be driving hazards. On the other hand, these same objects are clearly visible in onboard RGB sensors. In this work, we present an approach to seamlessly fuse RGB sensors into Lidar-based 3D recognition. Our approach takes a set of 2D detections to generate dense 3D virtual points to augment an otherwise sparse 3D point cloud. These virtual points naturally integrate into any standard Lidar-based 3D detectors along with regular Lidar measurements. The resulting multi-modal detector is simple and effective. Experimental results on the large-scale nuScenes dataset show that our framework improves a strong CenterPoint baseline by a significant $6.6$ mAP, and outperforms competing fusion approaches. Code and more visualizations are available at https://tianweiy.github.io/mvp/

        ----

        ## [1261] On Joint Learning for Solving Placement and Routing in Chip Design

        **Authors**: *Ruoyu Cheng, Junchi Yan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/898aef0932f6aaecda27aba8e9903991-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/898aef0932f6aaecda27aba8e9903991-Abstract.html)

        **Abstract**:

        For its advantage in GPU acceleration and less dependency on human experts, machine learning has been an emerging tool for solving the placement and routing problems, as two critical steps in modern chip design flow. Being still in its early stage, there are several fundamental issues unresolved: scalability, reward design, and end-to-end learning paradigm etc. To achieve end-to-end placement learning, we first propose a joint learning method for the placement of macros and standard cells, by the integration of reinforcement learning with a gradient based optimization scheme. To further bridge the placement with the subsequent routing task, we also develop a joint learning approach via reinforcement learning. One key design in our (reinforcement) learning paradigm involves a multi-view embedding model to encode both global graph level and local node level information of the input macros. Moreover, the random network distillation is devised to encourage exploration. Experiments on public chip design benchmarks show that our method can effectively learn from experience and also provide high-quality intermediate placement for the post standard cell placement, within few hours for training.

        ----

        ## [1262] Learning with Algorithmic Supervision via Continuous Relaxations

        **Authors**: *Felix Petersen, Christian Borgelt, Hilde Kuehne, Oliver Deussen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/89ae0fe22c47d374bc9350ef99e01685-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/89ae0fe22c47d374bc9350ef99e01685-Abstract.html)

        **Abstract**:

        The integration of algorithmic components into neural architectures has gained increased attention recently, as it allows training neural networks with new forms of supervision such as ordering constraints or silhouettes instead of using ground truth labels. Many approaches in the field focus on the continuous relaxation of a specific task and show promising results in this context. But the focus on single tasks also limits the applicability of the proposed concepts to a narrow range of applications. In this work, we build on those ideas to propose an approach that allows to integrate algorithms into end-to-end trainable neural network architectures based on a general approximation of discrete conditions. To this end, we relax these conditions in control structures such as conditional statements, loops, and indexing, so that resulting algorithms are smoothly differentiable. To obtain meaningful gradients, each relevant variable is perturbed via logistic distributions and the expectation value under this perturbation is approximated. We evaluate the proposed continuous relaxation model on four challenging tasks and show that it can keep up with relaxations specifically designed for each individual task.

        ----

        ## [1263] Differentiable Multiple Shooting Layers

        **Authors**: *Stefano Massaroli, Michael Poli, Sho Sonoda, Taiji Suzuki, Jinkyoo Park, Atsushi Yamashita, Hajime Asama*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/89b9c689a57b82e59074c6ba09aa394d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/89b9c689a57b82e59074c6ba09aa394d-Abstract.html)

        **Abstract**:

        We detail a novel class of implicit neural models. Leveraging time-parallel methods for differential equations, Multiple Shooting Layers  (MSLs) seek solutions of initial value problems via parallelizable root-finding algorithms. MSLs broadly serve as drop-in replacements for neural ordinary differential equations  (Neural ODEs) with improved efficiency in number of function evaluations (NFEs) and wall-clock inference time. We develop the algorithmic framework of MSLs, analyzing the different choices of solution methods from a theoretical and computational perspective. MSLs are showcased in long horizon optimal control of ODEs and PDEs and as latent models for sequence generation. Finally, we investigate the speedups obtained through application of MSL inference in neural controlled differential equations (Neural CDEs) for time series classification of medical data.

        ----

        ## [1264] Global-aware Beam Search for Neural Abstractive Summarization

        **Authors**: *Ye Ma, Zixun Lan, Lu Zong, Kaizhu Huang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/89d4402dc03d3b7318bbac10203034ab-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/89d4402dc03d3b7318bbac10203034ab-Abstract.html)

        **Abstract**:

        This study develops a calibrated beam-based algorithm with awareness of the global attention distribution for neural abstractive summarization, aiming to improve the local optimality problem of the original beam search in a rigorous way. Specifically, a novel global protocol is proposed based on the attention distribution to stipulate how a global optimal hypothesis should attend to the source. A global scoring mechanism is then developed to regulate beam search to generate summaries in a near-global optimal fashion. This novel design enjoys a distinctive property, i.e., the global attention distribution could be predicted before inference, enabling step-wise improvements on the beam search through the global scoring mechanism. Extensive experiments on nine datasets show that the global (attention)-aware inference significantly improves state-of-the-art summarization models even using empirical hyper-parameters. The algorithm is also proven robust as it remains to generate meaningful texts with corrupted attention distributions. The codes and a comprehensive set of examples are available.

        ----

        ## [1265] DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras

        **Authors**: *Zachary Teed, Jia Deng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/89fcd07f20b6785b92134bd6c1d0fa42-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/89fcd07f20b6785b92134bd6c1d0fa42-Abstract.html)

        **Abstract**:

        We introduce DROID-SLAM, a new deep learning based SLAM system. DROID-SLAM consists of recurrent iterative updates of camera pose and pixelwise depth through a Dense Bundle Adjustment layer. DROID-SLAM is accurate, achieving large improvements over prior work, and robust, suffering from substantially fewer catastrophic failures. Despite training on monocular video, it can leverage stereo or RGB-D video to achieve improved performance at test time. The URL to our open source code is https://github.com/princeton-vl/DROID-SLAM.

        ----

        ## [1266] Few-Shot Object Detection via Association and DIscrimination

        **Authors**: *Yuhang Cao, Jiaqi Wang, Ying Jin, Tong Wu, Kai Chen, Ziwei Liu, Dahua Lin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8a1e808b55fde9455cb3d8857ed88389-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8a1e808b55fde9455cb3d8857ed88389-Abstract.html)

        **Abstract**:

        Object detection has achieved substantial progress in the last decade. However, detecting novel classes with only few samples remains challenging, since deep learning under low data regime usually leads to a degraded feature space. Existing works employ a holistic fine-tuning paradigm to tackle this problem, where the model is first pre-trained on all base classes with abundant samples, and then it is used to carve the novel class feature space. Nonetheless, this paradigm is still imperfect. Durning fine-tuning, a novel class may implicitly leverage the knowledge of multiple base classes to construct its feature space, which induces a scattered feature space, hence violating the inter-class separability. To overcome these obstacles, we propose a two-step fine-tuning framework, Few-shot object detection via Association and DIscrimination (FADI), which builds up a discriminative feature space for each novel class with two integral steps. 1) In the association step, in contrast to implicitly leveraging multiple base classes, we construct a compact novel class feature space via explicitly imitating a specific base class feature space. Specifically, we associate each novel class with a base class according to their semantic similarity. After that, the feature space of a novel class can readily imitate the well-trained feature space of the associated base class. 2) In the discrimination step, to ensure the separability between the novel classes and associated base classes, we disentangle the classification branches for base and novel classes. To further enlarge the inter-class separability between all classes, a set-specialized margin loss is imposed. Extensive experiments on standard Pascal VOC and MS-COCO datasets demonstrate that FADI achieves new state-of-the-art performance, significantly improving the baseline in any shot/split by +18.7. Notably, the advantage of FADI is most announced on extremely few-shot scenarios (e.g. 1- and 3- shot).

        ----

        ## [1267] Neural Dubber: Dubbing for Videos According to Scripts

        **Authors**: *Chenxu Hu, Qiao Tian, Tingle Li, Yuping Wang, Yuxuan Wang, Hang Zhao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8a9c8ac001d3ef9e4ce39b1177295e03-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8a9c8ac001d3ef9e4ce39b1177295e03-Abstract.html)

        **Abstract**:

        Dubbing is a post-production process of re-recording actors’ dialogues, which is extensively used in filmmaking and video production. It is usually performed manually by professional voice actors who read lines with proper prosody, and in synchronization with the pre-recorded videos. In this work, we propose Neural Dubber, the first neural network model to solve a novel automatic video dubbing (AVD) task: synthesizing human speech synchronized with the given video from the text. Neural Dubber is a multi-modal text-to-speech (TTS) model that utilizes the lip movement in the video to control the prosody of the generated speech. Furthermore, an image-based speaker embedding (ISE) module is developed for the multi-speaker setting, which enables Neural Dubber to generate speech with a reasonable timbre according to the speaker’s face. Experiments on the chemistry lecture single-speaker dataset and LRS2 multi-speaker dataset show that Neural Dubber can generate speech audios on par with state-of-the-art TTS models in terms of speech quality. Most importantly, both qualitative and quantitative evaluations show that Neural Dubber can control the prosody of synthesized speech by the video, and generate high-fidelity speech temporally synchronized with the video.

        ----

        ## [1268] Neural Bootstrapper

        **Authors**: *Minsuk Shin, Hyungjoo Cho, Hyun-seok Min, Sungbin Lim*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8abfe8ac9ec214d68541fcb888c0b4c3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8abfe8ac9ec214d68541fcb888c0b4c3-Abstract.html)

        **Abstract**:

        Bootstrapping has been a primary tool for ensemble and uncertainty quantification in machine learning and statistics. However, due to its nature of multiple training and resampling, bootstrapping deep neural networks is computationally burdensome; hence it has difficulties in practical application to the uncertainty estimation and related tasks. To overcome this computational bottleneck, we propose a novel approach called Neural Bootstrapper (NeuBoots), which learns to generate bootstrapped neural networks through single model training. NeuBoots injects the bootstrap weights into the high-level feature layers of the backbone network and outputs the bootstrapped predictions of the target, without additional parameters and the repetitive computations from scratch. We apply NeuBoots to various machine learning tasks related to uncertainty quantification, including prediction calibrations in image classification and semantic segmentation, active learning, and detection of out-of-distribution samples. Our empirical results show that NeuBoots outperforms other bagging based methods under a much lower computational cost without losing the validity of bootstrapping.

        ----

        ## [1269] An Axiomatic Theory of Provably-Fair Welfare-Centric Machine Learning

        **Authors**: *Cyrus Cousins*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8b0bb3eff8c1e5bf7f206125959921d7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8b0bb3eff8c1e5bf7f206125959921d7-Abstract.html)

        **Abstract**:

        We address an inherent difficulty in welfare-theoretic fair machine learning (ML), by proposing an equivalently-axiomatically justified alternative setting, and studying the resulting computational and statistical learning questions. Welfare metrics quantify overall wellbeing across a population of groups, and welfare-based objectives and constraints have recently been proposed to incentivize fair ML methods to satisfy their diverse needs. However, many ML problems are cast as loss minimization tasks, rather than utility maximization, and thus require nontrivial modeling to construct utility functions. We define a complementary metric, termed malfare, measuring overall societal harm, with axiomatic justification via the standard axioms of cardinal welfare, and cast fair ML as malfare minimization over the risk values (expected losses) of each group. Surprisingly, the axioms of cardinal welfare (malfare) dictate that this is not equivalent to simply defining utility as negative loss and maximizing welfare. Building upon these concepts, we define fair-PAC learning, where a fair-PAC learner is an algorithm that learns an ε-δ malfare-optimal model with bounded sample complexity, for any data distribution and (axiomatically justified) malfare concept. Finally, we show conditions under which many standard PAC-learners may be converted to fair-PAC learners, which places fair-PAC learning on firm theoretical ground, as it yields statistical — and in some cases computational — efficiency guarantees for many well-studied ML models. Fair-PAC learning is also practically relevant, as it democratizes fair ML by providing concrete training algorithms with rigorous generalization guarantees.

        ----

        ## [1270] HSVA: Hierarchical Semantic-Visual Adaptation for Zero-Shot Learning

        **Authors**: *Shiming Chen, Guo-Sen Xie, Yang Liu, Qinmu Peng, Baigui Sun, Hao Li, Xinge You, Ling Shao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8b0d268963dd0cfb808aac48a549829f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8b0d268963dd0cfb808aac48a549829f-Abstract.html)

        **Abstract**:

        Zero-shot learning (ZSL) tackles the unseen class recognition problem,  transferring semantic knowledge from seen classes to unseen ones. Typically, to guarantee desirable knowledge transfer, a common (latent) space is adopted for associating the visual and semantic domains in ZSL.  However, existing common space learning methods align the semantic and visual domains by merely mitigating distribution disagreement through one-step adaptation. This strategy is usually ineffective due to the heterogeneous nature of the feature representations in the two domains, which intrinsically contain both distribution and structure variations. To address this and advance ZSL, we propose a novel hierarchical semantic-visual adaptation (HSVA) framework. Specifically, HSVA aligns the semantic and visual domains by adopting a hierarchical two-step adaptation, i.e., structure adaptation and distribution adaptation. In the structure adaptation step, we take two task-specific encoders to encode the source data (visual domain) and the target data (semantic domain) into a structure-aligned common space. To this end, a  supervised adversarial discrepancy (SAD)  module is proposed to adversarially minimize the discrepancy between the predictions of two task-specific classifiers, thus making the visual and semantic feature manifolds more closely aligned. In the distribution adaptation step, we directly minimize the Wasserstein distance between the latent multivariate Gaussian distributions to align the visual and semantic distributions using a common encoder. Finally, the structure and distribution adaptation are derived in a unified framework under two partially-aligned variational autoencoders. Extensive experiments on four benchmark datasets demonstrate that HSVA achieves superior performance on both conventional and generalized ZSL. The code is available at \url{https://github.com/shiming-chen/HSVA}.

        ----

        ## [1271] Higher Order Kernel Mean Embeddings to Capture Filtrations of Stochastic Processes

        **Authors**: *Cristopher Salvi, Maud Lemercier, Chong Liu, Blanka Horvath, Theodoros Damoulas, Terry J. Lyons*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8b2dfbe0c1d43f9537dae01e96458ff1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8b2dfbe0c1d43f9537dae01e96458ff1-Abstract.html)

        **Abstract**:

        Stochastic processes are random variables with values in some space of paths. However, reducing a stochastic process to a path-valued random variable ignores its filtration, i.e. the flow of information carried by the process through time. By conditioning the process on its filtration, we introduce a family of higher order kernel mean embeddings (KMEs) that generalizes the notion of KME to capture additional information related to the filtration. We derive empirical estimators for the associated higher order maximum mean discrepancies (MMDs) and prove consistency. We then construct a filtration-sensitive kernel two-sample test able to capture information that gets missed by the standard MMD test. In addition, leveraging our higher order MMDs we construct a family of universal kernels on stochastic processes that allows to solve real-world calibration and optimal stopping problems in quantitative finance (such as the pricing of American options) via classical kernel-based regression methods. Finally, adapting existing tests for conditional independence to the case of stochastic processes, we design a causal-discovery algorithm to recover the causal graph of structural dependencies among interacting bodies solely from observations of their multidimensional trajectories.

        ----

        ## [1272] Low-Rank Subspaces in GANs

        **Authors**: *Jiapeng Zhu, Ruili Feng, Yujun Shen, Deli Zhao, Zheng-Jun Zha, Jingren Zhou, Qifeng Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8b4066554730ddfaa0266346bdc1b202-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8b4066554730ddfaa0266346bdc1b202-Abstract.html)

        **Abstract**:

        The latent space of a Generative Adversarial Network (GAN) has been shown to encode rich semantics within some subspaces. To identify these subspaces, researchers typically analyze the statistical information from a collection of synthesized data, and the identified subspaces tend to control image attributes globally (i.e., manipulating an attribute causes the change of an entire image). By contrast, this work introduces low-rank subspaces that enable more precise control of GAN generation. Concretely, given an arbitrary image and a region of interest (e.g., eyes of face images), we manage to relate the latent space to the image region with the Jacobian matrix and then use low-rank factorization to discover steerable latent subspaces. There are three distinguishable strengths of our approach that can be aptly called LowRankGAN. First, compared to analytic algorithms in prior work, our low-rank factorization of Jacobians is able to find the low-dimensional representation of attribute manifold, making image editing more precise and controllable.  Second, low-rank factorization naturally yields a null space of attributes such that moving the latent code within it only affects the outer region of interest. Therefore, local image editing can be simply achieved by projecting an attribute vector into the null space without relying on a spatial mask as existing methods do. Third, our method can robustly work with a local region from one image for analysis yet well generalize to other images, making it much easy to use in practice. Extensive experiments on state-of-the-art GAN models (including StyleGAN2 and BigGAN) trained on various datasets demonstrate the effectiveness of our LowRankGAN.

        ----

        ## [1273] Neural Symplectic Form: Learning Hamiltonian Equations on General Coordinate Systems

        **Authors**: *Yuhan Chen, Takashi Matsubara, Takaharu Yaguchi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8b519f198dd26772e3e82874826b04aa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8b519f198dd26772e3e82874826b04aa-Abstract.html)

        **Abstract**:

        In recent years, substantial research on the methods for learning Hamiltonian equations has been conducted. Although these approaches are very promising, the commonly used representation of the Hamilton equation uses the generalized momenta, which are generally unknown. Therefore, the training data must be represented in this unknown coordinate system, and this causes difficulty in applying the model to real data. Meanwhile, Hamiltonian equations also have a coordinate-free expression that is expressed by using the symplectic 2-form. In this study, we propose a model that learns the symplectic form from data using neural networks, thereby providing a method for learning Hamiltonian equations from data represented in general coordinate systems, which are not limited to the generalized coordinates and the generalized momenta. Consequently, the proposed method is capable not only of modeling target equations of both Hamiltonian and Lagrangian formalisms but also of extracting unknown Hamiltonian structures hidden in the data. For example, many polynomial ordinary differential equations such as the Lotka-Volterra equation are known to admit non-trivial Hamiltonian structures, and our numerical experiments show that such structures can be certainly learned from data. Technically, each symplectic 2-form is associated with a skew-symmetric matrix, but not all skew-symmetric matrices define the symplectic 2-form. In the proposed method, using the fact that symplectic 2-forms are derived as the exterior derivative of certain differential 1-forms, we model the differential 1-form by neural networks, thereby improving the efficiency of learning.

        ----

        ## [1274] Sample-Efficient Reinforcement Learning Is Feasible for Linearly Realizable MDPs with Limited Revisiting

        **Authors**: *Gen Li, Yuxin Chen, Yuejie Chi, Yuantao Gu, Yuting Wei*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8b5700012be65c9da25f49408d959ca0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8b5700012be65c9da25f49408d959ca0-Abstract.html)

        **Abstract**:

        Low-complexity models such as linear function representation play a pivotal role in enabling sample-efficient reinforcement learning (RL). The current paper pertains to a scenario with value-based linear representation, which postulates linear realizability of the optimal Q-function (also called the ``linear $Q^{\star}$ problem''). While linear realizability alone does not allow for sample-efficient solutions in general, the presence of a large sub-optimality gap is a potential game changer, depending on the sampling mechanism in use.  Informally, sample efficiency is achievable with a large sub-optimality gap when a generative model is available, but is unfortunately infeasible when we turn to standard online RL settings.  We make progress towards understanding this linear $Q^{\star}$ problem by investigating a new sampling protocol, which draws samples in an online/exploratory fashion but allows one to backtrack and revisit previous states. This protocol is more flexible than the standard online RL setting, while being practically relevant and far more restrictive than the generative model. We develop an algorithm tailored to this setting, achieving a sample complexity that scales polynomially with the feature dimension, the horizon, and the inverse sub-optimality gap, but not the size of the state/action space. Our findings underscore the fundamental interplay between sampling protocols and low-complexity function representation in RL.

        ----

        ## [1275] Self-Paced Contrastive Learning for Semi-supervised Medical Image Segmentation with Meta-labels

        **Authors**: *Jizong Peng, Ping Wang, Christian Desrosiers, Marco Pedersoli*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8b5c8441a8ff8e151b191c53c1842a38-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8b5c8441a8ff8e151b191c53c1842a38-Abstract.html)

        **Abstract**:

        The contrastive pre-training of a recognition model on a large dataset of unlabeled data often boosts the modelâ€™s performance on downstream tasks like image classification. However, in domains such as medical imaging, collecting unlabeled data can be challenging and expensive. In this work, we consider the task of medical image segmentation and adapt contrastive learning with meta-label annotations to scenarios where no additional unlabeled data is available. Meta-labels, such as the location of a 2D slice in a 3D MRI scan, often come for free during the acquisition process. We use these meta-labels to pre-train the image encoder, as well as in a semi-supervised learning step that leverages a reduced set of annotated data. A self-paced learning strategy exploiting the weak annotations is proposed to furtherhelp the learning process and discriminate useful labels from noise. Results on five medical image segmentation datasets show that our approach: i) highly boosts the performance of a model trained on a few scans, ii) outperforms previous contrastive and semi-supervised approaches, and iii) reaches close to the performance of a model trained on the full data.

        ----

        ## [1276] Reverse engineering recurrent neural networks with Jacobian switching linear dynamical systems

        **Authors**: *Jimmy T. H. Smith, Scott W. Linderman, David Sussillo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8b77b4b5156dc11dec152c6c71481565-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8b77b4b5156dc11dec152c6c71481565-Abstract.html)

        **Abstract**:

        Recurrent neural networks (RNNs) are powerful models for processing time-series data, but it remains challenging to understand how they function. Improving this understanding is of substantial interest to both the machine learning and neuroscience communities. The framework of reverse engineering a trained RNN by linearizing around its fixed points has provided insight, but the approach has significant challenges. These include difficulty choosing which fixed point to expand around when studying RNN dynamics and error accumulation when reconstructing the nonlinear dynamics with the linearized dynamics. We present a new model that overcomes these limitations by co-training an RNN with a novel switching linear dynamical system (SLDS) formulation. A first-order Taylor series expansion of the co-trained RNN and an auxiliary function trained to pick out the RNN's fixed points govern the SLDS dynamics. The results are a trained SLDS variant that closely approximates the RNN, an auxiliary function that can produce a fixed point for each point in state-space, and a trained nonlinear RNN whose dynamics have been regularized such that its first-order terms perform the computation, if possible. This model removes the post-training fixed point optimization and allows us to unambiguously study the learned dynamics of the SLDS at any point in state-space.  It also generalizes SLDS models to continuous manifolds of switching points while sharing parameters across switches. We validate the utility of the model on two synthetic tasks relevant to previous work reverse engineering RNNs. We then show that our model can be used as a drop-in in more complex architectures, such as LFADS, and apply this LFADS hybrid to analyze single-trial spiking activity from the motor system of a non-human primate.

        ----

        ## [1277] Learning-Augmented Dynamic Power Management with Multiple States via New Ski Rental Bounds

        **Authors**: *Antonios Antoniadis, Christian Coester, Marek Eliás, Adam Polak, Bertrand Simon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8b8388180314a337c9aa3c5aa8e2f37a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8b8388180314a337c9aa3c5aa8e2f37a-Abstract.html)

        **Abstract**:

        We study the online problem of minimizing power consumption in systems with multiple power-saving states. During idle periods of unknown lengths, an algorithm has to choose between power-saving states of different energy consumption and wake-up costs. We develop a learning-augmented online algorithm that makes decisions based on (potentially inaccurate) predicted lengths of the idle periods. The algorithm's performance is near-optimal when predictions are accurate and degrades gracefully with increasing prediction error, with a worst-case guarantee almost identical to the optimal classical online algorithm for the problem. A key ingredient in our approach is a new algorithm for the online ski-rental problem in the learning augmented setting with tight dependence on the prediction error. We support our theoretical findings with experiments.

        ----

        ## [1278] Learning Equivariant Energy Based Models with Equivariant Stein Variational Gradient Descent

        **Authors**: *Priyank Jaini, Lars Holdijk, Max Welling*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8b9e7ab295e87570551db122a04c6f7c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8b9e7ab295e87570551db122a04c6f7c-Abstract.html)

        **Abstract**:

        We focus on the problem of efficient sampling and learning of probability densities by incorporating symmetries in probabilistic models. We first introduce Equivariant Stein Variational Gradient Descent algorithm -- an equivariant sampling method based on Stein's identity for sampling from densities with symmetries. Equivariant SVGD explicitly incorporates symmetry information in a density through equivariant kernels which makes the resultant sampler efficient both in terms of sample complexity and the quality of generated samples. Subsequently, we define equivariant energy based models to model invariant densities that are learned using contrastive divergence. By utilizing our equivariant SVGD for training equivariant EBMs, we propose new ways of improving and scaling up training of energy based models. We apply these equivariant energy models for modelling joint densities in regression and classification tasks for image datasets, many-body particle systems and molecular structure generation.

        ----

        ## [1279] Information Directed Sampling for Sparse Linear Bandits

        **Authors**: *Botao Hao, Tor Lattimore, Wei Deng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8ba6c657b03fc7c8dd4dff8e45defcd2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8ba6c657b03fc7c8dd4dff8e45defcd2-Abstract.html)

        **Abstract**:

        Stochastic sparse linear bandits offer a practical model for high-dimensional online decision-making problems and have a rich information-regret structure. In this work we explore the use of information-directed sampling (IDS), which naturally balances the information-regret trade-off. We develop a class of information-theoretic Bayesian regret bounds that nearly match existing lower bounds on a variety of problem instances, demonstrating the adaptivity of IDS. To efficiently implement sparse IDS, we propose an empirical Bayesian approach for sparse posterior sampling using a spike-and-slab Gaussian-Laplace prior.  Numerical results demonstrate significant regret reductions by sparse IDS relative to several baselines.

        ----

        ## [1280] Linear Convergence of Gradient Methods for Estimating Structured Transition Matrices in High-dimensional Vector Autoregressive Models

        **Authors**: *Xiao Lv, Wei Cui, Yulong Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8be627bc543fd91be4d7f26ee86f5ee9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8be627bc543fd91be4d7f26ee86f5ee9-Abstract.html)

        **Abstract**:

        In this paper, we present non-asymptotic optimization guarantees of gradient descent methods for estimating structured transition matrices in high-dimensional vector autoregressive (VAR) models. We adopt the projected gradient descent (PGD) for single-structured transition matrices and the alternating projected gradient descent (AltPGD) for superposition-structured ones. Our analysis demonstrates that both gradient algorithms converge linearly to the statistical error even though the strong convexity of the objective function is absent under the high-dimensional settings. Moreover our result is sharp (up to a constant factor) in the sense of matching the phase transition theory of the corresponding  model with independent samples. To the best of our knowledge, this analysis constitutes first non-asymptotic optimization guarantees of the linear rate for regularized estimation in  high-dimensional VAR models. Numerical results are provided to support our theoretical analysis.

        ----

        ## [1281] Large-Scale Unsupervised Object Discovery

        **Authors**: *Huy V. Vo, Elena Sizikova, Cordelia Schmid, Patrick Pérez, Jean Ponce*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8bf1211fd4b7b94528899de0a43b9fb3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8bf1211fd4b7b94528899de0a43b9fb3-Abstract.html)

        **Abstract**:

        Existing approaches to unsupervised object discovery (UOD) do not scale up to large datasets without approximations that compromise their performance. We propose a novel formulation of UOD as a ranking problem, amenable to the arsenal of distributed methods available for eigenvalue problems and link analysis. Through the use of self-supervised features, we also demonstrate the first  effective fully unsupervised pipeline for UOD. Extensive experiments on COCO~\cite{Lin2014cocodataset} and OpenImages~\cite{openimages} show that, in the single-object discovery setting where a single prominent object is sought in each image, the proposed LOD (Large-scale Object Discovery) approach is on par with, or better than the state of the art for medium-scale datasets (up to 120K images), and over 37\% better than the only other algorithms capable of scaling up to 1.7M images. In the multi-object discovery setting where multiple objects are sought in each image, the proposed LOD is over 14\% better in average precision (AP) than all other methods for datasets ranging from 20K to 1.7M images. Using self-supervised features, we also show that the proposed method obtains state-of-the-art UOD performance on OpenImages.

        ----

        ## [1282] Sparse Steerable Convolutions: An Efficient Learning of SE(3)-Equivariant Features for Estimation and Tracking of Object Poses in 3D Space

        **Authors**: *Jiehong Lin, Hongyang Li, Ke Chen, Jiangbo Lu, Kui Jia*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8c1b6fa97c4288a4514365198566c6fa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8c1b6fa97c4288a4514365198566c6fa-Abstract.html)

        **Abstract**:

        As a basic component of SE(3)-equivariant deep feature learning, steerable convolution has recently demonstrated its advantages for 3D semantic analysis. The advantages are, however, brought by expensive computations on dense, volumetric data, which prevent its practical use for efficient processing of 3D data that are inherently sparse. In this paper, we propose a novel design of Sparse Steerable Convolution (SS-Conv) to address the shortcoming; SS-Conv greatly accelerates steerable convolution with sparse tensors, while strictly preserving the property of SE(3)-equivariance. Based on SS-Conv, we propose a general pipeline for precise estimation of object poses, wherein a key design is a Feature-Steering module that  takes the full advantage of SE(3)-equivariance and is able to conduct an efficient pose refinement. To verify our designs, we conduct thorough experiments on three tasks of 3D object semantic analysis, including instance-level 6D pose estimation, category-level 6D pose and size estimation, and category-level 6D pose tracking. Our proposed pipeline based on SS-Conv outperforms existing methods on almost all the metrics evaluated by the three tasks. Ablation studies also show the superiority of our SS-Conv over alternative convolutions in terms of both accuracy and efficiency. Our code is released publicly at https://github.com/Gorilla-Lab-SCUT/SS-Conv.

        ----

        ## [1283] Noisy Adaptation Generates Lévy Flights in Attractor Neural Networks

        **Authors**: *Xingsi Dong, Tianhao Chu, Tiejun Huang, Zilong Ji, Si Wu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8c249675aea6c3cbd91661bbae767ff1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8c249675aea6c3cbd91661bbae767ff1-Abstract.html)

        **Abstract**:

        Lévy flights describe a special class of random walks whose step sizes satisfy a power-law tailed distribution. As being an efficientsearching strategy in unknown environments, Lévy flights are widely observed in animal foraging behaviors. Recent studies further showed that human cognitive functions also exhibit the characteristics of Lévy flights.  Despite being a general phenomenon, the neural mechanism at the circuit level for generating Lévy flights remains unresolved. Here, we investigate how Lévy flights can be achieved in attractor neural networks. To elucidate the underlying mechanism clearly, we first study continuous attractor neural networks (CANNs), and find that noisy neural adaptation, exemplified by spike frequency adaptation (SFA) in this work,  can generate Lévy flights representing transitions of the network state in the attractor space. Specifically, the strength of SFA defines a travelling wave boundary, below which the network state displays local Brownian motion, and above which the network state displays long-jump motion. Noises in neural adaptation causes the network state to intermittently switch between these two motion modes, manifesting the characteristics of Lévy flights. We further extend the study to a general attractor neural network, and demonstrate that our model can explain the Lévy-flight phenomenon observed during free memory retrieval of humans. We hope that this study will give us insight into understanding the neural mechanism for optimal information processing in the brain.

        ----

        ## [1284] On Linear Stability of SGD and Input-Smoothness of Neural Networks

        **Authors**: *Chao Ma, Lexing Ying*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8c26d2fad09dc76f3ff36b6ea752b0e1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8c26d2fad09dc76f3ff36b6ea752b0e1-Abstract.html)

        **Abstract**:

        The multiplicative structure of parameters and input data in the first layer of neural networks is explored to build connection between the landscape of the loss function with respect to parameters and the landscape of the model function with respect to input data. By this connection, it is shown that flat minima regularize the gradient of the model function, which explains the good generalization performance of flat minima. Then, we go beyond the flatness and consider high-order moments of the gradient noise, and show that Stochastic Gradient Dascent (SGD) tends to impose constraints on these moments by a linear stability analysis of SGD around global minima. Together with the multiplicative structure, we identify the Sobolev regularization effect of SGD, i.e. SGD regularizes the Sobolev seminorms of the model function with respect to the input data. Finally, bounds for generalization error and adversarial robustness are provided for solutions found by SGD under assumptions of the data distribution.

        ----

        ## [1285] Joint inference and input optimization in equilibrium networks

        **Authors**: *Swaminathan Gurumurthy, Shaojie Bai, Zachary Manchester, J. Zico Kolter*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8c3c27ac7d298331a1bdfd0a5e8703d3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8c3c27ac7d298331a1bdfd0a5e8703d3-Abstract.html)

        **Abstract**:

        Many tasks in deep learning involve optimizing over the inputs to a network to minimize or maximize some objective; examples include optimization over latent spaces in a generative model to match a target image, or adversarially perturbing an input to worsen classifier performance.  Performing such optimization, however, is traditionally quite costly, as it involves a complete forward and backward pass through the network for each gradient step.  In a separate line of work, a recent thread of research has developed the deep equilibrium (DEQ) model, a class of models that foregoes traditional network depth and instead computes the output of a network by finding the fixed point of a single nonlinear layer. In this paper, we show that there is a natural synergy between these two settings. Although, naively using DEQs for these optimization problems is expensive (owing to the time needed to compute a fixed point for each gradient step), we can leverage the fact that gradient-based optimization can itself be cast as a fixed point iteration to substantially improve the overall speed. That is, we simultaneously both solve for the DEQ fixed point and optimize over network inputs, all within a single "augmented" DEQ model that jointly encodes both the original network and the optimization process.  Indeed, the procedure is fast enough that it allows us to efficiently train DEQ models for tasks traditionally relying on an "inner" optimization loop.  We demonstrate this strategy on various tasks such as training generative models while optimizing over latent codes, training models for inverse problems like denoising and inpainting, adversarial training and gradient based meta-learning.

        ----

        ## [1286] A unified framework for bandit multiple testing

        **Authors**: *Ziyu Xu, Ruodu Wang, Aaditya Ramdas*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8c460674cd61bf189e62b4da4bd9d7c1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8c460674cd61bf189e62b4da4bd9d7c1-Abstract.html)

        **Abstract**:

        In bandit multiple hypothesis testing, each arm corresponds to a different null hypothesis that we wish to test, and the goal is to design adaptive algorithms that correctly identify large set of interesting arms (true discoveries), while only mistakenly identifying a few uninteresting ones (false discoveries). One common metric in non-bandit multiple testing is the false discovery rate (FDR). We propose a unified, modular framework for bandit FDR control that emphasizes the decoupling of exploration and summarization of evidence. We utilize the powerful martingale-based concept of "e-processes" to ensure FDR control for arbitrary composite nulls, exploration rules and stopping times in generic problem settings. In particular, valid FDR control holds even if the reward distributions of the arms could be dependent, multiple arms may be queried simultaneously, and multiple (cooperating or competing) agents may be querying arms, covering combinatorial semi-bandit type settings as well. Prior work has considered in great detail the setting where each arm's reward distribution is independent and sub-Gaussian, and a single arm is queried at each step. Our framework recovers matching sample complexity guarantees in this special case, and performs comparably or better in practice. For other settings, sample complexities will depend on the finer details of the problem (composite nulls being tested, exploration algorithm, data dependence structure, stopping rule) and we do not explore these; our contribution is to show that the FDR guarantee is clean and entirely agnostic to these details.

        ----

        ## [1287] Recovering Latent Causal Factor for Generalization to Distributional Shifts

        **Authors**: *Xinwei Sun, Botong Wu, Xiangyu Zheng, Chang Liu, Wei Chen, Tao Qin, Tie-Yan Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8c6744c9d42ec2cb9e8885b54ff744d0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8c6744c9d42ec2cb9e8885b54ff744d0-Abstract.html)

        **Abstract**:

        Distributional shifts between training and target domains may degrade the prediction accuracy of learned models, mainly because these models often learn features that possess only correlation rather than causal relation with the output. Such a correlation, which is known as ``spurious correlation'' statistically, is domain-dependent hence may fail to generalize to unseen domains. To avoid such a spurious correlation, we propose \textbf{La}tent \textbf{C}ausal \textbf{I}nvariance \textbf{M}odels (LaCIM) that specifies the underlying causal structure of the data and the source of distributional shifts, guiding us to pursue only causal factor for prediction. Specifically, the LaCIM introduces a pair of correlated latent factors: (a) causal factor and (b) others, while the extent of this correlation is governed by a domain variable that characterizes the distributional shifts. On the basis of this, we prove that the distribution of observed variables conditioning on latent variables is shift-invariant. Equipped with such an invariance, we prove that the causal factor can be recovered without mixing information from others, which induces the ground-truth predicting mechanism. We propose a Variational-Bayesian-based method to learn this invariance for prediction. The utility of our approach is verified by improved generalization to distributional shifts on various real-world data. Our code is freely available at \url{https://github.com/wubotong/LaCIM}.

        ----

        ## [1288] Graph Differentiable Architecture Search with Structure Learning

        **Authors**: *Yijian Qin, Xin Wang, Zeyang Zhang, Wenwu Zhu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8c9f32e03aeb2e3000825c8c875c4edd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8c9f32e03aeb2e3000825c8c875c4edd-Abstract.html)

        **Abstract**:

        Discovering ideal Graph Neural Networks (GNNs) architectures for different tasks is labor intensive and time consuming. To save human efforts, Neural Architecture Search (NAS) recently has been used to automatically discover adequate GNN architectures for certain tasks in order to achieve competitive or even better performance compared with manually designed architectures. However, existing works utilizing NAS to search GNN structures fail to answer the question: how NAS is able to select the desired GNN architectures? In this paper, we investigate this question to solve the problem, for the first time. We conduct a measurement study with experiments to discover that gradient based NAS methods tend to select proper architectures based on the usefulness of different types of information with respect to the target task. Our explorations further show that gradient based NAS also suffers from noises hidden in the graph, resulting in searching suboptimal GNN architectures. Based on our findings, we propose a Graph differentiable Architecture Search model with Structure Optimization (GASSO), which allows differentiable search of the architecture with gradient descent and is able to discover graph neural architectures with better performance through employing graph structure learning as a denoising process in the search procedure. The proposed GASSO model is capable of simultaneously searching the optimal architecture and adaptively adjusting graph structure by jointly optimizing graph architecture search and graph structure denoising. Extensive experiments on real-world graph datasets demonstrate that our proposed GASSO model is able to achieve state-of-the-art performance compared with existing baselines.

        ----

        ## [1289] Designing Counterfactual Generators using Deep Model Inversion

        **Authors**: *Jayaraman J. Thiagarajan, Vivek Sivaraman Narayanaswamy, Deepta Rajan, Jia Liang, Akshay Chaudhari, Andreas Spanias*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html)

        **Abstract**:

        Explanation techniques that synthesize small, interpretable changes to a given image while producing desired changes in the model prediction have become popular for introspecting black-box models. Commonly referred to as counterfactuals, the synthesized explanations are required to contain discernible changes (for easy interpretability) while also being realistic (consistency to the data manifold). In this paper, we focus on the case where we have access only to the trained deep classifier and not the actual training data. While the problem of inverting deep models to synthesize images from the training distribution has been explored, our goal is to develop a deep inversion approach to generate counterfactual explanations for a given query image. Despite their effectiveness in conditional image synthesis, we show that existing deep inversion methods are insufficient for producing meaningful counterfactuals. We propose DISC (Deep Inversion for Synthesizing Counterfactuals) that improves upon deep inversion by utilizing (a) stronger image priors, (b) incorporating a novel manifold consistency objective and (c) adopting a progressive optimization strategy. We find that, in addition to producing visually meaningful explanations, the counterfactuals from DISC are effective at learning classifier decision boundaries and are robust to unknown test-time corruptions.

        ----

        ## [1290] A Faster Maximum Cardinality Matching Algorithm with Applications in Machine Learning

        **Authors**: *Nathaniel Lahn, Sharath Raghvendra, Jiacheng Ye*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8ca696ca160520b1cf5a569b4be525e8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8ca696ca160520b1cf5a569b4be525e8-Abstract.html)

        **Abstract**:

        Maximum cardinality bipartite matching is an important graph optimization problem with several applications. For instance, maximum cardinality matching in a $\delta$-disc graph can be used in the computation of the bottleneck matching as well as the $\infty$-Wasserstein and the LÃ©vy-Prokhorov distances between probability distributions. For any point sets $A, B \subset \mathbb{R}^2$, the $\delta$-disc graph is a bipartite graph formed by connecting every pair of points $(a,b) \in A\times B$ by an edge if the Euclidean distance between them is at most $\delta$. Using the classical Hopcroft-Karp algorithm, a maximum-cardinality matching on any $\delta$-disc graph can be found in $\tilde{O}(n^{3/2})$ time.~\footnote{We use $\tilde{O}(\cdot)$ to suppress poly-logarithmic terms in the complexity.} In this paper, we present a simplification of a recent algorithm (Lahn and Raghvendra, JoCG 2021) for the maximum cardinality matching problem and describe how a maximum cardinality matching in a $\delta$-disc graph can be computed asymptotically faster than $O(n^{3/2})$ time for any moderately dense point set. As applications, we show that if $A$ and $B$ are point sets drawn uniformly at random from a unit square, an exact bottleneck matching can be computed in $\tilde{O}(n^{4/3})$ time. On the other hand, experiments suggest that the Hopcroft-Karp algorithm seems to take roughly $\Theta (n^{3/2})$ time for this case. This translates to substantial improvements in execution time for larger inputs.

        ----

        ## [1291] Dynamic population-based meta-learning for multi-agent communication with natural language

        **Authors**: *Abhinav Gupta, Marc Lanctot, Angeliki Lazaridou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8caa38721906c1a0bb95c80fab33a893-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8caa38721906c1a0bb95c80fab33a893-Abstract.html)

        **Abstract**:

        In this work, our goal is to train agents that can coordinate with seen, unseen as well as human partners in a multi-agent communication environment involving natural language. Previous work using a single set of agents has shown great progress in generalizing to known partners, however it struggles when coordinating with unfamiliar agents. To mitigate that, recent work explored the use of population-based approaches, where multiple agents interact with each other with the goal of learning more generic protocols. These methods, while able to result in good coordination between unseen partners, still only achieve so in cases of simple languages, thus failing to adapt to human partners using natural language. We attribute this to the use of static populations and instead propose a dynamic population-based meta-learning approach that builds such a population in an iterative manner. We perform a holistic evaluation of our method on two different referential games, and show that our agents outperform all prior work when communicating with seen partners and humans. Furthermore, we analyze the natural language generation skills of our agents, where we find that our agents also outperform strong baselines. Finally, we test the robustness of our agents when communicating with out-of-population agents and carefully test the importance of each component of our method through ablation studies.

        ----

        ## [1292] Adversarial Neuron Pruning Purifies Backdoored Deep Models

        **Authors**: *Dongxian Wu, Yisen Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8cbe9ce23f42628c98f80fa0fac8b19a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8cbe9ce23f42628c98f80fa0fac8b19a-Abstract.html)

        **Abstract**:

        As deep neural networks (DNNs) are growing larger, their requirements for computational resources become huge, which makes outsourcing training more popular. Training in a third-party platform, however, may introduce potential risks that a malicious trainer will return backdoored DNNs, which behave normally on clean samples but output targeted misclassifications whenever a trigger appears at the test time. Without any knowledge of the trigger, it is difficult to distinguish or recover benign DNNs from backdoored ones. In this paper, we first identify an unexpected sensitivity of backdoored DNNs, that is, they are much easier to collapse and tend to predict the target label on clean samples when their neurons are adversarially perturbed. Based on these observations, we propose a novel model repairing method, termed Adversarial Neuron Pruning (ANP), which prunes some sensitive neurons to purify the injected backdoor. Experiments show, even with only an extremely small amount of clean data (e.g., 1%), ANP effectively removes the injected backdoor without causing obvious performance degradation.

        ----

        ## [1293] Towards Robust and Reliable Algorithmic Recourse

        **Authors**: *Sohini Upadhyay, Shalmali Joshi, Himabindu Lakkaraju*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8ccfb1140664a5fa63177fb6e07352f0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8ccfb1140664a5fa63177fb6e07352f0-Abstract.html)

        **Abstract**:

        As predictive models are increasingly being deployed in high-stakes decision making (e.g., loan approvals), there has been growing interest in post-hoc techniques which provide recourse to affected individuals.  These techniques generate recourses under the assumption that the underlying predictive model does not change. However, in practice, models are often regularly updated for a variety of reasons (e.g., dataset shifts), thereby rendering previously prescribed recourses ineffective.To address this problem, we propose a novel framework, RObust Algorithmic Recourse (ROAR), that leverages adversarial training for finding recourses that are robust to model shifts. To the best of our knowledge, this work proposes the first ever solution to this critical problem. We also carry out theoretical analysis which underscores the importance of constructing recourses that are robust to model shifts: 1) We quantify the probability of invalidation for recourses generated without accounting for model shifts. 2) We prove that the additional cost incurred due to the robust recourses output by our framework is bounded. Experimental evaluation on multiple synthetic and real-world datasets demonstrates the efficacy of the proposed framework.

        ----

        ## [1294] Neural Rule-Execution Tracking Machine For Transformer-Based Text Generation

        **Authors**: *Yufei Wang, Can Xu, Huang Hu, Chongyang Tao, Stephen Wan, Mark Dras, Mark Johnson, Daxin Jiang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8ce241e1ed84937ee48322b170b9b18c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8ce241e1ed84937ee48322b170b9b18c-Abstract.html)

        **Abstract**:

        Sequence-to-Sequence (Seq2Seq) neural text generation models, especially the pre-trained ones (e.g., BART and T5), have exhibited compelling performance on various natural language generation tasks. However, the black-box nature of these models limits their application in tasks where specific rules (e.g., controllable constraints, prior knowledge) need to be executed. Previous works either design specific model structures (e.g., Copy Mechanism corresponding to the rule "the generated output should include certain words in the source input'') or implement specialized inference algorithms (e.g., Constrained Beam Search) to execute particular rules through the text generation. These methods require the careful design case-by-case and are difficult to support multiple rules concurrently. In this paper, we propose a novel module named Neural Rule-Execution Tracking Machine (NRETM) that can be equipped into various transformer-based generators to leverage multiple rules simultaneously to guide the neural generation model for superior generation performance in an unified and scalable way. Extensive experiments on several benchmarks verify the effectiveness of our proposed model in both controllable and general text generation tasks.

        ----

        ## [1295] Scalable Online Planning via Reinforcement Learning Fine-Tuning

        **Authors**: *Arnaud Fickinger, Hengyuan Hu, Brandon Amos, Stuart J. Russell, Noam Brown*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8ce8b102d40392a688f8c04b3cd6cae0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8ce8b102d40392a688f8c04b3cd6cae0-Abstract.html)

        **Abstract**:

        Lookahead search has been a critical component of recent AI successes, such as in the games of chess, go, and poker. However, the search methods used in these games, and in many other settings, are tabular. Tabular search methods do not scale well with the size of the search space, and this problem is exacerbated by stochasticity and partial observability. In this work we replace tabular search with online model-based fine-tuning of a policy neural network via reinforcement learning, and show that this approach outperforms state-of-the-art search algorithms in benchmark settings. In particular, we use our search algorithm to achieve a new state-of-the-art result in self-play Hanabi, and show the generality of our algorithm by also showing that it outperforms tabular search in the Atari game Ms. Pacman.

        ----

        ## [1296] Adversarial Regression with Doubly Non-negative Weighting Matrices

        **Authors**: *Tam Le, Truyen Nguyen, Makoto Yamada, Jose H. Blanchet, Viet Anh Nguyen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8cfef17bee2b7a75a3ce09d40b497f6b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8cfef17bee2b7a75a3ce09d40b497f6b-Abstract.html)

        **Abstract**:

        Many machine learning tasks that involve predicting an output response can be solved by training a weighted regression model. Unfortunately, the predictive power of this type of models may severely deteriorate under low sample sizes or under covariate perturbations. Reweighting the training samples has aroused as an effective mitigation strategy to these problems. In this paper, we propose a novel and coherent scheme for kernel-reweighted regression by reparametrizing the sample weights using a doubly non-negative matrix. When the weighting matrix is confined in an uncertainty set using either the log-determinant divergence or the Bures-Wasserstein distance, we show that the adversarially reweighted estimate can be solved efficiently using first-order methods. Numerical experiments show that our reweighting strategy delivers promising results on numerous datasets.

        ----

        ## [1297] Learned Robust PCA: A Scalable Deep Unfolding Approach for High-Dimensional Outlier Detection

        **Authors**: *HanQin Cai, Jialin Liu, Wotao Yin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8d2355364e9a2ba1f82f975414937b43-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8d2355364e9a2ba1f82f975414937b43-Abstract.html)

        **Abstract**:

        Robust principal component analysis (RPCA) is a critical tool in modern machine learning, which detects outliers in the task of low-rank matrix reconstruction. In this paper, we propose a scalable and learnable non-convex approach for high-dimensional RPCA problems, which we call Learned Robust PCA (LRPCA). LRPCA is highly efficient, and its free parameters can be effectively learned to optimize via deep unfolding. Moreover, we extend deep unfolding from finite iterations to infinite iterations via a novel feedforward-recurrent-mixed neural network model. We establish the recovery guarantee of LRPCA under mild assumptions for RPCA. Numerical experiments show that LRPCA outperforms the state-of-the-art RPCA algorithms, such as ScaledGD and AltProj, on both synthetic datasets and real-world applications.

        ----

        ## [1298] Proxy-Normalizing Activations to Match Batch Normalization while Removing Batch Dependence

        **Authors**: *Antoine Labatie, Dominic Masters, Zach Eaton-Rosen, Carlo Luschi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8d2a5f7d4afa5d0530789d3066945330-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8d2a5f7d4afa5d0530789d3066945330-Abstract.html)

        **Abstract**:

        We investigate the reasons for the performance degradation incurred with batch-independent normalization. We find that the prototypical techniques of layer normalization and instance normalization both induce the appearance of failure modes in the neural network's pre-activations: (i) layer normalization induces a collapse towards channel-wise constant functions; (ii) instance normalization induces a lack of variability in instance statistics, symptomatic of an alteration of the expressivity. To alleviate failure mode (i) without aggravating failure mode (ii), we introduce the technique "Proxy Normalization" that normalizes post-activations using a proxy distribution. When combined with layer normalization or group normalization, this batch-independent normalization emulates batch normalization's behavior and consistently matches or exceeds its performance.

        ----

        ## [1299] Dynamic Bottleneck for Robust Self-Supervised Exploration

        **Authors**: *Chenjia Bai, Lingxiao Wang, Lei Han, Animesh Garg, Jianye Hao, Peng Liu, Zhaoran Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8d3369c4c086f236fabf61d614a32818-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8d3369c4c086f236fabf61d614a32818-Abstract.html)

        **Abstract**:

        Exploration methods based on pseudo-count of transitions or curiosity of dynamics have achieved promising results in solving reinforcement learning with sparse rewards. However, such methods are usually sensitive to environmental dynamics-irrelevant information, e.g., white-noise. To handle such dynamics-irrelevant information, we propose a Dynamic Bottleneck (DB) model, which attains a dynamics-relevant representation based on the information-bottleneck principle. Based on the DB model, we further propose DB-bonus, which encourages the agent to explore state-action pairs with high information gain. We establish theoretical connections between the proposed DB-bonus, the upper confidence bound (UCB) for linear case, and the visiting count for tabular case. We evaluate the proposed method on Atari suits with dynamics-irrelevant noises. Our experiments show that exploration with DB bonus outperforms several state-of-the-art exploration methods in noisy environments.

        ----

        ## [1300] ProTo: Program-Guided Transformer for Program-Guided Tasks

        **Authors**: *Zelin Zhao, Karan Samel, Binghong Chen, Le Song*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8d34201a5b85900908db6cae92723617-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8d34201a5b85900908db6cae92723617-Abstract.html)

        **Abstract**:

        Programs, consisting of semantic and structural information, play an important role in the communication between humans and agents. Towards learning general program executors to unify perception, reasoning, and decision making, we formulate program-guided tasks which require learning to execute a given program on the observed task specification. Furthermore, we propose Program-Guided Transformers (ProTo), which integrates both semantic and structural guidance of a program by leveraging cross-attention and masked self-attention to pass messages between the specification and routines in the program. ProTo executes a program in a learned latent space and enjoys stronger representation ability than previous neural-symbolic approaches. We demonstrate that ProTo significantly outperforms the previous state-of-the-art methods on GQA visual reasoning and 2D Minecraft policy learning datasets. Additionally, ProTo demonstrates better generalization to unseen, complex, and human-written programs.

        ----

        ## [1301] An Efficient Transfer Learning Framework for Multiagent Reinforcement Learning

        **Authors**: *Tianpei Yang, Weixun Wang, Hongyao Tang, Jianye Hao, Zhaopeng Meng, Hangyu Mao, Dong Li, Wulong Liu, Yingfeng Chen, Yujing Hu, Changjie Fan, Chengwei Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8d9a6e908ed2b731fb96151d9bb94d49-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8d9a6e908ed2b731fb96151d9bb94d49-Abstract.html)

        **Abstract**:

        Transfer Learning has shown great potential to enhance single-agent Reinforcement Learning (RL) efficiency. Similarly, Multiagent RL (MARL) can also be accelerated if agents can share knowledge with each other. However, it remains a problem of how an agent should learn from other agents. In this paper, we propose a novel Multiagent Policy Transfer Framework (MAPTF) to improve MARL efficiency. MAPTF learns which agent's policy is the best to reuse for each agent and when to terminate it by modeling multiagent policy transfer as the option learning problem. Furthermore, in practice, the option module can only collect all agent's local experiences for update due to the partial observability of the environment. While in this setting, each agent's experience may be inconsistent with each other, which may cause the inaccuracy and oscillation of the option-value's estimation. Therefore, we propose a novel option learning algorithm, the successor representation option learning to solve it by decoupling the environment dynamics from rewards and learning the option-value under each agent's preference. MAPTF can be easily combined with existing deep RL and MARL approaches, and experimental results show it significantly boosts the performance of existing methods in both discrete and continuous state spaces.

        ----

        ## [1302] Learning to Time-Decode in Spiking Neural Networks Through the Information Bottleneck

        **Authors**: *Nicolas Skatchkovsky, Osvaldo Simeone, Hyeryung Jang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8da57fac3313174128cc5f13328d4573-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8da57fac3313174128cc5f13328d4573-Abstract.html)

        **Abstract**:

        One of the key challenges in training Spiking Neural Networks (SNNs) is that target outputs typically come in the form of natural signals, such as labels for classification or images for generative models, and need to be encoded into spikes. This is done by handcrafting target spiking signals, which in turn implicitly fixes the mechanisms used to decode spikes into natural signals, e.g., rate decoding. The arbitrary choice of target signals and decoding rule generally impairs the capacity of the SNN to encode and process information in the timing of spikes. To address this problem, this work introduces a hybrid variational autoencoder architecture, consisting of an encoding SNN and a decoding Artificial Neural Network (ANN). The role of the decoding ANN is to learn how to best convert the spiking signals output by the SNN into the target natural signal. A novel end-to-end learning rule is introduced that optimizes a directed information bottleneck training criterion via surrogate gradients. We demonstrate the applicability of the technique in an experimental settings on various tasks, including real-life datasets.

        ----

        ## [1303] NEO: Non Equilibrium Sampling on the Orbits of a Deterministic Transform

        **Authors**: *Achille Thin, Yazid Janati El Idrissi, Sylvain Le Corff, Charles Ollion, Eric Moulines, Arnaud Doucet, Alain Durmus, Christian X. Robert*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8dd291cbea8f231982db0fb1716dfc55-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8dd291cbea8f231982db0fb1716dfc55-Abstract.html)

        **Abstract**:

        Sampling from a complex distribution $\pi$ and approximating its intractable normalizing constant $\mathrm{Z}$ are challenging problems. In this paper, a novel family of importance samplers (IS) and Markov chain Monte Carlo (MCMC) samplers is derived. Given an invertible map $\mathrm{T}$, these schemes combine (with weights) elements from the forward and backward Orbits   through points sampled from a proposal distribution $\rho$. The map $\mathrm{T}$ does not leave the target $\pi$ invariant, hence the name NEO, standing for Non-Equilibrium Orbits. NEO-IS provides unbiased estimators of the normalizing constant and self-normalized IS estimators of expectations under $\pi$ while NEO-MCMC combines multiple NEO-IS estimates of the normalizing constant and an iterated sampling-importance resampling mechanism to sample from $\pi$. For $\mathrm{T}$ chosen as a discrete-time integrator of a conformal Hamiltonian system, NEO-IS achieves state-of-the art performance on difficult benchmarks and NEO-MCMC is able to explore highly multimodal targets. Additionally, we provide detailed theoretical results for both methods. In particular, we show that NEO-MCMC is uniformly geometrically ergodic and establish explicit mixing time estimates under mild conditions.

        ----

        ## [1304] Relaxing Local Robustness

        **Authors**: *Klas Leino, Matt Fredrikson*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8df6a65941e4c9da40a4fb899de65c55-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8df6a65941e4c9da40a4fb899de65c55-Abstract.html)

        **Abstract**:

        Certifiable local robustness, which rigorously precludes small-norm adversarial examples, has received significant attention as a means of addressing security concerns in deep learning. However, for some classification problems, local robustness is not a natural objective, even in the presence of adversaries; for example, if an image contains two classes of subjects, the correct label for the image may be considered arbitrary between the two, and thus enforcing strict separation between them is unnecessary. In this work, we introduce two relaxed safety properties for classifiers that address this observation: (1) relaxed top-k robustness, which serves as the analogue of top-k accuracy; and (2) affinity robustness, which specifies which sets of labels must be separated by a robustness margin, and which can be $\epsilon$-close in $\ell_p$ space. We show how to construct models that can be efficiently certified against each relaxed robustness property, and trained with very little overhead relative to standard gradient descent. Finally, we demonstrate experimentally that these relaxed variants of robustness are well-suited to several significant classification problems, leading to lower rejection rates and higher certified accuracies than can be obtained when certifying "standard" local robustness.

        ----

        ## [1305] Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer

        **Authors**: *Ge Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8df7c2e3c3c3be098ef7b382bd2c37ba-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8df7c2e3c3c3be098ef7b382bd2c37ba-Abstract.html)

        **Abstract**:

        Hyperparameter (HP) tuning in deep learning is an expensive process, prohibitively so for neural networks (NNs) with billions of parameters.We show that, in the recently discovered Maximal Update Parametrization ($\mu$P), many optimal HPs remain stable even as model size changes. This leads to a new HP tuning paradigm we call *$\mu$Transfer*: parametrize the target model in $\mu$P, tune the HP indirectly on a smaller model, and *zero-shot transfer* them to the full-sized model, i.e., without directly tuning the latter at all.We verify $\mu$Transfer on Transformer and ResNet. For example, 1) by transferring pretraining HPs from a model of 13M parameters, we outperform published numbers of BERT-large (350M parameters), with a total tuning cost equivalent to pretraining BERT-large once; 2) by transferring from 40M parameters, we outperform published numbers of the 6.7B GPT-3 model, with tuning cost only 7% of total pretraining cost. A Pytorch implementation of our technique can be found at github.com/microsoft/mup. See arxiv.org for the full, up-to-date version of this work.

        ----

        ## [1306] Statistical Regeneration Guarantees of the Wasserstein Autoencoder with Latent Space Consistency

        **Authors**: *Anish Chakrabarty, Swagatam Das*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8e036cc193d0af59aa9b22821248292b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8e036cc193d0af59aa9b22821248292b-Abstract.html)

        **Abstract**:

        The introduction of Variational Autoencoders (VAE) has been marked as a breakthrough in the history of representation learning models. Besides having several accolades of its own, VAE has successfully flagged off a series of inventions in the form of its immediate successors. Wasserstein Autoencoder (WAE), being an heir to that realm carries with it all of the goodness and heightened generative promises, matching even the generative adversarial networks (GANs). Needless to say, recent years have witnessed a remarkable resurgence in statistical analyses of the GANs. Similar examinations for Autoencoders however, despite their diverse applicability and notable empirical performance, remain largely absent. To close this gap, in this paper, we investigate the statistical properties of WAE. Firstly, we provide statistical guarantees that WAE achieves the target distribution in the latent space, utilizing the Vapnikâ€“Chervonenkis (VC) theory. The main result, consequently ensures the regeneration of the input distribution, harnessing the potential offered by Optimal Transport of measures under the Wasserstein metric. This study, in turn, hints at the class of distributions WAE can reconstruct after suffering a compression in the form of a latent law.

        ----

        ## [1307] Leveraging the Inductive Bias of Large Language Models for Abstract Textual Reasoning

        **Authors**: *Christopher Michael Rytting, David Wingate*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8e08227323cd829e449559bb381484b7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8e08227323cd829e449559bb381484b7-Abstract.html)

        **Abstract**:

        Large natural language models (LMs) (such as GPT-3 or T5) demonstrate impressive abilities across a range of general NLP tasks. Here, we show that the knowledge embedded in such models provides a useful inductive bias, not just on traditional NLP tasks, but also in the nontraditional task of training a symbolic reasoning engine. We observe that these engines learn quickly and generalize in a natural way that reflects human intuition. For example, training such a system to model block-stacking might naturally generalize to stacking other types of objects because of structure in the real world that has been partially captured by the language describing it. We study several abstract textual reasoning tasks, such as object manipulation and navigation, and demonstrate multiple types of generalization to novel scenarios and the symbols that comprise them. We also demonstrate the surprising utility of $\textit{compositional learning}$, where a learner dedicated to mastering a complicated task gains an advantage by training on relevant simpler tasks instead of jumping straight to the complicated task.

        ----

        ## [1308] Differentiable Simulation of Soft Multi-body Systems

        **Authors**: *Yi-Ling Qiao, Junbang Liang, Vladlen Koltun, Ming C. Lin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8e296a067a37563370ded05f5a3bf3ec-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8e296a067a37563370ded05f5a3bf3ec-Abstract.html)

        **Abstract**:

        We present a method for differentiable simulation of soft articulated bodies. Our work enables the integration of differentiable physical dynamics into gradient-based pipelines. We develop a top-down matrix assembly algorithm within Projective Dynamics and derive a generalized dry friction model for soft continuum using a new matrix splitting strategy. We derive a differentiable control framework for soft articulated bodies driven by muscles, joint torques, or pneumatic tubes. The experiments demonstrate that our designs make soft body simulation more stable and realistic compared to other frameworks. Our method accelerates the solution of system identification problems by more than an order of magnitude, and enables efficient gradient-based learning of motion control with soft robots.

        ----

        ## [1309] Good Classification Measures and How to Find Them

        **Authors**: *Martijn Gösgens, Anton Zhiyanov, Aleksey Tikhonov, Liudmila Prokhorenkova*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8e489b4966fe8f703b5be647f1cbae63-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8e489b4966fe8f703b5be647f1cbae63-Abstract.html)

        **Abstract**:

        Several performance measures can be used for evaluating classification results: accuracy, F-measure, and many others. Can we say that some of them are better than others, or, ideally, choose one measure that is best in all situations? To answer this question, we conduct a systematic analysis of classification performance measures: we formally define a list of desirable properties and theoretically analyze which measures satisfy which properties. We also prove an impossibility theorem: some desirable properties cannot be simultaneously satisfied. Finally, we propose a new family of measures satisfying all desirable properties except one. This family includes the Matthews Correlation Coefficient and a so-called Symmetric Balanced Accuracy that was not previously used in classification literature. We believe that our systematic approach gives an important tool to practitioners for adequately evaluating classification results.

        ----

        ## [1310] Distilling Robust and Non-Robust Features in Adversarial Examples by Information Bottleneck

        **Authors**: *Junho Kim, Byung-Kwan Lee, Yong Man Ro*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8e5e15c4e6d09c8333a17843461041a9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8e5e15c4e6d09c8333a17843461041a9-Abstract.html)

        **Abstract**:

        Adversarial examples, generated by carefully crafted perturbation, have attracted considerable attention in research fields. Recent works have argued that the existence of the robust and non-robust features is a primary cause of the adversarial examples, and investigated their internal interactions in the feature space. In this paper, we propose a way of explicitly distilling feature representation into the robust and non-robust features, using Information Bottleneck. Specifically, we inject noise variation to each feature unit and evaluate the information flow in the feature representation to dichotomize feature units either robust or non-robust, based on the noise variation magnitude. Through comprehensive experiments, we demonstrate that the distilled features are highly correlated with adversarial prediction, and they have human-perceptible semantic information by themselves. Furthermore, we present an attack mechanism intensifying the gradient of non-robust features that is directly related to the model prediction, and validate its effectiveness of breaking model robustness.

        ----

        ## [1311] Vector-valued Gaussian Processes on Riemannian Manifolds via Gauge Independent Projected Kernels

        **Authors**: *Michael J. Hutchinson, Alexander Terenin, Viacheslav Borovitskiy, So Takao, Yee Whye Teh, Marc Peter Deisenroth*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8e7991af8afa942dc572950e01177da5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8e7991af8afa942dc572950e01177da5-Abstract.html)

        **Abstract**:

        Gaussian processes are machine learning models capable of learning unknown functions in a way that represents uncertainty, thereby facilitating construction of optimal decision-making systems. Motivated by a desire to deploy Gaussian processes in novel areas of science, a rapidly-growing line of research has focused on constructively extending these models to handle non-Euclidean domains, including Riemannian manifolds, such as spheres and tori. We propose techniques that generalize this class to model vector fields on Riemannian manifolds, which are important in a number of application areas in the physical sciences. To do so, we present a general recipe for constructing gauge independent kernels, which induce Gaussian vector fields, i.e. vector-valued Gaussian processes coherent withgeometry, from scalar-valued Riemannian kernels. We extend standard Gaussian process training methods, such as variational inference, to this setting. This enables vector-valued Gaussian processes on Riemannian manifolds to be trained using standard methods and makes them accessible to machine learning practitioners.

        ----

        ## [1312] On the Representation Power of Set Pooling Networks

        **Authors**: *Christian Bueno, Alan Hylton*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8ea1e4f9f24c38f168d538c9cfc50a14-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8ea1e4f9f24c38f168d538c9cfc50a14-Abstract.html)

        **Abstract**:

        Point clouds and sets are input data-types which pose unique problems to deep learning. Since sets can have variable cardinality and are unchanged by permutation, the input space for these problems naturally form infinite-dimensional non-Euclidean spaces. Despite these mathematical difficulties, PointNet (Qi et al. 2017) and Deep Sets (Zaheer et al. 2017) introduced foundational neural network architectures to address these problems. In this paper we present a unified framework to study the expressive power of such networks as well as their extensions beyond point clouds (partially addressing a conjecture on the extendibility of DeepSets along the way). To this end, we demonstrate the crucial role that the Hausdorff and Wasserstein metrics play and prove new cardinality-agnostic universality results to characterize exactly which functions can be approximated by these models. In particular, these results imply that PointNet generally cannot approximate averages of continuous functions over sets (e.g. center-of-mass or higher moments) implying that DeepSets is strictly more expressive than PointNet in the constant cardinality setting. Moreover, we obtain explicit lower-bounds on the approximation error and present a simple method to produce arbitrarily many examples of this failure-mode. Counterintuitively, we also prove that in the unbounded cardinality setting that any function which can be uniformly approximated by both PointNet and normalized-DeepSets must be constant. Finally, we also prove theorems on the Lipschitz properties of PointNet and normalized-DeepSets which shed insight into exploitable inductive bias in these networks.

        ----

        ## [1313] Learning Policies with Zero or Bounded Constraint Violation for Constrained MDPs

        **Authors**: *Tao Liu, Ruida Zhou, Dileep Kalathil, Panganamala R. Kumar, Chao Tian*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8ec2ba5e96ec1c050bc631abda80f269-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8ec2ba5e96ec1c050bc631abda80f269-Abstract.html)

        **Abstract**:

        We address the issue of safety in reinforcement learning. We pose the problem in an episodic framework of a constrained Markov decision process. Existing results have shown that it is possible to achieve a reward regret of $\tilde{\mathcal{O}}(\sqrt{K})$ while allowing an $\tilde{\mathcal{O}}(\sqrt{K})$ constraint violation in $K$ episodes. A critical question that arises is whether it is possible to keep the constraint violation even smaller. We show that when a strictly safe policy is known, then one can confine the system to zero constraint violation with arbitrarily high probability while keeping the reward regret of order $\tilde{\mathcal{O}}(\sqrt{K})$. The algorithm which does so employs the principle of optimistic pessimism in the face of uncertainty to achieve safe exploration. When no strictly safe policy is known, though one is known to exist, then it is possible to restrict the system to bounded constraint violation with arbitrarily high probability. This is shown to be realized by a primal-dual algorithm with an optimistic primal estimate and a pessimistic dual update.

        ----

        ## [1314] A Prototype-Oriented Framework for Unsupervised Domain Adaptation

        **Authors**: *Korawat Tanwisuth, Xinjie Fan, Huangjie Zheng, Shujian Zhang, Hao Zhang, Bo Chen, Mingyuan Zhou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8edd72158ccd2a879f79cb2538568fdc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8edd72158ccd2a879f79cb2538568fdc-Abstract.html)

        **Abstract**:

        Existing methods for unsupervised domain adaptation often rely on minimizing some statistical distance between the source and target samples in the latent space. To avoid the sampling variability, class imbalance, and data-privacy concerns that often plague these methods, we instead provide a memory and computation-efficient probabilistic framework to extract class prototypes and align the target features with them. We demonstrate the general applicability of our method on a wide range of scenarios, including single-source, multi-source, class-imbalance, and source-private domain adaptation. Requiring no additional model parameters and having a moderate increase in computation over the source model alone, the proposed method achieves competitive performance with state-of-the-art methods.

        ----

        ## [1315] Mining the Benefits of Two-stage and One-stage HOI Detection

        **Authors**: *Aixi Zhang, Yue Liao, Si Liu, Miao Lu, Yongliang Wang, Chen Gao, Xiaobo Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html)

        **Abstract**:

        Two-stage methods have dominated Human-Object Interaction~(HOI) detection for several years. Recently, one-stage HOI detection methods have become popular. In this paper, we aim to explore the essential pros and cons of two-stage and one-stage methods. With this as the goal, we find that conventional two-stage methods mainly suffer from positioning positive interactive human-object pairs, while one-stage methods are challenging to make an appropriate trade-off on multi-task learning, \emph{i.e.}, object detection, and interaction classification.  Therefore, a core problem is how to take the essence and discard the dregs from the conventional two types of methods. To this end, we propose a novel one-stage framework with disentangling human-object detection and interaction classification in a cascade manner. In detail, we first design a human-object pair generator based on a state-of-the-art one-stage HOI detector by removing the interaction classification module or head and then design a relatively isolated interaction classifier to classify each human-object pair. Two cascade decoders in our proposed framework can focus on one specific task, detection or interaction classification. In terms of the specific implementation, we adopt a transformer-based HOI detector as our base model. The newly introduced disentangling paradigm outperforms existing methods by a large margin, with a significant relative mAP gain of 9.32% on HICO-Det. The source codes are available at https://github.com/YueLiao/CDN.

        ----

        ## [1316] Discerning Decision-Making Process of Deep Neural Networks with Hierarchical Voting Transformation

        **Authors**: *Ying Sun, Hengshu Zhu, Chuan Qin, Fuzhen Zhuang, Qing He, Hui Xiong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8f1fa0193ca2b5d2fa0695827d8270e9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8f1fa0193ca2b5d2fa0695827d8270e9-Abstract.html)

        **Abstract**:

        Neural network based deep learning techniques have shown great success for numerous applications. While it is expected to understand their intrinsic decision-making processes, these deep neural networks often work in a black-box way. To this end, in this paper, we aim to discern the decision-making processes of neural networks through a hierarchical voting strategy by developing an explainable deep learning model, namely Voting Transformation-based Explainable Neural Network (VOTEN). Specifically, instead of relying on massive feature combinations, VOTEN creatively models expressive single-valued voting functions between explicitly modeled latent concepts to achieve high fitting ability. Along this line, we first theoretically analyze the major components of VOTEN and prove the relationship and advantages of VOTEN compared with Multi-Layer Perceptron (MLP), the basic structure of deep neural networks. Moreover, we design efficient algorithms to improve the model usability by explicitly showing the decision processes of VOTEN. Finally, extensive experiments on multiple real-world datasets clearly validate the performances and explainability of VOTEN.

        ----

        ## [1317] Risk-averse Heteroscedastic Bayesian Optimization

        **Authors**: *Anastasia Makarova, Ilnura Usmanova, Ilija Bogunovic, Andreas Krause*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8f97d1d7e02158a83ceb2c14ff5372cd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8f97d1d7e02158a83ceb2c14ff5372cd-Abstract.html)

        **Abstract**:

        Many black-box optimization tasks arising in high-stakes applications require risk-averse decisions. The standard Bayesian optimization (BO) paradigm, however, optimizes the expected value only.  We generalize BO to trade mean and input-dependent variance of the objective, both of which we assume to be unknown a priori.  In particular, we propose a novel risk-averse heteroscedastic Bayesian optimization algorithm (RAHBO) that aims to identify a solution with high return and low noise variance, while learning the noise distribution on the fly.  To this end, we model both expectation and variance as (unknown) RKHS functions, and propose a novel risk-aware acquisition function.  We bound the regret for our approach and provide a robust rule to report the final decision point for applications where only a single solution must be identified. We demonstrate the effectiveness of RAHBO on synthetic benchmark functions and hyperparameter tuning tasks.

        ----

        ## [1318] Invertible DenseNets with Concatenated LipSwish

        **Authors**: *Yura Perugachi-Diaz, Jakub M. Tomczak, Sandjai Bhulai*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)

        **Abstract**:

        We introduce Invertible Dense Networks (i-DenseNets), a more parameter efficient extension of Residual Flows. The method relies on an analysis of the Lipschitz continuity of the concatenation in DenseNets, where we enforce invertibility of the network by satisfying the Lipschitz constant. Furthermore, we propose a learnable weighted concatenation, which not only improves the model performance but also indicates the importance of the concatenated weighted representation. Additionally, we introduce the Concatenated LipSwish as activation function, for which we show how to enforce the Lipschitz condition and which boosts performance. The new architecture, i-DenseNet, out-performs Residual Flow and other flow-based models on density estimation evaluated in bits per dimension, where we utilize an equal parameter budget. Moreover, we show that the proposed model out-performs Residual Flows when trained as a hybrid model where the model is both a generative and a discriminative model.

        ----

        ## [1319] Topological Detection of Trojaned Neural Networks

        **Authors**: *Songzhu Zheng, Yikai Zhang, Hubert Wagner, Mayank Goswami, Chao Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8fd7f981e10b41330b618129afcaab2d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8fd7f981e10b41330b618129afcaab2d-Abstract.html)

        **Abstract**:

        Deep neural networks are known to have security issues. One particular threat is the Trojan attack. It occurs when the attackers stealthily manipulate the model's behavior through Trojaned training samples, which can later be exploited. Guided by basic neuroscientific principles, we discover subtle -- yet critical -- structural deviation characterizing Trojaned models. In our analysis we use topological tools. They allow us to model high-order dependencies in the networks, robustly compare different networks, and localize structural abnormalities. One interesting observation is that Trojaned models develop short-cuts from shallow to deep layers. Inspired by these observations, we devise a strategy for robust detection of Trojaned models. Compared to standard baselines it displays better performance on multiple benchmarks.

        ----

        ## [1320] Provably Strict Generalisation Benefit for Invariance in Kernel Methods

        **Authors**: *Bryn Elesedy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/8fe04df45a22b63156ebabbb064fcd5e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8fe04df45a22b63156ebabbb064fcd5e-Abstract.html)

        **Abstract**:

        It is a commonly held belief that enforcing invariance improves generalisation. Although this approach enjoys widespread popularity, it is only very recently that a rigorous theoretical demonstration of this benefit has been established. In this work we build on the function space perspective of Elesedy and Zaidi [8] to derive a strictly non-zero generalisation benefit of incorporating invariance in kernel ridge regression when the target is invariant to the action of a compact group. We study invariance enforced by feature averaging and find that generalisation is governed by a notion of effective dimension that arises from the interplay between the kernel and the group. In building towards this result, we find that the action of the group induces an orthogonal decomposition of both the reproducing kernel Hilbert space and its kernel, which may be of interest in its own right.

        ----

        ## [1321] Formalizing the Generalization-Forgetting Trade-off in Continual Learning

        **Authors**: *Krishnan Raghavan, Prasanna Balaprakash*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/901797aebf0b23ecbab534d61ad33bb1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/901797aebf0b23ecbab534d61ad33bb1-Abstract.html)

        **Abstract**:

        We formulate the continual learning (CL) problem via dynamic programming and model the trade-off between catastrophic forgetting and generalization as a two-player sequential game. In this approach, player 1 maximizes the cost due to lack of generalization whereas player 2 minimizes the cost due to catastrophic forgetting. We show theoretically that a balance point between the two players exists for each task and that this point is stable (once the balance is achieved, the two players stay at the balance point). Next, we introduce balanced continual learning (BCL), which is designed to attain balance between generalization and forgetting and empirically demonstrate that BCL is comparable to or better than the state of the art.

        ----

        ## [1322] Risk-Aware Transfer in Reinforcement Learning using Successor Features

        **Authors**: *Michael Gimelfarb, André Barreto, Scott Sanner, Chi-Guhn Lee*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/90610aa0e24f63ec6d2637e06f9b9af2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/90610aa0e24f63ec6d2637e06f9b9af2-Abstract.html)

        **Abstract**:

        Sample efficiency and risk-awareness are central to the development of practical reinforcement learning (RL) for complex decision-making. The former can be addressed by transfer learning, while the latter by optimizing some utility function of the return. However, the problem of transferring skills in a risk-aware manner is not well-understood. In this paper, we address the problem of transferring policies between tasks in a common domain that differ only in their reward functions, in which risk is measured by the variance of reward streams. Our approach begins by extending the idea of generalized policy improvement to maximize entropic utilities, thus extending the dynamic programming's policy improvement operation to sets of policies \emph{and} levels of risk-aversion. Next, we extend the idea of successor features (SF), a value function representation that decouples the environment dynamics from the rewards, to capture the variance of returns. Our resulting risk-aware successor features (RaSF) integrate seamlessly within the RL framework, inherit the superior task generalization ability of SFs, while incorporating risk into the decision-making. Experiments on a discrete navigation domain and control of a simulated robotic arm demonstrate the ability of RaSFs to outperform alternative methods including SFs, when taking the risk of the learned policies into account.

        ----

        ## [1323] Causal Inference for Event Pairs in Multivariate Point Processes

        **Authors**: *Tian Gao, Dharmashankar Subramanian, Debarun Bhattacharjya, Xiao Shou, Nicholas Mattei, Kristin P. Bennett*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9078f2a8254704bd760460f027072e52-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9078f2a8254704bd760460f027072e52-Abstract.html)

        **Abstract**:

        Causal inference and discovery from observational data has been extensively studied across multiple fields. However, most prior work has focused on independent and identically distributed (i.i.d.) data. In this paper, we propose a formalization for causal inference between pairs of event variables in multivariate recurrent event streams by extending Rubin's framework for the average treatment effect (ATE) and propensity scores to multivariate point processes. Analogous to a joint probability distribution representing i.i.d. data, a multivariate point process represents data involving asynchronous and irregularly spaced occurrences of various types of events over a common timeline. We theoretically justify our point process causal framework and show how to obtain unbiased estimates of the proposed measure. We conduct an experimental investigation using synthetic and real-world event datasets, where our proposed causal inference framework is shown to exhibit superior performance against a set of baseline pairwise causal association scores.

        ----

        ## [1324] Evaluating model performance under worst-case subpopulations

        **Authors**: *Mike Li, Hongseok Namkoong, Shangzhou Xia*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/908075ea2c025c335f4865f7db427062-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/908075ea2c025c335f4865f7db427062-Abstract.html)

        **Abstract**:

        The performance of ML models degrades when the training population is different from that seen under operation. Towards assessing distributional robustness, we study the worst-case performance of a model over all subpopulations of a given size, defined with respect to core attributes $Z$. This notion of robustness can consider arbitrary (continuous) attributes $Z$, and automatically accounts for complex intersectionality in disadvantaged groups. We develop a scalable yet principled two-stage estimation procedure that can evaluate the robustness of state-of-the-art models. We prove that our procedure enjoys several finite-sample convergence guarantees, including dimension-free convergence. Instead of overly conservative notions based on Rademacher complexities, our evaluation error depends on the dimension of $Z$ only through the out-of-sample error in estimating the performance conditional on $Z$. On real datasets, we demonstrate that our method certifies the robustness of a model and prevents deployment of unreliable models.

        ----

        ## [1325] Privately Publishable Per-instance Privacy

        **Authors**: *Rachel Redberg, Yu-Xiang Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9087b0efc7c7acd1ef7e153678809c77-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9087b0efc7c7acd1ef7e153678809c77-Abstract.html)

        **Abstract**:

        We consider how to privately share the personalized privacy losses incurred by objective perturbation, using per-instance differential privacy (pDP). Standard differential privacy (DP) gives us a worst-case bound that might be orders of magnitude larger than the privacy loss to a particular individual relative to a fixed dataset. The pDP framework provides a more fine-grained analysis of the privacy guarantee to a target individual, but the per-instance privacy loss itself might be a function of sensitive data. In this paper, we analyze the per-instance privacy loss of releasing a private empirical risk minimizer learned via objective perturbation, and propose a group of methods to privately and accurately publish the pDP losses at little to no additional privacy cost.

        ----

        ## [1326] Understanding the Limits of Unsupervised Domain Adaptation via Data Poisoning

        **Authors**: *Akshay Mehra, Bhavya Kailkhura, Pin-Yu Chen, Jihun Hamm*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/90cc440b1b8caa520c562ac4e4bbcb51-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/90cc440b1b8caa520c562ac4e4bbcb51-Abstract.html)

        **Abstract**:

        Unsupervised domain adaptation (UDA) enables cross-domain learning without target domain labels by transferring knowledge from a labeled source domain whose distribution differs from that of the target. However, UDA is not always successful and several accounts of `negative transfer' have been reported in the literature. In this work, we prove a simple lower bound on the target domain error that complements the existing upper bound. Our bound shows the insufficiency of minimizing source domain error and marginal distribution mismatch for a guaranteed reduction in the target domain error, due to the possible increase of induced labeling function mismatch. This insufficiency is further illustrated through simple distributions for which the same UDA approach succeeds, fails, and may succeed or fail with an equal chance. Motivated from this, we propose novel data poisoning attacks to fool UDA methods into learning representations that produce large target domain errors. We evaluate the effect of these attacks on popular UDA methods using benchmark datasets where they have been previously shown to be successful. Our results show that poisoning can significantly decrease the target domain accuracy, dropping it to almost 0% in some cases, with the addition of only 10% poisoned data in the source domain. The failure of these UDA methods demonstrates their limitations at guaranteeing cross-domain generalization consistent with our lower bound. Thus, evaluating UDA methods in adversarial settings such as data poisoning provides a better sense of their robustness to data distributions unfavorable for UDA.

        ----

        ## [1327] Coresets for Clustering with Missing Values

        **Authors**: *Vladimir Braverman, Shaofeng H.-C. Jiang, Robert Krauthgamer, Xuan Wu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/90fd4f88f588ae64038134f1eeaa023f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/90fd4f88f588ae64038134f1eeaa023f-Abstract.html)

        **Abstract**:

        We provide the first coreset for clustering points in $\mathbb{R}^d$ that have multiple missing values (coordinates). Previous coreset constructions only allow one missing coordinate. The challenge in this setting is that objective functions, like \kMeans, are evaluated only on the set of available (non-missing) coordinates, which varies across points. Recall that an $\epsilon$-coreset of a large dataset is a small proxy, usually a reweighted subset of points, that $(1+\epsilon)$-approximates the clustering objective for every possible center set.Our coresets for $k$-Means and $k$-Median clustering have size $(jk)^{O(\min(j,k))} (\epsilon^{-1} d \log n)^2$, where $n$ is the number of data points, $d$ is the dimension and $j$ is the maximum number of missing coordinates for each data point. We further design an algorithm to construct these coresets in near-linear time, and consequently improve a recent quadratic-time PTAS for $k$-Means with missing values [Eiben et al., SODA 2021] to near-linear time.We validate our coreset construction, which is based on importance sampling and is easy to implement, on various real data sets. Our coreset exhibits a flexible tradeoff between coreset size and accuracy, and generally outperforms the uniform-sampling baseline. Furthermore, it significantly speeds up a Lloyd's-style heuristic for $k$-Means with missing values.

        ----

        ## [1328] Boosting with Multiple Sources

        **Authors**: *Corinna Cortes, Mehryar Mohri, Dmitry Storcheus, Ananda Theertha Suresh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9103820024efb30b451d006dc4ab3370-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9103820024efb30b451d006dc4ab3370-Abstract.html)

        **Abstract**:

        We study the problem of learning accurate ensemble predictors, in  particular boosting, in the presence of multiple source domains. We  show that the standard convex combination ensembles in general  cannot succeed in this scenario and adopt instead a domain-weighted  combination. We introduce and analyze a new boosting algorithm,  MULTIBOOST, for this scenario and show that it benefits from  favorable theoretical guarantees. We also report the results of  several experiments with our algorithm demonstrating that it  outperforms natural baselines on multi-source text-based,  image-based and tabular data. We further present an extension of our  algorithm to the federated learning scenario and report favorable  experimental results for that setting as well. Additionally, we  describe in detail an extension of our algorithm to the multi-class  setting, MCMULTIBOOST, for which we also report  experimental results.

        ----

        ## [1329] Dynamic Neural Representational Decoders for High-Resolution Semantic Segmentation

        **Authors**: *Bowen Zhang, Yifan Liu, Zhi Tian, Chunhua Shen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/912d2b1c7b2826caf99687388d2e8f7c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/912d2b1c7b2826caf99687388d2e8f7c-Abstract.html)

        **Abstract**:

        Semantic segmentation requires per-pixel prediction for a given image. Typically, the output resolution of a segmentation network is severely reduced due to the downsampling operations in the CNN backbone. Most previous methods employ upsampling decoders to recover the spatial resolution.Various decoders were designed in the literature. Here, we propose a novel decoder, termed dynamic neural representational decoder (NRD), which is simple yet significantly more efficient. As each location on the encoder's output corresponds to a local patch of the semantic labels, in this work, we represent these local patches of labels with compact neural networks. This neural representation enables our decoder to leverage the smoothness prior in the semantic label space, and thus makes our decoder more efficient. Furthermore, these neural representations are dynamically generated and conditioned on the outputs of the encoder networks. The desired semantic labels can be efficiently decoded from the neural representations, resulting in high-resolution semantic segmentation predictions.We empirically show that our proposed decoder can outperform the decoder in DeeplabV3+ with only $\sim$$30\%$ computational complexity, and achieve competitive performance with the methods using dilated encoders with only $\sim$$15\% $ computation. Experiments on Cityscapes, ADE20K, and Pascal Context demonstrate the effectiveness and efficiency of our proposed method.

        ----

        ## [1330] Dense Keypoints via Multiview Supervision

        **Authors**: *Zhixuan Yu, Haozheng Yu, Long Sha, Sujoy Ganguly, Hyun Soo Park*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/914101ec47c52b48a7b6ccc6f5a76f1f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/914101ec47c52b48a7b6ccc6f5a76f1f-Abstract.html)

        **Abstract**:

        This paper presents a new end-to-end semi-supervised framework to learn a dense keypoint detector using unlabeled multiview images. A key challenge lies in ﬁnding the exact correspondences between the dense keypoints in multiple views since the inverse of the keypoint mapping can be neither analytically derived nor differentiated. This limits applying existing multiview supervision approaches used to learn sparse keypoints that rely on the exact correspondences. To address this challenge, we derive a new probabilistic epipolar constraint that encodes the two desired properties. (1) Soft correspondence: we deﬁne a matchability, which measures a likelihood of a point matching to the other image’s corresponding point, thus relaxing the requirement of the exact correspondences. (2) Geometric consistency: every point in the continuous correspondence ﬁelds must satisfy the multiview consistency collectively. We formulate a probabilistic epipolar constraint using a weighted average of epipolar errors through the matchability thereby generalizing the point-to-point geometric error to the ﬁeld-to-ﬁeld geometric error. This generalization facilitates learning a geometrically coherent dense keypoint detection model by utilizing a large number of unlabeled multiview images. Additionally, to prevent degenerative cases, we employ a distillation-based regularization by using a pretrained model. Finally, we design a new neural network architecture, made of twin networks, that effectively minimizes the probabilistic epipolar errors of all possible correspondences between two view images by building afﬁnity matrices. Our method shows superior performance compared to existing methods, including non-differentiable bootstrapping in terms of keypoint accuracy, multiview consistency, and 3D reconstruction accuracy.

        ----

        ## [1331] Scatterbrain: Unifying Sparse and Low-rank Attention

        **Authors**: *Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, Christopher Ré*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9185f3ec501c674c7c788464a36e7fb3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9185f3ec501c674c7c788464a36e7fb3-Abstract.html)

        **Abstract**:

        Recent advances in efficient Transformers have exploited either the sparsity or low-rank properties of attention matrices to reduce the computational and memory bottlenecks of modeling long sequences. However, it is still challenging to balance the trade-off between model quality and efficiency to perform a one-size-fits-all approximation for different tasks. To better understand this trade-off, we observe that sparse and low-rank approximations excel in different regimes, determined by the softmax temperature in attention, and sparse + low-rank can outperform each individually. Inspired by the classical robust-PCA algorithm for sparse and low-rank decomposition, we propose Scatterbrain, a novel way to unify sparse (via locality sensitive hashing) and low-rank (via kernel feature map) attention for accurate and efficient approximation. The estimation is unbiased with provably low error. We empirically show that Scatterbrain can achieve $2.1 \times$ lower error than baselines when serving as a drop-in replacement in BigGAN image generation and pre-trained T2T-ViT. On a pre-trained T2T Vision transformer, even without fine-tuning, Scatterbrain can reduce $98\%$ of attention memory at the cost of only $1\%$ drop in accuracy. We demonstrate Scatterbrain for end-to-end training with up to $4$ points better perplexity and 5 points better average accuracy than sparse or low-rank efficient transformers on language modeling and long-range-arena tasks.

        ----

        ## [1332] PTR: A Benchmark for Part-based Conceptual, Relational, and Physical Reasoning

        **Authors**: *Yining Hong, Li Yi, Josh Tenenbaum, Antonio Torralba, Chuang Gan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/918f5cd5a5c0d48671d4d4fc54bab2e9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/918f5cd5a5c0d48671d4d4fc54bab2e9-Abstract.html)

        **Abstract**:

        A critical aspect of human visual perception is the ability to parse visual scenes into individual objects and further into object parts, forming part-whole hierarchies. Such composite structures could induce a rich set of semantic concepts and relations, thus playing an important role in the interpretation and organization of visual signals as well as for the generalization of visual perception and reasoning. However, existing visual reasoning benchmarks mostly focus on objects rather than parts. Visual reasoning based on the full part-whole hierarchy is much more challenging than object-centric reasoning due to finer-grained concepts, richer geometry relations, and more complex physics. Therefore, to better serve for part-based conceptual, relational and physical reasoning, we introduce a new large-scale diagnostic visual reasoning dataset named PTR. PTR contains around 80k RGBD synthetic images with ground truth object and part level annotations regarding semantic instance segmentation, color attributes, spatial and geometric relationships, and certain physical properties such as stability. These images are paired with 800k machine-generated questions covering various types of reasoning types, making them a good testbed for visual reasoning models. We examine several state-of-the-art visual reasoning models on this dataset and observe that they still make many surprising mistakes in situations where humans can easily infer the correct answer. We believe this dataset will open up new opportunities for part-based reasoning. PTR dataset and baseline models are publicly available.

        ----

        ## [1333] Property-Aware Relation Networks for Few-Shot Molecular Property Prediction

        **Authors**: *Yaqing Wang, Abulikemu Abuduweili, Quanming Yao, Dejing Dou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/91bc333f6967019ac47b49ca0f2fa757-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/91bc333f6967019ac47b49ca0f2fa757-Abstract.html)

        **Abstract**:

        Molecular property prediction plays a fundamental role in drug discovery to identify candidate molecules with target properties. However, molecular property prediction is essentially a few-shot problem, which makes it hard to use regular machine learning models. In this paper, we propose Property-Aware Relation networks (PAR) to handle this problem. In comparison to existing works, we leverage the fact that both relevant substructures and relationships among molecules change across different molecular properties. We first introduce a property-aware embedding function to transform the generic molecular embeddings to substructure-aware space relevant to the target property.  Further, we design an adaptive relation graph learning module to jointly estimate molecular relation graph and refine molecular embeddings w.r.t. the target property, such that the limited labels can be effectively propagated among similar molecules. We adopt a meta-learning strategy where the parameters are selectively updated within tasks in order to model generic and property-aware knowledge separately. Extensive experiments on benchmark molecular property prediction datasets show that PAR consistently outperforms existing methods and can obtain property-aware molecular embeddings and model molecular relation graph properly.

        ----

        ## [1334] Differentially Private Learning with Adaptive Clipping

        **Authors**: *Galen Andrew, Om Thakkar, Brendan McMahan, Swaroop Ramaswamy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/91cff01af640a24e7f9f7a5ab407889f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/91cff01af640a24e7f9f7a5ab407889f-Abstract.html)

        **Abstract**:

        Existing approaches for training neural networks with user-level differential privacy (e.g., DP Federated Averaging) in federated learning (FL) settings involve bounding the contribution of each user's model update by {\em clipping} it to some constant value. However there is no good {\em a priori} setting of the clipping norm across tasks and learning settings: the update norm distribution depends on the model architecture and loss, the amount of data on each device, the client learning rate, and possibly various other parameters. We propose a method wherein instead of a fixed clipping norm, one clips to a value at a specified quantile of the update norm distribution, where the value at the quantile is itself estimated online, with differential privacy. The method tracks the quantile closely, uses a negligible amount of privacy budget, is compatible with other federated learning technologies such as compression and secure aggregation, and has a straightforward joint DP analysis with DP-FedAvg. Experiments demonstrate that adaptive clipping to the median update norm works well across a range of federated learning tasks, eliminating the need to tune any clipping hyperparameter.

        ----

        ## [1335] Can Less be More? When Increasing-to-Balancing Label Noise Rates Considered Beneficial

        **Authors**: *Yang Liu, Jialu Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/91e50fe1e39af2869d3336eaaeebdb43-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/91e50fe1e39af2869d3336eaaeebdb43-Abstract.html)

        **Abstract**:

        In this paper, we answer the question of when inserting label noise (less informative labels) can instead return us more accurate and fair models. We are primarily inspired by three observations: 1) In contrast to reducing label noise rates, increasing the noise rates is easy to implement; 2) Increasing a certain class of instances' label noise to balance the noise rates (increasing-to-balancing) results in an easier learning problem; 3) Increasing-to-balancing improves fairness guarantees against label bias. In this paper, we first quantify the trade-offs introduced by increasing a certain group of instances' label noise rate w.r.t. the loss of label informativeness and the lowered learning difficulties. We analytically demonstrate when such an increase is beneficial, in terms of either improved generalization power or the fairness guarantees. Then we present a method to insert label noise properly for the task of learning with noisy labels, either without or with a fairness constraint. The primary technical challenge we face is due to the fact that we would not know which data instances are suffering from higher noise, and we would not have the ground truth labels to verify any possible hypothesis. We propose a detection method that informs us which group of labels might suffer from higher noise without using ground truth labels. We formally establish the effectiveness of the proposed solution and demonstrate it with extensive experiments.

        ----

        ## [1336] Projected GANs Converge Faster

        **Authors**: *Axel Sauer, Kashyap Chitta, Jens Müller, Andreas Geiger*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9219adc5c42107c4911e249155320648-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9219adc5c42107c4911e249155320648-Abstract.html)

        **Abstract**:

        Generative Adversarial Networks (GANs) produce high-quality images but are challenging to train. They need careful regularization, vast amounts of compute, and expensive hyper-parameter sweeps. We make significant headway on these issues by projecting generated and real samples into a fixed, pretrained feature space. Motivated by the finding that the discriminator cannot fully exploit features from deeper layers of the pretrained model, we propose a more effective strategy that mixes features across channels and resolutions. Our Projected GAN improves image quality, sample efficiency, and convergence speed. It is further compatible with resolutions of up to one Megapixel and advances the state-of-the-art Fréchet Inception Distance (FID) on twenty-two benchmark datasets. Importantly, Projected GANs match the previously lowest FIDs up to 40 times faster, cutting the wall-clock time from 5 days to less than 3 hours given the same computational resources.

        ----

        ## [1337] Generating High-Quality Explanations for Navigation in Partially-Revealed Environments

        **Authors**: *Gregory J. Stein*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/926ec030f29f83ce5318754fdb631a33-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/926ec030f29f83ce5318754fdb631a33-Abstract.html)

        **Abstract**:

        We present an approach for generating natural language explanations of high-level behavior of autonomous agents navigating in partially-revealed environments. Our counterfactual explanations communicate changes to interpratable statistics of the belief (e.g., the likelihood an exploratory action will reach the unseen goal) that are estimated from visual input via a deep neural network and used (via a Bellman equation variant) to inform planning far into the future. Additionally, our novel training procedure mimics explanation generation, allowing us to use planning performance as an objective measure of explanation quality. Simulated experiments validate that our explanations are both high quality and can be used in interventions to directly correct bad behavior; agents trained via our training-by-explaining procedure achieve 9.1% lower average cost than a non-learned baseline (12.7% after interventions) in environments derived from real-world floor plans.

        ----

        ## [1338] De-randomizing MCMC dynamics with the diffusion Stein operator

        **Authors**: *Zheyang Shen, Markus Heinonen, Samuel Kaski*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9271905e840548b8cada6d60c0cfd93b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9271905e840548b8cada6d60c0cfd93b-Abstract.html)

        **Abstract**:

        Approximate Bayesian inference estimates descriptors of an intractable target distribution - in essence, an optimization problem within a family of distributions. For example, Langevin dynamics (LD) extracts asymptotically exact samples from a diffusion process because the time evolution of its marginal distributions constitutes a curve that minimizes the KL-divergence via steepest descent in the Wasserstein space. Parallel to LD, Stein variational gradient descent (SVGD) similarly minimizes the KL, albeit endowed with a novel Stein-Wasserstein distance, by deterministically transporting a set of particle samples, thus de-randomizes the stochastic diffusion process. We propose de-randomized kernel-based particle samplers to all diffusion-based samplers known as MCMC dynamics. Following previous work in interpreting MCMC dynamics, we equip the Stein-Wasserstein space with a fiber-Riemannian Poisson structure, with the capacity of characterizing a fiber-gradient Hamiltonian flow that simulates MCMC dynamics. Such dynamics discretizes into generalized SVGD (GSVGD), a Stein-type deterministic particle sampler, with particle updates coinciding with applying the diffusion Stein operator to a kernel function. We demonstrate empirically that GSVGD can de-randomize complex MCMC dynamics, which combine the advantages of auxiliary momentum variables and Riemannian structure, while maintaining the high sample quality from an interacting particle system.

        ----

        ## [1339] Sparsely Changing Latent States for Prediction and Planning in Partially Observable Domains

        **Authors**: *Christian Gumbsch, Martin V. Butz, Georg Martius*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/927b028cfa24b23a09ff20c1a7f9b398-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/927b028cfa24b23a09ff20c1a7f9b398-Abstract.html)

        **Abstract**:

        A common approach to prediction and planning in partially observable domains is to use recurrent neural networks (RNNs), which ideally develop and maintain a latent memory about hidden, task-relevant factors. We hypothesize that many of these hidden factors in the physical world are constant over time, changing only sparsely. To study this hypothesis, we propose Gated $L_0$ Regularized Dynamics (GateL0RD), a novel recurrent architecture that incorporates the inductive bias to maintain stable, sparsely changing latent states.  The bias is implemented by means of a novel internal gating function and a penalty on the $L_0$ norm of latent state changes. We demonstrate that GateL0RD can compete with or outperform state-of-the-art RNNs in a variety of partially observable prediction and control tasks. GateL0RD tends to encode the underlying generative factors of the environment, ignores spurious temporal dependencies, and generalizes better, improving sampling efficiency and overall performance in model-based planning and reinforcement learning tasks. Moreover, we show that the developing latent states can be easily interpreted, which is a step towards better explainability in RNNs.

        ----

        ## [1340] PreferenceNet: Encoding Human Preferences in Auction Design with Deep Learning

        **Authors**: *Neehar Peri, Michael J. Curry, Samuel Dooley, John Dickerson*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/92977ae4d2ba21425a59afb269c2a14e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/92977ae4d2ba21425a59afb269c2a14e-Abstract.html)

        **Abstract**:

        The design of optimal auctions is a problem of interest in economics, game theory and computer science. Despite decades of effort, strategyproof, revenue-maximizing auction designs are still not known outside of restricted settings. However, recent methods using deep learning have shown some success in approximating optimal auctions, recovering several known solutions and outperforming strong baselines when optimal auctions are not known. In addition to maximizing revenue, auction mechanisms may also seek to encourage socially desirable constraints such as allocation fairness or diversity. However, these philosophical notions neither have standardization nor do they have widely accepted formal definitions. In this paper, we propose PreferenceNet, an extension of existing neural-network-based auction mechanisms to encode constraints using (potentially human-provided) exemplars of desirable allocations. In addition, we introduce a new metric to evaluate an auction allocations' adherence to such socially desirable constraints and demonstrate that our proposed method is competitive with current state-of-the-art neural-network based auction designs. We validate our approach through human subject research and show that we are able to effectively capture real human preferences.

        ----

        ## [1341] Large-Scale Learning with Fourier Features and Tensor Decompositions

        **Authors**: *Frederiek Wesel, Kim Batselier*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/92a08bf918f44ccd961477be30023da1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/92a08bf918f44ccd961477be30023da1-Abstract.html)

        **Abstract**:

        Random Fourier features provide a way to tackle large-scale machine learning problems with kernel methods. Their slow Monte Carlo convergence rate has motivated the research of deterministic Fourier features whose approximation error can decrease exponentially in the number of basis functions. However, due to their tensor product extension to multiple dimensions, these methods suffer heavily from the curse of dimensionality, limiting their applicability to one, two or three-dimensional scenarios. In our approach we overcome said curse of dimensionality by exploiting the tensor product structure of deterministic Fourier features, which enables us to represent the model parameters as a low-rank tensor decomposition. We derive a monotonically converging block coordinate descent algorithm with linear complexity in both the sample size and the dimensionality of the inputs for a regularized squared loss function, allowing to learn a parsimonious model in decomposed form using deterministic Fourier features.We demonstrate by means of numerical experiments how our low-rank tensor approach obtains the same performance of the corresponding nonparametric model, consistently outperforming random Fourier features.

        ----

        ## [1342] Hash Layers For Large Sparse Models

        **Authors**: *Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/92bf5e6240737e0326ea59846a83e076-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/92bf5e6240737e0326ea59846a83e076-Abstract.html)

        **Abstract**:

        We investigate the training of sparse layers that use different parameters for different inputs based on hashing in large Transformer models. Specifically, we modify the feedforward layer to hash to different sets of weights depending on the current token, over all tokens in the sequence. We show that this procedure either outperforms or is competitive with learning-to-route mixture-of-expert methods such as Switch Transformers and BASE Layers, while requiring no routing parameters or extra terms in the objective function such as a load balancing loss, and no sophisticated assignment algorithm. We study the performance of different hashing techniques,  hash sizes and input features,  and  show that  balanced and random hashes focused on the most local features work best, compared to either learning clusters or using longer-range context. We show our approach works well both on large language modeling and dialogue tasks, and on downstream fine-tuning tasks.

        ----

        ## [1343] Sliced Mutual Information: A Scalable Measure of Statistical Dependence

        **Authors**: *Ziv Goldfeld, Kristjan H. Greenewald*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/92c4661685bf6681f6a33b78ef729658-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/92c4661685bf6681f6a33b78ef729658-Abstract.html)

        **Abstract**:

        Mutual information (MI) is a fundamental measure of statistical dependence, with a myriad of applications to information theory, statistics, and machine learning. While it possesses many desirable structural properties, the estimation of high-dimensional MI from samples suffers from the curse of dimensionality. Motivated by statistical scalability to high dimensions, this paper proposes sliced MI (SMI) as a surrogate measure of dependence. SMI is defined as an average of MI terms between one-dimensional random projections. We show that it preserves many of the structural properties of classic MI, while gaining scalable computation and efficient estimation from samples. Furthermore, and in contrast to classic MI, SMI can grow as a result of deterministic transformations. This enables leveraging SMI for feature extraction by optimizing it over processing functions of raw data to identify useful representations thereof. Our theory is supported by numerical studies of independence testing and feature extraction, which demonstrate the potential gains SMI offers over classic MI for high-dimensional inference.

        ----

        ## [1344] Emergent Communication under Varying Sizes and Connectivities

        **Authors**: *Jooyeon Kim, Alice Oh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/92dfa194391a59dc65b88b704599dbd6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/92dfa194391a59dc65b88b704599dbd6-Abstract.html)

        **Abstract**:

        Recent advances in deep neural networks allowed artificial agents to derive their own emergent languages that promote interaction, coordination, and collaboration within a group. Just as we humans have succeeded in creating a shared language that allows us to interact within a large group, can the emergent communication within an artificial group converge to a shared, agreed language? This research provides an analytical study of the shared emergent language within the group communication settings of different sizes and connectivities. As the group size increases up to hundreds, agents start to speak dissimilar languages, but the rate at which they successfully communicate is maintained. We observe the emergence of different dialects when we restrict the group communication to have local connectivities only. Finally, we provide optimization results of group communication graphs when the number of agents one can communicate with is restricted or when we penalize communication between distant agent pairs. The optimized communication graphs show superior communication success rates compared to graphs with same number of links as well as the emergence of hub nodes and scale-free networks.

        ----

        ## [1345] Deep Bandits Show-Off: Simple and Efficient Exploration with Deep Networks

        **Authors**: *Rong Zhu, Mattia Rigotti*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/92fde850d824c2ba9b563cb6fa4078c3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/92fde850d824c2ba9b563cb6fa4078c3-Abstract.html)

        **Abstract**:

        Designing efficient exploration is central to Reinforcement Learning due to the fundamental problem posed by the exploration-exploitation dilemma.  Bayesian exploration strategies like Thompson Sampling resolve this trade-off in a principled way by modeling and updating the distribution of the parameters of the action-value function, the outcome model of the environment.However, this technique becomes infeasible for complex environments due to the computational intractability of maintaining probability distributions over parameters of outcome models of corresponding complexity.Moreover, the approximation techniques introduced to mitigate this issue typically result in poor exploration-exploitation trade-offs, as observed in the case of deep neural network models with approximate posterior methods that have been shown to underperform in the deep bandit scenario.In this paper we introduce Sample Average Uncertainty (SAU), a simple and efficient uncertainty measure for contextual bandits.While Bayesian approaches like Thompson Sampling estimate outcomes uncertainty indirectly by first quantifying the variability over the parameters of the outcome model, SAU is a frequentist approach that directly estimates the uncertainty of the outcomes based on the value predictions.Importantly, we show theoretically that the uncertainty measure estimated by SAU asymptotically matches the uncertainty provided by Thompson Sampling, as well as its regret bounds.Because of its simplicity SAU can be seamlessly applied to deep contextual bandits as a very scalable drop-in replacement for epsilon-greedy exploration.We confirm empirically our theory by showing that SAU-based exploration outperforms current state-of-the-art deep Bayesian bandit methods on several real-world datasets at modest computation cost, and make the code to reproduce our results available at \url{https://github.com/ibm/sau-explore}.

        ----

        ## [1346] Regret Minimization Experience Replay in Off-Policy Reinforcement Learning

        **Authors**: *Xu-Hui Liu, Zhenghai Xue, Jing-Cheng Pang, Shengyi Jiang, Feng Xu, Yang Yu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/931af583573227f0220bc568c65ce104-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/931af583573227f0220bc568c65ce104-Abstract.html)

        **Abstract**:

        In reinforcement learning, experience replay stores past samples for further reuse. Prioritized sampling is a promising technique to better utilize these samples. Previous criteria of prioritization include TD error, recentness and corrective feedback, which are mostly heuristically designed. In this work, we start from the regret minimization objective, and obtain an optimal prioritization strategy for Bellman update that can directly maximize the return of the policy. The theory suggests that data with higher hindsight TD error, better on-policiness and more accurate Q value should be assigned with higher weights during sampling. Thus most previous criteria only consider this strategy partially. We not only provide theoretical justifications for previous criteria, but also propose two new methods to compute the prioritization weight, namely ReMERN and ReMERT. ReMERN learns an error network, while ReMERT exploits the temporal ordering of states. Both methods outperform previous prioritized sampling algorithms in challenging RL benchmarks, including MuJoCo, Atari and Meta-World.

        ----

        ## [1347] Relative Uncertainty Learning for Facial Expression Recognition

        **Authors**: *Yuhang Zhang, Chengrui Wang, Weihong Deng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9332c513ef44b682e9347822c2e457ac-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9332c513ef44b682e9347822c2e457ac-Abstract.html)

        **Abstract**:

        In facial expression recognition (FER), the uncertainties introduced by inherent noises like ambiguous facial expressions and inconsistent labels raise concerns about the credibility of recognition results. To quantify these uncertainties and achieve good performance under noisy data, we regard uncertainty as a relative concept and propose an innovative uncertainty learning method called Relative Uncertainty Learning (RUL). Rather than assuming Gaussian uncertainty distributions for all datasets, RUL builds an extra branch to learn uncertainty from the relative difficulty of samples by feature mixup. Specifically, we use uncertainties as weights to mix facial features and design an add-up loss to encourage uncertainty learning. It is easy to implement and adds little or no extra computation overhead. Extensive experiments show that RUL outperforms state-of-the-art FER uncertainty learning methods in both real-world and synthetic noisy FER datasets. Besides, RUL also works well on other datasets such as CIFAR and Tiny ImageNet. The code is available at https://github.com/zyh-uaiaaaa/Relative-Uncertainty-Learning.

        ----

        ## [1348] An Information-theoretic Approach to Distribution Shifts

        **Authors**: *Marco Federici, Ryota Tomioka, Patrick Forré*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/93661c10ed346f9692f4d512319799b3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/93661c10ed346f9692f4d512319799b3-Abstract.html)

        **Abstract**:

        Safely deploying machine learning models to the real world is often a challenging process. For example, models trained with data obtained from a specific geographic location tend to fail when queried with data obtained elsewhere, agents trained in a simulation can struggle to adapt when deployed in the real world or novel environments, and neural networks that are fit to a subset of the population might carry some selection bias into their decision process.In this work, we describe the problem of data shift from an information-theoretic perspective by (i) identifying and describing the different sources of error, (ii) comparing some of the most promising objectives explored in the recent domain generalization and fair classification literature. From our theoretical analysis and empirical evaluation, we conclude that the model selection procedure needs to be guided by careful considerations regarding the observed data, the factors used for correction, and the structure of the data-generating process.

        ----

        ## [1349] TRS: Transferability Reduced Ensemble via Promoting Gradient Diversity and Model Smoothness

        **Authors**: *Zhuolin Yang, Linyi Li, Xiaojun Xu, Shiliang Zuo, Qian Chen, Pan Zhou, Benjamin I. P. Rubinstein, Ce Zhang, Bo Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/937936029af671cf479fa893db91cbdd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/937936029af671cf479fa893db91cbdd-Abstract.html)

        **Abstract**:

        Adversarial Transferability is an intriguing property - adversarial perturbation crafted against one model is also effective against another model, while these models are from different model families or training processes. To better protect ML systems against adversarial attacks, several questions are raised: what are the sufficient conditions for adversarial transferability, and how to bound it? Is there a way to reduce the adversarial transferability in order to improve the robustness of an ensemble ML model? To answer these questions, in this work we first theoretically analyze and outline sufficient conditions for adversarial transferability between models; then propose a practical algorithm to reduce the transferability between base models within an ensemble to improve its robustness. Our theoretical analysis shows that only promoting the orthogonality between gradients of base models is not enough to ensure low transferability; in the meantime, the model smoothness is an important factor to control the transferability. We also provide the lower and upper bounds of adversarial transferability under certain conditions. Inspired by our theoretical analysis, we propose an effective Transferability Reduced Smooth (TRS) ensemble training strategy to train a robust ensemble with low transferability by enforcing both gradient orthogonality and model smoothness between base models. We conduct extensive experiments on TRS and compare with 6 state-of-the-art ensemble baselines against 8 whitebox attacks on different datasets, demonstrating that the proposed TRS outperforms all baselines significantly.

        ----

        ## [1350] Towards Sample-Optimal Compressive Phase Retrieval with Sparse and Generative Priors

        **Authors**: *Zhaoqiang Liu, Subhroshekhar Ghosh, Jonathan Scarlett*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/939314105ce8701e67489642ef4d49e8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/939314105ce8701e67489642ef4d49e8-Abstract.html)

        **Abstract**:

        Compressive phase retrieval is a popular variant of the standard compressive sensing problem in which the measurements only contain magnitude information. In this paper, motivated by recent advances in deep generative models, we provide recovery guarantees with near-optimal sample complexity for phase retrieval with generative priors. We first show that when using i.i.d. Gaussian measurements and an $L$-Lipschitz continuous generative model with bounded $k$-dimensional inputs, roughly $O(k \log L)$ samples suffice to guarantee that any signal minimizing an amplitude-based empirical loss function is close to the true signal. Attaining this sample complexity with a practical algorithm remains a difficult challenge, and finding a good initialization for gradient-based methods has been observed to pose a major bottleneck. To partially address this, we further show that roughly $O(k \log L)$ samples ensure sufficient closeness between the underlying signal and any {\em globally optimal} solution to an optimization problem designed for spectral initialization (though finding such a solution may still be challenging). We also adapt this result to sparse phase retrieval, and show that $O(s \log n)$ samples are sufficient for a similar guarantee when the underlying signal is $s$-sparse and $n$-dimensional, matching an information-theoretic lower bound. While these guarantees do not directly correspond to a practical algorithm, we propose a practical spectral initialization method motivated by our findings, and experimentally observe performance gains over various existing spectral initialization methods for sparse phase retrieval.

        ----

        ## [1351] Moser Flow: Divergence-based Generative Modeling on Manifolds

        **Authors**: *Noam Rozen, Aditya Grover, Maximilian Nickel, Yaron Lipman*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/93a27b0bd99bac3e68a440b48aa421ab-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/93a27b0bd99bac3e68a440b48aa421ab-Abstract.html)

        **Abstract**:

        We are interested in learning generative models for complex geometries described via manifolds, such as spheres, tori, and other implicit surfaces. Current extensions of existing (Euclidean) generative models are restricted to specific geometries and typically suffer from high computational costs. We introduce Moser Flow (MF), a new class of generative models within the family of continuous normalizing flows (CNF). MF also produces a CNF via a solution to the change-of-variable formula, however differently from other CNF methods, its model (learned) density is parameterized as the source (prior) density minus the divergence of a neural network (NN). The divergence is a local, linear differential operator, easy to approximate and calculate on manifolds. Therefore, unlike other CNFs, MF does not require invoking or backpropagating through an ODE solver during training. Furthermore, representing the model density explicitly as the divergence of a NN rather than as a solution of an ODE facilitates learning high fidelity densities. Theoretically, we prove that MF constitutes a universal density approximator under suitable assumptions. Empirically, we demonstrate for the first time the use of flow models for sampling from general curved surfaces and achieve significant improvements in density estimation, sample quality, and training complexity over existing CNFs on challenging synthetic geometries and real-world benchmarks from the earth and climate sciences.

        ----

        ## [1352] Structure-Aware Random Fourier Kernel for Graphs

        **Authors**: *Jinyuan Fang, Qiang Zhang, Zaiqiao Meng, Shangsong Liang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/93da579a65ce84cd1d4c85c2cbb84fc5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/93da579a65ce84cd1d4c85c2cbb84fc5-Abstract.html)

        **Abstract**:

        Gaussian Processes (GPs) define distributions over functions and their generalization capabilities depend heavily on the choice of kernels. In this paper, we propose a novel structure-aware random Fourier (SRF) kernel for GPs that brings several benefits when modeling graph-structured data. First, SRF kernel is defined with a spectral distribution based on the Fourier duality given by the Bochner's theorem, transforming the kernel learning problem to a distribution inference problem. Second, SRF kernel admits a random Fourier feature formulation that makes the kernel scalable for optimization. Third, SRF kernel enables to leverage geometric structures by taking subgraphs as inputs. To effectively optimize GPs with SRF kernel, we develop a variational EM algorithm, which alternates between an inference procedure (E-step) and a learning procedure (M-step). Experimental results on five real-world datasets show that our model can achieve state-of-the-art performance in two typical graph learning tasks, i.e., object classification and link prediction.

        ----

        ## [1353] Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling

        **Authors**: *Valentin De Bortoli, James Thornton, Jeremy Heng, Arnaud Doucet*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html)

        **Abstract**:

        Progressively applying Gaussian noise transforms complex data distributions to approximately Gaussian. Reversing this dynamic defines a generative model. When the forward noising process is given by a Stochastic Differential Equation (SDE), Song et al (2021) demonstrate how the time inhomogeneous drift of the associated reverse-time SDE may be estimated using score-matching. A limitation of this approach is that the forward-time SDE must be run for a sufficiently long time for the final distribution to be approximately Gaussian. In contrast, solving the Schrödinger Bridge (SB) problem, i.e. an entropy-regularized optimal transport problem on path spaces, yields diffusions which generate samples from the data distribution in finite time. We present Diffusion SB (DSB), an original approximation of the Iterative Proportional Fitting (IPF) procedure to solve the SB problem, and provide theoretical analysis along with generative modeling experiments. The first DSB iteration recovers the methodology proposed by Song et al. (2021), with the flexibility of using shorter time intervals, as subsequent DSB iterations reduce the discrepancy between the final-time marginal of the forward (resp. backward) SDE with respect to the prior (resp. data) distribution. Beyond generative modeling, DSB offers a widely applicable computational optimal transport tool as the continuous state-space analogue of the popular Sinkhorn algorithm (Cuturi, 2013).

        ----

        ## [1354] Improving Transferability of Representations via Augmentation-Aware Self-Supervision

        **Authors**: *Hankook Lee, Kibok Lee, Kimin Lee, Honglak Lee, Jinwoo Shin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/94130ea17023c4837f0dcdda95034b65-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/94130ea17023c4837f0dcdda95034b65-Abstract.html)

        **Abstract**:

        Recent unsupervised representation learning methods have shown to be effective in a range of vision tasks by learning representations invariant to data augmentations such as random cropping and color jittering. However, such invariance could be harmful to downstream tasks if they rely on the characteristics of the data augmentations, e.g., location- or color-sensitive. This is not an issue just for unsupervised learning; we found that this occurs even in supervised learning because it also learns to predict the same label for all augmented samples of an instance. To avoid such failures and obtain more generalizable representations, we suggest to optimize an auxiliary self-supervised loss, coined AugSelf, that learns the difference of augmentation parameters (e.g., cropping positions, color adjustment intensities) between two randomly augmented samples. Our intuition is that AugSelf encourages to preserve augmentation-aware information in learned representations, which could be beneficial for their transferability. Furthermore, AugSelf can easily be incorporated into recent state-of-the-art representation learning methods with a negligible additional training cost. Extensive experiments demonstrate that our simple idea consistently improves the transferability of representations learned by supervised and unsupervised methods in various transfer learning scenarios. The code is available at https://github.com/hankook/AugSelf.

        ----

        ## [1355] Long-Short Transformer: Efficient Transformers for Language and Vision

        **Authors**: *Chen Zhu, Wei Ping, Chaowei Xiao, Mohammad Shoeybi, Tom Goldstein, Anima Anandkumar, Bryan Catanzaro*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9425be43ba92c2b4454ca7bf602efad8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9425be43ba92c2b4454ca7bf602efad8-Abstract.html)

        **Abstract**:

        Transformers have achieved success in both language and vision domains. However, it is prohibitively expensive to scale them to long sequences such as long documents or high-resolution images, because self-attention mechanism has quadratic time and memory complexities with respect to the input sequence length. In this paper, we propose Long-Short Transformer (Transformer-LS), an efficient self-attention mechanism for modeling long sequences with linear complexity for both language and vision tasks. It aggregates a novel long-range attention with dynamic projection to model distant correlations and a short-term attention to capture fine-grained local correlations. We propose a dual normalization strategy to account for the scale mismatch between the two attention mechanisms. Transformer-LS can be applied to both autoregressive and bidirectional models without additional complexity. Our method outperforms the state-of-the-art models on multiple tasks in language and vision domains, including the Long Range Arena benchmark, autoregressive language modeling, and ImageNet classification. For instance, Transformer-LS achieves 0.97 test BPC on enwik8 using half the number of parameters than previous method, while being faster and is able to handle 3x as long sequences compared to its full-attention version on the same hardware. On ImageNet, it can obtain the state-of-the-art results (e.g., a moderate size of 55.8M model solely trained on 224x224 ImageNet-1K can obtain Top-1 accuracy 84.1%), while being more scalable on high-resolution images. The source code and models are released at https://github.com/NVIDIA/transformer-ls.

        ----

        ## [1356] Post-Training Sparsity-Aware Quantization

        **Authors**: *Gil Shomron, Freddy Gabbay, Samer Kurzum, Uri C. Weiser*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9431c87f273e507e6040fcb07dcb4509-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9431c87f273e507e6040fcb07dcb4509-Abstract.html)

        **Abstract**:

        Quantization is a technique used in deep neural networks (DNNs) to increase execution performance and hardware efficiency. Uniform post-training quantization (PTQ) methods are common, since they can be implemented efficiently in hardware and do not require extensive hardware resources or a training set. Mapping FP32 models to INT8 using uniform PTQ yields models with negligible accuracy degradation; however, reducing precision below 8 bits with PTQ is challenging, as accuracy degradation becomes noticeable, due to the increase in quantization noise. In this paper, we propose a sparsity-aware quantization (SPARQ) method, in which the unstructured and dynamic activation sparsity is leveraged in different representation granularities. 4-bit quantization, for example, is employed by dynamically examining the bits of 8-bit values and choosing a window of 4 bits, while first skipping zero-value bits. Moreover, instead of quantizing activation-by-activation to 4 bits, we focus on pairs of 8-bit activations and examine whether one of the two is equal to zero. If one is equal to zero, the second can opportunistically use the other's 4-bit budget; if both do not equal zero, then each is dynamically quantized to 4 bits, as described. SPARQ achieves minor accuracy degradation and a practical hardware implementation.

        ----

        ## [1357] The Implicit Bias of Minima Stability: A View from Function Space

        **Authors**: *Rotem Mulayoff, Tomer Michaeli, Daniel Soudry*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/944a5ae3483ed5c1e10bbccb7942a279-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/944a5ae3483ed5c1e10bbccb7942a279-Abstract.html)

        **Abstract**:

        The loss terrains of over-parameterized neural networks have multiple global minima. However, it is well known that stochastic gradient descent (SGD) can stably converge only to minima that are sufficiently flat w.r.t. SGD's step size. In this paper we study the effect that this mechanism has on the function implemented by the trained model. First, we extend the existing knowledge on minima stability to non-differentiable minima, which are common in ReLU nets. We then use our stability results to study a single hidden layer univariate ReLU network. In this setting, we show that SGD is biased towards functions whose second derivative (w.r.t the input) has a bounded weighted $L_1$ norm, and this is regardless of the initialization. In particular, we show that the function implemented by the network upon convergence gets smoother as the learning rate increases. The weight multiplying the second derivative is larger around the center of the support of the training distribution, and smaller towards its boundaries, suggesting that a trained model tends to be smoother at the center of the training distribution.

        ----

        ## [1358] Breaking the Sample Complexity Barrier to Regret-Optimal Model-Free Reinforcement Learning

        **Authors**: *Gen Li, Laixi Shi, Yuxin Chen, Yuantao Gu, Yuejie Chi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/94739e5a5164b4d2396e253a11d57044-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/94739e5a5164b4d2396e253a11d57044-Abstract.html)

        **Abstract**:

        Achieving sample efficiency in online episodic reinforcement learning (RL) requires optimally balancing  exploration and exploitation. When it comes to a finite-horizon episodic Markov decision process with $S$ states, $A$ actions and horizon length $H$, substantial progress has been achieved towards characterizing the minimax-optimal regret, which scales on the order of $\sqrt{H^2SAT}$ (modulo log factors) with $T$ the total number of samples. While several competing solution paradigms have been proposed to minimize regret, they are either memory-inefficient, or fall short of optimality unless the sample size exceeds an enormous threshold (e.g., $S^6A^4 \,\mathrm{poly}(H)$ for existing model-free methods).To overcome such a large sample size barrier to efficient RL, we design a novel model-free algorithm, with space complexity $O(SAH)$, that achieves near-optimal regret as soon as the sample size exceeds the order of $SA\,\mathrm{poly}(H)$. In terms of this sample size requirement (also referred to the initial burn-in cost), our method improves --- by at least a factor of $S^5A^3$ --- upon any prior memory-efficient algorithm that is asymptotically regret-optimal. Leveraging the recently introduced variance reduction strategy (also called {\em reference-advantage decomposition}), the proposed algorithm employs an {\em early-settled}  reference update rule, with the aid of two Q-learning sequences with upper and lower confidence bounds. The design principle of our early-settled variance reduction method might be of independent interest to other RL settings that involve intricate exploration-exploitation trade-offs.

        ----

        ## [1359] Robust Auction Design in the Auto-bidding World

        **Authors**: *Santiago R. Balseiro, Yuan Deng, Jieming Mao, Vahab S. Mirrokni, Song Zuo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/948f847055c6bf156997ce9fb59919be-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/948f847055c6bf156997ce9fb59919be-Abstract.html)

        **Abstract**:

        In classic auction theory, reserve prices are known to be effective for improving revenue for the auctioneer against quasi-linear utility maximizing bidders. The introduction of reserve prices, however, usually do not help improve total welfare of the auctioneer and the bidders. In this paper, we focus on value maximizing bidders with return on spend constraints---a paradigm that has drawn considerable attention recently as more advertisers adopt auto-bidding algorithms in advertising platforms---and show that the introduction of reserve prices has a novel impact on the market. Namely, by choosing reserve prices appropriately the auctioneer can improve not only the total revenue but also the total welfare. Our results also demonstrate that reserve prices are robust to bidder types, i.e., reserve prices work well for different bidder types, such as value maximizers and utility maximizers, without using bidder type information. We generalize these results for a variety of auction mechanisms such as VCG, GSP, and first-price auctions. Moreover, we show how to combine these results with additive boosts to improve the welfare of the outcomes of the auction further. Finally, we complement our theoretical observations with an empirical study confirming the effectiveness of these ideas using data from online advertising auctions.

        ----

        ## [1360] Weighted model estimation for offline model-based reinforcement learning

        **Authors**: *Toru Hishinuma, Kei Senda*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/949694a5059302e7283073b502f094d7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/949694a5059302e7283073b502f094d7-Abstract.html)

        **Abstract**:

        This paper discusses model estimation in offline model-based reinforcement learning (MBRL), which is important for subsequent policy improvement using an estimated model. From the viewpoint of covariate shift, a natural idea is model estimation weighted by the ratio of the state-action distributions of offline data and real future data. However, estimating such a natural weight is one of the main challenges for off-policy evaluation, which is not easy to use. As an artificial alternative, this paper considers weighting with the state-action distribution ratio of offline data and simulated future data, which can be estimated relatively easily by standard density ratio estimation techniques for supervised learning. Based on the artificial weight, this paper defines a loss function for offline MBRL and presents an algorithm to optimize it. Weighting with the artificial weight is justified as evaluating an upper bound of the policy evaluation error. Numerical experiments demonstrate the effectiveness of weighting with the artificial weight.

        ----

        ## [1361] Practical, Provably-Correct Interactive Learning in the Realizable Setting: The Power of True Believers

        **Authors**: *Julian Katz-Samuels, Blake Mason, Kevin G. Jamieson, Robert Nowak*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/94aada62f90dd50a84ca74304563d5db-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/94aada62f90dd50a84ca74304563d5db-Abstract.html)

        **Abstract**:

        We consider interactive learning in the realizable setting and develop a general framework to handle problems ranging from best arm identification to active classification. We begin our investigation with the observation that agnostic algorithms \emph{cannot} be minimax-optimal in the realizable setting. Hence, we design novel computationally efficient algorithms for the realizable setting that match the minimax lower bound up to logarithmic factors and are general-purpose, accommodating a wide variety of function classes including kernel methods, H{\"o}lder smooth functions, and convex functions. The sample complexities of our algorithms can be quantified in terms of well-known quantities like the extended teaching dimension and haystack dimension. However, unlike algorithms based directly on those combinatorial quantities, our algorithms are computationally efficient. To achieve computational efficiency, our algorithms sample from the version space using Monte Carlo ``hit-and-run'' algorithms instead of maintaining the version space explicitly. Our approach has two key strengths. First, it is simple, consisting of two unifying, greedy algorithms. Second, our algorithms have the capability to seamlessly leverage prior knowledge that is often available and useful in practice. In addition to our new theoretical results, we demonstrate empirically that our algorithms are competitive with Gaussian process UCB methods.

        ----

        ## [1362] Deconditional Downscaling with Gaussian Processes

        **Authors**: *Siu Lun Chau, Shahine Bouabid, Dino Sejdinovic*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/94aef38441efa3380a3bed3faf1f9d5d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/94aef38441efa3380a3bed3faf1f9d5d-Abstract.html)

        **Abstract**:

        Refining low-resolution (LR) spatial fields with high-resolution (HR) information, often known as statistical downscaling, is challenging as the diversity of spatial datasets often prevents direct matching of observations. Yet, when LR samples are modeled as aggregate conditional means of HR samples with respect to a mediating variable that is globally observed, the recovery of the underlying fine-grained field can be framed as taking an "inverse" of the conditional expectation, namely a deconditioning problem. In this work, we propose a Bayesian formulation of deconditioning which naturally recovers the initial reproducing kernel Hilbert space formulation from Hsu and Ramos (2019). We extend deconditioning to a downscaling setup and devise efficient conditional mean embedding estimator for multiresolution data. By treating conditional expectations as inter-domain features of the underlying field, a posterior for the latent field can be established as a solution to the deconditioning problem. Furthermore, we show that this solution can be viewed as a two-staged vector-valued kernel ridge regressor and show that it has a minimax optimal convergence rate under mild assumptions. Lastly, we demonstrate its proficiency in a synthetic and a real-world atmospheric field downscaling problem, showing substantial improvements over existing methods.

        ----

        ## [1363] Image Generation using Continuous Filter Atoms

        **Authors**: *Ze Wang, Seunghyun Hwang, Zichen Miao, Qiang Qiu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/94c7bb58efc3b337800875b5d382a072-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/94c7bb58efc3b337800875b5d382a072-Abstract.html)

        **Abstract**:

        In this paper, we model the subspace of convolutional filters with a neural ordinary differential equation (ODE) to enable gradual changes in generated images. Decomposing convolutional filters over a set of filter atoms allows efficiently modeling and sampling from a subspace of high-dimensional filters. By further modeling filters atoms with a neural ODE, we show both empirically and theoretically that such introduced continuity can be propagated to the generated images, and thus achieves gradually evolved image generation. We support the proposed framework of image generation with continuous filter atoms using various experiments, including image-to-image translation and image generation conditioned on continuous labels. Without auxiliary network components and heavy supervision, the proposed continuous filter atoms allow us to easily manipulate the gradual change of generated images by controlling integration intervals of neural ordinary differential equation. This research sheds the light on using the subspace of network parameters to navigate the diverse appearance of image generation.

        ----

        ## [1364] Latent Equilibrium: Arbitrarily fast computation with arbitrarily slow neurons

        **Authors**: *Paul Haider, Benjamin Ellenberger, Laura Kriener, Jakob Jordan, Walter Senn, Mihai A. Petrovici*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/94cdbdb84e8e1de8a725fa2ed61498a4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/94cdbdb84e8e1de8a725fa2ed61498a4-Abstract.html)

        **Abstract**:

        The response time of physical computational elements is finite, and neurons are no exception. In hierarchical models of cortical networks each layer thus introduces a response lag. This inherent property of physical dynamical systems results in delayed processing of stimuli and causes a timing mismatch between network output and instructive signals, thus afflicting not only inference, but also learning. We introduce Latent Equilibrium, a new framework for inference and learning in networks of slow components which avoids these issues by harnessing the ability of biological neurons to phase-advance their output with respect to their membrane potential. This principle enables quasi-instantaneous inference independent of network depth and avoids the need for phased plasticity or computationally expensive network relaxation phases. We jointly derive disentangled neuron and synapse dynamics from a prospective energy function that depends on a network's generalized position and momentum. The resulting model can be interpreted as a biologically plausible approximation of error backpropagation in deep cortical networks with continuous-time, leaky neuronal dynamics and continuously active, local plasticity. We demonstrate successful learning of standard benchmark datasets, achieving competitive performance using both fully-connected and convolutional architectures, and show how our principle can be applied to detailed models of cortical microcircuitry. Furthermore, we study the robustness of our model to spatio-temporal substrate imperfections to demonstrate its feasibility for physical realization, be it in vivo or in silico.

        ----

        ## [1365] Learning Fast-Inference Bayesian Networks

        **Authors**: *Vaidyanathan Peruvemba Ramaswamy, Stefan Szeider*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/94e70705efae423efda1088614128d0b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/94e70705efae423efda1088614128d0b-Abstract.html)

        **Abstract**:

        We propose new methods for learning Bayesian networks (BNs) that reliably support fast inference. We utilize maximum state space size as a more fine-grained measure for the BN's reasoning complexity than the standard treewidth measure, thereby accommodating the possibility that variables range over domains of different sizes. Our methods combine heuristic BN structure learning algorithms with the recently introduced MaxSAT-powered local improvement method (Peruvemba Ramaswamy and Szeider, AAAI'21). Our experiments show that our new learning methods produce BNs that support significantly faster exact probabilistic inference than BNs learned with treewidth bounds.

        ----

        ## [1366] Per-Pixel Classification is Not All You Need for Semantic Segmentation

        **Authors**: *Bowen Cheng, Alexander G. Schwing, Alexander Kirillov*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/950a4152c2b4aa3ad78bdd6b366cc179-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/950a4152c2b4aa3ad78bdd6b366cc179-Abstract.html)

        **Abstract**:

        Modern approaches typically formulate semantic segmentation as a per-pixel classification task, while instance-level segmentation is handled with an alternative mask classification. Our key insight: mask classification is sufficiently general to solve both semantic- and instance-level segmentation tasks in a unified manner using the exact same model, loss, and training procedure. Following this observation, we propose MaskFormer, a simple mask classification model which predicts a set of binary masks, each associated with a single global class label prediction. Overall, the proposed mask classification-based method simplifies the landscape of effective approaches to semantic and panoptic segmentation tasks and shows excellent empirical results. In particular, we observe that MaskFormer outperforms per-pixel classification baselines when the number of classes is large. Our mask classification-based method outperforms both current state-of-the-art semantic (55.6 mIoU on ADE20K) and panoptic segmentation (52.7 PQ on COCO) models.

        ----

        ## [1367] Deep Markov Factor Analysis: Towards Concurrent Temporal and Spatial Analysis of fMRI Data

        **Authors**: *Amirreza Farnoosh, Sarah Ostadabbas*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/951124d4a093eeae83d9726a20295498-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/951124d4a093eeae83d9726a20295498-Abstract.html)

        **Abstract**:

        Factor analysis methods have been widely used in neuroimaging to transfer high dimensional imaging data into low dimensional, ideally interpretable representations. However, most of these methods overlook the highly nonlinear and complex temporal dynamics of neural processes when factorizing their imaging data. In this paper, we present deep Markov factor analysis (DMFA), a generative model that employs Markov property in a chain of low dimensional temporal embeddings together with spatial inductive assumptions, all related through neural networks, to capture temporal dynamics in functional magnetic resonance imaging (fMRI) data, and tackle their high spatial dimensionality, respectively. Augmented with a discrete latent, DMFA is able to cluster fMRI data in its low dimensional temporal embedding with regard to subject and cognitive state variability, therefore, enables validation of a variety of fMRI-driven neuroscientific hypotheses. Experimental results on both synthetic and real fMRI data demonstrate the capacity of DMFA in revealing interpretable clusters and capturing nonlinear temporal dependencies in these high dimensional imaging data.

        ----

        ## [1368] BooVAE: Boosting Approach for Continual Learning of VAE

        **Authors**: *Evgenii Egorov, Anna Kuzina, Evgeny Burnaev*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html)

        **Abstract**:

        Variational autoencoder (VAE) is a deep generative model for unsupervised learning, allowing to encode observations into the meaningful latent space. VAE is prone to catastrophic forgetting when tasks arrive sequentially, and only the data for the current one is available. We address this problem of continual learning for VAEs. It is known that the choice of the prior distribution over the latent space is crucial for VAE in the non-continual setting. We argue that it can also be helpful to avoid catastrophic forgetting. We learn the approximation of the aggregated posterior as a prior for each task. This approximation is parametrised as an additive mixture of distributions induced by an encoder evaluated at trainable pseudo-inputs. We use a greedy boosting-like approach with entropy regularisation to learn the components. This method encourages components diversity, which is essential as we aim at memorising the current task with the fewest components possible. Based on the learnable prior, we introduce an end-to-end approach for continual learning of VAEs and provide empirical studies on commonly used benchmarks (MNIST, Fashion MNIST, NotMNIST) and CelebA datasets. For each dataset, the proposed method avoids catastrophic forgetting in a fully automatic way.

        ----

        ## [1369] Handling Long-tailed Feature Distribution in AdderNets

        **Authors**: *Minjing Dong, Yunhe Wang, Xinghao Chen, Chang Xu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/95323660ed2124450caaac2c46b5ed90-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/95323660ed2124450caaac2c46b5ed90-Abstract.html)

        **Abstract**:

        Adder neural networks (ANNs) are designed for low energy cost which replace expensive multiplications in convolutional neural networks (CNNs) with cheaper additions to yield energy-efficient neural networks and hardware accelerations. Although ANNs achieve satisfactory efficiency, there exist gaps between ANNs and CNNs where the accuracy of ANNs can hardly be compared to CNNs without the assistance of other training tricks, such as knowledge distillation. The inherent discrepancy lies in the similarity measurement between filters and features, however how to alleviate this difference remains unexplored. To locate the potential problem of ANNs, we focus on the property difference due to similarity measurement. We demonstrate that unordered heavy tails in ANNs could be the key component which prevents ANNs from achieving superior classification performance since fatter tails tend to overlap in feature space. Through pre-defining Multivariate Skew Laplace distributions and embedding feature distributions into the loss function, ANN features can be fully controlled and designed for various properties. We further present a novel method for tackling existing heavy tails in ANNs with only a modification of classifier where ANN features are clustered with their tails well-formulated through proposed angle-based constraint on the distribution parameters to encourage high diversity of tails. Experiments conducted on several benchmarks and comparison with other distributions demonstrate the effectiveness of proposed approach for boosting the performance of ANNs.

        ----

        ## [1370] Pessimism Meets Invariance: Provably Efficient Offline Mean-Field Multi-Agent RL

        **Authors**: *Minshuo Chen, Yan Li, Ethan Wang, Zhuoran Yang, Zhaoran Wang, Tuo Zhao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9559fc73b13fa721a816958488a5b449-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9559fc73b13fa721a816958488a5b449-Abstract.html)

        **Abstract**:

        Mean-Field Multi-Agent Reinforcement Learning (MF-MARL) is attractive in the applications involving a large population of homogeneous agents, as it exploits the permutation invariance of agents and avoids the curse of many agents. Most existing results only focus on online settings, in which agents can interact with the environment during training. In some applications such as social welfare optimization, however, the interaction during training can be prohibitive or even unethical in the societal systems. To bridge such a gap, we propose a SAFARI (peSsimistic meAn-Field vAlue iteRatIon) algorithm for off-line MF-MARL, which only requires a handful of pre-collected experience data. Theoretically, under a weak coverage assumption that the experience dataset contains enough information about the optimal policy, we prove that for an episodic mean-field MDP with a horizon $H$ and $N$ training trajectories, SAFARI attains a sub-optimality gap of $\mathcal{O}(H^2d_{\rm eff} /\sqrt{N})$, where $d_{\rm eff}$ is the effective dimension of the function class for parameterizing the value function, but independent on the number of agents. Numerical experiments are provided.

        ----

        ## [1371] A Law of Iterated Logarithm for Multi-Agent Reinforcement Learning

        **Authors**: *Gugan Thoppe, Bhumesh Kumar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/955fd82131e15e7b5199cbc8f983306a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/955fd82131e15e7b5199cbc8f983306a-Abstract.html)

        **Abstract**:

        In Multi-Agent Reinforcement Learning (MARL), multiple agents interact with a common environment, as also with each other, for solving a shared problem in sequential decision-making. It has wide-ranging applications in gaming, robotics, finance, communication, etc. In this work, we derive a novel law of iterated logarithm for a family of  distributed nonlinear stochastic approximation schemes that is useful in MARL. In particular, our result describes the convergence rate on almost every sample path where the algorithm converges. This result is the first of its kind in the distributed setup and provides deeper insights than the existing ones, which only discuss convergence rates in the expected or the CLT sense. Importantly, our result holds under significantly weaker assumptions: neither the gossip matrix needs to be doubly stochastic nor the stepsizes square summable. As an application, we show  that, for the stepsize $n^{-\gamma}$ with $\gamma \in (0, 1),$ the distributed TD(0) algorithm with linear function approximation has a convergence rate of $O(\sqrt{n^{-\gamma} \ln n })$ a.s.; for the $1/n$ type stepsize, the same is $O(\sqrt{n^{-1} \ln \ln n})$ a.s. These decay rates do not depend on the graph depicting the interactions among the different agents.

        ----

        ## [1372] MOMA: Multi-Object Multi-Actor Activity Parsing

        **Authors**: *Zelun Luo, Wanze Xie, Siddharth Kapoor, Yiyun Liang, Michael Cooper, Juan Carlos Niebles, Ehsan Adeli, Fei-Fei Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/95688ba636a4720a85b3634acfec8cdd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/95688ba636a4720a85b3634acfec8cdd-Abstract.html)

        **Abstract**:

        Complex activities often involve multiple humans utilizing different objects to complete actions (e.g., in healthcare settings, physicians, nurses, and patients interact with each other and various medical devices). Recognizing activities poses a challenge that requires a detailed understanding of actors' roles, objects' affordances, and their associated relationships. Furthermore, these purposeful activities are composed of multiple achievable steps, including sub-activities and atomic actions, which jointly define a hierarchy of action parts. This paper introduces Activity Parsing as the overarching task of temporal segmentation and classification of activities, sub-activities, atomic actions, along with an instance-level understanding of actors, objects, and their relationships in videos. Involving multiple entities (actors and objects), we argue that traditional pair-wise relationships, often used in scene or action graphs, do not appropriately represent the dynamics between them. Hence, we introduce Action Hypergraph, a spatial-temporal graph containing hyperedges (i.e., edges with higher-order relationships), as a new representation. In addition, we introduce Multi-Object Multi-Actor (MOMA), the first benchmark and dataset dedicated to activity parsing. Lastly, to parse a video, we propose the HyperGraph Activity Parsing (HGAP) network, which outperforms several baselines, including those based on regular graphs and raw video data.

        ----

        ## [1373] The Pareto Frontier of model selection for general Contextual Bandits

        **Authors**: *Teodor Vanislavov Marinov, Julian Zimmert*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9570efef719d705326f0ff817ef084e6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9570efef719d705326f0ff817ef084e6-Abstract.html)

        **Abstract**:

        Recent progress in model selection raises the question of the fundamental limits of these techniques. Under specific scrutiny has been model selection for general contextual bandits with nested policy classes, resulting in a COLT2020 open problem. It asks whether it is possible to obtain simultaneously the optimal single algorithm guarantees over all policies in a nested sequence of policy classes, or if otherwise this is possible for a trade-off  $\alpha\in[\frac{1}{2},1)$ between complexity term and time: $\ln(|\Pi_m|)^{1-\alpha}T^\alpha$. We give a disappointing answer to this question. Even in the purely stochastic regime, the desired results are unobtainable. We present a Pareto frontier of up to logarithmic factors matching upper and lower bounds, thereby proving that an increase in the complexity term $\ln(|\Pi_m|)$ independent of $T$ is unavoidable for general policy classes.As a side result, we also resolve a COLT2016 open problem concerning second-order bounds in full-information games.

        ----

        ## [1374] Teaching an Active Learner with Contrastive Examples

        **Authors**: *Chaoqi Wang, Adish Singla, Yuxin Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/958adb57686c2fdec5796398de5f317a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/958adb57686c2fdec5796398de5f317a-Abstract.html)

        **Abstract**:

        We study the problem of active learning with the added twist that the learner is assisted by a helpful teacher. We consider the following natural interaction protocol: At each round, the learner proposes a query asking for the label of an instance $x^q$, the teacher provides the requested label $\{x^q, y^q\}$ along with explanatory information to guide the learning process. In this paper, we view this information in the form of an additional contrastive example ($\{x^c, y^c\}$) where $x^c$ is picked from a set constrained by $x^q$ (e.g., dissimilar instances with the same label). Our focus is to design a teaching algorithm that can provide an informative sequence of contrastive examples to the learner to speed up the learning process. We show that this leads to a challenging sequence optimization problem where the algorithm's choices at a given round depend on the history of interactions. We investigate an efficient teaching algorithm that adaptively picks these contrastive examples. We derive strong performance guarantees for our algorithm based on two problem-dependent parameters and further show that for specific types of active learners (e.g., a generalized binary search learner), the proposed teaching algorithm exhibits strong approximation guarantees. Finally, we illustrate our bounds and demonstrate the effectiveness of our teaching framework via two numerical case studies.

        ----

        ## [1375] Structured Denoising Diffusion Models in Discrete State-Spaces

        **Authors**: *Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html)

        **Abstract**:

        Denoising diffusion probabilistic models (DDPMs) [Ho et al. 2021] have shown impressive results on image and waveform generation in continuous state spaces. Here, we introduce Discrete Denoising Diffusion Probabilistic Models (D3PMs), diffusion-like generative models for discrete data that generalize the multinomial diffusion model of Hoogeboom et al. [2021], by going beyond corruption processes with uniform transition probabilities. This includes corruption with transition matrices that mimic Gaussian kernels in continuous space, matrices based on nearest neighbors in embedding space, and matrices that introduce absorbing states. The third allows us to draw a connection between diffusion models and autoregressive and mask-based generative models. We show that the choice of transition matrix is an important design decision that leads to improved results in image and text domains. We also introduce a new loss function that combines the variational lower bound with an auxiliary cross entropy loss.  For text, this model class achieves strong results on character-level text generation while scaling to large vocabularies on LM1B. On the image dataset CIFAR-10, our models approach the sample quality and exceed the log-likelihood of the continuous-space DDPM model.

        ----

        ## [1376] Emergent Communication of Generalizations

        **Authors**: *Jesse Mu, Noah D. Goodman*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9597353e41e6957b5e7aa79214fcb256-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9597353e41e6957b5e7aa79214fcb256-Abstract.html)

        **Abstract**:

        To build agents that can collaborate effectively with others, recent research has trained artificial agents to communicate with each other in Lewis-style referential games. However, this often leads to successful but uninterpretable communication. We argue that this is due to the game objective: communicating about a single object in a shared visual context is prone to overfitting and does not encourage language useful beyond concrete reference. In contrast, human language conveys a rich variety of abstract ideas. To promote such skills, we propose games that require communicating generalizations over sets of objects representing abstract visual concepts, optionally with separate contexts for each agent. We find that these games greatly improve systematicity and interpretability of the learned languages, according to several metrics in the literature. Finally, we propose a method for identifying logical operations embedded in the emergent languages by learning an approximate compositional reconstruction of the language.

        ----

        ## [1377] Distributed Machine Learning with Sparse Heterogeneous Data

        **Authors**: *Dominic Richards, Sahand Negahban, Patrick Rebeschini*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/959776b99b006e5785c3a3364949ce47-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/959776b99b006e5785c3a3364949ce47-Abstract.html)

        **Abstract**:

        Motivated by distributed machine learning settings such as Federated Learning, we consider the problem of fitting a statistical model across a distributed collection of heterogeneous data sets whose similarity structure is encoded by a graph topology. Precisely, we analyse the case where each node is associated with fitting a sparse linear model, and edges join two nodes if the difference of their solutions is also sparse. We propose a method based on Basis Pursuit Denoising with a total variation penalty, and provide finite sample guarantees for sub-Gaussian  design matrices. Taking the root of the tree as a reference node, we show that if the sparsity of the differences across nodes is smaller than the sparsity at the root, then recovery is successful with fewer samples than by solving the problems independently, or by using methods that rely on a large overlap in the signal supports, such as the group Lasso. We consider both the noiseless and noisy setting, and numerically investigate the performance of distributed methods based on Distributed Alternating Direction Methods of Multipliers (ADMM) and hyperspectral unmixing.

        ----

        ## [1378] Manipulating SGD with Data Ordering Attacks

        **Authors**: *Ilia Shumailov, Zakhar Shumaylov, Dmitry Kazhdan, Yiren Zhao, Nicolas Papernot, Murat A. Erdogdu, Ross J. Anderson*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/959ab9a0695c467e7caf75431a872e5c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/959ab9a0695c467e7caf75431a872e5c-Abstract.html)

        **Abstract**:

        Machine learning is vulnerable to a wide variety of attacks. It is now well understood that by changing the underlying data distribution, an adversary can poison the model trained with it or introduce backdoors. In this paper we present a novel class of training-time attacks that require no changes to the underlying dataset or model architecture, but instead only change the order in which data are supplied to the model. In particular, we find that the attacker can either prevent the model from learning, or poison it to learn behaviours specified by the attacker. Furthermore, we find that even a single adversarially-ordered epoch can be enough to slow down model learning, or even to reset all of the learning progress. Indeed, the attacks presented here are not specific to the model or dataset, but rather target the stochastic nature of modern learning procedures. We extensively evaluate our attacks on computer vision and natural language benchmarks to find that the adversary can disrupt model training and even introduce backdoors.

        ----

        ## [1379] Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification

        **Authors**: *Maximilian Stadler, Bertrand Charpentier, Simon Geisler, Daniel Zügner, Stephan Günnemann*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/95b431e51fc53692913da5263c214162-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/95b431e51fc53692913da5263c214162-Abstract.html)

        **Abstract**:

        The interdependence between nodes in graphs is key to improve class prediction on nodes, utilized in approaches like Label Probagation (LP) or in Graph Neural Networks (GNNs). Nonetheless, uncertainty estimation for non-independent node-level predictions is under-explored.  In this work, we explore uncertainty quantification for node classification in three ways: (1) We derive three axioms explicitly characterizing the expected predictive uncertainty behavior in homophilic attributed graphs.(2) We propose a new model Graph Posterior Network (GPN) which explicitly performs Bayesian posterior updates for predictions on interdependent nodes. GPN provably obeys the proposed axioms. (3) We extensively evaluate GPN and a strong set of baselines on semi-supervised node classification including detection of anomalous features, and detection of left-out classes. GPN outperforms existing approaches for uncertainty estimation in the experiments.

        ----

        ## [1380] Locality Sensitive Teaching

        **Authors**: *Zhaozhuo Xu, Beidi Chen, Chaojian Li, Weiyang Liu, Le Song, Yingyan Lin, Anshumali Shrivastava*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/95c3f1a8b262ec7a929a8739e21142d7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/95c3f1a8b262ec7a929a8739e21142d7-Abstract.html)

        **Abstract**:

        The emergence of the Internet-of-Things (IoT) sheds light on applying the machine teaching (MT) algorithms for online personalized education on home devices. This direction becomes more promising during the COVID-19 pandemic when in-person education becomes infeasible. However, as one of the most influential and practical MT paradigms, iterative machine teaching (IMT) is prohibited on IoT devices due to its inefficient and unscalable algorithms. IMT is a paradigm where a teacher feeds examples iteratively and intelligently based on the learner's status. In each iteration, current IMT algorithms greedily traverse the whole training set to find an example for the learner, which is computationally expensive in practice.  We propose a novel teaching framework, Locality Sensitive Teaching (LST), based on locality sensitive sampling, to overcome these challenges. LST has provable near-constant time complexity, which is exponentially better than the existing baseline. With at most 425.12x speedups and 99.76% energy savings over IMT, LST is the first algorithm that enables energy and time efficient machine teaching on IoT devices. Owing to LST's substantial efficiency and scalability, it is readily applicable in real-world education scenarios.

        ----

        ## [1381] No-Press Diplomacy from Scratch

        **Authors**: *Anton Bakhtin, David J. Wu, Adam Lerer, Noam Brown*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/95f2b84de5660ddf45c8a34933a2e66f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/95f2b84de5660ddf45c8a34933a2e66f-Abstract.html)

        **Abstract**:

        Prior AI successes in complex games have largely focused on settings with at most hundreds of actions at each decision point. In contrast, Diplomacy is a game with more than 10^20 possible actions per turn. Previous attempts to address games with large branching factors, such as Diplomacy, StarCraft, and Dota, used human data to bootstrap the policy or used handcrafted reward shaping. In this paper, we describe an algorithm for action exploration and equilibrium approximation in games with combinatorial action spaces. This algorithm simultaneously performs value iteration while learning a policy proposal network. A double oracle step is used to explore additional actions to add to the policy proposals. At each state, the target state value and policy for the model training are computed via an equilibrium search procedure. Using this algorithm, we train an agent, DORA, completely from scratch for a popular two-player variant of Diplomacy and show that it achieves superhuman performance. Additionally, we extend our methods to full-scale no-press Diplomacy and for the first time train an agent from scratch with no human data. We present evidence that this agent plays a strategy that is incompatible with human-data bootstrapped agents. This presents the first strong evidence of multiple equilibria in Diplomacy and suggests that self play alone may be insufficient for achieving superhuman performance in Diplomacy.

        ----

        ## [1382] Remember What You Want to Forget: Algorithms for Machine Unlearning

        **Authors**: *Ayush Sekhari, Jayadev Acharya, Gautam Kamath, Ananda Theertha Suresh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9627c45df543c816a3ddf2d8ea686a99-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9627c45df543c816a3ddf2d8ea686a99-Abstract.html)

        **Abstract**:

        We study the problem of unlearning datapoints from a learnt model. The learner first receives a dataset $S$ drawn i.i.d. from an unknown distribution, and outputs a model $\widehat{w}$ that performs well on  unseen samples from the same distribution. However, at some point in the future, any training datapoint $z \in S$ can request to be unlearned, thus prompting the learner to modify its output model while still ensuring the same accuracy guarantees.  We initiate a rigorous study of generalization in machine unlearning, where the goal is to perform well on previously unseen datapoints. Our focus is on both computational and storage complexity. For the setting of convex losses, we provide an unlearning algorithm that can unlearn up to $O(n/d^{1/4})$ samples, where $d$ is the problem dimension. In comparison, in general, differentially private learning (which implies unlearning) only guarantees deletion of $O(n/d^{1/2})$ samples. This demonstrates a novel separation between differential privacy and machine unlearning.

        ----

        ## [1383] Learning latent causal graphs via mixture oracles

        **Authors**: *Bohdan Kivva, Goutham Rajendran, Pradeep Ravikumar, Bryon Aragam*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/966aad8981dcc75b5b8ab04427a833b2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/966aad8981dcc75b5b8ab04427a833b2-Abstract.html)

        **Abstract**:

        We study the problem of reconstructing a causal graphical model from data in the presence of latent variables. The main problem of interest is recovering the causal structure over the latent variables while allowing for general, potentially nonlinear dependencies. In many practical problems, the dependence between raw observations (e.g. pixels in an image) is much less relevant than the dependence between certain high-level, latent features (e.g. concepts or objects), and this is the setting of interest. We provide conditions under which both the latent representations and the underlying latent causal model are identifiable by a reduction to a mixture oracle. These results highlight an intriguing connection between the well-studied problem of learning the order of a mixture model and the problem of learning the bipartite structure between observables and unobservables. The proof is constructive, and leads to several algorithms for explicitly reconstructing the full graphical model. We discuss efficient algorithms and provide experiments illustrating the algorithms in practice.

        ----

        ## [1384] ErrorCompensatedX: error compensation for variance reduced algorithms

        **Authors**: *Hanlin Tang, Yao Li, Ji Liu, Ming Yan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/968c9b4f09cbb7d7925f38aea3484111-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/968c9b4f09cbb7d7925f38aea3484111-Abstract.html)

        **Abstract**:

        Communication cost is one major bottleneck for the scalability for distributed learning. One approach to reduce the communication cost is to compress the gradient during communication. However, directly compressing the gradient decelerates the convergence speed, and the resulting algorithm may diverge for biased compression. Recent work addressed this problem for stochastic gradient descent by adding back the compression error from the previous step. This idea was further extended to one class of variance reduced algorithms, where the variance of the stochastic gradient is reduced by taking a moving average over all history gradients. However, our analysis shows that just adding the previous step's compression error, as done in existing work, does not fully compensate the compression error. So, we propose ErrorCompensateX, which uses the compression error from the previous two steps. We show that ErrorCompensateX can achieve the same asymptotic convergence rate with the training without compression. Moreover, we  provide a unified theoretical analysis framework for this class of variance reduced algorithms, with or without error compensation.

        ----

        ## [1385] Deep Contextual Video Compression

        **Authors**: *Jiahao Li, Bin Li, Yan Lu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/96b250a90d3cf0868c83f8c965142d2a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/96b250a90d3cf0868c83f8c965142d2a-Abstract.html)

        **Abstract**:

        Most of the existing neural video compression methods adopt the predictive coding framework, which first generates the predicted frame and then encodes its residue with the current frame. However, as for compression ratio, predictive coding is only a sub-optimal solution as it uses simple subtraction operation to remove the redundancy across frames. In this paper, we propose a deep contextual video compression framework to enable a paradigm shift from predictive coding to conditional coding. In particular, we try to answer the following questions: how to define, use, and learn condition under a deep video compression framework. To tap the potential of conditional coding, we propose using feature domain context as condition. This enables us to leverage the high dimension context to carry rich information to both the encoder and the decoder, which helps reconstruct the high-frequency contents for higher video quality. Our framework is also extensible, in which the condition can be flexibly designed. Experiments show that our method can significantly outperform the previous state-of-the-art (SOTA) deep video compression methods. When compared with x265 using veryslow preset, we can achieve 26.0% bitrate saving for 1080P standard test videos.

        ----

        ## [1386] On the Frequency Bias of Generative Models

        **Authors**: *Katja Schwarz, Yiyi Liao, Andreas Geiger*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/96bf57c6ff19504ff145e2a32991ea96-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/96bf57c6ff19504ff145e2a32991ea96-Abstract.html)

        **Abstract**:

        The key objective of Generative Adversarial Networks (GANs) is to generate new data with the same statistics as the provided training data. However, multiple recent works show that state-of-the-art architectures yet struggle to achieve this goal. In particular, they report an elevated amount of high frequencies in the spectral statistics which makes it straightforward to distinguish real and generated images. Explanations for this phenomenon are controversial: While most works attribute the artifacts to the generator, other works point to the discriminator.  We take a sober look at those explanations and provide insights on what makes proposed measures against high-frequency artifacts effective. To achieve this, we first independently assess the architectures of both the generator and discriminator and investigate if they exhibit a frequency bias that makes learning the distribution of high-frequency content particularly problematic. Based on these experiments, we make the following four observations: 1) Different upsampling operations bias the generator towards different spectral properties. 2) Checkerboard artifacts introduced by upsampling cannot explain the spectral discrepancies alone as the generator is able to compensate for these artifacts. 3) The discriminator does not struggle with detecting high frequencies per se but rather struggles with frequencies of low magnitude. 4) The downsampling operations in the discriminator can impair the quality of the training signal it provides.In light of these findings, we analyze proposed measures against high-frequency artifacts in state-of-the-art GAN training but find that none of the existing approaches can fully resolve spectral artifacts yet. Our results suggest that there is great potential in improving the discriminator and that this could be key to match the distribution of the training data more closely.

        ----

        ## [1387] Learning curves of generic features maps for realistic datasets with a teacher-student model

        **Authors**: *Bruno Loureiro, Cédric Gerbelot, Hugo Cui, Sebastian Goldt, Florent Krzakala, Marc Mézard, Lenka Zdeborová*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9704a4fc48ae88598dcbdcdf57f3fdef-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9704a4fc48ae88598dcbdcdf57f3fdef-Abstract.html)

        **Abstract**:

        Teacher-student models provide a framework in which the typical-case performance of high-dimensional supervised learning can be described in closed form. The assumptions of Gaussian i.i.d. input data underlying the canonical teacher-student model may, however, be perceived as too restrictive to capture the behaviour of realistic data sets. In this paper, we introduce a Gaussian covariate generalisation of the model where the teacher and student can act on different spaces, generated with fixed, but generic feature maps. While still solvable in a closed form, this generalization is able to capture the learning curves for a broad range of realistic data sets, thus redeeming the potential of the teacher-student framework. Our contribution is then two-fold: first, we prove a rigorous formula for the asymptotic training loss and generalisation error. Second, we present a number of situations where the learning curve of the model captures the one of a realistic data set learned with kernel regression and classification, with out-of-the-box feature maps such as random projections or scattering transforms, or with pre-learned ones - such as the features learned by training multi-layer neural networks. We discuss both the power and the limitations of the framework.

        ----

        ## [1388] It Has Potential: Gradient-Driven Denoisers for Convergent Solutions to Inverse Problems

        **Authors**: *Regev Cohen, Yochai Blau, Daniel Freedman, Ehud Rivlin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/97108695bd93b6be52fa0334874c8722-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/97108695bd93b6be52fa0334874c8722-Abstract.html)

        **Abstract**:

        In recent years there has been increasing interest in leveraging denoisers for solving general inverse problems. Two leading frameworks are regularization-by-denoising (RED) and plug-and-play priors (PnP) which incorporate explicit likelihood functions with priors induced by denoising algorithms.  RED and PnP have shown state-of-the-art performance in diverse imaging tasks when powerful denoisersare used, such as convolutional neural networks (CNNs). However, the study of their convergence remains an active line of research.  Recent works derive the convergence of RED and PnP methods by treating CNN denoisers as approximations for maximum a posteriori (MAP) or minimum mean square error (MMSE) estimators.  Yet, state-of-the-art denoisers cannot be interpreted as either MAPor MMSE estimators, since they typically do not exhibit symmetric Jacobians. Furthermore, obtaining stable inverse algorithms often requires controlling the Lipschitz constant of CNN denoisers during training.  Precisely enforcing this constraint is impractical, hence, convergence cannot be completely guaranteed. In this work, we introduce image denoisers derived as the gradients of smooth scalar-valued deep neural networks, acting as potentials. This ensures two things: (1) the proposed denoisers display symmetric Jacobians, allowing for MAP and MMSE estimators interpretation; (2) the denoisers may be integrated into RED and PnP schemes with backtracking step size, removing the need for enforcing their Lipschitz constant. To show the latter, we develop a simple inversion method that utilizes the proposed denoisers. We theoretically establish its convergence to stationary points of an underlying objective function consisting of the learned potentials. We numerically validate our method through various imaging experiments, showing improved results compared to standard RED and PnP methods, and with additional provable stability.

        ----

        ## [1389] Training Over-parameterized Models with Non-decomposable Objectives

        **Authors**: *Harikrishna Narasimhan, Aditya Krishna Menon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9713faa264b94e2bf346a1bb52587fd8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9713faa264b94e2bf346a1bb52587fd8-Abstract.html)

        **Abstract**:

        Many modern machine learning applications come with complex and nuanced design goals such as minimizing the worst-case error, satisfying a given precision or recall target, or enforcing group-fairness constraints. Popular techniques for optimizing such non-decomposable objectives reduce the problem into a sequence of cost-sensitive learning tasks, each of which is then solved by re-weighting the training loss with example-specific costs. We point out that the standard approach of re-weighting the loss to incorporate label costs can produce unsatisfactory results when used to train over-parameterized models. As a remedy, we propose new cost- sensitive losses that extend the classical idea of logit adjustment to handle more general cost matrices. Our losses are calibrated, and can be further improved with distilled labels from a teacher model. Through experiments on benchmark image datasets, we showcase the effectiveness of our approach in training ResNet models with common robust and constrained optimization objectives.

        ----

        ## [1390] Reinforcement learning for optimization of variational quantum circuit architectures

        **Authors**: *Mateusz Ostaszewski, Lea M. Trenkwalder, Wojciech Masarczyk, Eleanor Scerri, Vedran Dunjko*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9724412729185d53a2e3e7f889d9f057-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9724412729185d53a2e3e7f889d9f057-Abstract.html)

        **Abstract**:

        The study of Variational Quantum Eigensolvers (VQEs) has been in the spotlight in recent times as they may lead to real-world applications of near-term quantum devices. However, their performance depends on the structure of the used variational ansatz, which requires balancing the depth and expressivity of the corresponding circuit. At the same time, near-term restrictions limit the depth of the circuit we can expect to run. Thus, the optimization of the VQE ansatz requires maximizing the expressivity of the circuit while maintaining low depth. In recent years, various methods for VQE structure optimization have been introduced but the capacities of machine learning to aid with this problem have not yet been extensively investigated. In this work, we propose a reinforcement learning algorithm that autonomously explores the space of possible ansatzes, identifying economic circuits which still yield accurate ground energy estimates. The algorithm uses a feedback-driven curriculum learning method that autonomously adapts the complexity of the learning problem to the current performance of the learning algorithm and it incrementally improves the accuracy of the result while minimizing the circuit depth. We showcase the performance of our algorithm on the problem of estimating the ground-state energy of lithium hydride (LiH) in various configurations. In this well-known benchmark problem, we achieve chemical accuracy and state-of-the-art results in terms of circuit depth.

        ----

        ## [1391] Moshpit SGD: Communication-Efficient Decentralized Training on Heterogeneous Unreliable Devices

        **Authors**: *Max Ryabinin, Eduard Gorbunov, Vsevolod Plokhotnyuk, Gennady Pekhimenko*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/97275a23ca44226c9964043c8462be96-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/97275a23ca44226c9964043c8462be96-Abstract.html)

        **Abstract**:

        Training deep neural networks on large datasets can often be accelerated by using multiple compute nodes. This approach, known as distributed training, can utilize hundreds of computers via specialized message-passing protocols such as Ring All-Reduce.However, running these protocols at scale requires reliable high-speed networking that is only available in dedicated clusters.In contrast, many real-world applications, such as federated learning and cloud-based distributed training, operate on unreliable devices with unstable network bandwidth.As a result, these applications are restricted to using parameter servers or gossip-based averaging protocols.In this work, we lift that restriction by proposing Moshpit All-Reduce â€” an iterative averaging protocol that exponentially converges to the global average.We demonstrate the efficiency of our protocol for distributed optimization with strong theoretical guarantees.The experiments show 1.3x speedup for ResNet-50 training on ImageNet compared to competitive gossip-based strategies and 1.5x speedup when training ALBERT-large on preemptible compute nodes.

        ----

        ## [1392] IRM - when it works and when it doesn't: A test case of natural language inference

        **Authors**: *Yana Dranker, He He, Yonatan Belinkov*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/972cda1e62b72640cb7ac702714a115f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/972cda1e62b72640cb7ac702714a115f-Abstract.html)

        **Abstract**:

        Invariant Risk Minimization (IRM) is a recently proposed framework for out-of-distribution (o.o.d) generalization.  Most of the studies on IRM so far have focused on theoretical results, toy problems, and simple models. In this work, we investigate the applicability of IRM to bias mitigation-a special case of o.o.d generalization-in increasingly naturalistic settings and deep models. Using natural language inference (NLI) as a test case, we start with a setting where both the dataset and the bias are synthetic, continue with a natural dataset and synthetic bias, and end with a fully realistic setting with natural datasets and bias. Our results show that in naturalistic settings, learning complex features in place of the bias proves to be difficult, leading to a rather small improvement over empirical risk minimization. Moreover, we find that in addition to being sensitive to random seeds, the performance of IRM also depends on several critical factors, notably dataset size, bias prevalence, and bias strength, thus limiting IRM's advantage in practical scenarios. Our results  highlight key challenges in applying IRM to real-world scenarios, calling for a more naturalistic characterization of  the problem setup for o.o.d generalization.

        ----

        ## [1393] Self-Supervised Learning Disentangled Group Representation as Feature

        **Authors**: *Tan Wang, Zhongqi Yue, Jianqiang Huang, Qianru Sun, Hanwang Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/97416ac0f58056947e2eb5d5d253d4f2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/97416ac0f58056947e2eb5d5d253d4f2-Abstract.html)

        **Abstract**:

        A good visual representation is an inference map from observations (images) to features (vectors) that faithfully reflects the hidden modularized generative factors (semantics). In this paper, we formulate the notion of "good" representation from a group-theoretic view using Higgins' definition of disentangled representation, and show that existing Self-Supervised Learning (SSL) only disentangles simple augmentation features such as rotation and colorization, thus unable to modularize the remaining semantics. To break the limitation, we propose an iterative SSL algorithm: Iterative Partition-based Invariant Risk Minimization (IP-IRM), which successfully grounds the abstract semantics and the group acting on them into concrete contrastive learning. At each iteration, IP-IRM first partitions the training samples into two subsets that correspond to an entangled group element. Then, it minimizes a subset-invariant contrastive loss, where the invariance guarantees to disentangle the group element. We prove that IP-IRM converges to a fully disentangled representation and show its effectiveness on various benchmarks. Codes are available at https://github.com/Wangt-CN/IP-IRM.

        ----

        ## [1394] SalKG: Learning From Knowledge Graph Explanations for Commonsense Reasoning

        **Authors**: *Aaron Chan, Jiashu Xu, Boyuan Long, Soumya Sanyal, Tanishq Gupta, Xiang Ren*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9752d873fa71c19dc602bf2a0696f9b5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9752d873fa71c19dc602bf2a0696f9b5-Abstract.html)

        **Abstract**:

        Augmenting pre-trained language models with knowledge graphs (KGs) has achieved success on various commonsense reasoning tasks. However, for a given task instance, the KG, or certain parts of the KG, may not be useful. Although KG-augmented models often use attention to focus on specific KG components, the KG is still always used, and the attention mechanism is never explicitly taught which KG components should be used. Meanwhile, saliency methods can measure how much a KG feature (e.g., graph, node, path) influences the model to make the correct prediction, thus explaining which KG features are useful. This paper explores how saliency explanations can be used to improve KG-augmented models' performance. First, we propose to create coarse (Is the KG useful?) and fine (Which nodes/paths in the KG are useful?) saliency explanations. Second, to motivate saliency-based supervision, we analyze oracle KG-augmented models which directly use saliency explanations as extra inputs for guiding their attention. Third, we propose SalKG, a framework for KG-augmented models to learn from coarse and/or fine saliency explanations. Given saliency explanations created from a task's training set, SalKG jointly trains the model to predict the explanations, then solve the task by attending to KG features highlighted by the predicted explanations. On three popular commonsense QA benchmarks (CSQA, OBQA, CODAH) and a range of KG-augmented models, we show that SalKG can yield considerable performance gains --- up to 2.76% absolute improvement on CSQA.

        ----

        ## [1395] Supervising the Transfer of Reasoning Patterns in VQA

        **Authors**: *Corentin Kervadec, Christian Wolf, Grigory Antipov, Moez Baccouche, Madiha Nadri*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9766527f2b5d3e95d4a733fcfb77bd7e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9766527f2b5d3e95d4a733fcfb77bd7e-Abstract.html)

        **Abstract**:

        Methods for Visual Question Anwering (VQA) are notorious for leveraging dataset biases rather than performing reasoning, hindering generalization. It has been recently shown that better reasoning patterns emerge in attention layers of a state-of-the-art VQA model when they are trained on perfect (oracle) visual inputs. This provides evidence that deep neural networks can learn to reason when training conditions are favorable enough. However, transferring this learned knowledge to deployable models is a challenge, as much of it is lost during the transfer.We propose a method for knowledge transfer based on a regularization term in our loss function, supervising the sequence of required reasoning operations.We provide a theoretical analysis based on PAC-learning, showing that such program prediction can lead to decreased sample complexity under mild hypotheses. We also demonstrate the effectiveness of this approach experimentally on the GQA dataset and show its complementarity to BERT-like self-supervised pre-training.

        ----

        ## [1396] Conformal Bayesian Computation

        **Authors**: *Edwin Fong, Chris C. Holmes*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/97785e0500ad16c18574c64189ccf4b4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/97785e0500ad16c18574c64189ccf4b4-Abstract.html)

        **Abstract**:

        We develop scalable methods for producing conformal Bayesian predictive intervals with finite sample calibration guarantees. Bayesian posterior predictive distributions, $p(y \mid x)$,  characterize subjective beliefs on outcomes of interest, $y$, conditional on predictors, $x$. Bayesian prediction is well-calibrated when the model is true, but the predictive intervals may exhibit poor empirical coverage when the  model is misspecified, under the so called ${\cal{M}}$-open perspective. In contrast, conformal inference provides finite sample frequentist guarantees on predictive confidence intervals without the requirement of model fidelity. Using 'add-one-in' importance sampling, we show that conformal Bayesian predictive intervals are efficiently obtained from re-weighted posterior samples of model parameters. Our approach contrasts with existing conformal methods that require expensive refitting of models or data-splitting to achieve computational efficiency. We demonstrate the utility on a range of examples including extensions to partially exchangeable settings such as hierarchical models.

        ----

        ## [1397] A Unified Approach to Fair Online Learning via Blackwell Approachability

        **Authors**: *Evgenii Chzhen, Christophe Giraud, Gilles Stoltz*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/97ea3cfb64eeaa1edba65501d0bb3c86-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/97ea3cfb64eeaa1edba65501d0bb3c86-Abstract.html)

        **Abstract**:

        We provide a setting and a general approach to fair online learning with stochastic sensitive and non-sensitive contexts.The setting is a repeated game between the Player and Nature, where at each stage both pick actions based on the contexts. Inspired by the notion of unawareness, we assume that the Player can only access the non-sensitive context before making a decision, while we discuss both cases of Nature accessing the sensitive contexts and Nature unaware of the sensitive contexts. Adapting Blackwell's approachability theory to handle the case of an unknown contexts' distribution, we provide a general necessary and sufficient condition for learning objectives to be compatible with some fairness constraints. This condition is instantiated on (group-wise) no-regret and (group-wise) calibration objectives, and on demographic parity as an additional constraint. When the objective is not compatible with the constraint, the provided framework permits to characterise the optimal trade-off between the two.

        ----

        ## [1398] Training Neural Networks is ER-complete

        **Authors**: *Mikkel Abrahamsen, Linda Kleist, Tillmann Miltzow*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9813b270ed0288e7c0388f0fd4ec68f5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9813b270ed0288e7c0388f0fd4ec68f5-Abstract.html)

        **Abstract**:

        Given a neural network, training data, and a threshold, finding weights for the neural network such that the total error is below the threshold is known to be NP-hard. We determine the algorithmic complexity of this fundamental problem precisely, by showing that it is $\exists\mathbb R$-complete. This means that the problem is equivalent, up to polynomial time reductions, to deciding whether a system of polynomial equations and inequalities with integer coefficients and real unknowns has a solution. If, as widely expected, $\exists\mathbb R$ is strictly larger than NP, our work implies that the problem of training neural networks is not even in NP.Neural networks are usually trained using some variation of backpropagation. The result of this paper gives an explanation why techniques commonly used to solve big instances of NP-complete problems (such as SAT solvers, IP solvers, local search, dynamic programming, etc.) seem to be of no use to this task.

        ----

        ## [1399] Understanding the Under-Coverage Bias in Uncertainty Estimation

        **Authors**: *Yu Bai, Song Mei, Huan Wang, Caiming Xiong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9854d7afce413aa13cd0a1d39d0bcec5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9854d7afce413aa13cd0a1d39d0bcec5-Abstract.html)

        **Abstract**:

        Estimating the data uncertainty in regression tasks is often done by learning a quantile function or a prediction interval of the true label conditioned on the input. It is frequently observed that quantile regression---a vanilla algorithm for learning quantiles with asymptotic guarantees---tends to *under-cover* than the desired coverage level in reality. While various fixes have been proposed, a more fundamental understanding of why this under-coverage bias happens in the first place remains elusive.In this paper, we present a rigorous theoretical study on the coverage of uncertainty estimation algorithms in learning quantiles. We prove that quantile regression suffers from an inherent under-coverage bias, in a vanilla setting where we learn a realizable linear quantile function and there is more data than parameters. More quantitatively, for $\alpha>0.5$ and small $d/n$, the $\alpha$-quantile learned by quantile regression roughly achieves coverage $\alpha - (\alpha-1/2)\cdot d/n$ regardless of the noise distribution, where $d$ is the input dimension and $n$ is the number of training data. Our theory reveals that this under-coverage bias stems from a certain high-dimensional parameter estimation error that is not implied by existing theories on quantile regression. Experiments on simulated and real data verify our theory and further illustrate the effect of various factors such as sample size and model capacity on the under-coverage bias in more practical setups.

        ----

        

[Go to the previous page](NIPS-2021-list06.md)

[Go to the next page](NIPS-2021-list08.md)

[Go to the catalog section](README.md)